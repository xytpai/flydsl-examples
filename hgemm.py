import time
import torch
import argparse
import numpy as np
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass

import flydsl
from flydsl.lang.ir.types import T as FT
from flydsl.dialects.ext import flir, gpu, arith, buffer_ops
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.pipeline import Pipeline, run_pipeline
from flydsl.dialects.ext.python_control_flow import range_constexpr, lower_range_for_loops
from flydsl.utils import SmemAllocator, SmemPtr
fm_fast = flir.arith.FastMathFlags.fast

from _mlir import ir
from _mlir.ir import F16Type, BF16Type, F32Type, IntegerType, VectorType
import _mlir.extras.types as T


@dataclass
class Args:
    m: int
    n: int
    k: int
    dtype: torch.dtype


def create_inputs(args):
    a = torch.empty((args.m, args.k), dtype=args.dtype, device='cuda')
    a.uniform_(-1, 1)
    b = torch.empty((args.k, args.n), dtype=args.dtype, device='cuda')
    b.uniform_(-1, 1)
    return (a, b)


def create_outputs(args):
    c = torch.randn((args.m, args.n), dtype=args.dtype, device='cuda')
    return (c,)


def ref_func(a, b, c):
    torch.mm(a, b, out=c)


def create_hgemm_kernel(
    DTYPE,
    ELEMENT_BYTES: int,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    USE_ASYNC_COPY: bool,
):
    DYN = ir.ShapedType.get_dynamic_size()
    ARCH = get_rocm_arch()
    TILE_K_BYTES = TILE_K * ELEMENT_BYTES
    BLOCK_THREADS = 256
    allocator_pong = SmemAllocator(None, arch=ARCH, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=ARCH, global_sym_name="smem1")
    
    class HGEMM(flir.MlirModule):
        GPU_MODULE_NAME = "gemm_kernels"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{ARCH}">']

        def init_gpu_module(self):
            self.dtype = DTYPE.get()
            self.i32 = T.i32()
            self.acc_type = T.f32()
            self.lds_a_pong = allocator_pong.allocate_array(self.dtype, TILE_M * TILE_K)
            self.lds_a_ping = allocator_ping.allocate_array(self.dtype, TILE_M * TILE_K)
            allocator_pong.finalize()
            allocator_ping.finalize()

        @flir.kernel
        def hgemm_kernel(
            self: flir.T.i64,
            a: lambda: T.memref(DYN, DTYPE.get()),
            b: lambda: T.memref(DYN, DTYPE.get()),
            c: lambda: T.memref(DYN, DTYPE.get()),
            m: lambda: T.index(),
            n: lambda: T.index(),
            k: lambda: T.index(),
        ):
            layout_c = flir.make_layout((m, n), stride=(n, 1))
            k_bytes = k * ELEMENT_BYTES
            k_div4bytes = (k * ELEMENT_BYTES) / 4
            layout_a = flir.make_layout((m, k_bytes), stride=(k_bytes, 1))
            layout_a_div4 = flir.make_layout((m, k_div4bytes), stride=(k_div4bytes, 1))
            layout_lds = flir.make_layout((TILE_M, TILE_K), stride=(TILE_K, 1))

            # CK-style XOR16 swizzle parameter (const).
            k_blocks16 = arith.index(TILE_K_BYTES // 16)

            tid_x = flir.thread_idx("x")
            bid_x = flir.block_idx("x")
            bid_y = flir.block_idx("y")

            base_ptr0, base_ptr1 = allocator_pong.get_base(), allocator_ping.get_base()
            lds_a_pong_ptr = self.lds_a_pong(base_ptr0)
            lds_a_ping_ptr = self.lds_a_ping(base_ptr1)

            lds_a_pong = SmemPtr(
                base_ptr0, lds_a_pong_ptr.byte_offset, self.dtype, shape=(TILE_M * TILE_K,)
            ).get()
            lds_a_ping = SmemPtr(
                base_ptr1, lds_a_ping_ptr.byte_offset, self.dtype, shape=(TILE_M * TILE_K,)
            ).get()

            m_i32 = arith.index_cast(self.i32, m)
            n_i32 = arith.index_cast(self.i32, n)
            k_i32 = arith.index_cast(self.i32, k)

            a_bytes = m_i32 * k_i32 * arith.i32(ELEMENT_BYTES)
            c_bytes = m_i32 * n_i32 * arith.i32(ELEMENT_BYTES)
            a_rsrc = buffer_ops.create_buffer_resource(a, num_records_bytes=a_bytes)
            c_rsrc = buffer_ops.create_buffer_resource(c, num_records_bytes=c_bytes)
            b_rsrc = buffer_ops.create_buffer_resource(b, max_size=True)

            bx_m = bid_x * TILE_M
            by_n = bid_y * TILE_N

        @flir.jit
        def __call__(
            self: flir.T.i64,
            a: lambda: T.memref(DYN, DTYPE.get()),
            b: lambda: T.memref(DYN, DTYPE.get()),
            c: lambda: T.memref(DYN, DTYPE.get()),
            m: lambda: T.index(),
            n: lambda: T.index(),
            k: lambda: T.index(),
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(BLOCK_THREADS, index=True)
            tm = arith.constant(TILE_M, index=True)
            tn = arith.constant(TILE_N, index=True)
            one = arith.constant(1, index=True)
            gx = (m + tm - one) / tm
            gy = n / tn
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "hgemm_kernel"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[a, b, c, m, n, k],
            )

    return HGEMM().module


EXE = None
def func(a, b, c):
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    ELEMENT_BYTES = 2
    TILE_M = 128
    TILE_N = 128
    TILE_K = 32
    ASYNC_COPY = False
    global EXE
    if not EXE:
        if a.dtype == torch.half:
            module = create_hgemm_kernel(F16Type, ELEMENT_BYTES, TILE_M, TILE_N, TILE_K, ASYNC_COPY)
        elif a.dtype == torch.bfloat16:
            module = create_hgemm_kernel(BF16Type, ELEMENT_BYTES, TILE_M, TILE_N, TILE_K, ASYNC_COPY)
        optimized = run_pipeline(module, Pipeline().canonicalize().cse())
        EXE = flydsl.compile(optimized)
    EXE(a, b, c, m, n, k)
    torch.cuda.synchronize()


def benchmark(args, func, ref_func, warmup=20, niters=100):
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = inputs + outputs
    ref_inouts = inputs + ref_outputs
    func(*inouts)
    ref_func(*ref_inouts)
    for output, ref_output in zip(outputs, ref_outputs):
        is_allclose = torch.allclose(output, ref_output, atol=1e-3, rtol=1e-3)
        assert is_allclose == True
    print("validation passed!\n", flush=True)

    # get ref_func perf
    print("===================== [REF] =====================")
    for i in range(warmup):
        ref_func(*ref_inouts)
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        for i in range(niters):
            ref_func(*ref_inouts)
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    print(table)

    # get func perf
    print("===================== [FLYDSL] =====================")
    for i in range(warmup):
        func(*inouts)
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        for i in range(niters):
            func(*inouts)
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    print(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Examples")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    args = Args(**vars(args))
    benchmark(args, func, ref_func)
