import time
import torch
import argparse
import numpy as np
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass

import flydsl
from flydsl.dialects.ext import flir, gpu, arith
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.pipeline import Pipeline, run_pipeline
from flydsl.dialects.ext.python_control_flow import range_constexpr, lower_range_for_loops
from flydsl.utils import SmemAllocator
fm_fast = flir.arith.FastMathFlags.fast

from _mlir import ir
from _mlir.ir import F16Type, BF16Type, F32Type, IntegerType, VectorType
import _mlir.extras.types as T

from utils.ftensor import GTensor


@dataclass
class Args:
    m: int = 8
    n: int = 8
    tile_m: int = 4
    tile_n: int = 4
    tile_m_id: int = 1
    tile_n_id: int = 1
    dtype: torch.dtype = torch.float


def create_inputs(args):
    x = torch.randn((args.m, args.n), dtype=args.dtype, device='cuda')
    return (args, x)


def create_outputs(args):
    out = torch.zeros((args.m, args.n), dtype=args.dtype, device='cuda')
    return (out,)


def ref_func(args, x, out):
    tile_m = args.tile_m
    tile_n = args.tile_n
    start_m = args.tile_m_id * tile_m
    start_n = args.tile_n_id * tile_n
    out[start_m:start_m + tile_m, start_n:start_n + tile_n] = x[start_m:start_m + tile_m, start_n:start_n + tile_n]


def create_tile_copy_kernel(dtype, m: int, n: int, tile_m: int, tile_n: int):
    DYN = ir.ShapedType.get_dynamic_size()
    ARCH = get_rocm_arch()
    value_layout = (2, 2)
    thread_layout = (tile_m // value_layout[0], tile_n // value_layout[1])
    BLOCK_THREADS = thread_layout[0] * thread_layout[1]
    
    class TileCopy(flir.MlirModule):
        GPU_MODULE_NAME = "norm_kernels"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{ARCH}">']

        @flir.kernel
        def tile_copy_kernel(
            self: flir.T.i64,
            x: lambda: T.memref(DYN, dtype.get()),
            out: lambda: T.memref(DYN, dtype.get()),
        ):
            tid_x = flir.thread_idx("x")
            bid_x = flir.block_idx("x")
            th_m = tid_x / thread_layout[1]
            th_n = tid_x % thread_layout[1]
            x_tensor = GTensor(x, dtype.get(), (m, n))
            out_tensor = GTensor(out, dtype.get(), (m, n))
            x_tile = x_tensor.local_tile((tile_m, tile_n), (1, 1))
            out_tile = out_tensor.local_tile((tile_m, tile_n), (1, 1))
            out_tile.copy_(x_tile, thread_layout=thread_layout, value_layout=value_layout, 
                thread_idxs=(th_m, th_n), vec_size=1)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            x: lambda: T.memref(DYN, dtype.get()),
            out: lambda: T.memref(DYN, dtype.get()),
        ):
            c1 = arith.index(1)
            c2 = arith.index(2)
            bx = arith.index(BLOCK_THREADS)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "tile_copy_kernel"],
                grid_size=(c2, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[x, out],
            )

    return TileCopy().module


EXE = None
def func(args, x, out):
    m, n = args.m, args.n
    tile_m, tile_n = args.tile_m, args.tile_n
    global EXE
    if not EXE:
        if x.dtype == torch.float:
            module = create_tile_copy_kernel(F32Type, m, n, tile_m, tile_n)
        elif x.dtype == torch.half:
            module = create_tile_copy_kernel(F16Type, m, n, tile_m, tile_n)
        elif x.dtype == torch.bfloat16:
            module = create_tile_copy_kernel(BF16Type, m, n, tile_m, tile_n)
        optimized = run_pipeline(module, Pipeline().canonicalize().cse())
        EXE = flydsl.compile(optimized)
    EXE(x, out)
    torch.cuda.synchronize()


def benchmark(args, func, ref_func, warmup=20, niters=100):
    inputs = create_inputs(args)
    print(inputs[1])
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = inputs + outputs
    ref_inouts = inputs + ref_outputs
    func(*inouts)
    ref_func(*ref_inouts)
    for output, ref_output in zip(outputs, ref_outputs):
        is_allclose = torch.allclose(output, ref_output, atol=1e-3, rtol=1e-3)
        print(output)
        print(ref_output)
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
    torch.set_printoptions(threshold=float("inf"))
    parser = argparse.ArgumentParser(description="Examples")
    parser.add_argument("--dtype", type=str, required=True)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {'f32': torch.float, 'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    args = Args(**vars(args))
    benchmark(args, func, ref_func)
