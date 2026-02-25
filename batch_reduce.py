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


@dataclass
class Args:
    batch_size: int
    reduce_size: int
    dtype: torch.dtype


def create_inputs(args):
    x = torch.randn((args.batch_size, args.reduce_size), dtype=args.dtype, device='cuda')
    return (x,)


def create_outputs(args):
    y = torch.randn((args.batch_size, 1), dtype=args.dtype, device='cuda')
    return (y,)


def ref_func(x, y):
    torch.sum(x, dim=1, keepdim=True, out=y)


def make_block_reduce_add(tid, WARP_SIZE, RED_SLOTS):
    def block_reduce_add(val_f32, scratch_memref):
        arith_ops = flir.arith
        zero_idx = flir.const_index(0)

        if RED_SLOTS == 1:
            # Fast path: single-wave block (RED_SLOTS==1) needs no LDS and no barrier.
            # After xor-shuffle reduction, all lanes hold the same reduced value.
            width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
            w = arith.as_value(val_f32)
            for shift in [32, 16, 8, 4, 2, 1]:
                offset = arith.as_value(arith.constant(shift, type=T.i32()))
                peer = arith.as_value(gpu.ShuffleOp(arith.as_value(w), offset, width_i32, mode="xor").shuffleResult)
                w = arith.as_value(arith_ops.AddFOp(arith.as_value(w), peer, fastmath=fm_fast).result)
            return w

        scratch_tv = flir.make_tensor(scratch_memref, shape=(RED_SLOTS,), strides=(1,))
        tid_v = tid.value if hasattr(tid, "value") else tid
        tid_v = arith.as_value(tid_v)
        tid_i32 = arith.as_value(arith_ops.IndexCastOp(T.i32(), tid_v).result)
        c_warp_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
        lane_i32 = arith.as_value(arith_ops.RemUIOp(tid_i32, c_warp_i32).result)
        wave_i32 = arith.as_value(arith_ops.DivUIOp(tid_i32, c_warp_i32).result)
        width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
        # Use Flir layout algebra to compute LDS indices for the reduction scratch.
        c_num_waves = flir.const_index(RED_SLOTS)
        c1 = flir.const_index(1)
        shape_red = flir.make_shape(c_num_waves)
        stride_red = flir.make_stride(c1)
        layout_red = flir.make_layout(shape_red, stride_red)

        w = arith.as_value(val_f32)
        for sh in [32, 16, 8, 4, 2, 1]:
            off = arith.as_value(arith.constant(sh, type=T.i32()))
            peer = arith.as_value(gpu.ShuffleOp(arith.as_value(w), off, width_i32, mode="xor").shuffleResult)
            w = arith.as_value(arith_ops.AddFOp(arith.as_value(w), peer, fastmath=fm_fast).result)
        
        is_lane0 = arith.as_value(arith_ops.CmpIOp(
            arith_ops.CmpIPredicate.eq,
            lane_i32,
            arith.as_value(arith.constant(0, type=T.i32())),
        ).result)
        if is_lane0:
            wave_idx = arith_ops.IndexCastOp(T.index(), wave_i32).result
            red_idx = flir.crd2idx(flir.make_coord(wave_idx), layout_red)
            scratch_tv[red_idx] = w
        gpu.barrier()

        NUM_WAVES = RED_SLOTS
        is_wave0 = arith.as_value(arith_ops.CmpIOp(
            arith_ops.CmpIPredicate.eq,
            wave_i32,
            arith.as_value(arith.constant(0, type=T.i32())),
        ).result)
        # Only wave0 does final reduction and writes scratch[0].
        if is_wave0:
            in_range = arith.as_value(arith_ops.CmpIOp(
                arith_ops.CmpIPredicate.ult,
                lane_i32,
                arith.as_value(arith.constant(NUM_WAVES, type=T.i32())),
            ).result)

            c0_i32 = arith.as_value(arith.constant(0, type=T.i32()))
            lane_safe_i32 = arith.as_value(flir.arith.SelectOp(in_range, lane_i32, c0_i32).result)
            lane_safe_idx = arith.as_value(arith_ops.IndexCastOp(T.index(), lane_safe_i32).result)
            red_idx = flir.crd2idx(flir.make_coord(lane_safe_idx), layout_red)
            v = scratch_tv[red_idx]
            z = arith.as_value(arith.constant(0.0, type=T.f32()))
            ww = arith.as_value(flir.arith.SelectOp(in_range, v, z).result)

            for sh in [32, 16, 8, 4, 2, 1]:
                off = arith.as_value(arith.constant(sh, type=T.i32()))
                peer = arith.as_value(gpu.ShuffleOp(arith.as_value(ww), off, width_i32, mode="xor").shuffleResult)
                ww = arith.as_value(arith_ops.AddFOp(arith.as_value(ww), peer, fastmath=fm_fast).result)

            is_lane0_2 = arith.as_value(arith_ops.CmpIOp(
                arith_ops.CmpIPredicate.eq,
                lane_i32,
                arith.as_value(arith.constant(0, type=T.i32())),
            ).result)
            if is_lane0_2:
                red_idx0 = flir.crd2idx(flir.make_coord(zero_idx), layout_red)
                scratch_tv[red_idx0] = ww

        gpu.barrier()
        red_idx0 = flir.crd2idx(flir.make_coord(zero_idx), layout_red)
        return scratch_tv[red_idx0]

    try:
        return lower_range_for_loops(block_reduce_add)
    except Exception:
        return block_reduce_add


def create_reduce_kernel(dtype, VEC_SIZE: int):
    S = ir.ShapedType.get_dynamic_size()
    ARCH = get_rocm_arch()
    BLOCK_THREADS = 256
    WARP_SIZE = 64
    BLOCK_WORK_SIZE = BLOCK_THREADS * VEC_SIZE
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    allocator = SmemAllocator(None, arch=ARCH)
    
    class BatchReduce(flir.MlirModule):
        GPU_MODULE_NAME = "reduce_kernels"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{ARCH}">']

        def init_gpu_module(self):
            self.dtype = dtype.get()
            self.acc_type = T.f32()
            self.smem = allocator.allocate_array(T.f32(), RED_SLOTS)
            allocator.finalize()

        @flir.kernel
        def batch_reduce_kernel(
            self: flir.T.i64,
            X: lambda: T.memref(S, dtype.get()),
            Y: lambda: T.memref(S, dtype.get()),
            batch_size: lambda: T.index(),
            reduce_size: lambda: T.index(),
        ):
            tid_x = flir.thread_idx("x")
            bid_x = flir.block_idx("x")
            vec_type = VectorType.get([VEC_SIZE], self.dtype)
            acc_vec_type = VectorType.get([VEC_SIZE], self.acc_type)
            c_zero = arith.constant(0.0, type=self.acc_type)
            thread_sum = (c_zero)
            for vec_idx in range(tid_x * VEC_SIZE, reduce_size, BLOCK_WORK_SIZE):
                vec_addr = bid_x * reduce_size + vec_idx
                vec = flir.vector.load(vec_type, X, [arith.as_value(vec_addr)], alignment=16)
                vec = flir.arith.extf(acc_vec_type, arith.as_value(vec))
                red = flir.vector.reduction(self.acc_type, "add", arith.as_value(vec), fastmath=fm_fast)
                thread_sum = thread_sum + red
            block_reduce_add = make_block_reduce_add(tid_x, WARP_SIZE, RED_SLOTS)
            base_ptr = allocator.get_base()
            sum_val = block_reduce_add(thread_sum, self.smem(base_ptr).get())
            sum_val = flir.arith.truncf(self.dtype, (sum_val))
            flir.memref.store(arith.as_value(sum_val), Y, [flir.const_index(bid_x),])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            X: lambda: T.memref(S, dtype.get()),
            Y: lambda: T.memref(S, dtype.get()),
            batch_size: lambda: T.index(),
            reduce_size: lambda: T.index(),
        ):
            c1 = arith.index(1)
            bx = arith.index(BLOCK_THREADS)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "batch_reduce_kernel"],
                grid_size=(batch_size, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[X, Y, batch_size, reduce_size],
            )

    return BatchReduce().module


EXE = None
def func(x, y):
    global EXE
    if not EXE:
        if x.dtype == torch.float:
            module = create_reduce_kernel(F32Type, 4)
        elif x.dtype == torch.half:
            module = create_reduce_kernel(F16Type, 8)
        elif x.dtype == torch.bfloat16:
            module = create_reduce_kernel(BF16Type, 8)
        optimized = run_pipeline(module, Pipeline().canonicalize().cse())
        EXE = flydsl.compile(optimized)
    EXE(x, y, x.shape[0], x.shape[1])
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
        is_allclose = torch.allclose(output, ref_output)
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
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--reduce_size", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {'f32': torch.float, 'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    args = Args(**vars(args))
    benchmark(args, func, ref_func)
