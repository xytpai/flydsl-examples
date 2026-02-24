import time
import torch
import argparse
import numpy as np
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass

import flydsl
from flydsl.dialects.ext import flir
from flydsl.dialects.ext import arith
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.pipeline import Pipeline, run_pipeline
from _mlir import ir
from _mlir.ir import F16Type, BF16Type, F32Type, IntegerType
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


def create_reduce_kernel(dtype, VEC_SIZE: int):
    S = ir.ShapedType.get_dynamic_size()
    ARCH = get_rocm_arch()
    THR_M, THR_N = 1, 256
    VAL_M, VAL_N = VEC_SIZE, VEC_SIZE
    TILE_M = THR_M * VAL_M
    TILE_N = THR_N * VAL_N
    
    class BatchReduce(flir.MlirModule):
        GPU_MODULE_NAME = "reduce_kernels"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{ARCH}">']

        @flir.kernel
        def batch_reduce_kernel(
            self: flir.T.i64,
            X: lambda: T.memref(S, dtype.get()),
            Y: lambda: T.memref(S, dtype.get()),
            batch_size: lambda: T.index(),
            reduce_size: lambda: T.index(),
        ):
            # Get index
            tid_x = flir.thread_idx("x")
            bid_x = flir.block_idx("x")
            bid_y = flir.block_idx("y")

            # Create thread/value layouts
            thread_layout = flir.make_ordered_layout((THR_M, THR_N), order=(1, 0))
            value_layout = flir.make_ordered_layout((VAL_M, VAL_N), order=(1, 0))

            # Create copy atoms
            copy_atom_load = flir.make_copy_atom(dtype.get(), vector_size=VEC_SIZE)
            copy_atom_store = flir.make_copy_atom(dtype.get(), vector_size=VEC_SIZE)

            # Tiled Copies
            tiled_copy_X = flir.make_tiled_copy_tv(copy_atom_load, thread_layout, x_value_layout, thr_shape=(THR_M, THR_N), val_shape=(VAL_M, VAL_N))
            tiled_copy_Y = flir.make_tiled_copy_tv(copy_atom_store, thread_layout, value_layout, thr_shape=(THR_M, THR_N), val_shape=(VAL_M, VAL_N))

            # Specify input tensor layouts
            tensor_X = flir.make_tensor(A, shape=(batch_size, reduce_size), strides=(reduce_size, 1))
            tensor_Y = flir.make_tensor(B, shape=(batch_size, 1), strides=(1, 1))

            # Get per-block coordinates
            gX = flir.zipped_divide(tensor_X, (TILE_M, TILE_N))
            gY = flir.zipped_divide(tensor_Y, (TILE_M, 1))
            idX = flir.make_identity_tensor((batch_size, reduce_size)) # For tracking coordinates only
            idY = flir.make_identity_tensor((batch_size, 1)) # For tracking coordinates only
            cX = flir.zipped_divide(idX, (TILE_M, TILE_N))
            cY = flir.zipped_divide(idY, (TILE_M, 1))
            blkX = gX[(bid_y, bid_x)]
            blkY = gY[(bid_y, 1)]
            blkXrd = cX[(bid_y, bid_x)]
            blkYrd = cY[(bid_y, 1)]

            # Get per-thread coordinates
            thr_copy_X = tiled_copy_X.get_slice(tid_x)
            thr_copy_Y = tiled_copy_Y.get_slice(tid_x)
            thrX = thr_copy_X.partition_S(blkX)
            thrY = thr_copy_Y.partition_S(blkY)
            thrXrd = thr_copy_X.partition_S(blkXrd)
            thrYrd = thr_copy_Y.partition_S(blkYrd)

            frgX = flir.make_fragment_like(thrX, dtype.get())
            frgY = flir.make_fragment_like(thrY, dtype.get())

            frgPred = flir.make_rmem_tensor((THR_M, THR_N), IntegerType.get_signless(1))

            for idx_in_vec in range(THR_M * THR_N):
                idx_in_vec = flir.const_index(idx_in_vec)
                coords = thrXrd.coords_from_linear(idx_in_vec)
                pred_val = flir.elem_less(coords, (batch_size, reduce_size))
                pred_offsets = tuple(frgPred.offsets_from_linear(idx_in_vec))
                frgPred[pred_offsets] = pred_val

            flir.copy(tiled_copy_X, thrX, frgX, pred=frgPred)

            for v in range(VEC_SIZE):
                idx = flir.const_index(v)
                coords = (idx, )
                a_val = frgA[coords]
                b_val = frgB[coords]
                c_val = a_val + b_val
                frgC[coords] = c_val

            flir.copy(tiled_copy_C, frgC, thrC, pred=frgPred)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(S, dtype.get()),
            B: lambda: T.memref(S, dtype.get()),
            C: lambda: T.memref(S, dtype.get()),
            n: lambda: T.index(),
        ):
            c1 = arith.index(1)
            c_tile_elems = arith.index(BLOCK_WORK_SIZE)
            gx = (n + c_tile_elems - c1) // c_tile_elems
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "batch_reduce_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[A, B, C, n],
            )

    return BatchReduce().module


def func(x, y):
    ref_func(x, y)


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
