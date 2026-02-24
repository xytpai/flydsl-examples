import time
import torch
import argparse
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass

import flydsl
from flydsl.dialects.ext import flir
from flydsl.dialects.ext import arith
from _mlir import ir
from _mlir.ir import F16Type, BF16Type, F32Type, IntegerType
import _mlir.extras.types as T


@dataclass
class Args:
    n: int
    dtype: torch.dtype


def create_inputs(args):
    a = torch.randn(args.n, dtype=args.dtype).cuda()
    b = torch.randn(args.n, dtype=args.dtype).cuda()
    return a, b


def create_add_kernel(dtype, VEC_SIZE):
    # NOTE: dtype is static, n is dynamic

    # NOTE: Kernel operands in the lowered module use dynamic memref types.
    # Keep the host stub signature dynamic too so gpu.launch_func types match.
    S = ir.ShapedType.get_dynamic_size()
    BLOCK_SIZE = 256
    BLOCK_WORK_SIZE = BLOCK_SIZE * VEC_SIZE
    
    class PointwiseAdd(flir.MlirModule):
        GPU_MODULE_NAME = "pointwise_kernels"
        GPU_MODULE_TARGETS = ['#rocdl.target<abi = "500">']

        @flir.kernel
        def pointwise_add_kernel(
            self: flir.T.i64,
            A: lambda: T.memref(S, dtype.get()),
            B: lambda: T.memref(S, dtype.get()),
            C: lambda: T.memref(S, dtype.get()),
            n: lambda: T.index(),
        ):
            # Get index
            tid_x = flir.thread_idx("x")
            bid_x = flir.block_idx("x")            
            linear_x = bid_x * BLOCK_WORK_SIZE + tid_x * VEC_SIZE

            # Create thread/value layouts
            thread_layout = flir.make_ordered_layout((BLOCK_SIZE, ))
            value_layout = flir.make_ordered_layout((VEC_SIZE, ))

            # Create copy atoms
            copy_atom_load = flir.make_copy_atom(dtype.get(), vector_size=VEC_SIZE)
            copy_atom_store = flir.make_copy_atom(dtype.get(), vector_size=VEC_SIZE)

            # Tiled Copies
            tiled_copy_A = flir.make_tiled_copy_tv(copy_atom_load, thread_layout, value_layout, thr_shape=(BLOCK_SIZE,), val_shape=(VEC_SIZE,))
            tiled_copy_B = flir.make_tiled_copy_tv(copy_atom_load, thread_layout, value_layout, thr_shape=(BLOCK_SIZE,), val_shape=(VEC_SIZE,))
            tiled_copy_C = flir.make_tiled_copy_tv(copy_atom_store, thread_layout, value_layout, thr_shape=(BLOCK_SIZE,), val_shape=(VEC_SIZE,))

            # Specify input tensor layouts
            tensor_A = flir.make_tensor(A, shape=(n, ), strides=(1, ))
            tensor_B = flir.make_tensor(B, shape=(n, ), strides=(1, ))
            tensor_C = flir.make_tensor(C, shape=(n, ), strides=(1, ))

            # Get per-block coordinates
            gA = flir.zipped_divide(tensor_A, (BLOCK_WORK_SIZE, ))
            gB = flir.zipped_divide(tensor_B, (BLOCK_WORK_SIZE, ))
            gC = flir.zipped_divide(tensor_C, (BLOCK_WORK_SIZE, ))
            idC = flir.make_identity_tensor((n, )) # For tracking coordinates only
            cC = flir.zipped_divide(idC, (BLOCK_WORK_SIZE, ))
            blk_coord = (bid_x, )
            blkA = gA[blk_coord]
            blkB = gB[blk_coord]
            blkC = gC[blk_coord]
            blkCrd = cC[blk_coord]

            # Get per-thread coordinates
            thr_copy_A = tiled_copy_A.get_slice(tid_x)
            thr_copy_B = tiled_copy_B.get_slice(tid_x)
            thr_copy_C = tiled_copy_C.get_slice(tid_x)
            thrA = thr_copy_A.partition_S(blkA)
            thrB = thr_copy_B.partition_S(blkB)
            thrC = thr_copy_C.partition_S(blkC)
            thrCrd = thr_copy_C.partition_S(blkCrd)

            val_shape = tiled_copy_A.val_shape
            frgA = flir.make_fragment_like(thrA, dtype.get())
            frgB = flir.make_fragment_like(thrB, dtype.get())
            frgC = flir.make_fragment_like(thrC, dtype.get())
            pred_ty = IntegerType.get_signless(1)
            frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
            total_vals = val_shape[0]

            for linear in range(total_vals):
                linear_idx = flir.const_index(linear)
                coords = thrCrd.coords_from_linear(linear_idx)
                pred_val = flir.elem_less(coords, (n, ))
                pred_offsets = tuple(frgPred.offsets_from_linear(linear_idx))
                frgPred[pred_offsets] = pred_val

            flir.copy(tiled_copy_A, thrA, frgA, pred=frgPred)
            flir.copy(tiled_copy_B, thrB, frgB, pred=frgPred)

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
                [self.GPU_MODULE_NAME, "pointwise_add_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[A, B, C, n],
            )

    return PointwiseAdd()


def func(a, b):
    out = torch.empty_like(a)
    if a.dtype == torch.float:
        module = create_add_kernel(F32Type, 4)
    elif a.dtype == torch.half:
        module = create_add_kernel(F16Type, 8)
    elif a.dtype == torch.bfloat16:
        module = create_add_kernel(BF16Type, 8)
    exe = flydsl.compile(module)
    exe(a, b, out, out.numel())
    return (out, )


def ref_func(a, b):
    return (a + b, )


def benchmark(args, func, ref_func, warmup=10, niters=10, use_profiler=False):
    inputs = create_inputs(args)
    outputs = func(*inputs)
    ref_outputs = ref_func(*inputs)
    for output, ref_output in zip(outputs, ref_outputs):
        is_allclose = torch.allclose(output, ref_output)
        assert is_allclose == True
    print("validation passed!", flush=True)
    for i in range(warmup):
        outputs = func(*inputs)
    if use_profiler:
        with profile(activities=[ProfilerActivity.CUDA], ) as prof:
            for i in range(niters):
                outputs = func(*inputs)
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        print(table)
    else:
        torch.cuda.synchronize()
        start = time.time()
        for i in range(niters):
            outputs = func(*inputs)
        torch.cuda.synchronize()
        end = time.time()
        elapsed_per_iter = (end - start) / niters
        print(f"elapsed_per_iter:{elapsed_per_iter * 1e6} us")


if __name__ == '__main__':
    print(f"run: {__file__}")
    parser = argparse.ArgumentParser(description="Examples")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    args = parser.parse_args()
    print(args)
    dtype_convert = {'f32': torch.float, 'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    args = Args(**vars(args))
    benchmark(args, func, ref_func)
