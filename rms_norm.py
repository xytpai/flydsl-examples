import time
import torch
import argparse
import functools
import numpy as np
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl.expr import range_constexpr, arith, vector, gpu
from flydsl._mlir import ir
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.kernel_function import CompilationContext
import flydsl._mlir.dialects.gpu as mlir_gpu 

from utils.tensor_shim import get_dtype_in_kernel, GTensor, STensor, _to_raw
fm_fast = arith.FastMathFlags.fast


@dataclass
class Args:
    dtype: torch.dtype
    batch_size: int
    norm_size: int
    eps: float = 1e-6


def create_inputs(args):
    x = torch.randn((args.batch_size, args.norm_size), dtype=args.dtype, device='cuda')
    weight = torch.randn((args.norm_size), dtype=args.dtype, device='cuda')
    return (x, weight, args.eps)


def create_outputs(args):
    out = torch.zeros((args.batch_size, args.norm_size), dtype=args.dtype, device='cuda')
    return (out,)


def ref_func(x, weight, eps, out):
    def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor):
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = x.to(input_dtype)
        out.copy_(weight * x)
    rms_norm_forward(x, weight, eps, out)


@functools.lru_cache(maxsize=1024)
def compile_rmsnorm_kernel(dtype: str, batch_size: int, norm_size: int, eps: float):
    if dtype == 'f32':
        VEC_SIZE = 4
    elif dtype in ['f16', 'bf16']:
        VEC_SIZE = 8
    BLOCK_SIZE = 256
    BLOCK_WORK_SIZE = BLOCK_SIZE * VEC_SIZE
    WARP_SIZE = 64
    NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    WARP_SIZE_SHFL_OFFSETS = []
    offsets_ = WARP_SIZE // 2
    while offsets_ >= 1:
        WARP_SIZE_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2
    
    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem")
    smem_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_offset + NUM_WARPS * 4
    
    @flyc.kernel
    def rmsnorm_kernel(
        X: fx.Tensor,
        GAMMA: fx.Tensor,
        Y: fx.Tensor,
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        acc_vec_t = T.vec(VEC_SIZE, T.f32)
        vec_t = T.vec(VEC_SIZE, dtype_)

        bidx = fx.block_idx.x
        tidx = fx.thread_idx.x
        wid = fx.Index(tidx // WARP_SIZE)
        w_tid = fx.Int32(tidx % WARP_SIZE)

        X_ = GTensor(X, dtype=dtype_, shape=(batch_size, norm_size))
        GAMMA_ = GTensor(GAMMA, dtype=dtype_, shape=(norm_size,))
        Y_ = GTensor(Y, dtype=dtype_, shape=(batch_size, norm_size))

        c_zero_f = arith.constant(0.0, type=T.f32)
        init_state = [c_zero_f]
        for vec_idx, state in range(tidx * VEC_SIZE, fx.Int32(norm_size), fx.Int32(BLOCK_WORK_SIZE), init=init_state):
            value = state[0]
            x_vec = X_.vec_load((bidx, vec_idx), VEC_SIZE)
            x_vec = x_vec.extf(acc_vec_t)
            x_vec = x_vec * x_vec
            value = value + vector.ReductionOp(T.f32, vector.CombiningKind.ADD, x_vec).dest
            results = yield [value]
        
        for offset in WARP_SIZE_SHFL_OFFSETS:
            results = results + results.shuffle_xor(fx.Int32(offset), fx.Int32(WARP_SIZE))
        
        base_ptr = allocator.get_base()
        smem_ptr = SmemPtr(base_ptr, smem_offset, T.f32, shape=(NUM_WARPS,))
        smem_ = STensor(smem_ptr, T.f32, shape=(NUM_WARPS,))
        smem_[wid] = results

        gpu.barrier()
        
        sq_sum = c_zero_f
        cond = arith.cmpi(arith.CmpIPredicate.eq, w_tid, fx.Int32(0))
        if cond:
            for i in range_constexpr(NUM_WARPS):
                sq_sum = sq_sum + smem_[i]
        sq_sum = mlir_gpu.ShuffleOp(_to_raw(sq_sum), _to_raw(fx.Int32(0)), _to_raw(fx.Int32(WARP_SIZE)), mode="idx").shuffleResult

        sq_mean = sq_sum / norm_size + eps
        rrms = sq_mean.rsqrt(fastmath=fm_fast)
        rrms_vec = vector.broadcast(acc_vec_t, rrms)

        for vec_idx in range(tidx * VEC_SIZE, fx.Int32(norm_size), fx.Int32(BLOCK_WORK_SIZE)):
            x_vec = X_.vec_load((bidx, vec_idx), VEC_SIZE)
            w_vec = GAMMA_.vec_load((vec_idx,), VEC_SIZE)
            x_vec = x_vec.extf(acc_vec_t)
            w_vec = w_vec.extf(acc_vec_t)
            y_vec = x_vec * w_vec * rrms_vec
            y_vec = y_vec.truncf(vec_t)
            Y_.vec_store((bidx, vec_idx), y_vec, VEC_SIZE)
        return
    
    @flyc.jit
    def launch_rmsnorm_kernel(
        X: fx.Tensor,
        GAMMA: fx.Tensor,
        Y: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        
        rmsnorm_kernel(X, GAMMA, Y).launch(
            grid=(batch_size, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream
        )
    
    return launch_rmsnorm_kernel


def func(x, weight, eps, out):
    batch_size, norm_size = x.shape
    if x.dtype == torch.float:
        exe = compile_rmsnorm_kernel('f32', batch_size, norm_size, eps)
    elif x.dtype == torch.half:
        exe = compile_rmsnorm_kernel('f16', batch_size, norm_size, eps)
    elif x.dtype == torch.bfloat16:
        exe = compile_rmsnorm_kernel('bf16', batch_size, norm_size, eps)
    else:
        raise NotImplementedError()
    exe(x, weight, out, stream=torch.cuda.Stream())


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
        maxdiff_out = (output - ref_output).abs().max()
        # print(f"maxdiff_out:{maxdiff_out}")
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
    parser.add_argument("--norm_size", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {'f32': torch.float, 'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    args = Args(**vars(args))
    benchmark(args, func, ref_func)
