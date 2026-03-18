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
from flydsl.expr import range_constexpr, arith

from utils.tensor_shim import get_dtype_in_kernel, GTensor


@dataclass
class Args:
    dtype: torch.dtype
    n: int


def create_inputs(args):
    a = torch.randn(args.n, dtype=args.dtype, device='cuda')
    b = torch.randn(args.n, dtype=args.dtype, device='cuda')
    return a, b


def create_outputs(args):
    c = torch.randn(args.n, dtype=args.dtype, device='cuda')
    return (c,)


def ref_func(a, b, out):
    torch.add(a, b, out=out)


@functools.lru_cache(maxsize=1024)
def compile_pointwise_add_kernel(dtype: str, n: int):
    if dtype == 'f32':
        VEC_SIZE = 4
    elif dtype in ['f16', 'bf16']:
        VEC_SIZE = 8
    BLOCK_SIZE = 256
    BLOCK_WORK_SIZE = BLOCK_SIZE * VEC_SIZE

    @flyc.kernel
    def pointwise_add_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        bidx = fx.block_idx.x
        tidx = fx.thread_idx.x

        A_ = GTensor(A, dtype=dtype_, shape=(-1,))
        B_ = GTensor(B, dtype=dtype_, shape=(-1,))
        C_ = GTensor(C, dtype=dtype_, shape=(-1,))

        index = bidx * BLOCK_WORK_SIZE + tidx * VEC_SIZE
        remaining = n - index
        if arith.cmpi(arith.CmpIPredicate.ult, remaining, fx.Int32(VEC_SIZE)):
            for i in range_constexpr(VEC_SIZE):
                if arith.cmpi(arith.CmpIPredicate.ult, index + i, fx.Int32(n)):
                    C_[index + i] = A_[index + i] + B_[index + i]
        else:
            vec_a = A_.vec_load((index,), VEC_SIZE)
            vec_b = B_.vec_load((index,), VEC_SIZE)
            vec_c = vec_a + vec_b
            C_.vec_store((index,), vec_c, VEC_SIZE)
        return
    
    @flyc.jit
    def launch_pointwise_add_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        nblocks = (n + BLOCK_WORK_SIZE - 1) // BLOCK_WORK_SIZE
        pointwise_add_kernel(A, B, C).launch(
            grid=(nblocks, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream
        )
    
    return launch_pointwise_add_kernel


def func(a, b, out):
    n = out.numel()
    if a.dtype == torch.float:
        exe = compile_pointwise_add_kernel('f32', n)
    elif a.dtype == torch.half:
        exe = compile_pointwise_add_kernel('f16', n)
    elif a.dtype == torch.bfloat16:
        exe = compile_pointwise_add_kernel('bf16', n)
    else:
        raise NotImplementedError()
    exe(a, b, out, stream=torch.cuda.current_stream())


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
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {'f32': torch.float, 'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    args = Args(**vars(args))
    benchmark(args, func, ref_func)
