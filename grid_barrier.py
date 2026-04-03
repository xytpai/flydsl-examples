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
from flydsl._mlir import ir
from flydsl.expr import range_constexpr, arith, vector, gpu, rocdl
from flydsl._mlir.dialects import llvm, fly, memref, scf
from flydsl.compiler.protocol import fly_values
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch

from utils.tensor_shim import STensor, GTensor
from utils.cooperative_groups import CooperativeGroup


NBLOCKS = 128


@dataclass
class Args:
    block_size: int


def create_inputs(args):
    input = torch.randn(args.block_size, dtype=torch.float32, device='cuda')
    return (input,)


def create_outputs(args):
    output = torch.zeros(args.block_size, dtype=torch.float32, device='cuda')
    return (output,)


def ref_func(input, output):
    output.copy_(input)


@functools.lru_cache(maxsize=1024)
def compile_grid_barrier_kernel(block_size: int):
    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
    smem_ilb_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_ilb_offset + 4

    @flyc.kernel
    def grid_barrier_kernel(
        IN: fx.Tensor,
        OUT: fx.Tensor,
        COUNTER: fx.Tensor,
    ):
        bidx = fx.block_idx.x
        tidx = fx.thread_idx.x
        
        IN_ = GTensor(IN, dtype=fx.Float32, shape=(-1,))
        OUT_ = GTensor(OUT, dtype=fx.Float32, shape=(-1,))
        
        base_ptr = allocator.get_base()
        smem_ilb_ptr = SmemPtr(base_ptr, smem_ilb_offset, T.i32, shape=(1,))
        ilb_ = STensor(smem_ilb_ptr, T.i32, shape=(1,))

        cg = CooperativeGroup(COUNTER, ilb_, tidx, 0, NBLOCKS)

        data = IN_[tidx]

        cg.fetch_add_counter()
        cg.wait()

        IN_[tidx] += 1
        OUT_[tidx] = data
        
        return
    
    @flyc.jit
    def launch_grid_barrier_kernel(
        IN: fx.Tensor,
        OUT: fx.Tensor,
        COUNTER: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        
        grid_barrier_kernel(IN, OUT, COUNTER).launch(
            grid=(NBLOCKS, 1, 1), block=(block_size, 1, 1), stream=stream
        )
    
    return launch_grid_barrier_kernel


COUNTER = None
def func(input, output):
    global COUNTER
    block_size = input.shape[0]
    exe = compile_grid_barrier_kernel(block_size)
    if COUNTER is None:
        COUNTER = torch.zeros(1, dtype=torch.int32, device='cuda')
    exe(input, output, COUNTER, torch.cuda.current_stream())


def benchmark(args, func, ref_func, warmup=20, niters=100):
    inputs = create_inputs(args)
    ref_inputs = (inputs[0].clone(),)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = inputs + outputs
    ref_inouts = ref_inputs + ref_outputs
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
    parser.add_argument("--block_size", type=int, required=True)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    args = Args(**vars(args))
    benchmark(args, func, ref_func)
    # rm -rf ~/.flydsl/ ; python3 grid_barrier.py --block_size 256
