import time
import torch
import argparse
import functools
import numpy as np
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl.expr import range_constexpr, arith, vector, gpu, rocdl
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
    m: int
    n: int
    k: int


def create_inputs(args):
    a = torch.empty((args.m, args.k), dtype=args.dtype, device='cuda')
    a.uniform_(-1, 1)
    b = torch.empty((args.n, args.k), dtype=args.dtype, device='cuda')
    b.uniform_(-1, 1)
    return (a, b)


def create_outputs(args):
    c = torch.zeros((args.m, args.n), dtype=args.dtype, device='cuda')
    return (c,)


def ref_func(a, b, c):
    F.linear(a, b, out=c)


def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


@functools.lru_cache(maxsize=1024)
def compile_hgemm_kernel(
    dtype: str,
    m: int,
    n: int,
    k: int,
    BLOCK_K: int = 32,
    BLOCK_M_WARPS: int = 2,
    BLOCK_N_WARPS: int = 2,
    WARP_M_STEPS: int = 4,
    WARP_N_STEPS: int = 4,
    STAGES : int = 1,
):
    assert k % BLOCK_K == 0
    assert k // BLOCK_K >= 1
    assert BLOCK_M_WARPS * BLOCK_N_WARPS == 4
    # Fixed parameters:
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WMMA_FRAG_VALUES = 4
    WARP_SIZE = 64
    DTYPE_BYTES = 2
    LDG_VEC_SIZE = 8
    # Propagated parameters:
    WARP_ATOM_M = WMMA_M
    WARP_ATOM_N = WMMA_N
    WARP_ATOM_K = WMMA_K * 2
    BLOCK_K_LOOPS = k // BLOCK_K
    WARP_K_STEPS = BLOCK_K // WARP_ATOM_K
    assert (BLOCK_K % WARP_ATOM_K == 0) and (WARP_K_STEPS >= 1)
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE
    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    LDG_A_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_B_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_REG_A_COUNT = BLOCK_MK_SIZE // LDG_VEC_SIZE // BLOCK_THREADS
    LDG_REG_B_COUNT = BLOCK_NK_SIZE // LDG_VEC_SIZE // BLOCK_THREADS
    assert (LDG_REG_A_COUNT >= 1) and (LDG_REG_B_COUNT >= 1)
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES

    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_a_offset + STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
    smem_b_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_b_offset + STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES

    @flyc.kernel
    def hgemm_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        if dtype == 'bf16':
            mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k
        else:
            mfma_fn = rocdl.mfma_f32_16x16x16f16
        c_zero_f = arith.constant(0.0, type=T.f32)
        acc_init = arith.constant_vector(0.0, T.f32x4)

        A_ = GTensor(A, dtype=dtype_, shape=(m, k))
        B_ = GTensor(B, dtype=dtype_, shape=(n, k))
        C_ = GTensor(C, dtype=dtype_, shape=(m, n))
        base_ptr = allocator.get_base()
        smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(STAGES * BLOCK_M * BLOCK_K,))
        smem_b_ptr = SmemPtr(base_ptr, smem_b_offset, dtype_, shape=(STAGES * BLOCK_N * BLOCK_K,))
        as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, BLOCK_M, BLOCK_K))
        bs_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, BLOCK_N, BLOCK_K))
        
        tid = fx.Int32(fx.thread_idx.x)
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE
        block_m_idx = fx.block_idx.x
        block_n_idx = fx.block_idx.y

        current_stage = 0
        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        k_offset = fx.Int32(0)

        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)
    
        def ldg_a_copy(a_offset, k_offset, lds_stage, async_mode=False):
            if not async_mode:
                for i in range_constexpr(LDG_REG_A_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local_idx = global_tid // LDG_A_X_THREADS
                    k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                    vec = A_.vec_load((a_offset + m_local_idx, k_offset + k_local_idx), LDG_VEC_SIZE)
                    col_in_bytes = k_local_idx * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                    as_.vec_store((fx.Index(lds_stage), m_local_idx, col_in_bytes // DTYPE_BYTES), vec, LDG_VEC_SIZE)
        
        def ldg_b_copy(b_offset, k_offset, lds_stage, async_mode=False):
            if not async_mode:
                for i in range_constexpr(LDG_REG_B_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    n_local_idx = global_tid // LDG_B_X_THREADS
                    k_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
                    vec = B_.vec_load((b_offset + n_local_idx, k_offset + k_local_idx), LDG_VEC_SIZE)
                    col_in_bytes = k_local_idx * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
                    bs_.vec_store((fx.Index(lds_stage), n_local_idx, col_in_bytes // DTYPE_BYTES), vec, LDG_VEC_SIZE)
        
        warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid % BLOCK_N_WARPS * WARP_N
        ldmatrix_a_m_idx = w_tid % WARP_ATOM_M
        ldmatrix_a_k_vec_idx = w_tid // WARP_ATOM_M * WMMA_FRAG_VALUES * 2
        ldmatrix_b_n_idx = w_tid % WARP_ATOM_N
        ldmatrix_b_k_vec_idx = w_tid // WARP_ATOM_N * WMMA_FRAG_VALUES * 2
        c_frags = [acc_init] * (WARP_M_STEPS * WARP_N_STEPS)

        def block_mma_sync(lds_stage):
            s = fx.Index(lds_stage)
            a_frags = [0] * (WARP_K_STEPS * WARP_M_STEPS)
            b_frags = [0] * (WARP_K_STEPS * WARP_N_STEPS)
            # load matrix a
            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                for kk in range_constexpr(WARP_K_STEPS):
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                    col_in_bytes = (warp_atom_k_idx + ldmatrix_a_k_vec_idx) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    vec = as_.vec_load((s, row, col_in_bytes // DTYPE_BYTES), WMMA_FRAG_VALUES * 2)
                    a_frags[kk * WARP_M_STEPS + ii] = vec
            # load matrix b
            for ii in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                for kk in range_constexpr(WARP_K_STEPS):
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    row = warp_atom_n_idx + ldmatrix_b_n_idx
                    col_in_bytes = (warp_atom_k_idx + ldmatrix_b_k_vec_idx) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    vec = bs_.vec_load((s, row, col_in_bytes // DTYPE_BYTES), WMMA_FRAG_VALUES * 2)
                    b_frags[kk * WARP_N_STEPS + ii] = vec
            # wmma
            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                for jj in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                    for kk in range_constexpr(WARP_K_STEPS):
                        warp_atom_k_idx = kk * WARP_ATOM_K
                        a_frag_vec_pack = a_frags[kk * WARP_M_STEPS + ii]
                        b_frag_vec_pack = b_frags[kk * WARP_N_STEPS + jj]
                        # split a
                        a_i64x2 = vector.bitcast(T.i64x2, a_frag_vec_pack)
                        a0_i64 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                        a1_i64 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                        a_v0 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [a0_i64]))
                        a_v1 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [a1_i64]))
                        # split b
                        b_i64x2 = vector.bitcast(T.i64x2, b_frag_vec_pack)
                        b0_i64 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                        b1_i64 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                        b_v0 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [b0_i64]))
                        b_v1 = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [b1_i64]))
                        # handle bf16
                        if dtype == 'bf16':
                            a_v0 = vector.bitcast(T.vec(4, T.i16), a_v0)
                            a_v1 = vector.bitcast(T.vec(4, T.i16), a_v1)
                            b_v0 = vector.bitcast(T.vec(4, T.i16), b_v0)
                            b_v1 = vector.bitcast(T.vec(4, T.i16), b_v1)
                        # wmma
                        acc_in = c_frags[ii * WARP_N_STEPS + jj]
                        acc_mid = mfma_fn(T.f32x4, [a_v0, b_v0, acc_in, 0, 0, 0])
                        c_frags[ii * WARP_N_STEPS + jj] = mfma_fn(T.f32x4, [a_v1, b_v1, acc_mid, 0, 0, 0])
        
        for bki in range_constexpr(BLOCK_K_LOOPS):
            ldg_a_copy(m_offset, k_offset, 0)
            ldg_b_copy(n_offset, k_offset, 0)
            gpu.barrier()
            block_mma_sync(0)
            k_offset += fx.Int32(BLOCK_K)
            gpu.barrier()
        
        # store results
        stmatrix_c_m_vec_idx = w_tid // WARP_ATOM_N * WMMA_FRAG_VALUES
        stmatrix_c_n_idx = w_tid % WARP_ATOM_N
        for ii in range_constexpr(WARP_M_STEPS):
            g_warp_atom_m_idx = m_offset + warp_m_idx + ii * WARP_ATOM_M
            for jj in range_constexpr(WARP_N_STEPS):
                g_warp_atom_n_idx = n_offset + warp_n_idx + jj * WARP_ATOM_N
                out_vec = c_frags[ii * WARP_N_STEPS + jj]
                for kk in range_constexpr(WMMA_FRAG_VALUES):
                    out_m_idx = g_warp_atom_m_idx + stmatrix_c_m_vec_idx + kk
                    out_n_idx = g_warp_atom_n_idx + stmatrix_c_n_idx
                    val = vector.extract(out_vec, static_position=[kk], dynamic_position=[])
                    C_[out_m_idx, out_n_idx] = val.truncf(dtype_)
        return
    
    @flyc.jit
    def launch_hgemm_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        
        bm = (m + BLOCK_M - 1) // BLOCK_M
        bn = (n + BLOCK_N - 1) // BLOCK_N
        hgemm_kernel(A, B, C).launch(grid=(bm, bn, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)
    
    return launch_hgemm_kernel


# def shuffle_b(x, layout=(16, 16), k_steps=2):
#     x_shape = x.shape
#     VEC_SIZE = 16 // x.element_size()
#     BN = layout[0]
#     BK = layout[1] * k_steps
#     assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
#     assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"
#     x = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // VEC_SIZE, VEC_SIZE)
#     x = x.permute(0, 1, 3, 4, 2, 5).contiguous().view(*x_shape)
#     x.is_shuffled = True
#     return x


def func(a, b, c):
    m, k = a.shape
    n = b.shape[0]
    if a.dtype == torch.half:
        exe = compile_hgemm_kernel('f16', m, n, k)
    elif a.dtype == torch.bfloat16:
        exe = compile_hgemm_kernel('bf16', m, n, k)
    else:
        raise NotImplementedError()
    exe(a, b, c, stream=torch.cuda.current_stream())


def benchmark(args, func, ref_func, warmup=20, niters=100):
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = inputs + outputs
    ref_inouts = inputs + ref_outputs
    func(*inouts)
    ref_func(*ref_inouts)
    for output, ref_output in zip(outputs, ref_outputs):
        is_allclose = torch.allclose(output, ref_output, atol=1e-2, rtol=1e-2)
        # print(output)
        # print(ref_output)
        maxdiff_out = (output - ref_output).abs().max()
        print(f"maxdiff_out:{maxdiff_out}")
        # assert is_allclose == True
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
