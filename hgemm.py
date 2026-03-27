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
from flydsl._mlir.dialects import llvm, fly, memref, scf
from flydsl.compiler.protocol import fly_values

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
    c = torch.randn((args.m, args.n), dtype=args.dtype, device='cuda')
    return (c,)


def create_ref_outputs(args):
    c = torch.randn((args.m, args.n), dtype=args.dtype, device='cuda')
    return (c,)


def ref_func(a, b, c):
    F.linear(a, b, out=c)


def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


@functools.lru_cache(maxsize=1024)
def compile_hgemm_kernel(
    dtype: str,
    n: int,
    k: int,
    TILE_K: int = 64,
    BLOCK_M_WARPS: int = 1,
    BLOCK_N_WARPS: int = 4,
    TILE_M: int = 128,
    TILE_N: int = 128,
    STAGES : int = 2,
    ASYNC_COPY: bool = False,
    B_TO_LDS: bool = False,
    B_PRE_SHUFFLE: bool = True,
    SPLIT_K: int = 1,
    C_TO_LDS: bool = False,
):
    IS_SPLIT_K = SPLIT_K > 1
    BLOCK_K = TILE_K
    assert (k % SPLIT_K == 0) and (k // SPLIT_K >= 1)
    ks = k // SPLIT_K
    assert (ks % BLOCK_K == 0) and (ks // BLOCK_K >= 1)
    assert BLOCK_K >= 32
    assert BLOCK_M_WARPS * BLOCK_N_WARPS == 4
    assert STAGES in [2, 1]
    if B_PRE_SHUFFLE == True:
        assert B_TO_LDS == False

    # Fixed parameters:
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WMMA_FRAG_VALUES = 4
    WARP_SIZE = 64
    DTYPE_BYTES = 2
    LDG_VEC_SIZE = 8

    # Propagated parameters:
    MFMA_PER_WARP_K = LDG_VEC_SIZE // WMMA_FRAG_VALUES
    WARP_ATOM_M = WMMA_M
    WARP_ATOM_N = WMMA_N
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K
    BLOCK_K_LOOPS = ks // BLOCK_K
    WARP_K_STEPS = BLOCK_K // WARP_ATOM_K
    assert (BLOCK_K % WARP_ATOM_K == 0) and (WARP_K_STEPS >= 1)
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WARP_ATOM_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WARP_ATOM_N
    assert (WARP_M_STEPS >= 1) and (WARP_N_STEPS >= 1)
    assert TILE_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
    assert TILE_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    assert (n >= BLOCK_N) and (n % BLOCK_N == 0)
    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    LDG_A_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_B_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_C_X_THREADS = BLOCK_N // LDG_VEC_SIZE
    LDG_REG_A_COUNT = BLOCK_MK_SIZE // LDG_VEC_SIZE // BLOCK_THREADS
    LDG_REG_B_COUNT = BLOCK_NK_SIZE // LDG_VEC_SIZE // BLOCK_THREADS
    LDG_REG_C_COUNT = BLOCK_M * BLOCK_N // LDG_VEC_SIZE // BLOCK_THREADS
    assert (LDG_REG_A_COUNT >= 1) and (LDG_REG_B_COUNT >= 1)
    if IS_SPLIT_K:
        assert LDG_REG_C_COUNT >= 1
        C_TO_LDS = True
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES

    # LDS parameters:
    gpu_arch = get_rocm_arch()
    DMA_BYTES = 4 if gpu_arch == "gfx942" else 16
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    AS_BYTES = STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
    if C_TO_LDS:
        AS_BYTES = max(AS_BYTES, BLOCK_M * BLOCK_N * DTYPE_BYTES)
    allocator.ptr = smem_a_offset + AS_BYTES
    if B_TO_LDS:
        smem_b_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = smem_b_offset + STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES

    KERNEL_NAME = f"hgemm_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_S{STAGES}TN"
    if B_PRE_SHUFFLE:
        KERNEL_NAME += "_BP"
    if IS_SPLIT_K:
        KERNEL_NAME += f"_SPK{SPLIT_K}"
    elif C_TO_LDS:
        KERNEL_NAME += f"_CL"

    @flyc.kernel
    def hgemm_kernel(
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        m: fx.Int32,
        raster_factor: fx.Constexpr[int],
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        if dtype == 'bf16':
            mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k
        else:
            mfma_fn = rocdl.mfma_f32_16x16x16f16
        c_zero_f = arith.constant(0.0, type=T.f32)
        c_zero_d = arith.constant(0.0, type=dtype_)
        acc_init = arith.constant_vector(0.0, T.f32x4)

        A_ = GTensor(A, dtype=dtype_, shape=(-1, k))
        B_ = GTensor(B, dtype=dtype_, shape=(n, k))
        C_ = GTensor(C, dtype=dtype_, shape=(-1, n))
        base_ptr = allocator.get_base()
        smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(STAGES * BLOCK_M * BLOCK_K,))
        as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, BLOCK_M, BLOCK_K))
        if B_TO_LDS:
            smem_b_ptr = SmemPtr(base_ptr, smem_b_offset, dtype_, shape=(STAGES * BLOCK_N * BLOCK_K,))
            bs_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, BLOCK_N, BLOCK_K))
        if C_TO_LDS:
            smem_c_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(BLOCK_M * BLOCK_N,))
            cs_ = STensor(smem_c_ptr, dtype_, shape=(BLOCK_M, BLOCK_N))
        if B_PRE_SHUFFLE:
            # origin: n // WARP_ATOM_N, WARP_ATOM_N, k // WARP_ATOM_K, WARP_ATOM_K // LDG_VEC_SIZE, LDG_VEC_SIZE
            SHUFFLED_B_ = GTensor(B, dtype=dtype_, shape=(
                n // WARP_ATOM_N, k // WARP_ATOM_K, WARP_ATOM_K // LDG_VEC_SIZE, WARP_ATOM_N, LDG_VEC_SIZE))
        
        tid = fx.Int32(fx.thread_idx.x)
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE
        block_m_idx = fx.block_idx.x // raster_factor
        block_n_idx = fx.block_idx.x % raster_factor + fx.block_idx.y * raster_factor
        ks_idx = fx.Index(fx.block_idx.z)
        ks_begin = arith.index_cast(T.i32, ks_idx * ks)

        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)

        warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid % BLOCK_N_WARPS * WARP_N
        ldmatrix_a_m_idx = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_FRAG_VALUES * MFMA_PER_WARP_K
        ldmatrix_b_n_idx = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_FRAG_VALUES * MFMA_PER_WARP_K
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
        c_frags = [acc_init] * C_FRAGS_LEN
        
        def ldg_a(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                row_idx = m_offset + fx.Index(m_local_idx)
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset + k_local_idx)
                vec = A_.vec_load((safe_row_idx, col_idx), LDG_VEC_SIZE)
                vecs.append(vec)
            return vecs
        
        def sts_a(vecs, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                as_.vec_store((fx.Index(lds_stage), m_local_idx, col_in_bytes // DTYPE_BYTES), vecs[i], LDG_VEC_SIZE)
        
        def ldg_sts_a_async(k_offset, lds_stage):
            LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
            LDG_A_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
            LDG_REG_A_COUNT_AS = BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
            for i in range_constexpr(LDG_REG_A_COUNT_AS):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS_AS
                k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                row_idx = m_offset + fx.Index(m_local_idx)
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
                # get offset
                global_offset = A_.linear_offset((safe_row_idx, col_idx)) * DTYPE_BYTES
                global_offset = arith.index_cast(T.i32, global_offset)
                lds_offset = as_.linear_offset((fx.Index(lds_stage), m_local_idx, k_local_idx)) * DTYPE_BYTES
                # get lds ptr
                lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                lds_addr = memref.extract_aligned_pointer_as_index(as_.memptr) + lds_offset
                lds_addr_ = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                lds_ptr = llvm.inttoptr(lds_ptr_type, lds_addr_)
                # dma copy
                rocdl.raw_ptr_buffer_load_lds(
                    A_.rsrc,
                    lds_ptr,
                    arith.constant(DMA_BYTES, type=T.i32),
                    global_offset,
                    arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32),
                    arith.constant(1, type=T.i32),
                )
                # vec = A_.vec_load((m_offset + m_local_idx, k_offset + col_in_bytes // DTYPE_BYTES), LDG_ASYNC_VEC_SIZE)
                # as_.vec_store((fx.Index(lds_stage), m_local_idx, k_local_idx), vec, LDG_ASYNC_VEC_SIZE)
        
        def lds_matrix_a(lds_stage):
            s = fx.Index(lds_stage)
            a_frags = [0] * (WARP_K_STEPS * WARP_M_STEPS)
            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                for kk in range_constexpr(WARP_K_STEPS):
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                    col_in_bytes = (warp_atom_k_idx + ldmatrix_a_k_vec_idx) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    vec = as_.vec_load((s, row, col_in_bytes // DTYPE_BYTES), WMMA_FRAG_VALUES * MFMA_PER_WARP_K)
                    a_frags[kk * WARP_M_STEPS + ii] = vec
            return a_frags
        
        def ldg_matrix_b(k_offset):
            vecs = []
            b_n_intra_base = ldmatrix_b_n_idx
            b_k_intra_vec = ldmatrix_b_k_vec_idx // LDG_VEC_SIZE
            b_n0_base = n_offset // WARP_ATOM_N + warp_n_idx // WARP_ATOM_N
            b_k0_base = k_offset // WARP_ATOM_K
            for kk in range_constexpr(WARP_K_STEPS):
                b_k0 = b_k0_base + kk
                for ii in range_constexpr(WARP_N_STEPS):
                    b_n0 = b_n0_base + ii
                    if not B_PRE_SHUFFLE:
                        warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                        warp_atom_k_idx = kk * WARP_ATOM_K
                        n_idx = n_offset + warp_atom_n_idx + ldmatrix_b_n_idx
                        k_idx = k_offset + warp_atom_k_idx + ldmatrix_b_k_vec_idx
                        vec = B_.vec_load((n_idx, k_idx), WMMA_FRAG_VALUES * MFMA_PER_WARP_K)
                        vecs.append(vec)
                    else:
                        b_n_intra = b_n_intra_base  # idx_1
                        vec = SHUFFLED_B_.vec_load((b_n0, b_k0, b_k_intra_vec, b_n_intra, 0), LDG_VEC_SIZE)
                        vecs.append(vec)
            return vecs
        
        def block_mma_sync(a_frags, b_frags, c_frags):
            # wmma
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag_vec_pack = a_frags[kk * WARP_M_STEPS + ii]
                    for jj in range_constexpr(WARP_N_STEPS):
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
                        c_idx = ii * WARP_N_STEPS + jj
                        acc_in = c_frags[c_idx]
                        acc_mid = mfma_fn(T.f32x4, [a_v0, b_v0, acc_in, 0, 0, 0])
                        c_frags[c_idx] = mfma_fn(T.f32x4, [a_v1, b_v1, acc_mid, 0, 0, 0])
        
        if B_TO_LDS:
            # SLOW PATH
            raise NotImplementedError("B_TO_LDS not supported yet")
        else:

            if True:
                # ============ Main K-loop with scheduling ============
                # Initial scheduling barrier to reset hardware scheduler state
                rocdl.sched_barrier(0)
                def hot_loop_scheduler():
                    import math as _math

                    def _build_scheduler(numer: int, denom: int):
                        if denom <= 0:
                            return []
                        if numer <= 0:
                            return [0] * denom
                        out = []
                        prev = 0
                        for i in range_constexpr(denom):
                            cur = ((i + 1) * numer + (denom - 1)) // denom
                            out.append(cur - prev)
                            prev = cur
                        return out

                    if (gpu_arch == "gfx942") or (not ASYNC_COPY):
                        mfma_group =  WARP_N_STEPS
                        mfma_total = (WARP_K_STEPS  * 2) * WARP_M_STEPS * mfma_group
                        mfma_per_iter = 2 * mfma_group
                        sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                        rocdl.sched_dsrd(2)
                        rocdl.sched_mfma(1)
                        if TILE_M == 16:
                            rocdl.sched_vmem(1)
                        rocdl.sched_mfma(1)
                        if TILE_M == 16:
                            rocdl.sched_vmem(1)
                        if mfma_group < 4:
                            rocdl.sched_dsrd(1)
                            rocdl.sched_mfma(1)
                            if TILE_M == 16:
                                rocdl.sched_vmem(1)
                            rocdl.sched_dsrd(1)
                            rocdl.sched_mfma(1)
                            if TILE_M == 16:
                                rocdl.sched_vmem(1)
                            rocdl.sched_mfma(1)
                        dswr_tail = LDG_REG_A_COUNT
                        dstr_advance = 2
                        if dswr_tail > sche_iters:
                            dswr_tail = sche_iters
                        dswr_start = max(sche_iters - dswr_tail - dstr_advance, 0)
                        for sche_i in range_constexpr(sche_iters):
                            rocdl.sched_vmem(1)
                            rocdl.sched_mfma(mfma_group)
                            rocdl.sched_dsrd(1)
                            rocdl.sched_mfma(mfma_group)
                            if sche_i >= dswr_start - 1:
                                rocdl.sched_dswr(1)
                    rocdl.sched_barrier(0)

            a_regs = ldg_a(ks_begin)
            sts_a(a_regs, 0)
            b_frags = ldg_matrix_b(ks_begin)
            gpu.barrier()

            init_state = [ks_begin, arith.constant(0, index=True)] + c_frags + b_frags
            for bki, state in range(1, BLOCK_K_LOOPS, init=init_state):
                k_offset = state[0]
                if STAGES == 2:
                    current_stage = fx.Index(state[1])
                    next_stage = 1 - current_stage
                else:
                    current_stage = next_stage = 0
                c_frags = state[2 : 2 + C_FRAGS_LEN]
                b_frags = state[2 + C_FRAGS_LEN :]
                if not ASYNC_COPY:
                    a_regs_next = ldg_a(k_offset + BLOCK_K)
                else:
                    ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
                b_frags_next = ldg_matrix_b(k_offset + BLOCK_K)
                a_frags = lds_matrix_a(current_stage)
                block_mma_sync(a_frags, b_frags, c_frags)
                if STAGES == 1:
                    gpu.barrier()
                if not ASYNC_COPY:
                    sts_a(a_regs_next, next_stage)
                hot_loop_scheduler()
                gpu.barrier()
                k_offset = k_offset + fx.Int32(BLOCK_K)
                results = yield [k_offset, next_stage if STAGES == 2 else arith.constant(0, index=True)] + c_frags + b_frags_next
            
            k_offset = results[0]
            current_stage = results[1] if STAGES == 2 else 0
            c_frags = results[2 : 2 + C_FRAGS_LEN]
            b_frags = results[2 + C_FRAGS_LEN :]
            a_frags = lds_matrix_a(current_stage)
            block_mma_sync(a_frags, b_frags, c_frags)
        
        # store results
        stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_FRAG_VALUES
        stmatrix_c_n_idx = w_tid % WMMA_N
        if not C_TO_LDS:
            for ii in range_constexpr(WARP_M_STEPS):
                g_warp_atom_m_idx = m_offset + warp_m_idx + ii * WARP_ATOM_M
                for jj in range_constexpr(WARP_N_STEPS):
                    g_warp_atom_n_idx = n_offset + warp_n_idx + jj * WARP_ATOM_N
                    for kk in range_constexpr(WMMA_FRAG_VALUES):
                        out_m_idx = g_warp_atom_m_idx + stmatrix_c_m_vec_idx + kk
                        if arith.cmpi(arith.CmpIPredicate.ult, out_m_idx, fx.Index(m)):
                            out_n_idx = g_warp_atom_n_idx + stmatrix_c_n_idx
                            val = vector.extract(c_frags[ii * WARP_N_STEPS + jj], static_position=[kk], dynamic_position=[])
                            C_[out_m_idx, out_n_idx] = val.truncf(dtype_)
        else:
            gpu.barrier()
            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                for jj in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                    for kk in range_constexpr(WMMA_FRAG_VALUES):
                        lds_m_idx = fx.Index(warp_atom_m_idx + stmatrix_c_m_vec_idx + kk)
                        lds_n_idx = fx.Index(warp_atom_n_idx + stmatrix_c_n_idx)
                        val = vector.extract(c_frags[ii * WARP_N_STEPS + jj], static_position=[kk], dynamic_position=[])
                        cs_[lds_m_idx, lds_n_idx] = val.truncf(dtype_)
            gpu.barrier()
            if IS_SPLIT_K:
                _ptr_type = ir.Type.parse("!llvm.ptr<1>")
                _i64_type = T.i64
                out_raw = fly_values(C)[0]
                out_base_ptr = fly.extract_aligned_pointer_as_index(_ptr_type, out_raw)
                out_base_int = llvm.PtrToIntOp(_i64_type, out_base_ptr).result
                for i in range_constexpr(LDG_REG_C_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                    n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                    m_global_idx = m_offset + m_local_idx
                    n_global_idx = n_offset + n_local_idx
                    if arith.cmpi(arith.CmpIPredicate.ult, m_global_idx, fx.Index(m)):
                        pk_val = cs_.vec_load((m_local_idx, n_local_idx), LDG_VEC_SIZE)
                        linear_bytes_offset = C_.linear_offset((m_global_idx, n_global_idx)) * DTYPE_BYTES
                        byte_offset_i64 = arith.index_cast(T.i64, linear_bytes_offset)
                        addr_i64 = llvm.AddOp(out_base_int, byte_offset_i64, llvm.IntegerOverflowFlags(0)).result
                        out_ptr = llvm.IntToPtrOp(_ptr_type, addr_i64).result
                        out_ptr_v = out_ptr._value if hasattr(out_ptr, "_value") else out_ptr
                        pk_val_v = pk_val._value if hasattr(pk_val, "_value") else pk_val
                        # split to vec2s
                        vec2_ty = T.vec(2, dtype_)
                        for vec_idx in range_constexpr(LDG_VEC_SIZE // 2):
                            e0 = vector.extract(pk_val, static_position=[vec_idx * 2], dynamic_position=[])
                            e1 = vector.extract(pk_val, static_position=[vec_idx * 2 + 1], dynamic_position=[])
                            pair = vector.from_elements(vec2_ty, [e0, e1])
                            pair_byte_offset = arith.index_cast(T.i64, linear_bytes_offset + fx.Index(vec_idx * 2 * DTYPE_BYTES))
                            pair_addr_i64 = llvm.AddOp(out_base_int, pair_byte_offset, llvm.IntegerOverflowFlags(0)).result
                            pair_ptr = llvm.IntToPtrOp(_ptr_type, pair_addr_i64).result
                            pair_ptr_v = pair_ptr._value if hasattr(pair_ptr, "_value") else pair_ptr
                            pair_v = pair._value if hasattr(pair, "_value") else pair
                            llvm.AtomicRMWOp(
                                llvm.AtomicBinOp.fadd,
                                pair_ptr_v,
                                pair_v,
                                llvm.AtomicOrdering.monotonic,
                                syncscope="agent",
                                alignment=4,
                            )
            else:
                for i in range_constexpr(LDG_REG_C_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                    n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                    m_global_idx = m_offset + m_local_idx
                    if arith.cmpi(arith.CmpIPredicate.ult, m_global_idx, fx.Index(m)):
                        vec = cs_.vec_load((m_local_idx, n_local_idx), LDG_VEC_SIZE)
                        C_.vec_store((m_global_idx, n_offset + n_local_idx), vec, LDG_VEC_SIZE)
        return
    
    @flyc.jit
    def launch_hgemm_kernel(
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        
        bm = (m + BLOCK_M - 1) // BLOCK_M
        bn = n // BLOCK_N
        raster_factor = 1
        bm = bm * raster_factor
        bn = (bn + raster_factor - 1) // raster_factor
        hgemm_kernel._func.__name__ = KERNEL_NAME
        hgemm_kernel(C, A, B, m, raster_factor).launch(grid=(bm, bn, SPLIT_K), block=(BLOCK_THREADS, 1, 1), stream=stream)
    
    return launch_hgemm_kernel


def hgemm_shuffle_b(x, layout=(16, 16), pack_n=1, k_steps=2):
    x_shape = x.shape
    VEC_SIZE = 16 // x.element_size()
    BN = layout[0] * pack_n
    BK = layout[1] * k_steps
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"
    x = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // VEC_SIZE, VEC_SIZE)
    x = x.permute(0, 1, 3, 4, 2, 5).contiguous()
    x = x.view(*x_shape)
    x.is_shuffled = True
    return x


def get_default_kwargs(m, n, k):
    kwargs = {
        'TILE_K': 64,
        'BLOCK_M_WARPS': 1,
        'BLOCK_N_WARPS': 4,
        'TILE_M': 128,
        'TILE_N': 128,
        'STAGES' : 1,
        'ASYNC_COPY': False,
        'B_TO_LDS': False,
        'B_PRE_SHUFFLE': True,
        'SPLIT_K': 1,
        'C_TO_LDS': False,
    }
    if m <= 32 and n == 7168 and k == 2048:
        kwargs['TILE_K'] = 128
        kwargs['TILE_M'] = 32
        kwargs['TILE_N'] = 128
    if m <= 32 and n == 384 and k == 7168:
        kwargs['TILE_K'] = 128
        kwargs['TILE_M'] = 16
        kwargs['TILE_N'] = 128
    return kwargs


selections = {
    'TILE_K': [64, 128],
    'TILE_M': [16, 32, 48, 64, 96, 128],
    'TILE_N': [64, 128, 256],
}


def hgemm_(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    shuffle_b: bool=True,
    hgemm_kwargs: dict={},
):
    k = a.shape[-1]
    a = a.view(-1, k)
    m = a.shape[0]
    n = b.shape[0]
    assert b.shape[1] == k
    c = c.view(-1, n)
    assert c.shape[0] == m
    kwargs = get_default_kwargs(m, n, k)
    kwargs.update(hgemm_kwargs)
    if a.dtype == torch.half:
        exe = compile_hgemm_kernel('f16', n, k, **kwargs)
    elif a.dtype == torch.bfloat16:
        exe = compile_hgemm_kernel('bf16', n, k, **kwargs)
    else:
        raise NotImplementedError()
    if kwargs['B_PRE_SHUFFLE'] and shuffle_b:
        b = hgemm_shuffle_b(b)
    if kwargs['SPLIT_K'] > 1:
        c.zero_() # TODO: remove it
    exe(c, a, b, m, stream=torch.cuda.current_stream())


def func(a, b, c):
    hgemm_(a, b, c, shuffle_b=True)


def benchmark(args, func, ref_func, warmup=20, niters=100):
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_ref_outputs(args)
    inouts = inputs + outputs
    ref_inouts = inputs + ref_outputs
    for i in range(5):
        func(*inouts)
        ref_func(*ref_inouts)
        for output, ref_output in zip(outputs, ref_outputs):
            is_allclose = torch.allclose(output, ref_output, atol=1e-2, rtol=1e-2)
            # print(output)
            # print(ref_output)
            maxdiff_out = (output - ref_output).abs().max()
            print(f"maxdiff_out:{maxdiff_out}")
            # assert is_allclose == True

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
    # rm -rf ~/.flydsl/ ; python3 hgemm.py --m=4096 --n=4096 --k=4096 --dtype=bf16
    # rm -rf ~/.flydsl/ ; python3 hgemm.py --m=32 --n=7168 --k=2048 --dtype=bf16
    # rm -rf ~/.flydsl/ ; python3 hgemm.py --m=8192 --n=8192 --k=8192 --dtype=bf16
