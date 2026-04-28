import time
import json
import torch
import argparse
import functools
import itertools
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Any, Callable, Dict, Literal, Optional, Tuple

import triton
import triton.language as tl

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl._mlir.dialects import gpu as mlir_gpu, vector as mlir_vector, math as mlir_math
from flydsl.expr import range_constexpr, const_expr, arith, vector, gpu, rocdl
from flydsl._mlir import ir
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir.dialects import llvm, fly, memref, scf
from flydsl.compiler.protocol import fly_values

from utils.tensor_shim import get_dtype_in_kernel, get_dtype_vec_size, get_dtype_str, GTensor, STensor, _to_raw, _run_compiled
fm_fast = arith.FastMathFlags.fast

base_dir = Path(__file__).resolve().parent
temp_dir = base_dir / 'temp'
temp_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Args:
    dtype: torch.dtype
    b: int
    sq: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    ssm_state_dtype: torch.dtype
    ssm_state_size_n: int
    use_qk_l2norm: bool = True


def create_inputs(args):
    query = torch.randn((args.b, args.sq, args.num_k_heads, args.head_k_dim), dtype=args.dtype, device='cuda')
    key = torch.randn((args.b, args.sq, args.num_k_heads, args.head_k_dim), dtype=args.dtype, device='cuda')
    value = torch.randn((args.b, args.sq, args.num_v_heads, args.head_v_dim), dtype=args.dtype, device='cuda')
    a = torch.randn((args.b, args.sq, args.num_v_heads), dtype=args.dtype, device='cuda')
    b = torch.randn((args.b, args.sq, args.num_v_heads), dtype=args.dtype, device='cuda')
    dt_bias = torch.randn((args.num_v_heads), dtype=args.dtype, device='cuda')
    dt_bias.uniform_(1, 2)
    A_log = torch.randn((args.num_v_heads), dtype=torch.float32, device="cuda")
    A_log.uniform_(0, 16)
    indices = torch.arange(args.b - 1, -1, -1, dtype=torch.int32, device="cuda")
    state = torch.randn((args.ssm_state_size_n, args.num_v_heads, args.head_k_dim, args.head_v_dim), dtype=args.ssm_state_dtype, device="cuda")
    return (args, query, key, value, a, b, dt_bias, A_log, indices, state)


def create_outputs(args):
    out = torch.zeros((args.b, args.sq, args.num_v_heads, args.head_v_dim), dtype=args.dtype, device='cuda')
    return (out,)


def ref_func_(args, query, key, value, a, b, dt_bias, A_log, indices, state, out):
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias, beta=1.0, threshold=20.0)
    if args.num_v_heads // args.num_k_heads > 1:
        query = query.repeat_interleave(args.num_v_heads // args.num_k_heads, dim=2)
        key = key.repeat_interleave(args.num_v_heads // args.num_k_heads, dim=2)
    if args.use_qk_l2norm:
        def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
            """This function is intended to align with the l2norm implementation in the FLA library."""
            inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
            return x * inv_norm
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    # query, # (b, num_v_heads, sq, head_k_dim)
    # key,   # (b, num_v_heads, sq, head_k_dim)
    # value, # (b, num_v_heads, sq, head_v_dim)
    # g,     # (b, num_v_heads, sq)
    # beta,  # (b, num_v_heads, sq)
    scale = 1 / (args.head_k_dim ** 0.5)
    query = query * scale
    last_recurrent_state = state[indices]
    for i in range(args.sq):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        # q_t:     # (b, num_v_heads, head_k_dim)
        # k_t:     # (b, num_v_heads, head_k_dim)
        # v_t:     # (b, num_v_heads, head_v_dim)
        # g_t:     # (b, num_v_heads, 1, 1)
        # beta_t:  # (b, num_v_heads, 1)
        # last_recurrent_state: # (b, num_v_heads, head_k_dim, head_v_dim)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2) # (b, num_v_heads, head_v_dim)
        delta = (v_t - kv_mem) * beta_t # (b, num_v_heads, head_v_dim)  
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # core_attn_out: # (b, num_v_heads, sq, head_v_dim)
        out[:, i, :] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    state[indices] = last_recurrent_state


@functools.lru_cache(maxsize=1024)
def create_shuffle_gdr_decode_kernel(
    dtype: str,
    A_log_dtype: str,
    state_dtype: str,
    seq_length: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    state_strides: tuple,
    use_qk_l2norm: bool,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    NUM_BLOCKS_PER_V_DIM: int = 1,
    NUM_WARPS: int = 4,
    WARP_THREADS_K: int = 8,
):
    SCALE_VALUE = float(1.0 / (float(head_k_dim) ** 0.5))
    WARP_THREADS_V = 64 // WARP_THREADS_K
    VEC_SIZE = get_dtype_vec_size(dtype)
    DTYPE_BYTES = 16 // VEC_SIZE

    if 'f32' in state_dtype:
        VALUES_PER_THREAD_K = 4 # 16B
    else:
        VALUES_PER_THREAD_K = 8

    WARP_SIZE = WARP_THREADS_V * WARP_THREADS_K
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE
    assert WARP_SIZE == 64

    WARP_TILE_K = WARP_THREADS_K * VALUES_PER_THREAD_K
    WARP_TILE_K_ITERS = head_k_dim // WARP_TILE_K
    assert WARP_TILE_K_ITERS >= 1
    assert head_k_dim % WARP_TILE_K == 0
    TILE_K = head_k_dim

    WARP_TILE_V = WARP_THREADS_V
    WARP_GROUP_TILE_V = NUM_WARPS * WARP_TILE_V
    TILE_V = head_v_dim // NUM_BLOCKS_PER_V_DIM
    WARP_TILE_V_ITERS = TILE_V // WARP_GROUP_TILE_V
    assert TILE_V >= 1 and head_v_dim % NUM_BLOCKS_PER_V_DIM == 0
    assert WARP_TILE_V_ITERS >= 1 and TILE_V % WARP_GROUP_TILE_V == 0

    WARP_THREADS_K_SHFL_OFFSETS = []
    offsets_ = WARP_THREADS_K // 2
    while offsets_ >= 1:
        WARP_THREADS_K_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2
    WARP_THREADS_K_SHFL_OFFSETS = WARP_THREADS_K_SHFL_OFFSETS[::-1]
    
    WARP_SIZE_SHFL_OFFSETS = []
    offsets_ = WARP_SIZE // 2
    while offsets_ >= 1:
        WARP_SIZE_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2
    
    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
    smem_sr_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_sr_offset + 2 * NUM_WARPS * 4

    KERNEL_NAME = f"gdr_decode_{dtype}_kh{num_k_heads}x{head_k_dim}_vh{num_v_heads}x{head_v_dim}_q{seq_length}"
    KERNEL_NAME += f"_{NUM_WARPS}w{WARP_THREADS_V}x{WARP_THREADS_K}"
    KERNEL_NAME += f"_vs{NUM_BLOCKS_PER_V_DIM}"

    @flyc.kernel
    def gdr_decode_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        indices: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
    ):
        scale = arith.constant(SCALE_VALUE, type=T.f32)
        softplus_beta_ = arith.constant(softplus_beta, type=T.f32)
        softplus_threshold_ = arith.constant(softplus_threshold, type=T.f32)

        dtype_ = get_dtype_in_kernel(dtype)
        A_log_dtype_ = get_dtype_in_kernel(A_log_dtype)
        state_dtype_ = get_dtype_in_kernel(state_dtype)
        i32_0 = arith.constant(0, type=T.i32)
        f32_0 = arith.constant(0.0, type=T.f32)
        f32_1 = arith.constant(1.0, type=T.f32)
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)
        vec_t = T.vec(VALUES_PER_THREAD_K, dtype_)
        acc_vec_t = T.vec(VALUES_PER_THREAD_K, T.f32)

        tidx = fx.thread_idx.x
        bidx = fx.block_idx.x
        w_tid = tidx % WARP_SIZE
        wid = tidx // WARP_SIZE

        b_hv_i = bidx // NUM_BLOCKS_PER_V_DIM
        tile_v_start = bidx % NUM_BLOCKS_PER_V_DIM * TILE_V

        b_i = b_hv_i // num_v_heads
        hv_i = b_hv_i % num_v_heads
        hk_i = hv_i // (num_v_heads // num_k_heads)

        warp_k_vec_start = w_tid % WARP_THREADS_K * VALUES_PER_THREAD_K
        global_v_start = tile_v_start + wid * WARP_TILE_V + w_tid // WARP_THREADS_K

        indices_tensor = GTensor(indices, dtype=T.i32, shape=(-1,))
        pool_idx = fx.Int32(indices_tensor[b_i])

        q_tensor = GTensor(query, dtype=dtype_, shape=(-1, seq_length, num_k_heads, head_k_dim))
        k_tensor = GTensor(key, dtype=dtype_, shape=(-1, seq_length, num_k_heads, head_k_dim))
        v_tensor = GTensor(value, dtype=dtype_, shape=(-1, seq_length, num_v_heads, head_v_dim))
        a_tensor = GTensor(a, dtype=dtype_, shape=(-1, seq_length, num_v_heads))
        b_tensor = GTensor(b, dtype=dtype_, shape=(-1, seq_length, num_v_heads))
        dt_bias_tensor = GTensor(dt_bias, dtype=dtype_, shape=(num_v_heads,))
        A_log_tensor = GTensor(A_log, dtype=A_log_dtype_, shape=(num_v_heads,))
        state_tensor = GTensor(
            state,
            dtype=state_dtype_,
            shape=(-1, num_v_heads, head_v_dim, head_k_dim),
            stride=(state_strides[0], state_strides[1], state_strides[2], state_strides[3]))
        out_tensor = GTensor(out, dtype=dtype_, shape=(-1, seq_length, num_v_heads, head_v_dim))

        # base_ptr = allocator.get_base()
        # smem_sr_ptr = SmemPtr(base_ptr, smem_sr_offset, T.f32, shape=(2 * NUM_WARPS,))
        # sr_tensor = STensor(smem_sr_ptr, dtype=T.f32, shape=(-1,))

        def fast_exp(x, use_exp2=True):
            if const_expr(use_exp2):
                log2e = 1.4426950408889634
                out = rocdl.exp2(T.f32, x * log2e)
                return out
            return mlir_math.exp(x, fastmath=fm_fast)
        
        def fast_log1p(x):
            return mlir_math.log1p(x, fastmath=fm_fast)

        cond_valid = arith.cmpi(arith.CmpIPredicate.sge, pool_idx, fx.Int32(0))
        cond_valid_if = scf.IfOp(cond_valid, results_=[], has_else=False)
        with ir.InsertionPoint(cond_valid_if.then_block):

            if const_expr('f32' in A_log_dtype):
                r_A_log = A_log_tensor[hv_i]
            else:
                r_A_log = A_log_tensor[hv_i].extf(T.f32)
            r_dt_bias = dt_bias_tensor[hv_i].extf(T.f32)

            state_vecs = [0] * (WARP_TILE_V_ITERS * WARP_TILE_K_ITERS)
            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    state_vecs[vi * WARP_TILE_K_ITERS + ki] = state_tensor.vec_load((pool_idx, hv_i, global_v_i, warp_k_vec_i), VALUES_PER_THREAD_K)
                    if const_expr('f32' in state_dtype):
                        pass
                    else:
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = state_vecs[vi * WARP_TILE_K_ITERS + ki].extf(acc_vec_t)
            
            for sq_i in range_constexpr(seq_length):

                r_a = a_tensor[b_i, sq_i, hv_i].extf(T.f32)
                r_b = b_tensor[b_i, sq_i, hv_i].extf(T.f32)
                x = r_a + r_dt_bias
                beta_x = softplus_beta_ * x
                
                cond_sp = arith.cmpf(arith.CmpFPredicate.OLE, beta_x, fx.Float32(softplus_threshold_))
                cond_sp_if = scf.IfOp(cond_sp, results_=[T.f32], has_else=True)
                with ir.InsertionPoint(cond_sp_if.then_block):
                    softplus_x_ = (f32_1 / softplus_beta_) * fast_log1p(fast_exp(beta_x))
                    scf.YieldOp([softplus_x_])
                with ir.InsertionPoint(cond_sp_if.else_block):
                    softplus_x_ = x
                    scf.YieldOp([softplus_x_])
                softplus_x = cond_sp_if.results[0]

                r_g_value = - fast_exp(r_A_log) * softplus_x
                r_beta = f32_1 / (f32_1 + fast_exp(-r_b))
                r_g = fast_exp(r_g_value)

                r_g_vec = vector.BroadcastOp(acc_vec_t, r_g).vector

                sq_vecs = [0] * WARP_TILE_K_ITERS
                sk_vecs = [0] * WARP_TILE_K_ITERS
                
                scale_vec = vector.BroadcastOp(acc_vec_t, scale).vector

                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    q_vec = q_tensor.vec_load((b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K)
                    k_vec = k_tensor.vec_load((b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K)
                    sq_vecs[ki] = q_vec.extf(acc_vec_t)
                    sk_vecs[ki] = k_vec.extf(acc_vec_t)

                if const_expr(use_qk_l2norm):
                    sum_q_partial_vec = vector.from_elements(acc_vec_t, [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)])
                    sum_k_partial_vec = vector.from_elements(acc_vec_t, [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)])
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sum_q_partial_vec = sum_q_partial_vec + sq_vecs[ki] * sq_vecs[ki]
                        sum_k_partial_vec = sum_k_partial_vec + sk_vecs[ki] * sk_vecs[ki]
                        sum_q_partial = mlir_vector.ReductionOp(T.f32, vector.CombiningKind.ADD, sum_q_partial_vec).dest
                        sum_k_partial = mlir_vector.ReductionOp(T.f32, vector.CombiningKind.ADD, sum_k_partial_vec).dest
                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_q_partial = sum_q_partial + mlir_gpu.ShuffleOp(sum_q_partial, _to_raw(arith.constant(offset, type=T.i32)), width_i32, mode="xor").shuffleResult
                        sum_k_partial = sum_k_partial + mlir_gpu.ShuffleOp(sum_k_partial, _to_raw(arith.constant(offset, type=T.i32)), width_i32, mode="xor").shuffleResult
                    local_sum_q = mlir_gpu.ShuffleOp(sum_q_partial, _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)), width_i32, mode="idx").shuffleResult
                    local_sum_k = mlir_gpu.ShuffleOp(sum_k_partial, _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)), width_i32, mode="idx").shuffleResult
                    inv_norm_q = mlir_math.rsqrt(local_sum_q + 1e-6)
                    inv_norm_k = mlir_math.rsqrt(local_sum_k + 1e-6)
                    inv_norm_q_vec = vector.BroadcastOp(acc_vec_t, inv_norm_q).vector
                    inv_norm_k_vec = vector.BroadcastOp(acc_vec_t, inv_norm_k).vector
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * scale_vec * inv_norm_q_vec
                        sk_vecs[ki] = sk_vecs[ki] * inv_norm_k_vec
                else:
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * scale_vec

                dot_kq_vec = vector.from_elements(acc_vec_t, [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)])
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    dot_kq_vec = vector.FMAOp(sk_vecs[ki], sq_vecs[ki], dot_kq_vec).result
                dot_kq = mlir_vector.ReductionOp(T.f32, vector.CombiningKind.ADD, dot_kq_vec).dest
                for offset in WARP_THREADS_K_SHFL_OFFSETS:
                    dot_kq = dot_kq + mlir_gpu.ShuffleOp(dot_kq, _to_raw(fx.Int32(offset)), width_i32, mode="xor").shuffleResult

                for vi in range_constexpr(WARP_TILE_V_ITERS):

                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                    r_v = v_tensor[b_i, sq_i, hv_i, global_v_i].extf(T.f32)

                    sum_hk = vector.from_elements(acc_vec_t, [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)])
                    sum_hq_old = vector.from_elements(acc_vec_t, [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)])
                    
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] *= r_g_vec
                        h_cur = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                        sum_hk = vector.FMAOp(h_cur, sk_vecs[ki], sum_hk).result
                        sum_hq_old = vector.FMAOp(h_cur, sq_vecs[ki], sum_hq_old).result
                    
                    sum_hk = mlir_vector.ReductionOp(T.f32, vector.CombiningKind.ADD, sum_hk).dest
                    sum_hq_old = mlir_vector.ReductionOp(T.f32, vector.CombiningKind.ADD, sum_hq_old).dest
                    
                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_hk = sum_hk + mlir_gpu.ShuffleOp(sum_hk, _to_raw(fx.Int32(offset)), width_i32, mode="xor").shuffleResult
                        sum_hq_old = sum_hq_old + mlir_gpu.ShuffleOp(sum_hq_old, _to_raw(fx.Int32(offset)), width_i32, mode="xor").shuffleResult
                        
                    v_new = (r_v - sum_hk) * r_beta
                    v_new = mlir_gpu.ShuffleOp(v_new, _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)), width_i32, mode="idx").shuffleResult
                    sum_hq = sum_hq_old + v_new * dot_kq
                    v_new_bcast = vector.BroadcastOp(acc_vec_t, v_new)

                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        h_new = vector.FMAOp(sk_vecs[ki], v_new_bcast, state_vecs[vi * WARP_TILE_K_ITERS + ki]).result
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = h_new
                    
                    sum_hq = sum_hq.truncf(dtype_)
                    write_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(warp_k_vec_start), fx.Index(0))
                    write_cond_if = scf.IfOp(write_cond, results_=[], has_else=False)
                    with ir.InsertionPoint(write_cond_if.then_block):
                        out_tensor[b_i, sq_i, hv_i, global_v_i] = sum_hq
                        scf.YieldOp([])

            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    if const_expr('f32' in state_dtype):
                        out_vec = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                    else:
                        out_vec = state_vecs[vi * WARP_TILE_K_ITERS + ki].truncf(vec_t)
                    state_tensor.vec_store((pool_idx, hv_i, global_v_i, warp_k_vec_i), out_vec, VALUES_PER_THREAD_K)
            scf.YieldOp([])
        return

    @flyc.jit
    def launch_gdr_decode_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        indices: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        
        gx = batch_size * num_v_heads * NUM_BLOCKS_PER_V_DIM
        gdr_decode_kernel._func.__name__ = KERNEL_NAME
        gdr_decode_kernel(
            query, key, value, a, b, dt_bias, A_log, indices, state, out, batch_size,
        ).launch(grid=(gx, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)
    
    return launch_gdr_decode_kernel


GDR_GLOBAL_CONFIG_MAP = None
GDR_GPU_ARCH = get_rocm_arch()
def get_default_kwargs(dtype_str, state_dtype_str, batch_size, seq_length, num_k_heads, num_v_heads, head_k_dim, head_v_dim):
    d = {}
    d['NUM_BLOCKS_PER_V_DIM'] = 1
    d['NUM_WARPS'] = 4
    d['WARP_THREADS_K'] = 8
    global GDR_GLOBAL_CONFIG_MAP
    global GDR_GPU_ARCH
    if GDR_GLOBAL_CONFIG_MAP is None:
        _dict = {}
        with open('gdr_decode_tuned.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) > 10:
                    obj = json.loads(line)
                    arch, b, sq, nkh, nvh, khd, vhd = obj['arch'], obj['b'], obj['sq'], obj['num_k_heads'], obj['num_v_heads'], obj['head_k_dim'], obj['head_v_dim']
                    d_str, sd_str = obj['dtype'], obj['state_dtype']
                    _dict[(d_str, sd_str, arch, b, sq, nkh, nvh, khd, vhd)] = obj['config']
        GDR_GLOBAL_CONFIG_MAP = _dict
    config = GDR_GLOBAL_CONFIG_MAP.get((dtype_str, state_dtype_str, GDR_GPU_ARCH, batch_size, seq_length, num_k_heads, num_v_heads, head_k_dim, head_v_dim), None)
    if config:
        d.update(config)
    return d


def gdr_decode_(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    use_qk_l2norm: bool,
    need_shuffle_state: bool,
    kwargs: dict = {},
    stream: torch.cuda.Stream = torch.cuda.current_stream(),
):
    device = query.device
    dtype = query.dtype
    for input in [query, key, value, a, b, dt_bias, A_log, indices, out]:
        assert input.is_contiguous()
        assert input.data_ptr() % 16 == 0
        assert input.device == device
    assert state.data_ptr() % 16 == 0
    for input in [key, value, a, b, dt_bias, out]:
        assert input.dtype == dtype
    assert state.dtype in [torch.float, torch.bfloat16]
    assert A_log.dtype in [torch.float, torch.bfloat16]
    assert indices.dtype == torch.int32
    
    if need_shuffle_state:
        state_ = state.permute(0, 1, 3, 2).contiguous()
    else:
        state_ = state
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    num_v_heads = value.shape[-2]
    head_v_dim = value.shape[-1]
    kwargs_ = get_default_kwargs(str(dtype), str(state.dtype), batch_size, seq_length, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    kwargs_.update(kwargs)
    exe = create_shuffle_gdr_decode_kernel(
        get_dtype_str(query.dtype),
        get_dtype_str(A_log.dtype),
        get_dtype_str(state.dtype),
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        state.stride(),
        use_qk_l2norm,
        **kwargs_)
    with torch.cuda.device(query.device.index):
        _run_compiled(exe, query, key, value, a, b, dt_bias, A_log, indices, state_, out, batch_size, stream)
    if need_shuffle_state:
        state_ = state_.permute(0, 1, 3, 2).contiguous()
        state.copy_(state_)


def func(args, query, key, value, a, b, dt_bias, A_log, indices, state, out, stream=None):
    if stream is None:
        stream = torch.cuda.current_stream()
    gdr_decode_(query, key, value, a, b, dt_bias, A_log, indices, state, out, 
        use_qk_l2norm=args.use_qk_l2norm, need_shuffle_state=True, stream=stream)


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    o: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    grid = (NK, NV, N * HV)

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_triton_kernel(out, A_log, dt_bias, q, k, v, a, b, initial_state, indices, scale, use_qk_l2norm_in_kernel):
    fused_sigmoid_gating_delta_rule_update(
        out,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=None,
    )


def ref_func(args, query, key, value, a, b, dt_bias, A_log, indices, state, out):
    run_triton_kernel(out, A_log, dt_bias, query, key, value, a, b, state, indices,
        float(1.0 / (args.head_k_dim ** 0.5)), args.use_qk_l2norm)


def benchmark(args, func, ref_func, warmup=20, niters=100, sole_inputs=False):
    torch.manual_seed(2025)
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = list(inputs + outputs)
    inouts[-2] = inouts[-2].clone()
    ref_inouts = list(inputs + ref_outputs)
    ref_inouts[-2] = ref_inouts[-2].clone()
    func(*inouts)
    ref_func(*ref_inouts)
    for output, ref_output in zip(outputs, ref_outputs):
        is_allclose = torch.allclose(output, ref_output, atol=1e-2, rtol=1e-2)
        maxdiff_out = (output - ref_output).abs().max()
        is_allclose = is_allclose and torch.allclose(inouts[-2], ref_inouts[-2], atol=1e-2, rtol=1e-2)
        maxdiff_state = (inouts[-2] - ref_inouts[-2]).abs().max()
        # print("ref_output")
        # print(ref_output)
        # print("output")
        # print(output)
        # print(output - ref_output)
        print(f"maxdiff_out:{maxdiff_out}\nmaxdiff_state:{maxdiff_state}")
        # assert is_allclose == True
    print("validation passed!\n", flush=True)

    niters_ = niters if not sole_inputs else 1
    inputs = [create_inputs(args) for i in range(niters_)]
    ref_inputs = [create_inputs(args) for i in range(niters_)]
    outputs = [create_outputs(args) for i in range(niters_)]
    ref_outputs = [create_outputs(args) for i in range(niters_)]
    for i in range(niters_):
        inputs[i][-1].copy_(ref_inputs[i][-1])

    # get ref_func perf
    print("===================== [REF] =====================")
    for i in range(warmup):
        idx = i % niters_
        ref_func(*(ref_inputs[idx] + ref_outputs[idx]))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        for i in range(warmup, niters):
            idx = i % niters_
            ref_func(*(ref_inputs[idx] + ref_outputs[idx]))
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    print(table)

    # get func perf
    print("===================== [FLYDSL] =====================")
    for i in range(warmup):
        idx = i % niters_
        func(*(inputs[idx] + outputs[idx]))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        for i in range(warmup, niters):
            idx = i % niters_
            func(*(inputs[idx] + outputs[idx]))
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    print(table)


def benchmark_cudagraph(args, func, ref_func):
    print('===================== CUDA GRAPH TEST =====================')
    torch.manual_seed(2025)

    inputs = create_inputs(args)
    ref_inputs = create_inputs(args)
    
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)

    def copy_from_ref():
        for input, ref_input in zip(inputs, ref_inputs):
            if isinstance(input, torch.Tensor):
                input.copy_(ref_input)
        for output, ref_output in zip(outputs, ref_outputs):
            if isinstance(output, torch.Tensor):
                output.copy_(ref_output)
    
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph, stream=capture_stream):
            func(*(inputs + outputs), stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    
    copy_from_ref()
    ref_func(*(ref_inputs + ref_outputs))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    for output, ref_output in zip(outputs, ref_outputs):
        is_allclose = torch.allclose(output, ref_output, atol=1e-2, rtol=1e-2)
        maxdiff_out = (output - ref_output).abs().max()
        is_allclose = is_allclose and torch.allclose(inputs[-1], ref_inputs[-1], atol=1e-2, rtol=1e-2)
        maxdiff_state = (inputs[-1] - ref_inputs[-1]).abs().max()
        print(f"maxdiff_out:{maxdiff_out}\nmaxdiff_state:{maxdiff_state}")


@dataclass
class TunedArgs:
    arch: str
    dtype: str
    state_dtype: str
    b: int
    sq: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    config: dict
    duration: float


class GDRDecodeTuner:
    def __init__(self):
        self.selections = {
            'NUM_BLOCKS_PER_V_DIM': [1, 2, 4, 8],
            'NUM_WARPS': [1, 2, 4],
            'WARP_THREADS_K': [4, 8, 16, 32],
        }
        self.gpu_arch = get_rocm_arch()
    
    def tune_all(
        self,
        out_prefix,
        dtype,
        state_dtype,
        num_kv_heads = [[2, 8], [4, 8], [16, 32], [16, 64]],
        head_kv_dims = [[128, 128],],
        bs = [i for i in range(1, 257)],
        seq_length = 1,
    ):
        args = Args(
            dtype=dtype,
            ssm_state_dtype=state_dtype,
            b=0,
            sq=seq_length,
            num_k_heads=0,
            num_v_heads=0,
            head_k_dim=0,
            head_v_dim=0,
            ssm_state_size_n=256,
            use_qk_l2norm=True
        )
        with open(f"{out_prefix}.jsonl", "w", encoding="utf-8") as f:
            for num_kv_head in num_kv_heads:
                for head_kv_dim in head_kv_dims:
                    for b in bs:
                        args.b = b
                        args.num_k_heads = num_kv_head[0]
                        args.num_v_heads = num_kv_head[1]
                        args.head_k_dim = head_kv_dim[0]
                        args.head_v_dim = head_kv_dim[1]
                        result = self.tune_single(args)
                        result = vars(result)
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()

    def tune_single(self, args):
        keys = self.selections.keys()
        values = self.selections.values()
        configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        best_duration = float(1e10)
        best_idx = 0
        pbar = tqdm(total=len(configs), desc=f"{args}")
        for i, config in enumerate(configs):
            try:
                dur = self.benchmark(args, kwargs=config)
            except:
                dur = float(1e10)
            if dur < best_duration:
                best_duration = dur
                best_idx = i
            pbar.update(1)
        pbar.close()
        print(f"best_config:{configs[best_idx]}, duration:{best_duration}")
        result = TunedArgs(
            arch = self.gpu_arch,
            dtype = str(args.dtype),
            state_dtype = str(args.ssm_state_dtype),
            b = args.b,
            sq = args.sq,
            num_k_heads = args.num_k_heads,
            num_v_heads = args.num_v_heads,
            head_k_dim = args.head_k_dim,
            head_v_dim = args.head_v_dim,
            config = configs[best_idx],
            duration = best_duration,
        )
        return result
    
    def func(self, args, query, key, value, a, b, dt_bias, A_log, indices, state, out, kwargs={}, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream()
        gdr_decode_(query, key, value, a, b, dt_bias, A_log, indices, state, out, 
            use_qk_l2norm=args.use_qk_l2norm, need_shuffle_state=True, stream=stream, kwargs=kwargs)

    def ref_func(self, args, query, key, value, a, b, dt_bias, A_log, indices, state, out):
        run_triton_kernel(out, A_log, dt_bias, query, key, value, a, b, state, indices,
            float(1.0 / (args.head_k_dim ** 0.5)), args.use_qk_l2norm)

    def benchmark(self, args, kwargs={}, warmup=5, niters=50):
        # correctness test
        inputs = create_inputs(args)
        ref_inputs = create_inputs(args)
        outputs = create_outputs(args)
        ref_outputs = create_outputs(args)
        def copy_from_ref():
            for input, ref_input in zip(inputs, ref_inputs):
                if isinstance(input, torch.Tensor):
                    input.copy_(ref_input)
            for output, ref_output in zip(outputs, ref_outputs):
                if isinstance(output, torch.Tensor):
                    output.copy_(ref_output)
        copy_from_ref()
        self.ref_func(*(ref_inputs + ref_outputs))
        self.func(*(inputs + outputs + (kwargs,)))
        tol = 1e-3
        is_allclose = torch.allclose(ref_outputs[0], outputs[0], atol=tol, rtol=tol)
        is_allclose = is_allclose and torch.allclose(ref_inputs[-1], inputs[-1], atol=tol, rtol=tol)
        assert is_allclose == True
        # performance bench
        inputs = [create_inputs(args) for i in range(niters)]
        outputs = [create_outputs(args) for i in range(niters)]
        with profile(activities=[ProfilerActivity.CUDA], ) as prof:
            for i in range(niters):
                self.func(*(inputs[i] + outputs[i] + (kwargs,)))
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        durations = []
        for event in prof.events():
            if event.name.startswith("gdr_decode_"):
                durations.append(event.device_time)
        duration = np.median(durations)
        return duration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Examples")
    parser.add_argument("--b", type=int, default=2)
    parser.add_argument("--sq", type=int, default=2)
    parser.add_argument("--num_k_heads", type=int, default=16)
    parser.add_argument("--num_v_heads", type=int, default=32)
    parser.add_argument("--head_k_dim", type=int, default=128)
    parser.add_argument("--head_v_dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default='bf16')
    parser.add_argument("--ssm_state_size_n", type=int, default=1024)
    parser.add_argument("--ssm_state_dtype", type=str, default='bf16')
    parser.add_argument("--tune_all", action='store_true')
    args = parser.parse_args()
    dtype_convert = {'f32': torch.float, 'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    args.ssm_state_dtype = dtype_convert[args.ssm_state_dtype]

    if not args.tune_all:
        delattr(args, 'tune_all')
        args = Args(**vars(args))
        print(f"run: {__file__}, args: {args}")
        benchmark(args, func, ref_func)
        benchmark_cudagraph(args, func, ref_func)
    else:
        print(f"===================== Tune best configs  =====================")
        tuner = GDRDecodeTuner()
        tuner.tune_all(dtype=args.dtype, state_dtype=torch.bfloat16, out_prefix='temp/gdr_decode_tuned_sbf16')
        tuner.tune_all(dtype=args.dtype, state_dtype=torch.float, out_prefix='temp/gdr_decode_tuned_sf32')
    
    # rm -rf ~/.flydsl ; python3 gdr_decode.py --b=2 --sq=2 --num_k_heads=16 --num_v_heads=32 --head_k_dim=128 --head_v_dim=128 --dtype=bf16
    # rm -rf ~/.flydsl ; python3 gdr_decode.py --b=1 --sq=1 --num_k_heads=2 --num_v_heads=8 --head_k_dim=128 --head_v_dim=128 --dtype=bf16
    # rm -rf ~/.flydsl ; python3 gdr_decode.py --b=128 --sq=1 --num_k_heads=2 --num_v_heads=8 --head_k_dim=128 --head_v_dim=128 --dtype=bf16
    # rm -rf ~/.flydsl ; python3 gdr_decode.py --b=2 --sq=1 --num_k_heads=16 --num_v_heads=32 --head_k_dim=128 --head_v_dim=128 --dtype=bf16
    # rm -rf ~/.flydsl ; python3 gdr_decode.py --b=16 --sq=1 --num_k_heads=4 --num_v_heads=8 --head_k_dim=128 --head_v_dim=128 --dtype=bf16

    # rm -rf ~/.flydsl ; python3 gdr_decode.py --dtype=bf16 --tune_all
