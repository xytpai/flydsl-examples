import time
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from dataclasses import dataclass
import functools
from torch.profiler import profile, ProfilerActivity
from typing import Optional, Any, Callable, Dict, Literal, Optional, Tuple
import triton
import triton.language as tl

import flydsl
from flydsl.dialects.ext import flir, gpu, arith, rocdl, vector, buffer_ops, math
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.pipeline import Pipeline, run_pipeline
from flydsl.dialects.ext.python_control_flow import range_constexpr, lower_range_for_loops
from flydsl.utils import SmemAllocator
fm_fast = flir.arith.FastMathFlags.fast

from _mlir import ir
from _mlir.ir import F16Type, BF16Type, F32Type, IntegerType, VectorType
import _mlir.extras.types as T

from utils.ftensor import GTensor, STensor


@dataclass
class Args:
    dtype: torch.dtype
    b: int
    sq: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
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
    state = torch.randn((args.b, args.num_v_heads, args.head_k_dim, args.head_v_dim), dtype=torch.float32, device="cuda")
    return (args, query, key, value, a, b, dt_bias, A_log, indices, state)


def create_outputs(args):
    out = torch.zeros((args.b, args.sq, args.num_v_heads, args.head_v_dim), dtype=args.dtype, device='cuda')
    return (out,)


@torch.compile
def ref_func(args, query, key, value, a, b, dt_bias, A_log, indices, state, out):
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


def create_fused_gdn_kernel(
    dtype,
    VEC_SIZE: int,
    seq_length: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    use_qk_l2norm: bool,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    NUM_BLOCKS_PER_V_DIM = 2,
):
    _asv = arith.as_value
    _asid = flir.const_index
    _extf = flir.arith.extf
    fm_fast = flir.arith.FastMathFlags.fast
    def _extf32(value):
        return _extf(T.f32(), value)
    def _create_f32(value):
        return _extf32(_asv(float(value)))

    DYN = ir.ShapedType.get_dynamic_size()
    ARCH = get_rocm_arch()
    allocator = SmemAllocator(None, arch=ARCH)
    
    # NUM_WARPS = 4
    WARP_SIZE = 64
    WARP_TILE_V = 32

    TILE_V = head_v_dim // NUM_BLOCKS_PER_V_DIM
    NUM_WARPS = TILE_V // WARP_TILE_V

    BLOCK_THREADS = NUM_WARPS * WARP_SIZE
    VALUES_PER_THREAD_V = VEC_SIZE // 2
    WARP_TILE_V_THREADS = WARP_TILE_V // VALUES_PER_THREAD_V
    WARP_TILE_K_THREADS = WARP_SIZE // WARP_TILE_V_THREADS
    # TILE_V = NUM_WARPS * WARP_TILE_V
    TILE_K = head_k_dim
    VALUES_PER_THREAD_K = TILE_K // WARP_TILE_K_THREADS

    assert VALUES_PER_THREAD_K >= 1
    assert TILE_K % WARP_TILE_K_THREADS == 0
    assert TILE_K <= BLOCK_THREADS
    assert NUM_WARPS >= 1

    K_THREAD_SHFL_OFFSETS = []
    offsets_ = WARP_TILE_K_THREADS // 2
    while offsets_ >= 1:
        K_THREAD_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2
    
    WARP_SIZE_SHFL_OFFSETS = []
    offsets_ = WARP_SIZE // 2
    while offsets_ >= 1:
        WARP_SIZE_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2
    
    class FGDN(flir.MlirModule):
        GPU_MODULE_NAME = "linear_attention_kernels"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{ARCH}">']

        def init_gpu_module(self):
            self.dtype = dtype.get()
            self.acc_type = T.f32()
            self.sq = allocator.allocate_array(T.f32(), seq_length * TILE_K)
            self.sk = allocator.allocate_array(T.f32(), seq_length * TILE_K)
            self.sr = allocator.allocate_array(T.f32(), 2 * NUM_WARPS)
            allocator.finalize()

        @flir.kernel
        def fused_gdn_kernel(
            self: flir.T.i64,
            query: lambda: T.memref(DYN, dtype.get()),
            key: lambda: T.memref(DYN, dtype.get()),
            value: lambda: T.memref(DYN, dtype.get()),
            a: lambda: T.memref(DYN, dtype.get()),
            b: lambda: T.memref(DYN, dtype.get()),
            dt_bias: lambda: T.memref(DYN, dtype.get()),
            A_log: lambda: T.memref(DYN, F32Type.get()),
            indices: lambda: T.memref(DYN, T.i32()),
            state: lambda: T.memref(DYN, F32Type.get()),
            out: lambda: T.memref(DYN, dtype.get()),
            batch_size: lambda: T.index(),
            seq_length_: lambda: T.index(),
            num_k_heads_: lambda: T.index(),
            num_v_heads_: lambda: T.index(),
            head_k_dim_: lambda: T.index(),
            head_v_dim_: lambda: T.index(),
            scale: lambda: T.f32(),
        ):
            i32_0 = arith.constant(0, type=T.i32())
            width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
            acc_vec_t = VectorType.get([VALUES_PER_THREAD_V], self.acc_type)
            vec_t = VectorType.get([VALUES_PER_THREAD_V], self.dtype)

            tidx = flir.thread_idx("x")
            bidx = flir.block_idx("x")
            w_tid = tidx % WARP_SIZE
            wid = tidx // WARP_SIZE

            b_hv_i = bidx // NUM_BLOCKS_PER_V_DIM
            tile_v_start = bidx % NUM_BLOCKS_PER_V_DIM * TILE_V

            b_i = b_hv_i // num_v_heads
            hv_i = b_hv_i % num_v_heads
            hk_i = hv_i // (num_v_heads // num_k_heads)

            warp_k_begin_i = w_tid // WARP_TILE_V_THREADS
            warp_v_vec_i = w_tid % WARP_TILE_V_THREADS * VALUES_PER_THREAD_V

            indices_tensor = GTensor(indices, T.i32(), (-1,))
            pool_idx = arith.index_cast(T.index(), indices_tensor[b_i])

            q_tensor = GTensor(query, dtype.get(), shape=(-1, seq_length, num_k_heads, head_k_dim))
            k_tensor = GTensor(key, dtype.get(), shape=(-1, seq_length, num_k_heads, head_k_dim))
            v_tensor = GTensor(value, dtype.get(), shape=(-1, seq_length, num_v_heads, head_v_dim))
            a_tensor = GTensor(a, dtype.get(), shape=(-1, seq_length, num_v_heads))
            b_tensor = GTensor(b, dtype.get(), shape=(-1, seq_length, num_v_heads))
            dt_bias_tensor = GTensor(dt_bias, dtype.get(), shape=(num_v_heads,))
            A_log_tensor = GTensor(A_log, F32Type.get(), shape=(num_v_heads,))
            state_tensor = GTensor(state, F32Type.get(), shape=(-1, num_v_heads, head_k_dim, head_v_dim))
            out_tensor = GTensor(out, dtype.get(), shape=(-1, seq_length, num_v_heads, head_v_dim))

            sbase = allocator.get_base()
            sq_tensor = STensor(self.sq(sbase), T.f32(), shape=(seq_length, TILE_K,))
            sk_tensor = STensor(self.sk(sbase), T.f32(), shape=(seq_length, TILE_K,))
            sr_tensor = STensor(self.sr(sbase), T.f32(), shape=(-1,))

            if pool_idx >= 0:

                r_A_log = A_log_tensor[hv_i]
                r_dt_bias = _extf32(dt_bias_tensor[hv_i])

                for sq_i in range_constexpr(seq_length):
                    if use_qk_l2norm:
                        q_val = _create_f32(0)
                        k_val = _create_f32(0)
                        sum_q_partial = _create_f32(0)
                        sum_k_partial = _create_f32(0)
                        if tidx < TILE_K:
                            q_val = _extf32(q_tensor[b_i, sq_i, hk_i, tidx])
                            k_val = _extf32(k_tensor[b_i, sq_i, hk_i, tidx])
                            sum_q_partial = q_val * q_val
                            sum_k_partial = k_val * k_val
                        for offset in WARP_SIZE_SHFL_OFFSETS:
                            sum_q_partial = sum_q_partial + gpu.ShuffleOp(_asv(sum_q_partial), _asv(arith.constant(offset, type=T.i32())), width_i32, mode="xor").shuffleResult
                            sum_k_partial = sum_k_partial + gpu.ShuffleOp(_asv(sum_k_partial), _asv(arith.constant(offset, type=T.i32())), width_i32, mode="xor").shuffleResult
                        if w_tid == 0:
                            sr_tensor[wid] = sum_q_partial
                            sr_tensor[NUM_WARPS + wid] = sum_k_partial
                        gpu.barrier()
                        inv_norm_q = _create_f32(0)
                        inv_norm_k = _create_f32(0)
                        if wid == 0:
                            local_sum_q = _create_f32(0)
                            local_sum_k = _create_f32(0)
                            if w_tid < NUM_WARPS:
                                local_sum_q = sr_tensor[w_tid]
                                local_sum_k = sr_tensor[NUM_WARPS + w_tid]
                            for offset in WARP_SIZE_SHFL_OFFSETS:
                                local_sum_q = local_sum_q + gpu.ShuffleOp(_asv(local_sum_q), _asv(arith.constant(offset, type=T.i32())), width_i32, mode="xor").shuffleResult
                                local_sum_k = local_sum_k + gpu.ShuffleOp(_asv(local_sum_k), _asv(arith.constant(offset, type=T.i32())), width_i32, mode="xor").shuffleResult
                            if w_tid == 0:
                                sr_tensor[0] = _extf32(_asv(flir.math.rsqrt(_extf32(_asv(local_sum_q + 1e-6)).value)))
                                sr_tensor[1] = _extf32(_asv(flir.math.rsqrt(_extf32(_asv(local_sum_k + 1e-6)).value)))
                        gpu.barrier()
                        inv_norm_q = sr_tensor[0]
                        inv_norm_k = sr_tensor[1]
                        if tidx < TILE_K:
                            sq_tensor[sq_i, tidx] = q_val * scale * inv_norm_q
                            sk_tensor[sq_i, tidx] = k_val * inv_norm_k
                    else:
                        if tidx < TILE_K:
                            sq_tensor[sq_i, tidx] = _extf32(q_tensor[b_i, sq_i, hk_i, tidx]) * scale
                            sk_tensor[sq_i, tidx] = _extf32(k_tensor[b_i, sq_i, hk_i, tidx])
                    gpu.barrier()

                global_v_vec_i = tile_v_start + wid * WARP_TILE_V + warp_v_vec_i
                state_vecs = []
                for i in range_constexpr(VALUES_PER_THREAD_K):
                    state_ = state_tensor.vec_load((pool_idx, hv_i, warp_k_begin_i + i * WARP_TILE_K_THREADS, global_v_vec_i), VALUES_PER_THREAD_V)
                    state_vecs.append(state_)

                for sq_i in range_constexpr(seq_length):
                    if True:
                        r_g = _create_f32(0)
                        r_beta = _create_f32(0)
                        r_a = _extf32(a_tensor[b_i, sq_i, hv_i])
                        r_b = _extf32(b_tensor[b_i, sq_i, hv_i])
                        x = r_a + r_dt_bias
                        beta_x = _create_f32(softplus_beta) * x
                        softplus_x = _create_f32(0)
                        if beta_x <= softplus_threshold:
                            softplus_x = _create_f32(1.0 / softplus_beta) * flir.math.log1p(_asv(flir.math.exp(_asv(beta_x), fastmath=fm_fast)), fastmath=fm_fast)
                        else:
                            softplus_x = x
                        r_g_value = _create_f32(0) - flir.math.exp(_asv(r_A_log), fastmath=fm_fast) * softplus_x
                        r_beta = _create_f32(1) / (_create_f32(1) + flir.math.exp(_asv(_create_f32(0) - r_b), fastmath=fm_fast))
                        r_g = flir.math.exp(_asv(r_g_value), fastmath=fm_fast)

                    r_g = vector.BroadcastOp(acc_vec_t, _asv(r_g))
                    r_beta = vector.BroadcastOp(acc_vec_t, _asv(r_beta))

                    r_v = v_tensor.vec_load((b_i, sq_i, hv_i, global_v_vec_i), VALUES_PER_THREAD_V)
                    r_v = flir.arith.extf(acc_vec_t, _asv(r_v))

                    sum_hk = vector.from_elements(acc_vec_t, [_create_f32(0) for i in range_constexpr(VALUES_PER_THREAD_V)])
                    sum_hq = vector.from_elements(acc_vec_t, [_create_f32(0) for i in range_constexpr(VALUES_PER_THREAD_V)])

                    for i in range_constexpr(VALUES_PER_THREAD_K):
                        h_val = state_vecs[i] * r_g
                        r_k_val = vector.BroadcastOp(acc_vec_t, _asv(sk_tensor[sq_i, warp_k_begin_i + i * WARP_TILE_K_THREADS]))
                        sum_hk = vector.FMAOp(_asv(h_val), _asv(r_k_val), _asv(sum_hk)).result
                    
                    for offset in K_THREAD_SHFL_OFFSETS:
                        sum_hk = sum_hk + gpu.ShuffleOp(_asv(sum_hk), _asv(arith.constant(offset * WARP_TILE_V_THREADS, type=T.i32())), width_i32, mode="xor").shuffleResult
                    
                    v_new = (r_v - sum_hk) * r_beta
                    v_new = gpu.ShuffleOp(_asv(v_new), _asv(arith.index_cast(T.i32(), w_tid % WARP_TILE_V_THREADS)), width_i32, mode="idx").shuffleResult

                    for i in range_constexpr(VALUES_PER_THREAD_K):
                        h_old = state_vecs[i] * r_g
                        r_k_val = vector.BroadcastOp(acc_vec_t, _asv(sk_tensor[sq_i, warp_k_begin_i + i * WARP_TILE_K_THREADS]))
                        r_q_val = vector.BroadcastOp(acc_vec_t, _asv(sq_tensor[sq_i, warp_k_begin_i + i * WARP_TILE_K_THREADS]))
                        h_new = vector.FMAOp(_asv(r_k_val), _asv(v_new), _asv(h_old)).result
                        state_vecs[i] = h_new
                        sum_hq = vector.FMAOp(_asv(h_new), _asv(r_q_val), _asv(sum_hq)).result
                    
                    for offset in K_THREAD_SHFL_OFFSETS:
                        sum_hq = sum_hq + gpu.ShuffleOp(_asv(sum_hq), _asv(arith.constant(offset * WARP_TILE_V_THREADS, type=T.i32())), width_i32, mode="xor").shuffleResult

                    if warp_k_begin_i == 0:
                        sum_hq = flir.arith.truncf(vec_t, _asv(sum_hq))
                        out_tensor.vec_store((b_i, sq_i, hv_i, global_v_vec_i), sum_hq, VALUES_PER_THREAD_V)
                    pass
                
                for i in range_constexpr(VALUES_PER_THREAD_K):
                    state_tensor.vec_store((pool_idx, hv_i, warp_k_begin_i + i * WARP_TILE_K_THREADS, global_v_vec_i), state_vecs[i], VALUES_PER_THREAD_V)
            return

        @flir.jit
        def __call__(
            self: flir.T.i64,
            query: lambda: T.memref(DYN, dtype.get()),
            key: lambda: T.memref(DYN, dtype.get()),
            value: lambda: T.memref(DYN, dtype.get()),
            a: lambda: T.memref(DYN, dtype.get()),
            b: lambda: T.memref(DYN, dtype.get()),
            dt_bias: lambda: T.memref(DYN, dtype.get()),
            A_log: lambda: T.memref(DYN, F32Type.get()),
            indices: lambda: T.memref(DYN, T.i32()),
            state: lambda: T.memref(DYN, F32Type.get()),
            out: lambda: T.memref(DYN, dtype.get()),
            batch_size: lambda: T.index(),
            seq_length: lambda: T.index(),
            num_k_heads: lambda: T.index(),
            num_v_heads: lambda: T.index(),
            head_k_dim: lambda: T.index(),
            head_v_dim: lambda: T.index(),
            scale: lambda: T.f32(),
        ):
            c1 = arith.index(1)
            bx = arith.index(BLOCK_THREADS)
            gx = batch_size * num_v_heads * arith.index(NUM_BLOCKS_PER_V_DIM)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "fused_gdn_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[query, key, value, a, b, dt_bias, A_log, indices, state, out,
                    batch_size, seq_length, num_k_heads, num_v_heads, head_k_dim, head_v_dim, scale],
            )

    return FGDN().module


def choose_kwargs(args):
    d = {}
    if args.b == 1 or (args.b >= 16 and args.b <= 64):
        d['NUM_BLOCKS_PER_V_DIM'] = 2
    else:
        d['NUM_BLOCKS_PER_V_DIM'] = 1
    return d


EXE = None
def func(args, query, key, value, a, b, dt_bias, A_log, indices, state, out):
    global EXE
    if not EXE:
        kwargs = choose_kwargs(args)
        print(kwargs)
        if args.dtype == torch.float:
            module = create_fused_gdn_kernel(
                F32Type,
                4,
                args.sq,
                args.num_k_heads,
                args.num_v_heads,
                args.head_k_dim,
                args.head_v_dim,
                args.use_qk_l2norm,
                **kwargs)
        elif args.dtype == torch.half:
            module = create_fused_gdn_kernel(
                F16Type,
                8,
                args.sq,
                args.num_k_heads,
                args.num_v_heads,
                args.head_k_dim,
                args.head_v_dim,
                args.use_qk_l2norm,
                **kwargs)
        elif args.dtype == torch.bfloat16:
            module = create_fused_gdn_kernel(
                BF16Type,
                8,
                args.sq,
                args.num_k_heads,
                args.num_v_heads,
                args.head_k_dim,
                args.head_v_dim,
                args.use_qk_l2norm,
                **kwargs)
        optimized = run_pipeline(module, Pipeline().canonicalize().cse())
        EXE = flydsl.compile(optimized)
    EXE(query, key, value, a, b, dt_bias, A_log, indices, state, out, 
        args.b, args.sq, args.num_k_heads, args.num_v_heads, 
        args.head_k_dim, args.head_v_dim, float(1.0 / (args.head_k_dim ** 0.5)))
    torch.cuda.synchronize()


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


def benchmark(args, func, ref_func, warmup=20, niters=100):
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
        print(f"maxdiff_out:{maxdiff_out}\nmaxdiff_state:{maxdiff_state}")
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
    parser.add_argument("--b", type=int, required=True)
    parser.add_argument("--sq", type=int, required=True)
    parser.add_argument("--num_k_heads", type=int, required=True)
    parser.add_argument("--num_v_heads", type=int, required=True)
    parser.add_argument("--head_k_dim", type=int, required=True)
    parser.add_argument("--head_v_dim", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {'f32': torch.float, 'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    args = Args(**vars(args))
    benchmark(args, func, ref_func)
    # python3 gdn.py --b=2 --sq=2 --num_k_heads=16 --num_v_heads=32 --head_k_dim=128 --head_v_dim=128 --dtype=bf16
