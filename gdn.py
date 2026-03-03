import time
import torch
import torch.nn.functional as F
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
    A_log = torch.randn((args.num_v_heads), dtype=torch.float32, device="cuda")
    indices = torch.arange((args.b), dtype=torch.int32, device="cuda")
    state = torch.randn((args.b, args.num_v_heads, args.head_k_dim, args.head_v_dim), dtype=torch.float32, device="cuda")
    return (args, query, key, value, a, b, dt_bias, A_log, indices, state)


def create_outputs(args):
    out = torch.randn((args.b, args.sq, args.num_v_heads, args.head_v_dim), dtype=args.dtype, device='cuda')
    return (out,)


def ref_func(args, query, key, value, a, b, dt_bias, A_log, indices, state, out):
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
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
    last_recurrent_state = state.clone()
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
    out = out[indices]
    state.copy_(last_recurrent_state)


def create_fused_gdn_kernel(dtype, VEC_SIZE: int, use_qk_l2norm: bool, NUM_BLOCKS_PER_STATE: int = 2, TILE_V: int = 32):
    _asv = arith.as_value
    _asid = flir.const_index
    _extf = flir.arith.extf
    def _extf32(value):
        return _extf(T.f32(), value)
    def _create_f32_zero():
        return _extf32(_asv(0.0))

    DYN = ir.ShapedType.get_dynamic_size()
    ARCH = get_rocm_arch()
    allocator = SmemAllocator(None, arch=ARCH)
    BLOCK_THREADS = 128
    WARP_SIZE = 32
    NUM_WARPS = BLOCK_THREADS // WARP_SIZE
    V_PER_WARP = TILE_V // NUM_WARPS
    TILE_K = 128 # fixed
    
    class FGDN(flir.MlirModule):
        GPU_MODULE_NAME = "linear_attention_kernels"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{ARCH}">']

        def init_gpu_module(self):
            self.dtype = dtype.get()
            self.acc_type = T.f32()
            self.sq = allocator.allocate_array(T.f32(), TILE_K)
            self.sk = allocator.allocate_array(T.f32(), TILE_K)
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
            seq_length: lambda: T.index(),
            num_k_heads: lambda: T.index(),
            num_v_heads: lambda: T.index(),
            head_k_dim: lambda: T.index(),
            head_v_dim: lambda: T.index(),
        ):
            # state: # (b, num_v_heads, head_k_dim, NUM_BLOCKS_PER_STATE * num_v_tiles_per_block * TILE_V)
            # block_size = b * num_v_heads * NUM_BLOCKS_PER_STATE

            tidx = flir.thread_idx("x")
            bidx = flir.block_idx("x")
            base_ptr = allocator.get_base()
            w_tid = tidx % WARP_SIZE
            wid = tidx // WARP_SIZE

            b_hv_i = bidx // NUM_BLOCKS_PER_STATE
            b_i = b_hv_i // num_v_heads
            hv_i = b_hv_i % num_v_heads

            num_v_tiles = head_v_dim // TILE_V
            num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE
            tile_v_i = (bidx % NUM_BLOCKS_PER_STATE) * num_v_tiles_per_block * TILE_V
            
            hk_i = hv_i // (num_v_heads // num_k_heads)

            indices_tensor = GTensor(indices, T.i32(), (batch_size,))
            pool_idx = arith.index_cast(T.index(), indices_tensor[b_i])

            sq_tensor = STensor(self.sq(base_ptr), T.f32(), shape=(TILE_K,))
            sk_tensor = STensor(self.sk(base_ptr), T.f32(), shape=(TILE_K,))
            state_tensor = GTensor(state, F32Type.get(), shape=(batch_size, num_v_heads, head_k_dim, head_v_dim))

            q_tensor = GTensor(query, BF16Type.get(), (batch_size, seq_length, num_k_heads, head_k_dim))
            k_tensor = GTensor(key, BF16Type.get(), (batch_size, seq_length, num_k_heads, head_k_dim))

            A_log_tensor = GTensor(A_log, F32Type.get(), (num_v_heads,))
            dt_bias_tensor = GTensor(dt_bias, BF16Type.get(), (num_v_heads,))
            a_tensor = GTensor(a, BF16Type.get(), (batch_size, seq_length, num_v_heads))
            b_tensor = GTensor(b, BF16Type.get(), (batch_size, seq_length, num_v_heads))

            sq_i = 0
            
            if pool_idx >= 0:
                k_local = w_tid // V_PER_WARP
                v_local = w_tid % V_PER_WARP
                v_base = wid * V_PER_WARP
                v_idx = v_base + v_local
                
                if tidx < TILE_K:
                    sq_tensor[tidx] = _extf32(q_tensor[b_i, sq_i, hk_i, tidx])
                    sk_tensor[tidx] = _extf32(k_tensor[b_i, sq_i, hk_i, tidx])
                
                state_batch = state_tensor[pool_idx, hv_i, None, None]

                r_A_log = A_log_tensor[hv_i]
                r_dt_bias = _extf32(dt_bias_tensor[hv_i])
                r_a = _extf32(a_tensor[b_i, sq_i, hv_i])
                r_b = _extf32(b_tensor[b_i, sq_i, hv_i])

                r_g = _create_f32_zero()
                r_beta = _create_f32_zero()

                

            pass

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
        ):
            c1 = arith.index(1)
            bx = arith.index(BLOCK_THREADS)
            gx = batch_size * num_v_heads * arith.index(NUM_BLOCKS_PER_STATE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "fused_gdn_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[query, key, value, a, b, dt_bias, A_log, indices, state, out,
                    batch_size, seq_length, num_k_heads, num_v_heads, head_k_dim, head_v_dim],
            )

    return FGDN().module


EXE = None
def func(args, query, key, value, a, b, dt_bias, A_log, indices, state, out):
    global EXE
    if not EXE:
        if args.dtype == torch.float:
            module = create_fused_gdn_kernel(F32Type, 4, True)
        elif args.dtype == torch.half:
            module = create_fused_gdn_kernel(F16Type, 4, True)
        elif args.dtype == torch.bfloat16:
            module = create_fused_gdn_kernel(BF16Type, 4, True)
        optimized = run_pipeline(module, Pipeline().canonicalize().cse())
        EXE = flydsl.compile(optimized)
    EXE(query, key, value, a, b, dt_bias, A_log, indices, state, out, 
        args.b, args.sq, args.num_k_heads, args.num_v_heads, 
        args.head_k_dim, args.head_v_dim)
    torch.cuda.synchronize()


def benchmark(args, func, ref_func, warmup=20, niters=100):
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
        is_allclose = torch.allclose(output, ref_output, atol=1e-3, rtol=1e-3)
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
    # python3 gdn.py --b=2 --sq=4 --num_k_heads=16 --num_v_heads=32 --head_k_dim=128 --head_v_dim=128 --dtype=bf16
