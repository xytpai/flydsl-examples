from _mlir.dialects import builtin, gpu as _gpu
from flydsl.dialects.ext import buffer_ops
def stream_ptr_to_async_token(stream_ptr_value, loc=None, ip=None):
    stream_llvm_ptr = buffer_ops.create_llvm_ptr(stream_ptr_value)
    
    async_token_type = _gpu.AsyncTokenType.get()
    cast_op = builtin.UnrealizedConversionCastOp(
        [async_token_type], [stream_llvm_ptr], loc=loc, ip=ip
    )
    return cast_op.results[0]

import functools
import pytest
from typing import Optional, Any, Callable, Dict, Literal, Optional, Tuple
import logging
logger = logging.getLogger(__name__)
import torch
import triton
import triton.language as tl

import flydsl
from flydsl.dialects.ext import flir, gpu, arith
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.pipeline import Pipeline, run_pipeline
from flydsl.dialects.ext.python_control_flow import range_constexpr, lower_range_for_loops
from flydsl.utils import SmemAllocator
fm_fast = flir.arith.FastMathFlags.fast

from _mlir import ir
from _mlir.ir import F16Type, BF16Type, F32Type, IntegerType, VectorType, IndexType
import _mlir.extras.types as T


_compiled_kernels: Dict[Tuple, object] = {}
_cu_seqlens_cache: Dict[Tuple, torch.Tensor] = {}
TILE_K = 128
TILE_V = 32
TILE_V_PADDED = 36
TILE_V_SMALL = 16
TILE_V_SMALL_PADDED = 20
NUM_STAGES = 2
NUM_THREADS = 128
NUM_BLOCKS_PER_STATE_SMALL = 8
NUM_THREADS_LARGE = 256
NUM_WARPS_LARGE = 8
V_PER_WARP = 4
ROWS_PER_ITER = 8
NUM_K_ITERS = TILE_K // ROWS_PER_ITER
SMALL_BATCH_THRESHOLD = 32


def _create_jit_functions(
    softplus_beta: float,
    softplus_threshold: float,
    scale: float,
    B_compile: int,
    T_compile: int,
    H: int,
    K: int,
    V: int,
    HV: int,
    use_initial_state: bool,
    use_qk_l2norm: bool,
    N: int
):
    DYN = ir.ShapedType.get_dynamic_size()
    ARCH = get_rocm_arch()
    allocator = SmemAllocator(None, arch=ARCH)
    NUM_WARPS_SMALL = 4
    V_PER_WARP_SMALL = TILE_V_SMALL // NUM_WARPS_SMALL
    ROWS_PER_ITER_SMALL = 32 // V_PER_WARP_SMALL
    NUM_K_ITERS_SMALL = TILE_K // ROWS_PER_ITER_SMALL

    # gdn_small, gdn_small_varlen, gdn_large, gdn_large_varlen = _define_kernels()

    class RunSmallBatch(flir.MlirModule):
        GPU_MODULE_NAME = "gdn_kernels"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{ARCH}">']
        
        @flir.kernel
        def gdn_kernel_small_batch(
            self: flir.T.i64,
            h0_source: lambda: T.memref(DYN, F32Type.get()),
            num_v_tiles: lambda: T.index(),
            q: lambda: T.memref(DYN, BF16Type.get()),
            k: lambda: T.memref(DYN, BF16Type.get()),
            v: lambda: T.memref(DYN, BF16Type.get()),
            a: lambda: T.memref(DYN, BF16Type.get()),
            b: lambda: T.memref(DYN, BF16Type.get()),
            A_log: lambda: T.memref(DYN, F32Type.get()),
            dt_bias: lambda: T.memref(DYN, BF16Type.get()),
            o: lambda: T.memref(DYN, BF16Type.get()),
            h0_indices: lambda: T.memref(DYN, IntegerType.get_signless(32)),
        ):
            pass
        
        @flir.jit
        def __call__(
            self: flir.T.i64,
            cu_seqlens: lambda: T.memref(DYN, IntegerType.get_signless(32)),
            q: lambda: T.memref(DYN, BF16Type.get()),
            k: lambda: T.memref(DYN, BF16Type.get()),
            v: lambda: T.memref(DYN, BF16Type.get()),
            a: lambda: T.memref(DYN, BF16Type.get()),
            b: lambda: T.memref(DYN, BF16Type.get()),
            A_log: lambda: T.memref(DYN, F32Type.get()),
            dt_bias: lambda: T.memref(DYN, BF16Type.get()),
            h0_source: lambda: T.memref(DYN, F32Type.get()),
            h0_indices: lambda: T.memref(DYN, IntegerType.get_signless(32)),
            o: lambda: T.memref(DYN, BF16Type.get()),
            stream: lambda: T.i64(),
        ):
            batch_size = N * HV
            gx = arith.index(batch_size * NUM_BLOCKS_PER_STATE_SMALL)
            c1 = arith.index(1)
            bx = arith.index(NUM_THREADS)
            num_v_tiles_small = arith.index((V + TILE_V_SMALL - 1) // TILE_V_SMALL)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "gdn_kernel_small_batch"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[h0_source, num_v_tiles_small, q, k, v, a, b, A_log, dt_bias, o, h0_indices],
                async_dependencies=[stream_ptr_to_async_token(stream)],
            )
            
            # pool_size, hv_dim, k_dim, v_dim = h0_source.layout.shape
            # n_indices = h0_indices.layout.shape[0]
            # batch_size = N * HV

            # copy_atom = cute.make_copy_atom(
            #     cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            #     cutlass.Float32,
            #     num_bits_per_copy=128,
            # )
            # num_v_tiles_small = cute.ceil_div(v_dim, TILE_V_SMALL)
            # smem_layout_small = cute.make_layout(
            #     (TILE_K, TILE_V_SMALL, NUM_STAGES),
            #     stride=(TILE_V_SMALL_PADDED, 1, TILE_K * TILE_V_SMALL_PADDED),
            # )
            # thread_layout_small = cute.make_layout((32, 4), stride=(4, 1))
            # val_layout_small = cute.make_layout((1, 4))
            # tiled_copy_load_small = cute.make_tiled_copy_tv(
            #     copy_atom, thread_layout_small, val_layout_small
            # )
            # smem_bytes_small = (
            #     4 * TILE_K * TILE_V_SMALL_PADDED * NUM_STAGES
            #     + 4 * TILE_V_SMALL
            #     + 4 * TILE_K * 2
            #     + 64
            # )

            # gdn_small(
            #     tiled_copy_load_small,
            #     h0_source,
            #     smem_layout_small,
            #     num_v_tiles_small,
            #     q,
            #     k,
            #     v,
            #     a,
            #     b,
            #     A_log,
            #     dt_bias,
            #     o,
            #     h0_indices,
            #     softplus_beta,
            #     softplus_threshold,
            #     scale,
            #     H,
            #     HV,
            #     use_qk_l2norm,
            # ).launch(
            #     grid=(batch_size * NUM_BLOCKS_PER_STATE_SMALL, 1, 1),
            #     block=[NUM_THREADS, 1, 1],
            #     smem=smem_bytes_small,
            #     stream=stream,
            # )
    
    return RunSmallBatch().module


_jit_functions = None
def _get_jit_functions(*args, **kwargs):
    global _jit_functions
    if _jit_functions is None:
        _jit_functions = _create_jit_functions(*args, **kwargs)
    return _jit_functions


def _get_compiled_kernel(N, H, HV, K, V, pool_size, use_small_batch, is_varlen_decode):
    """Get or compile the kernel for given dimensions."""
    global _compiled_kernels

    key = (N, H, HV, K, V, pool_size, use_small_batch, is_varlen_decode)
    if key in _compiled_kernels:
        return _compiled_kernels[key]

    cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device="cuda")

    if is_varlen_decode:
        q = torch.zeros(1, N, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(1, N, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
        a = torch.zeros(N, HV, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(N, HV, dtype=torch.bfloat16, device="cuda")
        o = torch.zeros(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
    else:
        q = torch.zeros(N, 1, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(N, 1, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros(N, 1, HV, V, dtype=torch.bfloat16, device="cuda")
        a = torch.zeros(N, 1, HV, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(N, 1, HV, dtype=torch.bfloat16, device="cuda")
        o = torch.zeros(N, 1, HV, V, dtype=torch.bfloat16, device="cuda")

    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, dtype=torch.bfloat16, device="cuda")
    h0_source = torch.zeros(pool_size, HV, K, V, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(N, dtype=torch.int32, device="cuda")

    cu_seqlens_tensor = cu_seqlens
    q_tensor = q
    k_tensor = k
    v_tensor = v
    a_tensor = a
    b_tensor = b
    A_log_tensor = A_log
    dt_bias_tensor = dt_bias
    h0_source_tensor = h0_source
    h0_indices_tensor = h0_indices
    o_tensor = o

    stream = torch.cuda.current_stream().cuda_stream

    scale = K**-0.5
    softplus_beta = 1.0
    softplus_threshold = 20.0

    B_compile = 1 if is_varlen_decode else N
    T_compile = N if is_varlen_decode else 1

    # def get_kernel_func():
    #     run_small, run_small_varlen, run_large, run_large_varlen = _get_jit_functions()
    #     if use_small_batch:
    #         kernel_func = run_small_varlen if is_varlen_decode else run_small
    #     else:
    #         kernel_func = run_large_varlen if is_varlen_decode else run_large
    #     return kernel_func

    kernel_func = _get_jit_functions(
        softplus_beta,
        softplus_threshold,
        scale,
        B_compile,
        T_compile,
        H,
        K,
        V,
        HV,
        use_initial_state=True,
        use_qk_l2norm=True,
        N=N)
    compiled_kernel = flydsl.compile(kernel_func)
    _compiled_kernels[key] = compiled_kernel
    logger.info(
        f"FLY DSL GDN kernel compiled: N={N}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, small_batch={use_small_batch}, varlen={is_varlen_decode}"
    )

    return compiled_kernel


def flydsl_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """CuTe DSL implementation of fused sigmoid gating delta rule update."""

    B_q, T_q, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    N = initial_state_indices.shape[0]

    is_varlen_decode = B_q == 1 and T_q == N and N > 1
    if scale is None:
        scale = K**-0.5

    use_small_batch = N < SMALL_BATCH_THRESHOLD

    if initial_state_source.dim() == 1:
        pool_size = initial_state_source.numel() // (HV * K * V)
        h0_source = initial_state_source.view(pool_size, HV, K, V)
    elif initial_state_source.dim() == 4:
        pool_size = initial_state_source.shape[0]
        h0_source = initial_state_source
    else:
        raise ValueError(
            f"Unexpected initial_state_source shape: {initial_state_source.shape}"
        )

    if is_varlen_decode:
        if a.dim() == 3:
            a = a.squeeze(0)
        if b.dim() == 3:
            b = b.squeeze(0)
        o = q.new_empty(1, N, HV, V, dtype=torch.bfloat16)
    else:
        if a.dim() == 2:
            a = a.unsqueeze(1)
        if b.dim() == 2:
            b = b.unsqueeze(1)
        o = q.new_empty(N, 1, HV, V, dtype=torch.bfloat16)

    q, k, v = [t.contiguous() for t in (q, k, v)]

    global _cu_seqlens_cache
    if cu_seqlens is not None:
        cu_seqlens_to_use = cu_seqlens
    else:
        cache_key = (N, str(q.device))
        if cache_key not in _cu_seqlens_cache:
            _cu_seqlens_cache[cache_key] = torch.arange(
                N + 1, dtype=torch.int32, device=q.device
            )
        cu_seqlens_to_use = _cu_seqlens_cache[cache_key]

    cu_seqlens_tensor = cu_seqlens_to_use.detach()
    q_tensor = q.detach()
    k_tensor = k.detach()
    v_tensor = v.detach()
    a_tensor = a.detach()
    b_tensor = b.detach()
    A_log_tensor = A_log.detach()
    dt_bias_tensor = dt_bias.detach()
    h0_source_tensor = h0_source.detach()
    h0_indices_tensor = initial_state_indices.detach()
    o_tensor = o.detach()

    stream = torch.cuda.current_stream().cuda_stream

    compiled_kernel = _get_compiled_kernel(
        N, H, HV, K, V, pool_size, use_small_batch, is_varlen_decode
    )

    compiled_kernel(
        cu_seqlens_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        a_tensor,
        b_tensor,
        A_log_tensor,
        dt_bias_tensor,
        h0_source_tensor,
        h0_indices_tensor,
        o_tensor,
        stream,
    )

    return o


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.cuda.device(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


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


@input_guard
def fused_sigmoid_gating_delta_rule_update(
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
    use_qk_l2norm_in_kernel: bool = False,
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

    o = q.new_empty(NK, *v.shape)
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
    o = o.squeeze(0)
    return o


def run_triton_kernel(A_log, dt_bias, q, k, v, a, b, initial_state, indices, scale):
    return fused_sigmoid_gating_delta_rule_update(
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
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
    )


@pytest.mark.parametrize("B", [16, 128])
def test_cutedsl_gdn_precision(B: int):
    """Test precision of CuTe DSL GDN kernel against Triton reference."""
    torch.manual_seed(2025)
    T, H, K, V, HV = 1, 16, 128, 128, 32
    scale = K**-0.5

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(B, dtype=torch.int32, device="cuda")
    state_cutedsl = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    state_triton = state_cutedsl.clone().reshape(-1).contiguous()

    # Warmup compilation
    _ = flydsl_fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, q, k, v, a, b, state_cutedsl.clone(), indices, scale=scale
    )
    torch.cuda.synchronize()

    # Fresh state for actual test
    state_cutedsl = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    state_triton = state_cutedsl.clone().reshape(-1).contiguous()

    out_cutedsl = flydsl_fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, q, k, v, a, b, state_cutedsl, indices, scale=scale
    )
    out_triton = run_triton_kernel(
        A_log, dt_bias, q, k, v, a, b, state_triton, indices, scale
    )

    # Check precision: diff > 0.1 must be < 1% of elements
    abs_diff = (out_triton.float() - out_cutedsl.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    fail_rate = (abs_diff > 0.1).float().mean().item() * 100
    has_nan = torch.isnan(out_cutedsl).any() or torch.isinf(out_cutedsl).any()

    kernel_type = "SmallBatch" if B < 32 else "LargeBatch"
    print(
        f"\n  B={B} ({kernel_type}): max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, fail_rate={fail_rate:.2f}%"
    )

    assert not has_nan, "Output contains NaN/Inf"
    assert fail_rate < 1.0, f"Fail rate {fail_rate:.2f}% >= 1%"
