from dataclasses import dataclass
from typing import Any, Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch

from .common import run_cached
from .gemm_a16w16_gfx950 import (
    _dynamic_tensor_arg,
    get_split_k_buffers,
    write_cshuffle_vec_to_global,
)
from .gemm_a16w16_gfx950_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    SPLIT_K_SEMAPHORE_MAX_LEN,
    BlockSwizzle,
    SplitKProtocol,
    get_wave_lds_offset,
    make_fp8_lds_layout,
    swizzled_contiguous_idx,
    wait_vmcnt_and_barrier,
)

GEMM_FP8_PTPC_DTYPE_FP32 = 1
GEMM_FP8_PTPC_DTYPE_BF16 = 2
GEMM_FP8_PTPC_DTYPE_FP8 = 4
_SHUFFLE_DTYPE_BYTES = 2


@fx.struct
class GemmFp8PtpcGfx950Param:
    in_dtype_id: fx.Constexpr[int]
    out_dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    is_split_k: fx.Constexpr[bool]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    k_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    use_half_tile_interleaved: fx.Constexpr[bool]
    a_is_transposed: fx.Constexpr[bool]
    b_is_transposed: fx.Constexpr[bool]
    has_bias: fx.Constexpr[bool]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    # derived params
    async_load_bytes: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    out_data_bytes: fx.Constexpr[int]
    cshuffle_r2g_vec_size: fx.Constexpr[int]
    ldg_x_threads: fx.Constexpr[int]
    block_threads: fx.Constexpr[int]
    ldg_a_iters: fx.Constexpr[int]
    ldg_b_iters: fx.Constexpr[int]


@dataclass(slots=True, kw_only=True, eq=False)
class GemmABLoadContext:
    wave_offset: Any
    tid: Any
    ks_begin: Any
    param: GemmFp8PtpcGfx950Param
    async_g2s_copy_atom: Any
    a_s2r_copy_atom: Any
    b_s2r_copy_atom: Any
    thr_copy_a: Any
    thr_copy_b: Any


@dataclass(slots=True, kw_only=True, eq=False)
class AsyncLoadOperand:
    context: GemmABLoadContext
    src_base: Any
    lds_layout: Any
    outer_tile_size: Any
    outer_bound: Any
    leading_stride: Any
    load_iters: Any
    is_k_major: Any


def make_gemm_fp8_ptpc_gfx950_param(
    out_dtype_id: int,
    block_m: int = 256,
    block_n: int = 256,
    block_k: int = 128,
    stages: int = 2,
    split_k: int = 1,
    m_waves: int = 2,
    n_waves: int = 4,
    k_waves: int = 1,
    group_m: int = 0,
    use_half_tile_interleaved: bool = True,
    a_is_transposed: bool = False,
    b_is_transposed: bool = True,
    has_bias: bool = False,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 128,
    in_dtype_id: int = GEMM_FP8_PTPC_DTYPE_FP8,
) -> GemmFp8PtpcGfx950Param:
    if in_dtype_id != GEMM_FP8_PTPC_DTYPE_FP8:
        raise ValueError(f"unsupported in_dtype_id={in_dtype_id}")
    if out_dtype_id not in (GEMM_FP8_PTPC_DTYPE_BF16, GEMM_FP8_PTPC_DTYPE_FP32):
        raise ValueError(f"unsupported out_dtype_id={out_dtype_id}")
    if block_m <= 0 or block_n <= 0 or block_k <= 0 or stages <= 0 or split_k <= 0:
        raise ValueError("block_m, block_n, block_k, stages, and split_k must be positive")
    if (mma_m, mma_n, mma_k) != (16, 16, 128):
        raise ValueError("the gfx950 FP8 PTPC kernel requires mma=16x16x128")
    if block_k < 128 or block_k % 128 != 0:
        raise ValueError(f"block_k must be a multiple of 128 for FP8 PTPC, got {block_k}")
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0 or k_waves <= 0:
        raise ValueError("m_waves, n_waves, and k_waves must be positive")
    if m_waves * n_waves * k_waves > 16:
        raise ValueError("the workgroup cannot contain more than 16 waves")
    if group_m < 0:
        raise ValueError("group_m must be non-negative")
    in_dbytes = 1
    out_dbytes = 4 if out_dtype_id == GEMM_FP8_PTPC_DTYPE_FP32 else 2
    block_threads = m_waves * n_waves * k_waves * GFX950_WAVE_SIZE
    max_cshuffle_r2g_vec_size = 16 // out_dbytes
    if use_half_tile_interleaved:
        if k_waves != 1:
            raise ValueError("half-tile interleaved does not support slice-K")
        half_block_m = block_m // 2
        half_block_n = block_n // 2
        assert stages == 2
        assert m_waves == 2 and n_waves >= 2
        assert half_block_m * 2 == block_m
        assert half_block_n * 2 == block_n
        mma_m_half_repeat = half_block_m // m_waves // mma_m
        mma_n_half_repeat = half_block_n // n_waves // mma_n
        assert mma_m_half_repeat * m_waves * mma_m == half_block_m
        assert mma_n_half_repeat * n_waves * mma_n == half_block_n
        stg_size_per_m_step = m_waves * mma_m * half_block_n
        assert stg_size_per_m_step % block_threads == 0
        stg_work_size_per_m_step = stg_size_per_m_step // block_threads
        cshuffle_r2g_vec_size = min(max_cshuffle_r2g_vec_size, stg_work_size_per_m_step)
        assert cshuffle_r2g_vec_size in (4, 8)
        assert stg_work_size_per_m_step % cshuffle_r2g_vec_size == 0
        assert half_block_n % cshuffle_r2g_vec_size == 0
    else:
        cshuffle_r2g_vec_size = min(max_cshuffle_r2g_vec_size, 4) if split_k > 1 else max_cshuffle_r2g_vec_size
        assert block_n % cshuffle_r2g_vec_size == 0
    smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
    smem_bytes = max(smem_bytes, k_waves * block_m * block_n * _SHUFFLE_DTYPE_BYTES)
    arch = get_rocm_arch()
    SMEM_CAPACITY_MAP = {
        "gfx942": 65536,
        "gfx950": 163840,
    }
    smem_capacity = SMEM_CAPACITY_MAP[arch]
    if smem_bytes > smem_capacity:
        raise ValueError(
            "staged LDS buffers exceed the device shared-memory capacity: "
            f"stages={stages}, block_m={block_m}, block_n={block_n}, "
            f"block_k={block_k}, smem_bytes={smem_bytes}, "
            f"capacity={smem_capacity} for arch={arch}"
        )
    # async load check
    async_load_vec_size = GFX950_DMA_BYTES // in_dbytes
    ldg_x_threads = block_k // async_load_vec_size
    if ldg_x_threads * async_load_vec_size != block_k:
        raise ValueError(
            "block_k must be divisible by the async load vector size: "
            f"block_k={block_k}, async_load_vec_size={async_load_vec_size}, "
            f"covered_k={ldg_x_threads * async_load_vec_size}"
        )
    ldg_y_threads = block_threads // ldg_x_threads
    if ldg_y_threads * ldg_x_threads != block_threads:
        raise ValueError(
            "ldg thread layout must exactly cover the workgroup: "
            f"ldg_y_threads={ldg_y_threads}, ldg_x_threads={ldg_x_threads}, "
            f"block_threads={block_threads}"
        )
    ldg_a_iters = (block_m * block_k) // (block_threads * async_load_vec_size)
    ldg_b_iters = (block_n * block_k) // (block_threads * async_load_vec_size)
    if use_half_tile_interleaved:
        half_ldg_a_iters = ((block_m // 2) * block_k) // (block_threads * async_load_vec_size)
        half_ldg_b_iters = ((block_n // 2) * block_k) // (block_threads * async_load_vec_size)
        if half_ldg_a_iters * block_threads * async_load_vec_size != (block_m // 2) * block_k:
            raise ValueError(
                "Half-tile A async load tile must be exactly covered by whole-thread vector loads: "
                f"half_block_m={block_m // 2}, block_k={block_k}, "
                f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
                f"half_ldg_a_iters={half_ldg_a_iters}"
            )
        if half_ldg_b_iters * block_threads * async_load_vec_size != (block_n // 2) * block_k:
            raise ValueError(
                "Half-tile B async load tile must be exactly covered by whole-thread vector loads: "
                f"half_block_n={block_n // 2}, block_k={block_k}, "
                f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
                f"half_ldg_b_iters={half_ldg_b_iters}"
            )
    if ldg_a_iters * block_threads * async_load_vec_size != block_m * block_k:
        raise ValueError(
            "A async load tile must be exactly covered by whole-thread vector loads: "
            f"block_m={block_m}, block_k={block_k}, "
            f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
            f"ldg_a_iters={ldg_a_iters}, "
            f"covered={ldg_a_iters * block_threads * async_load_vec_size}, "
            f"required={block_m * block_k}"
        )
    if ldg_b_iters * block_threads * async_load_vec_size != block_n * block_k:
        raise ValueError(
            "B async load tile must be exactly covered by whole-thread vector loads: "
            f"block_n={block_n}, block_k={block_k}, "
            f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
            f"ldg_b_iters={ldg_b_iters}, "
            f"covered={ldg_b_iters * block_threads * async_load_vec_size}, "
            f"required={block_n * block_k}"
        )
    assert (stages - 2) * (ldg_a_iters + ldg_b_iters) < 63
    mma_m_repeat = block_m // m_waves // mma_m
    mma_n_repeat = block_n // n_waves // mma_n
    mma_k_repeat = block_k // k_waves // mma_k
    if mma_m_repeat * m_waves * mma_m != block_m:
        raise ValueError(
            "block_m must be divisible by m_waves * mma_m: "
            f"block_m={block_m}, m_waves={m_waves}, mma_m={mma_m}, "
            f"mma_m_repeat={mma_m_repeat}, covered_m={mma_m_repeat * m_waves * mma_m}"
        )
    if mma_n_repeat * n_waves * mma_n != block_n:
        raise ValueError(
            "block_n must be divisible by n_waves * mma_n: "
            f"block_n={block_n}, n_waves={n_waves}, mma_n={mma_n}, "
            f"mma_n_repeat={mma_n_repeat}, covered_n={mma_n_repeat * n_waves * mma_n}"
        )
    if mma_k_repeat * k_waves * mma_k != block_k:
        raise ValueError(
            "block_k must be divisible by k_waves * mma_k: "
            f"block_k={block_k}, k_waves={k_waves}, mma_k={mma_k}, "
            f"mma_k_repeat={mma_k_repeat}, "
            f"covered_k={mma_k_repeat * k_waves * mma_k}"
        )
    return GemmFp8PtpcGfx950Param(
        in_dtype_id=in_dtype_id,
        out_dtype_id=out_dtype_id,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        is_split_k=split_k > 1,
        m_waves=m_waves,
        n_waves=n_waves,
        k_waves=k_waves,
        group_m=group_m,
        use_half_tile_interleaved=use_half_tile_interleaved,
        a_is_transposed=a_is_transposed,
        b_is_transposed=b_is_transposed,
        has_bias=has_bias,
        async_load_bytes=GFX950_DMA_BYTES,
        in_data_bytes=in_dbytes,
        out_data_bytes=out_dbytes,
        cshuffle_r2g_vec_size=cshuffle_r2g_vec_size,
        ldg_x_threads=ldg_x_threads,
        block_threads=block_threads,
        ldg_a_iters=ldg_a_iters,
        ldg_b_iters=ldg_b_iters,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
    )


def make_gemm_fp8_ptpc_gfx950_kernel_name(param: GemmFp8PtpcGfx950Param):
    out_suffix = "_fp32" if param.out_dtype_id == GEMM_FP8_PTPC_DTYPE_FP32 else ""
    name = f"hgemm_fp8_ptpc{out_suffix}_t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}"
    name += "_ksd" if param.is_split_k else "_ks1"
    name += f"_w{param.m_waves}x{param.n_waves}x{param.k_waves}"
    name += f"_gm{param.group_m}"
    name += f"_bias{int(param.has_bias)}"
    a_layout = "t" if param.a_is_transposed else "n"
    b_layout = "t" if param.b_is_transposed else "n"
    name += f"_l{a_layout}{b_layout}"
    name += "_phti" if param.use_half_tile_interleaved else "_pft"
    return name


def make_gemm_ab_lds_layouts(rows_a, rows_b, block_k, a_is_transposed, b_is_transposed):
    return (
        make_fp8_lds_layout(rows_a, block_k, a_is_transposed),
        make_fp8_lds_layout(rows_b, block_k, not b_is_transposed),
    )


def load_fp8_kcontig_16b_i32x4(lds_base, lds_layout, row, col):
    offset = fx.get_scalar(fx.crd2idx((row, col), lds_layout))
    ptr_off = fx.add_offset(lds_base, fx.make_int_tuple(offset))
    i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
    return fx.make_view(i8_iter, fx.make_layout(16, 1)).load().bitcast(fx.Int32)


def load_fp8_kcontig_plus64_i32x8(lds_base, lds_layout, row, col_base):
    lo = load_fp8_kcontig_16b_i32x4(lds_base, lds_layout, row, col_base)
    hi = load_fp8_kcontig_16b_i32x4(lds_base, lds_layout, row, col_base + 64)
    return lo.shuffle(hi, list(range(8)))


def make_fp8_ptpc_tiled_mma(param: GemmFp8PtpcGfx950Param):
    # gfx950 peak FP8 is mfma.scale.f32.16x16x128.f8f6f4. Hardware
    # scale_a/scale_b stay at identity; PTPC f32 scales are applied
    # in the epilogue. MFMA_Scale has no rebuildStaticValue, so a
    # TiledMma kernel argument never folds; build it in-kernel.
    mma_atom = fx.make_mma_atom(
        fx.rocdl.cdna4.MFMA_Scale(param.mma_m, param.mma_n, param.mma_k, fx.Float8E4M3FN)
    )
    k_per_mfma_group = param.mma_k // 4
    return fx.make_tiled_mma(
        mma_atom,
        fx.make_layout(
            (param.m_waves, param.n_waves, 1),
            (param.n_waves, 1, 0),
        ),
        fx.make_tile(
            None,
            None,
            fx.make_layout(
                (k_per_mfma_group, 4),
                (1, k_per_mfma_group),
            ),
        ),
    )


def make_gemm_ab_load_context(
    elem_dtype,
    tiled_mma,
    copy_tid,
    load_tid,
    ks_begin,
    param: GemmFp8PtpcGfx950Param,
):
    uni_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    async_g2s_copy_atom = fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128)

    if const_expr(param.a_is_transposed):
        a_s2r_copy_atom = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans8_64b(), elem_dtype)
        a_tiled_copy_atom = a_s2r_copy_atom
    else:
        a_s2r_copy_atom = uni_copy_atom
        a_tiled_copy_atom = buffer_copy_atom
    if const_expr(not param.b_is_transposed):
        b_s2r_copy_atom = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans8_64b(), elem_dtype)
        b_tiled_copy_atom = b_s2r_copy_atom
    else:
        b_s2r_copy_atom = uni_copy_atom
        b_tiled_copy_atom = buffer_copy_atom

    return GemmABLoadContext(
        wave_offset=get_wave_lds_offset(load_tid, param.async_load_bytes),
        tid=load_tid,
        ks_begin=ks_begin,
        param=param,
        async_g2s_copy_atom=async_g2s_copy_atom,
        a_s2r_copy_atom=a_s2r_copy_atom,
        b_s2r_copy_atom=b_s2r_copy_atom,
        thr_copy_a=fx.make_tiled_copy_A(a_tiled_copy_atom, tiled_mma).get_slice(copy_tid),
        thr_copy_b=fx.make_tiled_copy_B(b_tiled_copy_atom, tiled_mma).get_slice(copy_tid),
    )


def async_load_operand(
    operand: AsyncLoadOperand,
    lds_base,
    global_outer_offset,
    k_tile,
):
    context = operand.context
    param = context.param
    tid = context.tid
    block_threads = param.block_threads
    async_load_vec_size = param.async_load_bytes // param.in_data_bytes
    ldg_x_threads = param.ldg_x_threads
    block_k = param.block_k
    elem_bytes = operand.src_base.dtype.width // 8
    lds_ptr = lds_base + fx.Int32(context.wave_offset) // elem_bytes
    g2s_copy_layout = fx.make_layout(async_load_vec_size, 1)
    for i in range_constexpr(operand.load_iters):
        global_tid = block_threads * i + tid
        if const_expr(operand.is_k_major):
            outer_x_threads = operand.outer_tile_size // async_load_vec_size
            outer_lds_idx = global_tid % outer_x_threads * async_load_vec_size
            k_local_idx = global_tid // outer_x_threads
            outer_local_idx = swizzled_contiguous_idx(
                outer_lds_idx,
                k_local_idx,
                operand.lds_layout,
                operand.outer_tile_size,
            )
            global_k_idx = context.ks_begin + k_tile * block_k + k_local_idx
        else:
            outer_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_k_idx = (
                context.ks_begin
                + k_tile * block_k
                + swizzled_contiguous_idx(
                    outer_local_idx,
                    k_local_idx,
                    operand.lds_layout,
                    block_k,
                )
            )
        global_outer_idx = global_outer_offset + outer_local_idx
        safe_global_outer_idx = (global_outer_idx < operand.outer_bound).select(global_outer_idx, 0)
        if const_expr(operand.is_k_major):
            global_offset = global_k_idx * operand.leading_stride + safe_global_outer_idx
        else:
            global_offset = safe_global_outer_idx * operand.leading_stride + global_k_idx
        src = fx.make_view(operand.src_base + global_offset, g2s_copy_layout)
        dst = fx.make_view(lds_ptr, g2s_copy_layout)
        rocdl.sched_barrier(0)
        fx.copy_atom_call(context.async_g2s_copy_atom, src, dst)
        rocdl.sched_barrier(0)
        if i < operand.load_iters - 1:
            lds_ptr = lds_ptr + block_threads * async_load_vec_size


@flyc.jit
def apply_fp8_ptpc_epilogue(
    acc,
    scale_a_buf,
    scale_b_buf,
    bias_buf,
    global_row,
    global_col,
    m,
    n,
    has_bias,
    is_split_k,
    is_slice_k,
    apply_bias,
):
    safe_m = (global_row < m).select(global_row, 0)
    safe_n = (global_col < n).select(global_col, 0)
    val = acc * scale_a_buf[safe_m] * scale_b_buf[safe_n]
    if const_expr(has_bias and not is_split_k):
        bias_val = bias_buf[safe_n].to(fx.Float32)
        if const_expr(is_slice_k):
            bias_val = apply_bias.select(bias_val, fx.Float32(0.0))
        val = val + bias_val
    return val.to(fx.BFloat16)


@flyc.kernel
def gemm_fp8_ptpc_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    scale_a: fx.Tensor,
    scale_b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    split_k: fx.Int32,
    working_k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    param: GemmFp8PtpcGfx950Param,
):
    tiled_mma = make_fp8_ptpc_tiled_mma(param)
    is_split_k = param.is_split_k
    is_slice_k = param.k_waves > 1
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    k_waves = param.k_waves
    k_mma_iters_per_wave = block_k // (k_waves * param.mma_k)
    stages = param.stages
    block_threads = param.block_threads
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters
    cshuffle_r2g_vec_size = param.cshuffle_r2g_vec_size
    elem_dtype = fx.Float8E4M3FN
    shuffle_dtype = fx.BFloat16
    global_output_dtype = fx.Float32 if const_expr(param.out_dtype_id == GEMM_FP8_PTPC_DTYPE_FP32) else shuffle_dtype
    if const_expr(is_split_k):
        splitk_protocol = SplitKProtocol(
            block_m,
            block_n,
            cshuffle_r2g_vec_size,
            param.out_data_bytes,
            block_threads,
            param.has_bias,
        )

    tid = fx.thread_idx.x
    threads_per_k_slice = param.m_waves * param.n_waves * GFX950_WAVE_SIZE
    tid_in_k_slice = tid % threads_per_k_slice
    k_wave_idx = tid // threads_per_k_slice
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    block_swizzle = BlockSwizzle(NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m)
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    ks_idx = fx.block_idx.y
    ks_begin = ks_idx * working_k
    ks_end = ks_begin + working_k
    ks_end = (ks_end < k).select(ks_end, k)
    k_tiles = (ks_end - ks_begin) // block_k
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[shuffle_dtype, k_waves * block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    scale_a_buf = fx.rocdl.make_buffer_tensor(scale_a, max_size=True)
    scale_b_buf = fx.rocdl.make_buffer_tensor(scale_b, max_size=True)
    if const_expr(param.has_bias):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

    if const_expr(is_split_k):
        splitk_protocol.init(
            semaphore,
            signal,
            out,
            bias,
            tid,
            ks_idx,
            m,
            n,
            block_m_offset,
            block_n_offset,
            global_output_dtype,
            fx.block_idx.x,
            n,
        )

    ab_load_context = make_gemm_ab_load_context(
        elem_dtype,
        tiled_mma,
        copy_tid=tid_in_k_slice,
        load_tid=tid,
        ks_begin=ks_begin,
        param=param,
    )
    a_s2r_copy_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_copy_atom = ab_load_context.b_s2r_copy_atom
    thr_copy_A = ab_load_context.thr_copy_a
    thr_copy_B = ab_load_context.thr_copy_b
    mma_atom = fx.make_mma_atom(
        fx.rocdl.cdna4.MFMA_Scale(param.mma_m, param.mma_n, param.mma_k, fx.Float8E4M3FN)
    )
    use_kcontig_atom_s2r = const_expr(not param.a_is_transposed) and const_expr(
        param.b_is_transposed
    )
    m_waves = param.m_waves
    n_waves = param.n_waves
    warp_m_steps = block_m // m_waves // param.mma_m
    warp_n_steps = block_n // n_waves // param.mma_n
    warp_m = warp_m_steps * param.mma_m
    warp_n = warp_n_steps * param.mma_n
    w_tid = tid_in_k_slice % GFX950_WAVE_SIZE
    wid = tid_in_k_slice // GFX950_WAVE_SIZE
    warp_m_idx = (wid // n_waves) * warp_m
    warp_n_idx = (wid % n_waves) * warp_n
    stmatrix_c_n_idx = w_tid % param.mma_n
    stmatrix_c_m_vec_idx = (w_tid // param.mma_n) * 4
    c_frags_len = warp_m_steps * warp_n_steps
    a_mma_rmem = fx.make_rmem_tensor(8, fx.Int32)
    b_mma_rmem = fx.make_rmem_tensor(8, fx.Int32)
    c_mma_rmem = fx.make_rmem_tensor(4, fx.Float32)
    acc_init = fx.Vector.filled(4, 0.0, fx.Float32)
    c_frags = [acc_init] * c_frags_len

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]

    thr_mma = tiled_mma.thr_slice(tid_in_k_slice)

    a_lds_layout, b_lds_layout = make_gemm_ab_lds_layouts(
        block_m,
        block_n,
        block_k,
        param.a_is_transposed,
        param.b_is_transposed,
    )
    a_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(a_buf),
        lds_layout=a_lds_layout,
        outer_tile_size=block_m,
        outer_bound=m,
        leading_stride=a_leading_stride,
        load_iters=ldg_a_iters,
        is_k_major=param.a_is_transposed,
    )
    b_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(b_buf),
        lds_layout=b_lds_layout,
        outer_tile_size=block_n,
        outer_bound=n,
        leading_stride=b_leading_stride,
        load_iters=ldg_b_iters,
        is_k_major=not param.b_is_transposed,
    )
    c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))

    sA = fx.make_view(smem_a, a_lds_layout)
    sB = fx.make_view(smem_b, b_lds_layout)
    sC_write = fx.make_view(smem_c + k_wave_idx * block_m * block_n, c_lds_layout)

    if const_expr(not use_kcontig_atom_s2r):
        frag_A = thr_mma.make_fragment_A(sA)
        frag_B = thr_mma.make_fragment_B(sB)
        frag_C = thr_mma.make_fragment_C(gC)
        frag_A_retile = thr_copy_A.retile(frag_A)
        frag_B_retile = thr_copy_B.retile(frag_B)
        frag_C.fill(0.0)

    row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
    thr_mma_cRow = thr_mma.partition_C(row_coords)
    thr_mma_cCol = thr_mma.partition_C(col_coords)

    if const_expr(is_split_k):
        splitk_protocol.zero_c()

    def async_load_a_to_lds(k_tile, stage):
        async_load_operand(
            a_load_operand,
            lds_base=smem_a + stage * block_m * block_k,
            global_outer_offset=block_m_offset,
            k_tile=k_tile,
        )

    def async_load_b_to_lds(k_tile, stage):
        async_load_operand(
            b_load_operand,
            lds_base=smem_b + stage * block_n * block_k,
            global_outer_offset=block_n_offset,
            k_tile=k_tile,
        )

    def compute_stage(read_stage, c_frags_in):
        lds_a = smem_a + read_stage * block_m * block_k
        lds_b = smem_b + read_stage * block_n * block_k
        if const_expr(use_kcontig_atom_s2r):
            c_frags_out = [cx for cx in c_frags_in]
            for block_k_iter in range_constexpr(k_mma_iters_per_wave):
                ki = k_wave_idx * k_mma_iters_per_wave + block_k_iter
                for ni in range_constexpr(warp_n_steps):
                    row_b = warp_n_idx + ni * param.mma_n + (w_tid % param.mma_n)
                    col_b = ki * param.mma_k + (w_tid // param.mma_n) * 16
                    b_vec = load_fp8_kcontig_plus64_i32x8(lds_b, b_lds_layout, row_b, col_b)
                    for mi in range_constexpr(warp_m_steps):
                        row_a = warp_m_idx + mi * param.mma_m + (w_tid % param.mma_m)
                        col_a = ki * param.mma_k + (w_tid // param.mma_m) * 16
                        c_idx = mi * warp_n_steps + ni
                        a_mma_rmem.store(
                            load_fp8_kcontig_plus64_i32x8(lds_a, a_lds_layout, row_a, col_a)
                        )
                        b_mma_rmem.store(b_vec)
                        c_mma_rmem.store(c_frags_out[c_idx])
                        fx.gemm(mma_atom, c_mma_rmem, a_mma_rmem, b_mma_rmem, c_mma_rmem)
                        c_frags_out[c_idx] = c_mma_rmem.load()
            return c_frags_out

        thr_sA_s2r = thr_copy_A.partition_S(fx.make_view(lds_a, a_lds_layout))
        thr_sB_s2r = thr_copy_B.partition_S(fx.make_view(lds_b, b_lds_layout))

        def compute_k_chunk(block_k_iter):
            frag_A_chunk = frag_A[None, None, block_k_iter]
            fx.copy(
                b_s2r_copy_atom,
                thr_sB_s2r[None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
            fx.copy(
                a_s2r_copy_atom,
                thr_sA_s2r[None, None, block_k_iter],
                frag_A_retile[None, None, block_k_iter],
            )
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A_chunk,
                frag_B[None, None, block_k_iter],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

        for k_slice in range_constexpr(k_waves):
            if k_wave_idx == k_slice:
                for block_k_iter in range_constexpr(k_mma_iters_per_wave):
                    k_iter = k_slice * k_mma_iters_per_wave + block_k_iter
                    compute_k_chunk(k_iter)
        return c_frags_in

    # Prime the staged LDS pipeline: preload the first `stages - 1` K tiles
    # before entering the main loop that overlaps async loads with compute.
    for stage in range_constexpr(stages - 1):
        async_load_b_to_lds(stage, stage)
        async_load_a_to_lds(stage, stage)
        rocdl.asyncmark()
    rocdl.sched_barrier(0)

    main_loop_end = k_tiles - (stages - 1)
    for k_tile in range(0, main_loop_end, 1):
        current_stage = k_tile % stages
        write_stage = (current_stage + stages - 1) % stages
        rocdl.wait_asyncmark(stages - 2)
        rocdl.s_barrier()
        async_load_b_to_lds(k_tile + (stages - 1), write_stage)
        async_load_a_to_lds(k_tile + (stages - 1), write_stage)
        rocdl.asyncmark()
        c_frags = compute_stage(current_stage, c_frags)
        rocdl.sched_barrier(0)

    current_stage = main_loop_end % stages
    for s in range_constexpr(0, stages - 1):
        rocdl.wait_asyncmark(stages - 2 - s)
        rocdl.s_barrier()
        c_frags = compute_stage(current_stage, c_frags)
        current_stage = (current_stage + 1) % stages

    gpu.barrier()
    apply_bias = k_wave_idx == 0
    if const_expr(use_kcontig_atom_s2r):
        for mi in range_constexpr(warp_m_steps):
            for ni in range_constexpr(warp_n_steps):
                c_vec = c_frags[mi * warp_n_steps + ni]
                col = warp_n_idx + ni * param.mma_n + stmatrix_c_n_idx
                for kk in range_constexpr(4):
                    row = warp_m_idx + mi * param.mma_m + stmatrix_c_m_vec_idx + kk
                    sC_write[row, col] = apply_fp8_ptpc_epilogue(
                        c_vec[kk],
                        scale_a_buf,
                        scale_b_buf,
                        bias_buf if const_expr(param.has_bias) else scale_b_buf,
                        block_m_offset + row,
                        block_n_offset + col,
                        m,
                        n,
                        param.has_bias,
                        is_split_k,
                        is_slice_k,
                        apply_bias,
                    )
    else:
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            sC_write[row, col] = apply_fp8_ptpc_epilogue(
                frag_C[i],
                scale_a_buf,
                scale_b_buf,
                bias_buf if const_expr(param.has_bias) else scale_b_buf,
                block_m_offset + row,
                block_n_offset + col,
                m,
                n,
                param.has_bias,
                is_split_k,
                is_slice_k,
                apply_bias,
            )

    if const_expr(is_split_k):
        splitk_protocol.wait_until_initialized()
    else:
        gpu.barrier()

    cshuffle_r2g_x_threads = block_n // cshuffle_r2g_vec_size
    cshuffle_vectors = block_m * block_n // cshuffle_r2g_vec_size
    cshuffle_iters = (cshuffle_vectors + block_threads - 1) // block_threads
    for i in range_constexpr(cshuffle_iters):
        vector_idx = block_threads * i + tid
        if vector_idx < cshuffle_vectors:
            local_row = vector_idx // cshuffle_r2g_x_threads
            local_col = vector_idx % cshuffle_r2g_x_threads * cshuffle_r2g_vec_size
            global_row = block_m_offset + local_row
            global_col = block_n_offset + local_col
            if (global_row < m) and (global_col < n):
                c_vec = fx.ptr_load(
                    smem_c + local_row * block_n + local_col,
                    result_type=fx.Vector.make_type(cshuffle_r2g_vec_size, shuffle_dtype),
                )
                for k_slice in range_constexpr(1, k_waves):
                    peer_c_vec = fx.ptr_load(
                        smem_c + k_slice * block_m * block_n + local_row * block_n + local_col,
                        result_type=fx.Vector.make_type(cshuffle_r2g_vec_size, shuffle_dtype),
                    )
                    c_vec = c_vec + peer_c_vec
                global_offset = global_row * n + global_col
                write_cshuffle_vec_to_global(
                    out,
                    out_buf,
                    global_offset,
                    c_vec,
                    is_split_k,
                    param.out_dtype_id == GEMM_FP8_PTPC_DTYPE_FP32,
                )
    if const_expr(is_split_k):
        splitk_protocol.finish_split(split_k)


@flyc.kernel
def gemm_fp8_ptpc_hti_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    scale_a: fx.Tensor,
    scale_b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    split_k: fx.Int32,
    working_k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    param: GemmFp8PtpcGfx950Param,
):
    tiled_mma = make_fp8_ptpc_tiled_mma(param)
    is_split_k = param.is_split_k
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    half_block_m = block_m // 2
    half_block_n = block_n // 2
    stages = param.stages
    block_threads = param.block_threads
    m_waves = param.m_waves
    n_waves = param.n_waves
    half_ldg_a_iters = param.ldg_a_iters // 2
    half_ldg_b_iters = param.ldg_b_iters // 2
    cshuffle_r2g_vec_size = param.cshuffle_r2g_vec_size
    elem_dtype = fx.Float8E4M3FN
    shuffle_dtype = fx.BFloat16
    global_output_dtype = fx.Float32 if const_expr(param.out_dtype_id == GEMM_FP8_PTPC_DTYPE_FP32) else shuffle_dtype
    if const_expr(is_split_k):
        splitk_protocol = SplitKProtocol(
            block_m,
            block_n,
            cshuffle_r2g_vec_size,
            param.out_data_bytes,
            block_threads,
            param.has_bias,
        )

    tid = fx.thread_idx.x
    wid = tid // GFX950_WAVE_SIZE
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    block_swizzle = BlockSwizzle(NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m)
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    ks_idx = fx.block_idx.y
    ks_begin = ks_idx * working_k
    ks_end = ks_begin + working_k
    ks_end = (ks_end < k).select(ks_end, k)
    k_tiles = (ks_end - ks_begin) // block_k
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[shuffle_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    scale_a_buf = fx.rocdl.make_buffer_tensor(scale_a, max_size=True)
    scale_b_buf = fx.rocdl.make_buffer_tensor(scale_b, max_size=True)
    if const_expr(param.has_bias):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

    if const_expr(is_split_k):
        splitk_protocol.init(
            semaphore,
            signal,
            out,
            bias,
            tid,
            ks_idx,
            m,
            n,
            block_m_offset,
            block_n_offset,
            global_output_dtype,
            fx.block_idx.x,
            n,
        )

    ab_load_context = make_gemm_ab_load_context(
        elem_dtype,
        tiled_mma,
        copy_tid=tid,
        load_tid=tid,
        ks_begin=ks_begin,
        param=param,
    )
    a_s2r_copy_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_copy_atom = ab_load_context.b_s2r_copy_atom
    thr_copy_A = ab_load_context.thr_copy_a
    thr_copy_B = ab_load_context.thr_copy_b
    thr_mma = tiled_mma.thr_slice(tid)
    mma_atom = fx.make_mma_atom(
        fx.rocdl.cdna4.MFMA_Scale(param.mma_m, param.mma_n, param.mma_k, fx.Float8E4M3FN)
    )
    # NT keeps the original hgemm S2R/MMA ABI (16B at K+0 and K+64 -> i32x8
    # atom). K-major layouts cannot pack that ABI, so they stay on tiled copy.
    use_kcontig_atom_s2r = const_expr(not param.a_is_transposed) and const_expr(
        param.b_is_transposed
    )
    warp_m_steps = half_block_m // m_waves // param.mma_m
    warp_n_steps = half_block_n // n_waves // param.mma_n
    warp_k_steps = block_k // param.mma_k
    warp_m = warp_m_steps * param.mma_m
    warp_n = warp_n_steps * param.mma_n
    w_tid = tid % GFX950_WAVE_SIZE
    warp_m_idx = (wid // n_waves) * warp_m
    warp_n_idx = (wid % n_waves) * warp_n
    stmatrix_c_n_idx = w_tid % param.mma_n
    stmatrix_c_m_vec_idx = (w_tid // param.mma_n) * 4
    c_frags_len = warp_m_steps * warp_n_steps
    stg_size_per_m_step = m_waves * param.mma_m * half_block_n
    stg_iters_per_m_step = (stg_size_per_m_step // block_threads) // cshuffle_r2g_vec_size
    stg_c_quad_x_threads = half_block_n // cshuffle_r2g_vec_size
    a_lds_layout, b_lds_layout = make_gemm_ab_lds_layouts(
        half_block_m,
        half_block_n,
        block_k,
        param.a_is_transposed,
        param.b_is_transposed,
    )
    a_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(a_buf),
        lds_layout=a_lds_layout,
        outer_tile_size=half_block_m,
        outer_bound=m,
        leading_stride=a_leading_stride,
        load_iters=half_ldg_a_iters,
        is_k_major=param.a_is_transposed,
    )
    b_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(b_buf),
        lds_layout=b_lds_layout,
        outer_tile_size=half_block_n,
        outer_bound=n,
        leading_stride=b_leading_stride,
        load_iters=half_ldg_b_iters,
        is_k_major=not param.b_is_transposed,
    )
    c_lds_layout = fx.make_layout((half_block_m, half_block_n), (half_block_n, 1))
    c_lds_layout_full = fx.make_layout((block_m, block_n), (block_n, 1))

    def half_a_base(stage, m_part):
        return smem_a + (stage * block_m + m_part * half_block_m) * block_k

    def half_b_base(stage, n_part):
        return smem_b + (stage * block_n + n_part * half_block_n) * block_k

    def half_c_base(m_part, n_part):
        tile_idx = m_part * 2 + n_part
        return smem_c + tile_idx * half_block_m * half_block_n

    def async_load_a_to_lds(m_part, k_tile, stage):
        async_load_operand(
            a_load_operand,
            lds_base=half_a_base(stage, m_part),
            global_outer_offset=block_m_offset + m_part * half_block_m,
            k_tile=k_tile,
        )

    def async_load_b_to_lds(n_part, k_tile, stage):
        async_load_operand(
            b_load_operand,
            lds_base=half_b_base(stage, n_part),
            global_outer_offset=block_n_offset + n_part * half_block_n,
            k_tile=k_tile,
        )

    def make_gC(m_part, n_part):
        return fx.flat_divide(out_buf, (half_block_m, half_block_n))[None, None, bid_m * 2 + m_part, bid_n * 2 + n_part]

    row_coords = fx.make_view(0, fx.make_layout((half_block_m, half_block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((half_block_m, half_block_n), (0, 1)))
    thr_mma_cRow = thr_mma.partition_C(row_coords)
    thr_mma_cCol = thr_mma.partition_C(col_coords)

    def make_c_fragment(m_part, n_part):
        gC = make_gC(m_part, n_part)
        return thr_mma.make_fragment_C(gC)

    def load_a_fragment(m_part, read_stage):
        sA = fx.make_view(half_a_base(read_stage, m_part), a_lds_layout)
        if const_expr(use_kcontig_atom_s2r):
            frag_A = [0] * (warp_k_steps * warp_m_steps)
            for block_k_iter in range_constexpr(block_k // param.mma_k):
                for mi in range_constexpr(warp_m_steps):
                    row = warp_m_idx + mi * param.mma_m + (w_tid % param.mma_m)
                    col_base = block_k_iter * param.mma_k + (w_tid // param.mma_m) * 16
                    frag_A[block_k_iter * warp_m_steps + mi] = load_fp8_kcontig_plus64_i32x8(
                        half_a_base(read_stage, m_part),
                        a_lds_layout,
                        row,
                        col_base,
                    )
            return frag_A
        frag_A = thr_mma.make_fragment_A(sA)
        frag_A_retile = thr_copy_A.retile(frag_A)
        thr_sA_s2r = thr_copy_A.partition_S(sA)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            fx.copy(
                a_s2r_copy_atom,
                thr_sA_s2r[None, None, block_k_iter],
                frag_A_retile[None, None, block_k_iter],
            )
        return frag_A

    def load_b_fragment(n_part, read_stage):
        sB = fx.make_view(half_b_base(read_stage, n_part), b_lds_layout)
        if const_expr(use_kcontig_atom_s2r):
            frag_B = [0] * (warp_k_steps * warp_n_steps)
            for block_k_iter in range_constexpr(block_k // param.mma_k):
                for ni in range_constexpr(warp_n_steps):
                    row = warp_n_idx + ni * param.mma_n + (w_tid % param.mma_n)
                    col_base = block_k_iter * param.mma_k + (w_tid // param.mma_n) * 16
                    frag_B[block_k_iter * warp_n_steps + ni] = load_fp8_kcontig_plus64_i32x8(
                        half_b_base(read_stage, n_part),
                        b_lds_layout,
                        row,
                        col_base,
                    )
            return frag_B
        frag_B = thr_mma.make_fragment_B(sB)
        frag_B_retile = thr_copy_B.retile(frag_B)
        thr_sB_s2r = thr_copy_B.partition_S(sB)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            fx.copy(
                b_s2r_copy_atom,
                thr_sB_s2r[None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
        return frag_B

    a_mma_rmem = fx.make_rmem_tensor(8, fx.Int32)
    b_mma_rmem = fx.make_rmem_tensor(8, fx.Int32)
    c_mma_rmem = fx.make_rmem_tensor(4, fx.Float32)

    def consume(frag_C, frag_A, frag_B, emit_sched_barrier):
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)
        if const_expr(use_kcontig_atom_s2r):
            frag_C_out = [cx for cx in frag_C]
            for block_k_iter in range_constexpr(block_k // param.mma_k):
                for ni in range_constexpr(warp_n_steps):
                    for mi in range_constexpr(warp_m_steps):
                        c_idx = mi * warp_n_steps + ni
                        a_mma_rmem.store(frag_A[block_k_iter * warp_m_steps + mi])
                        b_mma_rmem.store(frag_B[block_k_iter * warp_n_steps + ni])
                        c_mma_rmem.store(frag_C_out[c_idx])
                        fx.gemm(mma_atom, c_mma_rmem, a_mma_rmem, b_mma_rmem, c_mma_rmem)
                        frag_C_out[c_idx] = c_mma_rmem.load()
            if const_expr(emit_sched_barrier):
                rocdl.sched_barrier(0)
            return frag_C_out
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, block_k_iter],
                frag_B[None, None, block_k_iter],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)
        return frag_C

    def store_half_tile_to_lds(m_part, n_part, frag_C):
        if const_expr(use_kcontig_atom_s2r):
            sC = fx.make_view(smem_c, c_lds_layout_full)
            for mi in range_constexpr(warp_m_steps):
                for ni in range_constexpr(warp_n_steps):
                    c_vec = frag_C[mi * warp_n_steps + ni]
                    col = n_part * half_block_n + warp_n_idx + ni * param.mma_n + stmatrix_c_n_idx
                    global_col = block_n_offset + col
                    safe_n = (global_col < n).select(global_col, 0)
                    scale_b_val = scale_b_buf[safe_n]
                    if const_expr(param.has_bias and not is_split_k):
                        bias_val = bias_buf[safe_n].to(fx.Float32)
                    for kk in range_constexpr(4):
                        row = (
                            m_part * half_block_m
                            + warp_m_idx
                            + mi * param.mma_m
                            + stmatrix_c_m_vec_idx
                            + kk
                        )
                        global_row = block_m_offset + row
                        safe_m = (global_row < m).select(global_row, 0)
                        val = c_vec[kk] * scale_a_buf[safe_m] * scale_b_val
                        if const_expr(param.has_bias and not is_split_k):
                            val = val + bias_val
                        sC[row, col] = val.to(shuffle_dtype)
            return
        sC = fx.make_view(half_c_base(m_part, n_part), c_lds_layout)
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            global_row = block_m_offset + m_part * half_block_m + row
            global_col = block_n_offset + n_part * half_block_n + col
            safe_m = (global_row < m).select(global_row, 0)
            safe_n = (global_col < n).select(global_col, 0)
            val = frag_C[i] * scale_a_buf[safe_m] * scale_b_buf[safe_n]
            if const_expr(param.has_bias and not is_split_k):
                val = val + bias_buf[safe_n].to(fx.Float32)
            sC[row, col] = val.to(shuffle_dtype)

    def store_half_tile_to_global(m_part, n_part):
        if const_expr(use_kcontig_atom_s2r):
            for mi in range_constexpr(warp_m_steps):
                for i in range_constexpr(stg_iters_per_m_step):
                    global_tid = block_threads * i + tid
                    m_band_idx = global_tid // stg_c_quad_x_threads
                    n_local_idx = global_tid % stg_c_quad_x_threads * cshuffle_r2g_vec_size
                    warp_m_band = m_band_idx // param.mma_m
                    atom_m_idx = m_band_idx % param.mma_m
                    m_tile_idx = (
                        m_part * half_block_m
                        + warp_m_band * warp_m
                        + mi * param.mma_m
                        + atom_m_idx
                    )
                    n_tile_idx = n_part * half_block_n + n_local_idx
                    global_row = block_m_offset + m_tile_idx
                    global_col = block_n_offset + n_tile_idx
                    if (global_row < m) and (global_col < n):
                        c_vec = fx.ptr_load(
                            smem_c + m_tile_idx * block_n + n_tile_idx,
                            result_type=fx.Vector.make_type(cshuffle_r2g_vec_size, shuffle_dtype),
                        )
                        write_cshuffle_vec_to_global(
                            out,
                            out_buf,
                            global_row * n + global_col,
                            c_vec,
                            is_split_k,
                            param.out_dtype_id == GEMM_FP8_PTPC_DTYPE_FP32,
                        )
            return
        sC_base = half_c_base(m_part, n_part)
        cshuffle_r2g_x_threads = half_block_n // cshuffle_r2g_vec_size
        cshuffle_vectors = half_block_m * half_block_n // cshuffle_r2g_vec_size
        cshuffle_iters = (cshuffle_vectors + block_threads - 1) // block_threads
        for i in range_constexpr(cshuffle_iters):
            vector_idx = block_threads * i + tid
            if vector_idx < cshuffle_vectors:
                local_row = vector_idx // cshuffle_r2g_x_threads
                local_col = vector_idx % cshuffle_r2g_x_threads * cshuffle_r2g_vec_size
                global_row = block_m_offset + m_part * half_block_m + local_row
                global_col = block_n_offset + n_part * half_block_n + local_col
                if (global_row < m) and (global_col < n):
                    c_vec = fx.ptr_load(
                        sC_base + local_row * half_block_n + local_col,
                        result_type=fx.Vector.make_type(cshuffle_r2g_vec_size, shuffle_dtype),
                    )
                    global_offset = global_row * n + global_col
                    write_cshuffle_vec_to_global(
                        out,
                        out_buf,
                        global_offset,
                        c_vec,
                        is_split_k,
                        param.out_dtype_id == GEMM_FP8_PTPC_DTYPE_FP32,
                    )

    acc_init = fx.Vector.filled(4, 0.0, fx.Float32)
    c00 = [acc_init] * c_frags_len
    c01 = [acc_init] * c_frags_len
    c10 = [acc_init] * c_frags_len
    c11 = [acc_init] * c_frags_len
    if const_expr(not use_kcontig_atom_s2r):
        c00 = make_c_fragment(0, 0)
        c01 = make_c_fragment(0, 1)
        c10 = make_c_fragment(1, 0)
        c11 = make_c_fragment(1, 1)
        c00.fill(0.0)
        c01.fill(0.0)
        c10.fill(0.0)
        c11.fill(0.0)

    if const_expr(is_split_k):
        splitk_protocol.zero_c()

    async_load_b_to_lds(0, 0, 0)
    async_load_a_to_lds(0, 0, 0)
    async_load_b_to_lds(1, 0, 0)
    async_load_a_to_lds(1, 0, 0)
    rocdl.sched_barrier(0)
    if wid // n_waves == 1:
        rocdl.s_barrier()
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    async_load_b_to_lds(0, 1, 1)
    async_load_a_to_lds(0, 1, 1)
    async_load_b_to_lds(1, 1, 1)
    rocdl.sched_barrier(0)
    wait_vmcnt_and_barrier(half_ldg_b_iters + half_ldg_a_iters)

    main_loop_end = k_tiles - 2
    for k_tile in range(0, main_loop_end, 2):
        next_k_tile = k_tile + 2
        # 0
        b0 = load_b_fragment(0, 0)
        a0 = load_a_fragment(0, 0)
        async_load_a_to_lds(1, k_tile + 1, 1)
        rocdl.s_barrier()
        c00 = consume(c00, a0, b0, True)
        rocdl.s_barrier()
        b1 = load_b_fragment(1, 0)
        async_load_b_to_lds(0, next_k_tile, 0)
        rocdl.s_barrier()
        c01 = consume(c01, a0, b1, True)
        rocdl.s_barrier()
        a1 = load_a_fragment(1, 0)
        async_load_a_to_lds(0, next_k_tile, 0)
        rocdl.s_barrier()
        c10 = consume(c10, a1, b0, True)
        rocdl.s_barrier()
        b0 = load_b_fragment(0, 1)
        async_load_b_to_lds(1, next_k_tile, 0)
        wait_vmcnt_and_barrier(2 * half_ldg_b_iters + half_ldg_a_iters)
        c11 = consume(c11, a1, b1, True)
        rocdl.s_barrier()
        # 1
        a0 = load_a_fragment(0, 1)
        async_load_a_to_lds(1, next_k_tile, 0)
        rocdl.s_barrier()
        c00 = consume(c00, a0, b0, True)
        rocdl.s_barrier()
        b1 = load_b_fragment(1, 1)
        async_load_b_to_lds(0, next_k_tile + 1, 1)
        rocdl.s_barrier()
        c01 = consume(c01, a0, b1, True)
        rocdl.s_barrier()
        a1 = load_a_fragment(1, 1)
        async_load_a_to_lds(0, next_k_tile + 1, 1)
        rocdl.s_barrier()
        c10 = consume(c10, a1, b0, True)
        rocdl.s_barrier()
        async_load_b_to_lds(1, next_k_tile + 1, 1)
        wait_vmcnt_and_barrier(half_ldg_b_iters + half_ldg_a_iters)
        c11 = consume(c11, a1, b1, True)
        rocdl.s_barrier()

    k_tile = main_loop_end
    # 0
    if const_expr(is_split_k):
        wait_vmcnt_and_barrier(0)
    b0 = load_b_fragment(0, 0)
    a0 = load_a_fragment(0, 0)
    async_load_a_to_lds(1, k_tile + 1, 1)
    rocdl.s_barrier()
    c00 = consume(c00, a0, b0, True)
    rocdl.s_barrier()
    b1 = load_b_fragment(1, 0)
    rocdl.s_barrier()
    c01 = consume(c01, a0, b1, True)
    rocdl.s_barrier()
    a1 = load_a_fragment(1, 0)
    rocdl.s_barrier()
    c10 = consume(c10, a1, b0, True)
    rocdl.s_barrier()
    b0 = load_b_fragment(0, 1)
    rocdl.s_barrier()
    c11 = consume(c11, a1, b1, True)
    wait_vmcnt_and_barrier(0)
    # 1
    a0 = load_a_fragment(0, 1)
    rocdl.s_barrier()
    c00 = consume(c00, a0, b0, True)
    rocdl.s_barrier()
    b1 = load_b_fragment(1, 1)
    rocdl.s_barrier()
    c01 = consume(c01, a0, b1, True)
    rocdl.s_barrier()
    a1 = load_a_fragment(1, 1)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    store_half_tile_to_lds(0, 0, c00)
    c10 = consume(c10, a1, b0, False)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    store_half_tile_to_lds(0, 1, c01)
    c11 = consume(c11, a1, b1, False)
    rocdl.sched_barrier(0)
    if const_expr(is_split_k):
        if wid // n_waves == 0:
            rocdl.s_barrier()
        rocdl.s_barrier()
        store_half_tile_to_lds(1, 0, c10)
        store_half_tile_to_lds(1, 1, c11)
        splitk_protocol.wait_until_initialized()
        store_half_tile_to_global(0, 0)
        store_half_tile_to_global(0, 1)
        store_half_tile_to_global(1, 0)
        store_half_tile_to_global(1, 1)
        splitk_protocol.finish_split(split_k)
    else:
        wait_vmcnt_and_barrier(0)
        store_half_tile_to_global(0, 0)
        store_half_tile_to_global(0, 1)
        wait_vmcnt_and_barrier(0)
        store_half_tile_to_lds(1, 0, c10)
        store_half_tile_to_lds(1, 1, c11)
        wait_vmcnt_and_barrier(0)
        store_half_tile_to_global(1, 0)
        store_half_tile_to_global(1, 1)


@flyc.jit
def gemm_fp8_ptpc_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    scale_a: fx.Tensor,
    scale_b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    split_k: fx.Int32,
    param: GemmFp8PtpcGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    n = fx.Int32(fx.get_scalar(b.shape[1]))
    k = fx.Int32(fx.get_scalar(a.shape[1]))
    a_leading_stride = fx.Int32(fx.get_scalar(a.stride[1] if const_expr(param.a_is_transposed) else a.stride[0]))
    b_leading_stride = fx.Int32(fx.get_scalar(b.stride[1] if const_expr(param.b_is_transposed) else b.stride[0]))
    split_alignment = GFX950_DMA_BYTES // param.in_data_bytes
    working_k = (k + split_k - 1) // split_k
    working_k = (working_k + split_alignment - 1) // split_alignment * split_alignment
    num_pid_m = (m + param.block_m - 1) // param.block_m
    num_pid_n = (n + param.block_n - 1) // param.block_n
    gemm_fp8_ptpc_kernel_impl = (
        gemm_fp8_ptpc_hti_gfx950_kernel if param.use_half_tile_interleaved else gemm_fp8_ptpc_gfx950_kernel
    )
    gemm_fp8_ptpc_kernel_impl._known_block_size = [param.block_threads, 1, 1]
    gemm_fp8_ptpc_kernel_impl._func.__name__ = make_gemm_fp8_ptpc_gfx950_kernel_name(param)
    gemm_fp8_ptpc_kernel_impl(
        out,
        a,
        b,
        scale_a,
        scale_b,
        bias,
        semaphore,
        signal,
        m,
        n,
        k,
        split_k,
        working_k,
        a_leading_stride,
        b_leading_stride,
        param,
    ).launch(
        grid=(num_pid_m * num_pid_n, split_k, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


def make_gemm_fp8_ptpc_param_and_validate(m, n, k, kwargs):
    result = None
    try:
        result = make_gemm_fp8_ptpc_gfx950_param(**kwargs)
    except Exception:
        return None
    split_k = kwargs.get("split_k", 1)
    try:
        assert_no_k_tail(k, kwargs)
    except AssertionError:
        return None
    cshuffle_r2g_vec_size = result.cshuffle_r2g_vec_size
    if n % cshuffle_r2g_vec_size != 0:
        return None
    async_load_vec_size = GFX950_DMA_BYTES // result.in_data_bytes
    if result.a_is_transposed and m % async_load_vec_size != 0:
        return None
    if not result.b_is_transposed and n % async_load_vec_size != 0:
        return None
    if result.b_is_transposed and k % async_load_vec_size != 0:
        return None
    num_pid_m = (m + result.block_m - 1) // result.block_m
    num_pid_n = (n + result.block_n - 1) // result.block_n
    if split_k > 1:
        c_elements_per_iteration = result.block_threads * cshuffle_r2g_vec_size
        if (
            num_pid_m * num_pid_n > SPLIT_K_SEMAPHORE_MAX_LEN
            or result.block_m * result.block_n % c_elements_per_iteration != 0
        ):
            return None
    return result


def assert_no_k_tail(k: int, kwargs: dict):
    split_k = kwargs["split_k"]
    block_k = kwargs["block_k"]
    stages = kwargs["stages"]
    use_half_tile_interleaved = kwargs["use_half_tile_interleaved"]
    async_load_vec_size = GFX950_DMA_BYTES
    working_k = (k + split_k - 1) // split_k
    working_k = (working_k + async_load_vec_size - 1) // async_load_vec_size * async_load_vec_size
    last_working_k = k - (split_k - 1) * working_k
    assert (
        working_k % block_k == 0
    ), f"K-tail is unsupported: aligned split-K partition size {working_k} is not divisible by block_k={block_k}"
    assert last_working_k > 0 and last_working_k % block_k == 0, (
        "K-tail is unsupported: final split-K partition size "
        f"{last_working_k} must be positive and divisible by block_k={block_k}"
    )
    working_k_tiles = working_k // block_k
    last_working_k_tiles = last_working_k // block_k
    min_k_tiles = stages - 1
    assert (
        working_k_tiles >= min_k_tiles
    ), f"split-K partitions require at least {min_k_tiles} K tiles, got {working_k_tiles}"
    assert (
        last_working_k_tiles >= min_k_tiles
    ), f"the final split-K partition requires at least {min_k_tiles} K tiles, got {last_working_k_tiles}"
    if use_half_tile_interleaved:
        assert (
            working_k_tiles >= 2 and working_k_tiles % 2 == 0
        ), f"HTI requires at least two and an even number of K tiles per split-K partition, got {working_k_tiles}"
        assert last_working_k_tiles >= 2 and last_working_k_tiles % 2 == 0, (
            "HTI requires at least two and an even number of K tiles "
            "in the final split-K partition, "
            f"got {last_working_k_tiles}"
        )


def get_default_fp8_ptpc_kwargs(m, n, k):
    kwargs = {
        "block_m": 256,
        "block_n": 256,
        "block_k": 128,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 4,
        "k_waves": 1,
        "group_m": 0,
        "use_half_tile_interleaved": True,
    }
    if m == 2048 and n == 2048 and k == 2048:
        kwargs.update(
            {
                "block_m": 128,
                "block_n": 128,
                "block_k": 128,
                "stages": 4,
                "split_k": 1,
                "m_waves": 4,
                "n_waves": 4,
                "k_waves": 1,
                "use_half_tile_interleaved": False,
            }
        )
    elif m <= 32 and n == 384 and k == 7168:
        kwargs.update(
            {
                "block_m": 32,
                "block_n": 64,
                "block_k": 128,
                "stages": 5,
                "split_k": 8,
                "m_waves": 2,
                "n_waves": 2,
                "k_waves": 1,
                "use_half_tile_interleaved": False,
            }
        )
    return kwargs


def gemm_fp8_ptpc(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    user_kwargs: dict = {},
    stream: Optional[torch.cuda.Stream] = None,
    layout: str = "nt",
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute C[M, N] = (A[M, K] @ B[K, N]) * scale_a[M] * scale_b[N].

    A and B are ``float8_e4m3fn``. MMA uses gfx950 ``MFMA_Scale`` 16x16x128
    with hardware scale left at identity. Per-token f32 scales are applied
    in the epilogue as ``C = (A @ B) * scale_a[:, None] * scale_b[None, :]``.

    Each layout character controls only the corresponding tensor stride:
    N is row-major and T is column-major. Logical tensor shapes never change.
    Inputs that violate the selected layout or DMA alignment are rejected.
    Set ``user_kwargs["split_k"]`` above 1 to atomically reduce K partitions.
    Set ``user_kwargs["k_waves"]`` above 1 for full-tile workgroup-local slice-K.
    ``out_dtype`` may be ``torch.bfloat16`` (default) or ``torch.float32``.
    Bias, when present, is ``bfloat16`` and is added after scaling.
    """
    if stream is None:
        stream = torch.cuda.current_stream()
    layout = layout.lower()
    if layout not in ("nn", "nt", "tn", "tt"):
        raise ValueError(f"unsupported GEMM layout: {layout!r}; expected 'nn', 'nt', 'tn', or 'tt'")
    a_is_transposed = layout[0] == "t"
    b_is_transposed = layout[1] == "t"
    device = a.device
    assert a.device == b.device
    assert a.ndim == 2 and b.ndim == 2
    m, k = a.shape
    assert b.shape[0] == k
    n = b.shape[1]
    assert a.dtype == b.dtype
    assert a.dtype == torch.float8_e4m3fn
    if a_is_transposed:
        a_vec_size = GFX950_DMA_BYTES // a.element_size()
        if (
            a.stride(0) != 1
            or a.data_ptr() % GFX950_DMA_BYTES != 0
            or a.stride(1) * a.element_size() % GFX950_DMA_BYTES != 0
            or m % a_vec_size != 0
        ):
            raise ValueError(
                "A does not satisfy the GFX950 DMA requirements for a "
                "column-major input: expected stride(0) == 1, a "
                f"{GFX950_DMA_BYTES}-byte-aligned data pointer and leading "
                f"stride, and M divisible by {a_vec_size}; got "
                f"shape={tuple(a.shape)} and stride={a.stride()}"
            )
    else:
        if (
            a.stride(1) != 1
            or a.data_ptr() % GFX950_DMA_BYTES != 0
            or a.stride(0) * a.element_size() % GFX950_DMA_BYTES != 0
        ):
            raise ValueError(
                "A does not satisfy the GFX950 DMA requirements for a "
                "row-major input: expected stride(1) == 1 and a "
                f"{GFX950_DMA_BYTES}-byte-aligned data pointer and leading "
                f"stride; got shape={tuple(a.shape)} and stride={a.stride()}"
            )
    if b_is_transposed:
        if (
            b.stride(0) != 1
            or b.data_ptr() % GFX950_DMA_BYTES != 0
            or b.stride(1) * b.element_size() % GFX950_DMA_BYTES != 0
        ):
            raise ValueError(
                "B does not satisfy the GFX950 DMA requirements for a "
                "column-major input: expected stride(0) == 1 and a "
                f"{GFX950_DMA_BYTES}-byte-aligned data pointer and leading "
                f"stride; got shape={tuple(b.shape)} and stride={b.stride()}"
            )
    else:
        if (
            b.stride(1) != 1
            or b.data_ptr() % GFX950_DMA_BYTES != 0
            or b.stride(0) * b.element_size() % GFX950_DMA_BYTES != 0
        ):
            raise ValueError(
                "B does not satisfy the GFX950 DMA requirements for a "
                "row-major input: expected stride(1) == 1 and a "
                f"{GFX950_DMA_BYTES}-byte-aligned data pointer and leading "
                f"stride; got shape={tuple(b.shape)} and stride={b.stride()}"
            )
    assert scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32
    assert scale_a.device == device and scale_b.device == device
    assert scale_a.ndim == 1 and scale_a.shape[0] == m
    assert scale_b.ndim == 1 and scale_b.shape[0] == n
    if not scale_a.is_contiguous():
        scale_a = scale_a.contiguous()
    if not scale_b.is_contiguous():
        scale_b = scale_b.contiguous()
    if out_dtype is None:
        out_dtype = torch.bfloat16 if out is None else out.dtype
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported output dtype {out_dtype}; expected torch.bfloat16 or torch.float32")
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    else:
        assert out.dtype == out_dtype
        assert out.device == device
        assert out.is_contiguous()
    out = out.view(-1, n)
    assert out.shape[0] == m
    assert out.dtype == out_dtype

    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    kwargs = get_default_fp8_ptpc_kwargs(m, n, k)
    kwargs.update(user_kwargs)
    kwargs["a_is_transposed"] = a_is_transposed
    kwargs["b_is_transposed"] = b_is_transposed
    kwargs["in_dtype_id"] = GEMM_FP8_PTPC_DTYPE_FP8
    kwargs["out_dtype_id"] = (
        GEMM_FP8_PTPC_DTYPE_FP32 if out.dtype is torch.float32 else GEMM_FP8_PTPC_DTYPE_BF16
    )
    kwargs["has_bias"] = False if bias is None else True
    split_k = kwargs["split_k"]
    assert_no_k_tail(k, kwargs)

    if bias is not None:
        assert bias.shape[0] == n
        assert bias.dtype == torch.bfloat16

    param = make_gemm_fp8_ptpc_param_and_validate(m, n, k, kwargs)
    assert param is not None, "unsupported gemm_fp8_ptpc_gfx950 shape/config"
    semaphore, signal = get_split_k_buffers(stream, device)
    a_arg = _dynamic_tensor_arg(a, 0 if a_is_transposed else 1)
    b_arg = _dynamic_tensor_arg(b, 0 if b_is_transposed else 1)
    out_arg = _dynamic_tensor_arg(out, 1)
    scale_a_arg = _dynamic_tensor_arg(scale_a, 0)
    scale_b_arg = _dynamic_tensor_arg(scale_b, 0)
    bias_arg = scale_a_arg if bias is None else _dynamic_tensor_arg(bias, 0)
    dispatch_args = (
        out_arg,
        a_arg,
        b_arg,
        scale_a_arg,
        scale_b_arg,
        bias_arg,
        semaphore,
        signal,
        split_k,
        param,
        stream,
    )
    run_cached(
        gemm_fp8_ptpc_gfx950,
        *dispatch_args,
        constexpr_param=param,
        compiler=flyc.compile,
        dispatch_args=dispatch_args,
    )
    return out
