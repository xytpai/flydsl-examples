from dataclasses import dataclass
from typing import Any, Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import fly
from flydsl.expr import const_expr, range_constexpr, rocdl

from .common import run_cached
from .gemm_a16w16_gfx950 import (
    _dynamic_tensor_arg,
    get_split_k_buffers,
    write_cshuffle_vec_to_global,
)
from .gemm_a16w16_gfx950_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    BlockSwizzle,
    get_wave_lds_offset,
    make_fp8_lds_layout,
    swizzled_contiguous_idx,
    wait_vmcnt_and_barrier,
)

GEMM_FP8_PTPC_DTYPE_FP32 = 1
GEMM_FP8_PTPC_DTYPE_BF16 = 2
GEMM_FP8_PTPC_DTYPE_FP8 = 4


# The only supported specialization. Keep the existing Param/launch ABI and
# inline SSA MMA so this remains the pre-refine HTI implementation.
_FP8_PTPC_CONFIG = {
    "in_dtype_id": GEMM_FP8_PTPC_DTYPE_FP8,
    "out_dtype_id": GEMM_FP8_PTPC_DTYPE_BF16,
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
    "use_hti_ssa_mma": True,
    "a_is_transposed": False,
    "b_is_transposed": True,
    "has_bias": False,
    "mma_m": 16,
    "mma_n": 16,
    "mma_k": 128,
}
_FP8_PTPC_KERNEL_NAME = (
    "hgemm_fp8_ptpc_t256x256x128x2_ks1_w2x4x1_gm0_bias0_lnt_phti_ssa"
)


def validate_fp8_ptpc_config(kwargs):
    for key, value in kwargs.items():
        if key not in _FP8_PTPC_CONFIG:
            raise ValueError(f"unsupported FP8 PTPC option: {key!r}")
        expected = _FP8_PTPC_CONFIG[key]
        if value != expected:
            raise ValueError(
                f"only {_FP8_PTPC_KERNEL_NAME} is supported: "
                f"expected {key}={expected!r}, got {value!r}"
            )


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
    use_hti_ssa_mma: fx.Constexpr[bool]
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


@dataclass(slots=True, kw_only=True, eq=False)
class AsyncLoadOperand:
    context: GemmABLoadContext
    src_base: Any
    lds_layout: Any
    outer_tile_size: Any
    outer_bound: Any
    leading_stride: Any
    load_iters: Any


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
    use_hti_ssa_mma: bool = True,
) -> GemmFp8PtpcGfx950Param:
    validate_fp8_ptpc_config(locals())
    return GemmFp8PtpcGfx950Param(
        in_dtype_id=in_dtype_id,
        out_dtype_id=out_dtype_id,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        is_split_k=False,
        m_waves=m_waves,
        n_waves=n_waves,
        k_waves=k_waves,
        group_m=group_m,
        use_half_tile_interleaved=True,
        use_hti_ssa_mma=True,
        a_is_transposed=False,
        b_is_transposed=True,
        has_bias=False,
        async_load_bytes=GFX950_DMA_BYTES,
        in_data_bytes=1,
        out_data_bytes=2,
        cshuffle_r2g_vec_size=8,
        ldg_x_threads=8,
        block_threads=512,
        ldg_a_iters=4,
        ldg_b_iters=4,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
    )


def make_gemm_fp8_ptpc_gfx950_kernel_name(param: GemmFp8PtpcGfx950Param):
    return _FP8_PTPC_KERNEL_NAME


def make_gemm_ab_lds_layouts(rows_a, rows_b, block_k, a_is_transposed, b_is_transposed):
    return (
        make_fp8_lds_layout(rows_a, block_k, a_is_transposed),
        make_fp8_lds_layout(rows_b, block_k, not b_is_transposed),
    )


def load_fp8_kcontig_16b_i32x4(lds_base, lds_layout, row, col):
    offset = row * 128 + (col ^ ((row % 16) // 2 * 16))
    ptr_off = fx.add_offset(lds_base, fx.make_int_tuple(offset))
    i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
    return fx.make_view(i8_iter, fx.make_layout(16, 1)).load().bitcast(fx.Int32)


def load_fp8_kcontig_plus64_i32x8(lds_base, lds_layout, row, col_base):
    lo = load_fp8_kcontig_16b_i32x4(lds_base, lds_layout, row, col_base)
    hi = load_fp8_kcontig_16b_i32x4(lds_base, lds_layout, row, col_base + 64)
    return lo.shuffle(hi, list(range(8)))


def make_gemm_ab_load_context(load_tid, ks_begin, param: GemmFp8PtpcGfx950Param):
    return GemmABLoadContext(
        wave_offset=get_wave_lds_offset(load_tid, param.async_load_bytes),
        tid=load_tid,
        ks_begin=ks_begin,
        param=param,
        async_g2s_copy_atom=fx.make_copy_atom(
            fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128
        ),
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
    # HTI NT uses hand-counted vmcnt waits. Keep a scheduling boundary
    # between half-operand batches, but let the DMA instructions within a
    # batch schedule together with the preceding LDS reads/address work.
    for i in range_constexpr(operand.load_iters):
        global_tid = block_threads * i + tid
        outer_local_idx = global_tid // ldg_x_threads
        k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
        lane_k_idx = swizzled_contiguous_idx(
            outer_local_idx,
            k_local_idx,
            operand.lds_layout,
            block_k,
        )
        global_outer_idx = global_outer_offset + outer_local_idx
        safe_global_outer_idx = (global_outer_idx < operand.outer_bound).select(global_outer_idx, 0)
        lane_offset = safe_global_outer_idx * operand.leading_stride + lane_k_idx
        uniform_k_offset = context.ks_begin + k_tile * block_k
        src = fx.make_view(operand.src_base + lane_offset, g2s_copy_layout)
        dst = fx.make_view(lds_ptr, g2s_copy_layout)
        dma_atom = context.async_g2s_copy_atom.set_value(
            "soffset", fx.Int32(uniform_k_offset)
        )
        fx.copy_atom_call(dma_atom, src, dst)
        if i < operand.load_iters - 1:
            lds_ptr = lds_ptr + block_threads * async_load_vec_size
    rocdl.sched_barrier(0)


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

    ab_load_context = make_gemm_ab_load_context(
        load_tid=tid,
        ks_begin=ks_begin,
        param=param,
    )
    mma_atom = fx.make_mma_atom(
        fx.rocdl.cdna4.MFMA_Scale(param.mma_m, param.mma_n, param.mma_k, fx.Float8E4M3FN)
    )
    # Keep the packed K+0/K+64 operands and inline SSA MMA of the HTI kernel.
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
    )
    b_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(b_buf),
        lds_layout=b_lds_layout,
        outer_tile_size=half_block_n,
        outer_bound=n,
        leading_stride=b_leading_stride,
        load_iters=half_ldg_b_iters,
    )
    c_lds_layout_full = fx.make_layout((block_m, block_n), (block_n, 1))

    def half_a_base(stage, m_part):
        return smem_a + (stage * block_m + m_part * half_block_m) * block_k

    def half_b_base(stage, n_part):
        return smem_b + (stage * block_n + n_part * half_block_n) * block_k

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

    def load_a_fragment(m_part, read_stage):
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

    def load_b_fragment(n_part, read_stage):
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

    def consume(frag_C, frag_A, frag_B, emit_sched_barrier):
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)
        frag_C_out = [cx for cx in frag_C]
        for mma_idx in range_constexpr(warp_m_steps * warp_n_steps * warp_k_steps):
            block_k_iter = mma_idx % warp_k_steps
            c_idx = mma_idx // warp_k_steps
            mi = c_idx // warp_n_steps
            ni = c_idx % warp_n_steps
            frag_C_out[c_idx] = fx.Vector(
                fly.mma_atom_call_ssa(
                    [fx.Vector.make_type(4, fx.Float32)],
                    mma_atom,
                    frag_A[block_k_iter * warp_m_steps + mi],
                    frag_B[block_k_iter * warp_n_steps + ni],
                    frag_C_out[c_idx],
                )
            )
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)
        return frag_C_out

    def load_half_scale_b(n_part):
        # Load once per N fragment; reuse across all M repeats and both
        # M halves. Keep these loads in the epilogue, outside the K loop.
        scale_b_frags = [fx.Float32(0.0)] * warp_n_steps
        for ni in range_constexpr(warp_n_steps):
            col = n_part * half_block_n + warp_n_idx + ni * param.mma_n + stmatrix_c_n_idx
            global_col = block_n_offset + col
            safe_n = (global_col < n).select(global_col, 0)
            scale_b_frags[ni] = scale_b_buf[safe_n]
        return scale_b_frags

    def store_half_tile_to_lds(m_part, n_part, frag_C, scale_b_frags):
        sC = fx.make_view(smem_c, c_lds_layout_full)
        for mi in range_constexpr(warp_m_steps):
            row_base = m_part * half_block_m + warp_m_idx + mi * param.mma_m + stmatrix_c_m_vec_idx
            global_row_base = block_m_offset + row_base
            vals = [fx.Float32(0.0)] * 4
            for kk in range_constexpr(4):
                global_row = global_row_base + kk
                safe_m = (global_row < m).select(global_row, 0)
                vals[kk] = scale_a_buf[safe_m]
            scale_a_vec = fx.Vector.from_elements(vals, fx.Float32)
            for ni in range_constexpr(warp_n_steps):
                c_vec = frag_C[mi * warp_n_steps + ni]
                col = n_part * half_block_n + warp_n_idx + ni * param.mma_n + stmatrix_c_n_idx
                scale_b_val = scale_b_frags[ni]
                scaled = (c_vec * scale_a_vec * fx.Vector.filled(4, scale_b_val, fx.Float32)).to(shuffle_dtype)
                for kk in range_constexpr(4):
                    sC[row_base + kk, col] = scaled[kk]
        return

    def store_half_tile_to_global(m_part, n_part):
        for mi in range_constexpr(warp_m_steps):
            for i in range_constexpr(stg_iters_per_m_step):
                threads_per_m_group = n_waves * GFX950_WAVE_SIZE
                m_group = tid // threads_per_m_group
                tid_in_m_group = tid % threads_per_m_group
                global_tid = (
                    (m_group * stg_iters_per_m_step + i) * threads_per_m_group
                    + tid_in_m_group
                )
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

    acc_init = fx.Vector.filled(4, 0.0, fx.Float32)
    c00 = [acc_init] * c_frags_len
    c01 = [acc_init] * c_frags_len
    c10 = [acc_init] * c_frags_len
    c11 = [acc_init] * c_frags_len

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
    main_bulk_end = main_loop_end // 4 * 4
    for k_group in range(0, main_bulk_end, 4):
        for pair in range_constexpr(2):
            k_tile = k_group + pair * 2
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
    for k_tile in range(main_bulk_end, main_loop_end, 2):
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
    scale_b0 = load_half_scale_b(0)
    store_half_tile_to_lds(0, 0, c00, scale_b0)
    scale_b1 = load_half_scale_b(1)
    store_half_tile_to_lds(0, 1, c01, scale_b1)
    c10 = consume(c10, a1, b0, False)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    store_half_tile_to_global(0, 0)
    store_half_tile_to_global(0, 1)
    store_half_tile_to_lds(1, 0, c10, scale_b0)
    c11 = consume(c11, a1, b1, False)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    store_half_tile_to_global(1, 0)
    store_half_tile_to_lds(1, 1, c11, scale_b1)
    rocdl.s_barrier()
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
    gemm_fp8_ptpc_kernel_impl = gemm_fp8_ptpc_hti_gfx950_kernel
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
    validate_fp8_ptpc_config(kwargs)
    if m <= 0 or n <= 0 or n % 8 != 0:
        raise ValueError(f"HTI requires M > 0 and N > 0 divisible by 8, got M={m}, N={n}")
    assert_no_k_tail(k, kwargs)
    return make_gemm_fp8_ptpc_gfx950_param(**kwargs)


def assert_no_k_tail(k: int, kwargs: dict):
    assert k >= 256 and k % 256 == 0, (
        "HTI requires at least two and an even number of K tiles: "
        f"K must be a positive multiple of 256, got {k}"
    )


def get_default_fp8_ptpc_kwargs(m, n, k):
    return dict(_FP8_PTPC_CONFIG)


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
    """Compute scaled FP8 GEMM using the fixed SSA HTI specialization.

    A[M, K] is row-major FP8 E4M3FN; B[K, N] is column-major FP8 E4M3FN.
    Output is BF16, with FP32 per-row/per-column scales and no bias.
    Only tile=256x256x128, stages=2, waves=2x4, group_m=0, split_k=1
    is supported. M/N may have partial tiles; N must be divisible by 8,
    and K must be a positive multiple of 256. DMA pointers and leading
    strides must be 16-byte aligned. Other configurations are rejected.
    """
    layout = layout.lower()
    if layout != "nt":
        raise ValueError(f"only layout='nt' is supported, got {layout!r}")
    if bias is not None:
        raise ValueError("the fixed FP8 PTPC HTI kernel does not support bias")
    validate_fp8_ptpc_config(user_kwargs)
    if stream is None:
        stream = torch.cuda.current_stream()
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
    if out_dtype != torch.bfloat16:
        raise ValueError(f"only torch.bfloat16 output is supported, got {out_dtype}")
    kwargs = get_default_fp8_ptpc_kwargs(m, n, k)
    kwargs.update(user_kwargs)
    kwargs["a_is_transposed"] = a_is_transposed
    kwargs["b_is_transposed"] = b_is_transposed
    kwargs["in_dtype_id"] = GEMM_FP8_PTPC_DTYPE_FP8
    kwargs["out_dtype_id"] = GEMM_FP8_PTPC_DTYPE_BF16
    kwargs["has_bias"] = False
    split_k = kwargs["split_k"]
    assert_no_k_tail(k, kwargs)

    param = make_gemm_fp8_ptpc_param_and_validate(m, n, k, kwargs)
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    else:
        assert out.dtype == out_dtype
        assert out.device == device
        assert out.is_contiguous()
    out = out.view(-1, n)
    assert out.shape[0] == m
    assert out.dtype == out_dtype

    # Preserve the measured kernel's launch ABI; these buffers are unused
    # by the only remaining (non-split) specialization.
    semaphore, signal = get_split_k_buffers(stream, device)
    a_arg = _dynamic_tensor_arg(a, 0 if a_is_transposed else 1)
    b_arg = _dynamic_tensor_arg(b, 0 if b_is_transposed else 1)
    out_arg = _dynamic_tensor_arg(out, 1)
    scale_a_arg = _dynamic_tensor_arg(scale_a, 0)
    scale_b_arg = _dynamic_tensor_arg(scale_b, 0)
    bias_arg = scale_a_arg
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
