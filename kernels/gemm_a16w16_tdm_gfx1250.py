# SPDX-License-Identifier: MIT
"""Wave32 TDM GEMM for gfx1250, including FP8 per-token per-channel.

Optional NT quadrant interleaving uses opposite serpentine orders on even/odd
waves. Both schedules share one kernel, TDM operands and the output epilogue.
"""

import functools
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl, tdm_ops
from flydsl.expr.rocdl.cluster import cluster_barrier
from flydsl.expr.typing import T

from .common import run_cached
from .gemm_a16w16_gfx950_utils import BlockSwizzle


WAVE_SIZE = 32
LDS_PAD = 8
# MI455X SPX: 8 XCDs, 256 WGPs. Same pid threshold as the gfx950 kernel.
NUM_XCDS = 8
NUM_PIDS_THRESHOLD = 256


GEMM_DTYPE_FP32 = 1
GEMM_DTYPE_BF16 = 2
GEMM_DTYPE_FP16 = 3
GEMM_DTYPE_MXFP8 = 4
GEMM_DTYPE_MXFP4 = 5
GEMM_DTYPE_FP8_PTPC = 6
GEMM_DTYPE_NAMES = {
    GEMM_DTYPE_FP32: "fp32",
    GEMM_DTYPE_BF16: "bf16",
    GEMM_DTYPE_FP16: "fp16",
    GEMM_DTYPE_MXFP8: "mxfp8",
    GEMM_DTYPE_MXFP4: "mxfp4",
    GEMM_DTYPE_FP8_PTPC: "fp8_ptpc",
}
SUPPORTED_MMA = {
    GEMM_DTYPE_BF16: ((16, 16, 32),),
    GEMM_DTYPE_FP16: ((16, 16, 32),),
    GEMM_DTYPE_FP8_PTPC: ((16, 16, 128),),
}
GEMM_DTYPE_BITS = {
    GEMM_DTYPE_FP32: 32,
    GEMM_DTYPE_BF16: 16,
    GEMM_DTYPE_FP16: 16,
    GEMM_DTYPE_MXFP8: 8,
    GEMM_DTYPE_MXFP4: 4,
    GEMM_DTYPE_FP8_PTPC: 8,
}
FRAGMENT_LOAD_BITS = 16 * 8
STORE_VEC_BITS = 16 * 8


def _validate_interleaved_tile(
    block_m, block_n, m_waves, n_waves, mma_m, mma_n, mma_k, elem_bits,
    b_k_contiguous, interleave,
):
    """Each quadrant must hide the independent loads of its next half-fragment."""
    if type(interleave) is not bool:
        raise ValueError("interleave must be bool")
    if not interleave:
        return
    m_steps = block_m // (m_waves * mma_m)
    n_steps = block_n // (n_waves * mma_n)
    half_m_steps, half_n_steps = m_steps // 2, n_steps // 2
    fragment_loads = (mma_m * mma_k // WAVE_SIZE) // (FRAGMENT_LOAD_BITS // elem_bits)
    if (
        not b_k_contiguous or m_steps % 2 or n_steps % 2
        or half_m_steps * half_n_steps < max(half_m_steps, half_n_steps) * fragment_loads
    ):
        raise ValueError("interleave requires NT and even quadrants with room for LDS loads")


def validate_gemm_gfx1250_config(
    m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m,
    dtype_id, mma_m, mma_n, mma_k, b_k_contiguous=True, split_k=1, interleave=False,
):
    values = (m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, mma_m, mma_n, mma_k)
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError(
            "dimensions, tiles, stages, wave counts and WMMA sizes must be positive integers"
        )
    supported_mma = SUPPORTED_MMA.get(dtype_id)
    mma = (mma_m, mma_n, mma_k)
    if supported_mma is None or mma not in supported_mma:
        dtype_name = GEMM_DTYPE_NAMES.get(dtype_id, "unknown")
        raise ValueError(f"unsupported WMMA {mma_m}x{mma_n}x{mma_k} for {dtype_name}")
    if type(group_m) is not int or group_m < 0:
        raise ValueError("group_m must be a nonnegative integer")
    if not 2 <= stages <= 33:
        raise ValueError("stages must be in [2, 33] (two TDM loads per stage)")
    if block_k not in (32, 64, 128, 256, 512):
        raise ValueError("block_k must be a power of two in [32, 512]")
    waves = m_waves * n_waves
    if waves > 32:
        raise ValueError("a workgroup cannot exceed 1024 threads")
    if block_m % (m_waves * mma_m) or block_n % (n_waves * mma_n) or block_k % mma_k:
        raise ValueError("tile dimensions must be divisible by the WMMA atom")
    if not (
        1 <= block_m // (m_waves * mma_m) <= 8 and 1 <= block_n // (n_waves * mma_n) <= 8
    ):
        raise ValueError("each wave must compute 1..8 WMMA tiles along M and N")
    if block_m % waves or block_n % waves:
        raise ValueError("each wave must own a whole number of A and B rows")
    elem_bits = GEMM_DTYPE_BITS.get(dtype_id)
    if elem_bits is None or STORE_VEC_BITS % elem_bits:
        dtype_name = GEMM_DTYPE_NAMES.get(dtype_id, "unknown")
        raise ValueError(f"unsupported element size for {dtype_name}")
    _validate_interleaved_tile(
        block_m, block_n, m_waves, n_waves, mma_m, mma_n, mma_k, elem_bits,
        b_k_contiguous, interleave,
    )
    # FP8 PTPC writes BF16. The 16-byte store width follows the output type.
    out_bits = 16 if dtype_id == GEMM_DTYPE_FP8_PTPC else elem_bits
    if STORE_VEC_BITS % out_bits:
        raise ValueError("the output tile must support whole-workgroup 16-byte stores")
    out_vec_elems = STORE_VEC_BITS // out_bits
    threads = waves * WAVE_SIZE
    if (
        block_n % out_vec_elems
        or threads % (block_n // out_vec_elems)
        or block_m * block_n % (threads * out_vec_elems)
    ):
        raise ValueError("the output tile must support whole-workgroup 16-byte stores")
    lds_pad = LDS_PAD * 16 // elem_bits
    a_stage_size = block_m * (block_k + lds_pad)
    b_stage_size = block_n * (block_k + lds_pad) if b_k_contiguous else block_k * (block_n + lds_pad)
    ab_bits = stages * (a_stage_size + b_stage_size) * elem_bits
    # The interleaved schedule writes row-contiguous fragments into padded LDS.
    c_pitch = block_n + (STORE_VEC_BITS // out_bits if interleave else 0)
    c_bits = block_m * c_pitch * out_bits
    shared_bytes = (max(ab_bits, c_bits) + 7) // 8
    if shared_bytes > 320 * 1024:
        raise ValueError("pipeline exceeds gfx1250's 320 KiB LDS capacity")
    if split_k not in range(1, 10):
        raise ValueError("cluster split_k must be in 1..9")
    if k % (split_k * block_k):
        raise ValueError("K must be divisible by split_k * block_k")
    if k // split_k < (stages - 1) * block_k:
        raise ValueError("prefetch needs at least stages - 1 tiles in each K slice")
    n_blocks = (n + block_n - 1) // block_n
    if group_m * n_blocks > (2**32 - 1) // 2:
        raise ValueError("block swizzle offsets exceed the supported 32-bit range")


def make_gemm_a16w16_gfx1250_kernel_name(
    dtype_id, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias, b_k_contiguous,
    mma_m, mma_n, mma_k, split_k, interleave=False,
):
    dtype_name = GEMM_DTYPE_NAMES[dtype_id]
    layout = "nt" if b_k_contiguous else "nn"
    return (
        f"hgemm_{dtype_name}_t{block_m}x{block_n}x{block_k}x{stages}"
        f"_w{m_waves}x{n_waves}_gm{group_m}_bias{int(has_bias)}_l{layout}"
        f"_sk{split_k}_mma{mma_m}x{mma_n}x{mma_k}"
        f"{'_interleave' if interleave else ''}"
    )


@fx.struct
class GemmA16W16Gfx1250Param:
    dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    has_bias: fx.Constexpr[bool]
    b_k_contiguous: fx.Constexpr[bool]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    split_k: fx.Constexpr[int]
    interleave: fx.Constexpr[bool]


@functools.lru_cache(maxsize=128)
def build_gemm_gfx1250(
    dtype_id: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
    b_k_contiguous: bool,
    mma_m: int,
    mma_n: int,
    mma_k: int,
    split_k: int = 1,
    interleave: bool = False,
):
    is_fp8_ptpc = dtype_id == GEMM_DTYPE_FP8_PTPC
    if is_fp8_ptpc:
        elem_dtype = fx.Float8E4M3FN
        out_dtype = fx.BFloat16
    else:
        elem_dtype = {GEMM_DTYPE_FP16: fx.Float16, GEMM_DTYPE_BF16: fx.BFloat16}.get(dtype_id)
        out_dtype = elem_dtype
    if elem_dtype is None:
        name = GEMM_DTYPE_NAMES.get(dtype_id, "unknown")
        raise ValueError(f"unsupported dtype_id={dtype_id} ({name})")
    elem_bits = GEMM_DTYPE_BITS[dtype_id]
    lds_pad = LDS_PAD * 16 // elem_bits
    out_bits = 16 if is_fp8_ptpc else elem_bits
    out_vec_elems = STORE_VEC_BITS // out_bits
    frag_elems = mma_m * mma_k // WAVE_SIZE
    frag_load_elems = FRAGMENT_LOAD_BITS // elem_bits
    frag_loads = frag_elems // frag_load_elems
    frag_k_stride = 2 * frag_load_elems
    acc_elems = mma_m * mma_n // WAVE_SIZE
    waves = m_waves * n_waves
    threads = waves * WAVE_SIZE
    # Views are A[M, K] and B[K, N]. Pad whichever dimension is contiguous.
    a_pitch = block_k + lds_pad
    a_stage_size = block_m * a_pitch
    a_tile_stride = (a_pitch, 1)
    if b_k_contiguous:
        b_pitch = block_k + lds_pad
        b_stage_size = block_n * b_pitch
        b_tile_stride = (1, b_pitch)
    else:
        b_pitch = block_n + lds_pad
        b_stage_size = block_k * b_pitch
        b_tile_stride = (b_pitch, 1)
    # FP8 PTPC keeps A/B in FP8 and the output tile in BF16, so the arena is
    # counted in input elements and the C tile occupies out_bits/elem_bits slots.
    c_pitch = block_n + (STORE_VEC_BITS // out_bits if interleave else 0)
    c_slots = block_m * c_pitch * out_bits // elem_bits
    shared_elements = max(stages * (a_stage_size + b_stage_size), c_slots)
    m_steps, n_steps = block_m // (m_waves * mma_m), block_n // (n_waves * mma_n)
    half_m_steps, half_n_steps = m_steps // 2, n_steps // 2
    _validate_interleaved_tile(
        block_m, block_n, m_waves, n_waves, mma_m, mma_n, mma_k, elem_bits,
        b_k_contiguous, interleave,
    )

    @fx.struct
    class SharedStorage:
        data: fx.Array[elem_dtype, shared_elements, 16]

    @flyc.kernel
    def kernel(
        a: fx.Pointer,
        b: fx.Pointer,
        out: fx.Pointer,
        bias: fx.Pointer,
        workspace: fx.Pointer,
        scale_a: fx.Pointer,
        scale_b: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        a_leading_stride: fx.Int32,
        b_leading_stride: fx.Int32,
        c_leading_stride: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()
        tid = fx.thread_idx.x
        wave = rocdl.wave_id()
        lane = tid % WAVE_SIZE
        block_id = fx.block_idx.x
        m_blocks = (m + block_m - 1) // block_m
        n_blocks = (n + block_n - 1) // block_n
        block_swizzle = BlockSwizzle(
            NUM_XCDS=NUM_XCDS,
            NUM_PIDS_THRESHOLD=NUM_PIDS_THRESHOLD,
            GROUP_M=group_m,
        )
        block_row, block_col = block_swizzle.swizzle(m_blocks, n_blocks, block_id)
        row_start, col_start = block_row * block_m, block_col * block_n
        wave_row = wave // n_waves * (block_m // m_waves)
        wave_col = wave % n_waves * (block_n // n_waves)

        storage = fx.SharedAllocator(static=True).allocate(SharedStorage).peek()
        shared_a = storage.data.ptr
        shared_b = shared_a + stages * a_stage_size
        if const_expr(is_fp8_ptpc):
            shared_out = fx.recast_iter(out_dtype, shared_a)
        else:
            shared_out = shared_a

        # Full tensor, not the tile. A tile that hangs past M or N stays
        # block-sized in LDS; TDM zero-fills the out-of-range elements.
        def tdm_operand(global_pointer, global_shape, global_stride, tile_shape, tile_stride, store=False):
            global_view = fx.make_view(
                global_pointer,
                fx.make_layout(global_shape, global_stride),
            )
            atom, coords = rocdl.cdna5.make_tiled_tdm_atom(
                rocdl.TensorStore() if store else rocdl.TensorLoad(),
                global_view,
                fx.make_layout(tile_shape, tile_stride),
                tile_shape,
                num_warps=waves,
            )
            return atom, fx.zipped_divide(coords, tile_shape)

        a_global_stride = (a_leading_stride, 1)
        if const_expr(b_k_contiguous):
            b_global_stride = (1, b_leading_stride)
        else:
            b_global_stride = (b_leading_stride, 1)
        atom_a, coords_a = tdm_operand(
            a, (m, k), a_global_stride, (block_m, block_k), a_tile_stride,
        )
        atom_b, coords_b = tdm_operand(
            b, (k, n), b_global_stride, (block_k, block_n), b_tile_stride,
        )

        def copy_operand(atom, shared_pointer, tile_shape, tile_stride, stage_size, dst_stage, src_stage, coord):
            shared_view = fx.make_view(
                shared_pointer,
                fx.make_layout((tile_shape, stages), (tile_stride, stage_size)),
            )
            dst, src = rocdl.cdna5.tdm_partition(
                atom,
                wave,
                fx.make_layout(waves, 1),
                shared_view,
                coord,
            )
            fx.copy(atom, src[None, src_stage], dst[None, dst_stage])

        def copy_tile(dst_stage, src_stage):
            # Rest modes follow the view: A is (M tile, K tile), B is (K tile, N tile).
            copy_operand(
                atom_b, shared_b, (block_k, block_n), b_tile_stride, b_stage_size,
                dst_stage, src_stage, coords_b[None, (None, block_col)],
            )
            copy_operand(
                atom_a, shared_a, (block_m, block_k), a_tile_stride, a_stage_size,
                dst_stage, src_stage, coords_a[None, (block_row, None)],
            )
            rocdl.sched_barrier(0)

        def wait_tile(pending_stages):
            tdm_ops.tensor_wait(pending_stages * 2)
            # Per-wave completion is not enough: all waves share the tile.
            # This also protects the previous tile's LDS reads before recycling.
            fx.gpu.barrier()

        mma = fx.make_mma_atom(rocdl.WMMA(mma_m, mma_n, mma_k, elem_dtype, fx.Float32))
        accumulators = [
            fx.make_rmem_tensor(acc_elems, fx.Float32)
            for _ in range_constexpr(m_steps * n_steps)
        ]
        for acc in accumulators:
            acc.store(fx.full(acc_elems, 0.0, fx.Float32))

        def load_fragment(shared_pointer, stage_size, stage, row, column, k_contiguous=True):
            # One formula covers 16x16x32 FP16/BF16 and 16x16x128 FP8.
            # The two 16-lane groups own alternating 16-byte chunks along K.
            lane_k = lane // 16 * frag_load_elems

            def fragment_k_offset(i):
                return (
                    lane_k
                    + (i // frag_load_elems) * frag_k_stride
                    + i % frag_load_elems
                )

            if const_expr(k_contiguous):
                base = (
                    stage * stage_size
                    + (row + lane % 16) * (block_k + lds_pad)
                    + column
                )
                chunks = [
                    fx.ptr_load(
                        shared_pointer
                        + base
                        + fragment_k_offset(group * frag_load_elems),
                        result_type=T.vec(frag_load_elems, elem_dtype.ir_type),
                    )
                    for group in range_constexpr(frag_loads)
                ]
                values = fx.Vector.from_elements([
                    chunks[i // frag_load_elems][i % frag_load_elems]
                    for i in range_constexpr(frag_elems)
                ])
            else:
                base = stage * stage_size + row + lane % 16
                values = fx.Vector.from_elements([
                    fx.ptr_load(
                        shared_pointer
                        + base
                        + (column + fragment_k_offset(i)) * b_pitch
                    )
                    for i in range_constexpr(frag_elems)
                ])
            fragment = fx.make_rmem_tensor(frag_elems, elem_dtype)
            fragment.store(values)
            return fragment

        def compute_tile(stage):
            for ki in range_constexpr(block_k // mma_k):
                b_fragments = [
                    load_fragment(
                        shared_b, b_stage_size, stage, wave_col + ni * mma_n, ki * mma_k,
                        k_contiguous=b_k_contiguous,
                    )
                    for ni in range_constexpr(n_steps)
                ]
                rocdl.sched_barrier(0)
                for mi in range_constexpr(m_steps):
                    a_fragment = load_fragment(
                        shared_a, a_stage_size, stage, wave_row + mi * mma_m, ki * mma_k
                    )
                    rocdl.sched_barrier(0)
                    for ni in range_constexpr(n_steps):
                        acc = accumulators[mi * n_steps + ni]
                        fx.gemm(mma, acc, a_fragment, b_fragments[ni], acc)

        # Each cluster rank owns an equal K slice. Tile indices stay in the
        # full-tensor coordinate space.
        k_rank = fx.block_idx.y
        k_tiles = (k // split_k) // block_k
        k_tile_base = k_rank * k_tiles
        if const_expr(interleave):
            # Two b128 loads per BF16/FP16 fragment, four per FP8 fragment.
            fragment_words = frag_elems * elem_bits // 32
            quadrant_steps = half_m_steps * half_n_steps
            wmma_k_steps = block_k // mma_k
            load_atom = fx.make_copy_atom(fx.UniversalCopy(FRAGMENT_LOAD_BITS), fx.Int32)
            load_layout = fx.make_layout(FRAGMENT_LOAD_BITS // 32, 1)
            aligned_lds_type = fx.PointerType.get(
                elem_ty=fx.Int32.ir_type,
                address_space=fx.AddressSpace.Shared,
                alignment=16,
            )

            def fragment_pointer(operand, stage, k_step):
                if const_expr(operand == "a"):
                    pointer, stage_size, wave_start, pitch = shared_a, a_stage_size, wave_row, a_pitch
                else:
                    pointer, stage_size, wave_start, pitch = shared_b, b_stage_size, wave_col, b_pitch
                pointer = (
                    pointer + stage * stage_size + (wave_start + lane % 16) * pitch
                    + k_step * mma_k + lane // 16 * frag_load_elems
                )
                return pointer, mma_m * pitch

            def load_fragment_part(pointer, element_offset):
                address = fx.Int32(fx.ptrtoint(pointer)) + fx.Int32(element_offset) * (elem_bits // 8)
                source = fx.inttoptr(aligned_lds_type, address)
                values = fx.make_rmem_tensor(load_layout, fx.Int32)
                fx.copy(load_atom, fx.make_view(source, load_layout), values)
                return values.load()

            def join_fragment_parts(parts):
                values = parts[0].shuffle(parts[1], list(range(8)))
                if const_expr(frag_loads == 4):
                    values = values.shuffle(
                        parts[2].shuffle(parts[3], list(range(8))), list(range(16))
                    )
                return values

            def load_packed_fragment(pointer, element_offset):
                return join_fragment_parts([
                    load_fragment_part(pointer, element_offset + part * frag_k_stride)
                    for part in range_constexpr(frag_loads)
                ])

            a_first_half = [
                fx.make_rmem_tensor(fragment_words, fx.Int32)
                for _ in range_constexpr(half_m_steps)
            ]
            b_first_half = [
                fx.make_rmem_tensor(fragment_words, fx.Int32)
                for _ in range_constexpr(half_n_steps)
            ]

            def first_half_loads(stage, k_step, parity):
                a_pointer, a_row_stride = fragment_pointer("a", stage, k_step)
                b_pointer, b_row_stride = fragment_pointer("b", stage, k_step)
                a_loads = [
                    lambda index=index: a_first_half[index].store(
                        load_packed_fragment(a_pointer, index * a_row_stride)
                    )
                    for index in range_constexpr(half_m_steps)
                ]
                b_loads = [
                    lambda index=index: b_first_half[index].store(
                        load_packed_fragment(b_pointer, index * b_row_stride)
                    )
                    for index in range_constexpr(half_n_steps)
                ]
                return (a_loads, b_loads) if parity == 0 else (b_loads, a_loads)

            def second_half_loads(operand, stage, k_step):
                count = half_m_steps if operand == "a" else half_n_steps
                pointer, row_stride = fragment_pointer(operand, stage, k_step)
                fragments = [None] * count
                parts = [[] for _ in range_constexpr(count)]

                def load_part(index, part):
                    parts[index].append(load_fragment_part(
                        pointer, (count + index) * row_stride + part * frag_k_stride
                    ))
                    if const_expr(part == frag_loads - 1):
                        fragment = fx.make_rmem_tensor(fragment_words, fx.Int32)
                        fragment.store(join_fragment_parts(parts[index]))
                        fragments[index] = fragment

                loads = {
                    index * frag_loads + part: (
                        lambda index=index, part=part: load_part(index, part)
                    )
                    for index in range(count) for part in range(frag_loads)
                }
                return loads, fragments

            def compute_quadrant(
                m_step_start, n_step_start, a_fragments, b_fragments,
                loads, n_fast, before=None,
            ):
                if const_expr(before is not None):
                    before()
                for step in range_constexpr(quadrant_steps):
                    if const_expr(n_fast):
                        m_index, n_index = step // half_n_steps, step % half_n_steps
                    else:
                        m_index, n_index = step % half_m_steps, step // half_m_steps
                    acc = accumulators[(m_step_start + m_index) * n_steps + n_step_start + n_index]
                    # Swap operands so each lane owns consecutive output columns.
                    # The interleaved epilogue below restores logical C[M, N].
                    fx.gemm(mma, acc, b_fragments[n_index], a_fragments[m_index], acc)
                    if const_expr(step in loads):
                        loads[step]()
                    if const_expr(step == max(loads, default=-1)):
                        for _ in range_constexpr(len(loads)):
                            rocdl.sched_mfma(1)
                            rocdl.sched_dsrd(frag_loads // 2)
                        rocdl.sched_barrier(0)

            def compute_interleaved_tile(k_tile, has_next_tile, parity, read_stage=None):
                read_stage = k_tile % stages if read_stage is None else read_stage
                for k_step in range_constexpr(wmma_k_steps):
                    at_boundary = k_step == wmma_k_steps - 1
                    next_stage = (
                        ((1 - read_stage) if stages == 2 else (k_tile + 1) % stages)
                        if at_boundary else read_stage
                    )
                    next_k_step = 0 if at_boundary else k_step + 1

                    def signal_next_stage(at_boundary=at_boundary):
                        rocdl.sched_barrier(0)
                        rocdl.s_wait_dscnt(0)
                        if const_expr(at_boundary and has_next_tile):
                            tdm_ops.tensor_wait(2 * (stages - 2))
                            rocdl.s_barrier_signal(-1)
                        rocdl.sched_barrier(0)

                    def refill_stage(at_boundary=at_boundary):
                        if const_expr(at_boundary and has_next_tile):
                            rocdl.s_barrier_wait(-1)
                            copy_tile(read_stage, fx.min(k_tile_base + k_tile + stages, last_tile))
                        rocdl.sched_barrier(0)

                    has_next_step = not at_boundary or has_next_tile
                    early_loads, late_loads = first_half_loads(next_stage, next_k_step, parity) if has_next_step else ([], [])
                    refill_step = quadrant_steps // 4
                    early_load_slots = quadrant_steps - refill_step
                    third_quadrant_loads = {refill_step - 1: refill_stage}
                    third_quadrant_loads.update({
                        refill_step + index: load for index, load in enumerate(early_loads[:early_load_slots])
                    })
                    last_quadrant_loads = dict(enumerate(early_loads[early_load_slots:]))
                    last_quadrant_loads.update({
                        quadrant_steps - len(late_loads) + index: load
                        for index, load in enumerate(late_loads)
                    })
                    # Even: Q00 -> Q01 -> Q11 -> Q10. Odd: Q00 -> Q10 -> Q11 -> Q01.
                    # The next first half replaces its old registers only after
                    # the final quadrant that reads them; no double RF buffer.
                    if parity == 0:
                        loads, b_second_half = second_half_loads("b", read_stage, k_step)
                        compute_quadrant(0, 0, a_first_half, b_first_half, loads, False)
                        loads, a_second_half = second_half_loads("a", read_stage, k_step)
                        compute_quadrant(0, half_n_steps, a_first_half, b_second_half, loads, False)
                        compute_quadrant(
                            half_m_steps, half_n_steps, a_second_half, b_second_half,
                            third_quadrant_loads, True, signal_next_stage,
                        )
                        compute_quadrant(
                            half_m_steps, 0, a_second_half, b_first_half,
                            last_quadrant_loads, True,
                        )
                    else:
                        loads, a_second_half = second_half_loads("a", read_stage, k_step)
                        compute_quadrant(0, 0, a_first_half, b_first_half, loads, True)
                        loads, b_second_half = second_half_loads("b", read_stage, k_step)
                        compute_quadrant(half_m_steps, 0, a_second_half, b_first_half, loads, True)
                        compute_quadrant(
                            half_m_steps, half_n_steps, a_second_half, b_second_half,
                            third_quadrant_loads, False, signal_next_stage,
                        )
                        compute_quadrant(
                            0, half_n_steps, a_first_half, b_second_half,
                            last_quadrant_loads, False,
                        )

            # Clamp unused tail prefetches to the split's last tile. This also
            # admits the existing minimum stages-1 tiles without reading past K.
            last_tile = k_tile_base + k_tiles - 1
            for stage in range_constexpr(stages):
                copy_tile(fx.Int32(stage), fx.min(k_tile_base + stage, last_tile))
            tdm_ops.tensor_wait(2 * (stages - 1))
            fx.gpu.barrier()
            for loads in first_half_loads(fx.Int32(0), 0, 0):
                for load in loads:
                    load()

            def run_interleaved(parity):
                for pair in range((k_tiles - 1) // 2):
                    for step in range_constexpr(2):
                        compute_interleaved_tile(
                            pair * 2 + step, True, parity, step if stages == 2 else None
                        )
                for remaining in range((k_tiles - 1) % 2):
                    compute_interleaved_tile(k_tiles - 2, True, parity)
                compute_interleaved_tile(k_tiles - 1, False, parity)

            wave_parity = fx.Int32(rocdl.readfirstlane(T.i32, wave % 2))
            if wave_parity == 0:
                run_interleaved(0)
            else:
                run_interleaved(1)
            tdm_ops.tensor_wait(0)
        else:
            for stage in range_constexpr(stages - 1):
                copy_tile(stage, k_tile_base + fx.Int32(stage))

            # Two tiles per iteration. Double buffering retains literal LDS offsets;
            # deeper pipelines advance the ring modulo stages.
            main_tiles = k_tiles - (stages - 1)
            current_stage = fx.Int32(0)
            for pair in range(main_tiles // 2):
                for step in range_constexpr(2):
                    read_stage = step if stages == 2 else current_stage
                    wait_tile(stages - 2)
                    copy_tile(
                        (read_stage + stages - 1) % stages,
                        k_tile_base + pair * 2 + step + stages - 1,
                    )
                    compute_tile(read_stage)
                    current_stage = (current_stage + 1) % stages

            # A one-trip loop keeps the accumulators loop-carried instead of
            # materializing both sides of a predicated remainder branch.
            for remaining in range(main_tiles % 2):
                wait_tile(stages - 2)
                copy_tile(
                    (current_stage + stages - 1) % stages,
                    k_tile_base + main_tiles + stages - 2,
                )
                compute_tile(current_stage)
                current_stage = (current_stage + 1) % stages

            # Drain: retire the remaining loads in order without issuing more.
            for tail in range_constexpr(stages - 1):
                wait_tile(stages - 2 - tail)
                compute_tile(current_stage)
                current_stage = (current_stage + 1) % stages

        # Reuse the A/B arena only after all TDM and fragment reads have retired.
        fx.gpu.barrier()
        if const_expr(interleave):
            if const_expr(is_fp8_ptpc and split_k == 1):
                row_scales = [
                    fx.ptr_load(
                        scale_a + fx.min(row_start + wave_row + mi * mma_m + lane % 16, m - 1),
                        result_type=fx.Float32.ir_type,
                    )
                    for mi in range_constexpr(m_steps)
                ]
                col_scales = [
                    fx.Vector.from_elements([
                        fx.ptr_load(
                            scale_b + fx.min(
                                col_start + wave_col + ni * mma_n + lane // 16 * acc_elems + i, n - 1
                            ),
                            result_type=fx.Float32.ir_type,
                        )
                        for i in range_constexpr(acc_elems)
                    ])
                    for ni in range_constexpr(n_steps)
                ]
            for mi in range_constexpr(m_steps):
                for ni in range_constexpr(n_steps):
                    row = wave_row + mi * mma_m + lane % 16
                    col = wave_col + ni * mma_n + lane // 16 * acc_elems
                    values = accumulators[mi * n_steps + ni].load()
                    if const_expr(is_fp8_ptpc and split_k == 1):
                        values = values * row_scales[mi] * col_scales[ni]
                    fx.make_view(
                        shared_out + row * c_pitch + col, fx.make_layout(acc_elems, 1)
                    ).store(values.to(out_dtype))
        else:
            # Reuse row/column scales across accumulator tiles. Clamped edge
            # lanes are never committed to global output.
            if const_expr(is_fp8_ptpc and split_k == 1):
                row_scales = []
                for mi in range_constexpr(m_steps):
                    row = row_start + wave_row + mi * mma_m + lane // 16 * acc_elems
                    row_scales.append(fx.Vector.from_elements([
                        fx.ptr_load(scale_a + fx.min(row + i, m - 1), result_type=fx.Float32.ir_type)
                        for i in range_constexpr(acc_elems)
                    ]))
                col_scales = []
                for ni in range_constexpr(n_steps):
                    col = col_start + wave_col + ni * mma_n + lane % 16
                    col_scales.append(fx.ptr_load(scale_b + fx.min(col, n - 1), result_type=fx.Float32.ir_type))
            for mi in range_constexpr(m_steps):
                for ni in range_constexpr(n_steps):
                    row = wave_row + mi * mma_m + lane // 16 * acc_elems
                    col = wave_col + ni * mma_n + lane % 16
                    values = accumulators[mi * n_steps + ni].load()
                    if const_expr(is_fp8_ptpc and split_k == 1):
                        values = values * row_scales[mi] * col_scales[ni]
                    values = values.to(out_dtype)
                    for i in range_constexpr(acc_elems):
                        fx.ptr_store(values[i], shared_out + (row + i) * block_n + col)
        fx.gpu.barrier()

        def commit_output(values, row, col, apply_scale=False):
            destination = (
                out
                + fx.Int64(row_start + row) * fx.Int64(c_leading_stride)
                + col_start
                + col
            )
            # Interior tiles store one 16-byte vector along N. An edge tile may
            # cover rows or columns past M or N; those elements stay unstored.
            if (row_start + block_m <= m) and (col_start + block_n <= n):
                # Apply PTPC scaling in FP32, then round before adding bias.
                if const_expr(apply_scale):
                    scale_a_value = fx.ptr_load(
                        scale_a + row_start + row, result_type=fx.Float32.ir_type
                    )
                    scaled = []
                    for element in range_constexpr(out_vec_elems):
                        scale_b_value = fx.ptr_load(
                            scale_b + col_start + col + element,
                            result_type=fx.Float32.ir_type,
                        )
                        scaled.append(
                            (values[element].to(fx.Float32) * scale_a_value * scale_b_value).to(
                                out_dtype
                            )
                        )
                    output_values = fx.Vector.from_elements(scaled)
                else:
                    output_values = values.to(out_dtype)
                if const_expr(has_bias):
                    bias_values = fx.ptr_load(
                        bias + col_start + col, result_type=T.vec(out_vec_elems, out_dtype.ir_type)
                    )
                    output_values = (
                        output_values.to(fx.Float32) + bias_values.to(fx.Float32)
                    ).to(out_dtype)
                fx.make_view(destination, fx.make_layout(out_vec_elems, 1)).store(
                    output_values
                )
            else:
                for element in range_constexpr(out_vec_elems):
                    if (row_start + row < m) and (col_start + col + element < n):
                        value = values[element]
                        if const_expr(apply_scale):
                            scale_a_value = fx.ptr_load(
                                scale_a + row_start + row, result_type=fx.Float32.ir_type
                            )
                            scale_b_value = fx.ptr_load(
                                scale_b + col_start + col + element,
                                result_type=fx.Float32.ir_type,
                            )
                            value = (
                                value.to(fx.Float32) * scale_a_value * scale_b_value
                            ).to(out_dtype)
                        else:
                            value = value.to(out_dtype)
                        if const_expr(has_bias):
                            bias_value = fx.ptr_load(
                                bias + col_start + col + element, result_type=out_dtype.ir_type
                            )
                            value = (value.to(fx.Float32) + bias_value.to(fx.Float32)).to(out_dtype)
                        fx.ptr_store(value, destination + element)

        tile_elems = block_m * block_n
        if const_expr(interleave and split_k == 1 and not has_bias):
            c_tile = (block_m, c_pitch)
            atom_c, coords_c = tdm_operand(
                out + fx.Int64(row_start) * fx.Int64(c_leading_stride) + col_start,
                (fx.max(0, fx.min(block_m, m - row_start)),
                 fx.max(0, fx.min(block_n, n - col_start))),
                (c_leading_stride, 1), c_tile, (c_pitch, 1), store=True,
            )
            shared_c = fx.make_view(
                shared_out,
                fx.make_layout((c_tile, 1), ((c_pitch, 1), block_m * c_pitch)),
            )
            src, dst = rocdl.cdna5.tdm_partition(
                atom_c, wave, fx.make_layout(waves, 1),
                shared_c, coords_c[None, (0, None)],
            )
            fx.copy(atom_c, src[None, 0], dst[None, 0])
            tdm_ops.tensor_wait(0)
        elif const_expr(split_k == 1):
            for i in range_constexpr(tile_elems // (threads * out_vec_elems)):
                offset = (i * threads + tid) * out_vec_elems
                row, col = offset // block_n, offset % block_n
                values = fx.ptr_load(shared_out + row * c_pitch + col, result_type=T.vec(out_vec_elems, out_dtype.ir_type))
                commit_output(values, row, col)
        else:
            for i in range_constexpr(tile_elems // (threads * out_vec_elems)):
                offset = (i * threads + tid) * out_vec_elems
                row, col = offset // block_n, offset % block_n
                values = fx.ptr_load(shared_out + row * c_pitch + col, result_type=T.vec(out_vec_elems, out_dtype.ir_type))
                partial = (
                    workspace
                    + fx.Int64(block_id) * (split_k * tile_elems)
                    + fx.Int64(k_rank) * tile_elems
                    + row * block_n
                    + col
                )
                fx.make_view(partial, fx.make_layout(out_vec_elems, 1)).store(values)
            cluster_barrier()
            if k_rank == 0:
                for i in range_constexpr(tile_elems // (threads * out_vec_elems)):
                    offset = (i * threads + tid) * out_vec_elems
                    row, col = offset // block_n, offset % block_n
                    total = fx.ptr_load(
                        workspace + fx.Int64(block_id) * (split_k * tile_elems) + offset,
                        result_type=T.vec(out_vec_elems, out_dtype.ir_type),
                    ).to(fx.Float32)
                    for part in range_constexpr(split_k - 1):
                        partial = fx.ptr_load(
                            workspace
                            + fx.Int64(block_id) * (split_k * tile_elems)
                            + (part + 1) * tile_elems
                            + offset,
                            result_type=T.vec(out_vec_elems, out_dtype.ir_type),
                        )
                        total = total + partial.to(fx.Float32)
                    commit_output(total, row, col, apply_scale=is_fp8_ptpc)

    @flyc.jit
    def launch(
        a: fx.Pointer,
        b: fx.Pointer,
        out: fx.Pointer,
        bias: fx.Pointer,
        workspace: fx.Pointer,
        scale_a: fx.Pointer,
        scale_b: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        a_leading_stride: fx.Int32,
        b_leading_stride: fx.Int32,
        c_leading_stride: fx.Int32,
        stream: fx.Stream,
    ):
        mn_tiles = (m + block_m - 1) // block_m * ((n + block_n - 1) // block_n)
        if split_k == 1:
            kernel(
                a, b, out, bias, workspace, scale_a, scale_b, m, n, k,
                a_leading_stride, b_leading_stride, c_leading_stride,
            ).launch(
                grid=(mn_tiles, 1, 1),
                block=(threads, 1, 1),
                stream=stream,
            )
        else:
            kernel(
                a, b, out, bias, workspace, scale_a, scale_b, m, n, k,
                a_leading_stride, b_leading_stride, c_leading_stride,
                value_attrs={"rocdl.cluster_dims": f"1,{split_k},1"},
            ).launch(
                grid=(mn_tiles, split_k, 1),
                block=(threads, 1, 1),
                stream=stream,
                cluster=(1, split_k, 1),
            )

    kernel._func.__name__ = make_gemm_a16w16_gfx1250_kernel_name(
        dtype_id,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        group_m,
        has_bias,
        b_k_contiguous,
        mma_m,
        mma_n,
        mma_k,
        split_k,
        interleave,
    )
    if interleave:
        launch.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
            "amdgpu-sched-strategy": "coexec",
            "enable-post-misched": True,
            "amdgpu-barrier-signal-wait-latency": 32,
        }
    return launch


def gemm_a16w16_gfx1250(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    *,
    block_m: int = 256,
    block_n: int = 256,
    block_k: int = 128,
    stages: int = 2,
    m_waves: int = 4,
    n_waves: int = 4,
    group_m: int = 0,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
    split_k: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
    layout: str = "nt",
    interleave: bool = False,
) -> torch.Tensor:
    """C = A[M,K] @ B[K,N]; FP16/BF16 output, optional bias.

    ``layout`` is ``nt`` or ``nn``. N means that dimension is contiguous
    (stride 1); T means the other dimension is contiguous. The other stride
    is the leading stride and may be larger than the contiguous size, so
    padded rows are allowed. C is always row-major. M and N may end on a
    partial tile; K must contain at least ``stages - 1`` complete tiles.
    No split-K tail: K must be divisible by split_k * block_k. split_k is the
    cluster size along the K split and must be in 1..9.
    ``interleave=True`` uses four quadrant tiles per wave and opposite serpentine
    orders on even/odd waves. It requires NT B and sufficiently large even wave
    tiles; the default schedule and split-K reduction semantics are unchanged.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise ValueError("A and B must have the same FP16 or BF16 dtype")
    dtype_id = GEMM_DTYPE_FP16 if a.dtype is torch.float16 else GEMM_DTYPE_BF16
    if a.device != b.device or a.device.type != "cuda":
        raise ValueError("A and B must be on the same ROCm device")
    layout = layout.lower()
    if layout not in ("nn", "nt"):
        raise ValueError("layout must be 'nn' or 'nt'")
    m, k = a.shape
    n = b.shape[1]
    validate_gemm_gfx1250_config(
        m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m,
        dtype_id, mma_m, mma_n, mma_k, layout == "nt", split_k, interleave,
    )
    a_leading_stride = _leading_stride(a, 1, 0, k, f"A does not satisfy the {layout.upper()} layout")
    if layout == "nt":
        b_leading_stride = _leading_stride(b, 0, 1, k, "B does not satisfy the NT layout")
    else:
        b_leading_stride = _leading_stride(b, 1, 0, n, "B does not satisfy the NN layout")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if out.shape != (m, n) or out.dtype != a.dtype or out.device != a.device:
        raise ValueError("out must be [M,N] with the input dtype and device")
    c_leading_stride = _leading_stride(out, 1, 0, n, "out must be row-major [M,N]")
    if bias is not None and (
        bias.shape != (n,)
        or bias.dtype != a.dtype
        or bias.device != a.device
        or not bias.is_contiguous()
    ):
        raise ValueError("bias must be contiguous [N] with the input dtype and device")
    tensors = [a, b, out] + ([] if bias is None else [bias])
    if any(t.data_ptr() % 16 for t in tensors):
        raise ValueError("all pointers must be 16-byte aligned")
    if max(m * a_leading_stride, k * b_leading_stride, n * b_leading_stride, m * c_leading_stride) > (2**32 - 1) // 2:
        raise ValueError("tensor offsets exceed the supported 32-bit range")
    out_lo, out_hi = _storage_span(out)
    if any(_overlaps(out_lo, out_hi, t) for t in [a, b] + ([] if bias is None else [bias])):
        raise ValueError("out must not overlap the inputs")
    with torch.cuda.device(a.device):
        arch = torch.cuda.get_device_properties(a.device).gcnArchName.split(":")[0]
        if arch != "gfx1250":
            raise ValueError(f"TDM GEMM requires gfx1250, got {arch}")
        if stream is None:
            stream = torch.cuda.current_stream()
        if stream.device != a.device:
            raise ValueError("stream must belong to the input device")
        dtype = fx.Float16 if dtype_id == GEMM_DTYPE_FP16 else fx.BFloat16
        mn_tiles = ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n)
        workspace = (
            out
            if split_k == 1
            else torch.empty(
                (mn_tiles * split_k * block_m * block_n,),
                device=a.device,
                dtype=a.dtype,
            )
        )
        pointers = [
            flyc.from_c_void_p(dtype, t.data_ptr())
            for t in (a, b, out, a if bias is None else bias, workspace)
        ]
        # FP16/BF16 leaves the scale pointers unused. The FP8 PTPC entry
        # passes real per-token and per-channel scales.
        pointers.extend((
            flyc.from_c_void_p(fx.Float32, a.data_ptr()),
            flyc.from_c_void_p(fx.Float32, a.data_ptr()),
        ))
        has_bias = bias is not None
        b_k_contiguous = layout == "nt"
        param = GemmA16W16Gfx1250Param(
            dtype_id=dtype_id,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            group_m=group_m,
            has_bias=has_bias,
            b_k_contiguous=b_k_contiguous,
            mma_m=mma_m,
            mma_n=mma_n,
            mma_k=mma_k,
            split_k=split_k,
            interleave=interleave,
        )
        launch = build_gemm_gfx1250(
            dtype_id,
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            group_m,
            has_bias,
            b_k_contiguous,
            mma_m,
            mma_n,
            mma_k,
            split_k,
            interleave,
        )
        dispatch_args = (
            *pointers,
            m,
            n,
            k,
            a_leading_stride,
            b_leading_stride,
            c_leading_stride,
            stream,
        )
        run_cached(
            launch,
            *dispatch_args,
            constexpr_param=param,
            compiler=flyc.compile,
            dispatch_args=dispatch_args,
        )
    return out


def gemm_fp8_ptpc_gfx1250(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    *,
    block_m: int = 256,
    block_n: int = 256,
    block_k: int = 128,
    stages: int = 2,
    m_waves: int = 4,
    n_waves: int = 4,
    group_m: int = 0,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 128,
    split_k: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
    layout: str = "nt",
    interleave: bool = False,
) -> torch.Tensor:
    """C = scale_a[M] * scale_b[N] * (A[M,K] @ B[K,N]), BF16 output.

    A and B are FP8 E4M3. The WMMA loop accumulates the unscaled product in
    fp32. scale_a and scale_b are FP32 and are multiplied once in the
    epilogue, after a cluster split-K sum. Optional bias is BF16 and is added
    after that result is rounded. ``interleave=True`` uses the same optional
    NT quadrant schedule as the BF16/FP16 entrypoint; scaling is unchanged.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    fp8 = torch.float8_e4m3fn
    if a.dtype is not fp8 or b.dtype is not fp8:
        raise ValueError("A and B must be float8_e4m3fn")
    if a.device != b.device or a.device.type != "cuda":
        raise ValueError("A and B must be on the same ROCm device")
    layout = layout.lower()
    if layout not in ("nn", "nt"):
        raise ValueError("layout must be 'nn' or 'nt'")
    m, k = a.shape
    n = b.shape[1]
    validate_gemm_gfx1250_config(
        m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m,
        GEMM_DTYPE_FP8_PTPC, mma_m, mma_n, mma_k, layout == "nt", split_k, interleave,
    )
    a_leading_stride = _leading_stride(a, 1, 0, k, f"A does not satisfy the {layout.upper()} layout")
    if layout == "nt":
        b_leading_stride = _leading_stride(b, 0, 1, k, "B does not satisfy the NT layout")
    else:
        b_leading_stride = _leading_stride(b, 1, 0, n, "B does not satisfy the NN layout")
    if scale_a.shape != (m,) or scale_a.dtype is not torch.float32 or scale_a.device != a.device or not scale_a.is_contiguous():
        raise ValueError("scale_a must be contiguous FP32 [M] on the input device")
    if scale_b.shape != (n,) or scale_b.dtype is not torch.float32 or scale_b.device != a.device or not scale_b.is_contiguous():
        raise ValueError("scale_b must be contiguous FP32 [N] on the input device")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    if out.shape != (m, n) or out.dtype is not torch.bfloat16 or out.device != a.device:
        raise ValueError("out must be BF16 [M,N] on the input device")
    c_leading_stride = _leading_stride(out, 1, 0, n, "out must be row-major [M,N]")
    if bias is not None and (
        bias.shape != (n,)
        or bias.dtype is not torch.bfloat16
        or bias.device != a.device
        or not bias.is_contiguous()
    ):
        raise ValueError("bias must be contiguous BF16 [N] on the input device")
    tensors = [a, b, out, scale_a, scale_b] + ([] if bias is None else [bias])
    if any(t.data_ptr() % 16 for t in tensors):
        raise ValueError("all pointers must be 16-byte aligned")
    if max(m * a_leading_stride, k * b_leading_stride, n * b_leading_stride, m * c_leading_stride) > (2**32 - 1) // 2:
        raise ValueError("tensor offsets exceed the supported 32-bit range")
    out_lo, out_hi = _storage_span(out)
    if any(_overlaps(out_lo, out_hi, t) for t in [a, b, scale_a, scale_b] + ([] if bias is None else [bias])):
        raise ValueError("out must not overlap the inputs")
    with torch.cuda.device(a.device):
        arch = torch.cuda.get_device_properties(a.device).gcnArchName.split(":")[0]
        if arch != "gfx1250":
            raise ValueError(f"TDM GEMM requires gfx1250, got {arch}")
        if stream is None:
            stream = torch.cuda.current_stream()
        if stream.device != a.device:
            raise ValueError("stream must belong to the input device")
        mn_tiles = ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n)
        workspace = (
            out
            if split_k == 1
            else torch.empty(
                (mn_tiles * split_k * block_m * block_n,),
                device=a.device,
                dtype=torch.bfloat16,
            )
        )
        pointers = [
            flyc.from_c_void_p(fx.Float8E4M3FN, a.data_ptr()),
            flyc.from_c_void_p(fx.Float8E4M3FN, b.data_ptr()),
            flyc.from_c_void_p(fx.BFloat16, out.data_ptr()),
            flyc.from_c_void_p(fx.BFloat16, (out if bias is None else bias).data_ptr()),
            flyc.from_c_void_p(fx.BFloat16, workspace.data_ptr()),
            flyc.from_c_void_p(fx.Float32, scale_a.data_ptr()),
            flyc.from_c_void_p(fx.Float32, scale_b.data_ptr()),
        ]
        has_bias = bias is not None
        b_k_contiguous = layout == "nt"
        param = GemmA16W16Gfx1250Param(
            dtype_id=GEMM_DTYPE_FP8_PTPC,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            group_m=group_m,
            has_bias=has_bias,
            b_k_contiguous=b_k_contiguous,
            mma_m=mma_m,
            mma_n=mma_n,
            mma_k=mma_k,
            split_k=split_k,
            interleave=interleave,
        )
        launch = build_gemm_gfx1250(
            GEMM_DTYPE_FP8_PTPC,
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            group_m,
            has_bias,
            b_k_contiguous,
            mma_m,
            mma_n,
            mma_k,
            split_k,
            interleave,
        )
        dispatch_args = (
            *pointers,
            m,
            n,
            k,
            a_leading_stride,
            b_leading_stride,
            c_leading_stride,
            stream,
        )
        run_cached(
            launch,
            *dispatch_args,
            constexpr_param=param,
            compiler=flyc.compile,
            dispatch_args=dispatch_args,
        )
    return out


def _leading_stride(tensor, unit_dim, leading_dim, min_leading, message):
    unit, leading = tensor.stride(unit_dim), tensor.stride(leading_dim)
    if (
        unit != 1
        or leading < min_leading
        or leading * tensor.element_size() % 16 != 0
    ):
        raise ValueError(
            f"{message}: expected stride({unit_dim}) == 1 and a 16-byte-aligned "
            f"leading stride({leading_dim}) >= {min_leading}; got stride={tuple(tensor.stride())}"
        )
    return int(leading)


def _storage_span(tensor):
    last = (tensor.shape[0] - 1) * tensor.stride(0) + (tensor.shape[1] - 1) * tensor.stride(1)
    start = tensor.data_ptr()
    return start, start + (last + 1) * tensor.element_size()


def _overlaps(out_lo, out_hi, tensor):
    lo, hi = _storage_span(tensor) if tensor.ndim == 2 else (
        tensor.data_ptr(),
        tensor.data_ptr() + tensor.numel() * tensor.element_size(),
    )
    return out_lo < hi and lo < out_hi
