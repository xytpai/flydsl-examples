# SPDX-License-Identifier: MIT
"""PeakGemm-style wave32 TDM GEMM: NT or NN layout, full tiles, no split-K."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl, tdm_ops
from flydsl.expr.typing import T

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
GEMM_DTYPE_NAMES = {
    GEMM_DTYPE_FP32: "fp32",
    GEMM_DTYPE_BF16: "bf16",
    GEMM_DTYPE_FP16: "fp16",
    GEMM_DTYPE_MXFP8: "mxfp8",
    GEMM_DTYPE_MXFP4: "mxfp4",
}


def _check_config(
    m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m,
    b_k_contiguous=True,
):
    values = (m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves)
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError(
            "dimensions, tiles, stages and wave counts must be positive integers"
        )
    if type(group_m) is not int or group_m < 0:
        raise ValueError("group_m must be a nonnegative integer")
    if not 2 <= stages <= 33:
        raise ValueError("stages must be in [2, 33] (two TDM loads per stage)")
    if block_k not in (32, 64, 128, 256, 512):
        raise ValueError("block_k must be a power of two in [32, 512]")
    waves = m_waves * n_waves
    if waves > 32:
        raise ValueError("a workgroup cannot exceed 1024 threads")
    if block_m % (m_waves * 16) or block_n % (n_waves * 16):
        raise ValueError(
            "tile dimensions must be divisible by their wave count times 16"
        )
    if not (
        1 <= block_m // (m_waves * 16) <= 8 and 1 <= block_n // (n_waves * 16) <= 8
    ):
        raise ValueError("each wave must compute 1..8 WMMA tiles along M and N")
    if block_m % waves or block_n % waves:
        raise ValueError("each wave must own a whole number of A and B rows")
    threads = waves * WAVE_SIZE
    if threads % (block_n // 8) or block_m * block_n % (threads * 8):
        raise ValueError("the output tile must support whole-workgroup 16-byte stores")
    a_stage_size = block_m * (block_k + LDS_PAD)
    b_stage_size = block_n * (block_k + LDS_PAD) if b_k_contiguous else block_k * (block_n + LDS_PAD)
    shared_bytes = 2 * max(stages * (a_stage_size + b_stage_size), block_m * block_n)
    if shared_bytes > 320 * 1024:
        raise ValueError("pipeline exceeds gfx1250's 320 KiB LDS capacity")
    if k % block_k:
        raise ValueError("K must be block-tile aligned")
    if k < (stages - 1) * block_k:
        raise ValueError("K needs at least stages - 1 tiles for the prefetch prologue")
    n_blocks = (n + block_n - 1) // block_n
    if group_m * n_blocks > (2**32 - 1) // 2:
        raise ValueError("block swizzle offsets exceed the supported 32-bit range")


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
):
    elem = {GEMM_DTYPE_FP16: fx.Float16, GEMM_DTYPE_BF16: fx.BFloat16}.get(dtype_id)
    if elem is None:
        name = GEMM_DTYPE_NAMES.get(dtype_id, "unknown")
        raise ValueError(f"unsupported dtype_id={dtype_id} ({name})")
    waves = m_waves * n_waves
    threads = waves * WAVE_SIZE
    # Views are A[M, K] and B[K, N]. Pad whichever dimension is contiguous.
    a_pitch = block_k + LDS_PAD
    a_stage_size = block_m * a_pitch
    a_shared_stride = (a_pitch, 1)
    if b_k_contiguous:
        b_pitch = block_k + LDS_PAD
        b_stage_size = block_n * b_pitch
        b_shared_stride = (1, b_pitch)
    else:
        b_pitch = block_n + LDS_PAD
        b_stage_size = block_k * b_pitch
        b_shared_stride = (b_pitch, 1)
    shared_elements = max(stages * (a_stage_size + b_stage_size), block_m * block_n)
    m_steps, n_steps = block_m // (m_waves * 16), block_n // (n_waves * 16)

    @fx.struct
    class SharedStorage:
        data: fx.Array[elem, shared_elements, 16]

    @flyc.kernel
    def kernel(
        a: fx.Pointer,
        b: fx.Pointer,
        out: fx.Pointer,
        bias: fx.Pointer,
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

        # Full tensor, not the tile. A tile that hangs past M or N stays
        # block-sized in LDS; TDM zero-fills the out-of-range elements.
        def tdm_operand(pointer, shape, tile, global_stride, shared_stride):
            global_view = fx.make_view(
                pointer,
                fx.make_layout(shape, global_stride),
            )
            atom, coords = rocdl.cdna5.make_tiled_tdm_atom(
                rocdl.TensorLoad(),
                global_view,
                fx.make_layout(tile, shared_stride),
                tile,
                num_warps=waves,
            )
            return atom, fx.zipped_divide(coords, tile)

        a_global_stride = (a_leading_stride, 1)
        if const_expr(b_k_contiguous):
            b_global_stride = (1, b_leading_stride)
        else:
            b_global_stride = (b_leading_stride, 1)
        atom_a, coords_a = tdm_operand(
            a, (m, k), (block_m, block_k), a_global_stride, a_shared_stride,
        )
        atom_b, coords_b = tdm_operand(
            b, (k, n), (block_k, block_n), b_global_stride, b_shared_stride,
        )

        def copy_operand(atom, shared, tile_shape, stage, tile_index, shared_stride, stage_size, coord):
            shared_view = fx.make_view(
                shared + stage * stage_size,
                fx.make_layout((tile_shape, 1), (shared_stride, stage_size)),
            )
            dst, src = rocdl.cdna5.tdm_partition(
                atom,
                wave,
                fx.make_layout(waves, 1),
                shared_view,
                coord,
            )
            fx.copy(atom, src[None, tile_index], dst[None, 0])

        def copy_tile(stage, tile_index):
            # Rest modes follow the view: A is (M tile, K tile), B is (K tile, N tile).
            copy_operand(
                atom_b, shared_b, (block_k, block_n), stage, tile_index,
                b_shared_stride, b_stage_size, coords_b[None, (None, block_col)],
            )
            copy_operand(
                atom_a, shared_a, (block_m, block_k), stage, tile_index,
                a_shared_stride, a_stage_size, coords_a[None, (block_row, None)],
            )
            rocdl.sched_barrier(0)

        def wait_tile(pending_stages):
            tdm_ops.tensor_wait(pending_stages * 2)
            # Per-wave completion is not enough: all waves share the tile.
            # This also protects the previous tile's LDS reads before recycling.
            fx.gpu.barrier()

        mma = fx.make_mma_atom(rocdl.WMMA(16, 16, 32, elem, fx.Float32))
        accumulators = [
            fx.make_rmem_tensor(8, fx.Float32)
            for _ in range_constexpr(m_steps * n_steps)
        ]
        for acc in accumulators:
            acc.store(fx.full(8, 0.0, fx.Float32))

        def load_fragment(shared, stage, stage_size, row, column, is_k_major=True):
            if const_expr(is_k_major):
                offset = stage * stage_size + (row + lane % 16) * a_pitch + column + lane // 16 * 8
                low = fx.ptr_load(shared + offset, result_type=T.vec(8, elem.ir_type))
                high = fx.ptr_load(shared + offset + 16, result_type=T.vec(8, elem.ir_type))
                values = low.shuffle(high, list(range(16)))
            else:
                offset = stage * stage_size + row + lane % 16 + (column + lane // 16 * 8) * b_pitch
                values = fx.Vector.from_elements([
                    fx.ptr_load(shared + offset + (i % 8 + i // 8 * 16) * b_pitch)
                    for i in range_constexpr(16)
                ])
            fragment = fx.make_rmem_tensor(16, elem)
            fragment.store(values)
            return fragment

        def compute_tile(stage):
            for ki in range_constexpr(block_k // 32):
                b_fragments = [
                    load_fragment(
                        shared_b, stage, b_stage_size, wave_col + ni * 16, ki * 32,
                        is_k_major=b_k_contiguous,
                    )
                    for ni in range_constexpr(n_steps)
                ]
                rocdl.sched_barrier(0)
                for mi in range_constexpr(m_steps):
                    a_fragment = load_fragment(
                        shared_a, stage, a_stage_size, wave_row + mi * 16, ki * 32
                    )
                    rocdl.sched_barrier(0)
                    for ni in range_constexpr(n_steps):
                        acc = accumulators[mi * n_steps + ni]
                        fx.gemm(mma, acc, a_fragment, b_fragments[ni], acc)

        # Prologue: fill stages - 1 slots, leaving one slot for the next tile.
        for stage in range_constexpr(stages - 1):
            copy_tile(stage, fx.Int32(stage))

        # Two tiles per iteration, matching PeakGemm. Double buffering retains
        # literal LDS offsets; deeper pipelines advance the ring modulo stages.
        main_tiles = k // block_k - (stages - 1)
        current_stage = fx.Int32(0)
        for pair in range(main_tiles // 2):
            for step in range_constexpr(2):
                read_stage = step if stages == 2 else current_stage
                wait_tile(stages - 2)
                copy_tile(
                    (read_stage + stages - 1) % stages, pair * 2 + step + stages - 1
                )
                compute_tile(read_stage)
                current_stage = (current_stage + 1) % stages
        # A one-trip loop keeps the accumulators loop-carried instead of
        # materializing both sides of a predicated remainder branch.
        for remaining in range(main_tiles % 2):
            wait_tile(stages - 2)
            copy_tile((current_stage + stages - 1) % stages, main_tiles + stages - 2)
            compute_tile(current_stage)
            current_stage = (current_stage + 1) % stages

        # Drain: retire the remaining loads in order without issuing more.
        for tail in range_constexpr(stages - 1):
            wait_tile(stages - 2 - tail)
            compute_tile(current_stage)
            current_stage = (current_stage + 1) % stages

        # Reuse the A/B LDS arena for coalesced output stores (PeakGemm cshuffle).
        # Native WMMA output: column=lane%16, row=lane//16*8+register_index.
        fx.gpu.barrier()
        for mi in range_constexpr(m_steps):
            for ni in range_constexpr(n_steps):
                values = accumulators[mi * n_steps + ni].load().to(elem)
                row = wave_row + mi * 16 + lane // 16 * 8
                col = wave_col + ni * 16 + lane % 16
                for i in range_constexpr(8):
                    fx.ptr_store(values[i], shared_a + (row + i) * block_n + col)
        fx.gpu.barrier()

        for i in range_constexpr(block_m * block_n // (threads * 8)):
            offset = (i * threads + tid) * 8
            row, col = offset // block_n, offset % block_n
            values = fx.ptr_load(shared_a + offset, result_type=T.vec(8, elem.ir_type))
            destination = (
                out
                + fx.Int64(row_start + row) * fx.Int64(c_leading_stride)
                + col_start
                + col
            )
            # Interior tiles store 8 contiguous N elements. An edge tile may
            # cover rows or columns past M or N; those elements stay unstored.
            if (row_start + block_m <= m) and (col_start + block_n <= n):
                if const_expr(has_bias):
                    bias_values = fx.ptr_load(
                        bias + col_start + col, result_type=T.vec(8, elem.ir_type)
                    )
                    # Round C to input dtype BEFORE bias, as PeakGemm does.
                    values = (values.to(fx.Float32) + bias_values.to(fx.Float32)).to(elem)
                fx.make_view(destination, fx.make_layout(8, 1)).store(values)
            else:
                for element in range_constexpr(8):
                    if (row_start + row < m) and (col_start + col + element < n):
                        value = values[element]
                        if const_expr(has_bias):
                            bias_value = fx.ptr_load(
                                bias + col_start + col + element, result_type=elem.ir_type
                            )
                            value = (value.to(fx.Float32) + bias_value.to(fx.Float32)).to(elem)
                        fx.ptr_store(value, destination + element)

    @flyc.jit
    def launch(
        a: fx.Pointer,
        b: fx.Pointer,
        out: fx.Pointer,
        bias: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        a_leading_stride: fx.Int32,
        b_leading_stride: fx.Int32,
        c_leading_stride: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(a, b, out, bias, m, n, k, a_leading_stride, b_leading_stride, c_leading_stride).launch(
            grid=((m + block_m - 1) // block_m * ((n + block_n - 1) // block_n), 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch


def gemm_a16w16(
    a,
    b,
    out=None,
    bias=None,
    *,
    block_m=256,
    block_n=256,
    block_k=128,
    stages=2,
    m_waves=4,
    n_waves=4,
    group_m=0,
    stream=None,
    layout="nt",
):
    """C = A[M,K] @ B[K,N]; FP16/BF16 output, optional bias.

    ``layout`` is ``nt`` or ``nn``. N means that dimension is contiguous
    (stride 1); T means the other dimension is contiguous. The other stride
    is the leading stride and may be larger than the contiguous size, so
    padded rows are allowed. C is always row-major. M and N may end on a
    partial tile; K must contain at least ``stages - 1`` complete tiles.
    No split-K or K tail path.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise ValueError("A and B must have the same FP16 or BF16 dtype")
    if a.device != b.device or a.device.type != "cuda":
        raise ValueError("A and B must be on the same ROCm device")
    layout = layout.lower()
    if layout not in ("nn", "nt"):
        raise ValueError("layout must be 'nn' or 'nt'")
    m, k = a.shape
    n = b.shape[1]
    _check_config(m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, layout == "nt")
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
        dtype_id = GEMM_DTYPE_FP16 if a.dtype is torch.float16 else GEMM_DTYPE_BF16
        dtype = fx.Float16 if dtype_id == GEMM_DTYPE_FP16 else fx.BFloat16
        pointers = [
            flyc.from_c_void_p(dtype, t.data_ptr())
            for t in (a, b, out, a if bias is None else bias)
        ]
        launch = build_gemm_gfx1250(
            dtype_id,
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            group_m,
            bias is not None,
            layout == "nt",
        )
        flyc.compile(launch, *pointers, m, n, k, a_leading_stride, b_leading_stride, c_leading_stride, stream)
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
