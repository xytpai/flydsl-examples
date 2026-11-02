# SPDX-License-Identifier: MIT
"""PeakGemm-style wave32 TDM GEMM: NT layout, full tiles, no split-K."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl, tdm_ops
from flydsl.expr.typing import T

WAVE_SIZE = 32
LDS_PAD = 8


def _check_config(
    m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m
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
    shared_bytes = 2 * max(
        stages * (block_m + block_n) * (block_k + LDS_PAD), block_m * block_n
    )
    if shared_bytes > 320 * 1024:
        raise ValueError("pipeline exceeds gfx1250's 320 KiB LDS capacity")
    if m % block_m or n % block_n or k % block_k:
        raise ValueError("M, N and K must be block-tile aligned")
    if k < (stages - 1) * block_k:
        raise ValueError("K needs at least stages - 1 tiles for the prefetch prologue")
    if max(m * k, n * k, m * n, group_m * (n // block_n)) > (2**32 - 1) // 2:
        raise ValueError("tensor or block offsets exceed the supported 32-bit range")


@functools.lru_cache(maxsize=128)
def _build_kernel(
    is_fp16, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias
):
    elem = fx.Float16 if is_fp16 else fx.BFloat16
    waves = m_waves * n_waves
    threads = waves * WAVE_SIZE
    pitch = block_k + LDS_PAD
    a_stage_size, b_stage_size = block_m * pitch, block_n * pitch
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
    ):
        rocdl.disable_xdl_arb_stall()
        tid = fx.thread_idx.x
        wave = rocdl.wave_id()
        lane = tid % WAVE_SIZE
        block_id = fx.block_idx.x
        m_blocks, n_blocks = m // block_m, n // block_n
        if const_expr(group_m == 0):
            block_row, block_col = block_id // n_blocks, block_id % n_blocks
        else:
            group_size = group_m * n_blocks
            first_row = block_id // group_size * group_m
            rows_in_group = fx.min(m_blocks - first_row, group_m)
            within_group = block_id % group_size
            block_row = first_row + within_group % rows_in_group
            block_col = within_group // rows_in_group
        row_start, col_start = block_row * block_m, block_col * block_n
        wave_row = wave // n_waves * (block_m // m_waves)
        wave_col = wave % n_waves * (block_n // n_waves)

        storage = fx.SharedAllocator(static=True).allocate(SharedStorage).peek()
        shared_a = storage.data.ptr
        shared_b = shared_a + stages * a_stage_size

        # Describe the entire K range once. TDM splits disjoint rows across waves
        # and derives the 8-element LDS padding from the destination layout.
        def tdm_operand(pointer, start, rows):
            global_view = fx.make_view(
                pointer + fx.Int64(start) * fx.Int64(k),
                fx.make_layout((rows, k), (k, 1)),
            )
            tile = (rows, block_k)
            atom, coords = rocdl.cdna5.make_tiled_tdm_atom(
                rocdl.TensorLoad(),
                global_view,
                fx.make_layout(tile, (pitch, 1)),
                tile,
                num_warps=waves,
            )
            return atom, fx.zipped_divide(coords, tile)[None, (0, None)]

        atom_a, coords_a = tdm_operand(a, row_start, block_m)
        atom_b, coords_b = tdm_operand(b, col_start, block_n)

        def copy_operand(atom, coords, shared, rows, stage, tile_index):
            shared_view = fx.make_view(
                shared + stage * rows * pitch,
                fx.make_layout(((rows, block_k), 1), ((pitch, 1), rows * pitch)),
            )
            dst, src = rocdl.cdna5.tdm_partition(
                atom,
                wave,
                fx.make_layout(waves, 1),
                shared_view,
                coords,
            )
            fx.copy(atom, src[None, tile_index], dst[None, 0])

        def copy_tile(stage, tile_index):
            copy_operand(atom_b, coords_b, shared_b, block_n, stage, tile_index)
            copy_operand(atom_a, coords_a, shared_a, block_m, stage, tile_index)
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

        def load_fragment(shared, stage, stage_size, row, column):
            offset = (
                stage * stage_size + (row + lane % 16) * pitch + column + lane // 16 * 8
            )
            low = fx.ptr_load(shared + offset, result_type=T.vec(8, elem.ir_type))
            high = fx.ptr_load(shared + offset + 16, result_type=T.vec(8, elem.ir_type))
            fragment = fx.make_rmem_tensor(16, elem)
            fragment.store(low.shuffle(high, list(range(16))))
            return fragment

        def compute_tile(stage):
            for ki in range_constexpr(block_k // 32):
                b_fragments = [
                    load_fragment(
                        shared_b, stage, b_stage_size, wave_col + ni * 16, ki * 32
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
        if main_tiles % 2 != 0:
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
            if const_expr(has_bias):
                bias_values = fx.ptr_load(
                    bias + col_start + col, result_type=T.vec(8, elem.ir_type)
                )
                # Round C to input dtype BEFORE bias, as PeakGemm does.
                values = (values.to(fx.Float32) + bias_values.to(fx.Float32)).to(elem)
            destination = (
                out + fx.Int64(row_start + row) * fx.Int64(n) + col_start + col
            )
            fx.make_view(destination, fx.make_layout(8, 1)).store(values)

    @flyc.jit
    def launch(
        a: fx.Pointer,
        b: fx.Pointer,
        out: fx.Pointer,
        bias: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(a, b, out, bias, m, n, k).launch(
            grid=(m // block_m * (n // block_n), 1, 1),
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
):
    """C = A[M,K] @ B[K,N], NT storage; FP16/BF16 output, optional bias.

    A and C are contiguous. B is a transposed contiguous [N,K] tensor.
    Full tiles only; K must contain at least stages - 1 complete tiles.
    Unlike the gfx950 example, this first port has no split-K or tail path.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise ValueError("A and B must have the same FP16 or BF16 dtype")
    if a.device != b.device or a.device.type != "cuda":
        raise ValueError("A and B must be on the same ROCm device")
    m, k = a.shape
    n = b.shape[1]
    _check_config(m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m)
    if not a.is_contiguous() or b.stride() != (1, k):
        raise ValueError(
            "NT layout requires contiguous A and B = contiguous_weights.t()"
        )
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if (
        out.shape != (m, n)
        or out.dtype != a.dtype
        or out.device != a.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous [M,N] with the input dtype and device")
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
    lo, hi = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
    if any(
        lo < t.data_ptr() + t.numel() * t.element_size() and t.data_ptr() < hi
        for t in [a, b] + ([] if bias is None else [bias])
    ):
        raise ValueError("out must not overlap the inputs")
    with torch.cuda.device(a.device):
        arch = torch.cuda.get_device_properties(a.device).gcnArchName.split(":")[0]
        if arch != "gfx1250":
            raise ValueError(f"TDM GEMM requires gfx1250, got {arch}")
        if stream is None:
            stream = torch.cuda.current_stream()
        if stream.device != a.device:
            raise ValueError("stream must belong to the input device")
        dtype = fx.Float16 if a.dtype == torch.float16 else fx.BFloat16
        pointers = [
            flyc.from_c_void_p(dtype, t.data_ptr())
            for t in (a, b, out, a if bias is None else bias)
        ]
        launch = _build_kernel(
            a.dtype == torch.float16,
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            group_m,
            bias is not None,
        )
        flyc.compile(launch, *pointers, m, n, k, stream)
    return out
