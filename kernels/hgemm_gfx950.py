import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
STAGES_A = 2
WARP_SIZE = 64
DEFAULT_BLOCK_M_WARPS = 2
DEFAULT_BLOCK_N_WARPS = 4
DMA_BYTES = 16
IN_DTYPE_BYTES = 2
LDG_ASYNC_VEC_SIZE = DMA_BYTES // IN_DTYPE_BYTES
LDG_X_THREADS = BLOCK_K // LDG_ASYNC_VEC_SIZE


@fx.struct
class _LayoutGemmSharedStorage:
    a0: fx.Array[fx.Float16, BLOCK_M * BLOCK_K, 16]
    a1: fx.Array[fx.Float16, BLOCK_M * BLOCK_K, 16]
    b0: fx.Array[fx.Float16, BLOCK_N * BLOCK_K, 16]
    b1: fx.Array[fx.Float16, BLOCK_N * BLOCK_K, 16]


@flyc.kernel
def hgemm_wmma_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    k_tiles: fx.Constexpr[int],
    block_threads: fx.Constexpr[int],
    ldg_a_iters: fx.Constexpr[int],
    ldg_b_iters: fx.Constexpr[int],
    tiled_mma: fx.TiledMma,
):
    tid = fx.thread_idx.x
    bid_m, bid_n, _ = fx.block_idx

    a_bytes = fx.rocdl.make_buffer_tensor(
        fx.Tensor(
            fx.make_view(
                fx.recast_iter(fx.Int8, fx.get_iter(A)),
                fx.make_layout(65536 * k_tiles * BLOCK_K * IN_DTYPE_BYTES, 1),
            )
        ),
        max_size=True,
    )
    b_bytes = fx.rocdl.make_buffer_tensor(
        fx.Tensor(
            fx.make_view(
                fx.recast_iter(fx.Int8, fx.get_iter(B)),
                fx.make_layout(65536 * k_tiles * BLOCK_K * IN_DTYPE_BYTES, 1),
            )
        ),
        max_size=True,
    )
    a_bytes = fx.logical_divide(a_bytes, fx.make_layout(1, 1))
    b_bytes = fx.logical_divide(b_bytes, fx.make_layout(1, 1))
    C = fx.rocdl.make_buffer_tensor(C, max_size=False)

    gC = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_m, bid_n]

    thr_mma = tiled_mma.thr_slice(tid)

    uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float16)
    copy_ab = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float16)
    copy_g2s = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

    thr_copy_s2r_A = fx.make_tiled_copy_A(copy_ab, tiled_mma).get_slice(tid)
    thr_copy_s2r_B = fx.make_tiled_copy_B(copy_ab, tiled_mma).get_slice(tid)
    thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)

    lds = fx.SharedAllocator().allocate(_LayoutGemmSharedStorage).peek()
    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))

    def make_lds_view(ptr, rows):
        return fx.make_view(
            ptr,
            fx.make_composed_layout(
                swizzle,
                fx.make_ordered_layout((rows, BLOCK_K), (1, 0)),
            ),
        )

    sA_stages = [
        make_lds_view(lds.a0.ptr, BLOCK_M),
        make_lds_view(lds.a1.ptr, BLOCK_M),
    ]
    sB_stages = [
        make_lds_view(lds.b0.ptr, BLOCK_N),
        make_lds_view(lds.b1.ptr, BLOCK_N),
    ]

    thr_sA_s2r_stages = [thr_copy_s2r_A.partition_S(sA) for sA in sA_stages]
    thr_sB_s2r_stages = [thr_copy_s2r_B.partition_S(sB) for sB in sB_stages]
    thr_gC = thr_copy_C.partition_S(gC)

    frag_A = thr_mma.make_fragment_A(sA_stages[0])
    frag_B = thr_mma.make_fragment_B(sB_stages[0])
    frag_C = thr_mma.make_fragment_C(gC)

    frag_A_retile = thr_copy_s2r_A.retile(frag_A)
    frag_B_retile = thr_copy_s2r_B.retile(frag_B)
    frag_C_retile = thr_copy_C.retile(frag_C)

    frag_C.fill(0.0)

    warp_offset = rocdl.readfirstlane(
        fx.Int32.ir_type,
        fx.Int32((tid // WARP_SIZE) * WARP_SIZE * DMA_BYTES),
    )

    def make_warp_lds_ptr(ptr):
        return fx.add_offset(fx.recast_iter(fx.Int8, ptr), warp_offset)

    a_lds_ptrs = [
        make_warp_lds_ptr(lds.a0.ptr),
        make_warp_lds_ptr(lds.a1.ptr),
    ]
    b_lds_ptrs = [
        make_warp_lds_ptr(lds.b0.ptr),
        make_warp_lds_ptr(lds.b1.ptr),
    ]

    k_blocks16 = (BLOCK_K * IN_DTYPE_BYTES) // 16

    def swizzle_col_bytes(row, col_in_bytes):
        return col_in_bytes ^ ((row % k_blocks16) * 16)

    def advance_lds_ptr(lds_ptr):
        return fx.add_offset(lds_ptr, block_threads * DMA_BYTES)

    def async_load_a_to_lds(k_tile, stage):
        lds_ptr = a_lds_ptrs[stage]
        for i in range_constexpr(ldg_a_iters):
            global_tid = fx.Index(block_threads * i) + fx.Index(tid)
            m_local_idx = global_tid // LDG_X_THREADS
            k_local_idx = global_tid % LDG_X_THREADS * LDG_ASYNC_VEC_SIZE
            col_in_bytes = swizzle_col_bytes(m_local_idx, k_local_idx * IN_DTYPE_BYTES)
            global_m_idx = fx.Index(bid_m) * BLOCK_M + m_local_idx
            global_k_idx = fx.Index(k_tile * BLOCK_K) + col_in_bytes // IN_DTYPE_BYTES
            global_offset = (
                global_m_idx * (k_tiles * BLOCK_K) + global_k_idx
            ) * IN_DTYPE_BYTES
            src = fx.slice(a_bytes, (None, arith.index_cast(T.i32, global_offset)))
            dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
            fx.copy(copy_g2s, src, dst)
            if i < ldg_a_iters - 1:
                lds_ptr = advance_lds_ptr(lds_ptr)

    def async_load_b_to_lds(k_tile, stage):
        lds_ptr = b_lds_ptrs[stage]
        for i in range_constexpr(ldg_b_iters):
            global_tid = fx.Index(block_threads * i) + fx.Index(tid)
            n_local_idx = global_tid // LDG_X_THREADS
            k_local_idx = global_tid % LDG_X_THREADS * LDG_ASYNC_VEC_SIZE
            col_in_bytes = swizzle_col_bytes(n_local_idx, k_local_idx * IN_DTYPE_BYTES)
            global_n_idx = fx.Index(bid_n) * BLOCK_N + n_local_idx
            global_k_idx = fx.Index(k_tile * BLOCK_K) + col_in_bytes // IN_DTYPE_BYTES
            global_offset = (
                global_n_idx * (k_tiles * BLOCK_K) + global_k_idx
            ) * IN_DTYPE_BYTES
            src = fx.slice(b_bytes, (None, arith.index_cast(T.i32, global_offset)))
            dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
            fx.copy(copy_g2s, src, dst)
            if i < ldg_b_iters - 1:
                lds_ptr = advance_lds_ptr(lds_ptr)

    def hot_loop_scheduler_tile(read_next=True):
        if read_next:
            for _ in range_constexpr(ldg_a_iters):
                rocdl.sched_vmem(1)  # async_load_a_to_lds next
            for _ in range_constexpr(ldg_b_iters):
                rocdl.sched_vmem(1)  # async_load_b_to_lds next
        for _ in range_constexpr(BLOCK_K // 32):
            rocdl.sched_dsrd(2)  # LDS -> frag_A and LDS -> frag_B
            rocdl.sched_mfma(1)  # one tiled_mma k32 step
        rocdl.sched_barrier(0)

    def hot_loop_scheduler_double_tile(second_read_next=True):
        hot_loop_scheduler_tile(read_next=True)
        hot_loop_scheduler_tile(read_next=second_read_next)

    def run_pipeline_stage(read_stage, next_k=None, read_next=True):
        write_stage = read_stage ^ 1
        if read_next:
            async_load_a_to_lds(next_k, write_stage)
            async_load_b_to_lds(next_k, write_stage)

        for block_k_iter in range_constexpr(BLOCK_K // 32):
            fx.copy(
                uni_copy,
                thr_sA_s2r_stages[read_stage][None, None, block_k_iter],
                frag_A_retile[None, None, block_k_iter],
            )
            fx.copy(
                uni_copy,
                thr_sB_s2r_stages[read_stage][None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, block_k_iter],
                frag_B[None, None, block_k_iter],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

        if read_next:
            rocdl.s_waitcnt(0)
            fx.gpu.barrier()

    async_load_a_to_lds(0, 0)
    async_load_b_to_lds(0, 0)
    for _ in range_constexpr(ldg_a_iters + ldg_b_iters):
        rocdl.sched_vmem(1)
    rocdl.sched_barrier(0)
    # rocdl.s_waitcnt(0)
    fx.gpu.barrier()

    for k_tile in range(0, k_tiles - 2, 2):
        run_pipeline_stage(read_stage=0, next_k=k_tile + 1)
        run_pipeline_stage(read_stage=1, next_k=k_tile + 2)
        # hot_loop_scheduler_double_tile()

    run_pipeline_stage(read_stage=0, next_k=k_tiles - 1)
    run_pipeline_stage(read_stage=1, read_next=False)
    # hot_loop_scheduler_double_tile(second_read_next=False)

    fx.copy(copy_c, frag_C_retile, thr_gC)


@flyc.jit
def _layout_gemm_f16_f32(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    m: fx.Constexpr[int],
    n: fx.Constexpr[int],
    k: fx.Constexpr[int],
    block_m_warps: fx.Constexpr[int],
    block_n_warps: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float16))
    block_threads = block_m_warps * block_n_warps * WARP_SIZE
    ldg_a_iters = (BLOCK_M * BLOCK_K) // (block_threads * LDG_ASYNC_VEC_SIZE)
    ldg_b_iters = (BLOCK_N * BLOCK_K) // (block_threads * LDG_ASYNC_VEC_SIZE)
    atom_layout_stride_m = 0 if block_m_warps == 1 else 1
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout(
            (block_m_warps, block_n_warps, 1),
            (atom_layout_stride_m, block_m_warps, 0),
        ),
        fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),
    )
    hgemm_wmma_kernel._known_block_size = [block_threads, 1, 1]
    hgemm_wmma_kernel(
        A, B, C, k // BLOCK_K, block_threads, ldg_a_iters, ldg_b_iters, tiled_mma
    ).launch(
        grid=(m // BLOCK_M, n // BLOCK_N, 1),
        block=(block_threads, 1, 1),
        stream=stream,
    )


def _check_layout_gemm_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    block_m_warps: int,
    block_n_warps: int,
):
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise ValueError("layout_gemm expects 2D tensors")
    if not (a.is_cuda and b.is_cuda and c.is_cuda):
        raise ValueError("layout_gemm expects CUDA tensors")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("layout_gemm currently supports fp16 A/B only")
    if c.dtype != torch.float32:
        raise TypeError("layout_gemm currently writes fp32 C")

    m, k = a.shape
    n, kb = b.shape
    if kb != k:
        raise ValueError(
            f"inner dimensions must match, got A.shape={a.shape}, B.shape={b.shape}"
        )
    if c.shape != (m, n):
        raise ValueError(f"C must have shape {(m, n)}, got {tuple(c.shape)}")
    if not a.is_contiguous() or not b.is_contiguous() or not c.is_contiguous():
        raise ValueError("layout_gemm expects contiguous row-major tensors")
    if m % BLOCK_M or n % BLOCK_N or k % BLOCK_K:
        raise ValueError(
            f"M/N/K must be multiples of {(BLOCK_M, BLOCK_N, BLOCK_K)}, got {(m, n, k)}"
        )
    if k // BLOCK_K < STAGES_A:
        raise ValueError(f"K must cover at least {STAGES_A} K tiles for ping-pong")
    if block_m_warps <= 0 or block_n_warps <= 0:
        raise ValueError("block_m_warps and block_n_warps must be positive")
    block_threads = block_m_warps * block_n_warps * WARP_SIZE
    if block_threads > 1024:
        raise ValueError(f"too many block threads: {block_threads}")
    if block_threads not in (64, 128, 256, 512, 1024):
        raise ValueError(
            "unsupported block_m_warps * block_n_warps; "
            "supported total waves are 1, 2, 4, 8, or 16"
        )
    if (BLOCK_M * BLOCK_K) % (block_threads * LDG_ASYNC_VEC_SIZE):
        raise ValueError(
            "block_m_warps * block_n_warps does not evenly cover A async loads"
        )
    if (BLOCK_N * BLOCK_K) % (block_threads * LDG_ASYNC_VEC_SIZE):
        raise ValueError(
            "block_m_warps * block_n_warps does not evenly cover B async loads"
        )


def layout_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None = None,
    *,
    block_m_warps: int = DEFAULT_BLOCK_M_WARPS,
    block_n_warps: int = DEFAULT_BLOCK_N_WARPS,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Compute C = A @ B.T with the FlyDSL layout-system tiled MMA path.

    This intentionally keeps the public wrapper narrow: fp16 inputs, fp32 output,
    row-major contiguous tensors, and tile-aligned dimensions.
    """

    if c is None:
        c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.float32, device=a.device)
    _check_layout_gemm_inputs(a, b, c, block_m_warps, block_n_warps)

    launch_stream = torch.cuda.current_stream(a.device) if stream is None else stream
    _layout_gemm_f16_f32(
        a,
        b,
        c,
        a.shape[0],
        b.shape[0],
        a.shape[1],
        block_m_warps,
        block_n_warps,
        stream=launch_stream,
    )
    return c


def benchmark_layout_gemm(
    m: int = 8192,
    n: int = 8192,
    k: int = 8192,
    *,
    warmup: int = 10,
    iters: int = 50,
    block_m_warps: int = DEFAULT_BLOCK_M_WARPS,
    block_n_warps: int = DEFAULT_BLOCK_N_WARPS,
) -> dict[str, float]:
    """Run a simple CUDA-event benchmark for the layout-system GEMM kernel."""

    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((n, k), dtype=torch.float16, device="cuda")
    c = torch.empty((m, n), dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()

    layout_gemm(
        a, b, c, block_m_warps=block_m_warps, block_n_warps=block_n_warps, stream=stream
    )
    torch.cuda.synchronize()

    for _ in range(warmup):
        layout_gemm(
            a,
            b,
            c,
            block_m_warps=block_m_warps,
            block_n_warps=block_n_warps,
            stream=stream,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iters):
        layout_gemm(
            a,
            b,
            c,
            block_m_warps=block_m_warps,
            block_n_warps=block_n_warps,
            stream=stream,
        )
    end.record(stream)
    end.synchronize()

    avg_ms = start.elapsed_time(end) / iters
    tflops = (2.0 * m * n * k) / (avg_ms / 1e3) / 1e12
    result = {
        "m": float(m),
        "n": float(n),
        "k": float(k),
        "avg_ms": avg_ms,
        "tflops": tflops,
        "block_m_warps": float(block_m_warps),
        "block_n_warps": float(block_n_warps),
    }
    print(
        "layout_gemm "
        f"tile={BLOCK_M}x{BLOCK_N}x{BLOCK_K} "
        f"warps={block_m_warps}x{block_n_warps} M={m} N={n} K={k}: "
        f"{avg_ms:.4f} ms, {tflops:.2f} TFLOP/s"
    )
    return result
