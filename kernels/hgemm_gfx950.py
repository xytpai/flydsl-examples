import torch

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP
from flydsl._mlir.dialects import llvm

GFX950_DMA_BYTES = 16
GFX950_WAVE_SIZE = 64


@fx.struct
class HGemmGfx950Param:
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    # derived params
    async_load_bytes: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    out_data_bytes: fx.Constexpr[int]
    ldg_x_threads: fx.Constexpr[int]
    block_threads: fx.Constexpr[int]
    ldg_a_iters: fx.Constexpr[int]
    ldg_b_iters: fx.Constexpr[int]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    mma_m_repeat: fx.Constexpr[int]
    mma_n_repeat: fx.Constexpr[int]


def make_hgemm_gfx950_param(
    block_m: int = 256,
    block_n: int = 256,
    block_k: int = 64,
    stages: int = 2,
    m_waves: int = 2,
    n_waves: int = 4,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
) -> HGemmGfx950Param:
    if block_m <= 0 or block_n <= 0 or block_k <= 0 or stages <= 0:
        raise ValueError("block_m, block_n, block_k, and stages must be positive")
    if (mma_m, mma_n, mma_k) != (16, 16, 32):
        raise ValueError("the gfx950 layout kernel currently requires mma=16x16x32")
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0:
        raise ValueError("m_waves, and n_waves must be positive")
    in_dbytes = out_dbytes = 2  # for hgemm
    smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
    arch = get_rocm_arch()
    smem_capacity = SMEM_CAPACITY_MAP.get(arch)
    if smem_capacity is not None and smem_bytes > smem_capacity:
        raise ValueError(
            "staged LDS buffers exceed the device shared-memory capacity: "
            f"stages={stages}, block_m={block_m}, block_n={block_n}, "
            f"block_k={block_k}, smem_bytes={smem_bytes}, "
            f"capacity={smem_capacity} for arch={arch}"
        )
    async_load_vec_size = GFX950_DMA_BYTES // in_dbytes
    ldg_x_threads = block_k // async_load_vec_size
    if ldg_x_threads * async_load_vec_size != block_k:
        raise ValueError(
            "block_k must be divisible by the async load vector size: "
            f"block_k={block_k}, async_load_vec_size={async_load_vec_size}, "
            f"covered_k={ldg_x_threads * async_load_vec_size}"
        )
    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    ldg_a_iters = (block_m * block_k) // (block_threads * async_load_vec_size)
    ldg_b_iters = (block_n * block_k) // (block_threads * async_load_vec_size)
    mma_m_repeat = block_m // m_waves // mma_m
    mma_n_repeat = block_n // n_waves // mma_n
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
    return HGemmGfx950Param(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        async_load_bytes=GFX950_DMA_BYTES,
        in_data_bytes=in_dbytes,
        out_data_bytes=out_dbytes,
        ldg_x_threads=ldg_x_threads,
        block_threads=block_threads,
        ldg_a_iters=ldg_a_iters,
        ldg_b_iters=ldg_b_iters,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
        mma_m_repeat=mma_m_repeat,
        mma_n_repeat=mma_n_repeat,
    )


def make_hgemm_gfx950_kernel_name(param: HGemmGfx950Param):
    name = f"hgemm_t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}"
    name += f"_w{param.m_waves}x{param.n_waves}"
    name += "_nt"
    return name


def __barrier(vmcnt=0):
    llvm.InlineAsmOp(
        None,
        [],
        f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
        "",
        has_side_effects=True,
    )


def buffer_load_lds_inline(rsrc, lds_ptr, global_offset, DMA_BYTES):
    buffer_load_asm_dict = {
        16: "buffer_load_dwordx4",
        8: "buffer_load_dwordx2",
        4: "buffer_load_dword",
    }
    llvm.InlineAsmOp(
        None,
        [
            llvm.IntToPtrOp(
                flydsl._mlir.ir.Type.parse("!llvm.ptr<3>"),
                fx.as_ir_value(fx.ptrtoint(lds_ptr)),
            ).result,
            fx.as_ir_value(global_offset),
            fx.as_ir_value(rsrc),
        ],
        f"s_mov_b32 m0, $0\n\t{buffer_load_asm_dict[DMA_BYTES]} $1, $2, 0 offen sc0 lds",
        "s,v,s",
        has_side_effects=True,
    )


@flyc.kernel
def hgemm_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: HGemmGfx950Param,
):
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    wave_size = GFX950_WAVE_SIZE
    async_load_bytes = param.async_load_bytes
    in_data_bytes = param.in_data_bytes
    async_load_vec_size = async_load_bytes // in_data_bytes
    ldg_x_threads = param.ldg_x_threads
    block_threads = param.block_threads
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters
    mma_m_repeat = param.mma_m_repeat
    mma_n_repeat = param.mma_n_repeat

    tid = fx.thread_idx.x
    bid_m, bid_n, _ = fx.block_idx
    k_tiles = k // block_k

    @fx.struct
    class LayoutGemmSharedStorage:
        a: fx.Array[fx.BFloat16, stages * block_m * block_k, 16]
        b: fx.Array[fx.BFloat16, stages * block_n * block_k, 16]

    lds = fx.SharedAllocator().allocate(LayoutGemmSharedStorage).peek()
    smem_a = lds.a.ptr
    smem_b = lds.b.ptr

    wave_offset = rocdl.readfirstlane(
        T.i64,
        fx.Int64(tid // wave_size * wave_size * async_load_bytes),
    )

    def make_warp_lds_ptr(ptr):
        return fx.recast_iter(fx.Int8, ptr) + wave_offset

    a_rsrc = buffer_ops.create_buffer_resource(a, max_size=True)
    b_rsrc = buffer_ops.create_buffer_resource(b, max_size=True)
    out = fx.rocdl.make_buffer_tensor(out, max_size=False)
    gC = fx.flat_divide(out, (block_m, block_n))[None, None, bid_m, bid_n]

    thr_mma = tiled_mma.thr_slice(tid)
    uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    copy_ab = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    thr_copy_s2r_A = fx.make_tiled_copy_A(copy_ab, tiled_mma).get_slice(tid)
    thr_copy_s2r_B = fx.make_tiled_copy_B(copy_ab, tiled_mma).get_slice(tid)
    thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)

    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))

    def make_lds_layout(rows):
        return fx.make_composed_layout(
            swizzle,
            fx.make_ordered_layout((rows, block_k), (1, 0)),
        )

    a_lds_layout = make_lds_layout(block_m)
    b_lds_layout = make_lds_layout(block_n)

    sA0 = fx.make_view(smem_a, a_lds_layout)
    sB0 = fx.make_view(smem_b, b_lds_layout)
    thr_gC = thr_copy_C.partition_S(gC)

    frag_A = thr_mma.make_fragment_A(sA0)
    frag_B = thr_mma.make_fragment_B(sB0)
    frag_C = thr_mma.make_fragment_C(gC)
    frag_C_bf16 = fx.make_fragment_like(frag_C, fx.BFloat16.ir_type)

    frag_A_retile = thr_copy_s2r_A.retile(frag_A)
    frag_B_retile = thr_copy_s2r_B.retile(frag_B)
    frag_C_retile = thr_copy_C.retile(frag_C_bf16)

    frag_C.fill(0.0)

    def swizzled_col_idx(row, col, layout):
        elem_offset = fx.get_scalar(
            fx.crd2idx(
                (arith.index_cast(T.i32, row), arith.index_cast(T.i32, col)),
                layout,
            )
        )
        return elem_offset % block_k

    def advance_lds_ptr(lds_ptr):
        return lds_ptr + block_threads * async_load_bytes

    def get_a_stage_ptr(stage):
        return fx.add_offset(smem_a, stage * block_m * block_k)

    def get_b_stage_ptr(stage):
        return fx.add_offset(smem_b, stage * block_n * block_k)

    def get_a_stage_view(stage):
        return fx.make_view(get_a_stage_ptr(stage), a_lds_layout)

    def get_b_stage_view(stage):
        return fx.make_view(get_b_stage_ptr(stage), b_lds_layout)

    def async_load_a_to_lds(k_tile, stage):
        lds_ptr = make_warp_lds_ptr(get_a_stage_ptr(stage))
        for i in range_constexpr(ldg_a_iters):
            global_tid = block_threads * i + tid
            m_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_m_idx = bid_m * block_m + m_local_idx
            global_k_idx = k_tile * block_k + swizzled_col_idx(
                m_local_idx,
                k_local_idx,
                a_lds_layout,
            )
            global_offset = (global_m_idx * k + global_k_idx) * in_data_bytes
            global_offset = arith.index_cast(T.i32, global_offset)
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < ldg_a_iters - 1:
                lds_ptr = advance_lds_ptr(lds_ptr)

    def async_load_b_to_lds(k_tile, stage):
        lds_ptr = make_warp_lds_ptr(get_b_stage_ptr(stage))
        for i in range_constexpr(ldg_b_iters):
            global_tid = block_threads * i + tid
            n_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_n_idx = bid_n * block_n + n_local_idx
            global_k_idx = k_tile * block_k + swizzled_col_idx(
                n_local_idx,
                k_local_idx,
                b_lds_layout,
            )
            global_offset = (global_n_idx * k + global_k_idx) * in_data_bytes
            global_offset = arith.index_cast(T.i32, global_offset)
            buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < ldg_b_iters - 1:
                lds_ptr = advance_lds_ptr(lds_ptr)

    def compute_stage(read_stage):
        thr_sA_s2r = thr_copy_s2r_A.partition_S(get_a_stage_view(read_stage))
        thr_sB_s2r = thr_copy_s2r_B.partition_S(get_b_stage_view(read_stage))
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            fx.copy(
                uni_copy,
                thr_sB_s2r[None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
            fx.copy(
                uni_copy,
                thr_sA_s2r[None, None, block_k_iter],
                frag_A_retile[None, None, block_k_iter],
            )
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, block_k_iter],
                frag_B[None, None, block_k_iter],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

    LDG_WAIT_COUNT = ldg_a_iters + ldg_b_iters

    for stage in range_constexpr(stages - 1):
        async_load_b_to_lds(stage, stage)
        async_load_a_to_lds(stage, stage)
    rocdl.sched_barrier(0)

    main_loop_end = k_tiles - (stages - 1)
    for k_tile in range(0, main_loop_end, 1):
        current_stage = k_tile % stages
        write_stage = (current_stage + stages - 1) % stages
        __barrier((stages - 2) * LDG_WAIT_COUNT)
        async_load_b_to_lds(k_tile + (stages - 1), write_stage)
        async_load_a_to_lds(k_tile + (stages - 1), write_stage)
        compute_stage(current_stage)

    current_stage = main_loop_end % stages
    for s in range_constexpr(0, stages - 1):
        __barrier((stages - 2 - s) * LDG_WAIT_COUNT)
        compute_stage(current_stage)
        current_stage = (current_stage + 1) % stages

    frag_C_bf16.store(fx.Vector(frag_C.load()).to(fx.BFloat16).ir_value())
    fx.copy(copy_c, frag_C_retile, thr_gC)


@flyc.jit
def launch_hgemm_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    param: HGemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    mma_atom = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, fx.BFloat16)
    )
    k_per_mfma_group = param.mma_k // 4
    tiled_mma = fx.make_tiled_mma(
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
    num_pid_m = (m + param.block_m - 1) // param.block_m
    num_pid_n = (n + param.block_n - 1) // param.block_n
    hgemm_gfx950_kernel._known_block_size = [param.block_threads, 1, 1]
    hgemm_gfx950_kernel._func.__name__ = make_hgemm_gfx950_kernel_name(param)
    hgemm_gfx950_kernel(out, a, b, m, n, k, tiled_mma, param).launch(
        grid=(num_pid_m, num_pid_n, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


def layout_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None = None,
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    stages: int = 5,
    m_waves: int = 4,
    n_waves: int = 4,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Compute C = A @ B.T with the FlyDSL layout-system tiled MMA path."""

    if c is None:
        c = torch.empty((a.shape[0], b.shape[0]), dtype=a.dtype, device=a.device)
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise ValueError("layout_gemm expects 2D tensors")
    if not (a.is_cuda and b.is_cuda and c.is_cuda):
        raise ValueError("layout_gemm expects CUDA tensors")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("layout_gemm currently supports bf16 A/B only")
    if c.dtype != a.dtype:
        raise TypeError("layout_gemm expects C to have the same dtype as A/B")
    if not a.is_contiguous() or not b.is_contiguous() or not c.is_contiguous():
        raise ValueError("layout_gemm expects contiguous row-major tensors")
    m, k = a.shape
    n, kb = b.shape
    if kb != k:
        raise ValueError(
            f"inner dimensions must match, got A.shape={a.shape}, B.shape={b.shape}"
        )
    if c.shape != (m, n):
        raise ValueError(f"C must have shape {(m, n)}, got {tuple(c.shape)}")
    if m % block_m or n % block_n or k % block_k:
        raise ValueError(
            f"M/N/K must be multiples of {(block_m, block_n, block_k)}, got {(m, n, k)}"
        )
    effective_stages = min(stages, k // block_k)
    if effective_stages < 2:
        raise ValueError(
            f"K must contain at least 2 block_k tiles for the staged pipeline, got K={k}, block_k={block_k}"
        )

    launch_stream = torch.cuda.current_stream(a.device) if stream is None else stream
    param = make_hgemm_gfx950_param(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=effective_stages,
        m_waves=m_waves,
        n_waves=n_waves,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
    )
    launch_hgemm_gfx950(
        c,
        a,
        b,
        a.shape[0],
        b.shape[0],
        a.shape[1],
        param,
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
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    stages: int = 4,
    m_waves: int = 4,
    n_waves: int = 4,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
    profiler_row_limit: int = 20,
) -> dict[str, float]:
    """Run a profiler-backed benchmark for the layout-system GEMM kernel."""

    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((m, n), dtype=a.dtype, device="cuda")
    stream = torch.cuda.current_stream()

    layout_gemm(
        a,
        b,
        c,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
        stream=stream,
    )
    torch.cuda.synchronize()

    for _ in range(warmup):
        layout_gemm(
            a,
            b,
            c,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            mma_m=mma_m,
            mma_n=mma_n,
            mma_k=mma_k,
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
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            mma_m=mma_m,
            mma_n=mma_n,
            mma_k=mma_k,
            stream=stream,
        )
    end.record(stream)
    end.synchronize()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as prof:
        for _ in range(iters):
            layout_gemm(
                a,
                b,
                c,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                stages=stages,
                m_waves=m_waves,
                n_waves=n_waves,
                mma_m=mma_m,
                mma_n=mma_n,
                mma_k=mma_k,
                stream=stream,
            )
        torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / iters
    tflops = (2.0 * m * n * k) / (avg_ms / 1e3) / 1e12
    result = {
        "m": float(m),
        "n": float(n),
        "k": float(k),
        "avg_ms": avg_ms,
        "tflops": tflops,
        "block_m": float(block_m),
        "block_n": float(block_n),
        "block_k": float(block_k),
        "stages": float(stages),
        "m_waves": float(m_waves),
        "n_waves": float(n_waves),
        "mma_m": float(mma_m),
        "mma_n": float(mma_n),
        "mma_k": float(mma_k),
    }
    print(
        "layout_gemm "
        f"tile={block_m}x{block_n}x{block_k}x{stages} "
        f"waves={m_waves}x{n_waves} M={m} N={n} K={k}: "
        f"{avg_ms:.4f} ms, {tflops:.2f} TFLOP/s"
    )
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=profiler_row_limit,
        )
    )
    return result
