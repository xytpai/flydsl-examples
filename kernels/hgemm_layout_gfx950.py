import torch
from typing import Optional

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch
from flydsl._mlir.dialects import llvm

from .hgemm_wmma_gfx950_utils import BlockSwizzle

GFX950_DMA_BYTES = 16
GFX950_WAVE_SIZE = 64
HGEMM_DTYPE_BF16 = 2
HGEMM_DTYPE_FP16 = 3


@fx.struct
class HGemmGfx950Param:
    dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    has_bias: fx.Constexpr[bool]
    has_k_tail: fx.Constexpr[bool]
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


def make_hgemm_gfx950_param(
    dtype_id: int = HGEMM_DTYPE_BF16,
    block_m: int = 256,
    block_n: int = 256,
    block_k: int = 64,
    stages: int = 2,
    m_waves: int = 2,
    n_waves: int = 4,
    group_m: int = 0,
    has_bias: bool = False,
    has_k_tail: bool = False,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
) -> HGemmGfx950Param:
    if dtype_id not in (HGEMM_DTYPE_BF16, HGEMM_DTYPE_FP16):
        raise ValueError(f"unsupported dtype_id={dtype_id}")
    if block_m <= 0 or block_n <= 0 or block_k <= 0 or stages <= 0:
        raise ValueError("block_m, block_n, block_k, and stages must be positive")
    if (mma_m, mma_n, mma_k) != (16, 16, 32):
        raise ValueError("the gfx950 layout kernel currently requires mma=16x16x32")
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0:
        raise ValueError("m_waves, and n_waves must be positive")
    if group_m < 0:
        raise ValueError("group_m must be non-negative")
    in_dbytes = out_dbytes = 2  # for hgemm
    smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
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
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        has_bias=has_bias,
        has_k_tail=has_k_tail,
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
    )


def make_hgemm_gfx950_kernel_name(param: HGemmGfx950Param):
    dtype_str = "fp16" if param.dtype_id == HGEMM_DTYPE_FP16 else "bf16"
    name = f"hgemm_{dtype_str}_t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}"
    name += f"_w{param.m_waves}x{param.n_waves}"
    name += f"_gm{param.group_m}"
    name += f"_bias{int(param.has_bias)}"
    name += f"_ktail{int(param.has_k_tail)}"
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
    bias: fx.Tensor,
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
    has_k_tail = param.has_k_tail
    async_load_bytes = param.async_load_bytes
    in_data_bytes = param.in_data_bytes
    async_load_vec_size = async_load_bytes // in_data_bytes
    ldg_x_threads = param.ldg_x_threads
    block_threads = param.block_threads
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters
    ldg_wait_count = ldg_a_iters + ldg_b_iters
    elem_dtype = (
        fx.Float16 if const_expr(param.dtype_id == HGEMM_DTYPE_FP16) else fx.BFloat16
    )

    tid = fx.thread_idx.x
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    k_tiles = (k + block_k - 1) // block_k

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[elem_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    # Wrap tensor arguments as ROCDL buffer tensors so later code can use
    # buffer resources/copy atoms while preserving the original tensor layout.
    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out = fx.rocdl.make_buffer_tensor(out, max_size=False)
    if const_expr(param.has_bias):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

    # Extract raw buffer descriptors for the inline global-to-LDS load path.
    a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(a_buf))
    b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(b_buf))

    s2r_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    g2r_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    r2g_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)

    gC = fx.flat_divide(out, (block_m, block_n))[None, None, bid_m, bid_n]

    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(g2r_copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(g2r_copy_atom, tiled_mma).get_slice(tid)

    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))

    def make_lds_layout(rows):
        return fx.make_composed_layout(
            swizzle,
            fx.make_ordered_layout((rows, block_k), (1, 0)),
        )

    a_lds_layout = make_lds_layout(block_m)
    b_lds_layout = make_lds_layout(block_n)
    c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))

    sA = fx.make_view(smem_a, a_lds_layout)
    sB = fx.make_view(smem_b, b_lds_layout)
    sC = fx.make_view(smem_c, c_lds_layout)

    frag_A = thr_mma.make_fragment_A(sA)
    frag_B = thr_mma.make_fragment_B(sB)
    frag_C = thr_mma.make_fragment_C(gC)

    # `retile` does not allocate new data; it reinterprets the MMA register
    # fragments with the tiled-copy layout so LDS-to-register `fx.copy` can fill them.
    frag_A_retile = thr_copy_A.retile(frag_A)
    frag_B_retile = thr_copy_B.retile(frag_B)

    row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
    thr_mma_cRow = thr_mma.partition_C(row_coords)
    thr_mma_cCol = thr_mma.partition_C(col_coords)

    cshuffle_vec_size = GFX950_DMA_BYTES // param.out_data_bytes
    cshuffle_x_threads = block_n // cshuffle_vec_size
    cshuffle_thr_layout = fx.make_layout(
        (block_threads // cshuffle_x_threads, cshuffle_x_threads),
        (cshuffle_x_threads, 1),
    )
    cshuffle_val_layout = fx.make_layout((1, cshuffle_vec_size), (1, 1))
    cshuffle_tile, cshuffle_tv_layout = fx.make_layout_tv(
        cshuffle_thr_layout,
        cshuffle_val_layout,
    )
    tiled_copy_cshuffle = fx.make_tiled_copy(
        r2g_copy_atom,
        cshuffle_tv_layout,
        cshuffle_tile,
    )
    thr_copy_cshuffle = tiled_copy_cshuffle.get_slice(tid)
    thr_sC = thr_copy_cshuffle.partition_S(sC)
    thr_gC = thr_copy_cshuffle.partition_D(gC)
    thr_cRow = thr_copy_cshuffle.partition_S(row_coords)[(0, None), None, None]
    thr_cCol = thr_copy_cshuffle.partition_S(col_coords)[(0, None), None, None]
    frag_C_cshuffle = fx.make_fragment_like(thr_sC)
    pred_C = fx.make_fragment_like(thr_cRow, dtype=fx.Boolean)

    frag_C.fill(0.0)
    if const_expr(param.has_bias):
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            col_idx = fx.get_scalar(thr_mma_cCol[i])
            global_n_idx = bid_n * block_n + col_idx
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            bias_val = bias_buf[safe_global_n_idx].to(fx.Float32)
            frag_C[i] = bias_val

    for i in range_constexpr(fx.size(pred_C.shape).unpack()):
        row_idx = bid_m * block_m + fx.get_scalar(thr_cRow[i])
        col_idx = bid_n * block_n + fx.get_scalar(thr_cCol[i])
        pred_C[i] = (row_idx < m) & (col_idx < n)

    wave_offset = rocdl.readfirstlane(
        fx.Int64.ir_type,
        fx.Int64(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * async_load_bytes),
    )

    def make_wave_lds_ptr(ptr):
        return fx.recast_iter(fx.Int8, ptr) + fx.Int32(wave_offset)

    def swizzled_col_idx(row, col, layout):
        elem_offset = fx.get_scalar(
            fx.crd2idx(
                (row, col),
                layout,
            )
        )
        return elem_offset % block_k

    def async_load_a_to_lds(k_tile, stage):
        lds_ptr = make_wave_lds_ptr(smem_a + stage * block_m * block_k)
        for i in range_constexpr(ldg_a_iters):
            global_tid = block_threads * i + tid
            m_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_m_idx = bid_m * block_m + m_local_idx
            safe_global_m_idx = (global_m_idx < m).select(global_m_idx, 0)
            global_k_idx = k_tile * block_k + swizzled_col_idx(
                m_local_idx,
                k_local_idx,
                a_lds_layout,
            )
            if const_expr(has_k_tail):
                safe_global_k_idx = (global_k_idx < k).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            global_offset = (safe_global_m_idx * k + safe_global_k_idx) * in_data_bytes
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < ldg_a_iters - 1:
                lds_ptr = lds_ptr + block_threads * async_load_bytes

    def async_load_b_to_lds(k_tile, stage):
        lds_ptr = make_wave_lds_ptr(smem_b + stage * block_n * block_k)
        for i in range_constexpr(ldg_b_iters):
            global_tid = block_threads * i + tid
            n_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_n_idx = bid_n * block_n + n_local_idx
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            global_k_idx = k_tile * block_k + swizzled_col_idx(
                n_local_idx,
                k_local_idx,
                b_lds_layout,
            )
            if const_expr(has_k_tail):
                safe_global_k_idx = (global_k_idx < k).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            global_offset = (safe_global_n_idx * k + safe_global_k_idx) * in_data_bytes
            buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < ldg_b_iters - 1:
                lds_ptr = lds_ptr + block_threads * async_load_bytes

    def compute_stage(read_stage, k_tile):
        thr_sA_s2r = thr_copy_A.partition_S(
            fx.make_view(smem_a + read_stage * block_m * block_k, a_lds_layout)
        )
        thr_sB_s2r = thr_copy_B.partition_S(
            fx.make_view(smem_b + read_stage * block_n * block_k, b_lds_layout)
        )

        def compute_k_chunk(block_k_iter):
            fx.copy(
                s2r_copy_atom,
                thr_sB_s2r[None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
            fx.copy(
                s2r_copy_atom,
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

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    compute_k_chunk(block_k_iter)
            else:
                compute_k_chunk(block_k_iter)

    # Prime the staged LDS pipeline: preload the first `stages - 1` K tiles
    # before entering the main loop that overlaps async loads with compute.
    for stage in range_constexpr(stages - 1):
        async_load_b_to_lds(stage, stage)
        async_load_a_to_lds(stage, stage)
    rocdl.sched_barrier(0)

    if const_expr(has_k_tail):
        main_loop_end = (k_tiles > stages - 1).select(k_tiles - (stages - 1), 0)
    else:
        main_loop_end = k_tiles - (stages - 1)
    for k_tile in range(0, main_loop_end, 1):
        current_stage = k_tile % stages
        write_stage = (current_stage + stages - 1) % stages
        __barrier((stages - 2) * ldg_wait_count)
        async_load_b_to_lds(k_tile + (stages - 1), write_stage)
        async_load_a_to_lds(k_tile + (stages - 1), write_stage)
        compute_stage(current_stage, k_tile)

    current_stage = main_loop_end % stages
    for s in range_constexpr(0, stages - 1):
        __barrier((stages - 2 - s) * ldg_wait_count)
        compute_stage(current_stage, main_loop_end + s)
        current_stage = (current_stage + 1) % stages

    frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
    frag_C_out.store(frag_C.load().to(elem_dtype))

    fx.gpu.barrier()
    for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
        row = fx.get_scalar(thr_mma_cRow[i])
        col = fx.get_scalar(thr_mma_cCol[i])
        sC[row, col] = frag_C_out[i]

    fx.gpu.barrier()
    fx.copy(s2r_copy_atom, thr_sC, frag_C_cshuffle)
    fx.copy(r2g_copy_atom, frag_C_cshuffle, thr_gC, pred=pred_C)


@flyc.jit
def launch_hgemm_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    param: HGemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    elem_dtype = (
        fx.Float16 if const_expr(param.dtype_id == HGEMM_DTYPE_FP16) else fx.BFloat16
    )
    mma_atom = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
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
    hgemm_gfx950_kernel(out, a, b, bias, m, n, k, tiled_mma, param).launch(
        grid=(num_pid_m * num_pid_n, 1, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


def make_hgemm_param_and_validate(m, n, k, kwargs):
    result = None
    try:
        result = make_hgemm_gfx950_param(**kwargs)
    except Exception as e:
        return None
    if not ((n % 32 == 0) and (k % result.mma_k == 0)):
        return None
    return result


def infer_has_k_tail(k: int, block_k: int, stages: int):
    k_tiles = (k + block_k - 1) // block_k
    return (k % block_k != 0) or (k_tiles < stages - 1)


def hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    user_kwargs: dict = {},
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    if stream is None:
        stream = torch.cuda.current_stream()
    device = a.device
    assert a.device == b.device
    k = a.shape[-1]
    a = a.view(-1, k)
    m = a.shape[0]
    n = b.shape[0]
    assert b.shape[1] == k
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    assert a.dtype == b.dtype
    assert a.dtype in (torch.float16, torch.bfloat16)
    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    else:
        assert out.dtype == a.dtype
        assert out.device == device
        assert out.is_contiguous()
    out = out.view(-1, n)
    assert out.shape[0] == m
    assert out.dtype == a.dtype

    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    kwargs = {
        "block_m": 256,
        "block_n": 256,
        "block_k": 64,
        "stages": 2,
        "m_waves": 4,
        "n_waves": 4,
        "group_m": 0,
    }
    if m == 2048 and n == 2048 and k == 2048:
        kwargs = {
            "block_m": 128,
            "block_n": 128,
            "block_k": 64,
            "stages": 4,
            "m_waves": 4,
            "n_waves": 4,
            "group_m": 0,
        }
    kwargs.update(user_kwargs)
    kwargs["dtype_id"] = (
        HGEMM_DTYPE_FP16 if a.dtype is torch.float16 else HGEMM_DTYPE_BF16
    )
    kwargs["has_bias"] = False if bias is None else True
    kwargs["has_k_tail"] = infer_has_k_tail(
        k=k,
        block_k=kwargs["block_k"],
        stages=kwargs["stages"],
    )
    bias_tensor = a if bias is None else bias

    if bias is not None:
        assert bias.shape[0] == n
        assert bias.dtype == a.dtype

    param = make_hgemm_param_and_validate(m, n, k, kwargs)
    assert param is not None, "unsupported hgemm_layout_gfx950 shape/config"
    launch_hgemm_gfx950(out, a, b, bias_tensor, m, n, k, param, stream)
    return out
