import functools
import torch
from typing import Optional

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

from .hgemm_wmma_gfx950_utils import (
    BlockSwizzle,
    atomic_add_stg_vec,
    get_llvm_ptr,
)

GFX950_DMA_BYTES = 16
GFX950_WAVE_SIZE = 64
SPLIT_K_SEMAPHORE_MAX_LEN = 256
HGEMM_DTYPE_BF16 = 2
HGEMM_DTYPE_FP16 = 3


def make_buffer_rsrc(tensor):
    """Build a ROCDL buffer resource over a global tensor's base address."""
    addr = fx.ptrtoint(fx.get_iter(tensor))
    base = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr"), fx.as_ir_value(addr)
    ).result
    num_records = fx.Int64(0xFFFFFFFF)
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(get_rocm_arch()):
        flags |= 1 << 24
        flags |= 2 << 28

    return rocdl.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"),
        base,
        fx.Int16(0).ir_value(),
        num_records.ir_value(),
        fx.Int32(flags).ir_value(),
    ).result


class SplitKProtocol:
    def __init__(
        self,
        SPLIT_K,
        BLOCK_M,
        BLOCK_N,
        STG_VEC_SIZE,
        C_DTYPE_BYTES,
        BLOCK_THREADS,
        HAS_BIAS,
    ):
        self.SPLIT_K = SPLIT_K
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.STG_VEC_SIZE = STG_VEC_SIZE
        self.C_DTYPE_BYTES = C_DTYPE_BYTES
        self.BLOCK_THREADS = BLOCK_THREADS
        self.HAS_BIAS = HAS_BIAS
        self.STG_C_X_THREADS = BLOCK_N // STG_VEC_SIZE
        assert self.STG_C_X_THREADS * STG_VEC_SIZE == BLOCK_N
        self.STG_C_ITERS = BLOCK_M * BLOCK_N // BLOCK_THREADS // STG_VEC_SIZE
        assert (
            self.STG_C_ITERS * BLOCK_THREADS * STG_VEC_SIZE == BLOCK_M * BLOCK_N
        )

    @flyc.jit
    def init(
        self,
        semaphore_ptr,
        signal_ptr,
        c_ptr,
        bias_buf,
        tid,
        ks_idx,
        m,
        n,
        block_m_offset,
        block_n_offset,
        out_dtype_,
        signal_idx,
        c_stride,
    ):
        self.semaphore_ptr = semaphore_ptr
        self.signal_ptr = signal_ptr
        self.c_ptr = c_ptr
        self.bias_buf = bias_buf
        self.tid = tid
        self.ks_idx = ks_idx
        self.m = m
        self.n = n
        self.block_m_offset = block_m_offset
        self.block_n_offset = block_n_offset
        self.out_dtype_ = out_dtype_
        self.signal_idx = signal_idx
        self.c_stride = c_stride
        self.semaphore_buf = rocdl.make_buffer_tensor(semaphore_ptr)
        self.signal_buf = rocdl.make_buffer_tensor(signal_ptr)
        if const_expr(self.HAS_BIAS):
            self.bias_vecs = fx.logical_divide(
                self.bias_buf, fx.make_layout(self.STG_VEC_SIZE, 1)
            )

    @flyc.jit
    def zero_c(self):
        if self.ks_idx == 0:
            if const_expr(self.STG_VEC_SIZE == 4):
                store_asm = "global_store_dwordx2 $0, $1, off sc0 sc1"
            elif const_expr(self.STG_VEC_SIZE == 8):
                store_asm = "global_store_dwordx4 $0, $1, off sc0 sc1"
            else:
                raise NotImplementedError(f"STG_VEC_SIZE={self.STG_VEC_SIZE}")
            zero_vec = fx.full(self.STG_VEC_SIZE, 0.0, self.out_dtype_)
            for i in range_constexpr(self.STG_C_ITERS):
                global_tid = self.BLOCK_THREADS * i + self.tid
                m_local_idx = global_tid // self.STG_C_X_THREADS
                n_local_idx = global_tid % self.STG_C_X_THREADS * self.STG_VEC_SIZE
                global_m_idx = self.block_m_offset + m_local_idx
                global_n_idx = self.block_n_offset + n_local_idx
                safe_global_n_idx = (global_n_idx < self.n).select(global_n_idx, 0)
                if const_expr(self.HAS_BIAS):
                    init_vec = self.bias_vecs[
                        None, safe_global_n_idx // self.STG_VEC_SIZE
                    ].load()
                else:
                    init_vec = zero_vec
                if global_m_idx < self.m and global_n_idx < self.n:
                    c_offset = global_m_idx * self.c_stride + global_n_idx
                    c_ptr = get_llvm_ptr(
                        self.c_ptr,
                        c_offset,
                        self.C_DTYPE_BYTES,
                        ir.Type.parse("!llvm.ptr<1>"),
                    )
                    llvm.InlineAsmOp(
                        None,
                        [c_ptr, init_vec],
                        store_asm,
                        "v,v",
                        has_side_effects=True,
                    )
            gpu.barrier()
            if self.tid == 0:
                signal_ptr = get_llvm_ptr(
                    self.signal_ptr,
                    self.signal_idx,
                    4,
                    ir.Type.parse("!llvm.ptr<1>"),
                )
                llvm.InlineAsmOp(
                    None,
                    [signal_ptr, arith.constant(1, type=T.i32)],
                    "global_store_dword $0, $1, off sc0 sc1",
                    "v,v",
                    has_side_effects=True,
                )
            gpu.barrier()

    @flyc.jit
    def reset_sync_state(self):
        self.semaphore_buf[self.signal_idx] = 0
        self.signal_buf[self.signal_idx] = 0

    @flyc.jit
    def split_k_barrier(self):
        if self.tid == 0:
            cur = fx.Int32(0)
            while cur == 0:
                signal_ptr = get_llvm_ptr(
                    self.signal_ptr,
                    self.signal_idx,
                    4,
                    ir.Type.parse("!llvm.ptr<1>"),
                )
                cur = llvm.InlineAsmOp(
                    T.i32,
                    [signal_ptr],
                    "global_load_dword $0, $1, off sc1",
                    "=v,v",
                    has_side_effects=True,
                ).result
                rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)
        gpu.barrier()
        if self.tid == 0:
            semaphore_ptr = get_llvm_ptr(
                self.semaphore_ptr,
                self.signal_idx,
                4,
                ir.Type.parse("!llvm.ptr<1>"),
            )
            arrive_idx = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                semaphore_ptr,
                arith.constant(1, type=T.i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
            if arrive_idx == self.SPLIT_K - 1:
                self.reset_sync_state()
        gpu.barrier()


@fx.struct
class HGemmGfx950Param:
    dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    split_k: fx.Constexpr[int]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    use_half_tile_interleaved: fx.Constexpr[bool]
    a_is_transposed: fx.Constexpr[bool]
    b_is_transposed: fx.Constexpr[bool]
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
    use_half_tile_interleaved: bool = False,
    a_is_transposed: bool = False,
    b_is_transposed: bool = True,
    has_bias: bool = False,
    has_k_tail: bool = False,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
    split_k: int = 1,
) -> HGemmGfx950Param:
    if dtype_id not in (HGEMM_DTYPE_BF16, HGEMM_DTYPE_FP16):
        raise ValueError(f"unsupported dtype_id={dtype_id}")
    if (
        block_m <= 0
        or block_n <= 0
        or block_k <= 0
        or stages <= 0
        or split_k <= 0
    ):
        raise ValueError(
            "block_m, block_n, block_k, stages, and split_k must be positive"
        )
    if (mma_m, mma_n, mma_k) != (16, 16, 32):
        raise ValueError("the gfx950 layout kernel currently requires mma=16x16x32")
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0:
        raise ValueError("m_waves, and n_waves must be positive")
    if group_m < 0:
        raise ValueError("group_m must be non-negative")
    in_dbytes = out_dbytes = 2  # for hgemm
    cshuffle_vec_size = 16 // in_dbytes
    if use_half_tile_interleaved:
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
        assert half_block_n % cshuffle_vec_size == 0
    else:
        assert block_n % cshuffle_vec_size == 0
    smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
    smem_bytes = max(smem_bytes, block_m * block_n * in_dbytes)
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
    if use_half_tile_interleaved:
        half_ldg_a_iters = ((block_m // 2) * block_k) // (
            block_threads * async_load_vec_size
        )
        half_ldg_b_iters = ((block_n // 2) * block_k) // (
            block_threads * async_load_vec_size
        )
        if (
            half_ldg_a_iters * block_threads * async_load_vec_size
            != (block_m // 2) * block_k
        ):
            raise ValueError(
                "Half-tile A async load tile must be exactly covered by whole-thread vector loads: "
                f"half_block_m={block_m // 2}, block_k={block_k}, "
                f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
                f"half_ldg_a_iters={half_ldg_a_iters}"
            )
        if (
            half_ldg_b_iters * block_threads * async_load_vec_size
            != (block_n // 2) * block_k
        ):
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
        split_k=split_k,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        use_half_tile_interleaved=use_half_tile_interleaved,
        a_is_transposed=a_is_transposed,
        b_is_transposed=b_is_transposed,
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
    name += f"_ks{param.split_k}"
    name += f"_w{param.m_waves}x{param.n_waves}"
    name += f"_gm{param.group_m}"
    name += f"_bias{int(param.has_bias)}"
    name += f"_ktail{int(param.has_k_tail)}"
    a_layout = "t" if param.a_is_transposed else "n"
    b_layout = "t" if param.b_is_transposed else "n"
    name += f"_layout{a_layout}{b_layout}"
    name += "_phti" if param.use_half_tile_interleaved else "_pft"
    return name


def __barrier(vmcnt=0):
    llvm.InlineAsmOp(
        None,
        [],
        f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
        "",
        has_side_effects=True,
    )


def __waitcnt(vmcnt=0):
    llvm.InlineAsmOp(None, [], f"s_waitcnt vmcnt({vmcnt})", "", has_side_effects=True)


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
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    working_k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: HGemmGfx950Param,
):
    is_split_k = param.split_k > 1
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
    cshuffle_vec_size = GFX950_DMA_BYTES // param.out_data_bytes
    elem_dtype = (
        fx.Float16 if const_expr(param.dtype_id == HGEMM_DTYPE_FP16) else fx.BFloat16
    )
    if const_expr(is_split_k):
        splitk_protocol = SplitKProtocol(
            param.split_k,
            block_m,
            block_n,
            cshuffle_vec_size,
            param.out_data_bytes,
            block_threads,
            param.has_bias,
        )

    tid = fx.thread_idx.x
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    ks_idx = fx.block_idx.y
    ks_begin = ks_idx * working_k
    ks_end = ks_begin + working_k
    ks_end = (ks_end < k).select(ks_end, k)
    k_tiles = (ks_end - ks_begin + block_k - 1) // block_k
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

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

    # Preserve a buffer tensor for tiled C stores while using raw ROCDL buffer
    # resources for global loads and split-K synchronization.
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
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
            elem_dtype,
            fx.block_idx.x,
            n,
        )

    a_rsrc = make_buffer_rsrc(a)
    b_rsrc = make_buffer_rsrc(b)

    s2r_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    g2r_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    r2g_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    if const_expr(param.a_is_transposed):
        a_s2r_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype
        )
        a_tiled_copy_atom = a_s2r_copy_atom
    else:
        a_tiled_copy_atom = g2r_copy_atom
        a_s2r_copy_atom = s2r_copy_atom
    if const_expr(not param.b_is_transposed):
        b_s2r_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype
        )
        b_tiled_copy_atom = b_s2r_copy_atom
    else:
        b_tiled_copy_atom = g2r_copy_atom
        b_s2r_copy_atom = s2r_copy_atom

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]

    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(a_tiled_copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(b_tiled_copy_atom, tiled_mma).get_slice(tid)

    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))

    def make_lds_layout(rows):
        return fx.make_composed_layout(
            swizzle,
            fx.make_ordered_layout((rows, block_k), (1, 0)),
        )

    def make_transposed_lds_layout(rows):
        # ds_read_tr16 transposes within 16-element groups. Preserve those
        # groups and XOR two K bits into contiguous-dimension bits [4:6]:
        # contiguous_idx ^ ((k_idx & 3) << 4).
        base_layout = fx.make_ordered_layout((rows, block_k), (0, 1))
        if const_expr(rows == 64):
            trans_swizzle = fx.static(fx.SwizzleType.get(2, 4, 2))
            return fx.make_composed_layout(trans_swizzle, base_layout)
        if const_expr(rows == 128):
            trans_swizzle = fx.static(fx.SwizzleType.get(2, 4, 3))
            return fx.make_composed_layout(trans_swizzle, base_layout)
        if const_expr(rows == 256):
            trans_swizzle = fx.static(fx.SwizzleType.get(2, 4, 4))
            return fx.make_composed_layout(trans_swizzle, base_layout)
        return base_layout

    # T-layout A[M, K] is M-contiguous; N-layout B[K, N] is N-contiguous.
    # Keep those physical orders in LDS and use the gfx950 transpose read to
    # form the K-contiguous MFMA register fragments. The compatible swizzle
    # preserves the transpose atom's 16-element lane grouping.
    a_lds_layout = (
        make_transposed_lds_layout(block_m)
        if const_expr(param.a_is_transposed)
        else make_lds_layout(block_m)
    )
    b_lds_layout = (
        make_transposed_lds_layout(block_n)
        if const_expr(not param.b_is_transposed)
        else make_lds_layout(block_n)
    )
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
    a_k_coords = fx.make_view(0, fx.make_layout((block_m, block_k), (0, 1)))
    thr_mma_cRow = thr_mma.partition_C(row_coords)
    thr_mma_cCol = thr_mma.partition_C(col_coords)
    thr_mma_aK = thr_mma.partition_A(a_k_coords)

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
    if const_expr(param.has_bias and not is_split_k):
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            col_idx = fx.get_scalar(thr_mma_cCol[i])
            global_n_idx = block_n_offset + col_idx
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            bias_val = bias_buf[safe_global_n_idx].to(fx.Float32)
            frag_C[i] = bias_val

    for i in range_constexpr(fx.size(pred_C.shape).unpack()):
        local_row = fx.get_scalar(thr_cRow[i])
        local_col = fx.get_scalar(thr_cCol[i])
        row_idx = bid_m * block_m + local_row
        col_idx = bid_n * block_n + local_col
        pred_C[i] = (
            (local_row < block_m)
            & (local_col < block_n)
            & (row_idx < m)
            & (col_idx < n)
        )

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

    def transposed_contiguous_idx(idx, k_idx, layout, rows):
        # The XOR swizzle is self-inverse. Given the physical contiguous
        # position written by direct-to-LDS DMA, select the logical global
        # vector that belongs at that position.
        elem_offset = fx.get_scalar(fx.crd2idx((idx, k_idx), layout))
        return elem_offset % rows

    def async_load_a_to_lds(k_tile, stage):
        lds_ptr = make_wave_lds_ptr(smem_a + stage * block_m * block_k)
        for i in range_constexpr(ldg_a_iters):
            global_tid = block_threads * i + tid
            if const_expr(param.a_is_transposed):
                ldg_a_x_threads = block_m // async_load_vec_size
                m_lds_idx = (
                    global_tid % ldg_a_x_threads * async_load_vec_size
                )
                k_local_idx = global_tid // ldg_a_x_threads
                m_local_idx = transposed_contiguous_idx(
                    m_lds_idx,
                    k_local_idx,
                    a_lds_layout,
                    block_m,
                )
                global_m_idx = block_m_offset + m_local_idx
                global_k_idx = ks_begin + k_tile * block_k + k_local_idx
            else:
                m_local_idx = global_tid // ldg_x_threads
                k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
                global_m_idx = block_m_offset + m_local_idx
                global_k_idx = ks_begin + k_tile * block_k + swizzled_col_idx(
                    m_local_idx,
                    k_local_idx,
                    a_lds_layout,
                )
            safe_global_m_idx = (global_m_idx < m).select(global_m_idx, 0)
            if const_expr(has_k_tail):
                safe_global_k_idx = (global_k_idx < ks_end).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            if const_expr(param.a_is_transposed):
                global_offset = (
                    safe_global_k_idx * a_leading_stride + safe_global_m_idx
                ) * in_data_bytes
            else:
                global_offset = (
                    safe_global_m_idx * a_leading_stride + safe_global_k_idx
                ) * in_data_bytes
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < ldg_a_iters - 1:
                lds_ptr = lds_ptr + block_threads * async_load_bytes

    def async_load_b_to_lds(k_tile, stage):
        lds_ptr = make_wave_lds_ptr(smem_b + stage * block_n * block_k)
        for i in range_constexpr(ldg_b_iters):
            global_tid = block_threads * i + tid
            if const_expr(not param.b_is_transposed):
                ldg_b_x_threads = block_n // async_load_vec_size
                n_lds_idx = (
                    global_tid % ldg_b_x_threads * async_load_vec_size
                )
                k_local_idx = global_tid // ldg_b_x_threads
                n_local_idx = transposed_contiguous_idx(
                    n_lds_idx,
                    k_local_idx,
                    b_lds_layout,
                    block_n,
                )
                global_n_idx = block_n_offset + n_local_idx
                global_k_idx = ks_begin + k_tile * block_k + k_local_idx
            else:
                n_local_idx = global_tid // ldg_x_threads
                k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
                global_n_idx = block_n_offset + n_local_idx
                global_k_idx = ks_begin + k_tile * block_k + swizzled_col_idx(
                    n_local_idx,
                    k_local_idx,
                    b_lds_layout,
                )
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            if const_expr(has_k_tail):
                safe_global_k_idx = (global_k_idx < ks_end).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            if const_expr(not param.b_is_transposed):
                global_offset = (
                    safe_global_k_idx * b_leading_stride + safe_global_n_idx
                ) * in_data_bytes
            else:
                global_offset = (
                    safe_global_n_idx * b_leading_stride + safe_global_k_idx
                ) * in_data_bytes
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
            if const_expr(has_k_tail):
                global_k_iter = (
                    ks_begin + k_tile * block_k + block_k_iter * param.mma_k
                )
                frag_a_k_coords = thr_mma_aK[None, None, block_k_iter]
                for i in range_constexpr(fx.size(frag_A_chunk.shape).unpack()):
                    local_k_idx = fx.get_scalar(frag_a_k_coords[i])
                    global_k_idx = (
                        ks_begin + k_tile * block_k + local_k_idx
                    )
                    valid_k = global_k_idx < ks_end
                    frag_A_chunk[i] = valid_k.select(
                        frag_A_chunk[i], elem_dtype(0.0)
                    )
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A_chunk,
                frag_B[None, None, block_k_iter],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = (
                    ks_begin + k_tile * block_k + block_k_iter * param.mma_k
                )
                if global_k_iter < ks_end:
                    compute_k_chunk(block_k_iter)
            else:
                compute_k_chunk(block_k_iter)

    if const_expr(is_split_k):
        splitk_protocol.zero_c()

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
    if const_expr(is_split_k):
        splitk_protocol.split_k_barrier()
        cshuffle_iters = (
            block_m * block_n // block_threads // cshuffle_vec_size
        )
        for i in range_constexpr(cshuffle_iters):
            global_tid = block_threads * i + tid
            local_row = global_tid // cshuffle_x_threads
            local_col = (
                global_tid % cshuffle_x_threads * cshuffle_vec_size
            )
            global_row = block_m_offset + local_row
            global_col = block_n_offset + local_col
            if (global_row < m) and (global_col < n):
                c_vec = fx.ptr_load(
                    smem_c + local_row * block_n + local_col,
                    result_type=fx.Vector.make_type(
                        cshuffle_vec_size, elem_dtype
                    ),
                ).ir_value()
                atomic_add_stg_vec(
                    out,
                    global_row * n + global_col,
                    c_vec,
                    cshuffle_vec_size,
                    param.out_data_bytes,
                    elem_dtype.ir_type,
                )
    else:
        fx.copy(s2r_copy_atom, thr_sC, frag_C_cshuffle)
        fx.copy(r2g_copy_atom, frag_C_cshuffle, thr_gC, pred=pred_C)


@flyc.kernel
def hgemm_hti_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    working_k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: HGemmGfx950Param,
):
    is_split_k = param.split_k > 1
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    half_block_m = block_m // 2
    half_block_n = block_n // 2
    stages = param.stages
    has_k_tail = param.has_k_tail
    async_load_bytes = param.async_load_bytes
    in_data_bytes = param.in_data_bytes
    async_load_vec_size = async_load_bytes // in_data_bytes
    ldg_x_threads = param.ldg_x_threads
    block_threads = param.block_threads
    n_waves = param.n_waves
    half_ldg_a_iters = param.ldg_a_iters // 2
    half_ldg_b_iters = param.ldg_b_iters // 2
    cshuffle_vec_size = GFX950_DMA_BYTES // param.out_data_bytes
    elem_dtype = (
        fx.Float16 if const_expr(param.dtype_id == HGEMM_DTYPE_FP16) else fx.BFloat16
    )
    if const_expr(is_split_k):
        splitk_protocol = SplitKProtocol(
            param.split_k,
            block_m,
            block_n,
            cshuffle_vec_size,
            param.out_data_bytes,
            block_threads,
            param.has_bias,
        )

    tid = fx.thread_idx.x
    wid = tid // GFX950_WAVE_SIZE
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    ks_idx = fx.block_idx.y
    ks_begin = ks_idx * working_k
    ks_end = ks_begin + working_k
    ks_end = (ks_end < k).select(ks_end, k)
    k_tiles = (ks_end - ks_begin + block_k - 1) // block_k
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

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

    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=False)
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
            elem_dtype,
            fx.block_idx.x,
            n,
        )

    a_rsrc = make_buffer_rsrc(a)
    b_rsrc = make_buffer_rsrc(b)

    s2r_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    g2r_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    r2g_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    if const_expr(param.a_is_transposed):
        a_s2r_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype
        )
        a_tiled_copy_atom = a_s2r_copy_atom
    else:
        a_tiled_copy_atom = g2r_copy_atom
        a_s2r_copy_atom = s2r_copy_atom
    if const_expr(not param.b_is_transposed):
        b_s2r_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype
        )
        b_tiled_copy_atom = b_s2r_copy_atom
    else:
        b_tiled_copy_atom = g2r_copy_atom
        b_s2r_copy_atom = s2r_copy_atom

    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(a_tiled_copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(b_tiled_copy_atom, tiled_mma).get_slice(tid)

    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))

    def make_lds_layout(rows):
        return fx.make_composed_layout(
            swizzle,
            fx.make_ordered_layout((rows, block_k), (1, 0)),
        )

    def make_transposed_lds_layout(rows):
        # Keep the 16-element groups required by ds_read_tr16 intact while
        # spreading the groups across LDS banks with two low K bits.
        base_layout = fx.make_ordered_layout((rows, block_k), (0, 1))
        if const_expr(rows == 64):
            trans_swizzle = fx.static(fx.SwizzleType.get(2, 4, 2))
            return fx.make_composed_layout(trans_swizzle, base_layout)
        if const_expr(rows == 128):
            trans_swizzle = fx.static(fx.SwizzleType.get(2, 4, 3))
            return fx.make_composed_layout(trans_swizzle, base_layout)
        if const_expr(rows == 256):
            trans_swizzle = fx.static(fx.SwizzleType.get(2, 4, 4))
            return fx.make_composed_layout(trans_swizzle, base_layout)
        return base_layout

    a_lds_layout = (
        make_transposed_lds_layout(half_block_m)
        if const_expr(param.a_is_transposed)
        else make_lds_layout(half_block_m)
    )
    b_lds_layout = (
        make_transposed_lds_layout(half_block_n)
        if const_expr(not param.b_is_transposed)
        else make_lds_layout(half_block_n)
    )
    c_lds_layout = fx.make_layout((half_block_m, half_block_n), (half_block_n, 1))
    a_k_coords = fx.make_view(
        0, fx.make_layout((half_block_m, block_k), (0, 1))
    )
    thr_mma_aK = thr_mma.partition_A(a_k_coords)

    wave_offset = rocdl.readfirstlane(
        fx.Int64.ir_type,
        fx.Int64(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * async_load_bytes),
    )

    def make_wave_lds_ptr(ptr):
        return fx.recast_iter(fx.Int8, ptr) + fx.Int32(wave_offset)

    def swizzled_col_idx(row, col, layout):
        elem_offset = fx.get_scalar(fx.crd2idx((row, col), layout))
        return elem_offset % block_k

    def transposed_contiguous_idx(idx, k_idx, layout, rows):
        elem_offset = fx.get_scalar(fx.crd2idx((idx, k_idx), layout))
        return elem_offset % rows

    def half_a_base(stage, m_part):
        return smem_a + (stage * block_m + m_part * half_block_m) * block_k

    def half_b_base(stage, n_part):
        return smem_b + (stage * block_n + n_part * half_block_n) * block_k

    def async_load_a_to_lds(m_part, k_tile, stage):
        lds_ptr = make_wave_lds_ptr(half_a_base(stage, m_part))
        for i in range_constexpr(half_ldg_a_iters):
            global_tid = block_threads * i + tid
            if const_expr(param.a_is_transposed):
                ldg_a_x_threads = half_block_m // async_load_vec_size
                m_lds_idx = (
                    global_tid % ldg_a_x_threads * async_load_vec_size
                )
                k_local_idx = global_tid // ldg_a_x_threads
                m_local_idx = transposed_contiguous_idx(
                    m_lds_idx,
                    k_local_idx,
                    a_lds_layout,
                    half_block_m,
                )
                global_m_idx = (
                    block_m_offset + m_part * half_block_m + m_local_idx
                )
                global_k_idx = ks_begin + k_tile * block_k + k_local_idx
            else:
                m_local_idx = global_tid // ldg_x_threads
                k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
                global_m_idx = (
                    block_m_offset + m_part * half_block_m + m_local_idx
                )
                global_k_idx = ks_begin + k_tile * block_k + swizzled_col_idx(
                    m_local_idx,
                    k_local_idx,
                    a_lds_layout,
                )
            safe_global_m_idx = (global_m_idx < m).select(global_m_idx, 0)
            if const_expr(has_k_tail):
                safe_global_k_idx = (global_k_idx < ks_end).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            if const_expr(param.a_is_transposed):
                global_offset = (
                    safe_global_k_idx * a_leading_stride + safe_global_m_idx
                ) * in_data_bytes
            else:
                global_offset = (
                    safe_global_m_idx * a_leading_stride + safe_global_k_idx
                ) * in_data_bytes
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < half_ldg_a_iters - 1:
                lds_ptr = lds_ptr + block_threads * async_load_bytes

    def async_load_b_to_lds(n_part, k_tile, stage):
        lds_ptr = make_wave_lds_ptr(half_b_base(stage, n_part))
        for i in range_constexpr(half_ldg_b_iters):
            global_tid = block_threads * i + tid
            if const_expr(not param.b_is_transposed):
                ldg_b_x_threads = half_block_n // async_load_vec_size
                n_lds_idx = (
                    global_tid % ldg_b_x_threads * async_load_vec_size
                )
                k_local_idx = global_tid // ldg_b_x_threads
                n_local_idx = transposed_contiguous_idx(
                    n_lds_idx,
                    k_local_idx,
                    b_lds_layout,
                    half_block_n,
                )
                global_n_idx = (
                    block_n_offset + n_part * half_block_n + n_local_idx
                )
                global_k_idx = ks_begin + k_tile * block_k + k_local_idx
            else:
                n_local_idx = global_tid // ldg_x_threads
                k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
                global_n_idx = (
                    block_n_offset + n_part * half_block_n + n_local_idx
                )
                global_k_idx = ks_begin + k_tile * block_k + swizzled_col_idx(
                    n_local_idx,
                    k_local_idx,
                    b_lds_layout,
                )
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            if const_expr(has_k_tail):
                safe_global_k_idx = (global_k_idx < ks_end).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            if const_expr(not param.b_is_transposed):
                global_offset = (
                    safe_global_k_idx * b_leading_stride + safe_global_n_idx
                ) * in_data_bytes
            else:
                global_offset = (
                    safe_global_n_idx * b_leading_stride + safe_global_k_idx
                ) * in_data_bytes
            buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < half_ldg_b_iters - 1:
                lds_ptr = lds_ptr + block_threads * async_load_bytes

    def make_gC(m_part, n_part):
        return fx.flat_divide(out_buf, (half_block_m, half_block_n))[
            None, None, bid_m * 2 + m_part, bid_n * 2 + n_part
        ]

    def make_c_fragment(m_part, n_part):
        gC = make_gC(m_part, n_part)
        frag_C = thr_mma.make_fragment_C(gC)
        frag_C.fill(0.0)
        return frag_C

    def load_a_fragment(m_part, read_stage, k_tile):
        sA = fx.make_view(half_a_base(read_stage, m_part), a_lds_layout)
        frag_A = thr_mma.make_fragment_A(sA)
        frag_A_retile = thr_copy_A.retile(frag_A)
        thr_sA_s2r = thr_copy_A.partition_S(sA)

        def copy_k_chunk(block_k_iter):
            frag_A_chunk = frag_A[None, None, block_k_iter]
            fx.copy(
                a_s2r_copy_atom,
                thr_sA_s2r[None, None, block_k_iter],
                frag_A_retile[None, None, block_k_iter],
            )
            if const_expr(has_k_tail):
                frag_a_k_coords = thr_mma_aK[None, None, block_k_iter]
                global_k_base = ks_begin + k_tile * block_k
                for i in range_constexpr(fx.size(frag_A_chunk.shape).unpack()):
                    local_k_idx = fx.get_scalar(frag_a_k_coords[i])
                    valid_k = global_k_base + local_k_idx < ks_end
                    frag_A_chunk[i] = valid_k.select(
                        frag_A_chunk[i], elem_dtype(0.0)
                    )

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = (
                    ks_begin + k_tile * block_k + block_k_iter * param.mma_k
                )
                if global_k_iter < ks_end:
                    copy_k_chunk(block_k_iter)
            else:
                copy_k_chunk(block_k_iter)
        return frag_A

    def load_b_fragment(n_part, read_stage, k_tile):
        sB = fx.make_view(half_b_base(read_stage, n_part), b_lds_layout)
        frag_B = thr_mma.make_fragment_B(sB)
        frag_B_retile = thr_copy_B.retile(frag_B)
        thr_sB_s2r = thr_copy_B.partition_S(sB)

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = (
                    ks_begin + k_tile * block_k + block_k_iter * param.mma_k
                )
                if global_k_iter < ks_end:
                    fx.copy(
                        b_s2r_copy_atom,
                        thr_sB_s2r[None, None, block_k_iter],
                        frag_B_retile[None, None, block_k_iter],
                    )
            else:
                fx.copy(
                    b_s2r_copy_atom,
                    thr_sB_s2r[None, None, block_k_iter],
                    frag_B_retile[None, None, block_k_iter],
                )
        return frag_B

    def consume(k_tile, frag_C, frag_A, frag_B, emit_sched_barrier):
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = (
                    ks_begin + k_tile * block_k + block_k_iter * param.mma_k
                )
                if global_k_iter < ks_end:
                    fx.gemm(
                        tiled_mma,
                        frag_C,
                        frag_A[None, None, block_k_iter],
                        frag_B[None, None, block_k_iter],
                        frag_C,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )
            else:
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

    def store_half_tile(m_part, n_part, frag_C):
        gC = fx.flat_divide(out_buf, (half_block_m, half_block_n))[
            None, None, bid_m * 2 + m_part, bid_n * 2 + n_part
        ]
        sC = fx.make_view(smem_c, c_lds_layout)

        row_coords = fx.make_view(
            0, fx.make_layout((half_block_m, half_block_n), (1, 0))
        )
        col_coords = fx.make_view(
            0, fx.make_layout((half_block_m, half_block_n), (0, 1))
        )
        thr_mma_cRow = thr_mma.partition_C(row_coords)
        thr_mma_cCol = thr_mma.partition_C(col_coords)

        cshuffle_x_threads = half_block_n // cshuffle_vec_size
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

        for i in range_constexpr(fx.size(pred_C.shape).unpack()):
            local_row = fx.get_scalar(thr_cRow[i])
            local_col = fx.get_scalar(thr_cCol[i])
            row_idx = block_m_offset + m_part * half_block_m + local_row
            col_idx = block_n_offset + n_part * half_block_n + local_col
            pred_C[i] = (
                (local_row < half_block_m)
                & (local_col < half_block_n)
                & (row_idx < m)
                & (col_idx < n)
            )

        frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            val = frag_C[i]
            if const_expr(param.has_bias and not is_split_k):
                col = fx.get_scalar(thr_mma_cCol[i])
                global_n_idx = block_n_offset + n_part * half_block_n + col
                safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
                val = val + bias_buf[safe_global_n_idx].to(fx.Float32)
            frag_C_out[i] = val.to(elem_dtype)

        fx.gpu.barrier()
        for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            sC[row, col] = frag_C_out[i]

        fx.gpu.barrier()
        if const_expr(is_split_k):
            cshuffle_vectors = (
                half_block_m * half_block_n // cshuffle_vec_size
            )
            cshuffle_iters = (
                cshuffle_vectors + block_threads - 1
            ) // block_threads
            for i in range_constexpr(cshuffle_iters):
                vector_idx = block_threads * i + tid
                if vector_idx < cshuffle_vectors:
                    local_row = vector_idx // cshuffle_x_threads
                    local_col = (
                        vector_idx % cshuffle_x_threads * cshuffle_vec_size
                    )
                    global_row = (
                        block_m_offset
                        + m_part * half_block_m
                        + local_row
                    )
                    global_col = (
                        block_n_offset
                        + n_part * half_block_n
                        + local_col
                    )
                    if (global_row < m) and (global_col < n):
                        c_vec = fx.ptr_load(
                            smem_c
                            + local_row * half_block_n
                            + local_col,
                            result_type=fx.Vector.make_type(
                                cshuffle_vec_size, elem_dtype
                            ),
                        ).ir_value()
                        atomic_add_stg_vec(
                            out,
                            global_row * n + global_col,
                            c_vec,
                            cshuffle_vec_size,
                            param.out_data_bytes,
                            elem_dtype.ir_type,
                        )
        else:
            fx.copy(s2r_copy_atom, thr_sC, frag_C_cshuffle)
            fx.copy(r2g_copy_atom, frag_C_cshuffle, thr_gC, pred=pred_C)
        fx.gpu.barrier()

    c00 = make_c_fragment(0, 0)
    c01 = make_c_fragment(0, 1)
    c10 = make_c_fragment(1, 0)
    c11 = make_c_fragment(1, 1)

    def compute_loaded_tile(k_tile, read_stage):
        b0 = load_b_fragment(0, read_stage, k_tile)
        a0 = load_a_fragment(0, read_stage, k_tile)
        consume(k_tile, c00, a0, b0, True)

        b1 = load_b_fragment(1, read_stage, k_tile)
        consume(k_tile, c01, a0, b1, True)

        a1 = load_a_fragment(1, read_stage, k_tile)
        consume(k_tile, c10, a1, b0, True)
        consume(k_tile, c11, a1, b1, True)

    def compute_double_tile(k_tile, prefetch_next):
        next_k_tile = k_tile + 2

        b0 = load_b_fragment(0, 0, k_tile)
        a0 = load_a_fragment(0, 0, k_tile)
        async_load_a_to_lds(1, k_tile + 1, 1)
        rocdl.s_barrier()
        consume(k_tile, c00, a0, b0, True)
        rocdl.s_barrier()

        b1 = load_b_fragment(1, 0, k_tile)
        if const_expr(prefetch_next):
            async_load_b_to_lds(0, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile, c01, a0, b1, True)
        rocdl.s_barrier()

        a1 = load_a_fragment(1, 0, k_tile)
        if const_expr(prefetch_next):
            async_load_a_to_lds(0, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile, c10, a1, b0, True)
        rocdl.s_barrier()

        b0 = load_b_fragment(0, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_b_to_lds(1, next_k_tile, 0)
            __barrier(2 * half_ldg_b_iters + 1 * half_ldg_a_iters)
        consume(k_tile, c11, a1, b1, True)
        if const_expr(not prefetch_next):
            __waitcnt(0)
        rocdl.s_barrier()

        a0 = load_a_fragment(0, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_a_to_lds(1, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile + 1, c00, a0, b0, True)
        rocdl.s_barrier()

        b1 = load_b_fragment(1, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_b_to_lds(0, next_k_tile + 1, 1)
            rocdl.s_barrier()
        consume(k_tile + 1, c01, a0, b1, True)
        rocdl.s_barrier()

        a1 = load_a_fragment(1, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_a_to_lds(0, next_k_tile + 1, 1)
            rocdl.s_barrier()
        consume(k_tile + 1, c10, a1, b0, True)
        rocdl.s_barrier()

        if const_expr(prefetch_next):
            async_load_b_to_lds(1, next_k_tile + 1, 1)
            __barrier(1 * half_ldg_b_iters + 1 * half_ldg_a_iters)
        consume(k_tile + 1, c11, a1, b1, True)
        rocdl.s_barrier()

    if const_expr(is_split_k):
        splitk_protocol.zero_c()

    if k_tiles == 2:
        # The interleaved pipeline below intentionally staggers wave barriers
        # while prefetching later double tiles. With no later tile, use a
        # symmetric preload path so all four A/B halves are visible before use.
        async_load_b_to_lds(0, 0, 0)
        async_load_a_to_lds(0, 0, 0)
        async_load_b_to_lds(1, 0, 0)
        async_load_a_to_lds(1, 0, 0)
        async_load_b_to_lds(0, 1, 1)
        async_load_a_to_lds(0, 1, 1)
        async_load_b_to_lds(1, 1, 1)
        async_load_a_to_lds(1, 1, 1)
        __waitcnt(0)
        rocdl.s_barrier()

        compute_loaded_tile(0, 0)
        compute_loaded_tile(1, 1)
    else:
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
        __barrier(1 * half_ldg_b_iters + 1 * half_ldg_a_iters)

        final_double_tile = ((k_tiles % 2) == 0).select(k_tiles - 2, k_tiles - 1)
        main_loop_end = (k_tiles > 2).select(final_double_tile, 0)
        for k_tile in range(0, main_loop_end, 2):
            compute_double_tile(k_tile, True)

        compute_double_tile(main_loop_end, False)

    if const_expr(is_split_k):
        splitk_protocol.split_k_barrier()

    store_half_tile(0, 0, c00)
    store_half_tile(0, 1, c01)
    store_half_tile(1, 0, c10)
    store_half_tile(1, 1, c11)


@flyc.jit
def launch_hgemm_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    param: HGemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    n = fx.Int32(fx.get_scalar(b.shape[1]))
    k = fx.Int32(fx.get_scalar(a.shape[1]))
    a_leading_stride = fx.Int32(
        fx.get_scalar(
            a.stride[1] if const_expr(param.a_is_transposed) else a.stride[0]
        )
    )
    b_leading_stride = fx.Int32(
        fx.get_scalar(
            b.stride[1] if const_expr(param.b_is_transposed) else b.stride[0]
        )
    )
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
    working_k = (k + param.split_k - 1) // param.split_k
    num_pid_m = (m + param.block_m - 1) // param.block_m
    num_pid_n = (n + param.block_n - 1) // param.block_n
    hgemm_kernel_impl = (
        hgemm_hti_gfx950_kernel
        if param.use_half_tile_interleaved
        else hgemm_gfx950_kernel
    )
    hgemm_kernel_impl._known_block_size = [param.block_threads, 1, 1]
    hgemm_kernel_impl._func.__name__ = make_hgemm_gfx950_kernel_name(param)
    hgemm_kernel_impl(
        out,
        a,
        b,
        bias,
        semaphore,
        signal,
        m,
        n,
        k,
        working_k,
        a_leading_stride,
        b_leading_stride,
        tiled_mma,
        param,
    ).launch(
        grid=(num_pid_m * num_pid_n, param.split_k, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


def make_hgemm_param_and_validate(m, n, k, kwargs):
    result = None
    try:
        result = make_hgemm_gfx950_param(**kwargs)
    except Exception:
        return None
    working_k = (k + result.split_k - 1) // result.split_k
    last_working_k = k - (result.split_k - 1) * working_k
    cshuffle_vec_size = GFX950_DMA_BYTES // result.out_data_bytes
    async_load_vec_size = GFX950_DMA_BYTES // result.in_data_bytes
    if (
        n % cshuffle_vec_size != 0
        or k % async_load_vec_size != 0
        or last_working_k <= 0
    ):
        return None
    num_pid_m = (m + result.block_m - 1) // result.block_m
    num_pid_n = (n + result.block_n - 1) // result.block_n
    if result.split_k > 1:
        c_elements_per_iteration = result.block_threads * cshuffle_vec_size
        if (
            num_pid_m * num_pid_n > SPLIT_K_SEMAPHORE_MAX_LEN
            or result.block_m * result.block_n % c_elements_per_iteration != 0
        ):
            return None
    return result


def infer_has_k_tail(
    k: int,
    split_k: int,
    block_k: int,
    stages: int,
    use_half_tile_interleaved: bool,
):
    working_k = (k + split_k - 1) // split_k
    last_working_k = k - (split_k - 1) * working_k
    working_k_tiles = (working_k + block_k - 1) // block_k
    last_working_k_tiles = (last_working_k + block_k - 1) // block_k
    has_k_tail = (
        working_k % block_k != 0
        or last_working_k % block_k != 0
        or working_k_tiles < stages - 1
        or last_working_k_tiles < stages - 1
    )
    if use_half_tile_interleaved:
        has_k_tail = (
            has_k_tail
            or working_k_tiles % 2 != 0
            or last_working_k_tiles % 2 != 0
        )
    return has_k_tail


@functools.lru_cache(maxsize=128)
def get_split_k_buffers(stream, device):
    semaphore = torch.zeros(
        (SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device
    )
    signal = torch.zeros(
        (SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device
    )
    return semaphore, signal


def hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    user_kwargs: dict = {},
    stream: Optional[torch.cuda.Stream] = None,
    layout: str = "nt",
) -> torch.Tensor:
    """Compute C[M, N] = A[M, K] @ B[K, N].

    Each layout character controls only the corresponding tensor stride:
    N is row-major and T is column-major. Logical tensor shapes never change.
    Set ``user_kwargs["split_k"]`` above 1 to atomically reduce K partitions.
    """
    if stream is None:
        stream = torch.cuda.current_stream()
    layout = layout.lower()
    if layout not in ("nn", "nt", "tn", "tt"):
        raise ValueError(
            f"unsupported GEMM layout: {layout!r}; "
            "expected 'nn', 'nt', 'tn', or 'tt'"
        )
    a_is_transposed = layout[0] == "t"
    b_is_transposed = layout[1] == "t"
    device = a.device
    assert a.device == b.device
    assert a.ndim == 2 and b.ndim == 2
    m, k = a.shape
    assert b.shape[0] == k
    n = b.shape[1]
    assert a.dtype == b.dtype
    assert a.dtype in (torch.float16, torch.bfloat16)
    if a_is_transposed:
        a_vec_size = GFX950_DMA_BYTES // a.element_size()
        if (
            a.stride(0) != 1
            or a.data_ptr() % GFX950_DMA_BYTES != 0
            or a.stride(1) * a.element_size() % GFX950_DMA_BYTES != 0
            or m % a_vec_size != 0
        ):
            padded_m = (m + a_vec_size - 1) // a_vec_size * a_vec_size
            a_storage = torch.zeros(
                (k, padded_m),
                dtype=a.dtype,
                device=a.device,
            )
            a_column_major = a_storage[:, :m].t()
            a_column_major.copy_(a)
            a = a_column_major
    else:
        if (
            a.stride(1) != 1
            or a.data_ptr() % GFX950_DMA_BYTES != 0
            or a.stride(0) * a.element_size() % GFX950_DMA_BYTES != 0
        ):
            a = a.contiguous()
    if b_is_transposed:
        if (
            b.stride(0) != 1
            or b.data_ptr() % GFX950_DMA_BYTES != 0
            or b.stride(1) * b.element_size() % GFX950_DMA_BYTES != 0
        ):
            b = b.t().contiguous().t()
    else:
        if (
            b.stride(1) != 1
            or b.data_ptr() % GFX950_DMA_BYTES != 0
            or b.stride(0) * b.element_size() % GFX950_DMA_BYTES != 0
        ):
            b = b.contiguous()
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
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 4,
        "group_m": 0,
        "use_half_tile_interleaved": True,
    }

    kwargs.update(user_kwargs)
    kwargs["a_is_transposed"] = a_is_transposed
    kwargs["b_is_transposed"] = b_is_transposed
    kwargs["dtype_id"] = (
        HGEMM_DTYPE_FP16 if a.dtype is torch.float16 else HGEMM_DTYPE_BF16
    )
    kwargs["has_bias"] = False if bias is None else True
    has_k_tail = infer_has_k_tail(
        k=k,
        split_k=kwargs["split_k"],
        block_k=kwargs["block_k"],
        stages=kwargs["stages"],
        use_half_tile_interleaved=kwargs["use_half_tile_interleaved"],
    )
    kwargs["has_k_tail"] = has_k_tail
    bias_tensor = a if bias is None else bias

    if bias is not None:
        assert bias.shape[0] == n
        assert bias.dtype == a.dtype

    param = make_hgemm_param_and_validate(m, n, k, kwargs)
    assert param is not None, "unsupported hgemm_layout_gfx950 shape/config"
    semaphore, signal = get_split_k_buffers(stream, device)
    launch_hgemm_gfx950(
        out,
        a,
        b,
        bias_tensor,
        semaphore,
        signal,
        param,
        stream,
    )
    return out
