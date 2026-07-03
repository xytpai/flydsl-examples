from abc import ABC, abstractmethod
from flydsl.expr import (
    range_constexpr,
    const_expr,
    arith,
    vector,
    gpu,
    rocdl,
    buffer_ops,
)
import flydsl.expr as fx
from utils.tensor_shim import (
    get_dtype_in_kernel,
    GTensor,
    STensor,
    _to_raw,
    _run_compiled,
)
from flydsl.expr.typing import T, Vector as Vec
from flydsl._mlir.dialects import llvm, fly, memref, scf
from flydsl._mlir import ir


class WmmaHalfBase(ABC):
    @abstractmethod
    def __init__(self, dtype: str):
        pass

    @abstractmethod
    def __call__(self, a_frag, b_frag, c_frag):
        pass


class WmmaHalf_m16n16k16(WmmaHalfBase):
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WMMA_A_FRAG_VALUES = 4
    WMMA_B_FRAG_VALUES = 4
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, a_frag, b_frag, c_frag):
        if self.dtype == "bf16":
            a_frag_vi16 = vector.bitcast(T.vec(self.WMMA_A_FRAG_VALUES, T.i16), a_frag)
            b_frag_vi16 = vector.bitcast(T.vec(self.WMMA_B_FRAG_VALUES, T.i16), b_frag)
            c_frag_new = rocdl.mfma_f32_16x16x16bf16_1k(
                T.f32x4, [a_frag_vi16, b_frag_vi16, c_frag, 0, 0, 0]
            )
            return c_frag_new
        else:
            c_frag_new = rocdl.mfma_f32_16x16x16f16(
                T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
            )
            return c_frag_new


class WmmaHalf_m16n16k32(WmmaHalfBase):
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 32
    WMMA_A_FRAG_VALUES = 8
    WMMA_B_FRAG_VALUES = 8
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str, agpr: bool = False):
        self.dtype = dtype
        self.agpr = agpr

    def __call__(self, a_frag, b_frag, c_frag):
        res_ty = T.vec(self.WMMA_C_FRAG_VALUES, T.f32)
        operands = [a_frag, b_frag, c_frag, 0, 0, 0]
        if self.dtype == "bf16":
            if self.agpr:
                return llvm.inline_asm(
                    res_ty,
                    [_to_raw(c_frag), _to_raw(a_frag), _to_raw(b_frag)],
                    "v_mfma_f32_16x16x32_bf16 $0, $2, $3, $0",
                    "=a,0,v,v",
                    has_side_effects=False,
                )
            else:
                return rocdl.mfma_f32_16x16x32_bf16(res_ty, operands)
        else:
            if self.agpr:
                return llvm.inline_asm(
                    res_ty,
                    [_to_raw(c_frag), _to_raw(a_frag), _to_raw(b_frag)],
                    "v_mfma_f32_16x16x32_f16 $0, $2, $3, $0",
                    "=a,0,v,v",
                    has_side_effects=False,
                )
            return rocdl.mfma_f32_16x16x32_f16(res_ty, operands)


class WmmaFp8_m16n16k128:
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 128
    WMMA_A_FRAG_VALUES = 32
    WMMA_B_FRAG_VALUES = 32
    WMMA_C_FRAG_VALUES = 4

    def _make_atom_call(self, a_frag, b_frag, c_frag):
        atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
        accum_type = Vec.make_type(4, fx.Float32)
        return fly.mma_atom_call_ssa([accum_type], atom, a_frag, b_frag, c_frag)

    def __call__(self, a_frag, b_frag, c_frag):
        return self._make_atom_call(a_frag, b_frag, c_frag)


def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


def swizzle_fp8_128(row, col_in_bytes):
    return col_in_bytes ^ (((row % 16) // 2) * 16)


def get_llvm_ptr(ptr, offset, dtype_bytes, ptr_type):
    base_ptr = fly.extract_aligned_pointer_as_index(ptr_type, ptr)
    base_ptr = llvm.PtrToIntOp(T.i64, base_ptr).result
    byte_offset = arith.index_cast(T.i64, fx.Index(offset) * fx.Index(dtype_bytes))
    llvm_ptr = llvm.AddOp(base_ptr, byte_offset, llvm.IntegerOverflowFlags(0)).result
    llvm_ptr = llvm.IntToPtrOp(ptr_type, llvm_ptr).result
    ptr_v = llvm_ptr._value if const_expr(hasattr(llvm_ptr, "_value")) else llvm_ptr
    return ptr_v


def __barrier(vmcnt=0):
    llvm.InlineAsmOp(
        None,
        [],
        f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
        "",
        has_side_effects=True,
    )


def __s_barrier():
    llvm.InlineAsmOp(None, [], "s_barrier", "", has_side_effects=True)


def buffer_load_lds_inline(rsrc, lds_ptr, global_offset, DMA_BYTES):
    if const_expr(DMA_BYTES == 16):
        asm = "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, 0 offen sc0 lds"
    elif const_expr(DMA_BYTES == 8):
        asm = "s_mov_b32 m0, $0\n\tbuffer_load_dwordx2 $1, $2, 0 offen sc0 lds"
    elif const_expr(DMA_BYTES == 4):
        asm = "s_mov_b32 m0, $0\n\tbuffer_load_dword $1, $2, 0 offen sc0 lds"
    else:
        raise NotImplementedError(f"DMA_BYTES={DMA_BYTES} not supported")
    llvm.InlineAsmOp(
        None,
        [lds_ptr, global_offset, rsrc],
        asm,
        "s,v,s",
        has_side_effects=True,
    )


class SplitKProtocol:
    def __init__(
        self,
        SPLIT_K,
        BLOCK_M,
        BLOCK_N,
        STG_VEC_SIZE,
        DTYPE_BYTES,
        BLOCK_THREADS,
        HAS_BIAS,
    ):
        self.SPLIT_K = SPLIT_K
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.STG_VEC_SIZE = STG_VEC_SIZE
        self.DTYPE_BYTES = DTYPE_BYTES
        self.BLOCK_THREADS = BLOCK_THREADS
        self.HAS_BIAS = HAS_BIAS
        self.STG_C_X_THREADS = BLOCK_N // STG_VEC_SIZE
        assert (self.STG_C_X_THREADS >= 1) and (BLOCK_N % STG_VEC_SIZE == 0)
        self.STG_REG_C_COUNT = BLOCK_M * BLOCK_N // BLOCK_THREADS // STG_VEC_SIZE
        assert (self.STG_REG_C_COUNT >= 1) and (
            (BLOCK_M * BLOCK_N) % (BLOCK_THREADS * STG_VEC_SIZE) == 0
        )

    def init(
        self,
        tid,
        ks_idx,
        m,
        m_offset,
        n_offset,
        in_dtype_,
        out_dtype_,
        signal,
        signal_idx,
        semaphore,
    ):
        self.tid = tid
        self.ks_idx = ks_idx
        self.m = m
        self.m_offset = m_offset
        self.n_offset = n_offset
        self.in_dtype_ = in_dtype_
        self.c_zero_d = arith.constant(0.0, type=out_dtype_)
        self.signal = signal
        self.signal_idx = signal_idx
        self.semaphore = semaphore
        self.semaphore_ = GTensor(semaphore, dtype=T.i32, shape=(-1,))
        self.signal_ = GTensor(signal, dtype=T.i32, shape=(-1,))

    def zero_c(self, C, C_, BIAS_=None):
        # zero c if current block is the first block
        is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(self.tid), fx.Index(0))
        cond_ks0 = arith.cmpi(arith.CmpIPredicate.eq, self.ks_idx, fx.Index(0))
        cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
        with ir.InsertionPoint(cond_ks0_if.then_block):
            zero_vec = vector.broadcast(
                T.vec(self.STG_VEC_SIZE, self.in_dtype_), self.c_zero_d
            )
            for i in range_constexpr(self.STG_REG_C_COUNT):
                global_tid = self.BLOCK_THREADS * i + self.tid
                m_local_idx = global_tid // self.STG_C_X_THREADS
                n_local_idx = global_tid % self.STG_C_X_THREADS * self.STG_VEC_SIZE
                row_idx = self.m_offset + fx.Index(m_local_idx)
                init_vec = zero_vec
                if const_expr(self.HAS_BIAS):
                    init_vec = BIAS_.vec_load(
                        (self.n_offset + n_local_idx,), self.STG_VEC_SIZE
                    )
                cond_boundary = arith.cmpi(
                    arith.CmpIPredicate.ult, row_idx, fx.Index(self.m)
                )
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    bytes_offset = C_.linear_offset(
                        (row_idx, self.n_offset + n_local_idx)
                    )
                    bytes_offset_i32 = arith.index_cast(T.i32, bytes_offset)
                    c_ptr = get_llvm_ptr(
                        C,
                        bytes_offset_i32,
                        self.DTYPE_BYTES,
                        ir.Type.parse("!llvm.ptr<1>"),
                    )
                    llvm.InlineAsmOp(
                        None,
                        [c_ptr, init_vec],
                        "global_store_dwordx4 $0, $1, off sc0 sc1",
                        "v,v",
                        has_side_effects=True,
                    )
                    scf.YieldOp([])
            gpu.barrier()
            # trigger signal when zeroc is done by the first arrived block
            is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_cond_if.then_block):
                signal_ptr = get_llvm_ptr(
                    self.signal, self.signal_idx, 4, ir.Type.parse("!llvm.ptr<1>")
                )
                llvm.InlineAsmOp(
                    None,
                    [signal_ptr, arith.constant(1, type=T.i32)],
                    "global_store_dword $0, $1, off sc0 sc1",
                    "v,v",
                    has_side_effects=True,
                )
                scf.YieldOp([])
            gpu.barrier()
            scf.YieldOp([])

    def split_k_barrier(self):
        # spin-wait until signal triggered
        is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(self.tid), fx.Index(0))
        is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
        with ir.InsertionPoint(is_t0_cond_if.then_block):
            init_cur = arith.constant(0, type=T.i32)
            w = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.eq, cur, arith.constant(0, type=T.i32)
                ).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                signal_ptr = get_llvm_ptr(
                    self.signal, self.signal_idx, 4, ir.Type.parse("!llvm.ptr<1>")
                )
                data = llvm.InlineAsmOp(
                    T.i32,
                    [signal_ptr],
                    "global_load_dword $0, $1, off sc1",
                    "=v,v",
                    has_side_effects=True,
                ).result
                rocdl.s_waitcnt(0)
                scf.YieldOp([data])
            scf.YieldOp([])
        rocdl.sched_barrier(0)
        gpu.barrier()
        # clean semaphore and signal if this is the last block within split-k group
        is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
        with ir.InsertionPoint(is_t0_cond_if.then_block):
            semaphore_ptr = get_llvm_ptr(
                self.semaphore, self.signal_idx, 4, ir.Type.parse("!llvm.ptr<1>")
            )
            arrive_idx = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                semaphore_ptr,
                arith.constant(1, type=T.i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
            cond_ksl = arith.cmpi(
                arith.CmpIPredicate.eq, fx.Index(arrive_idx), fx.Index(self.SPLIT_K - 1)
            )
            cond_ksl_if = scf.IfOp(cond_ksl, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ksl_if.then_block):
                self.semaphore_[self.signal_idx] = arith.constant(0, type=T.i32)
                self.signal_[self.signal_idx] = arith.constant(0, type=T.i32)
                scf.YieldOp([])
            scf.YieldOp([])
        gpu.barrier()


class BlockSwizzle:
    def __init__(self, NUM_XCDS, NUM_CUS, GROUP_M, BLOCK_M, BLOCK_N, N):
        self.NUM_XCDS = NUM_XCDS
        self.NUM_CUS = NUM_CUS
        self.GROUP_M = GROUP_M
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.N = N
        self.N_BLOCKS = N // BLOCK_N

    def swizzle(self, m, pid):
        simple_m = fx.Index(pid // self.N_BLOCKS)
        simple_n = fx.Index(pid % self.N_BLOCKS)
        if const_expr(self.GROUP_M <= 0):
            return simple_m, simple_n
        num_xcds = fx.Index(self.NUM_XCDS)
        num_cus = fx.Index(self.NUM_CUS)
        swizzle_threshold = num_cus
        num_pid_n = fx.Index(self.N_BLOCKS)
        num_pid_m = (fx.Index(m) + fx.Index(self.BLOCK_M - 1)) // fx.Index(self.BLOCK_M)
        num_wg = num_pid_m * num_pid_n
        linear_id = fx.Index(pid)
        intra_xcd = linear_id // num_xcds
        xcd = linear_id % num_xcds
        wgid = xcd * (num_wg // num_xcds) + intra_xcd
        group_m = fx.Index(self.GROUP_M)
        wgid_per_group = group_m * num_pid_n
        group_id = wgid // wgid_per_group
        intra_group = wgid % wgid_per_group
        first_pid_m = group_id * group_m
        remaining_m = num_pid_m - first_pid_m
        group_size_m = arith.select(
            arith.cmpi(arith.CmpIPredicate.ult, remaining_m, group_m),
            remaining_m,
            group_m,
        )
        swizzled_n = intra_group // group_size_m
        swizzled_m = first_pid_m + (intra_group % group_size_m)
        use_simple = arith.cmpi(
            arith.CmpIPredicate.ult, num_wg, swizzle_threshold
        ) | arith.cmpi(arith.CmpIPredicate.ne, num_wg % num_xcds, fx.Index(0))
        return (
            arith.select(use_simple, simple_m, swizzled_m),
            arith.select(use_simple, simple_n, swizzled_n),
        )
