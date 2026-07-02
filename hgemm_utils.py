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
from flydsl._mlir.dialects import llvm, fly
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
