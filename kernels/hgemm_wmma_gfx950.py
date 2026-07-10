import torch
import functools
import itertools
from typing import Optional, Tuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl.expr import (
    range_constexpr,
    const_expr,
    arith,
    vector,
    gpu,
    rocdl,
)
from flydsl._mlir import ir
from flydsl.runtime.device import get_rocm_arch
from flydsl._mlir.dialects import llvm

SMEM_CAPACITY_MAP = {
    "gfx942": 65536,
    "gfx950": 163840,
}

from .hgemm_wmma_gfx950_utils import (
    _run_compiled,
    WmmaHalf_m16n16k16,
    WmmaHalf_m16n16k32,
    WmmaFp8_m16n16k128,
    swizzle_xor16,
    swizzle_fp8_128,
    get_llvm_ptr,
    make_buffer_rsrc,
    make_hgemm_lds,
    __barrier,
    __s_barrier,
    buffer_load_lds_inline,
    SplitKProtocol,
    BlockSwizzle,
)

SPLIT_K_SEMAPHORE_MAX_LEN = 256


HGEMM_DTYPE_F32 = 1
HGEMM_DTYPE_BF16 = 2
HGEMM_DTYPE_F16 = 3
HGEMM_DTYPE_FP8_PTPC = 4


HGEMM_DTYPE_STR_MAP = {
    HGEMM_DTYPE_F32: "f32",
    HGEMM_DTYPE_BF16: "bf16",
    HGEMM_DTYPE_F16: "f16",
    HGEMM_DTYPE_FP8_PTPC: "fp8_ptpc",
}

@flyc.jit
def get_dtype_in_kernel(dtype_id):
    return {
        HGEMM_DTYPE_F32: fx.Float32,
        HGEMM_DTYPE_BF16: fx.BFloat16,
        HGEMM_DTYPE_F16: fx.Float16,
        HGEMM_DTYPE_FP8_PTPC: fx.Float8E4M3FN,
    }[dtype_id]


@fx.struct
class HGemmWmmaConstexprParam:
    DTYPE_ID: fx.Constexpr[int]
    BLOCK_M: fx.Constexpr[int]
    BLOCK_N: fx.Constexpr[int]
    BLOCK_K: fx.Constexpr[int]
    STAGES: fx.Constexpr[int]
    SPLIT_K: fx.Constexpr[int]
    BLOCK_M_WARPS: fx.Constexpr[int]
    BLOCK_N_WARPS: fx.Constexpr[int]
    BLOCK_K_WARPS: fx.Constexpr[int]
    HAS_BIAS: fx.Constexpr[bool]
    HAS_K_TAIL: fx.Constexpr[bool]
    GROUP_M: fx.Constexpr[int]
    USE_HALF_TILE_INTERLEAVED: fx.Constexpr[bool]
    # derived params
    IS_SPLIT_K: fx.Constexpr[bool]
    IS_SLICE_K: fx.Constexpr[bool]
    HALF_BLOCK_M: fx.Constexpr[int]
    HALF_BLOCK_N: fx.Constexpr[int]
    IS_FP8: fx.Constexpr[bool]
    IS_FP8_PTPC: fx.Constexpr[bool]
    WARP_SIZE: fx.Constexpr[int]
    IN_DTYPE_BYTES: fx.Constexpr[int]
    OUT_DTYPE_BYTES: fx.Constexpr[int]
    LDG_VEC_SIZE: fx.Constexpr[int]
    DMA_BYTES: fx.Constexpr[int]
    MFMA_PER_WARP_K: fx.Constexpr[int]
    WMMA_M: fx.Constexpr[int]
    WMMA_N: fx.Constexpr[int]
    WMMA_K: fx.Constexpr[int]
    WMMA_A_FRAG_VALUES: fx.Constexpr[int]
    WMMA_B_FRAG_VALUES: fx.Constexpr[int]
    WMMA_C_FRAG_VALUES: fx.Constexpr[int]
    WARP_ATOM_M: fx.Constexpr[int]
    WARP_ATOM_N: fx.Constexpr[int]
    WARP_ATOM_K: fx.Constexpr[int]
    WARP_K_STEPS: fx.Constexpr[int]
    K_SLICE: fx.Constexpr[int]
    BLOCK_THREADS: fx.Constexpr[int]
    BLOCK_MN_WARPS: fx.Constexpr[int]
    WARP_M_STEPS: fx.Constexpr[int]
    WARP_N_STEPS: fx.Constexpr[int]
    WARP_M: fx.Constexpr[int]
    WARP_N: fx.Constexpr[int]
    STG_WORK_SIZE_PER_M_STEP: fx.Constexpr[int]
    STG_VEC_SIZE: fx.Constexpr[int]
    STG_C_X_THREADS: fx.Constexpr[int]
    STG_C_QUAD_X_THREADS: fx.Constexpr[int]
    STG_C_ITERS: fx.Constexpr[int]
    BLOCK_K_BYTES: fx.Constexpr[int]
    LDG_ASYNC_VEC_SIZE: fx.Constexpr[int]
    LDG_A_X_THREADS_AS: fx.Constexpr[int]
    LDG_B_X_THREADS_AS: fx.Constexpr[int]
    LDG_A_ITERS_AS: fx.Constexpr[int]
    LDG_B_ITERS_AS: fx.Constexpr[int]
    LDG_WAIT_COUNT: fx.Constexpr[int]
    A_FRAGS_LEN: fx.Constexpr[int]
    B_FRAGS_LEN: fx.Constexpr[int]
    C_FRAGS_LEN: fx.Constexpr[int]


def init_hgemm_wmma_constexpr_param(
    DTYPE_ID: int,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    STAGES: int,
    SPLIT_K: int,
    BLOCK_M_WARPS: int,
    BLOCK_N_WARPS: int,
    BLOCK_K_WARPS: int,
    HAS_BIAS: bool,
    HAS_K_TAIL: bool,
    GROUP_M: int,
    USE_HALF_TILE_INTERLEAVED: bool,
):
    DTYPE_STR = HGEMM_DTYPE_STR_MAP[DTYPE_ID]
    assert BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS <= 16
    assert STAGES >= 2
    IS_SPLIT_K = SPLIT_K > 1
    IS_SLICE_K = BLOCK_K_WARPS > 1
    IS_HT = USE_HALF_TILE_INTERLEAVED
    BLOCK_M = TILE_M
    BLOCK_N = TILE_N
    BLOCK_K = TILE_K
    HALF_BLOCK_M = BLOCK_M // 2
    HALF_BLOCK_N = BLOCK_N // 2
    IS_FP8 = "fp8" in DTYPE_STR
    IS_FP8_PTPC = DTYPE_STR == "fp8_ptpc"
    WARP_SIZE = 64
    IN_DTYPE_BYTES = 1 if IS_FP8 else 2
    OUT_DTYPE_BYTES = 2
    LDG_VEC_SIZE = 16 if IS_FP8 else 8
    GPU_ARCH = get_rocm_arch()

    if IS_FP8:
        assert BLOCK_K >= 128
        WMMA_IMPL = WmmaFp8_m16n16k128()
        DMA_BYTES = 16
        MFMA_PER_WARP_K = 1
    elif GPU_ARCH == "gfx942":
        assert BLOCK_K >= 16
        WMMA_IMPL = WmmaHalf_m16n16k16(DTYPE_STR)
        DMA_BYTES = 4
        MFMA_PER_WARP_K = 2
    else:
        assert BLOCK_K >= 32
        WMMA_IMPL = WmmaHalf_m16n16k32(DTYPE_STR)
        DMA_BYTES = 16
        MFMA_PER_WARP_K = 1

    if IS_HT:
        assert STAGES == 2
        assert BLOCK_M_WARPS == 2
        assert BLOCK_N_WARPS >= 2
        assert BLOCK_K_WARPS == 1
        assert HALF_BLOCK_M * 2 == BLOCK_M
        assert HALF_BLOCK_N * 2 == BLOCK_N

    WMMA_M = WMMA_IMPL.WMMA_M
    WMMA_N = WMMA_IMPL.WMMA_N
    WMMA_K = WMMA_IMPL.WMMA_K
    WMMA_A_FRAG_VALUES = WMMA_IMPL.WMMA_A_FRAG_VALUES
    WMMA_B_FRAG_VALUES = WMMA_IMPL.WMMA_B_FRAG_VALUES
    WMMA_C_FRAG_VALUES = WMMA_IMPL.WMMA_C_FRAG_VALUES
    WARP_ATOM_M = WMMA_M
    WARP_ATOM_N = WMMA_N
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K
    WARP_GROUP_K = BLOCK_K_WARPS * WARP_ATOM_K

    WARP_K_STEPS = BLOCK_K // WARP_GROUP_K
    assert WARP_K_STEPS * WARP_GROUP_K == BLOCK_K

    K_SLICE = BLOCK_K // BLOCK_K_WARPS
    assert K_SLICE * BLOCK_K_WARPS == BLOCK_K
    assert K_SLICE % WARP_ATOM_K == 0

    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS * WARP_SIZE
    BLOCK_MN_WARPS = BLOCK_M_WARPS * BLOCK_N_WARPS
    if IS_HT:
        WARP_M_STEPS = HALF_BLOCK_M // BLOCK_M_WARPS // WARP_ATOM_M
        WARP_N_STEPS = HALF_BLOCK_N // BLOCK_N_WARPS // WARP_ATOM_N
        assert (WARP_M_STEPS >= 1) and (WARP_N_STEPS >= 1)
        assert HALF_BLOCK_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
        assert HALF_BLOCK_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    else:
        WARP_M_STEPS = BLOCK_M // BLOCK_M_WARPS // WARP_ATOM_M
        WARP_N_STEPS = BLOCK_N // BLOCK_N_WARPS // WARP_ATOM_N
        assert (WARP_M_STEPS >= 1) and (WARP_N_STEPS >= 1)
        assert BLOCK_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
        assert BLOCK_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    if IS_HT:
        assert HALF_BLOCK_M == BLOCK_M_WARPS * WARP_M
        assert HALF_BLOCK_N == BLOCK_N_WARPS * WARP_N
    else:
        assert BLOCK_M == BLOCK_M_WARPS * WARP_M
        assert BLOCK_N == BLOCK_N_WARPS * WARP_N
    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    BLOCK_MN_SIZE = BLOCK_M * BLOCK_N

    STG_SIZE_PER_M_STEP = BLOCK_M_WARPS * WARP_ATOM_M * HALF_BLOCK_N
    STG_WORK_SIZE_PER_M_STEP = STG_SIZE_PER_M_STEP // BLOCK_THREADS

    if IS_HT:
        assert (STG_SIZE_PER_M_STEP % BLOCK_THREADS == 0) and (
            STG_WORK_SIZE_PER_M_STEP >= 1
        )
        STG_VEC_SIZE = 8 if STG_WORK_SIZE_PER_M_STEP >= 8 else STG_WORK_SIZE_PER_M_STEP
        assert STG_VEC_SIZE in [8, 4]
        assert (STG_WORK_SIZE_PER_M_STEP % STG_VEC_SIZE == 0) and (
            STG_WORK_SIZE_PER_M_STEP // STG_VEC_SIZE >= 1
        )
    else:
        STG_VEC_SIZE = 8

    BLOCK_VECS_STG = STG_VEC_SIZE * BLOCK_THREADS
    STG_C_X_THREADS = BLOCK_N // STG_VEC_SIZE
    STG_C_QUAD_X_THREADS = HALF_BLOCK_N // STG_VEC_SIZE
    assert STG_C_X_THREADS * STG_VEC_SIZE == BLOCK_N
    STG_C_ITERS = BLOCK_MN_SIZE // BLOCK_VECS_STG
    assert STG_C_ITERS * BLOCK_VECS_STG == BLOCK_MN_SIZE
    BLOCK_K_BYTES = BLOCK_K * IN_DTYPE_BYTES
    LDG_ASYNC_VEC_SIZE = DMA_BYTES // IN_DTYPE_BYTES
    assert LDG_ASYNC_VEC_SIZE * IN_DTYPE_BYTES == DMA_BYTES
    LDG_A_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    assert LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE == BLOCK_K
    LDG_B_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    assert LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE == BLOCK_K
    assert (BLOCK_M * BLOCK_N) % (BLOCK_THREADS * STG_VEC_SIZE) == 0

    A_FRAGS_LEN = WARP_K_STEPS * WARP_M_STEPS
    B_FRAGS_LEN = WARP_K_STEPS * WARP_N_STEPS
    C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS

    if IS_HT:
        LDG_A_ITERS_AS = HALF_BLOCK_M * BLOCK_K // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
        assert (
            LDG_A_ITERS_AS * LDG_ASYNC_VEC_SIZE * BLOCK_THREADS
            == HALF_BLOCK_M * BLOCK_K
        )
        LDG_B_ITERS_AS = HALF_BLOCK_N * BLOCK_K // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
        assert (
            LDG_B_ITERS_AS * LDG_ASYNC_VEC_SIZE * BLOCK_THREADS
            == HALF_BLOCK_N * BLOCK_K
        )
        assert (2 * LDG_B_ITERS_AS + 2 * LDG_A_ITERS_AS) < 63
        assert HALF_BLOCK_N % STG_VEC_SIZE == 0
    else:
        LDG_A_ITERS_AS = BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
        assert LDG_A_ITERS_AS * LDG_ASYNC_VEC_SIZE * BLOCK_THREADS == BLOCK_MK_SIZE
        LDG_B_ITERS_AS = BLOCK_NK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
        assert LDG_B_ITERS_AS * LDG_ASYNC_VEC_SIZE * BLOCK_THREADS == BLOCK_NK_SIZE

    LDG_WAIT_COUNT = LDG_B_ITERS_AS + LDG_A_ITERS_AS

    # LDS parameters:
    AS_BYTES = STAGES * BLOCK_M * BLOCK_K * IN_DTYPE_BYTES
    SMEM_USE = AS_BYTES
    BS_BYTES = STAGES * BLOCK_N * BLOCK_K * IN_DTYPE_BYTES
    SMEM_USE += BS_BYTES
    SMEM_USE_ = max(SMEM_USE, BLOCK_K_WARPS * BLOCK_M * BLOCK_N * OUT_DTYPE_BYTES)
    assert SMEM_USE_ <= SMEM_CAPACITY_MAP[GPU_ARCH]

    return HGemmWmmaConstexprParam(
        DTYPE_ID=DTYPE_ID,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        STAGES=STAGES,
        SPLIT_K=SPLIT_K,
        BLOCK_M_WARPS=BLOCK_M_WARPS,
        BLOCK_N_WARPS=BLOCK_N_WARPS,
        BLOCK_K_WARPS=BLOCK_K_WARPS,
        HAS_BIAS=HAS_BIAS,
        HAS_K_TAIL=HAS_K_TAIL,
        GROUP_M=GROUP_M,
        USE_HALF_TILE_INTERLEAVED=USE_HALF_TILE_INTERLEAVED,
        # derived params
        IS_SPLIT_K=IS_SPLIT_K,
        IS_SLICE_K=IS_SLICE_K,
        HALF_BLOCK_M=HALF_BLOCK_M,
        HALF_BLOCK_N=HALF_BLOCK_N,
        IS_FP8=IS_FP8,
        IS_FP8_PTPC=IS_FP8_PTPC,
        WARP_SIZE=WARP_SIZE,
        IN_DTYPE_BYTES=IN_DTYPE_BYTES,
        OUT_DTYPE_BYTES=OUT_DTYPE_BYTES,
        LDG_VEC_SIZE=LDG_VEC_SIZE,
        DMA_BYTES=DMA_BYTES,
        MFMA_PER_WARP_K=MFMA_PER_WARP_K,
        WMMA_M=WMMA_M,
        WMMA_N=WMMA_N,
        WMMA_K=WMMA_K,
        WMMA_A_FRAG_VALUES=WMMA_A_FRAG_VALUES,
        WMMA_B_FRAG_VALUES=WMMA_B_FRAG_VALUES,
        WMMA_C_FRAG_VALUES=WMMA_C_FRAG_VALUES,
        WARP_ATOM_M=WARP_ATOM_M,
        WARP_ATOM_N=WARP_ATOM_N,
        WARP_ATOM_K=WARP_ATOM_K,
        WARP_K_STEPS=WARP_K_STEPS,
        K_SLICE=K_SLICE,
        BLOCK_THREADS=BLOCK_THREADS,
        BLOCK_MN_WARPS=BLOCK_MN_WARPS,
        WARP_M_STEPS=WARP_M_STEPS,
        WARP_N_STEPS=WARP_N_STEPS,
        WARP_M=WARP_M,
        WARP_N=WARP_N,
        STG_WORK_SIZE_PER_M_STEP=STG_WORK_SIZE_PER_M_STEP,
        STG_VEC_SIZE=STG_VEC_SIZE,
        STG_C_X_THREADS=STG_C_X_THREADS,
        STG_C_QUAD_X_THREADS=STG_C_QUAD_X_THREADS,
        STG_C_ITERS=STG_C_ITERS,
        BLOCK_K_BYTES=BLOCK_K_BYTES,
        LDG_ASYNC_VEC_SIZE=LDG_ASYNC_VEC_SIZE,
        LDG_A_X_THREADS_AS=LDG_A_X_THREADS_AS,
        LDG_B_X_THREADS_AS=LDG_B_X_THREADS_AS,
        LDG_A_ITERS_AS=LDG_A_ITERS_AS,
        LDG_B_ITERS_AS=LDG_B_ITERS_AS,
        LDG_WAIT_COUNT=LDG_WAIT_COUNT,
        A_FRAGS_LEN=A_FRAGS_LEN,
        B_FRAGS_LEN=B_FRAGS_LEN,
        C_FRAGS_LEN=C_FRAGS_LEN,
    )


def assert_hgemm_wmma_kernel(
    DTYPE_ID: int,
    TILE_M: int = 128,
    TILE_N: int = 128,
    TILE_K: int = 64,
    STAGES: int = 2,
    SPLIT_K: int = 1,
    BLOCK_M_WARPS: int = 2,
    BLOCK_N_WARPS: int = 2,
    BLOCK_K_WARPS: int = 1,
    HAS_BIAS: bool = False,
    HAS_K_TAIL: bool = False,
    GROUP_M: int = 0,
    USE_HALF_TILE_INTERLEAVED: bool = False,
    WMMA_STEP_CONSTRAIN: bool = False,
):
    param = init_hgemm_wmma_constexpr_param(
        DTYPE_ID,
        TILE_M,
        TILE_N,
        TILE_K,
        STAGES,
        SPLIT_K,
        BLOCK_M_WARPS,
        BLOCK_N_WARPS,
        BLOCK_K_WARPS,
        HAS_BIAS,
        HAS_K_TAIL,
        GROUP_M,
        USE_HALF_TILE_INTERLEAVED,
    )

    # NOTE: threshold for tuning
    if WMMA_STEP_CONSTRAIN:
        assert param.WARP_M_STEPS <= 4
        assert param.WARP_N_STEPS <= 4

    splitk_protocol = SplitKProtocol(
        param.SPLIT_K,
        param.BLOCK_M,
        param.BLOCK_N,
        param.STG_VEC_SIZE,
        param.OUT_DTYPE_BYTES,
        param.BLOCK_THREADS,
        param.HAS_BIAS,
    )
    assert splitk_protocol is not None

    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.GROUP_M
    )
    assert block_swizzle is not None

    return


def _make_hgemm_wmma_kernel_name(param: HGemmWmmaConstexprParam):
    dtype_str = HGEMM_DTYPE_STR_MAP[param.DTYPE_ID]
    out_dtype_str = "bf16" if "fp8" in dtype_str else dtype_str
    policy = "ht" if param.USE_HALF_TILE_INTERLEAVED else "ft"
    name = f"hgemm_a{dtype_str}_w{dtype_str}_{out_dtype_str}_t{param.BLOCK_M}x{param.BLOCK_N}x{param.BLOCK_K}x{param.STAGES}_ks{param.SPLIT_K}"
    name += f"_w{param.BLOCK_M_WARPS}x{param.BLOCK_N_WARPS}x{param.BLOCK_K_WARPS}_bias{int(param.HAS_BIAS)}_ktail{int(param.HAS_K_TAIL)}"
    name += f"_gm{param.GROUP_M}_p{policy}"
    name += "nt"
    return name


def _make_hgemm_wmma_impl(param: HGemmWmmaConstexprParam):
    GPU_ARCH = get_rocm_arch()
    dtype_str = HGEMM_DTYPE_STR_MAP[param.DTYPE_ID]
    if param.IS_FP8:
        return WmmaFp8_m16n16k128()
    elif GPU_ARCH == "gfx942":
        return WmmaHalf_m16n16k16(dtype_str)
    return WmmaHalf_m16n16k32(dtype_str)


@flyc.kernel
def hgemm_kernel(
    c_ptr: fx.Tensor,
    a_ptr: fx.Tensor,
    b_ptr: fx.Tensor,
    scale_a_ptr: fx.Tensor,
    scale_b_ptr: fx.Tensor,
    bias_ptr: fx.Tensor,
    semaphore_ptr: fx.Tensor,
    signal_ptr: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    working_k: fx.Int32,
    num_pid_m: fx.Int32,
    num_pid_n: fx.Int32,
    c_stride: fx.Int32,
    a_stride: fx.Int32,
    b_stride: fx.Int32,
    param: HGemmWmmaConstexprParam,
):
    DTYPE_ID = param.DTYPE_ID
    IS_SPLIT_K = param.IS_SPLIT_K
    IS_SLICE_K = param.IS_SLICE_K
    BLOCK_M = param.BLOCK_M
    BLOCK_N = param.BLOCK_N
    BLOCK_K = param.BLOCK_K
    IS_FP8 = param.IS_FP8
    IS_FP8_PTPC = param.IS_FP8_PTPC
    WARP_SIZE = param.WARP_SIZE
    STAGES = param.STAGES
    SPLIT_K = param.SPLIT_K
    BLOCK_N_WARPS = param.BLOCK_N_WARPS
    BLOCK_K_WARPS = param.BLOCK_K_WARPS
    HAS_BIAS = param.HAS_BIAS
    HAS_K_TAIL = param.HAS_K_TAIL
    IN_DTYPE_BYTES = param.IN_DTYPE_BYTES
    OUT_DTYPE_BYTES = param.OUT_DTYPE_BYTES
    LDG_VEC_SIZE = param.LDG_VEC_SIZE
    DMA_BYTES = param.DMA_BYTES
    MFMA_PER_WARP_K = param.MFMA_PER_WARP_K
    WMMA_IMPL = _make_hgemm_wmma_impl(param)
    WMMA_M = param.WMMA_M
    WMMA_N = param.WMMA_N
    WMMA_A_FRAG_VALUES = param.WMMA_A_FRAG_VALUES
    WMMA_B_FRAG_VALUES = param.WMMA_B_FRAG_VALUES
    WMMA_C_FRAG_VALUES = param.WMMA_C_FRAG_VALUES
    WARP_ATOM_M = param.WARP_ATOM_M
    WARP_ATOM_N = param.WARP_ATOM_N
    WARP_ATOM_K = param.WARP_ATOM_K
    WARP_K_STEPS = param.WARP_K_STEPS
    K_SLICE = param.K_SLICE
    BLOCK_THREADS = param.BLOCK_THREADS
    BLOCK_MN_WARPS = param.BLOCK_MN_WARPS
    WARP_M_STEPS = param.WARP_M_STEPS
    WARP_N_STEPS = param.WARP_N_STEPS
    WARP_M = param.WARP_M
    WARP_N = param.WARP_N
    BLOCK_K_BYTES = param.BLOCK_K_BYTES
    STG_VEC_SIZE = param.STG_VEC_SIZE
    STG_C_X_THREADS = param.STG_C_X_THREADS
    STG_C_ITERS = param.STG_C_ITERS
    LDG_ASYNC_VEC_SIZE = param.LDG_ASYNC_VEC_SIZE
    LDG_A_X_THREADS_AS = param.LDG_A_X_THREADS_AS
    LDG_B_X_THREADS_AS = param.LDG_B_X_THREADS_AS
    LDG_A_ITERS_AS = param.LDG_A_ITERS_AS
    LDG_B_ITERS_AS = param.LDG_B_ITERS_AS
    LDG_WAIT_COUNT = param.LDG_WAIT_COUNT
    splitk_protocol = SplitKProtocol(
        SPLIT_K,
        BLOCK_M,
        BLOCK_N,
        STG_VEC_SIZE,
        OUT_DTYPE_BYTES,
        BLOCK_THREADS,
        HAS_BIAS,
    )
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.GROUP_M
    )

    # kernel impl

    input_dtype_ = get_dtype_in_kernel(DTYPE_ID)
    output_dtype_ = fx.BFloat16 if const_expr(IS_FP8) else input_dtype_
    smem_input_dtype_ = fx.Int8 if const_expr(IS_FP8) else input_dtype_

    a_rsrc = make_buffer_rsrc(a_ptr)
    b_rsrc = make_buffer_rsrc(b_ptr)
    c_flat = fx.make_view(fx.get_iter(c_ptr), fx.make_layout(m * c_stride, 1))
    c_buf = rocdl.make_buffer_tensor(c_flat)
    c_vecs = fx.logical_divide(c_buf, fx.make_layout(STG_VEC_SIZE, 1))
    if const_expr(IS_FP8_PTPC):
        scale_a_buf = rocdl.make_buffer_tensor(scale_a_ptr)
        scale_b_buf = rocdl.make_buffer_tensor(scale_b_ptr)
    else:
        scale_a_buf = scale_b_buf = None
    if const_expr(HAS_BIAS):
        bias_buf = rocdl.make_buffer_tensor(bias_ptr)
    else:
        bias_buf = None

    smem_a_ptr, smem_b_ptr, smem_c_ptr = make_hgemm_lds(
        param, smem_input_dtype_, output_dtype_
    )

    tid = fx.thread_idx.x
    wid = tid // WARP_SIZE
    wid_mn = wid % BLOCK_MN_WARPS
    wid_k = wid // BLOCK_MN_WARPS
    w_tid = tid % WARP_SIZE

    block_m_idx, block_n_idx = block_swizzle.swizzle(
        num_pid_m, num_pid_n, fx.block_idx.x
    )
    ks_idx = fx.block_idx.y
    ks_begin = ks_idx * working_k
    ks_end = ks_begin + working_k
    ks_end = (ks_end < k).select(ks_end, k)

    block_m_offset = block_m_idx * BLOCK_M
    block_n_offset = block_n_idx * BLOCK_N
    k_blocks16 = BLOCK_K_BYTES // 16

    warp_m_idx = wid_mn // BLOCK_N_WARPS * WARP_M
    warp_n_idx = wid_mn % BLOCK_N_WARPS * WARP_N
    ldmatrix_a_m_idx = w_tid % WMMA_M
    ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
    ldmatrix_b_n_idx = w_tid % WMMA_N
    ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
    warp_k_slice_base = wid_k * K_SLICE

    acc_init = fx.full(WMMA_C_FRAG_VALUES, 0.0, fx.Float32)
    C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
    c_frags = [acc_init] * C_FRAGS_LEN
    stmatrix_c_n_idx = w_tid % WMMA_N

    if const_expr(HAS_BIAS and not IS_SPLIT_K and not IS_FP8_PTPC):
        bias_frags = [acc_init] * WARP_N_STEPS
        for jj in range_constexpr(WARP_N_STEPS):
            warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
            lds_n_idx = warp_atom_n_idx + stmatrix_c_n_idx
            global_n_idx = block_n_offset + lds_n_idx
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            bias_val = bias_buf[safe_global_n_idx].to(fx.Float32)
            if const_expr(IS_SLICE_K):
                is_first_k_slice = wid_k == 0
                bias_val = is_first_k_slice.select(bias_val, fx.Float32(0.0))
            bias_frags[jj] = vector.broadcast(
                T.vec(WMMA_C_FRAG_VALUES, T.f32), arith.unwrap(bias_val)
            )
        for ii in range_constexpr(WARP_M_STEPS):
            for jj in range_constexpr(WARP_N_STEPS):
                c_frags[ii * WARP_N_STEPS + jj] = bias_frags[jj]

    if const_expr(IS_SPLIT_K):
        signal_idx = fx.block_idx.x
        splitk_protocol.init(
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
            output_dtype_,
            signal_idx,
            c_stride,
        )

    def get_dma_copy_warp_offset():
        return rocdl.readfirstlane(T.i64, fx.Int64(wid * WARP_SIZE * DMA_BYTES))

    warp_offset = get_dma_copy_warp_offset()

    as_warp_ptr = fx.recast_iter(fx.Int8, smem_a_ptr) + warp_offset
    bs_warp_ptr = fx.recast_iter(fx.Int8, smem_b_ptr) + warp_offset

    def ldg_sts_a_async_one(ii, k_offset, write_stage, lds_ptr=None):
        global_tid = BLOCK_THREADS * ii + tid
        m_local_idx = global_tid // LDG_A_X_THREADS_AS
        k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
        if const_expr(IS_FP8_PTPC):
            col_in_bytes = swizzle_fp8_128(m_local_idx, k_local_idx)
        else:
            col_in_bytes = k_local_idx * IN_DTYPE_BYTES
            col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
        global_m_idx = block_m_offset + m_local_idx
        safe_global_m_idx = (global_m_idx < m).select(global_m_idx, 0)
        global_k_idx = k_offset + col_in_bytes // IN_DTYPE_BYTES
        if const_expr(HAS_K_TAIL):
            safe_global_k_idx = (global_k_idx < ks_end).select(global_k_idx, 0)
        else:
            safe_global_k_idx = global_k_idx
        global_offset_in_bytes = (
            safe_global_m_idx * a_stride + safe_global_k_idx
        ) * IN_DTYPE_BYTES
        global_offset_in_bytes = fx.Int32(global_offset_in_bytes)
        dynamic_bytes_offset = write_stage * BLOCK_M * BLOCK_K * IN_DTYPE_BYTES
        if const_expr(lds_ptr is None):
            lds_ptr = as_warp_ptr + dynamic_bytes_offset
        else:
            lds_ptr = lds_ptr + BLOCK_THREADS * DMA_BYTES
        buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset_in_bytes, DMA_BYTES)
        return lds_ptr

    def ldg_sts_b_async_one(ii, k_offset, write_stage, lds_ptr=None):
        global_tid = BLOCK_THREADS * ii + tid
        n_local_idx = global_tid // LDG_B_X_THREADS_AS
        k_local_idx = global_tid % LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
        if const_expr(IS_FP8_PTPC):
            col_in_bytes = swizzle_fp8_128(n_local_idx, k_local_idx)
        else:
            col_in_bytes = k_local_idx * IN_DTYPE_BYTES
            col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
        global_n_idx = block_n_offset + fx.Uint64(n_local_idx)
        safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
        global_k_idx = k_offset + col_in_bytes // IN_DTYPE_BYTES
        if const_expr(HAS_K_TAIL):
            safe_global_k_idx = (global_k_idx < ks_end).select(global_k_idx, 0)
        else:
            safe_global_k_idx = global_k_idx
        global_offset_in_bytes = (
            safe_global_n_idx * b_stride + safe_global_k_idx
        ) * IN_DTYPE_BYTES
        global_offset_in_bytes = fx.Int32(global_offset_in_bytes)
        dynamic_bytes_offset = write_stage * BLOCK_N * BLOCK_K * IN_DTYPE_BYTES
        if const_expr(lds_ptr is None):
            lds_ptr = bs_warp_ptr + dynamic_bytes_offset
        else:
            lds_ptr = lds_ptr + BLOCK_THREADS * DMA_BYTES
        buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset_in_bytes, DMA_BYTES)
        return lds_ptr

    def ldg_sts_a_async(k_offset, lds_stage):
        lds_ptr = None
        for ii in range_constexpr(LDG_A_ITERS_AS):
            lds_ptr = ldg_sts_a_async_one(
                ii, k_offset, lds_stage, lds_ptr if ii > 0 else None
            )

    def ldg_sts_b_async(k_offset, lds_stage):
        lds_ptr = None
        for ii in range_constexpr(LDG_B_ITERS_AS):
            lds_ptr = ldg_sts_b_async_one(
                ii, k_offset, lds_stage, lds_ptr if ii > 0 else None
            )

    def mask_tail_k_frag(frag, k_base, frag_values):
        elems = [0] * frag_values
        for vi in range_constexpr(frag_values):
            global_k_idx = k_base + vi
            valid_k = global_k_idx < ks_end
            elem = frag[vi]
            elems[vi] = valid_k.select(elem, 0)
        return fx.Vector.from_elements(elems)

    def ldmatrix_compute_tile_streaming(
        lds_stage, k_offset, c_frags, mask_k_tail=False
    ):
        s = lds_stage
        c_frags_new = [cx for cx in c_frags]
        for kk in range_constexpr(WARP_K_STEPS):
            warp_atom_k_idx = warp_k_slice_base + kk * WARP_ATOM_K
            # ldmatrix b
            b_frags = [0] * WARP_N_STEPS
            for ii in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                row = warp_atom_n_idx + ldmatrix_b_n_idx
                if const_expr(IS_FP8_PTPC):

                    def load_i32x4_at_b(col_delta):
                        col_base = warp_atom_k_idx + (w_tid // WMMA_N) * 16 + col_delta
                        col_swz = swizzle_fp8_128(row, col_base)
                        flat_offset = (s * BLOCK_N + row) * BLOCK_K + col_swz
                        v16 = fx.ptr_load(smem_b_ptr + flat_offset, fx.Vector.make_type(LDG_VEC_SIZE, smem_input_dtype_))
                        if const_expr(mask_k_tail):
                            v16 = mask_tail_k_frag(
                                v16, k_offset + col_base, LDG_VEC_SIZE
                            )
                        return v16.bitcast(fx.Int32)

                    lo = load_i32x4_at_b(0)
                    hi = load_i32x4_at_b(64)
                    b_frags[ii] = lo.shuffle(hi, list(range(8)))
                else:
                    frag_k_base = warp_atom_k_idx + ldmatrix_b_k_vec_idx
                    col_in_bytes = frag_k_base * IN_DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    flat_offset = (
                        s * BLOCK_N + row
                    ) * BLOCK_K + col_in_bytes // IN_DTYPE_BYTES
                    vec = fx.ptr_load(smem_b_ptr + flat_offset, fx.Vector.make_type(WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K, smem_input_dtype_))
                    if const_expr(mask_k_tail):
                        b_frags[ii] = mask_tail_k_frag(
                            vec,
                            k_offset + frag_k_base,
                            WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                        )
                    else:
                        b_frags[ii] = vec
            a_frags = [0] * WARP_M_STEPS
            for ii in range_constexpr(WARP_M_STEPS):
                # ldmatrix a
                warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                row = warp_atom_m_idx + ldmatrix_a_m_idx
                if const_expr(IS_FP8_PTPC):

                    def load_i32x4_at_a(col_delta):
                        col_base = warp_atom_k_idx + (w_tid // WMMA_M) * 16 + col_delta
                        col_swz = swizzle_fp8_128(row, col_base)
                        flat_offset = (s * BLOCK_M + row) * BLOCK_K + col_swz
                        v16 = fx.ptr_load(smem_a_ptr + flat_offset, fx.Vector.make_type(LDG_VEC_SIZE, smem_input_dtype_))
                        if const_expr(mask_k_tail):
                            v16 = mask_tail_k_frag(
                                v16, k_offset + col_base, LDG_VEC_SIZE
                            )
                        return v16.bitcast(fx.Int32)

                    lo = load_i32x4_at_a(0)
                    hi = load_i32x4_at_a(64)
                    a_frags[ii] = lo.shuffle(hi, list(range(8)))
                else:
                    frag_k_base = warp_atom_k_idx + ldmatrix_a_k_vec_idx
                    col_in_bytes = frag_k_base * IN_DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    flat_offset = (
                        s * BLOCK_M + row
                    ) * BLOCK_K + col_in_bytes // IN_DTYPE_BYTES
                    vec = fx.ptr_load(smem_a_ptr + flat_offset, fx.Vector.make_type(WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K, smem_input_dtype_))
                    if const_expr(mask_k_tail):
                        a_frags[ii] = mask_tail_k_frag(
                            vec,
                            k_offset + frag_k_base,
                            WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                        )
                    else:
                        a_frags[ii] = vec
            rocdl.sched_barrier(0)
            for ii in range_constexpr(WARP_M_STEPS):
                a_frag = a_frags[ii]
                for jj in range_constexpr(WARP_N_STEPS):
                    b_frag = b_frags[jj]
                    if const_expr(MFMA_PER_WARP_K == 2):
                        # split a
                        a_i64x2 = a_frag.bitcast(fx.Int64)
                        a_v0 = fx.Vector.from_elements([a_i64x2[0]]).bitcast(
                            fx.Float16
                        )
                        a_v1 = fx.Vector.from_elements([a_i64x2[1]]).bitcast(
                            fx.Float16
                        )
                        # split b
                        b_i64x2 = b_frag.bitcast(fx.Int64)
                        b_v0 = fx.Vector.from_elements([b_i64x2[0]]).bitcast(
                            fx.Float16
                        )
                        b_v1 = fx.Vector.from_elements([b_i64x2[1]]).bitcast(
                            fx.Float16
                        )
                        # wmma
                        c_idx = ii * WARP_N_STEPS + jj
                        acc_in = c_frags_new[c_idx]
                        acc_mid = WMMA_IMPL(a_v0, b_v0, acc_in)
                        c_frags_new[c_idx] = WMMA_IMPL(a_v1, b_v1, acc_mid)
                    elif const_expr(MFMA_PER_WARP_K == 1):
                        c_idx = ii * WARP_N_STEPS + jj
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a_frag, b_frag, c_frags_new[c_idx]
                        )
                    else:
                        raise NotImplementedError(
                            f"MFMA_PER_WARP_K={MFMA_PER_WARP_K} not supported"
                        )
        return c_frags_new

    if const_expr(IS_SPLIT_K):
        splitk_protocol.zero_c()

    for s in range_constexpr(STAGES - 1):
        ldg_sts_b_async(ks_begin + s * BLOCK_K, s)
        ldg_sts_a_async(ks_begin + s * BLOCK_K, s)
    rocdl.sched_barrier(0)

    def hot_loop_scheduler():
        # ================ Ordered ================
        for i in range_constexpr(LDG_B_ITERS_AS):
            rocdl.sched_vmem(1)  # ldg_sts_b_async next
        for i in range_constexpr(LDG_A_ITERS_AS):
            rocdl.sched_vmem(1)  # ldg_sts_a_async next
        for ki in range_constexpr(WARP_K_STEPS):
            for i in range_constexpr(WARP_N_STEPS):
                rocdl.sched_dsrd(1)  # lds_matrix_b current
            for i in range_constexpr(WARP_M_STEPS):
                rocdl.sched_dsrd(1)  # lds_matrix_a current
            for i in range_constexpr(WARP_M_STEPS):
                rocdl.sched_mfma(WARP_N_STEPS)
        # ================ Reordered ================
        rocdl.sched_barrier(0)

    BLOCK_K_LOOPS = (working_k + BLOCK_K - 1) // BLOCK_K
    init_state = [ks_begin, fx.Int32(0)] + c_frags
    for _, state in range(0, BLOCK_K_LOOPS - (STAGES - 1), 1, init=init_state):
        k_offset = state[0]
        current_stage = state[1]
        c_frags = state[2:]
        next_stage = (current_stage + 1) % STAGES
        write_stage = (current_stage + STAGES - 1) % STAGES
        __barrier((STAGES - 2) * LDG_WAIT_COUNT)
        ldg_sts_b_async(k_offset + (STAGES - 1) * BLOCK_K, write_stage)
        ldg_sts_a_async(k_offset + (STAGES - 1) * BLOCK_K, write_stage)
        c_frags_new = ldmatrix_compute_tile_streaming(current_stage, k_offset, c_frags)
        k_offset_next = k_offset + BLOCK_K
        hot_loop_scheduler()
        results = yield [k_offset_next, next_stage] + c_frags_new
    current_stage = results[1]
    k_offset = results[0]
    c_frags = results[2:]
    for s in range_constexpr(0, STAGES - 1):
        __barrier((STAGES - 2 - s) * LDG_WAIT_COUNT)
        mask_k_tail = HAS_K_TAIL
        c_frags = ldmatrix_compute_tile_streaming(
            current_stage, k_offset, c_frags, mask_k_tail
        )
        k_offset = k_offset + BLOCK_K
        current_stage = (current_stage + 1) % STAGES

    # write to lds
    stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
    gpu.barrier()
    for ii in range_constexpr(WARP_M_STEPS):
        warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
        for jj in range_constexpr(WARP_N_STEPS):
            warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
            for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                lds_m_idx = fx.Uint64(warp_atom_m_idx + stmatrix_c_m_vec_idx + kk)
                lds_n_idx = fx.Uint64(warp_atom_n_idx + stmatrix_c_n_idx)
                val = c_frags[ii * WARP_N_STEPS + jj][kk]
                if const_expr(IS_FP8_PTPC):
                    row_global = block_m_offset + lds_m_idx
                    col_global = block_n_offset + lds_n_idx
                    scale_a_offset = (row_global < m).select(row_global, 0)
                    scale_a = scale_a_buf[scale_a_offset]
                    scale_b_offset = (col_global < n).select(col_global, 0)
                    scale_b = scale_b_buf[scale_b_offset]
                    val = val * scale_a * scale_b
                    if const_expr(HAS_BIAS and not IS_SPLIT_K):
                        bias = bias_buf[scale_b_offset].to(fx.Float32)
                        val = val + bias
                val = val.to(output_dtype_)
                flat_offset = (wid_k * BLOCK_M + lds_m_idx) * BLOCK_N + lds_n_idx
                fx.ptr_store(val, smem_c_ptr + flat_offset)

    # write back to global
    if const_expr(IS_SPLIT_K):
        splitk_protocol.split_k_barrier()
    else:
        gpu.barrier()
    for i in range_constexpr(STG_C_ITERS):
        global_tid = BLOCK_THREADS * i + tid
        m_local_idx = global_tid // STG_C_X_THREADS
        n_local_idx = global_tid % STG_C_X_THREADS * STG_VEC_SIZE
        global_m_idx = block_m_offset + m_local_idx
        global_n_idx = block_n_offset + n_local_idx
        if (global_m_idx < m) and (global_n_idx < n):
            flat_offset = m_local_idx * BLOCK_N + n_local_idx
            c_vec = fx.ptr_load(smem_c_ptr + flat_offset, fx.Vector.make_type(STG_VEC_SIZE, output_dtype_))
            for ksi in range_constexpr(1, BLOCK_K_WARPS):
                peer_c_vec = fx.ptr_load(smem_c_ptr + (flat_offset + ksi * BLOCK_M * BLOCK_N), fx.Vector.make_type(STG_VEC_SIZE, output_dtype_))
                c_vec += peer_c_vec
            global_offset = global_m_idx * c_stride + global_n_idx
            if const_expr(IS_SPLIT_K):
                # split to vec2s
                vec2_ty = fx.Vector.make_type(2, output_dtype_)
                for vec_idx in range_constexpr(STG_VEC_SIZE // 2):
                    e0 = c_vec[vec_idx * 2]
                    e1 = c_vec[vec_idx * 2 + 1]
                    pair = fx.Vector.from_elements([e0, e1])
                    pair_v = (
                        pair._value if const_expr(hasattr(pair, "_value")) else pair
                    )
                    pair_ptr_v = get_llvm_ptr(
                        c_ptr,
                        global_offset + vec_idx * 2,
                        OUT_DTYPE_BYTES,
                        ir.Type.parse("!llvm.ptr<1>"),
                    )
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.fadd,
                        pair_ptr_v,
                        pair_v,
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=4,
                    )
            else:
                c_vecs[None, global_offset // STG_VEC_SIZE] = c_vec
    return


@flyc.kernel
def hgemm_ht_kernel(
    c_ptr: fx.Tensor,
    a_ptr: fx.Tensor,
    b_ptr: fx.Tensor,
    scale_a_ptr: fx.Tensor,
    scale_b_ptr: fx.Tensor,
    bias_ptr: fx.Tensor,
    semaphore_ptr: fx.Tensor,
    signal_ptr: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    working_k: fx.Int32,
    num_pid_m: fx.Int32,
    num_pid_n: fx.Int32,
    c_stride: fx.Int32,
    a_stride: fx.Int32,
    b_stride: fx.Int32,
    param: HGemmWmmaConstexprParam,
):
    DTYPE_ID = param.DTYPE_ID
    IS_SPLIT_K = param.IS_SPLIT_K
    BLOCK_M = param.BLOCK_M
    BLOCK_N = param.BLOCK_N
    BLOCK_K = param.BLOCK_K
    HALF_BLOCK_M = param.HALF_BLOCK_M
    HALF_BLOCK_N = param.HALF_BLOCK_N
    IS_FP8 = param.IS_FP8
    IS_FP8_PTPC = param.IS_FP8_PTPC
    WARP_SIZE = param.WARP_SIZE
    STAGES = param.STAGES
    SPLIT_K = param.SPLIT_K
    BLOCK_N_WARPS = param.BLOCK_N_WARPS
    BLOCK_K_WARPS = param.BLOCK_K_WARPS
    HAS_BIAS = param.HAS_BIAS
    HAS_K_TAIL = param.HAS_K_TAIL
    IN_DTYPE_BYTES = param.IN_DTYPE_BYTES
    OUT_DTYPE_BYTES = param.OUT_DTYPE_BYTES
    LDG_VEC_SIZE = param.LDG_VEC_SIZE
    DMA_BYTES = param.DMA_BYTES
    MFMA_PER_WARP_K = param.MFMA_PER_WARP_K
    WMMA_IMPL = _make_hgemm_wmma_impl(param)
    WMMA_M = param.WMMA_M
    WMMA_N = param.WMMA_N
    WMMA_K = param.WMMA_K
    WMMA_A_FRAG_VALUES = param.WMMA_A_FRAG_VALUES
    WMMA_B_FRAG_VALUES = param.WMMA_B_FRAG_VALUES
    WMMA_C_FRAG_VALUES = param.WMMA_C_FRAG_VALUES
    WARP_ATOM_M = param.WARP_ATOM_M
    WARP_ATOM_N = param.WARP_ATOM_N
    WARP_ATOM_K = param.WARP_ATOM_K
    WARP_K_STEPS = param.WARP_K_STEPS
    BLOCK_THREADS = param.BLOCK_THREADS
    WARP_M_STEPS = param.WARP_M_STEPS
    WARP_N_STEPS = param.WARP_N_STEPS
    WARP_M = param.WARP_M
    WARP_N = param.WARP_N
    BLOCK_K_BYTES = param.BLOCK_K_BYTES
    STG_WORK_SIZE_PER_M_STEP = param.STG_WORK_SIZE_PER_M_STEP
    STG_VEC_SIZE = param.STG_VEC_SIZE
    STG_C_QUAD_X_THREADS = param.STG_C_QUAD_X_THREADS
    LDG_ASYNC_VEC_SIZE = param.LDG_ASYNC_VEC_SIZE
    LDG_A_X_THREADS_AS = param.LDG_A_X_THREADS_AS
    LDG_B_X_THREADS_AS = param.LDG_B_X_THREADS_AS
    LDG_A_ITERS_AS = param.LDG_A_ITERS_AS
    LDG_B_ITERS_AS = param.LDG_B_ITERS_AS
    A_FRAGS_LEN = param.A_FRAGS_LEN
    B_FRAGS_LEN = param.B_FRAGS_LEN
    C_FRAGS_LEN = param.C_FRAGS_LEN
    splitk_protocol = SplitKProtocol(
        SPLIT_K,
        BLOCK_M,
        BLOCK_N,
        STG_VEC_SIZE,
        OUT_DTYPE_BYTES,
        BLOCK_THREADS,
        HAS_BIAS,
    )
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.GROUP_M
    )

    # kernel impl

    input_dtype_ = get_dtype_in_kernel(DTYPE_ID)
    output_dtype_ = fx.BFloat16 if const_expr(IS_FP8) else input_dtype_
    smem_input_dtype_ = fx.Int8 if const_expr(IS_FP8) else input_dtype_
    acc_init = fx.full(WMMA_C_FRAG_VALUES, 0.0, fx.Float32)

    a_rsrc = make_buffer_rsrc(a_ptr)
    b_rsrc = make_buffer_rsrc(b_ptr)
    c_flat = fx.make_view(fx.get_iter(c_ptr), fx.make_layout(m * c_stride, 1))
    c_buf = rocdl.make_buffer_tensor(c_flat)
    c_vecs = fx.logical_divide(c_buf, fx.make_layout(STG_VEC_SIZE, 1))
    if const_expr(IS_FP8_PTPC):
        scale_a_buf = rocdl.make_buffer_tensor(scale_a_ptr)
        scale_b_buf = rocdl.make_buffer_tensor(scale_b_ptr)
    else:
        scale_a_buf = scale_b_buf = None
    if const_expr(HAS_BIAS):
        bias_buf = rocdl.make_buffer_tensor(bias_ptr)
    else:
        bias_buf = None

    smem_a_ptr, smem_b_ptr, smem_c_ptr = make_hgemm_lds(
        param, smem_input_dtype_, output_dtype_
    )

    tid = fx.thread_idx.x
    wid = tid // WARP_SIZE
    w_tid = tid % WARP_SIZE

    block_m_idx, block_n_idx = block_swizzle.swizzle(
        num_pid_m, num_pid_n, fx.block_idx.x
    )
    ks_idx = fx.block_idx.y
    ks_begin = ks_idx * working_k
    ks_end = ks_begin + working_k
    ks_end = (ks_end < k).select(ks_end, k)
    block_m_offset = block_m_idx * BLOCK_M
    block_n_offset = block_n_idx * BLOCK_N
    k_blocks16 = BLOCK_K_BYTES // 16

    warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
    warp_n_idx = wid % BLOCK_N_WARPS * WARP_N
    ldmatrix_a_m_idx = w_tid % WMMA_M
    ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
    ldmatrix_b_n_idx = w_tid % WMMA_N
    ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K

    c_frags = [acc_init] * (4 * C_FRAGS_LEN)
    stmatrix_c_n_idx = w_tid % WMMA_N
    stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES

    if const_expr(HAS_BIAS and not IS_SPLIT_K and not IS_FP8_PTPC):
        bias_frags = [acc_init] * (2 * WARP_N_STEPS)
        for n_part in range_constexpr(2):
            for ni in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + ni * WARP_ATOM_N
                lds_n_idx = n_part * HALF_BLOCK_N + warp_atom_n_idx + stmatrix_c_n_idx
                global_n_idx = block_n_offset + lds_n_idx
                safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
                bias_val = bias_buf[safe_global_n_idx].to(fx.Float32)
                bias_frags[n_part * WARP_N_STEPS + ni] = vector.broadcast(
                    T.vec(WMMA_C_FRAG_VALUES, T.f32), bias_val
                )
        for m_part in range_constexpr(2):
            for n_part in range_constexpr(2):
                for mi in range_constexpr(WARP_M_STEPS):
                    for ni in range_constexpr(WARP_N_STEPS):
                        c_frags[
                            (m_part * 2 + n_part) * C_FRAGS_LEN + mi * WARP_N_STEPS + ni
                        ] = bias_frags[n_part * WARP_N_STEPS + ni]

    swizzle_cache_a = [0] * LDG_A_ITERS_AS
    for i in range_constexpr(LDG_A_ITERS_AS):
        global_tid = BLOCK_THREADS * i + tid
        m_local_idx = global_tid // LDG_A_X_THREADS_AS
        k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
        if const_expr(IS_FP8_PTPC):
            col_in_bytes = swizzle_fp8_128(m_local_idx, k_local_idx)
        else:
            col_in_bytes = swizzle_xor16(
                m_local_idx, k_local_idx * IN_DTYPE_BYTES, k_blocks16
            )
        swizzle_cache_a[i] = col_in_bytes // IN_DTYPE_BYTES

    swizzle_cache_b = [0] * LDG_B_ITERS_AS
    for i in range_constexpr(LDG_B_ITERS_AS):
        global_tid = BLOCK_THREADS * i + tid
        n_local_idx = global_tid // LDG_B_X_THREADS_AS
        k_local_idx = global_tid % LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
        if const_expr(IS_FP8_PTPC):
            col_in_bytes = swizzle_fp8_128(n_local_idx, k_local_idx)
        else:
            col_in_bytes = swizzle_xor16(
                n_local_idx, k_local_idx * IN_DTYPE_BYTES, k_blocks16
            )
        swizzle_cache_b[i] = col_in_bytes // IN_DTYPE_BYTES

    if const_expr(IS_SPLIT_K):
        signal_idx = fx.block_idx.x
        splitk_protocol.init(
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
            output_dtype_,
            signal_idx,
            c_stride,
        )

    def get_dma_copy_warp_offset():
        return rocdl.readfirstlane(T.i64, fx.Int64(wid * WARP_SIZE * DMA_BYTES))

    warp_offset = get_dma_copy_warp_offset()

    as_warp_ptr = fx.recast_iter(fx.Int8, smem_a_ptr) + warp_offset
    bs_warp_ptr = fx.recast_iter(fx.Int8, smem_b_ptr) + warp_offset

    def ldg_sts_a_async_one(m_part, k_buf, k_offset, ii, lds_ptr=None):
        global_tid = BLOCK_THREADS * ii + tid
        m_local_idx = global_tid // LDG_A_X_THREADS_AS
        global_m_idx = block_m_offset + m_part * HALF_BLOCK_M + m_local_idx
        safe_global_m_idx = (global_m_idx < m).select(global_m_idx, 0)
        global_k_idx = k_offset + k_buf * BLOCK_K + swizzle_cache_a[ii]
        if const_expr(HAS_K_TAIL):
            safe_global_k_idx = (global_k_idx < ks_end).select(global_k_idx, 0)
        else:
            safe_global_k_idx = global_k_idx
        global_offset = (
            safe_global_m_idx * a_stride + safe_global_k_idx
        ) * IN_DTYPE_BYTES
        global_offset = fx.Int32(global_offset)
        static_bytes_offset = (
            (k_buf * 2 + m_part) * HALF_BLOCK_M * BLOCK_K * IN_DTYPE_BYTES
        )
        if const_expr(lds_ptr is None):
            lds_ptr = as_warp_ptr + static_bytes_offset
        else:
            lds_ptr = lds_ptr + BLOCK_THREADS * DMA_BYTES
        buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, DMA_BYTES)
        return lds_ptr

    def ldg_sts_b_async_one(n_part, k_buf, k_offset, ii, lds_ptr=None):
        global_tid = BLOCK_THREADS * ii + tid
        n_local_idx = global_tid // LDG_B_X_THREADS_AS
        global_n_idx = block_n_offset + n_part * HALF_BLOCK_N + n_local_idx
        safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
        global_k_idx = k_offset + k_buf * BLOCK_K + swizzle_cache_b[ii]
        if const_expr(HAS_K_TAIL):
            safe_global_k_idx = (global_k_idx < ks_end).select(global_k_idx, 0)
        else:
            safe_global_k_idx = global_k_idx
        global_offset = (
            safe_global_n_idx * b_stride + safe_global_k_idx
        ) * IN_DTYPE_BYTES
        global_offset = fx.Int32(global_offset)
        static_bytes_offset = (
            (k_buf * 2 + n_part) * HALF_BLOCK_N * BLOCK_K * IN_DTYPE_BYTES
        )
        if const_expr(lds_ptr is None):
            lds_ptr = bs_warp_ptr + static_bytes_offset
        else:
            lds_ptr = lds_ptr + BLOCK_THREADS * DMA_BYTES
        buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset, DMA_BYTES)
        return lds_ptr

    def ldg_sts_a_async(m_part, k_buf, k_offset):
        lds_ptr = None
        for i in range_constexpr(LDG_A_ITERS_AS):
            lds_ptr = ldg_sts_a_async_one(
                m_part, k_buf, k_offset, i, lds_ptr if i > 0 else None
            )

    def ldg_sts_b_async(n_part, k_buf, k_offset):
        lds_ptr = None
        for i in range_constexpr(LDG_B_ITERS_AS):
            lds_ptr = ldg_sts_b_async_one(
                n_part, k_buf, k_offset, i, lds_ptr if i > 0 else None
            )

    def mask_tail_k_frag(frag, k_base, frag_values):
        elems = [0] * frag_values
        for vi in range_constexpr(frag_values):
            global_k_idx = k_base + vi
            valid_k = global_k_idx < ks_end
            elem = frag[vi]
            elems[vi] = valid_k.select( elem, smem_input_dtype_(0)
            )
        return fx.Vector.from_elements(elems)

    def ldmatrix_a(m_part, k_buf, k_tile_offset, mask_k_tail=False):
        a_frags = [0] * A_FRAGS_LEN
        for mi in range_constexpr(WARP_M_STEPS):
            warp_atom_m_idx = warp_m_idx + mi * WARP_ATOM_M
            for ki in range_constexpr(WARP_K_STEPS):
                if const_expr(IS_FP8_PTPC):
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                else:
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                    warp_atom_k_idx = ki * WARP_ATOM_K
                    col_in_bytes = (
                        warp_atom_k_idx + ldmatrix_a_k_vec_idx
                    ) * IN_DTYPE_BYTES
                if const_expr(IS_FP8_PTPC):

                    def load_i32x4_at(col_delta):
                        col_base = ki * WMMA_K + (w_tid // WMMA_M) * 16
                        col_swz = swizzle_fp8_128(row, col_base + col_delta)
                        flat_offset = (
                            (k_buf * 2 + m_part) * HALF_BLOCK_M + row
                        ) * BLOCK_K + col_swz
                        v16 = fx.ptr_load(smem_a_ptr + flat_offset, fx.Vector.make_type(LDG_VEC_SIZE, smem_input_dtype_))
                        if const_expr(mask_k_tail):
                            v16 = mask_tail_k_frag(
                                v16,
                                k_tile_offset + col_base + col_delta,
                                LDG_VEC_SIZE,
                            )
                        return v16.bitcast(fx.Int32)

                    lo = load_i32x4_at(0)
                    hi = load_i32x4_at(64)
                    a_frags[ki * WARP_M_STEPS + mi] = lo.shuffle(
                        hi, list(range(8))
                    )
                else:
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    flat_offset = (
                        (k_buf * 2 + m_part) * HALF_BLOCK_M + row
                    ) * BLOCK_K + col_in_bytes // IN_DTYPE_BYTES
                    vec = fx.ptr_load(smem_a_ptr + flat_offset, fx.Vector.make_type(WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K, smem_input_dtype_))
                    if const_expr(mask_k_tail):
                        a_frags[ki * WARP_M_STEPS + mi] = mask_tail_k_frag(
                            vec,
                            k_tile_offset + warp_atom_k_idx + ldmatrix_a_k_vec_idx,
                            WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                        )
                    else:
                        a_frags[ki * WARP_M_STEPS + mi] = vec
        return a_frags

    def ldmatrix_b(n_part, k_buf, k_tile_offset, mask_k_tail=False):
        b_frags = [0] * B_FRAGS_LEN
        for ni in range_constexpr(WARP_N_STEPS):
            warp_atom_n_idx = warp_n_idx + ni * WARP_ATOM_N
            for ki in range_constexpr(WARP_K_STEPS):
                if const_expr(IS_FP8_PTPC):
                    row = warp_atom_n_idx + ldmatrix_b_n_idx
                else:
                    row = warp_atom_n_idx + ldmatrix_b_n_idx
                    warp_atom_k_idx = ki * WARP_ATOM_K
                    col_in_bytes = (
                        warp_atom_k_idx + ldmatrix_b_k_vec_idx
                    ) * IN_DTYPE_BYTES
                if const_expr(IS_FP8_PTPC):

                    def load_i32x4_at(col_delta):
                        col_base = ki * WMMA_K + (w_tid // WMMA_N) * 16
                        col_swz = swizzle_fp8_128(row, col_base + col_delta)
                        flat_offset = (
                            (k_buf * 2 + n_part) * HALF_BLOCK_N + row
                        ) * BLOCK_K + col_swz
                        v16 = fx.ptr_load(smem_b_ptr + flat_offset, fx.Vector.make_type(LDG_VEC_SIZE, smem_input_dtype_))
                        if const_expr(mask_k_tail):
                            v16 = mask_tail_k_frag(
                                v16,
                                k_tile_offset + col_base + col_delta,
                                LDG_VEC_SIZE,
                            )
                        return v16.bitcast(fx.Int32)

                    lo = load_i32x4_at(0)
                    hi = load_i32x4_at(64)
                    b_frags[ki * WARP_N_STEPS + ni] = lo.shuffle(
                        hi, list(range(8))
                    )
                else:
                    col_in_bytes = swizzle_xor16(
                        row, col_in_bytes, fx.Int32(k_blocks16)
                    )
                    flat_offset = (
                        (k_buf * 2 + n_part) * HALF_BLOCK_N + row
                    ) * BLOCK_K + col_in_bytes // IN_DTYPE_BYTES
                    vec = fx.ptr_load(smem_b_ptr + flat_offset, fx.Vector.make_type(WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K, smem_input_dtype_))
                    if const_expr(mask_k_tail):
                        b_frags[ki * WARP_N_STEPS + ni] = mask_tail_k_frag(
                            vec,
                            k_tile_offset + warp_atom_k_idx + ldmatrix_b_k_vec_idx,
                            WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                        )
                    else:
                        b_frags[ki * WARP_N_STEPS + ni] = vec
        return b_frags

    def consume(m_part, n_part, a_frags, b_frags, c_frags_in, emit_sched_barrier):
        c_frags_out = [cx for cx in c_frags_in]
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)
        for mi in range_constexpr(WARP_M_STEPS):
            for ni in range_constexpr(WARP_N_STEPS):
                for ki in range_constexpr(WARP_K_STEPS):
                    c_idx = (m_part * 2 + n_part) * C_FRAGS_LEN + mi * WARP_N_STEPS + ni
                    c_frags_out[c_idx] = WMMA_IMPL(
                        a_frags[ki * WARP_M_STEPS + mi],
                        b_frags[ki * WARP_N_STEPS + ni],
                        c_frags_out[c_idx],
                    )
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)
        return c_frags_out

    if const_expr(IS_SPLIT_K):
        splitk_protocol.zero_c()

    ldg_sts_b_async(0, 0, ks_begin)
    ldg_sts_a_async(0, 0, ks_begin)
    ldg_sts_b_async(1, 0, ks_begin)
    ldg_sts_a_async(1, 0, ks_begin)
    rocdl.sched_barrier(0)
    if wid // BLOCK_N_WARPS == 1:
        __s_barrier()
    rocdl.sched_barrier(0)
    __s_barrier()
    rocdl.sched_barrier(0)
    ldg_sts_b_async(0, 1, ks_begin)
    ldg_sts_a_async(0, 1, ks_begin)
    ldg_sts_b_async(1, 1, ks_begin)
    __barrier(1 * LDG_B_ITERS_AS + 1 * LDG_A_ITERS_AS)

    def compute_double_tile(k_offset, c_frags_in):
        next_k_offset = k_offset + 2 * BLOCK_K
        # 0
        b0_frags = ldmatrix_b(0, 0, k_offset)
        a0_frags = ldmatrix_a(0, 0, k_offset)
        ldg_sts_a_async(1, 1, k_offset)
        __s_barrier()
        c_frags_out = consume(0, 0, a0_frags, b0_frags, c_frags_in, True)
        __s_barrier()
        b1_frags = ldmatrix_b(1, 0, k_offset)
        ldg_sts_b_async(0, 0, next_k_offset)
        __s_barrier()
        c_frags_out = consume(0, 1, a0_frags, b1_frags, c_frags_out, True)
        __s_barrier()
        a1_frags = ldmatrix_a(1, 0, k_offset)
        ldg_sts_a_async(0, 0, next_k_offset)
        __s_barrier()
        c_frags_out = consume(1, 0, a1_frags, b0_frags, c_frags_out, True)
        __s_barrier()
        b0_frags = ldmatrix_b(0, 1, k_offset + BLOCK_K)
        ldg_sts_b_async(1, 0, next_k_offset)
        __barrier(2 * LDG_B_ITERS_AS + 1 * LDG_A_ITERS_AS)
        c_frags_out = consume(1, 1, a1_frags, b1_frags, c_frags_out, True)
        __s_barrier()
        # 1
        a0_frags = ldmatrix_a(0, 1, k_offset + BLOCK_K)
        ldg_sts_a_async(1, 0, next_k_offset)
        __s_barrier()
        c_frags_out = consume(0, 0, a0_frags, b0_frags, c_frags_out, True)
        __s_barrier()
        b1_frags = ldmatrix_b(1, 1, k_offset + BLOCK_K)
        ldg_sts_b_async(0, 1, next_k_offset)
        __s_barrier()
        c_frags_out = consume(0, 1, a0_frags, b1_frags, c_frags_out, True)
        __s_barrier()
        a1_frags = ldmatrix_a(1, 1, k_offset + BLOCK_K)
        ldg_sts_a_async(0, 1, next_k_offset)
        __s_barrier()
        c_frags_out = consume(1, 0, a1_frags, b0_frags, c_frags_out, True)
        __s_barrier()
        ldg_sts_b_async(1, 1, next_k_offset)
        __barrier(1 * LDG_B_ITERS_AS + 1 * LDG_A_ITERS_AS)
        c_frags_out = consume(1, 1, a1_frags, b1_frags, c_frags_out, True)
        __s_barrier()
        return c_frags_out

    BLOCK_K_LOOPS = (working_k + BLOCK_K - 1) // BLOCK_K
    loop_end = (BLOCK_K_LOOPS > 2).select(BLOCK_K_LOOPS - 2, 0)
    init_state = [ks_begin] + c_frags
    for _, state in range(0, loop_end, 2, init=init_state):
        k_offset = state[0]
        c_frags = state[1:]
        c_frags = compute_double_tile(k_offset, c_frags)
        results = yield [k_offset + 2 * BLOCK_K] + c_frags
    k_offset = results[0]
    c_frags = results[1:]

    def load_scale_b(n_part):
        scale_b_frags = [fx.Float32(0.0)] * WARP_N_STEPS
        for ni in range_constexpr(WARP_N_STEPS):
            warp_atom_n_idx = warp_n_idx + ni * WARP_ATOM_N
            local_n_idx = fx.Uint64(
                n_part * HALF_BLOCK_N + warp_atom_n_idx + stmatrix_c_n_idx
            )
            global_n_offset = block_n_offset + local_n_idx
            scale_b_offset = (global_n_offset < n).select(global_n_offset, 0)
            scale_b = scale_b_buf[scale_b_offset]
            scale_b_frags[ni] = scale_b
        return scale_b_frags

    def store_matrix_to_lds(m_, n_, c_frags, scale_b_frags=None):
        c_base = (m_ * 2 + n_) * C_FRAGS_LEN
        for mi in range_constexpr(WARP_M_STEPS):
            warp_atom_m_idx = warp_m_idx + mi * WARP_ATOM_M
            lds_m_base = m_ * HALF_BLOCK_M + warp_atom_m_idx + stmatrix_c_m_vec_idx
            for ni in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + ni * WARP_ATOM_N
                c_idx = c_base + mi * WARP_N_STEPS + ni
                lds_n_idx = n_ * HALF_BLOCK_N + warp_atom_n_idx + stmatrix_c_n_idx
                col_global = block_n_offset + lds_n_idx
                safe_col_global = (col_global < n).select(col_global, 0)
                if const_expr(HAS_BIAS and IS_FP8_PTPC and not IS_SPLIT_K):
                    bias = bias_buf[safe_col_global].to(fx.Float32)
                if const_expr(IS_FP8_PTPC):
                    scale_b = scale_b_frags[ni]
                for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                    lds_m_idx = lds_m_base + kk
                    val = c_frags[c_idx][kk]
                    if const_expr(IS_FP8_PTPC):
                        row_global = block_m_offset + lds_m_idx
                        scale_a_offset = (row_global < m).select(row_global, 0)
                        scale_a = scale_a_buf[scale_a_offset]
                        val = val * scale_a * scale_b
                        if const_expr(HAS_BIAS and not IS_SPLIT_K):
                            val = val + bias
                    val = val.to(output_dtype_)
                    flat_offset = lds_m_idx * BLOCK_N + lds_n_idx
                    fx.ptr_store(val, smem_c_ptr + flat_offset)

    def atomic_add_vec_to_c(global_m_idx, global_n_idx, vec):
        global_offset = global_m_idx * c_stride + global_n_idx
        vec2_ty = fx.Vector.make_type(2, output_dtype_)
        for vec_idx in range_constexpr(STG_VEC_SIZE // 2):
            e0 = vec[vec_idx * 2]
            e1 = vec[vec_idx * 2 + 1]
            pair = fx.Vector.from_elements([e0, e1])
            pair_v = pair._value if const_expr(hasattr(pair, "_value")) else pair
            pair_ptr_v = get_llvm_ptr(
                c_ptr,
                global_offset + vec_idx * 2,
                OUT_DTYPE_BYTES,
                ir.Type.parse("!llvm.ptr<1>"),
            )
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.fadd,
                pair_ptr_v,
                pair_v,
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )

    def store_vec_to_c(global_m_idx, global_n_idx, vec):
        if const_expr(IS_SPLIT_K):
            atomic_add_vec_to_c(global_m_idx, global_n_idx, vec)
        else:
            global_offset = global_m_idx * c_stride + global_n_idx
            c_vecs[None, global_offset // STG_VEC_SIZE] = vec

    def store_matrix_from_lds(m_, n_):
        for mi in range_constexpr(WARP_M_STEPS):
            for i in range_constexpr(STG_WORK_SIZE_PER_M_STEP // STG_VEC_SIZE):
                global_tid = BLOCK_THREADS * i + tid
                m_band_idx = global_tid // STG_C_QUAD_X_THREADS
                n_local_idx = global_tid % STG_C_QUAD_X_THREADS * STG_VEC_SIZE
                warp_m_band = m_band_idx // WARP_ATOM_M
                atom_m_idx = m_band_idx % WARP_ATOM_M
                m_tile_idx = (
                    m_ * HALF_BLOCK_M
                    + warp_m_band * WARP_M
                    + mi * WARP_ATOM_M
                    + atom_m_idx
                )
                n_tile_idx = n_ * HALF_BLOCK_N + n_local_idx
                global_m_idx = block_m_offset + m_tile_idx
                global_n_idx = block_n_offset + n_tile_idx

                if (global_m_idx < m) and (global_n_idx < n):
                    flat_offset = m_tile_idx * BLOCK_N + n_tile_idx
                    c_vec = fx.ptr_load(smem_c_ptr + flat_offset, fx.Vector.make_type(STG_VEC_SIZE, output_dtype_))
                    store_vec_to_c(
                        global_m_idx,
                        global_n_idx,
                        c_vec,
                    )

    def compute_final_tile_and_epilogue(k_offset, c_frags_in):
        # 0
        b0_frags = ldmatrix_b(0, 0, k_offset, HAS_K_TAIL)
        a0_frags = ldmatrix_a(0, 0, k_offset, HAS_K_TAIL)
        ldg_sts_a_async(1, 1, k_offset)
        __s_barrier()
        c_frags_out = consume(0, 0, a0_frags, b0_frags, c_frags_in, True)
        __s_barrier()
        b1_frags = ldmatrix_b(1, 0, k_offset, HAS_K_TAIL)
        __s_barrier()
        c_frags_out = consume(0, 1, a0_frags, b1_frags, c_frags_out, True)
        __s_barrier()
        a1_frags = ldmatrix_a(1, 0, k_offset, HAS_K_TAIL)
        __s_barrier()
        c_frags_out = consume(1, 0, a1_frags, b0_frags, c_frags_out, True)
        __s_barrier()
        b0_frags = ldmatrix_b(0, 1, k_offset + BLOCK_K, HAS_K_TAIL)
        __s_barrier()
        c_frags_out = consume(1, 1, a1_frags, b1_frags, c_frags_out, True)
        # 1
        __barrier(0)
        a0_frags = ldmatrix_a(0, 1, k_offset + BLOCK_K, HAS_K_TAIL)
        __s_barrier()
        c_frags_out = consume(0, 0, a0_frags, b0_frags, c_frags_out, True)
        __s_barrier()
        b1_frags = ldmatrix_b(1, 1, k_offset + BLOCK_K, HAS_K_TAIL)
        __s_barrier()
        c_frags_out = consume(0, 1, a0_frags, b1_frags, c_frags_out, True)
        __s_barrier()
        a1_frags = ldmatrix_a(1, 1, k_offset + BLOCK_K, HAS_K_TAIL)
        __s_barrier()
        rocdl.sched_barrier(0)
        if const_expr(IS_FP8_PTPC):
            scale_b0 = load_scale_b(0)
            store_matrix_to_lds(0, 0, c_frags_out, scale_b0)
            scale_b1 = load_scale_b(1)
            store_matrix_to_lds(0, 1, c_frags_out, scale_b1)
        else:
            store_matrix_to_lds(0, 0, c_frags_out)
            store_matrix_to_lds(0, 1, c_frags_out)
        c_frags_out = consume(1, 0, a1_frags, b0_frags, c_frags_out, False)
        rocdl.sched_barrier(0)
        __s_barrier()
        rocdl.sched_barrier(0)
        if const_expr(IS_SPLIT_K):
            splitk_protocol.split_k_barrier()
        store_matrix_from_lds(0, 0)
        store_matrix_from_lds(0, 1)
        if const_expr(IS_FP8_PTPC):
            store_matrix_to_lds(1, 0, c_frags_out, scale_b0)
        else:
            store_matrix_to_lds(1, 0, c_frags_out)
        c_frags_out = consume(1, 1, a1_frags, b1_frags, c_frags_out, False)
        rocdl.sched_barrier(0)
        __s_barrier()
        store_matrix_from_lds(1, 0)
        if const_expr(IS_FP8_PTPC):
            store_matrix_to_lds(1, 1, c_frags_out, scale_b1)
        else:
            store_matrix_to_lds(1, 1, c_frags_out)
        __s_barrier()
        store_matrix_from_lds(1, 1)
        return c_frags_out

    c_frags = compute_final_tile_and_epilogue(k_offset, c_frags)
    return


@flyc.jit
def hgemm_wmma(
    c_ptr: fx.Tensor,
    a_ptr: fx.Tensor,
    b_ptr: fx.Tensor,
    scale_a_ptr: fx.Tensor,
    scale_b_ptr: fx.Tensor,
    bias_ptr: fx.Tensor,
    semaphore_ptr: fx.Tensor,
    signal_ptr: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    stream: fx.Stream,
    param: HGemmWmmaConstexprParam,
):
    working_k = (k + param.SPLIT_K - 1) // param.SPLIT_K
    num_pid_m = (m + param.BLOCK_M - 1) // param.BLOCK_M
    num_pid_n = (n + param.BLOCK_N - 1) // param.BLOCK_N
    c_stride = n
    a_stride = k
    b_stride = k
    hgemm_kernel_impl = (
        hgemm_ht_kernel if param.USE_HALF_TILE_INTERLEAVED else hgemm_kernel
    )
    hgemm_kernel_impl._func.__name__ = _make_hgemm_wmma_kernel_name(param)
    hgemm_kernel_impl._known_block_size = [param.BLOCK_THREADS, 1, 1]
    hgemm_kernel_impl(
        c_ptr,
        a_ptr,
        b_ptr,
        scale_a_ptr,
        scale_b_ptr,
        bias_ptr,
        semaphore_ptr,
        signal_ptr,
        m,
        n,
        k,
        working_k,
        num_pid_m,
        num_pid_n,
        c_stride,
        a_stride,
        b_stride,
        param,
    ).launch(
        grid=(num_pid_m * num_pid_n, param.SPLIT_K, 1),
        block=(param.BLOCK_THREADS, 1, 1),
        stream=stream,
    )


def get_default_b16_kwargs(m, n, k):
    kwargs = {
        "TILE_M": 256,
        "TILE_N": 256,
        "TILE_K": 64,
        "STAGES": 2,
        "SPLIT_K": 1,
        "BLOCK_M_WARPS": 2,
        "BLOCK_N_WARPS": 4,
        "BLOCK_K_WARPS": 1,
        "USE_HALF_TILE_INTERLEAVED": True,
        "GROUP_M": 0,
    }
    if m == 2048 and n == 2048 and k == 2048:
        kwargs["TILE_M"] = 128
        kwargs["TILE_N"] = 128
        kwargs["TILE_K"] = 64
        kwargs["STAGES"] = 4
        kwargs["SPLIT_K"] = 1
        kwargs["BLOCK_M_WARPS"] = 4
        kwargs["BLOCK_N_WARPS"] = 4
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 32 and n == 384 and k == 7168:
        kwargs["TILE_M"] = 32
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 64
        kwargs["STAGES"] = 5
        kwargs["SPLIT_K"] = 16
        kwargs["BLOCK_M_WARPS"] = 2
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 32 and n == 7168 and k == 2048:
        kwargs["TILE_M"] = 16
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 128
        kwargs["STAGES"] = 4
        kwargs["SPLIT_K"] = 1
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 1
        kwargs["BLOCK_K_WARPS"] = 2
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 32 and n == 384 and k == 16384:
        kwargs["TILE_M"] = 32
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 256
        kwargs["STAGES"] = 3
        kwargs["SPLIT_K"] = 16
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 4
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 16 and n == 5120 and k == 2880:
        kwargs["TILE_M"] = 16
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 64
        kwargs["STAGES"] = 5
        kwargs["SPLIT_K"] = 3
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 32 and n == 2880 and k == 2048:
        kwargs["TILE_M"] = 16
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 128
        kwargs["STAGES"] = 5
        kwargs["SPLIT_K"] = 2
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 800 and n == 384 and k == 7168:
        kwargs["TILE_M"] = 32
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 128
        kwargs["STAGES"] = 6
        kwargs["SPLIT_K"] = 1
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 2
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    return kwargs


def get_default_b8_kwargs(m, n, k):
    kwargs = {
        "TILE_M": 256,
        "TILE_N": 256,
        "TILE_K": 128,
        "STAGES": 2,
        "SPLIT_K": 1,
        "BLOCK_M_WARPS": 2,
        "BLOCK_N_WARPS": 4,
        "BLOCK_K_WARPS": 1,
        "USE_HALF_TILE_INTERLEAVED": True,
        "GROUP_M": 0,
    }
    if m == 2048 and n == 2048 and k == 2048:
        kwargs["TILE_M"] = 128
        kwargs["TILE_N"] = 128
        kwargs["TILE_K"] = 128
        kwargs["STAGES"] = 4
        kwargs["SPLIT_K"] = 1
        kwargs["BLOCK_M_WARPS"] = 4
        kwargs["BLOCK_N_WARPS"] = 4
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 32 and n == 384 and k == 7168:
        kwargs["TILE_M"] = 32
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 128
        kwargs["STAGES"] = 5
        kwargs["SPLIT_K"] = 8
        kwargs["BLOCK_M_WARPS"] = 2
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    return kwargs


def hgemm_validate(dtype_id, m, n, k, kwargs, tune_mode=False):
    TILE_M = kwargs["TILE_M"]
    TILE_N = kwargs["TILE_N"]
    TILE_K = kwargs["TILE_K"]
    STAGES = kwargs["STAGES"]
    SPLIT_K = kwargs["SPLIT_K"]
    BLOCK_M_WARPS = kwargs["BLOCK_M_WARPS"]
    BLOCK_N_WARPS = kwargs["BLOCK_N_WARPS"]
    BLOCK_K_WARPS = kwargs["BLOCK_K_WARPS"]
    USE_HALF_TILE_INTERLEAVED = kwargs["USE_HALF_TILE_INTERLEAVED"]
    GPU_ARCH = get_rocm_arch()
    IS_FP8 = "fp8" in HGEMM_DTYPE_STR_MAP[dtype_id]
    DTYPE_BYTES = 1 if IS_FP8 else 2

    try:
        assert_hgemm_wmma_kernel(
            DTYPE_ID=dtype_id,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            STAGES=STAGES,
            SPLIT_K=SPLIT_K,
            BLOCK_M_WARPS=BLOCK_M_WARPS,
            BLOCK_N_WARPS=BLOCK_N_WARPS,
            BLOCK_K_WARPS=BLOCK_K_WARPS,
            HAS_BIAS=kwargs["HAS_BIAS"],
            HAS_K_TAIL=kwargs["HAS_K_TAIL"],
            GROUP_M=kwargs["GROUP_M"],
            USE_HALF_TILE_INTERLEAVED=USE_HALF_TILE_INTERLEAVED,
            WMMA_STEP_CONSTRAIN=tune_mode,
        )
    except Exception as e:
        return False

    if USE_HALF_TILE_INTERLEAVED:
        if not (
            STAGES == 2
            and BLOCK_K_WARPS == 1
            and BLOCK_M_WARPS == 2
            and BLOCK_N_WARPS >= 2
        ):
            return False

    def get_stage_smem_use(stages_):
        SMEM_USE = stages_ * TILE_M * TILE_K * DTYPE_BYTES
        SMEM_USE += stages_ * TILE_N * TILE_K * DTYPE_BYTES
        SMEM_USE = max(SMEM_USE, BLOCK_K_WARPS * TILE_M * TILE_N * DTYPE_BYTES)
        return SMEM_USE

    smem_use_s0 = get_stage_smem_use(STAGES)
    # smem_use_s1 = get_stage_smem_use(STAGES + 3)
    smem_cap = SMEM_CAPACITY_MAP[GPU_ARCH]
    if not (smem_use_s0 <= smem_cap):
        return False
    if m >= 4096 and n >= 4096 and k >= 4096:
        if not (
            TILE_M == 256 and TILE_N == 256 and TILE_K == 128
            if IS_FP8
            else 64
            and STAGES == 2
            and SPLIT_K == 1
            and BLOCK_M_WARPS == 2
            and BLOCK_N_WARPS == 4
            and BLOCK_K_WARPS == 1
        ):
            return False
    if kwargs["SPLIT_K"] > 1:
        global SPLIT_K_SEMAPHORE_MAX_LEN
        bm = (m + TILE_M - 1) // TILE_M
        bn = (n + TILE_N - 1) // TILE_N
        if not (bm * bn <= SPLIT_K_SEMAPHORE_MAX_LEN):
            return False
    if not ((n % 16 == 0) and (k % 16 == 0)):
        return False
    return True


def _ceil_div(a, b):
    return (a + b - 1) // b


def _hgemm_split_k_padded(k, tile_k, split_k):
    working_k = _ceil_div(k, split_k)
    padded_k = 0
    for split_idx in range(split_k):
        remaining_k = max(k - split_idx * working_k, 0)
        part_k = min(working_k, remaining_k)
        if part_k > 0:
            padded_k += _ceil_div(part_k, tile_k) * tile_k
    return padded_k


def _hgemm_tile_iou(m, n, k, tile_m, tile_n, tile_k, split_k):
    padded_m = _ceil_div(m, tile_m) * tile_m
    padded_n = _ceil_div(n, tile_n) * tile_n
    padded_k = _hgemm_split_k_padded(k, tile_k, split_k)
    return (m * n * k) / (padded_m * padded_n * padded_k)


def _hgemm_tile_iou_threshold(selections, m, n, k, keep_ratio):
    best_iou = 0.0
    for tile_m, tile_n, tile_k, split_k in itertools.product(
        selections["TILE_M"],
        selections["TILE_N"],
        selections["TILE_K"],
        selections["SPLIT_K"],
    ):
        best_iou = max(
            best_iou, _hgemm_tile_iou(m, n, k, tile_m, tile_n, tile_k, split_k)
        )
    return best_iou * keep_ratio


def _hgemm_config_tile_iou(m, n, k, config):
    return _hgemm_tile_iou(
        m,
        n,
        k,
        config["TILE_M"],
        config["TILE_N"],
        config["TILE_K"],
        config["SPLIT_K"],
    )


def _hgemm_has_smaller_supported_split_k(config, supported_split_k, m, n):
    bm = _ceil_div(m, config["TILE_M"])
    bn = _ceil_div(n, config["TILE_N"])
    block_count = bm * bn * config["SPLIT_K"]
    if block_count <= 1024:
        return False
    return any(
        smaller_split_k < config["SPLIT_K"] and bm * bn * smaller_split_k > 1024
        for smaller_split_k in supported_split_k
    )


def hgemm_get_configs(dtype_id, m, n, k):
    selections = {
        "TILE_M": [16, 32, 48, 64, 80, 96, 128, 256],
        "TILE_N": [64, 80, 96, 128, 256],
        "TILE_K": [64, 128, 256],
        "STAGES": [i for i in range(2, 10)],
        "SPLIT_K": [i for i in range(1, 10)],
        "BLOCK_M_WARPS": [1, 2, 4],
        "BLOCK_N_WARPS": [1, 2, 4],
        "BLOCK_K_WARPS": [1, 2],
        "GROUP_M": [0, 4],
        "USE_HALF_TILE_INTERLEAVED": [False, True],
    }
    keep_ratio = 0.95
    if m is not None and n is not None and k is not None:
        if m <= 256:
            selections["GROUP_M"] = [
                0,
            ]
            selections["USE_HALF_TILE_INTERLEAVED"] = [
                False,
            ]
            if m <= 32:
                selections["TILE_M"] = [16, 32]
                keep_ratio = 0.75
            elif m <= 128:
                selections["TILE_M"] = [16, 32, 48, 64, 80, 96, 128]
                keep_ratio = 0.85
        selections["SPLIT_K"] = [ks for ks in selections["SPLIT_K"] if k % ks == 0]
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    if m is None or n is None or k is None:
        pass
    else:
        tile_iou_threshold = _hgemm_tile_iou_threshold(selections, m, n, k, keep_ratio)
        configs = [
            config
            for config in configs
            if not _hgemm_has_smaller_supported_split_k(
                config, selections["SPLIT_K"], m, n
            )
            and _hgemm_config_tile_iou(m, n, k, config) >= tile_iou_threshold
            and hgemm_validate(dtype_id, m, n, k, config, True)
        ]
    return configs


@functools.lru_cache(maxsize=128)
def get_semaphore(stream, device):
    semaphore = torch.zeros(
        (SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device
    )
    signal = torch.zeros((SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device)
    return semaphore, signal


def infer_has_k_tail(k: int, split_k: int, tile_k: int, stages: int, is_ht: bool):
    working_k = (k + split_k - 1) // split_k
    last_working_k = k - (split_k - 1) * working_k
    working_k_tiles = (working_k + tile_k - 1) // tile_k
    last_working_k_tiles = (last_working_k + tile_k - 1) // tile_k
    has_k_tail = (working_k % tile_k != 0) or (last_working_k % tile_k != 0)
    has_k_tail = (
        has_k_tail
        or (working_k_tiles < stages - 1)
        or (last_working_k_tiles < stages - 1)
    )
    if is_ht:
        has_k_tail = (
            has_k_tail or (working_k_tiles % 2 != 0) or (last_working_k_tiles % 2 != 0)
        )
    return has_k_tail


def hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    user_kwargs: dict = {},
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    if stream is None:
        stream = torch.cuda.current_stream()
    global SPLIT_K_SEMAPHORE_MAX_LEN
    device = a.device
    semaphore, signal = get_semaphore(stream, device)
    k = a.shape[-1]
    a = a.view(-1, k)
    m = a.shape[0]
    n = b.shape[0]
    assert a.device == b.device
    assert a.dtype == b.dtype
    assert b.shape[1] == k
    assert b.numel() == n * k

    is_fp8_ptpc = scale_a is not None or scale_b is not None

    if c is None:
        if is_fp8_ptpc:
            c = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        else:
            c = torch.empty((m, n), dtype=a.dtype, device=a.device)
    c = c.view(-1, n)
    assert c.shape[0] == m
    assert a.device == c.device

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    kwargs = (
        get_default_b8_kwargs(m, n, k)
        if is_fp8_ptpc
        else get_default_b16_kwargs(m, n, k)
    )
    kwargs.update(user_kwargs)

    kwargs["HAS_K_TAIL"] = infer_has_k_tail(
        k=k,
        split_k=kwargs["SPLIT_K"],
        tile_k=kwargs["TILE_K"],
        stages=kwargs["STAGES"],
        is_ht=kwargs["USE_HALF_TILE_INTERLEAVED"],
    )

    kwargs["HAS_BIAS"] = False if bias is None else True

    bias_tensor = a if bias is None else bias
    if bias is not None:
        assert bias.shape[0] == n

    if is_fp8_ptpc:
        assert scale_a is not None and scale_b is not None
        assert scale_a.shape[0] == m
        assert scale_b.shape[0] == n
        assert c.dtype == torch.bfloat16
        dtype_id = HGEMM_DTYPE_FP8_PTPC
        hgemm_validate(dtype_id, m, n, k, kwargs)
        constexpr_param = init_hgemm_wmma_constexpr_param(
            dtype_id,
            TILE_M=kwargs["TILE_M"],
            TILE_N=kwargs["TILE_N"],
            TILE_K=kwargs["TILE_K"],
            STAGES=kwargs["STAGES"],
            SPLIT_K=kwargs["SPLIT_K"],
            BLOCK_M_WARPS=kwargs["BLOCK_M_WARPS"],
            BLOCK_N_WARPS=kwargs["BLOCK_N_WARPS"],
            BLOCK_K_WARPS=kwargs["BLOCK_K_WARPS"],
            HAS_BIAS=kwargs["HAS_BIAS"],
            HAS_K_TAIL=kwargs["HAS_K_TAIL"],
            GROUP_M=kwargs["GROUP_M"],
            USE_HALF_TILE_INTERLEAVED=kwargs["USE_HALF_TILE_INTERLEAVED"],
        )
        _run_compiled(
            hgemm_wmma,
            c,
            a,
            b,
            scale_a,
            scale_b,
            bias_tensor,
            semaphore,
            signal,
            m,
            n,
            k,
            stream,
            constexpr_param=constexpr_param,
        )
    else:
        dtype_id = HGEMM_DTYPE_F16 if a.dtype == torch.half else HGEMM_DTYPE_BF16
        hgemm_validate(dtype_id, m, n, k, kwargs)
        constexpr_param = init_hgemm_wmma_constexpr_param(
            dtype_id,
            TILE_M=kwargs["TILE_M"],
            TILE_N=kwargs["TILE_N"],
            TILE_K=kwargs["TILE_K"],
            STAGES=kwargs["STAGES"],
            SPLIT_K=kwargs["SPLIT_K"],
            BLOCK_M_WARPS=kwargs["BLOCK_M_WARPS"],
            BLOCK_N_WARPS=kwargs["BLOCK_N_WARPS"],
            BLOCK_K_WARPS=kwargs["BLOCK_K_WARPS"],
            HAS_BIAS=kwargs["HAS_BIAS"],
            HAS_K_TAIL=kwargs["HAS_K_TAIL"],
            GROUP_M=kwargs["GROUP_M"],
            USE_HALF_TILE_INTERLEAVED=kwargs["USE_HALF_TILE_INTERLEAVED"],
        )
        _run_compiled(
            hgemm_wmma,
            c,
            a,
            b,
            a,
            b,
            bias_tensor,
            semaphore,
            signal,
            m,
            n,
            k,
            stream,
            constexpr_param=constexpr_param,
        )
    return c
