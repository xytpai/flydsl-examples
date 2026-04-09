"""Hybrid tiled-MMA GEMM example.

This version keeps the high-level `tiled_mma` style, but follows the structure
of `04-preshuffle_gemm.py` instead of launching one output tile at a time from
Python:

- one kernel grid covers all M/N output tiles
- the K loop lives inside the kernel
- B can be provided as preshuffled or regular weight
- A uses a two-stage global -> LDS -> register pipeline

What it supports:
- `f16` / `bf16` input and output
- fp32 accumulation inside the GEMM K loop
- kernel-native `preshuffle` and `non-shuffle` B paths
- manual block-tile tuning via `--block-m/n/k`
- split-K via `--split-k`

What it does not support:
- fp32 input
- ragged edge tiles
- autotuning search
"""

import argparse
import functools
from dataclasses import dataclass

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, scf
from flydsl.expr import arith, gpu, rocdl, vector
from flydsl.compiler.protocol import fly_values
from flydsl.runtime.device import get_rocm_arch

from aiter.test_common import run_perftest
from utils.tensor_shim import GTensor

DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 128
DEFAULT_BLOCK_K = 64
DEFAULT_BLOCK_THREADS = 256
DEFAULT_STAGES_A = 2
DEFAULT_SPLIT_K = 1

MMA_TILE_M = 16
MMA_TILE_N = 64
MMA_TILE_K = 32
MFMA_INSTR_M = 16
MFMA_INSTR_N = 16
MFMA_INSTR_K = 16
AMD_WAVE_SIZE = 64
ELEMENT_BYTES = 2
G2S_COPY_BYTES = 16
VMEM_COPY_ELEMS = G2S_COPY_BYTES // ELEMENT_BYTES


class OnlineScheduler:
    def __init__(self, total_signals: int, init_count: int = 0):
        self.total_signals = total_signals
        self.current_signal_id = init_count
        self.remaining = init_count

    def release(self, count: int):
        count = min(count, self.total_signals - self.current_signal_id)
        self.current_signal_id += count
        self.remaining += count

    def consume(self, count: int):
        count = min(count, self.remaining)
        self.remaining -= count
        return count


@dataclass(frozen=True)
class KernelConfig:
    block_m: int = DEFAULT_BLOCK_M
    block_n: int = DEFAULT_BLOCK_N
    block_k: int = DEFAULT_BLOCK_K
    block_threads: int = DEFAULT_BLOCK_THREADS
    stages_a: int = DEFAULT_STAGES_A


DEFAULT_KERNEL_CONFIG = KernelConfig()
_SPLIT_K_COUNTERS = {}


def _get_dtype_spec(dtype_name: str):
    if dtype_name == "f16":
        return {
            "torch_dtype": torch.float16,
            "fx_dtype": fx.Float16,
        }
    if dtype_name == "bf16":
        return {
            "torch_dtype": torch.bfloat16,
            "fx_dtype": fx.BFloat16,
        }
    raise ValueError(f"Unsupported dtype_name={dtype_name!r}")


def _dtype_name_from_torch(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"Unsupported torch dtype {dtype!r}")


def _a_g2s_tile_rows(config: KernelConfig) -> int:
    val_per_thr = G2S_COPY_BYTES // ELEMENT_BYTES
    if config.block_k % val_per_thr != 0:
        raise ValueError(f"block_k={config.block_k} must be divisible by {val_per_thr}")
    thrs_col = config.block_k // val_per_thr
    if config.block_threads % thrs_col != 0:
        raise ValueError(
            f"block_threads={config.block_threads} must be divisible by A-copy columns={thrs_col}"
        )
    return config.block_threads // thrs_col


def _c_frag_values(config: KernelConfig) -> int:
    if (config.block_m * config.block_n) % config.block_threads != 0:
        raise ValueError(
            "Per-thread C fragment size must be integral: "
            f"{config.block_m} * {config.block_n} % {config.block_threads} != 0"
        )
    return (config.block_m * config.block_n) // config.block_threads


def _resolve_c_epilogue_vec_size(config: KernelConfig) -> int:
    for vec in (8, 4, 2):
        if config.block_n % vec != 0:
            continue
        if (config.block_m * config.block_n) % (config.block_threads * vec) != 0:
            continue
        return vec
    raise ValueError(
        "Unable to find split-K epilogue vector width for "
        f"(block_m, block_n, block_threads)=({config.block_m}, {config.block_n}, {config.block_threads})"
    )


def _div_exact(numerator: int, denominator: int, what: str) -> int:
    if numerator % denominator != 0:
        raise ValueError(f"{what} must divide exactly: {numerator} % {denominator} != 0")
    return numerator // denominator


def _nonshuffle_scheduler_counts(config: KernelConfig):
    a_vmem = _div_exact(
        config.block_m * config.block_k,
        config.block_threads * VMEM_COPY_ELEMS,
        "non-shuffle A vmem count",
    )
    b_vmem = _div_exact(
        config.block_n * config.block_k,
        config.block_threads * VMEM_COPY_ELEMS,
        "non-shuffle B vmem count",
    )
    wave_count = _div_exact(config.block_threads, AMD_WAVE_SIZE, "wave count")
    mfma_total = _div_exact(
        config.block_m * config.block_n * config.block_k,
        wave_count * MFMA_INSTR_M * MFMA_INSTR_N * MFMA_INSTR_K,
        "non-shuffle MFMA count",
    )
    return a_vmem, b_vmem, mfma_total


def _counter_span(m: int, n: int, config: KernelConfig) -> int:
    return (m // config.block_m) * (n // config.block_n)


def _stream_cache_key(stream: torch.cuda.Stream, counter_span: int):
    return (stream.device.index, int(stream.cuda_stream), counter_span)


def _get_split_k_counter(counter_span: int, stream: torch.cuda.Stream) -> torch.Tensor:
    key = _stream_cache_key(stream, counter_span)
    counter = _SPLIT_K_COUNTERS.get(key)
    if counter is None:
        counter = torch.zeros((counter_span,), dtype=torch.int32, device=stream.device)
        _SPLIT_K_COUNTERS[key] = counter
    return counter


def _validate_kernel_config(config: KernelConfig):
    a_g2s_tile_rows = _a_g2s_tile_rows(config)
    if config.block_threads != DEFAULT_BLOCK_THREADS:
        raise ValueError(
            f"Only block_threads={DEFAULT_BLOCK_THREADS} is supported for now, got {config.block_threads}"
        )
    if config.stages_a != DEFAULT_STAGES_A:
        raise ValueError(
            f"Only stages_a={DEFAULT_STAGES_A} is supported for now, got {config.stages_a}"
        )
    if config.block_m <= 0 or config.block_n <= 0 or config.block_k <= 0:
        raise ValueError(f"Tile sizes must be positive, got {config}")
    if config.block_m % MMA_TILE_M != 0:
        raise ValueError(f"block_m={config.block_m} must be divisible by {MMA_TILE_M}")
    if config.block_n % MMA_TILE_N != 0:
        raise ValueError(f"block_n={config.block_n} must be divisible by {MMA_TILE_N}")
    if config.block_k % MMA_TILE_K != 0:
        raise ValueError(f"block_k={config.block_k} must be divisible by {MMA_TILE_K}")
    if config.block_m % a_g2s_tile_rows != 0:
        raise ValueError(
            f"block_m={config.block_m} must be divisible by A g2s tile rows={a_g2s_tile_rows}"
        )
    _c_frag_values(config)
    _resolve_c_epilogue_vec_size(config)


def _validate_shape(m: int, n: int, k: int, config: KernelConfig, split_k: int = DEFAULT_SPLIT_K):
    _validate_kernel_config(config)
    if split_k <= 0:
        raise ValueError(f"split_k must be positive, got {split_k}")
    if m % config.block_m != 0 or n % config.block_n != 0 or k % config.block_k != 0:
        raise ValueError(
            f"Shape {(m, n, k)} must be divisible by tile "
            f"{(config.block_m, config.block_n, config.block_k)}"
        )
    if k % split_k != 0:
        raise ValueError(f"k={k} must be divisible by split_k={split_k}")
    if (k // split_k) % config.block_k != 0:
        raise ValueError(
            f"Per-split K={k // split_k} must be divisible by block_k={config.block_k}"
        )


def _shuffle_b(b: torch.Tensor) -> torch.Tensor:
    bn = 16
    bk = 32
    vec = 16 // b.element_size()
    if b.shape[-2] % bn != 0 or b.shape[-1] % bk != 0:
        raise ValueError(
            f"Shape {tuple(b.shape)} is incompatible with preshuffle {(bn, bk)}"
        )
    shuffled = b.view(-1, b.shape[-2] // bn, bn, b.shape[-1] // bk, bk // vec, vec)
    shuffled = shuffled.permute(0, 1, 3, 4, 2, 5).contiguous()
    shuffled = shuffled.view(*b.shape)
    shuffled.is_shuffled = True
    return shuffled


def _tflops(us: float, m: int, n: int, k: int) -> float:
    return (2 * m * n * k) / (us / 1e6) / 1e12


def create_inputs(args):
    spec = _get_dtype_spec(args.dtype)
    a = torch.randn((args.m, args.k), dtype=spec["torch_dtype"], device="cuda")
    b = torch.randn((args.n, args.k), dtype=spec["torch_dtype"], device="cuda")
    preshuffle_b = _shuffle_b(b)
    return a, b, preshuffle_b


def create_outputs(args):
    spec = _get_dtype_spec(args.dtype)
    c = torch.zeros((args.m, args.n), dtype=spec["torch_dtype"], device="cuda")
    return (c,)


def ref_func(a, b, out):
    return torch.matmul(a, b.t(), out=out)


def _make_tiled_copy_g2s_A(config: KernelConfig, fx_dtype):
    val_per_thr = G2S_COPY_BYTES // ELEMENT_BYTES
    thrs_col = config.block_k // val_per_thr
    thrs_row = config.block_threads // thrs_col
    return fx.make_tiled_copy(
        fx.make_copy_atom(fx.UniversalCopy128b(), fx_dtype),
        fx.make_layout(
            ((thrs_col, thrs_row), (1, val_per_thr)),
            ((thrs_row * val_per_thr, 1), (1, thrs_row)),
        ),
        fx.make_tile(thrs_row, config.block_k),
    )


def _make_tiled_mma(fx_dtype):
    return fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx_dtype, fx.Float32)),
        fx.make_layout((1, 4, 1), (0, 1, 0)),
        fx.make_tile(16, 64, fx.make_layout((4, 4, 2), (1, 8, 4))),
    )


@functools.lru_cache(maxsize=128)
def compile_hybrid_kernel(
    dtype_name: str,
    m: int,
    n: int,
    k: int,
    config: KernelConfig,
    split_k: int,
    b_preshuffle: bool,
):
    _validate_shape(m, n, k, config, split_k=split_k)
    spec = _get_dtype_spec(dtype_name)
    fx_dtype = spec["fx_dtype"]
    block_m = config.block_m
    block_n = config.block_n
    block_k = config.block_k
    block_threads = config.block_threads
    stages_a = config.stages_a
    c_frag_values = _c_frag_values(config)
    c_epilogue_vec_size = _resolve_c_epilogue_vec_size(config)
    c_epilogue_x_threads = block_n // c_epilogue_vec_size
    c_epilogue_reg_count = (block_m * block_n) // (block_threads * c_epilogue_vec_size)
    is_split_k = split_k > 1
    k_tiles_per_split = (k // split_k) // block_k
    tiles_n = n // block_n
    gpu_arch = get_rocm_arch()
    needs_global_atomic_bf16 = dtype_name == "bf16" and gpu_arch == "gfx942"
    nonshuffle_a_vmem = nonshuffle_b_vmem = nonshuffle_mfma_total = 0
    nonshuffle_block_n_waves = 0
    nonshuffle_warp_n_steps = nonshuffle_warp_k_steps = 0
    if not b_preshuffle:
        nonshuffle_a_vmem, nonshuffle_b_vmem, nonshuffle_mfma_total = _nonshuffle_scheduler_counts(
            config
        )
        nonshuffle_block_n_waves = _div_exact(
            block_threads,
            AMD_WAVE_SIZE,
            "non-shuffle block N wave count",
        )
        nonshuffle_warp_n_steps = _div_exact(
            block_n,
            MMA_TILE_N,
            "non-shuffle warp N steps",
        )
        nonshuffle_warp_k_steps = _div_exact(
            block_k,
            MMA_TILE_K,
            "non-shuffle warp K steps",
        )

    @flyc.kernel
    def gemm_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        COUNTER: fx.Tensor,
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s_A: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        tid_i32 = fx.Int32(tid)
        bid_x, bid_y, bid_z = fx.block_idx

        A_raw = A
        B_raw = B
        C_raw = C
        COUNTER_raw = COUNTER

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        bid_x_idx = fx.Index(bid_x)
        bid_y_idx = fx.Index(bid_y)
        bid_z_idx = fx.Index(bid_z)
        m_offset = bid_x_idx * block_m
        n_offset = bid_y_idx * block_n
        output_zero_vec = fx.arith.constant_vector(
            0.0,
            fx.T.VectorType.get([c_frag_values], fx_dtype.ir_type),
        )
        output_pair_type = fx.T.VectorType.get([2], fx_dtype.ir_type)
        zero_i32 = fx.Int32(0)
        one_i32 = fx.Int32(1)
        elem_bytes_i32 = fx.Int32(ELEMENT_BYTES)
        k_tile_base_idx = fx.Index(0)
        k_tile_base_i32 = fx.Int32(0)
        global_ptr_type = ir.Type.parse("!llvm.ptr<1>")
        i64_type = fx.T.i64()

        if is_split_k:
            out_rsrc = fx.buffer_ops.create_buffer_resource(C_raw, max_size=True)
            out_base_idx = None
            if needs_global_atomic_bf16:
                out_base_idx = fx.buffer_ops.extract_base_index(C_raw)
            counter_idx = fx.Int32(arith.index_cast(fx.T.i32(), bid_x_idx * tiles_n + bid_y_idx))
            k_tile_base_idx = bid_z_idx * k_tiles_per_split
            k_tile_base_i32 = arith.index_cast(fx.T.i32(), k_tile_base_idx)
            counter_base_ptr = fly.extract_aligned_pointer_as_index(global_ptr_type, fly_values(COUNTER_raw)[0])
            counter_base_i64 = llvm.PtrToIntOp(i64_type, counter_base_ptr).result

            def get_counter_ptr():
                counter_idx_index = arith.index_cast(fx.T.index(), counter_idx)
                counter_byte_offset = arith.index_cast(i64_type, counter_idx_index * fx.Index(4))
                counter_addr_i64 = llvm.AddOp(
                    counter_base_i64,
                    counter_byte_offset,
                    llvm.IntegerOverflowFlags(0),
                ).result
                return llvm.IntToPtrOp(global_ptr_type, counter_addr_i64).result

        gA_k = fx.flat_divide(A, fx.make_tile(block_m, block_k))[None, None, bid_x, None]
        gB_k = fx.flat_divide(B, fx.make_tile(block_n, block_k))[None, None, bid_y, None]
        gC = fx.flat_divide(C, fx.make_tile(block_m, block_n))[None, None, bid_x, bid_y]

        thr_mma = tiled_mma.thr_slice(tid)
        thr_copy_g2s_A = tiled_copy_g2s_A.get_slice(tid)

        uni_copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx_dtype)
        buffer_copy_128b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx_dtype)
        buffer_copy_16b = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx_dtype)
        buffer_copy_B = buffer_copy_128b

        thr_copy_s2r_A = fx.make_tiled_copy_A(buffer_copy_128b, tiled_mma).get_slice(tid)
        thr_copy_g2r_B = fx.make_tiled_copy_B(buffer_copy_B, tiled_mma).get_slice(tid)
        thr_copy_r2g_C = fx.make_tiled_copy_C(buffer_copy_16b, tiled_mma).get_slice(tid)
        thr_gB_k = None
        b_tensor = None
        if b_preshuffle:
            thr_gB_k = thr_copy_g2r_B.partition_S(gB_k)
        else:
            b_tensor = GTensor(B_raw, dtype=fx_dtype.ir_type, shape=(n, k))
            w_tid = tid_i32 % AMD_WAVE_SIZE
            wid = tid_i32 // AMD_WAVE_SIZE
            ldmatrix_b_n_idx = w_tid % MFMA_INSTR_N
            ldmatrix_b_k_vec_idx = (w_tid // MFMA_INSTR_N) * VMEM_COPY_ELEMS
            warp_n_idx = (wid % nonshuffle_block_n_waves) * MFMA_INSTR_N

        smem_ptr = fx.get_dyn_shared()
        smem_ptr_ab = fx.recast_iter(
            fx.PointerType.get(fx_dtype.ir_type, fx.AddressSpace.Shared, 512),
            smem_ptr,
        )

        composed_layout_A = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(3, 3, 3)),
            fx.make_ordered_layout((block_m, block_k, stages_a), (1, 0, 2)),
        )
        sA = fx.make_view(smem_ptr_ab, composed_layout_A)

        thr_gA_k = thr_copy_g2s_A.partition_S(gA_k)
        thr_sA = thr_copy_g2s_A.partition_D(sA)
        thr_sA_s2r = thr_copy_s2r_A.partition_S(sA)
        thr_gC = thr_copy_r2g_C.partition_S(gC)

        copy_frag_A = fx.make_fragment_like(thr_sA[None, None, None, 0])

        mma_frag_A = thr_mma.make_fragment_A(sA[None, None, 0])
        mma_frag_B = fx.make_fragment_like(
            fx.flat_product(
                thr_mma.partition_B(gB_k).layout(None, None, None, 0),
                fx.make_layout(2, 1),
            ),
            fx_dtype.ir_type,
        )
        mma_frag_C = thr_mma.make_fragment_C(gC)

        mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)
        mma_frag_B_retile = thr_copy_g2r_B.retile(mma_frag_B)
        mma_frag_C_out = fx.make_fragment_like(mma_frag_C, fx_dtype.ir_type)
        mma_frag_C_retile = thr_copy_r2g_C.retile(mma_frag_C_out)

        if is_split_k:
            uni_copy_16b = fx.make_copy_atom(fx.UniversalCopy16b(), fx_dtype)
            thr_copy_r2s_C = fx.make_tiled_copy_C(uni_copy_16b, tiled_mma).get_slice(tid)
            sC = fx.make_view(smem_ptr_ab, fx.make_ordered_layout((block_m, block_n), (1, 0)))
            thr_sC = thr_copy_r2s_C.partition_D(sC)
            mma_frag_C_retile_s = thr_copy_r2s_C.retile(mma_frag_C_out)

            def zero_output_tile():
                cond_z0 = arith.cmpi(arith.CmpIPredicate.eq, bid_z_idx, fx.Index(0))
                cond_z0_if = scf.IfOp(cond_z0, results_=[], has_else=False)
                with ir.InsertionPoint(cond_z0_if.then_block):
                    mma_frag_C_out.store(output_zero_vec)
                    fx.copy(buffer_copy_16b, mma_frag_C_retile, thr_gC)
                    rocdl.s_waitcnt(0)
                    scf.YieldOp([])
                rocdl.sched_barrier(0)
                gpu.barrier()

                cond_z0_if = scf.IfOp(cond_z0, results_=[], has_else=False)
                with ir.InsertionPoint(cond_z0_if.then_block):
                    tid0 = arith.cmpi(arith.CmpIPredicate.eq, tid_i32, zero_i32)
                    tid0_if = scf.IfOp(tid0, results_=[], has_else=False)
                    with ir.InsertionPoint(tid0_if.then_block):
                        counter_ptr = get_counter_ptr()
                        counter_ptr_v = counter_ptr._value if hasattr(counter_ptr, "_value") else counter_ptr
                        one_i32_v = one_i32.ir_value() if hasattr(one_i32, "ir_value") else one_i32
                        llvm.InlineAsmOp(None, [], "buffer_wbl2 sc0 sc1", "", has_side_effects=True)
                        llvm.InlineAsmOp(
                            None,
                            [counter_ptr_v, one_i32_v],
                            "global_store_dword $0, $1, off sc0 sc1",
                            "v,v",
                            has_side_effects=True,
                        )
                        rocdl.s_waitcnt(0)
                        scf.YieldOp([])
                    scf.YieldOp([])
                rocdl.sched_barrier(0)
                gpu.barrier()

            def split_k_barrier():
                wait_loop = scf.WhileOp([fx.T.i32()], [zero_i32.ir_value()])
                before = ir.Block.create_at_start(wait_loop.before, [fx.T.i32()])
                after = ir.Block.create_at_start(wait_loop.after, [fx.T.i32()])

                with ir.InsertionPoint(before):
                    cur = before.arguments[0]
                    need_wait = arith.cmpi(arith.CmpIPredicate.eq, cur, zero_i32)
                    scf.ConditionOp(need_wait, [cur])

                with ir.InsertionPoint(after):
                    counter_ptr = get_counter_ptr()
                    counter_ptr_v = counter_ptr._value if hasattr(counter_ptr, "_value") else counter_ptr
                    data = llvm.InlineAsmOp(
                        fx.T.i32(),
                        [counter_ptr_v],
                        "global_load_dword $0, $1, off sc1",
                        "=v,v",
                        has_side_effects=True,
                    ).result
                    rocdl.s_waitcnt(0)
                    scf.YieldOp([data])

                gpu.barrier()

            def atomic_add_output_pair(pair, byte_off_i32):
                if needs_global_atomic_bf16:
                    byte_off_idx = arith.index_cast(fx.T.index(), byte_off_i32)
                    ptr_addr_idx = out_base_idx + byte_off_idx
                    out_ptr = fx.buffer_ops.create_llvm_ptr(ptr_addr_idx, address_space=1)
                    out_ptr_v = out_ptr._value if hasattr(out_ptr, "_value") else out_ptr
                    pair_v = pair._value if hasattr(pair, "_value") else pair
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.fadd,
                        out_ptr_v,
                        pair_v,
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=4,
                    )
                else:
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        pair,
                        out_rsrc,
                        byte_off_i32,
                        zero_i32,
                        zero_i32,
                    )

            zero_output_tile()

        def load_b_fragment(k_tile_idx_i32, stage):
            if b_preshuffle:
                fx.copy(
                    buffer_copy_B,
                    thr_gB_k[None, None, None, k_tile_idx_i32],
                    mma_frag_B_retile[None, None, None, stage],
                )
                return

            # Match hgemm's non-shuffle per-lane 128b B loads.
            k_offset = k_tile_idx_i32 * block_k
            for kk in fx.range_constexpr(nonshuffle_warp_k_steps):
                warp_atom_k_idx = kk * MMA_TILE_K
                for ii in fx.range_constexpr(nonshuffle_warp_n_steps):
                    n_idx = n_offset + warp_n_idx + ii * MMA_TILE_N + ldmatrix_b_n_idx
                    k_idx = k_offset + warp_atom_k_idx + ldmatrix_b_k_vec_idx
                    mma_frag_B_retile[None, ii, kk, stage].store(
                        b_tensor.vec_load((n_idx, k_idx), VMEM_COPY_ELEMS)
                    )

        def run_pipeline_stage(read_stage, next_k, read_next=True):
            write_stage = read_stage ^ 1

            if read_next:
                next_k = arith.index_cast(fx.T.i32(), next_k)
                fx.copy(buffer_copy_128b, thr_gA_k[None, None, None, next_k], copy_frag_A)
                load_b_fragment(next_k, write_stage)

            for block_k_iter in fx.range_constexpr(block_k // MMA_TILE_K):
                fx.copy(
                    uni_copy_128b,
                    thr_sA_s2r[None, None, block_k_iter, read_stage],
                    mma_frag_A_retile[None, None, block_k_iter],
                )
                fx.gemm(
                    tiled_mma,
                    mma_frag_C,
                    mma_frag_A[None, None, (None, block_k_iter)],
                    mma_frag_B[None, None, (None, block_k_iter), read_stage],
                    mma_frag_C,
                )

            fx.copy(uni_copy_128b, copy_frag_A, thr_sA[None, None, None, write_stage])
            fx.gpu.barrier()

            def hot_loop_scheduler():
                if b_preshuffle:
                    fx.rocdl.sched_dsrd(2)
                    fx.rocdl.sched_mfma(2)
                    fx.rocdl.sched_dsrd(1)
                    fx.rocdl.sched_mfma(1)
                    fx.rocdl.sched_dsrd(1)
                    fx.rocdl.sched_mfma(2)

                    def sched_main_iter(with_vmem=False, with_dswr=False):
                        if with_vmem:
                            fx.rocdl.sched_vmem(1)
                        fx.rocdl.sched_mfma(2)
                        fx.rocdl.sched_dsrd(1)
                        fx.rocdl.sched_mfma(2)
                        if with_dswr:
                            fx.rocdl.sched_dswr(1)

                    for _ in fx.range_constexpr(8):
                        sched_main_iter(with_vmem=True)
                    sched_main_iter()
                    for _ in fx.range_constexpr(7):
                        sched_main_iter(with_dswr=True)
                    fx.rocdl.sched_barrier(0)
                    return

                ldg_total = nonshuffle_a_vmem + nonshuffle_b_vmem
                ldg_sts_total = ldg_total + nonshuffle_a_vmem
                avg_mfma_count = (nonshuffle_mfma_total + ldg_sts_total - 1) // ldg_sts_total
                mfma_sched = OnlineScheduler(nonshuffle_mfma_total, nonshuffle_mfma_total)
                ldg_sched = OnlineScheduler(ldg_total, ldg_total)

                for _ in fx.range_constexpr(ldg_total):
                    fx.rocdl.sched_vmem(ldg_sched.consume(1))
                    fx.rocdl.sched_mfma(mfma_sched.consume(avg_mfma_count))
                for _ in fx.range_constexpr(nonshuffle_a_vmem):
                    fx.rocdl.sched_dswr(1)
                    fx.rocdl.sched_mfma(mfma_sched.consume(avg_mfma_count))
                fx.rocdl.sched_barrier(0)

            hot_loop_scheduler()

        fx.copy(buffer_copy_128b, thr_gA_k[None, None, None, k_tile_base_i32], copy_frag_A)
        load_b_fragment(k_tile_base_i32, 0)

        mma_frag_C.store(
            fx.arith.constant_vector(0.0, fx.T.VectorType.get([c_frag_values], fx.T.f32()))
        )

        fx.copy(uni_copy_128b, copy_frag_A, thr_sA[None, None, None, 0])
        fx.gpu.barrier()

        if k_tiles_per_split == 1:
            run_pipeline_stage(read_stage=0, next_k=None, read_next=False)
        elif k_tiles_per_split % 2 == 0:
            for k_iter in range(0, k_tiles_per_split - 2, 2):
                run_pipeline_stage(read_stage=0, next_k=k_tile_base_idx + k_iter + 1)
                run_pipeline_stage(read_stage=1, next_k=k_tile_base_idx + k_iter + 2)
            run_pipeline_stage(read_stage=0, next_k=k_tile_base_idx + k_tiles_per_split - 1)
            run_pipeline_stage(read_stage=1, next_k=None, read_next=False)
        else:
            for k_iter in range(0, k_tiles_per_split - 3, 2):
                run_pipeline_stage(read_stage=0, next_k=k_tile_base_idx + k_iter + 1)
                run_pipeline_stage(read_stage=1, next_k=k_tile_base_idx + k_iter + 2)
            run_pipeline_stage(read_stage=0, next_k=k_tile_base_idx + k_tiles_per_split - 2)
            run_pipeline_stage(read_stage=1, next_k=k_tile_base_idx + k_tiles_per_split - 1)
            run_pipeline_stage(read_stage=0, next_k=None, read_next=False)

        mma_frag_C_out.store(
            fx.arith.trunc_f(
                fx.T.VectorType.get([c_frag_values], fx_dtype.ir_type),
                mma_frag_C.load(),
            )
        )

        if is_split_k:
            fx.copy(uni_copy_16b, mma_frag_C_retile_s, thr_sC)
            gpu.barrier()
            split_k_barrier()

            for i in fx.range_constexpr(c_epilogue_reg_count):
                global_tid = block_threads * i + tid_i32
                m_local_idx = arith.index_cast(fx.T.index(), global_tid // c_epilogue_x_threads)
                n_local_idx = arith.index_cast(
                    fx.T.index(),
                    (global_tid % c_epilogue_x_threads) * c_epilogue_vec_size,
                )
                m_global_i32 = arith.index_cast(fx.T.i32(), m_offset + m_local_idx)
                n_global_i32 = arith.index_cast(fx.T.i32(), n_offset + n_local_idx)
                pair_base_elem = m_global_i32 * fx.Int32(n) + n_global_i32
                m_local_i32 = arith.index_cast(fx.T.i32(), m_local_idx)

                for pair_idx in fx.range_constexpr(c_epilogue_vec_size // 2):
                    pair_col_idx = n_local_idx + fx.Index(pair_idx * 2)
                    pair_col_i32 = arith.index_cast(fx.T.i32(), pair_col_idx)
                    e0 = sC[m_local_i32, pair_col_i32]
                    e1 = sC[m_local_i32, pair_col_i32 + fx.Int32(1)]
                    pair = vector.from_elements(output_pair_type, [e0, e1])
                    pair_elem = pair_base_elem + fx.Int32(pair_idx * 2)
                    atomic_add_output_pair(pair, pair_elem * elem_bytes_i32)
        else:
            fx.copy(buffer_copy_16b, mma_frag_C_retile, thr_gC)

    @flyc.jit
    def launch_hybrid_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        COUNTER: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        if b_preshuffle:
            b_view_layout = fx.make_layout(
                ((16, n // 16), (8, 4, k // 32)),
                ((8, 16 * k), (1, 8 * 16, 8 * 16 * 4)),
            )
            kernel_B = fx.Tensor(fx.make_view(fx.get_iter(B), b_view_layout))
        else:
            kernel_B = B

        tiled_copy_g2s_A = _make_tiled_copy_g2s_A(config, fx_dtype)
        tiled_mma = _make_tiled_mma(fx_dtype)
        smem_elements = stages_a * block_m * block_k
        if is_split_k:
            smem_elements = max(smem_elements, block_m * block_n)

        gemm_kernel(A, kernel_B, C, COUNTER, tiled_mma, tiled_copy_g2s_A).launch(
            grid=(m // block_m, n // block_n, split_k),
            block=(block_threads, 1, 1),
            smem=smem_elements * ELEMENT_BYTES,
            stream=stream,
        )

    return launch_hybrid_kernel


def _launch_kernel(
    a,
    b,
    out,
    b_preshuffle: bool,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
    split_k: int = DEFAULT_SPLIT_K,
):
    dtype_name = _dtype_name_from_torch(a.dtype)
    m, k = a.shape
    n = b.shape[0]
    _validate_shape(m, n, k, config, split_k=split_k)
    if out.dtype != a.dtype:
        raise ValueError(f"Output dtype {out.dtype} must match input dtype {a.dtype}")
    exe = compile_hybrid_kernel(
        dtype_name,
        m,
        n,
        k,
        config,
        split_k,
        b_preshuffle,
    )
    stream = torch.cuda.current_stream()
    counter_span = _counter_span(m, n, config) if split_k > 1 else 1
    counter = _get_split_k_counter(counter_span, stream)
    if split_k > 1:
        counter.zero_()
    exe(a, b, out, counter, stream=stream)


def _launch_preshuffled(
    a,
    preshuffle_b,
    out,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
    split_k: int = DEFAULT_SPLIT_K,
):
    _launch_kernel(a, preshuffle_b, out, b_preshuffle=True, config=config, split_k=split_k)


def _launch_nonshuffled(
    a,
    b,
    out,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
    split_k: int = DEFAULT_SPLIT_K,
):
    _launch_kernel(
        a,
        b,
        out,
        b_preshuffle=False,
        config=config,
        split_k=split_k,
    )


def func(
    a,
    b,
    out,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
    split_k: int = DEFAULT_SPLIT_K,
):
    if getattr(b, "is_shuffled", False):
        _launch_preshuffled(a, b, out, config=config, split_k=split_k)
    else:
        _launch_nonshuffled(a, b, out, config=config, split_k=split_k)


def bench_func_preshuffled(
    a,
    preshuffle_b,
    out,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
    split_k: int = DEFAULT_SPLIT_K,
):
    _launch_preshuffled(a, preshuffle_b, out, config=config, split_k=split_k)


def benchmark(args):
    a, b, preshuffle_b = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    config = args.kernel_config
    if args.weight_layout == "preshuffle":
        kernel_func = functools.partial(
            bench_func_preshuffled,
            config=config,
            split_k=args.split_k,
        )
        kernel_weight = preshuffle_b
    else:
        kernel_func = functools.partial(
            _launch_nonshuffled,
            config=config,
            split_k=args.split_k,
        )
        kernel_weight = b
    func_inouts = (a, kernel_weight, *outputs)
    ref_inouts = (a, b, *ref_outputs)

    kernel_func(*func_inouts)
    ref_func(*ref_inouts)
    torch.cuda.synchronize()

    atol = 1e-2 if args.dtype == "f16" else 2e-2
    if args.split_k > 1:
        # Match hgemm.py behavior: split-K accumulates partial tiles through
        # dtype-sized atomics, so bf16/f16 outputs see larger rounding error
        # than the split_k=1 path.
        atol = max(atol, 0.125 if args.dtype == "f16" else 0.5)
    rtol = atol
    for output, ref_output in zip(outputs, ref_outputs):
        is_allclose = torch.allclose(output, ref_output, atol=atol, rtol=rtol)
        maxdiff = (output - ref_output).abs().max().item()
        print(f"validation passed: {is_allclose}, maxdiff={maxdiff}")
        assert is_allclose

    print("===================== [REF] =====================")
    _, ref_us = run_perftest(
        ref_func,
        *ref_inouts,
        num_warmup=args.warmup,
        num_iters=args.niters,
    )
    print(f"avg: {ref_us:.4f} us/iter, {_tflops(ref_us, args.m, args.n, args.k):.2f} TFLOPS")

    print("===================== [TILED MMA] =====================")
    _, mma_us = run_perftest(
        kernel_func,
        *func_inouts,
        num_warmup=args.warmup,
        num_iters=args.niters,
    )
    print(f"avg: {mma_us:.4f} us/iter, {_tflops(mma_us, args.m, args.n, args.k):.2f} TFLOPS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid tiled-MMA GEMM demo")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", choices=["f16", "bf16"], required=True)
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N)
    parser.add_argument("--block-k", type=int, default=DEFAULT_BLOCK_K)
    parser.add_argument("--split-k", type=int, default=DEFAULT_SPLIT_K)
    parser.add_argument(
        "--weight-layout",
        choices=["preshuffle", "non-shuffle"],
        default="preshuffle",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--niters", type=int, default=100)
    args = parser.parse_args()
    args.kernel_config = KernelConfig(
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
    )
    _validate_kernel_config(args.kernel_config)
    _validate_shape(args.m, args.n, args.k, args.kernel_config, split_k=args.split_k)

    print(
        "run tiled_mma_gemm.py with "
        f"M={args.m}, N={args.n}, K={args.k}, dtype={args.dtype}, "
        f"weight_layout={args.weight_layout}, "
        f"split_k={args.split_k}, "
        f"tile=({args.kernel_config.block_m}, {args.kernel_config.block_n}, {args.kernel_config.block_k}), "
        f"warmup={args.warmup}, niters={args.niters}"
    )
    benchmark(args)
