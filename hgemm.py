import time
import torch
import argparse
import numpy as np
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass

import flydsl
from flydsl.lang.ir.types import T as FT
from flydsl.dialects.ext import flir, gpu, arith, buffer_ops, rocdl, vector
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.pipeline import Pipeline, run_pipeline
from flydsl.dialects.ext.python_control_flow import range_constexpr, lower_range_for_loops
from flydsl.utils import SmemAllocator, SmemPtr
fm_fast = flir.arith.FastMathFlags.fast

from _mlir import ir
from _mlir.ir import F16Type, BF16Type, F32Type, IntegerType, VectorType
import _mlir.extras.types as T

from utils.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_load_pack_k32,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
)
from utils.mfma_epilogues import mfma_epilog


@dataclass
class Args:
    m: int
    n: int
    k: int
    dtype: torch.dtype


def create_inputs(args):
    a = torch.empty((args.m, args.k), dtype=args.dtype, device='cuda')
    a.uniform_(-1, 1)
    b = torch.empty((args.k, args.n), dtype=args.dtype, device='cuda')
    b.uniform_(-1, 1)
    return (a, b)


def create_outputs(args):
    c = torch.randn((args.m, args.n), dtype=args.dtype, device='cuda')
    return (c,)


def ref_func(a, b, c):
    torch.mm(a, b, out=c)


def create_hgemm_kernel(
    DTYPE,
    ELEMENT_BYTES: int,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    USE_ASYNC_COPY: bool,
    IS_FP16: bool,
):
    IS_BF16 = not IS_FP16
    DYN = ir.ShapedType.get_dynamic_size()
    ARCH = get_rocm_arch()
    VEC_SIZE = 16 // ELEMENT_BYTES
    TILE_K_BYTES = TILE_K * ELEMENT_BYTES
    BLOCK_THREADS = 256
    WAVE_SIZE = 64
    NUM_WAVES = BLOCK_THREADS // WAVE_SIZE
    BYTES_PER_A_TILE = TILE_M * TILE_K * ELEMENT_BYTES
    BYTES_PER_THREAD_A = BYTES_PER_A_TILE // BLOCK_THREADS
    A_ASYNC_LOAD_BYTES = 4 if ARCH == "gfx942" else 16
    A_ASYNC_LOAD_DWORD = A_ASYNC_LOAD_BYTES // 4
    NUM_A_LOADS = BYTES_PER_THREAD_A // 16
    NUM_A_ASYNC_LOADS = BYTES_PER_THREAD_A // A_ASYNC_LOAD_BYTES
    allocator_pong = SmemAllocator(None, arch=ARCH, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=ARCH, global_sym_name="smem1")
    
    class HGEMM(flir.MlirModule):
        GPU_MODULE_NAME = "gemm_kernels"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{ARCH}">']

        def init_gpu_module(self):
            self.dtype = DTYPE.get()
            self.i32 = T.i32()
            self.acc_type = T.f32()
            self.lds_a_pong = allocator_pong.allocate_array(self.dtype, TILE_M * TILE_K)
            self.lds_a_ping = allocator_ping.allocate_array(self.dtype, TILE_M * TILE_K)
            allocator_pong.finalize()
            allocator_ping.finalize()

        @flir.kernel
        def hgemm_kernel(
            self: flir.T.i64,
            a: lambda: T.memref(DYN, DTYPE.get()),
            b: lambda: T.memref(DYN, DTYPE.get()),
            c: lambda: T.memref(DYN, DTYPE.get()),
            m: lambda: T.index(),
            n: lambda: T.index(),
            k: lambda: T.index(),
        ):
            layout_c = flir.make_layout((m, n), stride=(n, 1))
            k_bytes = k * ELEMENT_BYTES
            k_div4bytes = (k * ELEMENT_BYTES) / 4
            layout_a = flir.make_layout((m, k_bytes), stride=(k_bytes, 1))
            layout_a_div4 = flir.make_layout((m, k_div4bytes), stride=(k_div4bytes, 1))
            layout_lds = flir.make_layout((TILE_M, TILE_K), stride=(TILE_K, 1))

            # CK-style XOR16 swizzle parameter (const).
            k_blocks16 = arith.index(TILE_K_BYTES // 16)

            tid_x = flir.thread_idx("x")
            bid_x = flir.block_idx("x")
            bid_y = flir.block_idx("y")

            base_ptr0, base_ptr1 = allocator_pong.get_base(), allocator_ping.get_base()
            lds_a_pong_ptr = self.lds_a_pong(base_ptr0)
            lds_a_ping_ptr = self.lds_a_ping(base_ptr1)

            lds_a_pong = SmemPtr(
                base_ptr0, lds_a_pong_ptr.byte_offset, self.dtype, shape=(TILE_M * TILE_K,)
            ).get()
            lds_a_ping = SmemPtr(
                base_ptr1, lds_a_ping_ptr.byte_offset, self.dtype, shape=(TILE_M * TILE_K,)
            ).get()

            m_i32 = arith.index_cast(self.i32, m)
            n_i32 = arith.index_cast(self.i32, n)
            k_i32 = arith.index_cast(self.i32, k)

            a_bytes = m_i32 * k_i32 * arith.i32(ELEMENT_BYTES)
            c_bytes = m_i32 * n_i32 * arith.i32(ELEMENT_BYTES)
            a_rsrc = buffer_ops.create_buffer_resource(a, num_records_bytes=a_bytes)
            c_rsrc = buffer_ops.create_buffer_resource(c, num_records_bytes=c_bytes)
            b_rsrc = buffer_ops.create_buffer_resource(b, max_size=True)

            bx_m = bid_x * TILE_M
            by_n = bid_y * TILE_N

            layout_wave_lane = flir.make_layout((4, WAVE_SIZE), stride=(WAVE_SIZE, 1)) # 256 Block size
            coord_wave_lane = flir.idx2crd(tid_x, layout_wave_lane)
            wave_id = flir.get(coord_wave_lane, 0)
            lane_id = flir.get(coord_wave_lane, 1)

            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))
            coord_lane16 = flir.idx2crd(lane_id, layout_lane16)
            lane_div_16 = flir.get(coord_lane16, 0)
            lane_mod_16 = flir.get(coord_lane16, 1)

            row_a_lds = lane_mod_16
            # Per-`k1` (KLane) base offset along K inside a 64B K0 block.
            col_offset_base = lane_div_16 * arith.constant(VEC_SIZE, index=True)
            col_offset_base_bytes = col_offset_base * arith.constant(ELEMENT_BYTES, index=True)

            # --- Dynamic tiling along N (4 waves) ---
            m_repeat = TILE_M // 16
            k_unroll = TILE_K_BYTES // 64
            n_per_wave = TILE_N // NUM_WAVES
            num_acc_n = n_per_wave // 16

            n_per_wave = arith.constant(n_per_wave, index=True)
            n_tile_base = wave_id * n_per_wave

            acc_init = arith.unwrap(arith.constant_vector(0.0, FT.f32x4))
            c4 = arith.constant(4, index=True)

            def a_tile_chunk_coord_i32(i: int, tx_i32_base: int, chunk_i32: int = 4):
                tile_k_dwords = (TILE_K * 2) // 4
                layout_a_tile_div4 = flir.make_layout((TILE_M, tile_k_dwords), stride=(tile_k_dwords, 1))
                return tile_chunk_coord_i32(
                    flir,
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=BLOCK_THREADS,
                    layout_tile_div4=layout_a_tile_div4,
                    chunk_i32=chunk_i32,
                )

            # DMA async version: direct global-to-LDS transfer
            def dma_a_tile_to_lds(base_k_div4, lds_buffer):
                from _mlir.dialects import llvm, memref as memref_dialect

                dma_bytes = A_ASYNC_LOAD_BYTES
                bytes_per_dword = 4
                chunk_i32 = dma_bytes // bytes_per_dword

                for i in range_constexpr(NUM_A_ASYNC_LOADS):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i, tid_x * A_ASYNC_LOAD_DWORD, chunk_i32=chunk_i32)
                    col_a_local_sw = flir.swizzle_xor16(row_a_local, col_a_local_i32 * c4, k_blocks16)
                    row_a_global = bx_m + row_a_local
                    coord_a_g = flir.make_coord(row_a_global, base_k_div4 * c4 + col_a_local_sw)
                    global_offset = arith.index_cast(FT.i32, flir.crd2idx(coord_a_g, layout_a))

                    if i == 0:
                        lds_addr = memref_dialect.extract_aligned_pointer_as_index(lds_buffer) + wave_id * WAVE_SIZE * dma_bytes
                        lds_ptr_i64_lane0 = rocdl.readfirstlane(FT.i64, arith.index_cast(FT.i64, lds_addr))
                    else:
                        lds_ptr_i64_lane0 += BLOCK_THREADS * dma_bytes
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_ptr_i64_lane0, address_space=3)

                    # DMA from global to LDS using buffer_load_lds
                    size_i32 = arith.constant(dma_bytes, type=FT.i32)
                    soffset = arith.constant(0, type=FT.i32)
                    offset_imm = arith.constant(0, type=FT.i32)
                    aux = arith.constant(1, type=FT.i32)

                    rocdl.raw_ptr_buffer_load_lds(
                        a_rsrc,
                        lds_ptr,
                        arith.unwrap(size_i32),
                        arith.unwrap(global_offset),
                        arith.unwrap(soffset),
                        arith.unwrap(offset_imm),
                        arith.unwrap(aux),
                    )

            def prefetch_a_to_lds(base_k, lds_buffer):
                base_k_div4 = base_k / 4
                dma_a_tile_to_lds(base_k_div4, lds_buffer)
            
            # Decompose global_n -> (n_blk, n_intra) once per ni.
            c_n0 = n / 16
            layout_n_blk_intra = flir.make_layout((c_n0, 16), stride=(16, 1))
            n_intra_list = []
            n_blk_list = []
            for i in range_constexpr(num_acc_n):
                offset = i * 16
                c_offset = arith.constant(offset, index=True)
                global_n = by_n + n_tile_base + c_offset + lane_mod_16
                coord_n = flir.idx2crd(global_n, layout_n_blk_intra)
                n_blk_list.append(flir.get(coord_n, 0))
                n_intra_list.append(flir.get(coord_n, 1))
            layout_b = make_preshuffle_b_layout(
                flir, arith, c_n=n, c_k=k, kpack_bytes=16, elem_bytes=ELEMENT_BYTES
            ).layout_b
            
            def load_b_packs_k64(base_k, ku: int, ni: int):
                # FP8/INT8/FP16/BF16: load 16 bytes (one full KPack).
                base_k_bytes = base_k * arith.constant(ELEMENT_BYTES, index=True)
                k0_base = base_k_bytes / 64
                k0 = k0_base + ku
                k1 = lane_div_16
                coord_pack = flir.make_coord(n_blk_list[ni], k0, k1, n_intra_list[ni], 0)
                idx_pack = flir.crd2idx(coord_pack, layout_b)
                b_view = flir.TensorView(
                    b,
                    (VEC_SIZE,),
                    strides=(1,),
                    base_indices=(idx_pack,),
                    element_type=self.dtype,
                )
                b16 = flir.copy(
                    flir.make_copy_atom(self.dtype, vector_size=VEC_SIZE),
                    b_view,
                    None,
                    alignment=8,
                    return_vector=True,
                    src_buffer_offset_in_bytes=False,
                )
                # Split 16B pack into two 8B halves.
                b_i64x2 = vector.bitcast(FT.i64x2, b16)
                b0_i64 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                b1_i64 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])

                # For fp16/bf16 MFMA 16x16x16, operands are 8B packs:
                # - fp16: v4f16
                # - bf16: v4i16 (bit pattern) for *_bf16_1k
                vec1_i64_ty = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
                b0_v1 = vector.from_elements(vec1_i64_ty, [b0_i64])
                b1_v1 = vector.from_elements(vec1_i64_ty, [b1_i64])
                if IS_FP16:
                    return vector.bitcast(FT.f16x4, b0_v1), vector.bitcast(FT.f16x4, b1_v1)
                return vector.bitcast(FT.i16x4, b0_v1), vector.bitcast(FT.i16x4, b1_v1)

            def load_b_tile(base_k):
                # b_tile[ku] = (packs_half0[ni], packs_half1[ni])
                b_tile = []
                for ku in range_constexpr(k_unroll):
                    packs0 = []
                    packs1 = []
                    for ni in range_constexpr(num_acc_n):
                        b0, b1 = load_b_packs_k64(base_k, ku, ni)
                        packs0.append(b0)
                        packs1.append(b1)
                    b_tile.append((packs0, packs1))
                return b_tile
            
            def _vec16_type():
                if IS_FP16:
                    return FT.f16x8  # 16B
                if IS_BF16:
                    return FT.bf16x8  # 16B
            
            def lds_load_16b(curr_row_a_lds, col_base, lds_buffer):
                # Swizzle in bytes, then convert to element offset for memref indexing.
                col_base_swz_bytes = flir.swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                col_base_swz = col_base_swz_bytes / 2
                coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swz)
                idx_a16 = flir.crd2idx(coord_a16, layout_lds)
                return vector.load_op(_vec16_type(), lds_buffer, [idx_a16])
            
            def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
                loaded_a16 = lds_load_16b(curr_row_a_lds, col_base, lds_buffer)
                a_i64x2 = vector.bitcast(FT.i64x2, loaded_a16)
                a0_i64 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1_i64 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                vec1_i64_ty = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
                a0_v1 = vector.from_elements(vec1_i64_ty, [a0_i64])
                a1_v1 = vector.from_elements(vec1_i64_ty, [a1_i64])
                if IS_FP16:
                    return vector.bitcast(FT.f16x4, a0_v1), vector.bitcast(FT.f16x4, a1_v1)
                return vector.bitcast(FT.i16x4, a0_v1), vector.bitcast(FT.i16x4, a1_v1)
            
            def compute_tile(accs_in, b_tile_in, lds_buffer, *, is_last_tile=False, a0_prefetch=None):
                scales_pf = {}

                current_accs_list = list(accs_in)

                mfma_res_ty = FT.f32x4

                if IS_FP16:
                    # gfx942 fp16 MFMA: 16x16x16 f16 (operands are v4f16, 8B packs)
                    mfma_fn = rocdl.mfma_f32_16x16x16f16
                elif IS_BF16:
                    # bf16 MFMA K16 variant uses i16 bit-pattern packs (v4i16)
                    mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k
                else:
                    mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

                def mfma_step(acc_in, a, b):
                    return mfma_fn(mfma_res_ty, [a, b, acc_in, 0, 0, 0])

                # "K64-byte wrapper": two back-to-back MFMA/WMMA ops using the two 8B halves.
                def mfma_k64_bytes(acc_in, a0, a1, b0, b1):
                    acc_mid = mfma_step(acc_in, a0, b0)
                    return mfma_step(acc_mid, a1, b1)

                for ku in range_constexpr(k_unroll):
                    b_packs0, b_packs1 = b_tile_in[ku]
                    # Byte-addressed K stepping (64B per ku).
                    ki64 = ku * 64
                    col_base = col_offset_base_bytes + ki64
                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row_a_lds = row_a_lds + mi_val
                        if (a0_prefetch is not None) and (ku == 0) and (mi == 0):
                            a0, a1 = a0_prefetch
                        else:
                            a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer)
                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            current_accs_list[acc_idx] = mfma_k64_bytes(
                                current_accs_list[acc_idx],
                                a0,
                                a1,
                                b_packs0[ni],
                                b_packs1[ni],
                            )
                return current_accs_list, scales_pf
            
            # ---------------- Scheduling hints (match CK-style) ----------------
            # These sched_group_barrier hints help the backend interleave VMEM/DS/MFMA
            # similarly to CK's tuned pipelines.
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                # - MFMA group size per "slot": num_acc_n
                # - Total MFMA per tile: (2*K32 per K64) * k_unroll * m_repeat * num_acc_n
                # - We emit (mfma_group + dsrd + mfma_group) per scheduler iteration.
                mfma_group = num_acc_n
                mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)

                # DS-read preload (CK default is 2).
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
                if TILE_M == 16:
                    rocdl.sched_vmem(1)
                rocdl.sched_mfma(1)
                if TILE_M == 16:
                    rocdl.sched_vmem(1)
                if num_acc_n < 4:
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if TILE_M == 16:
                        rocdl.sched_vmem(1)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if TILE_M == 16:
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)

                # DS-write hints near the end: match total A LDS-store micro-ops per thread.
                dswr_tail = NUM_A_LOADS
                if dswr_tail > sche_iters:
                    dswr_tail = sche_iters
                dswr_start = sche_iters - dswr_tail

                for sche_i in range_constexpr(sche_iters):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(mfma_group)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(mfma_group)
                    if (not USE_ASYNC_COPY) and (sche_i >= dswr_start - 1):
                        rocdl.sched_dswr(1)
                rocdl.sched_barrier(0)

            k0 = arith.constant(0, index=True)
            prefetch_a_to_lds(k0, lds_a_pong)  # Single buffer (reuses same)
            b_tile0 = load_b_tile(k0)
            gpu.barrier()
            accs = [acc_init] * (num_acc_n * m_repeat)
            b_tile_cur = b_tile0
            for k_base in range(0, k - TILE_K, TILE_K):
                next_k = k_base + TILE_K
                prefetch_a_to_lds(next_k, lds_a_pong)  # Reuse same buffer
                b_next = load_b_tile(next_k)
                accs, _ = compute_tile(accs, b_tile_cur, lds_a_pong)
                # Single LDS buffer: ensure *all* waves are done reading A from LDS
                # before any wave overwrites it with the next tile.
                gpu.barrier()
                hot_loop_scheduler()
                gpu.barrier()
                b_tile_cur = b_next

            final_accs, scales = compute_tile(
                accs, b_tile_cur, lds_a_pong, is_last_tile=True
            )

            def store_output(final_accs, scales):
                s_b_vals = None
                s_a_vecs = None
                
                def body_row(*, mi: int, ii: int, row_in_tile, row):
                    col_base = by_n + n_tile_base + lane_mod_16
                    idx_base = flir.crd2idx(flir.make_coord(row, col_base), layout_c)
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                        val_s = val
                        val_f16 = arith.trunc_f(FT.f16, val_s)
                        idx_out = idx_base + arith.constant(ni * 16, index=True)
                        buffer_ops.buffer_store(val_f16, c_rsrc, idx_out)

                mfma_epilog(
                    use_cshuffle=False,
                    arith=arith,
                    range_constexpr=range_constexpr,
                    m_repeat=m_repeat,
                    lane_div_16=lane_div_16,
                    bx_m=bx_m,
                    body_row=body_row,
                )
            
            store_output(final_accs, scales)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            a: lambda: T.memref(DYN, DTYPE.get()),
            b: lambda: T.memref(DYN, DTYPE.get()),
            c: lambda: T.memref(DYN, DTYPE.get()),
            m: lambda: T.index(),
            n: lambda: T.index(),
            k: lambda: T.index(),
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(BLOCK_THREADS, index=True)
            tm = arith.constant(TILE_M, index=True)
            tn = arith.constant(TILE_N, index=True)
            one = arith.constant(1, index=True)
            gx = (m + tm - one) / tm
            gy = n / tn
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "hgemm_kernel"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[a, b, c, m, n, k],
            )

    return HGEMM().module


EXE = None
def func(a, b, c):
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    ELEMENT_BYTES = 2
    TILE_M = 128
    TILE_N = 128
    TILE_K = 32
    ASYNC_COPY = True
    global EXE
    if not EXE:
        if a.dtype == torch.half:
            module = create_hgemm_kernel(F16Type, ELEMENT_BYTES, TILE_M, TILE_N, TILE_K, ASYNC_COPY, True)
        elif a.dtype == torch.bfloat16:
            module = create_hgemm_kernel(BF16Type, ELEMENT_BYTES, TILE_M, TILE_N, TILE_K, ASYNC_COPY, False)
        optimized = run_pipeline(module, Pipeline().canonicalize().cse())
        EXE = flydsl.compile(optimized)
    EXE(a, b, c, m, n, k)
    torch.cuda.synchronize()


def benchmark(args, func, ref_func, warmup=20, niters=100):
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = inputs + outputs
    ref_inouts = inputs + ref_outputs
    func(*inouts)
    ref_func(*ref_inouts)
    for output, ref_output in zip(outputs, ref_outputs):
        is_allclose = torch.allclose(output, ref_output, atol=1e-3, rtol=1e-3)
        print(output)
        print(ref_output)
        assert is_allclose == True
    print("validation passed!\n", flush=True)

    # get ref_func perf
    print("===================== [REF] =====================")
    for i in range(warmup):
        ref_func(*ref_inouts)
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        for i in range(niters):
            ref_func(*ref_inouts)
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    print(table)

    # get func perf
    print("===================== [FLYDSL] =====================")
    for i in range(warmup):
        func(*inouts)
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        for i in range(niters):
            func(*inouts)
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    print(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Examples")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    args = Args(**vars(args))
    benchmark(args, func, ref_func)
