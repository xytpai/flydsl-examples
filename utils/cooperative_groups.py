import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl.expr import range_constexpr, arith, vector, gpu, rocdl
from flydsl._mlir.dialects import llvm, fly, memref, scf
from flydsl.compiler.protocol import fly_values
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch

from .tensor_shim import GTensor


class CooperativeGroup:
    def __init__(self, counter, smem_flag, tidx, group_id, group_size):
        self.counter = counter
        self.counter_ = GTensor(counter, dtype=fx.Int32, shape=(-1,))
        self.smem_flag = smem_flag
        self.tidx = tidx
        self.group_id = group_id
        self.group_size = group_size
        _ptr_type = ir.Type.parse("!llvm.ptr<1>")
        counter_base_ptr = fly.extract_aligned_pointer_as_index(_ptr_type, fly_values(self.counter)[0])
        counter_base_ptr = llvm.PtrToIntOp(T.i64, counter_base_ptr).result
        counter_byte_offset = arith.index_cast(T.i64, fx.Index(self.group_id))
        counter_ptr = llvm.AddOp(counter_base_ptr, counter_byte_offset, llvm.IntegerOverflowFlags(0)).result
        counter_ptr = llvm.IntToPtrOp(_ptr_type, counter_ptr).result
        self.counter_ptr_v = counter_ptr._value if hasattr(counter_ptr, "_value") else counter_ptr
    
    def fetch_add_counter(self):
        rocdl.sched_barrier(0)
        is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(self.tidx), fx.Index(0))
        is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
        with ir.InsertionPoint(is_t0_cond_if.then_block):
            prev = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                self.counter_ptr_v,
                arith.constant(1, type=T.i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
            self.smem_flag[fx.Index(0)] = prev
            scf.YieldOp([])
    
    def wait(self):
        rocdl.sched_barrier(0)
        gpu.barrier()
        flag = self.smem_flag[fx.Index(0)]
        ilb_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(flag), fx.Index(self.group_size - 1))
        ilb_cond_if = scf.IfOp(ilb_cond, results_=[], has_else=True)
        with ir.InsertionPoint(ilb_cond_if.then_block):
            self.counter_[fx.Int32(0)] = arith.constant(0, type=T.i32)
            rocdl.s_waitcnt(0)
            rocdl.sched_barrier(0)
            scf.YieldOp([])
        with ir.InsertionPoint(ilb_cond_if.else_block):
            init_cur = arith.constant(1, type=T.i32)
            w = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(arith.CmpIPredicate.ne, cur, arith.constant(0, type=T.i32)).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                value = llvm.InlineAsmOp(
                    T.i32, [self.counter_ptr_v],
                    "global_load_dword $0, $1, off sc1", "=v,v",
                    has_side_effects=True,
                ).result
                rocdl.s_waitcnt(0)
                scf.YieldOp([value])
            scf.YieldOp([])
        gpu.barrier()
