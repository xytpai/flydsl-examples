# flydsl: pre_v0.1

import numpy as np
import flydsl
from flydsl.dialects.ext import flir, gpu, arith, buffer_ops
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.pipeline import Pipeline, run_pipeline
from flydsl.dialects.ext.python_control_flow import range_constexpr, lower_range_for_loops
from flydsl.utils import SmemAllocator
fm_fast = flir.arith.FastMathFlags.fast
_asid = flir.const_index
from _mlir import ir
from _mlir.ir import F16Type, BF16Type, F32Type, IntegerType, VectorType, IndexType
import _mlir.extras.types as T


class FTensorView:
    def __init__(self, dtype, shape, stride, base_offset, load_impl, store_impl):
        self.dtype = dtype
        self.shape = shape
        if stride is None:
            self.stride = tuple((np.cumprod(shape[::-1])[::-1].tolist()+[1,])[1:])
        else:
            self.stride = stride
        self.base_offset = base_offset
        self.load_impl = load_impl
        self.store_impl = store_impl
    
    def linear_offset(self, idxs):
        slice_shape = []
        slice_stride = []
        offset = self.base_offset
        for i in range_constexpr(len(idxs)):
            d = idxs[i]
            if d is None:
                slice_shape.append(self.shape[i])
                slice_stride.append(self.stride[i])
                d = 0
            offset = offset + d * self.stride[i]
        if len(slice_shape) > 0:
            return offset, slice_shape, slice_stride
        else:
            return (offset,)
    
    def __getitem__(self, idxs):
        offset = self.linear_offset(idxs)
        if len(offset) == 1:
            return self.load_impl(offset[0])
        else:
            return FTensorView(self.dtype, offset[1], offset[2], offset[0], self.load_impl, self.store_impl)
    
    def __setitem__(self, idxs, value):
        offset = self.linear_offset(idxs)
        assert len(offset) == 1
        self.store_impl(offset[0], value)


class GTensor:
    def __init__(self, fx_tensor, dtype, shape, stride=None, base_offset=0, vec_size=1, init=True):
        self.tensor_view = None
        self.fx_tensor = fx_tensor
        self.dtype = dtype
        self.shape = shape
        self.stride = stride
        self.base_offset = base_offset
        self.vec_size = vec_size
        if init:
            self.rsrc = buffer_ops.create_buffer_resource(self.fx_tensor, max_size=True)
        else:
            self.rsrc = None
    
    def load(self, offset):
        return buffer_ops.buffer_load(self.rsrc, offset, vec_width=self.vec_size, dtype=self.dtype)
    
    def store(self, offset, value):
        buffer_ops.buffer_store(value, self.rsrc, offset, offset_is_bytes=False)

    def __getitem__(self, idxs):
        if self.tensor_view is None:
            self.tensor_view = FTensorView(
                self.dtype, self.shape, self.stride, self.base_offset, self.load, self.store)
        return self.tensor_view[idxs]

    def __setitem__(self, idxs, value):
        if self.tensor_view is None:
            self.tensor_view = FTensorView(
                self.dtype, self.shape, self.stride, self.base_offset, self.load, self.store)
        self.tensor_view[idxs] = value


class STensor(GTensor):
    def __init__(self, fx_tensor, dtype, shape, stride=None, base_offset=0):
        super().__init__(fx_tensor, dtype, shape, stride, base_offset, 1, False)
    
    def load(self, offset):
        return self.fx_tensor.load([arith.as_value(offset)])
    
    def store(self, offset, value):
        self.fx_tensor.store(value, [arith.as_value(offset)])
