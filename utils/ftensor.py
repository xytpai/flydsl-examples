# flydsl: pre_v0.1

import numpy as np
from itertools import product
from abc import ABC, abstractmethod

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
    
    def _lazy_init(self):
        pass
    
    def __repr__(self):
        return f"TensorView(offset={self.base_offset}, shape={self.shape}, stride={self.stride}, dtype={self.dtype})"

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
    
    def local_tile(self, tile_shape, tile_idxs):
        offset = self.base_offset
        for i in range_constexpr(len(tile_idxs)):
            offset = offset + tile_idxs[i] * tile_shape[i] * self.stride[i]
        return FTensorView(self.dtype, tile_shape, self.stride, offset, self.load_impl, self.store_impl)
    
    def copy_(self, src_tensor, thread_layout, value_layout, thread_idxs, vec_size):
        src_tensor._lazy_init()
        ndim = len(thread_layout)
        src_offset = src_tensor.base_offset
        dst_offset = self.base_offset
        for d in range_constexpr(ndim):
            thread_offset_d = thread_idxs[d] * value_layout[d]
            src_offset = src_offset + thread_offset_d * src_tensor.stride[d]
            dst_offset = dst_offset + thread_offset_d * self.stride[d]
        value_layout_v = value_layout[:-1] + (value_layout[-1] // vec_size,)
        coords = list(product(*(range(s) for s in value_layout_v)))
        for coord in coords:
            src_vec_offset = src_offset
            dst_vec_offset = dst_offset
            for d in range_constexpr(len(coord)):
                if d == len(coord) - 1:
                    src_vec_offset = src_vec_offset + coord[d] * src_tensor.stride[d] * vec_size
                    dst_vec_offset = dst_vec_offset + coord[d] * self.stride[d] * vec_size
                else:
                    src_vec_offset = src_vec_offset + coord[d] * src_tensor.stride[d]
                    dst_vec_offset = dst_vec_offset + coord[d] * self.stride[d]
            value = src_tensor.load_impl(src_vec_offset, vec_size=vec_size)
            self.store_impl(dst_vec_offset, value, vec_size=vec_size)


class FTensorBase(ABC):
    def __init__(self, dtype, shape, stride=None, base_offset=0):
        self.tensor_view = None
        self.dtype = dtype
        self.shape = shape
        self.stride = stride
        self.base_offset = base_offset
    
    @abstractmethod
    def load(self, offset):
        return None
    
    @abstractmethod
    def store(self, offset, value):
        pass
    
    def _lazy_init(self):
        if self.tensor_view is None:
            self.tensor_view = FTensorView(
                self.dtype, self.shape, self.stride, self.base_offset, self.load, self.store)
            self.stride = self.tensor_view.stride
            self.load_impl = self.tensor_view.load_impl
            self.store_impl = self.tensor_view.store_impl
    
    def __repr__(self):
        self._lazy_init()
        return self.tensor_view.__repr__()

    def __getitem__(self, idxs):
        self._lazy_init()
        return self.tensor_view[idxs]

    def __setitem__(self, idxs, value):
        self._lazy_init()
        self.tensor_view[idxs] = value
    
    def local_tile(self, tile_shape, tile_idxs):
        self._lazy_init()
        return self.tensor_view.local_tile(tile_shape, tile_idxs)
    
    def copy_(src_tensor, thread_layout, value_layout, thread_idxs, vec_size):
        self._lazy_init()
        self.tensor_view.copy_(src_tensor, thread_layout, value_layout, thread_idxs, vec_size)


class TorchTensor(FTensorBase):
    def __init__(self, torch_tensor, dtype, shape, stride=None, base_offset=0):
        super().__init__(dtype, shape, stride, base_offset)
        self.torch_tensor = torch_tensor
    
    def load(self, offset, vec_size=1):
        return self.torch_tensor.view(-1)[offset:offset+vec_size]
    
    def store(self, offset, value, vec_size=1):
        self.torch_tensor.view(-1)[offset:offset+vec_size] = value


class GTensor(FTensorBase):
    def __init__(self, fx_tensor, dtype, shape, stride=None, base_offset=0):
        super().__init__(dtype, shape, stride, base_offset)
        self.fx_tensor = fx_tensor
        self.rsrc = buffer_ops.create_buffer_resource(self.fx_tensor, max_size=True)
    
    def load(self, offset, vec_size=1):
        return buffer_ops.buffer_load(self.rsrc, offset, vec_width=vec_size, dtype=self.dtype)
    
    def store(self, offset, value, vec_size=1):
        buffer_ops.buffer_store(value, self.rsrc, offset, offset_is_bytes=False, vec_width=vec_size)


class STensor(FTensorBase):
    def __init__(self, fx_tensor, dtype, shape, stride=None, base_offset=0):
        super().__init__(dtype, shape, stride, base_offset)
        self.fx_tensor = fx_tensor
    
    def load(self, offset, vec_size=1):
        return self.fx_tensor.load([arith.as_value(offset)])
    
    def store(self, offset, value, vec_size=1):
        self.fx_tensor.store(value, [arith.as_value(offset)])


if __name__ == '__main__':
    print('==== test ftensor ===')
    import torch
    a_shape = (4, 8)
    b_shape = (4, 4)
    tile_shape = (4, 4)
    tile_idxs = (0, 1)
    thread_layout = (2, 2)
    value_layout = (2, 2)
    thread_idxs = (0, 1)
    print(
        f"a_shape:{a_shape}",
        f"b_shape:{b_shape}",
        f"tile_shape:{tile_shape}",
        f"tile_idxs:{tile_idxs}",
        f"thread_layout:{thread_layout}",
        f"value_layout:{value_layout}",
        f"thread_idxs:{thread_idxs}",
    )
    a = torch.zeros(a_shape)
    b = torch.FloatTensor([[1,2,3,4],[5,6,7,8],[2,3,4,5],[6,7,8,9]])
    print(a)
    print(b)
    a_tensor = TorchTensor(a, torch.float, a_shape)
    b_tensor = TorchTensor(b, torch.float, b_shape)
    a_tensor[(1, None)][(7,)] = 9
    assert a_tensor[(1, None)][(7,)].item() == 9
    a_tensor[(1, 7)] = 0
    a_tensor_tiled = a_tensor.local_tile(
        tile_shape=tile_shape,
        tile_idxs=tile_idxs
    ).copy_(
        b_tensor,
        thread_layout=thread_layout,
        value_layout=value_layout,
        thread_idxs=thread_idxs,
        vec_size=2
    )
    assert a_tensor[(0, 6)].item() == 3
    assert a_tensor[(0, 7)].item() == 4
    assert a_tensor[(1, 6)].item() == 7
    assert a_tensor[(1, 7)].item() == 8
    print(a)
