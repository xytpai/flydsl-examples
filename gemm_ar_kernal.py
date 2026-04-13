# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL HGEMM + tensor-parallel all-reduce APIs."""

from __future__ import annotations

from typing import Optional

import torch
from flydsl.expr.typing import Int32, Int64
from flydsl.runtime.device import get_rocm_arch

from aiter.dist.parallel_state import get_tp_group

from gemm_kernal import (
    SPLIT_K_COUNTER_MAX_LEN,
    _advance_split_k_signal_state,
    _get_flydsl_shuffle_layout,
    _get_split_k_global_semaphore,
    _get_split_k_signal_state,
    _normalize_launch_stream,
    _to_kernel_dtype,
    _validate_hgemm_inputs,
    _validate_hgemm_tiling,
    shuffle_weight,
)
from utils.custom_all_reduce import FlyDSLAllreduce, _DEFAULT_MAX_SIZE

__all__ = ["flydsl_hgemm_ar"]

FIXED_STAGE = 2
FIXED_C_TO_LDS = False
KERNEL_ASYNC_COPY = get_rocm_arch() != "gfx942"
HGEMM_AR_MAX_BLOCKS = 80

_BACKEND_CACHE: dict[tuple[tuple[int, ...], int, int], "_GEMMARBackend"] = {}

_KWARG_MAP = {
    "tile_m": "TILE_M",
    "tile_n": "TILE_N",
    "tile_k": "TILE_K",
    "split_k": "SPLIT_K",
    "block_m_warps": "BLOCK_M_WARPS",
    "block_n_warps": "BLOCK_N_WARPS",
    "b_preshuffle": "B_PRE_SHUFFLE",
    "b_to_lds": "B_TO_LDS",
    "use_atomic_add": "USE_CROSS_DEVICE_ATOMIC",
}

_REQUIRED_HGEMM_AR_KWARGS = (
    "TILE_M",
    "TILE_N",
    "TILE_K",
    "SPLIT_K",
    "BLOCK_M_WARPS",
    "BLOCK_N_WARPS",
    "B_PRE_SHUFFLE",
    "B_TO_LDS",
)


def _compile_hgemm_ar_kernel(*args, **kwargs):
    from hgemm_ar import compile_hgemm_ar_kernel

    return compile_hgemm_ar_kernel(*args, **kwargs)


def _normalize_supported_kernel_metadata(
    *,
    stage: int,
    async_copy: bool,
    c_to_lds: bool,
) -> tuple[int, bool, bool]:
    if int(stage) != FIXED_STAGE:
        raise ValueError(
            f"Current kernel only supports stage={FIXED_STAGE}; got stage={stage}"
        )
    if bool(async_copy) != KERNEL_ASYNC_COPY:
        raise ValueError(
            "Current kernel fixes async_copy based on GPU arch; "
            f"got async_copy={async_copy}, expected {KERNEL_ASYNC_COPY}"
        )
    if bool(c_to_lds) != FIXED_C_TO_LDS:
        raise ValueError(
            f"Current kernel only supports c_to_lds={FIXED_C_TO_LDS}; got {c_to_lds}"
        )
    return FIXED_STAGE, KERNEL_ASYNC_COPY, FIXED_C_TO_LDS


def _check_hgemm_ar_block_capacity(
    m: int,
    n: int,
    tile_m: int,
    tile_n: int,
    split_k: int,
) -> None:
    bm = (m + tile_m - 1) // tile_m
    bn = n // tile_n
    required = bm * bn * split_k
    limit = min(HGEMM_AR_MAX_BLOCKS, SPLIT_K_COUNTER_MAX_LEN)
    if required > limit:
        raise ValueError(
            "Fused FlyDSL HGEMM AR block capacity exceeded: "
            f"requires {required} logical blocks, max supported is {limit}"
        )


def _normalize_hgemm_ar_kwargs(
    hgemm_kwargs: Optional[dict[str, object]],
) -> dict[str, object]:
    if not hgemm_kwargs:
        raise ValueError(
            "FlyDSL HGEMM AR requires an explicit kernel config; "
            "no default kernel config is defined."
        )

    kwargs: dict[str, object] = {}
    for key, value in hgemm_kwargs.items():
        if value is None:
            continue
        kwargs[_KWARG_MAP.get(key, key)] = value

    missing = [key for key in _REQUIRED_HGEMM_AR_KWARGS if key not in kwargs]
    if missing:
        raise ValueError(
            "FlyDSL HGEMM AR requires explicit kernel config values; "
            f"missing: {', '.join(missing)}"
        )

    if bool(kwargs["B_PRE_SHUFFLE"]) and bool(kwargs["B_TO_LDS"]):
        raise ValueError(
            "FlyDSL HGEMM AR does not support b_preshuffle=True with b_to_lds=True"
        )

    kwargs["USE_CROSS_DEVICE_ATOMIC"] = bool(
        kwargs.get("USE_CROSS_DEVICE_ATOMIC", False)
    ) or int(kwargs["SPLIT_K"]) > 1
    return kwargs


def _run_hgemm_ar(
    world_size: int,
    rank: Int32,
    self_sg: Int64,
    sg_ptrs: Int64,
    tmp_ptrs: Int64,
    out_ptrs: Int64,
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    shuffle_b: bool = False,
    hgemm_kwargs: Optional[dict[str, object]] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    launch_stream = _normalize_launch_stream(a.device, stream)
    signal_state = _get_split_k_signal_state(launch_stream)
    semaphore = _get_split_k_global_semaphore(launch_stream)

    k = a.shape[-1]
    a = a.view(-1, k)
    m = a.shape[0]
    n = b.shape[0]
    assert b.shape[1] == k
    c = c.view(-1, n)
    assert c.shape[0] == m

    kwargs = _normalize_hgemm_ar_kwargs(hgemm_kwargs)
    exe = _compile_hgemm_ar_kernel(world_size, _to_kernel_dtype(a.dtype), n, k, **kwargs)

    if bool(kwargs["B_PRE_SHUFFLE"]) and shuffle_b:
        b = shuffle_weight(b, layout=_get_flydsl_shuffle_layout(1))

    _check_hgemm_ar_block_capacity(
        m,
        n,
        int(kwargs["TILE_M"]),
        int(kwargs["TILE_N"]),
        int(kwargs["SPLIT_K"]),
    )

    exe_compiled = exe.compile(
        rank,
        self_sg,
        sg_ptrs,
        tmp_ptrs,
        out_ptrs,
        c,
        a,
        b,
        m,
        semaphore,
        signal_state,
        launch_stream,
    )
    exe_compiled(
        rank,
        self_sg,
        sg_ptrs,
        tmp_ptrs,
        out_ptrs,
        c,
        a,
        b,
        m,
        semaphore,
        signal_state,
        launch_stream,
    )
    if int(kwargs["SPLIT_K"]) > 1:
        _advance_split_k_signal_state(launch_stream)


class _GEMMARBackend(FlyDSLAllreduce):
    def should_fuse(self, out: torch.Tensor) -> bool:
        out_bytes = int(out.numel()) * int(out.element_size())
        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            return False
        if out.dtype not in self._SUPPORTED_DTYPES:
            return False
        if out_bytes % 16 != 0 or out_bytes > self.max_size:
            return False
        if self.world_size > 2 and not self.full_nvlink:
            return False
        return True

    def hgemm_ar_fusion(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        *,
        shuffle_b: bool = False,
        hgemm_kwargs: Optional[dict[str, object]] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        m, n = a.shape[0], b.shape[0]
        if c is None:
            c = torch.empty((m, n), dtype=a.dtype, device=a.device)
        if not self.should_fuse(c):
            out_bytes = int(c.numel()) * int(c.element_size())
            raise ValueError(
                f"Output tensor with {out_bytes} bytes is not supported by FlyDSL HGEMM AR"
            )

        launch_stream = _normalize_launch_stream(a.device, stream)
        rank = Int32(self.rank)
        self_sg = Int64(self._self_sg)
        sg_ptrs = Int64(int(self._gpu_sg_ptrs_array.data_ptr()))
        tmp_ptrs = Int64(int(self._gpu_tmp_ptrs_array.data_ptr()))
        out_bytes = int(c.numel()) * int(c.element_size())
        is_graph_capture = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        if is_graph_capture:
            graph_out_ptrs = self._get_or_create_graph_ptrs(c, False)
            out_ptrs = Int64(int(graph_out_ptrs.data_ptr()))
        else:
            out_ptrs = Int64(int(self._gpu_output_buffer_ptrs_array.data_ptr()))

        with torch.cuda.stream(launch_stream):
            _run_hgemm_ar(
                self.world_size,
                rank,
                self_sg,
                sg_ptrs,
                tmp_ptrs,
                out_ptrs,
                c,
                a,
                b,
                shuffle_b=shuffle_b,
                hgemm_kwargs=hgemm_kwargs,
                stream=launch_stream,
            )
            if not is_graph_capture:
                c.view(-1).view(torch.uint8)[:out_bytes].copy_(
                    self.output_buffer[:out_bytes]
                )
        return c


def _get_tp_flydsl_hgemm_ar_backend(
    device: torch.device,
    *,
    max_size: int = _DEFAULT_MAX_SIZE,
) -> _GEMMARBackend:
    tp_group = get_tp_group()
    device_index = device.index
    if device_index is None:
        raise ValueError(f"Unable to determine device index for {device}")

    ranks = tuple(tp_group.ranks)
    cache_key = (ranks, device_index, int(max_size))
    backend = _BACKEND_CACHE.get(cache_key)
    if backend is not None:
        return backend

    ca_comm = getattr(getattr(tp_group, "device_communicator", None), "ca_comm", None)
    full_nvlink = getattr(ca_comm, "fully_connected", True)
    backend = _GEMMARBackend(
        group=tp_group.cpu_group,
        device=device,
        max_size=max_size,
        world_size=tp_group.world_size,
        rank=tp_group.rank_in_group,
        full_nvlink=full_nvlink,
    )
    _BACKEND_CACHE[cache_key] = backend
    return backend


def _get_tp_flydsl_hgemm_ar_capture_context(
    device: Optional[torch.device] = None,
    *,
    max_size: int = _DEFAULT_MAX_SIZE,
):
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return _get_tp_flydsl_hgemm_ar_backend(device, max_size=max_size).capture()


def flydsl_hgemm_ar(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    pack_n: int = 1,
    split_k: int = 1,
    block_m_warps: int = 2,
    block_n_warps: int = 2,
    stages: int = FIXED_STAGE,
    async_copy: bool = KERNEL_ASYNC_COPY,
    b_to_lds: bool = True,
    b_preshuffle: bool = False,
    use_atomic_add: bool = False,
    auto_shuffle_b: bool = False,
    c_to_lds: bool = FIXED_C_TO_LDS,
    stream: Optional[torch.cuda.Stream] = None,
    max_size: int = _DEFAULT_MAX_SIZE,
) -> torch.Tensor:
    m, n, k = _validate_hgemm_inputs(a, b, out, None)
    _normalize_supported_kernel_metadata(
        stage=stages,
        async_copy=async_copy,
        c_to_lds=c_to_lds,
    )
    _validate_hgemm_tiling(
        m,
        n,
        k,
        dtype=_to_kernel_dtype(a.dtype),
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        pack_n=pack_n,
        split_k=split_k,
        stages=stages,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        b_to_lds=b_to_lds,
    )

    if b_preshuffle and b_to_lds:
        raise ValueError(
            "FlyDSL HGEMM AR does not support b_preshuffle=True with b_to_lds=True"
        )

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    if b_preshuffle and not getattr(b, "is_shuffled", False):
        if auto_shuffle_b:
            b = shuffle_weight(b, layout=_get_flydsl_shuffle_layout(pack_n))
        else:
            raise ValueError(
                "`b_preshuffle=True` expects `b` to be pre-shuffled. "
                f"Use `shuffle_weight(b, layout={_get_flydsl_shuffle_layout(pack_n)})` "
                "first or pass `auto_shuffle_b=True`."
            )

    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    backend = _get_tp_flydsl_hgemm_ar_backend(a.device, max_size=max_size)
    if not backend.should_fuse(out):
        raise ValueError("Current tensor-parallel setup does not support FlyDSL HGEMM AR")

    return backend.hgemm_ar_fusion(
        a,
        b,
        out,
        shuffle_b=False,
        hgemm_kwargs={
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "split_k": split_k,
            "block_m_warps": block_m_warps,
            "block_n_warps": block_n_warps,
            "b_to_lds": b_to_lds,
            "b_preshuffle": b_preshuffle,
            "use_atomic_add": use_atomic_add,
        },
        stream=stream,
    )
