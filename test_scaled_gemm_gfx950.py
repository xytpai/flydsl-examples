"""Scaled GEMM path regression tests.

Correctness (no autotuning / large rotary allocations):
    HIP_VISIBLE_DEVICES=0 pytest -q test_scaled_gemm_gfx950.py -k "not benchmark"

Focused suites:
    pytest -q test_scaled_gemm_gfx950.py -k "path_matrix or pipeline_paths"
    pytest -q test_scaled_gemm_gfx950.py -k "stream or graph or dispatch"
    pytest -q test_scaled_gemm_gfx950.py -k "rejects or invalid or param"

Coverage:
- 16x16x128 / 32x32x64 MMA; full-tile / HTI; NN/NT/TN/TT.
- Split-K, local slice-K=2/4, split+slice, multiple MMA K steps.
- BF16/FP32 output x bias on/off; exact and dense-random references.
- Drain/main/stage wrap; unequal valid split partitions; group_m branches.
- M/N tails, small M/N, padded strides/offsets, strided scales/bias, guards.
- Allocation/dtype inference, dynamic dispatch, stream-local sync and graphs.
- Policy/shape/DMA/API rejection and benchmark helper control flow.

Known implementation/compiler defects remain executable strict xfails:
- TN + mma16 + slice4, with/without split-K: LLVM dominance failure.
  Four-slice tests run in subprocesses so a compiler abort cannot kill pytest.
- Singleton non-unit-stride scale: Torch contiguous != FlyDSL unit stride.
- Nonpositive M/N accepted by the shape validator (host-only tests).

The three original performance benchmarks are intentionally separate.
Benchmark-helper smoke tests mock optional comparators; they do not claim
to validate the Triton autotuner or replace real performance measurements.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from dataclasses import dataclass, replace
from torch.profiler import ProfilerActivity, profile

from kernels.gemm_a16w16_gfx950_utils import GFX950_DMA_BYTES
from kernels.scaled_gemm_gfx950 import (
    SCALED_GEMM_DTYPE_FP8,
    SCALED_GEMM_DTYPE_BF16,
    SCALED_GEMM_DTYPE_FP32,
    assert_no_k_tail,
    make_scaled_gemm_gfx950_param,
    make_scaled_gemm_gfx950_kernel_name,
    make_scaled_gemm_param_and_validate,
    scaled_gemm,
)
import kernels.scaled_gemm_gfx950 as scaled_module

ROTARY_INPUTS_TARGET_BYTES = 8 * 1024**3


@dataclass
class _TestArgs:
    m: int
    n: int
    k: int
    block_m: int
    block_n: int
    block_k: int
    stages: int
    m_waves: int
    n_waves: int
    k_waves: int
    group_m: int
    has_bias: bool
    layout: str = "nt"
    split_k: int = 1
    out_dtype: torch.dtype = torch.bfloat16
    mma_m: int = 16
    mma_n: int = 16
    mma_k: int = 128
    use_half_tile_interleaved: bool = False


def _skip_if_no_fp8():
    if not torch.cuda.is_available():
        pytest.skip("GPU is required")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    if arch.split(":")[0] != "gfx950":
        pytest.skip("scaled GEMM requires gfx950")


def _a_dma_vec_size() -> int:
    return GFX950_DMA_BYTES


def _skip_unsupported_accuracy_layout(args: _TestArgs):
    if args.layout[0] == "t" and args.m % _a_dma_vec_size() != 0:
        pytest.skip(
            "column-major A requires M divisible by "
            f"{_a_dma_vec_size()} for GFX950 DMA; got M={args.m}"
        )


def empty_layout_matrix(rows: int, cols: int, dtype: torch.dtype, is_t: bool):
    if is_t:
        return torch.empty((cols, rows), dtype=dtype, device="cuda").t()
    return torch.empty((rows, cols), dtype=dtype, device="cuda")


def _fill_fp8(tensor):
    tmp = torch.empty(tensor.shape, dtype=torch.bfloat16, device=tensor.device)
    tmp.uniform_(-1, 1)
    tensor.copy_(tmp.to(torch.float8_e4m3fn))
    return tensor


def create_inputs(args: _TestArgs):
    a = _fill_fp8(
        empty_layout_matrix(args.m, args.k, torch.float8_e4m3fn, args.layout[0] == "t")
    )
    b = _fill_fp8(
        empty_layout_matrix(args.k, args.n, torch.float8_e4m3fn, args.layout[1] == "t")
    )
    scale_a = torch.rand(args.m, device="cuda") * 0.25 + 0.05
    scale_b = torch.rand(args.n, device="cuda") * 0.25 + 0.05
    if args.has_bias:
        bias = torch.empty((args.n,), dtype=args.out_dtype, device="cuda")
        bias.uniform_(0.1, 1.0)
    else:
        bias = None
    return a, b, scale_a, scale_b, bias


def create_outputs(args: _TestArgs):
    return (torch.empty((args.m, args.n), dtype=args.out_dtype, device="cuda"),)


def ref_func(a, b, scale_a, scale_b, bias, c):
    ref = torch.mm(a.float(), b.float()) * scale_a[:, None] * scale_b[None, :]
    if bias is not None:
        ref = ref + bias.float()
    c.copy_(ref.to(c.dtype))


def func(a, b, scale_a, scale_b, bias, c, kwargs, layout):
    scaled_gemm(
        a,
        b,
        scale_a,
        scale_b,
        out=c,
        bias=bias,
        user_kwargs=kwargs,
        layout=layout,
        out_dtype=c.dtype,
    )


def _ptpc_scale_views(scale_a, scale_b):
    return scale_a.view(-1, 1).contiguous(), scale_b.view(1, -1).contiguous()


def scaled_mm_func(a, b, scale_a_2d, scale_b_2d, c):
    torch._scaled_mm(
        a,
        b,
        scale_a=scale_a_2d,
        scale_b=scale_b_2d,
        out_dtype=c.dtype,
        out=c,
    )


def make_triton_maxautotune_func():
    import torch._inductor.config as inductor_config

    inductor_config.max_autotune_gemm_backends = "TRITON"
    inductor_config.max_autotune_gemm_search_space = "DEFAULT"
    torch._dynamo.reset()

    def triton_maxautotune_func(a, b, scale_a_2d, scale_b_2d, c):
        out = torch._scaled_mm(
            a,
            b,
            scale_a=scale_a_2d,
            scale_b=scale_b_2d,
            out_dtype=c.dtype,
        )
        c.copy_(out)

    return torch.compile(triton_maxautotune_func, mode="max-autotune", fullgraph=True)


def tensor_nbytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def get_rotary_inputs(sample_inputs, sample_outputs):
    slot_bytes = 2 * (tensor_nbytes(sample_inputs) + tensor_nbytes(sample_outputs))
    return max(1, ROTARY_INPUTS_TARGET_BYTES // slot_bytes)


def kernel_kwargs(args: _TestArgs):
    return {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "stages": args.stages,
        "m_waves": args.m_waves,
        "n_waves": args.n_waves,
        "k_waves": args.k_waves,
        "group_m": args.group_m,
        "use_half_tile_interleaved": args.use_half_tile_interleaved,
        "split_k": args.split_k,
        "mma_m": args.mma_m,
        "mma_n": args.mma_n,
        "mma_k": args.mma_k,
    }


def check_acc(args: _TestArgs):
    _skip_if_no_fp8()
    _skip_unsupported_accuracy_layout(args)
    kwargs = kernel_kwargs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    k_scale = (args.k / 8192) ** 0.5 * args.split_k * args.k_waves
    atol = 0.5 * k_scale * (1.5 if args.has_bias else 1.0)
    rtol = 0.5
    maxdiff_out_ = []
    for _ in range(3):
        inputs = create_inputs(args)
        for output in outputs:
            output.fill_(float("nan"))
        func(*inputs, *outputs, kwargs, args.layout)
        ref_func(*inputs, *ref_outputs)
        for output, ref_output in zip(outputs, ref_outputs):
            maxdiff_out = (output.float() - ref_output.float()).abs().max().item()
            maxdiff_out_.append(maxdiff_out)
            print(maxdiff_out, flush=True)
            torch.testing.assert_close(
                output.float(),
                ref_output.float(),
                atol=atol,
                rtol=rtol,
            )
    print(f"\n{args}\nmaxdiff_out:{maxdiff_out_}")


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias",
    [
        (512, 512, 512, 256, 256, 128, 2, 2, 4, 0, False),
        (512, 512, 512, 256, 256, 128, 2, 2, 4, 4, True),
        (528, 528, 512, 256, 256, 128, 2, 2, 4, 0, False),
        (512, 512, 512 + 128, 256, 256, 128, 2, 2, 4, 0, False),
        (2048, 2048, 2048, 128, 128, 128, 2, 4, 4, 0, False),
    ],
)
def test_scaled_gemm_acc_main_loop(
    layout: str,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
):
    check_acc(
        _TestArgs(
            m,
            n,
            k,
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            1,
            group_m,
            has_bias,
            layout,
        )
    )


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, split_k, "
    "m_waves, n_waves, k_waves, has_bias, group_m",
    [
        (32, 384, 7168, 32, 64, 128, 5, 8, 2, 2, 1, True, 0),
        (32, 384, 7168, 32, 64, 128, 5, 8, 2, 2, 1, False, 0),
        (800, 384, 7168, 32, 64, 256, 6, 1, 1, 2, 2, True, 0),
        (800, 384, 7168, 32, 64, 256, 6, 2, 1, 2, 2, False, 0),
    ],
)
def test_scaled_gemm_acc_split_k(
    layout: str,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    split_k: int,
    m_waves: int,
    n_waves: int,
    k_waves: int,
    has_bias: bool,
    group_m: int,
):
    assert k_waves > 0
    assert split_k > 1 or k_waves > 1
    check_acc(
        _TestArgs(
            m=m,
            n=n,
            k=k,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            k_waves=k_waves,
            group_m=group_m,
            has_bias=has_bias,
            layout=layout,
            split_k=split_k,
        )
    )


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("block_k", [64, 128])
def test_scaled_gemm_acc_mma_32x32x64(layout: str, has_bias: bool, block_k: int):
    check_acc(
        _TestArgs(
            m=64,
            n=64,
            k=256,
            block_m=64,
            block_n=64,
            block_k=block_k,
            stages=2,
            m_waves=2,
            n_waves=2,
            k_waves=1,
            group_m=0,
            has_bias=has_bias,
            layout=layout,
            mma_m=32,
            mma_n=32,
            mma_k=64,
        )
    )


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias",
    [
        # Peak NT tile and group_m / bias.
        (512, 512, 512, 256, 256, 128, 2, 2, 4, 0, False),
        (512, 512, 512, 256, 256, 128, 2, 2, 4, 4, True),
        # Partial M/N that still satisfy DMA alignment for all layouts.
        (528, 528, 512, 256, 256, 128, 2, 2, 4, 0, False),
        (80, 80, 512, 64, 64, 128, 2, 2, 2, 0, True),
        # Minimum even K-tile count (2).
        (256, 256, 256, 256, 256, 128, 2, 2, 4, 0, False),
        (64, 64, 256, 64, 64, 128, 2, 2, 2, 0, False),
        # Smaller tiles and mixed MN.
        (64, 64, 512, 64, 64, 128, 2, 2, 2, 0, False),
        (128, 128, 512, 128, 128, 128, 2, 2, 2, 0, False),
        (256, 256, 512, 128, 256, 128, 2, 2, 4, 0, False),
        (256, 128, 512, 256, 128, 128, 2, 2, 2, 4, False),
        (2048, 2048, 2048, 128, 128, 128, 2, 2, 2, 0, False),
        # Wider wave maps and block_k variants that still cover a half-tile DMA.
        (256, 256, 512, 256, 256, 128, 2, 2, 8, 0, False),
        (64, 64, 512, 64, 64, 256, 2, 2, 2, 0, False),
    ],
)
def test_scaled_gemm_acc_hti(
    layout: str,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
):
    check_acc(
        _TestArgs(
            m=m,
            n=n,
            k=k,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            k_waves=1,
            group_m=group_m,
            has_bias=has_bias,
            layout=layout,
            use_half_tile_interleaved=True,
        )
    )


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_bias", [False, True])
def test_scaled_gemm_acc_hti_out_dtype(layout: str, out_dtype: torch.dtype, has_bias: bool):
    check_acc(
        _TestArgs(
            m=256,
            n=256,
            k=512,
            block_m=256,
            block_n=256,
            block_k=128,
            stages=2,
            m_waves=2,
            n_waves=4,
            k_waves=1,
            group_m=0,
            has_bias=has_bias,
            layout=layout,
            out_dtype=out_dtype,
            use_half_tile_interleaved=True,
        )
    )


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("block_k", [64, 128])
def test_scaled_gemm_acc_hti_mma_32x32x64(layout: str, has_bias: bool, block_k: int):
    check_acc(
        _TestArgs(
            m=128,
            n=128,
            k=256,
            block_m=128,
            block_n=128,
            block_k=block_k,
            stages=2,
            m_waves=2,
            n_waves=2,
            k_waves=1,
            group_m=0,
            has_bias=has_bias,
            layout=layout,
            mma_m=32,
            mma_n=32,
            mma_k=64,
            use_half_tile_interleaved=True,
        )
    )


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, split_k, "
    "m_waves, n_waves, has_bias, group_m",
    [
        (64, 64, 512, 64, 64, 128, 2, 2, 2, 2, True, 0),
        (64, 64, 512, 64, 64, 128, 2, 2, 2, 2, False, 0),
        (80, 96, 1024, 64, 64, 128, 2, 2, 2, 2, False, 0),
        (64, 384, 7168, 64, 64, 128, 2, 7, 2, 2, False, 0),
        (64, 384, 7168, 64, 64, 128, 2, 7, 2, 2, True, 4),
        (128, 128, 1024, 128, 128, 128, 2, 4, 2, 2, False, 4),
        (256, 256, 1024, 256, 256, 128, 2, 2, 2, 4, True, 0),
        (2048, 2048, 2048, 128, 128, 128, 2, 4, 2, 2, True, 0),
    ],
)
def test_scaled_gemm_acc_hti_split_k(
    layout: str,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    split_k: int,
    m_waves: int,
    n_waves: int,
    has_bias: bool,
    group_m: int,
):
    check_acc(
        _TestArgs(
            m=m,
            n=n,
            k=k,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            k_waves=1,
            group_m=group_m,
            has_bias=has_bias,
            layout=layout,
            split_k=split_k,
            use_half_tile_interleaved=True,
        )
    )


# Path coverage is intentionally separate from performance benchmarks.
# 2 MMA shapes x 6 reduction modes x 4 layouts x 2 outputs x bias on/off.
# Inputs are sparse, exactly representable FP8 integers and binary scales:
# every partial sum/bias fits BF16 exactly, including split/slice reductions.
# Thus atol=rtol=0 catches lost K slices, double bias, bad lane maps and tails.
_PATH_MODES = ("full", "full-split", "slice2", "split-slice2", "hti", "hti-split")
_MMA_SHAPES = ((16, 16, 128), (32, 32, 64))


def _path_args(mma, mode, *, layout="nt", out_dtype=torch.bfloat16, has_bias=False):
    mma_m, mma_n, mma_k = mma
    hti = mode.startswith("hti")
    tile = 128 if hti else 64
    k_waves = 2 if "slice" in mode else 1
    split_k = 2 if "split" in mode else 1
    block_k = mma_k * k_waves
    return _TestArgs(
        m=tile + 16, n=tile + 16, k=4 * block_k * split_k,
        block_m=tile, block_n=tile, block_k=block_k, stages=2,
        m_waves=2, n_waves=2, k_waves=k_waves, group_m=0,
        has_bias=has_bias, layout=layout, split_k=split_k, out_dtype=out_dtype,
        mma_m=mma_m, mma_n=mma_n, mma_k=mma_k, use_half_tile_interleaved=hti,
    )


def _padded_matrix(rows, cols, dtype, is_t):
    # Leading strides and base offset obey DMA alignment, but differ from
    # logical extents. Fill padding with poison to detect wrong addressing.
    outer, inner = (cols, rows) if is_t else (rows, cols)
    stride = ((inner + 15) // 16) * 16 + 32
    storage = torch.empty((outer, stride), dtype=dtype, device="cuda")
    storage.copy_(torch.full(storage.shape, float("nan"), device="cuda").to(dtype))
    view = storage[:, 16 : inner + 16]
    return view.t() if is_t else view


def _exact_inputs(args, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = _padded_matrix(args.m, args.k, torch.float8_e4m3fn, args.layout[0] == "t")
    b = _padded_matrix(args.k, args.n, torch.float8_e4m3fn, args.layout[1] == "t")
    av = torch.zeros((args.m, args.k), device="cuda")
    av[:, ::32] = torch.randint(-1, 2, av[:, ::32].shape, device="cuda", generator=generator)
    a.copy_(av.to(a.dtype))
    b.copy_(torch.randint(-1, 2, b.shape, device="cuda", generator=generator).to(b.dtype))
    # Non-contiguous scales/bias exercise host materialization; include
    # negative, zero and non-unit scales (not just all-ones scale tests).
    sa = (torch.randint(-2, 3, (2 * args.m,), device="cuda", generator=generator).float() * 0.25)[::2]
    sb = (torch.randint(-2, 3, (2 * args.n,), device="cuda", generator=generator).float() * 0.25)[1::2]
    bias = None
    if args.has_bias:
        bias = (torch.randint(-2, 3, (2 * args.n,), device="cuda", generator=generator).to(args.out_dtype) * 0.25)[::2]
    return a, b, sa, sb, bias


def _exact_reference(inputs, dtype):
    a, b, sa, sb, bias = inputs
    ref = (a.float() @ b.float()) * sa[:, None] * sb[None, :]
    if bias is not None:
        ref += bias.float()
    return ref.to(dtype)


def _check_exact_path(args, repeats=2):
    _skip_if_no_fp8()
    kwargs = kernel_kwargs(args)
    param_kwargs = dict(
        kwargs, in_dtype_id=SCALED_GEMM_DTYPE_FP8,
        out_dtype_id=SCALED_GEMM_DTYPE_FP32 if args.out_dtype == torch.float32 else SCALED_GEMM_DTYPE_BF16,
        a_is_transposed=args.layout[0] == "t", b_is_transposed=args.layout[1] == "t",
        has_bias=args.has_bias,
    )
    assert make_scaled_gemm_param_and_validate(args.m, args.n, args.k, param_kwargs) is not None
    storage = torch.full((args.m * args.n + 16,), 123, dtype=args.out_dtype, device="cuda")
    out = storage[8:-8].view(args.m, args.n)
    for seed in range(7919, 7919 + repeats):
        inputs = _exact_inputs(args, seed)
        out.fill_(float("nan"))
        result = scaled_gemm(
            *inputs[:4], bias=inputs[4], out=out, user_kwargs=kwargs,
            layout=args.layout, out_dtype=args.out_dtype,
        )
        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(out, _exact_reference(inputs, args.out_dtype), atol=0, rtol=0)
        assert torch.all(storage[:8] == 123).item()
        assert torch.all(storage[-8:] == 123).item()
    if args.has_bias:
        inputs[0].zero_()
        out.fill_(float("nan"))
        scaled_gemm(*inputs[:4], bias=inputs[4], out=out, user_kwargs=kwargs, layout=args.layout)
        torch.testing.assert_close(out, inputs[4].expand_as(out), atol=0, rtol=0)


@pytest.mark.parametrize("mma", _MMA_SHAPES, ids=["mma16", "mma32"])
@pytest.mark.parametrize("mode", _PATH_MODES)
@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["no-bias", "bias"])
def test_scaled_gemm_path_matrix(mma, mode, layout, out_dtype, has_bias):
    _check_exact_path(_path_args(mma, mode, layout=layout, out_dtype=out_dtype, has_bias=has_bias))


# Pipeline control-flow coverage: drain only, main+drain, stage wraparound,
# multiple MMA steps per slice, four K waves, and unequal valid split sizes.
_PIPELINES = (
    ("full-drain", "full", dict(stages=2), 1),
    ("full-3stage-drain", "full", dict(stages=3), 2),
    ("full-3stage-wrap", "full", dict(stages=3), 7),
    ("full-5stage-wrap", "full", dict(stages=5), 9),
    ("full-multi-mma", "full", dict(block_k=256), 4),
    ("slice-multi-mma", "slice2", dict(block_k=512), 4),
    ("slice4", "slice2", dict(block_k=512, k_waves=4), 4),
    ("split-slice4", "split-slice2", dict(block_k=512, k_waves=4), 8),
    ("hti-drain", "hti", {}, 2),
    ("hti-one-pair", "hti", {}, 4),
    ("hti-wrap", "hti", {}, 6),
    ("hti-long", "hti", {}, 64),
    ("hti-multi-mma", "hti", dict(block_k=256), 4),
    ("hti-split-drain", "hti-split", {}, 4),
    ("full-unequal-split", "full-split", dict(split_k=9), 26),
)


@pytest.mark.parametrize("case,mode,overrides,tiles", _PIPELINES, ids=[p[0] for p in _PIPELINES])
@pytest.mark.parametrize("mma", _MMA_SHAPES, ids=["mma16", "mma32"])
@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
def test_scaled_gemm_pipeline_paths(case, mode, overrides, tiles, mma, layout, request):
    if case in ("slice4", "split-slice4") and mma == (16, 16, 128) and layout == "tn":
        request.node.add_marker(pytest.mark.xfail(
            strict=True, raises=_KnownCompilerFailure,
            reason="LLVM: virtual-register defs do not dominate uses for TN/mma16/slice4",
        ))
    args = replace(_path_args(mma, mode, layout=layout, has_bias=True), **overrides)
    args = replace(args, k=args.block_k * tiles)
    if args.k_waves == 4:
        # LLVM can abort rather than raise a Python exception; protect the
        # rest of the suite while still testing every four-slice configuration.
        _check_path_in_subprocess(args)
    else:
        _check_exact_path(args, repeats=1)


class _KnownCompilerFailure(RuntimeError):
    pass


def _check_path_in_subprocess(args):
    _skip_if_no_fp8()
    from dataclasses import asdict
    payload = asdict(args)
    payload["out_dtype"] = str(args.out_dtype).split(".")[-1]
    script = (
        "import json, resource, sys, torch; "
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); "
        "from test_scaled_gemm_gfx950 import _TestArgs, _check_exact_path; "
        "p=json.loads(sys.argv[1]); p['out_dtype']=getattr(torch, p['out_dtype']); "
        "_check_exact_path(_TestArgs(**p), repeats=1)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, json.dumps(payload)],
        cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=120,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0 and "Virtual register defs don't dominate all uses" in output:
        raise _KnownCompilerFailure(output[-3500:])
    assert proc.returncode == 0, output[-6000:]


# Host validation tests are GPU-independent: only parameter arithmetic is
# tested, with the architectural capacity explicitly fixed to gfx950.
@pytest.fixture
def gfx950_params(monkeypatch):
    monkeypatch.setattr(scaled_module, "get_rocm_arch", lambda: "gfx950")


def _full_kwargs(**overrides):
    return {
        **kernel_kwargs(_path_args((16, 16, 128), "full")), **overrides,
    }


@pytest.mark.parametrize("override,match", [
    ({"in_dtype_id": 2}, "unsupported in_dtype_id"),
    ({"out_dtype_id": 3}, "unsupported out_dtype_id"),
    ({"block_m": 0}, "must be positive"),
    ({"block_n": -1}, "must be positive"),
    ({"block_k": 0}, "must be positive"),
    ({"stages": 0}, "must be positive"),
    ({"split_k": 0}, "must be positive"),
    ({"stages": 1}, "stages must be at least 2"),
    ({"m_waves": 0}, "must be positive"),
    ({"n_waves": 0}, "must be positive"),
    ({"k_waves": 0}, "must be positive"),
    ({"m_waves": 8, "n_waves": 4}, "more than 16 waves"),
    ({"group_m": -1}, "group_m"),
    ({"mma_m": 32}, "requires mma"),
    ({"mma_n": 32}, "requires mma"),
    ({"mma_k": 64}, "requires mma"),
    ({"block_m": 4096, "block_n": 4096}, "shared-memory capacity"),
    ({"block_k": 64}, "divisible by k_waves"),
    ({"use_half_tile_interleaved": True, "k_waves": 2}, "does not support slice-K"),
])
def test_scaled_gemm_invalid_param_values(gfx950_params, override, match):
    kwargs = _full_kwargs(**override)
    with pytest.raises(ValueError, match=match):
        make_scaled_gemm_gfx950_param(**kwargs)
    assert make_scaled_gemm_param_and_validate(128, 128, 1024, kwargs) is None


@pytest.mark.parametrize("override", [
    {"use_half_tile_interleaved": True, "stages": 3},
    {"use_half_tile_interleaved": True, "m_waves": 1},
    {"use_half_tile_interleaved": True, "n_waves": 1},
    {"use_half_tile_interleaved": True, "block_m": 63},
    {"use_half_tile_interleaved": True, "block_n": 63},
])
def test_scaled_gemm_invalid_hti_geometry(gfx950_params, override):
    with pytest.raises(AssertionError):
        make_scaled_gemm_gfx950_param(**_full_kwargs(**override))
    assert make_scaled_gemm_param_and_validate(128, 128, 1024, _full_kwargs(**override)) is None


def test_scaled_gemm_unknown_option(gfx950_params):
    kwargs = _full_kwargs(not_a_policy_option=True)
    with pytest.raises(TypeError, match="unexpected keyword"):
        make_scaled_gemm_gfx950_param(**kwargs)
    assert make_scaled_gemm_param_and_validate(128, 128, 512, kwargs) is None


@pytest.mark.parametrize("mma", _MMA_SHAPES, ids=["mma16", "mma32"])
@pytest.mark.parametrize("mode", _PATH_MODES)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_scaled_gemm_param_derived_fields_and_name(gfx950_params, mma, mode, out_dtype):
    args = _path_args(mma, mode, out_dtype=out_dtype, layout="tn", has_bias=True)
    kwargs = dict(kernel_kwargs(args), out_dtype_id=1 if out_dtype == torch.float32 else 2,
                  a_is_transposed=True, b_is_transposed=False, has_bias=True)
    param = make_scaled_gemm_gfx950_param(**kwargs)
    assert param.block_threads == args.m_waves * args.n_waves * args.k_waves * 64
    assert param.ldg_x_threads == args.block_k // 16
    assert param.ldg_a_iters * param.block_threads * 16 == args.block_m * args.block_k
    assert param.ldg_b_iters * param.block_threads * 16 == args.block_n * args.block_k
    assert param.in_data_bytes == 1
    assert param.out_data_bytes == (4 if out_dtype == torch.float32 else 2)
    assert param.is_split_k == (args.split_k > 1)
    name = make_scaled_gemm_gfx950_kernel_name(param)
    assert f"_t{args.block_m}x{args.block_n}x{args.block_k}x{args.stages}" in name
    assert ("_ksd" if args.split_k > 1 else "_ks1") in name
    assert f"_w{args.m_waves}x{args.n_waves}x{args.k_waves}" in name
    assert "_gm0_bias1_ltn_" in name
    assert name.endswith("_phti" if args.use_half_tile_interleaved else "_pft")
    assert ("fp32" in name) == (out_dtype == torch.float32)
    assert make_scaled_gemm_param_and_validate(args.m, args.n, args.k, kwargs) is not None


@pytest.mark.parametrize("k,override", [
    (0, {}), (129, {}), (384, {"split_k": 2}), (1024, {"split_k": 3}),
    (256, {"split_k": 8}), (256, {"stages": 4}),
    (128, {"use_half_tile_interleaved": True}),
    (384, {"use_half_tile_interleaved": True}),
    (768, {"use_half_tile_interleaved": True, "split_k": 2}),
])
def test_scaled_gemm_rejects_k_partitioning(gfx950_params, k, override):
    kwargs = _full_kwargs(**override)
    with pytest.raises(AssertionError):
        assert_no_k_tail(k, kwargs)
    assert make_scaled_gemm_param_and_validate(128, 128, k, kwargs) is None


@pytest.mark.parametrize("m,n,override", [
    (128, 7, {}),
    (127, 128, {"a_is_transposed": True}),
    (128, 136, {"b_is_transposed": False}),
])
def test_scaled_gemm_shape_alignment_rejections(gfx950_params, m, n, override):
    assert make_scaled_gemm_param_and_validate(m, n, 512, _full_kwargs(**override)) is None


def test_scaled_gemm_split_k_grid_capacity(gfx950_params):
    from kernels.gemm_a16w16_gfx950_utils import SPLIT_K_SEMAPHORE_MAX_LEN
    kwargs = _full_kwargs(split_k=2)
    assert make_scaled_gemm_param_and_validate(64, 64 * SPLIT_K_SEMAPHORE_MAX_LEN, 512, kwargs) is not None
    assert make_scaled_gemm_param_and_validate(64, 64 * (SPLIT_K_SEMAPHORE_MAX_LEN + 1), 512, kwargs) is None
    # Same large grid is legal with split-K disabled.
    assert make_scaled_gemm_param_and_validate(64, 64 * (SPLIT_K_SEMAPHORE_MAX_LEN + 1), 512, _full_kwargs()) is not None


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("operand", ["a", "b"])
@pytest.mark.parametrize("fault", ["minor-stride", "pointer", "leading-stride"])
def test_scaled_gemm_rejects_bad_input_strides(layout, operand, fault):
    _skip_if_no_fp8()
    args = _path_args((16, 16, 128), "full", layout=layout)
    a, b, sa, sb, _ = create_inputs(args)
    tensor = a if operand == "a" else b
    is_t = layout[0 if operand == "a" else 1] == "t"
    rows, cols = tensor.shape
    outer, inner = (cols, rows) if is_t else (rows, cols)
    if fault == "minor-stride":
        bad = torch.empty((outer, inner * 2), device="cuda", dtype=tensor.dtype)[:, ::2]
    elif fault == "pointer":
        bad = torch.empty((outer, inner + 32), device="cuda", dtype=tensor.dtype)[:, 1:inner + 1]
    else:
        bad = torch.empty((outer, inner + 1), device="cuda", dtype=tensor.dtype)[:, :inner]
    bad = bad.t() if is_t else bad
    if operand == "a":
        a = bad
    else:
        b = bad
    with pytest.raises(ValueError, match="DMA requirements"):
        scaled_gemm(a, b, sa, sb, user_kwargs=kernel_kwargs(args), layout=layout)


@pytest.mark.parametrize("fault", [
    "layout", "input-dtype", "input-shape", "input-rank",
    "scale-a-dtype", "scale-b-dtype", "scale-a-shape", "scale-b-shape", "scale-a-rank",
    "scale-device", "out-dtype", "out-shape", "out-stride", "out-device",
    "out-dtype-conflict", "bias-shape", "bias-dtype", "unknown-option",
])
def test_scaled_gemm_rejects_bad_api_arguments(fault):
    _skip_if_no_fp8()
    args = _path_args((16, 16, 128), "full")
    a, b, sa, sb, _ = create_inputs(args)
    options = dict(user_kwargs=kernel_kwargs(args))
    expected = AssertionError
    if fault == "layout":
        options["layout"] = "invalid"
        expected = ValueError
    elif fault == "input-dtype":
        a = a.bfloat16()
    elif fault == "input-shape":
        b = b[:-16]
    elif fault == "input-rank":
        a = a.flatten()
    elif fault == "scale-a-dtype":
        sa = sa.bfloat16()
    elif fault == "scale-b-dtype":
        sb = sb.bfloat16()
    elif fault == "scale-a-shape":
        sa = sa[:-1]
    elif fault == "scale-b-shape":
        sb = sb[:-1]
    elif fault == "scale-a-rank":
        sa = sa.view(-1, 1)
    elif fault == "scale-device":
        sa = sa.cpu()
    elif fault == "out-dtype":
        options["out_dtype"] = torch.float16
        expected = ValueError
    elif fault == "out-shape":
        options["out"] = torch.empty((args.m + 1, args.n), device="cuda", dtype=torch.bfloat16)
    elif fault == "out-stride":
        options["out"] = torch.empty((args.m, args.n * 2), device="cuda", dtype=torch.bfloat16)[:, ::2]
    elif fault == "out-device":
        options["out"] = torch.empty((args.m, args.n), dtype=torch.bfloat16)
    elif fault == "out-dtype-conflict":
        options.update(out=torch.empty((args.m, args.n), device="cuda", dtype=torch.bfloat16), out_dtype=torch.float32)
    elif fault == "bias-shape":
        options["bias"] = torch.ones(args.n + 1, device="cuda", dtype=torch.bfloat16)
    elif fault == "bias-dtype":
        options["bias"] = torch.ones(args.n, device="cuda", dtype=torch.float32)
    elif fault == "unknown-option":
        options["user_kwargs"] = dict(options["user_kwargs"], unknown=True)
    with pytest.raises(expected):
        scaled_gemm(a, b, sa, sb, **options)


@pytest.mark.parametrize("hti", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("out_mode", ["allocate", "infer", "flat"])
def test_scaled_gemm_output_api(hti, out_dtype, out_mode):
    _skip_if_no_fp8()
    args = _path_args((16, 16, 128), "hti" if hti else "full", out_dtype=out_dtype)
    inputs = _exact_inputs(args, 123)
    kwargs = kernel_kwargs(args)
    saved_kwargs = dict(kwargs)
    options = dict(user_kwargs=kwargs, layout="NT")
    if out_mode == "allocate":
        options["out_dtype"] = out_dtype
    else:
        shape = (args.m, args.n) if out_mode == "infer" else (args.m * args.n,)
        options["out"] = torch.empty(shape, device="cuda", dtype=out_dtype)
    out = scaled_gemm(*inputs[:4], **options)
    assert out.shape == (args.m, args.n) and out.dtype == out_dtype and out.device == inputs[0].device
    if out_mode != "allocate":
        assert out.data_ptr() == options["out"].data_ptr()
    assert kwargs == saved_kwargs
    torch.testing.assert_close(out, _exact_reference(inputs, out_dtype), atol=0, rtol=0)


def test_scaled_gemm_default_policy():
    _skip_if_no_fp8()
    args = replace(_path_args((16, 16, 128), "full"), m=256, n=256)
    inputs = _exact_inputs(args, 123)
    out = scaled_gemm(*inputs[:4])
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, _exact_reference(inputs, out.dtype), atol=0, rtol=0)


@pytest.mark.parametrize("mma", _MMA_SHAPES, ids=["mma16", "mma32"])
@pytest.mark.parametrize("mode", ["full", "full-split", "hti", "hti-split"])
@pytest.mark.parametrize("m,n", [(1, 8), (17, 24)])
def test_scaled_gemm_small_m_and_minimum_n(mma, mode, m, n):
    args = replace(_path_args(mma, mode, has_bias=True), m=m, n=n)
    # Unit stride for length-1 scale; the singleton non-unit stride bug has
    # its own targeted regression below rather than hiding it here.
    _skip_if_no_fp8()
    inputs = list(_exact_inputs(args, 456))
    if m == 1:
        inputs[2] = inputs[2].clone(memory_format=torch.contiguous_format)
    out = scaled_gemm(*inputs[:4], bias=inputs[4], user_kwargs=kernel_kwargs(args))
    torch.testing.assert_close(out, _exact_reference(inputs, out.dtype), atol=0, rtol=0)


@pytest.mark.parametrize("hti", [False, True])
@pytest.mark.parametrize("n_tiles", [31, 32, 33], ids=["below-256", "256", "264"])
def test_scaled_gemm_block_swizzle_thresholds(hti, n_tiles):
    # 8 M tiles: 248 WGs (simple), 256/264 WGs (XCD grouped), and a partial
    # final M group when GROUP_M=3. Exact data keeps the reference inexpensive.
    args = _path_args((16, 16, 128), "hti" if hti else "full")
    args = replace(args, m=8 * args.block_m - 1, n=n_tiles * args.block_n,
                   k=256, group_m=3)
    _check_exact_path(args, repeats=1)


def test_scaled_gemm_block_swizzle_non_xcd_multiple():
    args = replace(_path_args((16, 16, 128), "full"), m=9 * 64, n=29 * 64,
                   k=256, group_m=4)
    _check_exact_path(args, repeats=1)  # 261 WGs => simple mapping


def _dispatch_param(args):
    return make_scaled_gemm_gfx950_param(
        **kernel_kwargs(args),
        in_dtype_id=SCALED_GEMM_DTYPE_FP8,
        out_dtype_id=SCALED_GEMM_DTYPE_FP32 if args.out_dtype == torch.float32 else SCALED_GEMM_DTYPE_BF16,
        has_bias=args.has_bias, a_is_transposed=args.layout[0] == "t",
        b_is_transposed=args.layout[1] == "t",
    )


@pytest.mark.parametrize("mode", _PATH_MODES)
def test_scaled_gemm_dynamic_shape_stride_and_split_dispatch(mode):
    _skip_if_no_fp8()
    args = _path_args((16, 16, 128), mode, has_bias=True)
    cached = []
    for i in range(3):
        shape_args = replace(args, m=args.m + i * 16, n=args.n + i * 16)
        if "split" in mode:
            shape_args = replace(shape_args, split_k=(2, 4, 2)[i],
                                 k=4 * args.block_k * (2, 4, 2)[i])
        else:
            shape_args = replace(shape_args, k=args.block_k * (4 + i * 2))
        _check_exact_path(shape_args, repeats=1)
        cached.append(scaled_module.scaled_gemm_gfx950._compiled_cache[
            _dispatch_param(shape_args).__cache_signature__()
        ])
    assert cached[0] is cached[1] is cached[2]


@pytest.mark.parametrize("mma", _MMA_SHAPES, ids=["mma16", "mma32"])
@pytest.mark.parametrize("mode", _PATH_MODES)
def test_scaled_gemm_stream_graph_replay(mma, mode):
    _skip_if_no_fp8()
    from kernels.gemm_a16w16_gfx950 import get_split_k_buffers
    args = _path_args(mma, mode, has_bias=True, out_dtype=torch.float32)
    kwargs = kernel_kwargs(args)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    slots = []  # Retain every captured tensor until all replays have finished.
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            inputs = _exact_inputs(args, 101 + i)
            out = torch.empty((args.m, args.n), device="cuda", dtype=args.out_dtype)
            ref = _exact_reference(inputs, args.out_dtype)
            scaled_gemm(*inputs[:4], bias=inputs[4], out=out, user_kwargs=kwargs, stream=stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                scaled_gemm(*inputs[:4], bias=inputs[4], out=out, user_kwargs=kwargs, stream=stream)
        slots.append((inputs, out, ref, graph))
    for _ in range(5):
        for stream, (_, out, _, graph) in zip(streams, slots):
            with torch.cuda.stream(stream):
                out.fill_(float("nan"))
                graph.replay()
    for stream, (inputs, out, ref, _) in zip(streams, slots):
        stream.synchronize()
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
        if args.split_k > 1:
            for buf in get_split_k_buffers(stream, inputs[0].device):
                assert torch.count_nonzero(buf).item() == 0
    if args.split_k > 1:
        buffers = [get_split_k_buffers(s, slots[0][0][0].device) for s in streams]
        assert buffers[0][0].data_ptr() != buffers[1][0].data_ptr()
        assert buffers[0][1].data_ptr() != buffers[1][1].data_ptr()


@pytest.mark.parametrize("hti", [False, True])
def test_scaled_gemm_nonsplit_does_not_allocate_sync_buffers(monkeypatch, hti):
    _skip_if_no_fp8()
    def unexpected_sync_allocation(*args, **kwargs):
        pytest.fail("non-split GEMM must not allocate split-K synchronization buffers")
    monkeypatch.setattr(scaled_module, "get_split_k_buffers", unexpected_sync_allocation)
    _check_exact_path(_path_args((16, 16, 128), "hti" if hti else "full"), repeats=1)


@pytest.mark.parametrize("mode", ["full-split", "split-slice2", "hti-split"])
def test_scaled_gemm_repeated_bias_initialization(mode):
    args = _path_args((16, 16, 128), mode, out_dtype=torch.float32, has_bias=True)
    for _ in range(20):
        _check_exact_path(args, repeats=1)


@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="length-1 stride-2 scale passes Torch is_contiguous but FlyDSL requires stride 1",
)
def test_scaled_gemm_singleton_nonunit_scale_stride():
    _skip_if_no_fp8()
    args = replace(_path_args((16, 16, 128), "full"), m=1, n=8)
    inputs = _exact_inputs(args, 789)
    assert inputs[2].stride() == (2,) and inputs[2].is_contiguous()
    try:
        out = scaled_gemm(*inputs[:4], user_kwargs=kernel_kwargs(args))
    except RuntimeError as exc:
        # Only the known failure is expected, not unrelated runtime errors.
        if "Leading dimension must have stride 1" not in str(exc):
            raise AssertionError("unexpected runtime failure") from exc
        raise
    torch.testing.assert_close(out, _exact_reference(inputs, out.dtype), atol=0, rtol=0)


@pytest.mark.parametrize("m,n", [(0, 64), (64, 0), (-1, 64)])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="shape validator does not reject nonpositive M/N")
def test_scaled_gemm_shape_validator_rejects_nonpositive_mn(gfx950_params, m, n):
    # Host only: never launch a zero/negative grid while checking this defect.
    assert make_scaled_gemm_param_and_validate(m, n, 512, _full_kwargs()) is None


def test_scaled_gemm_rejects_unaligned_m_for_transposed_a():
    _skip_if_no_fp8()
    args = replace(_path_args((16, 16, 128), "full", layout="tn"), m=65)
    inputs = _exact_inputs(args, 123)
    with pytest.raises(ValueError, match="DMA requirements"):
        scaled_gemm(*inputs[:4], user_kwargs=kernel_kwargs(args), layout=args.layout)


@pytest.mark.parametrize("override,match", [
    ({"block_k": 130}, "async load vector size"),
    ({"block_k": 192}, "ldg thread layout"),
    ({"block_m": 65}, "A async load tile"),
    ({"block_n": 48}, "B async load tile"),
    ({"block_m": 96, "m_waves": 4, "n_waves": 1}, "divisible by m_waves"),
    ({"block_n": 96, "m_waves": 1, "n_waves": 4}, "divisible by n_waves"),
    ({"block_k": 128, "k_waves": 2}, "divisible by k_waves"),
])
def test_scaled_gemm_dma_and_mma_geometry_rejections(gfx950_params, override, match):
    kwargs = _full_kwargs(**override)
    with pytest.raises(ValueError, match=match):
        make_scaled_gemm_gfx950_param(**kwargs)
    assert make_scaled_gemm_param_and_validate(128, 128, 1024, kwargs) is None


@pytest.mark.parametrize("override", [
    {"block_n": 65},
    {"use_half_tile_interleaved": True, "block_m": 96, "block_n": 128, "n_waves": 4},
    {"use_half_tile_interleaved": True, "block_m": 128, "block_n": 96},
])
def test_scaled_gemm_vector_geometry_rejections(gfx950_params, override):
    with pytest.raises(AssertionError):
        make_scaled_gemm_gfx950_param(**_full_kwargs(**override))


@pytest.mark.parametrize("mode,out_dtype", [
    ("full", torch.float32), ("full-split", torch.bfloat16),
    ("full-split", torch.float32), ("hti", torch.float32), ("hti-split", torch.float32),
])
def test_scaled_gemm_four_element_output_vector(mode, out_dtype):
    args = replace(_path_args((16, 16, 128), mode, out_dtype=out_dtype, has_bias=True), m=17, n=4)
    _check_exact_path(args, repeats=1)


@pytest.mark.parametrize("mode", _PATH_MODES)
@pytest.mark.parametrize("mma", _MMA_SHAPES, ids=["mma16", "mma32"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_scaled_gemm_dense_random_paths(mode, mma, out_dtype):
    _skip_if_no_fp8()
    args = _path_args(mma, mode, has_bias=True, out_dtype=out_dtype)
    generator = torch.Generator(device="cuda").manual_seed(314159)
    a = torch.randn((args.m, args.k), device="cuda", generator=generator).clamp(-2, 2).to(torch.float8_e4m3fn)
    b = torch.randn((args.n, args.k), device="cuda", generator=generator).clamp(-2, 2).to(a.dtype).t()
    sa = torch.rand(args.m, device="cuda", generator=generator) * 0.125 + 0.0625
    sb = torch.rand(args.n, device="cuda", generator=generator) * 0.125 + 0.0625
    bias = torch.randn(args.n, device="cuda", generator=generator).to(out_dtype) * 0.25
    out = scaled_gemm(a, b, sa, sb, bias=bias, user_kwargs=kernel_kwargs(args), out_dtype=out_dtype)
    ref = ((a.float() @ b.float()) * sa[:, None] * sb[None, :] + bias.float()).to(out_dtype)
    # BF16 C-shuffle and BF16 split/slice partials require rounding tolerance
    # even when the final output is FP32. Much tighter than the legacy rtol=.5.
    torch.testing.assert_close(out.float(), ref.float(), atol=0.025, rtol=0.02)


@pytest.mark.parametrize("k,overrides", [
    (3328, {"split_k": 9}),  # partitions 384,...,384,256
    (128, {}),             # one full-tile drain
    (256, {"use_half_tile_interleaved": True}),
])
def test_scaled_gemm_valid_partition_boundaries(gfx950_params, k, overrides):
    kwargs = _full_kwargs(**overrides)
    assert_no_k_tail(k, kwargs)
    assert make_scaled_gemm_param_and_validate(128, 128, k, kwargs) is not None


def test_scaled_gemm_ptpc_scale_views_and_rotary_count(monkeypatch):
    # Benchmark helpers need not allocate an 8 GiB rotary pool in a unit test.
    sa = torch.arange(12, dtype=torch.float32)[::2]
    sb = torch.arange(16, dtype=torch.float32)[::2]
    va, vb = _ptpc_scale_views(sa, sb)
    assert va.shape == (6, 1) and vb.shape == (1, 8)
    assert va.is_contiguous() and vb.is_contiguous()
    torch.testing.assert_close(va[:, 0], sa)
    torch.testing.assert_close(vb[0], sb)
    inp, out = (sa, None), (sb,)
    assert tensor_nbytes(inp) == sa.numel() * 4
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "ROTARY_INPUTS_TARGET_BYTES", 1)
    assert get_rotary_inputs(inp, out) == 1
    slot_bytes = 2 * (tensor_nbytes(inp) + tensor_nbytes(out))
    monkeypatch.setattr(module, "ROTARY_INPUTS_TARGET_BYTES", 3 * slot_bytes)
    assert get_rotary_inputs(inp, out) == 3


@pytest.mark.parametrize("comparators", ["unavailable", "torch-only", "torch-and-triton"])
def test_scaled_gemm_bench_helper_smoke(monkeypatch, comparators):
    _skip_if_no_fp8()
    module = sys.modules[__name__]
    calls = dict(torch=0, triton=0)
    def reference(a, b, sa, sb, c):
        calls["torch"] += 1
        if comparators == "unavailable":
            raise RuntimeError("deliberately unavailable comparator")
        c.copy_(((a.float() @ b.float()) * sa * sb).to(c.dtype))
    def triton_factory():
        if comparators != "torch-and-triton":
            raise RuntimeError("deliberately unavailable autotuner")
        # Test comparator scheduling, not the third-party Triton compiler.
        def comparator(a, b, sa, sb, c):
            calls["triton"] += 1
            c.copy_(((a.float() @ b.float()) * sa * sb).to(c.dtype))
        return comparator
    monkeypatch.setattr(module, "scaled_mm_func", reference)
    monkeypatch.setattr(module, "make_triton_maxautotune_func", triton_factory)
    monkeypatch.setattr(module, "ROTARY_INPUTS_TARGET_BYTES", 1)
    original_timer = _cuda_time_ms
    monkeypatch.setattr(module, "_cuda_time_ms", lambda run, slots: original_timer(run, slots, niters=2))
    benchmark(_path_args((16, 16, 128), "hti"), warmup=2, niters=4)
    assert calls["torch"] >= 1
    assert (calls["triton"] > 0) == (comparators == "torch-and-triton")




@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("tile_m,tile_n,n_waves,split_k", [
    (128, 128, 2, 1), (256, 256, 4, 1), (128, 128, 2, 4),
])
def test_scaled_gemm_rejects_hti_block64_with_mma128(gfx950_params, layout, tile_m, tile_n, n_waves, split_k):
    # These used to be accuracy cases, but a 64-wide K tile cannot hold a
    # 16x16x128 MMA. The legal block_k=64 path uses 32x32x64 (matrix above).
    kwargs = _full_kwargs(
        block_m=tile_m, block_n=tile_n, block_k=64,
        m_waves=2, n_waves=n_waves, split_k=split_k,
        use_half_tile_interleaved=True,
        a_is_transposed=layout[0] == "t", b_is_transposed=layout[1] == "t",
    )
    with pytest.raises(ValueError, match="divisible by k_waves"):
        make_scaled_gemm_gfx950_param(**kwargs)
    assert make_scaled_gemm_param_and_validate(256, 256, 1024, kwargs) is None


@pytest.mark.parametrize("mode", ["full", "full-split", "hti", "hti-split"])
def test_scaled_gemm_uses_current_stream_when_stream_omitted(mode):
    _skip_if_no_fp8()
    args = _path_args((16, 16, 128), mode, has_bias=True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        inputs = _exact_inputs(args, 888)
        out = scaled_gemm(*inputs[:4], bias=inputs[4], user_kwargs=kernel_kwargs(args))
        ref = _exact_reference(inputs, args.out_dtype)
        finished = torch.cuda.Event()
        finished.record()
    torch.cuda.current_stream().wait_event(finished)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["full-split", "split-slice2", "hti-split"])
def test_scaled_gemm_graph_observes_changed_inputs(mode):
    _skip_if_no_fp8()
    args = _path_args((16, 16, 128), mode, has_bias=True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        inputs = _exact_inputs(args, 2468)
        a, b, sa, sb, bias = inputs
        out = torch.empty((args.m, args.n), device="cuda", dtype=args.out_dtype)
        scaled_gemm(a, b, sa, sb, bias=bias, out=out, user_kwargs=kernel_kwargs(args))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            scaled_gemm(a, b, sa, sb, bias=bias, out=out, user_kwargs=kernel_kwargs(args))
        for i in range(3):
            sa.fill_((i - 1) * 0.25)
            bias.fill_(i * 0.25)
            ref = _exact_reference(inputs, args.out_dtype)
            out.fill_(float("nan"))
            graph.replay()
            stream.synchronize()
            torch.testing.assert_close(out, ref, atol=0, rtol=0)

def _cuda_time_ms(run, rotary_inputs, niters=50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(niters):
        run(i % rotary_inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / niters


def benchmark(args: _TestArgs, warmup: int = 500, niters: int = 600):
    _skip_if_no_fp8()
    assert args.layout == "nt", "torch._scaled_mm comparison is NT-only"
    assert not args.has_bias, "benchmark compares the no-bias scaled-mm path"
    kwargs = kernel_kwargs(args)
    sample_inputs = create_inputs(args)
    sample_outputs = create_outputs(args)
    rotary_inputs = get_rotary_inputs(sample_inputs, sample_outputs)
    inputs = [sample_inputs] + [create_inputs(args) for _ in range(rotary_inputs - 1)]
    outputs = [sample_outputs] + [
        create_outputs(args) for _ in range(rotary_inputs - 1)
    ]
    sm_scales = [_ptpc_scale_views(inp[2], inp[3]) for inp in inputs]
    flops = 2.0 * args.m * args.n * args.k
    print(
        f"{args}\nrotary_inputs:{rotary_inputs}, target_bytes:{ROTARY_INPUTS_TARGET_BYTES}, "
        f"warmup:{warmup}, niters:{niters}, flops:{flops:.3e}"
    )

    def run_flydsl(idx):
        func(*inputs[idx], *outputs[idx], kwargs, args.layout)

    has_scaled_mm = hasattr(torch, "_scaled_mm")
    if has_scaled_mm:
        try:
            a, b, _sa, _sb, _bias = inputs[0]
            scale_a_2d, scale_b_2d = sm_scales[0]
            (c,) = outputs[0]
            scaled_mm_func(a, b, scale_a_2d, scale_b_2d, c)
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"torch._scaled_mm unavailable: {exc}")
            has_scaled_mm = False

    def run_scaled_mm(idx):
        a, b, _sa, _sb, _bias = inputs[idx]
        scale_a_2d, scale_b_2d = sm_scales[idx]
        (c,) = outputs[idx]
        scaled_mm_func(a, b, scale_a_2d, scale_b_2d, c)

    has_triton = has_scaled_mm
    triton_fn = None
    if has_triton:
        try:
            triton_fn = make_triton_maxautotune_func()
            a, b, _sa, _sb, _bias = inputs[0]
            scale_a_2d, scale_b_2d = sm_scales[0]
            (c,) = outputs[0]
            triton_fn(a, b, scale_a_2d, scale_b_2d, c)
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"triton max-autotune unavailable: {exc}")
            has_triton = False

    def run_triton(idx):
        a, b, _sa, _sb, _bias = inputs[idx]
        scale_a_2d, scale_b_2d = sm_scales[idx]
        (c,) = outputs[idx]
        triton_fn(a, b, scale_a_2d, scale_b_2d, c)

    runners = [("flydsl", run_flydsl)]
    if has_scaled_mm:
        runners.append(("scaled_mm", run_scaled_mm))
    if has_triton:
        runners.append(("triton", run_triton))

    def run_once(i, idx):
        n = len(runners)
        for off in range(n):
            runners[(i + off) % n][1](idx)

    print("===================== [INTERLEAVED] =====================")
    for i in range(warmup):
        run_once(i, i % rotary_inputs)
        torch.cuda.synchronize()

    print("===================== [CUDA EVENTS] =====================")
    for name, run in runners:
        ms = _cuda_time_ms(run, rotary_inputs)
        print(f"{name}: {ms:.3f} ms, {flops / ms / 1e9:.2f} TFLOPS")

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(warmup, niters):
            run_once(i, i % rotary_inputs)
            torch.cuda.synchronize()
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1, max_name_column_width=140
        )
    )


@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves",
    [
        (2048, 2048, 2048, 128, 128, 128, 2, 4, 4),
        (4096, 4096, 4096, 256, 256, 128, 2, 2, 4),
    ],
)
def test_scaled_gemm_benchmark_nt(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
):
    benchmark(
        _TestArgs(
            m=m,
            n=n,
            k=k,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            k_waves=1,
            group_m=0,
            has_bias=False,
            layout="nt",
        )
    )


def test_scaled_gemm_benchmark_8192_nt():
    benchmark(
        _TestArgs(
            m=8192,
            n=8192,
            k=8192,
            block_m=256,
            block_n=256,
            block_k=128,
            stages=2,
            m_waves=2,
            n_waves=4,
            k_waves=1,
            group_m=0,
            has_bias=False,
            layout="nt",
            use_half_tile_interleaved=True,
        )
    )
