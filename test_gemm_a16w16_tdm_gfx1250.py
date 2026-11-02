"""Full-tile PeakGemm TDM port: FP32 reference, pipeline boundaries and replay."""

import argparse
import itertools

import pytest
import torch

from kernels.gemm_a16w16_tdm_gfx1250 import _check_config, gemm_a16w16


def require_gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
        pytest.skip("requires gfx1250")


def reference(a, b, bias=None):
    # PeakGemm rounds the accumulator before bias, not after a fused FP32 add.
    out = (a.float() @ b.float()).to(a.dtype)
    return out if bias is None else (out.float() + bias.float()).to(a.dtype)


def inputs(m, n, k, dtype):
    torch.manual_seed(1102)
    a = (torch.randint(-4, 5, (m, k), device="cuda").float() / 16).to(dtype)
    b = (torch.randint(-4, 5, (n, k), device="cuda").float() / 16).to(dtype).t()
    return a, b


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stages", [2, 3, 4, 5])
@pytest.mark.parametrize("extra_tiles", [0, 1, 2, 7])
@pytest.mark.parametrize("use_bias", [False, True])
def test_pipeline(dtype, stages, extra_tiles, use_bias):
    require_gfx1250()
    tiles = stages - 1 + extra_tiles
    a, b = inputs(128, 384, tiles * 32, dtype)
    bias = (
        (torch.arange(384, device="cuda").remainder(5) / 16).to(dtype)
        if use_bias
        else None
    )
    storage = torch.full((128 * 384 + 32,), 83, device="cuda", dtype=dtype)
    out = storage[16:-16].view(128, 384)
    kwargs = {
        "block_m": 64,
        "block_n": 128,
        "block_k": 32,
        "m_waves": 2,
        "n_waves": 2,
        "stages": stages,
    }
    expected = reference(a, b, bias)
    torch.cuda.synchronize()
    actual = gemm_a16w16(a, b, out, bias, **kwargs)
    torch.cuda.synchronize()
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (storage[:16] == 83).all() and (storage[-16:] == 83).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stages", [2, 3, 4])
def test_graph_changed_input(dtype, stages):
    require_gfx1250()
    a, b = inputs(192, 256, (2 * stages + 1) * 64, dtype)
    bias = torch.full((256,), 0.125, device="cuda", dtype=dtype)
    out = torch.empty((192, 256), device="cuda", dtype=dtype)
    kwargs = {
        "block_m": 64,
        "block_n": 128,
        "block_k": 64,
        "stages": stages,
        "m_waves": 2,
        "n_waves": 2,
        "group_m": 2,
    }
    gemm_a16w16(a, b, out, bias, **kwargs)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        gemm_a16w16(a, b, out, bias, stream=stream, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        a.neg_()
        bias.neg_()
        expected = reference(a, b, bias)
        torch.cuda.synchronize()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,config",
    [
        (
            (256, 256, 128),
            {
                "block_m": 256,
                "block_n": 256,
                "block_k": 128,
                "stages": 2,
                "m_waves": 4,
                "n_waves": 4,
            },
        ),
        (
            (256, 256, 192),
            {
                "block_m": 256,
                "block_n": 256,
                "block_k": 64,
                "stages": 3,
                "m_waves": 4,
                "n_waves": 4,
            },
        ),
        (
            (256, 256, 448),
            {
                "block_m": 128,
                "block_n": 256,
                "block_k": 64,
                "stages": 5,
                "m_waves": 2,
                "n_waves": 4,
            },
        ),
        (
            (128, 192, 1088),
            {
                "block_m": 64,
                "block_n": 64,
                "block_k": 64,
                "stages": 8,
                "m_waves": 2,
                "n_waves": 2,
            },
        ),
    ],
)
def test_tile_variants(dtype, shape, config):
    require_gfx1250()
    m, n, k = shape
    a, b = inputs(m, n, k, dtype)
    expected = reference(a, b)
    torch.cuda.synchronize()
    out = gemm_a16w16(a, b, **config)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "changed,message",
    [
        ({"stages": 1}, "stages"),
        ({"stages": 34}, "stages"),
        ({"k": 32, "stages": 3}, "prefetch"),
        ({"m": 65}, "aligned"),
        ({"block_k": 48}, "power of two"),
        (
            {
                "block_m": 256,
                "block_n": 256,
                "block_k": 128,
                "stages": 3,
                "m_waves": 4,
                "n_waves": 4,
            },
            "LDS",
        ),
        ({"m_waves": 0}, "positive"),
        ({"group_m": -1}, "nonnegative"),
    ],
)
def test_config_rejections(changed, message):
    config = {
        "m": 128,
        "n": 256,
        "k": 256,
        "block_m": 64,
        "block_n": 128,
        "block_k": 32,
        "stages": 2,
        "m_waves": 2,
        "n_waves": 2,
        "group_m": 0,
    }
    config.update(changed)
    with pytest.raises(ValueError, match=message):
        _check_config(**config)


def test_input_contract():
    require_gfx1250()
    a, b = inputs(64, 128, 64, torch.float16)
    config = {"block_m": 64, "block_n": 128, "block_k": 32, "m_waves": 2, "n_waves": 2}
    with pytest.raises(ValueError, match="NT layout"):
        gemm_a16w16(a, b.contiguous(), **config)
    with pytest.raises(ValueError, match="out must"):
        gemm_a16w16(
            a,
            b,
            out=torch.empty((64, 128), device="cuda", dtype=torch.float32),
            **config,
        )
    with pytest.raises(ValueError, match="bias must"):
        gemm_a16w16(
            a, b, bias=torch.empty((127,), device="cuda", dtype=a.dtype), **config
        )
    square_a, square_b = inputs(64, 64, 64, torch.float16)
    with pytest.raises(ValueError, match="overlap"):
        gemm_a16w16(
            square_a,
            square_b,
            out=square_a,
            block_m=64,
            block_n=64,
            block_k=32,
            m_waves=2,
            n_waves=2,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stages", [2, 3, 4])
@pytest.mark.parametrize("shape", [(512, 768, 1024), (2048, 4096, 2048)])
def test_random_matrices(dtype, stages, shape):
    require_gfx1250()
    torch.manual_seed(1102)
    m, n, k = shape
    a = (torch.randn((m, k), device="cuda") / 8).to(dtype)
    b = (torch.randn((n, k), device="cuda") / 8).to(dtype).t()
    bias = (torch.randn((n,), device="cuda") / 8).to(dtype)
    expected = reference(a, b, bias)
    torch.cuda.synchronize()
    actual = gemm_a16w16(
        a,
        b,
        bias=bias,
        block_m=128,
        block_n=256,
        block_k=64,
        stages=stages,
        m_waves=2,
        n_waves=4,
        group_m=3,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("zero_input", [False, True])
def test_explicit_stream(dtype, zero_input):
    require_gfx1250()
    a, b = inputs(192, 384, 320, dtype)
    if zero_input:
        a.zero_()
    bias = (torch.arange(384, device="cuda").remainder(7) / 16).to(dtype)
    expected = reference(a, b, bias)
    out = torch.full((192, 384), float("nan"), device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Pass an explicit stream without changing PyTorch's current stream.
    gemm_a16w16(
        a,
        b,
        out,
        bias,
        block_m=64,
        block_n=128,
        block_k=64,
        stages=4,
        m_waves=2,
        n_waves=2,
        group_m=2,
        stream=stream,
    )
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bias_rounding_order(dtype):
    require_gfx1250()
    # Rounding 1 + half an output ULP before bias gives 1. Adding bias in
    # FP32 first would instead produce 1 + one output ULP.
    half_ulp = 2 ** (-11 if dtype == torch.float16 else -8)
    a = torch.zeros((64, 64), device="cuda", dtype=dtype)
    weights = torch.zeros((128, 64), device="cuda", dtype=dtype)
    a[:, 0], a[:, 1] = 1, half_ulp
    weights[:, :2] = 1
    b = weights.t()
    bias = torch.full((128,), half_ulp, device="cuda", dtype=dtype)
    expected = reference(a, b, bias)
    fused = (a.float() @ b.float() + bias.float()).to(dtype)
    assert (expected != fused).all()
    torch.cuda.synchronize()
    actual = gemm_a16w16(
        a,
        b,
        bias=bias,
        block_m=64,
        block_n=128,
        block_k=32,
        stages=3,
        m_waves=2,
        n_waves=2,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnk", nargs=3, type=int, default=[8192, 8192, 8192])
    parser.add_argument("--tile", nargs=3, type=int, default=[256, 256, 128])
    parser.add_argument("--waves", nargs=2, type=int, default=[4, 4])
    parser.add_argument("--stages", nargs="+", type=int, default=[2])
    parser.add_argument(
        "--dtype", nargs="+", choices=["fp16", "bf16"], default=["fp16", "bf16"]
    )
    args = parser.parse_args()
    require_gfx1250()
    m, n, k = args.mnk
    rows = []
    for name, stages in itertools.product(args.dtype, args.stages):
        dtype = torch.float16 if name == "fp16" else torch.bfloat16
        a, b = inputs(m, n, k, dtype)
        out = torch.empty((m, n), dtype=dtype, device="cuda")
        config = dict(zip(("block_m", "block_n", "block_k"), args.tile))
        config.update(m_waves=args.waves[0], n_waves=args.waves[1], stages=stages)
        expected = reference(a, b)
        torch.cuda.synchronize()
        gemm_a16w16(a, b, out, **config)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
        for _ in range(10):
            gemm_a16w16(a, b, out, **config)
        torch.cuda.synchronize()
        # Capture repeated launches so the timings measure GPU work, not Python.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                gemm_a16w16(a, b, out, **config)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(9):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / 20)
        ms = sorted(times)[4]
        rows.append((name, stages, ms, 2 * m * n * k / ms / 1e9))
    print("| Dtype | Stages | ms | TFLOPS |")
    print("|---|---:|---:|---:|")
    for name, stages, ms, tflops in rows:
        print(f"| {name} | {stages} | {ms:.6f} | {tflops:.2f} |")


if __name__ == "__main__":
    main()
