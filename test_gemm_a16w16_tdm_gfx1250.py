"""Full-tile PeakGemm TDM port: FP32 reference, pipeline boundaries and replay."""

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

import pytest
import torch

from kernels.gemm_a16w16_tdm_gfx1250 import (
    GEMM_DTYPE_BF16,
    validate_gemm_gfx1250_config,
    gemm_a16w16_gfx1250,
)


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
    actual = gemm_a16w16_gfx1250(a, b, out, bias, **kwargs)
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
    gemm_a16w16_gfx1250(a, b, out, bias, **kwargs)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        gemm_a16w16_gfx1250(a, b, out, bias, stream=stream, **kwargs)
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
    out = gemm_a16w16_gfx1250(a, b, **config)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "changed,message",
    [
        ({"stages": 1}, "stages"),
        ({"stages": 34}, "stages"),
        ({"k": 32, "stages": 3}, "prefetch"),
        ({"k": 40}, "K must"),
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
        "dtype_id": GEMM_DTYPE_BF16,
        "mma_m": 16,
        "mma_n": 16,
        "mma_k": 32,
    }
    config.update(changed)
    with pytest.raises(ValueError, match=message):
        validate_gemm_gfx1250_config(**config)


def test_input_contract():
    require_gfx1250()
    a, b = inputs(64, 128, 64, torch.float16)
    config = {"block_m": 64, "block_n": 128, "block_k": 32, "m_waves": 2, "n_waves": 2}
    with pytest.raises(ValueError, match="NT layout"):
        gemm_a16w16_gfx1250(a, b.contiguous(), **config)
    with pytest.raises(ValueError, match="out must"):
        gemm_a16w16_gfx1250(
            a,
            b,
            out=torch.empty((64, 128), device="cuda", dtype=torch.float32),
            **config,
        )
    with pytest.raises(ValueError, match="bias must"):
        gemm_a16w16_gfx1250(
            a, b, bias=torch.empty((127,), device="cuda", dtype=a.dtype), **config
        )
    square_a, square_b = inputs(64, 64, 64, torch.float16)
    with pytest.raises(ValueError, match="overlap"):
        gemm_a16w16_gfx1250(
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
@pytest.mark.parametrize("layout", ["nt", "nn"])
def test_padded_leading_strides(dtype, layout):
    require_gfx1250()
    m, n, k = 64, 128, 64
    torch.manual_seed(1102)
    a = (torch.randint(-4, 5, (m, 80), device="cuda").float() / 16).to(dtype)[:, :k]
    if layout == "nt":
        b = (torch.randint(-4, 5, (n, 96), device="cuda").float() / 16).to(dtype)[:, :k].t()
    else:
        b = (torch.randint(-4, 5, (k, 160), device="cuda").float() / 16).to(dtype)[:, :n]
    storage = torch.full((m, 192), 83, device="cuda", dtype=dtype)
    out = storage[:, :n]
    expected = reference(a, b)
    torch.cuda.synchronize()
    actual = gemm_a16w16_gfx1250(
        a,
        b,
        out,
        layout=layout,
        block_m=64,
        block_n=128,
        block_k=32,
        m_waves=2,
        n_waves=2,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (storage[:, n:] == 83).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["nt", "nn"])
def test_partial_mn_tile(dtype, layout):
    require_gfx1250()
    m, n, k = 80, 200, 64
    torch.manual_seed(1102)
    a = (torch.randint(-4, 5, (m, k), device="cuda").float() / 16).to(dtype)
    if layout == "nt":
        b = (torch.randint(-4, 5, (n, k), device="cuda").float() / 16).to(dtype).t()
    else:
        b = (torch.randint(-4, 5, (k, n), device="cuda").float() / 16).to(dtype)
    expected = reference(a, b)
    torch.cuda.synchronize()
    actual = gemm_a16w16_gfx1250(
        a,
        b,
        layout=layout,
        block_m=64,
        block_n=128,
        block_k=32,
        m_waves=2,
        n_waves=2,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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
    actual = gemm_a16w16_gfx1250(
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
    gemm_a16w16_gfx1250(
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
    actual = gemm_a16w16_gfx1250(
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


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias",
    [
        (8, 4096, 4096, 16, 16, 128, 8, 1, 1, 4, True),
        (16, 4096, 4096, 16, 16, 128, 8, 1, 1, 4, True),
        (32, 4096, 4096, 16, 16, 64, 8, 1, 1, 0, True),
        (64, 4096, 4096, 32, 32, 64, 8, 2, 2, 4, True),
        (128, 4096, 4096, 64, 32, 128, 4, 4, 2, 4, True),
        (256, 4096, 4096, 64, 64, 64, 7, 4, 2, 4, True),
        (512, 4096, 4096, 64, 128, 64, 6, 2, 4, 4, True),
        (1024, 4096, 4096, 128, 128, 64, 4, 2, 4, 4, True),
        (2048, 4096, 4096, 128, 256, 64, 3, 4, 4, 4, True),
        (1024, 1024, 1024, 64, 64, 64, 4, 2, 4, 0, True),
        (2048, 2048, 2048, 128, 128, 64, 3, 4, 2, 0, True),
        (4096, 4096, 4096, 256, 256, 64, 2, 2, 4, 4, True),
        (4096, 4096, 8192, 256, 256, 64, 2, 2, 4, 4, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, True),
        (16384, 16384, 16384, 256, 256, 64, 2, 2, 4, 4, True),
        (8, 7168, 2048, 16, 16, 128, 8, 1, 1, 4, True),
        (32, 384, 7168, 16, 16, 128, 8, 1, 1, 0, True),
        (32, 14336, 4096, 32, 64, 64, 8, 2, 2, 0, True),
        (16, 28672, 4096, 16, 64, 128, 3, 1, 4, 4, True),
        (4096, 256, 4096, 64, 64, 64, 6, 4, 2, 4, True),
    ],
)
def test_gemm_a16w16_benchmark_smoke(
    layout,
    dtype,
    m,
    n,
    k,
    block_m,
    block_n,
    block_k,
    stages,
    m_waves,
    n_waves,
    group_m,
    has_bias,
):
    require_gfx1250()
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    torch.manual_seed(1102)
    a = (torch.randn((m, k), device="cuda") / 8).to(torch_dtype)
    if layout == "nt":
        b = (torch.randn((n, k), device="cuda") / 8).to(torch_dtype).t()
    else:
        b = (torch.randn((k, n), device="cuda") / 8).to(torch_dtype)
    bias = (torch.randn((n,), device="cuda") / 8).to(torch_dtype) if has_bias else None
    config = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "stages": stages,
        "m_waves": m_waves,
        "n_waves": n_waves,
        "group_m": group_m,
        "layout": layout,
    }
    out = torch.empty((m, n), device="cuda", dtype=torch_dtype)
    expected = reference(a, b, bias)
    torch.cuda.synchronize()
    gemm_a16w16_gfx1250(a, b, out, bias, **config)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)

    for _ in range(3):
        gemm_a16w16_gfx1250(a, b, out, bias, **config)
    torch.cuda.synchronize()
    nodes = 8
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            gemm_a16w16_gfx1250(a, b, out, bias, **config)
    samples = benchmark_graph(graph, nodes, "legacy")
    graph.reset()
    ms = statistics.median(samples)
    print(
        f"layout={layout} shape=({m},{n},{k}) "
        f"tile=({block_m},{block_n},{block_k}) stages={stages} "
        f"waves=({m_waves},{n_waves}) group_m={group_m} "
        f"ms={ms:.6f} tflops={2 * m * n * k / ms / 1e9:.2f}"
    )


def benchmark_graph(graph, nodes, mode):
    """Return all event samples in ms/GEMM; never select the fastest sample."""
    if mode == "burst":
        # Explicit idle interval: this measures a burst, NOT sustained throughput.
        torch.cuda.synchronize()
        time.sleep(0.5)
        graph.replay()
        torch.cuda.synchronize()
        rounds, replays = 5, 1
    elif mode == "sustained":
        begin = time.monotonic()
        while time.monotonic() - begin < 0.5:
            graph.replay()
            torch.cuda.synchronize()
        rounds, replays = 21, 4
    else:
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        rounds, replays = 9, 1
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / (nodes * replays))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnk", nargs=3, type=int, default=[8192, 8192, 8192])
    parser.add_argument("--tile", nargs=3, type=int, default=[256, 256, 128])
    parser.add_argument("--waves", nargs=2, type=int, default=[4, 4])
    parser.add_argument("--stages", nargs="+", type=int, default=[2])
    parser.add_argument(
        "--dtype", nargs="+", choices=["fp16", "bf16"], default=["fp16", "bf16"]
    )
    parser.add_argument(
        "--mode",
        choices=["legacy", "burst", "sustained", "both"],
        default="legacy",
        help="legacy keeps the original timing; burst includes a 0.5s idle interval; "
        "sustained includes >=0.5s continuous warmup; both reports them separately",
    )
    parser.add_argument(
        "--json", type=Path, help="save configuration and every timing sample"
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
        gemm_a16w16_gfx1250(a, b, out, **config)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
        for _ in range(10):
            gemm_a16w16_gfx1250(a, b, out, **config)
        torch.cuda.synchronize()
        # Keep the original default protocol. Explicit phase modes use eight
        # nodes, matching the paired BF16 power-state study in docs/.
        nodes = 20 if args.mode == "legacy" else 8
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(nodes):
                gemm_a16w16_gfx1250(a, b, out, **config)
        modes = ["burst", "sustained"] if args.mode == "both" else [args.mode]
        repeats = 1 if args.mode == "legacy" else 3
        for repeat in range(repeats):
            for mode in modes:
                samples = benchmark_graph(graph, nodes, mode)
                ms = statistics.median(samples)
                rows.append(
                    {
                        "dtype": name,
                        "shape": [m, n, k],
                        "config": config,
                        "mode": mode,
                        "repeat": repeat,
                        "graph_nodes": nodes,
                        "samples_ms": samples,
                        "median_ms": ms,
                        "tflops": 2 * m * n * k / ms / 1e9,
                    }
                )
                # Clear C and replay after timing; a stale result cannot pass.
                out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
        graph.reset()
    print("| Dtype | Stages | Mode | Repeat | ms | TFLOPS |")
    print("|---|---:|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['dtype']} | {row['config']['stages']} | {row['mode']} | "
            f"{row['repeat']} | {row['median_ms']:.6f} | {row['tflops']:.2f} |"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "hip": torch.version.hip,
                    "device": str(torch.cuda.get_device_properties(0)),
                    "protocol": args.mode,
                    "results": rows,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
