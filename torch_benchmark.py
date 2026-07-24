import argparse
import json
import statistics
from pathlib import Path
from unittest import mock

import torch
import torch._dynamo
import torch._inductor.config as inductor_config


SHAPES = [
    (8, 4096, 4096),
    (16, 4096, 4096),
    (32, 4096, 4096),
    (64, 4096, 4096),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
    (2048, 4096, 4096),
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    (8192, 8192, 8192),
    (8160, 8160, 8160),
    (32, 14336, 4096),
    (16, 28672, 4096),
    (4096, 256, 4096),
]


BACKEND_PATCHES = {
    "aten": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN",
        "max_autotune_gemm_search_space": "DEFAULT",
    },
    "triton": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "TRITON",
        "max_autotune_gemm_search_space": "EXHAUSTIVE",
    },
    "flydsl": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "FLYDSL",
        "max_autotune_gemm_search_space": "EXHAUSTIVE",
        "flydsl_enable_autotuning": True,
    },
}


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def mm_nt(a, b):
    return torch.mm(a, b.t())


def run_padded_stride_regression(args):
    """Check aligned padded rows and reject unaligned FlyDSL row strides."""
    from torch._inductor.heuristics.template import flydsl as flydsl_heuristics
    from torch._inductor.utils import run_and_get_code

    dtype = DTYPES[args.dtype]
    common_config = {
        "TILE_M": 128,
        "TILE_N": 128,
        "TILE_K": 64,
        "STAGES": 2,
        "SPLIT_K": 1,
        "BLOCK_K_WARPS": 1,
        "GROUP_M": 0,
        "B_TO_LDS": True,
    }
    cases = (
        {
            "name": "full-tile-small",
            "shape": (64, 64, 128),
            "a_row_stride": 160,
            "b_row_stride": 160,
            "expect_flydsl": True,
            "kernel_config": {
                **common_config,
                "BLOCK_M_WARPS": 4,
                "BLOCK_N_WARPS": 4,
                "USE_HALF_TILE_INTERLEAVED": False,
            },
        },
        {
            "name": "hti-large",
            "shape": (1024, 1024, 1024),
            "a_row_stride": 1056,
            "b_row_stride": 1088,
            "expect_flydsl": True,
            "kernel_config": {
                **common_config,
                "BLOCK_M_WARPS": 2,
                "BLOCK_N_WARPS": 2,
                "USE_HALF_TILE_INTERLEAVED": True,
            },
        },
        {
            "name": "unaligned-row-stride",
            "shape": (64, 64, 128),
            "a_row_stride": 129,
            "b_row_stride": 129,
            "expect_flydsl": False,
            "kernel_config": {
                **common_config,
                "BLOCK_M_WARPS": 4,
                "BLOCK_N_WARPS": 4,
                "USE_HALF_TILE_INTERLEAVED": False,
            },
        },
    )

    torch.manual_seed(0)
    regression_config = {
        **BACKEND_PATCHES["flydsl"],
        "max_autotune_gemm_search_space": "DEFAULT",
        "flydsl_enable_autotuning": False,
    }
    for case in cases:
        name = case["name"]
        m, n, k = case["shape"]
        a_row_stride = case["a_row_stride"]
        b_row_stride = case["b_row_stride"]
        expect_flydsl = case["expect_flydsl"]

        torch._dynamo.reset()
        torch.cuda.empty_cache()

        a_storage = torch.randn(
            (m, a_row_stride), device="cuda", dtype=dtype
        )
        b_storage = torch.randn(
            (n, b_row_stride), device="cuda", dtype=dtype
        )
        a = a_storage[:, :k]
        b = b_storage[:, :k]

        assert a.stride() == (a_row_stride, 1)
        assert b.stride() == (b_row_stride, 1)
        assert b.t().stride() == (1, b_row_stride)

        case_config = regression_config
        if not expect_flydsl:
            case_config = {
                **regression_config,
                "max_autotune_gemm_backends": "ATEN,FLYDSL",
            }

        with (
            inductor_config.patch(**case_config),
            mock.patch.object(
                flydsl_heuristics,
                "get_gemm_configs",
                return_value=[case["kernel_config"]],
            ) as get_gemm_configs,
        ):
            compiled = torch.compile(mm_nt, backend="inductor")
            result, (code,) = run_and_get_code(compiled, a, b)

        uses_flydsl = "async_compile.flydsl" in code
        assert uses_flydsl == expect_flydsl
        if expect_flydsl:
            get_gemm_configs.assert_called()
        else:
            get_gemm_configs.assert_not_called()

        ref = mm_nt(a, b)
        abs_diff = (result.float() - ref.float()).abs()
        max_diff = abs_diff.max().item()
        mismatches = (abs_diff > 3e-2).sum().item()

        print(
            f"FlyDSL padded-stride regression [{name}]: "
            f"M={m} N={n} K={k}, "
            f"A.stride={a.stride()}, B.stride={b.stride()}, "
            f"B.T.stride={b.t().stride()}, "
            f"selected={'FlyDSL' if uses_flydsl else 'fallback'}"
        )
        print(f"max absolute error: {max_diff}")
        print(f"entries with absolute error > 0.03: {mismatches}/{m * n}")

        torch.testing.assert_close(result, ref, atol=3e-2, rtol=3e-2)


def tflops(m, n, k, ms):
    return 2.0 * m * n * k / (ms * 1.0e9)


def e2e_bench(fn, a, b, warmup, reps, rounds):
    for _ in range(warmup):
        fn(a, b)
    torch.cuda.synchronize()
    
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn(a, b)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / reps)
    return statistics.median(samples)


def cuda_graph_bench(fn, a, b, warmup, reps, rounds):
    # Replaying a captured graph bypasses the backend's Python launch path.
    # Comparing this with e2e_bench separates launch bubbles from GPU
    # kernel execution.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fn(a, b)

    def replay(_a, _b):
        graph.replay()
        return graph_output

    return e2e_bench(replay, a, b, warmup, reps, rounds)


def run_case(backend, m, n, k, args):
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    dtype = DTYPES[args.dtype]
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((n, k), device="cuda", dtype=dtype)

    with inductor_config.patch(**BACKEND_PATCHES[backend]):
        compiled = torch.compile(mm_nt, backend="inductor")
        result = compiled(a, b)

    ref = mm_nt(a, b)
    float_max_diff = (result.float() - ref.float()).abs().max().item()
    ok = torch.allclose(result, ref, atol=3e-2, rtol=3e-2)

    e2e_ms = e2e_bench(
        compiled,
        a,
        b,
        warmup=args.warmup,
        reps=args.reps,
        rounds=args.rounds,
    )
    graph_error = None
    try:
        graph_ms = cuda_graph_bench(
            compiled,
            a,
            b,
            warmup=args.warmup,
            reps=args.reps,
            rounds=args.rounds,
        )
    except Exception as exc:
        graph_ms = None
        graph_error = repr(exc)

    return {
        "backend": backend,
        "m": m,
        "n": n,
        "k": k,
        "dtype": args.dtype,
        "ok": bool(ok),
        "float_max_diff": float_max_diff,
        "e2e_ms": e2e_ms,
        "graph_ms": graph_ms,
        "graph_error": graph_error,
        "e2e_tflops": tflops(m, n, k, e2e_ms),
        "graph_tflops": tflops(m, n, k, graph_ms) if graph_ms is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["aten", "triton", "flydsl", "all"],
        default="all",
    )
    parser.add_argument("--output", default="./temp")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--shape-index", type=int, default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument(
        "--padded-stride-regression",
        action="store_true",
        help="run aligned padded-row and unaligned-stride FlyDSL regressions",
    )
    args = parser.parse_args()

    if args.padded_stride_regression:
        run_padded_stride_regression(args)
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backends = ["aten", "triton", "flydsl"] if args.backend == "all" else [args.backend]
    shapes = SHAPES if args.shape_index is None else [SHAPES[args.shape_index]]

    header = (
        f"{'Backend':<8} {'Shape (M/N/K)':<28} {'DType':<10} {'Accuracy':<8} "
        f"{'E2E ms':>10} {'E2E TFLOPS':>12} "
        f"{'Graph ms':>10} {'Graph TFLOPS':>13} {'Max diff':>11}  Error"
    )
    print(header)
    print("-" * len(header))

    with out_path.open("a", buffering=1) as f:
        for backend in backends:
            for m, n, k in shapes:
                try:
                    row = run_case(backend, m, n, k, args)
                except Exception as exc:
                    row = {
                        "backend": backend,
                        "m": m,
                        "n": n,
                        "k": k,
                        "dtype": args.dtype,
                        "ok": False,
                        "error": repr(exc),
                    }
                f.write(json.dumps(row, sort_keys=True) + "\n")
                accuracy = (
                    "ERROR"
                    if row.get("error")
                    else "PASS"
                    if row["ok"]
                    else "FAIL"
                )
                e2e_ms = (
                    f"{row['e2e_ms']:.4f}" if row.get("e2e_ms") is not None else "-"
                )
                e2e_tflops = (
                    f"{row['e2e_tflops']:.1f}"
                    if row.get("e2e_tflops") is not None
                    else "-"
                )
                graph_ms = (
                    f"{row['graph_ms']:.4f}"
                    if row.get("graph_ms") is not None
                    else "-"
                )
                graph_tflops = (
                    f"{row['graph_tflops']:.1f}"
                    if row.get("graph_tflops") is not None
                    else "-"
                )
                max_diff = (
                    f"{row['float_max_diff']:.3e}"
                    if row.get("float_max_diff") is not None
                    else "-"
                )
                error = row.get("error") or row.get("graph_error") or ""
                shape = f"M={m} N={n} K={k}"
                print(
                    f"{backend.upper():<8} {shape:<28} "
                    f"{args.dtype:<10} {accuracy:<8} "
                    f"{e2e_ms:>10} {e2e_tflops:>12} "
                    f"{graph_ms:>10} {graph_tflops:>13} "
                    f"{max_diff:>11}  {error}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
