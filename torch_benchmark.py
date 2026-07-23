import argparse
import json
import statistics
from pathlib import Path

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
    args = parser.parse_args()

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
