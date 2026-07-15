#!/usr/bin/env python3
"""Compare standalone FlyDSL JIT with AITER Triton JIT for RMSNorm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


def _parse_shape(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"shape must be M,N or MxN, got {value!r}")
    try:
        m, n = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}") from exc
    return m, n


def _format_shape(shape: tuple[int, int]) -> str:
    m, n = shape
    return f"{m}x{n}"


def _resolve_input_dtype(dtype_name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise RuntimeError(f"unsupported dtype {dtype_name!r}; use fp32, fp16, or bf16")
    return mapping[key]


def _flydsl_dtype_name(dtype_name: str) -> str:
    mapping = {
        "float32": "f32",
        "fp32": "f32",
        "float16": "f16",
        "fp16": "f16",
        "half": "f16",
        "bfloat16": "bf16",
        "bf16": "bf16",
    }
    return mapping[dtype_name.lower()]


def _dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _cuda_event_bench(callable_, *, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()

    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iters):
        callable_()
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) / iters


def _prewarm_cuda_runtime() -> None:
    import torch

    x = torch.empty((1,), device="cuda")
    x.zero_()
    torch.cuda.synchronize()


def _disable_flydsl_source_locations() -> None:
    import flydsl.expr.meta as meta
    from flydsl._mlir import ir

    if getattr(meta, "_benchmark_source_locs_disabled", False):
        return

    meta.capture_user_location = lambda: ir.Location.unknown()
    meta._benchmark_source_locs_disabled = True


def _prewarm_triton_runtime() -> None:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _prewarm_kernel(x):
        pid = tl.program_id(0)
        val = tl.load(x + pid)
        tl.store(x + pid, val + 1.0)

    x = torch.zeros((1,), device="cuda", dtype=torch.float32)
    _prewarm_kernel[(1,)](x, num_warps=1)
    torch.cuda.synchronize()


_FLYDSL_PREWARM_FUNCS: tuple[Any, Any] | None = None


def _prewarm_flydsl_runtime() -> None:
    import torch

    import flydsl.compiler as flyc
    import flydsl.expr as fx

    globals()["fx"] = fx
    global _FLYDSL_PREWARM_FUNCS
    if _FLYDSL_PREWARM_FUNCS is None:

        @flyc.kernel
        def _prewarm_vector_add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
            bid = fx.block_idx.x
            tid = fx.thread_idx.x
            block_dim = 64

            A = fx.rocdl.make_buffer_tensor(A)
            tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
            tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
            tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))
            tA = fx.logical_divide(fx.slice(tA, (None, bid)), fx.make_layout(1, 1))
            tB = fx.logical_divide(fx.slice(tB, (None, bid)), fx.make_layout(1, 1))
            tC = fx.logical_divide(fx.slice(tC, (None, bid)), fx.make_layout(1, 1))

            copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
            copy_atom_buffer = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            rA = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
            rB = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
            rC = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
            fx.copy_atom_call(copy_atom_buffer, fx.slice(tA, (None, tid)), rA)
            fx.copy_atom_call(copy_atom, fx.slice(tB, (None, tid)), rB)
            vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
            fx.memref_store_vec(vC, rC)
            fx.copy_atom_call(copy_atom, rC, fx.slice(tC, (None, tid)))

        @flyc.jit
        def _prewarm_vector_add(
            A: fx.Tensor,
            B: fx.Tensor,
            C: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            block_dim = 64
            _prewarm_vector_add_kernel(A, B, C).launch(
                grid=(1, 1, 1), block=(block_dim, 1, 1), stream=stream
            )

        _FLYDSL_PREWARM_FUNCS = (_prewarm_vector_add_kernel, _prewarm_vector_add)

    n = 64
    a = torch.ones((n,), dtype=torch.float32, device="cuda")
    b = torch.ones((n,), dtype=torch.float32, device="cuda")
    c = torch.empty((n,), dtype=torch.float32, device="cuda")
    _, prewarm_vector_add = _FLYDSL_PREWARM_FUNCS
    prewarm_vector_add(a, b, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()


def _add_jit_estimate(result: dict[str, Any]) -> dict[str, Any]:
    result["jit_estimate_s"] = max(0.0, result["first_call_s"] - result["avg_ms"] / 1e3)
    result["jit_estimate_ms"] = result["jit_estimate_s"] * 1e3
    result["first_call_ms"] = result["first_call_s"] * 1e3
    return result


def _approx_rmsnorm_bytes(m: int, n: int, dtype) -> int:
    # Approximate traffic: read input + read gamma + write output.
    elem_bytes = dtype.itemsize
    return 3 * m * n * elem_bytes


def _gbps(num_bytes: int, avg_ms: float) -> float:
    return num_bytes / (avg_ms / 1e3) / 1e9


def _reference_rmsnorm(x, weight, eps: float):
    import torch

    x_f = x.float()
    w_f = weight.float()
    return (x_f / torch.sqrt((x_f * x_f).mean(dim=1, keepdim=True) + eps) * w_f).to(
        x.dtype
    )


def _bench_flydsl(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not args.flydsl_source_locs:
        _disable_flydsl_source_locations()

    import flydsl.compiler as flyc
    from kernels.rmsnorm_kernel import build_rmsnorm_module

    m, n = args.m, args.n
    input_dtype = _resolve_input_dtype(args.dtype)
    flydsl_dtype = _flydsl_dtype_name(args.dtype)

    torch.manual_seed(args.seed)
    x = torch.randn((m, n), device="cuda", dtype=input_dtype).contiguous()
    weight = torch.rand((n,), device="cuda", dtype=input_dtype).contiguous()
    out = torch.empty((m, n), device="cuda", dtype=input_dtype)

    if args.prewarm_runtime:
        _prewarm_cuda_runtime()
        _prewarm_flydsl_runtime()

    launch_fn = build_rmsnorm_module(n, flydsl_dtype)
    stream = torch.cuda.current_stream()

    torch.cuda.synchronize()
    start = time.perf_counter()
    compiled_fn = flyc.compile(launch_fn, x, weight, out, m, stream)
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - start

    def run() -> None:
        compiled_fn(x, weight, out, m, stream)

    avg_ms = _cuda_event_bench(run, warmup=args.warmup, iters=args.iters)

    if args.check:
        ref = _reference_rmsnorm(x, weight, args.eps)
        torch.testing.assert_close(out, ref, atol=args.atol, rtol=args.rtol)

    num_bytes = _approx_rmsnorm_bytes(m, n, input_dtype)
    return _add_jit_estimate(
        {
            "backend": "flydsl-rmsnorm",
            "shape": _format_shape((m, n)),
            "first_call_s": first_call_s,
            "avg_ms": avg_ms,
            "gbps": _gbps(num_bytes, avg_ms),
            "bytes": num_bytes,
            "input_dtype": _dtype_name(input_dtype),
            "output_dtype": _dtype_name(input_dtype),
            "flydsl_source_locs": args.flydsl_source_locs,
        }
    )


def _bench_aiter_triton(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import triton

    from kernels.triton_rmsnorm import _rms_norm_kernel as rms_norm_kernel

    m, n = args.m, args.n
    input_dtype = _resolve_input_dtype(args.dtype)

    torch.manual_seed(args.seed)
    x = torch.randn((m, n), device="cuda", dtype=input_dtype).contiguous()
    weight = torch.rand((n,), device="cuda", dtype=input_dtype).contiguous()

    if args.prewarm_runtime:
        _prewarm_cuda_runtime()
        _prewarm_triton_runtime()

    def num_programs() -> int:
        return min(m, torch.cuda.get_device_properties(x.device).multi_processor_count)

    def block_size() -> int:
        return min(65536 // x.element_size(), triton.next_power_of_2(n))

    y = torch.empty_like(x)
    rsigma = torch.empty((m,), dtype=torch.float32, device=x.device)

    def run_rms_norm():
        blk_size = block_size()
        use_blocked = n > blk_size
        num_prgms = num_programs()
        grid = lambda meta: (num_prgms,)  # noqa: E731
        rms_norm_kernel[grid](
            x,
            y,
            weight,
            rsigma,
            x.stride(0),
            y.stride(0),
            m,
            n,
            args.eps,
            blk_size,
            use_blocked,
            num_prgms,
        )
        return y

    torch.cuda.synchronize()
    start = time.perf_counter()
    out = run_rms_norm()
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - start

    def run() -> None:
        run_rms_norm()

    avg_ms = _cuda_event_bench(run, warmup=args.warmup, iters=args.iters)

    if args.check:
        ref = _reference_rmsnorm(x, weight, args.eps)
        torch.testing.assert_close(out, ref, atol=args.atol, rtol=args.rtol)

    num_bytes = _approx_rmsnorm_bytes(m, n, input_dtype)
    return _add_jit_estimate(
        {
            "backend": "aiter-triton-rmsnorm",
            "shape": _format_shape((m, n)),
            "first_call_s": first_call_s,
            "avg_ms": avg_ms,
            "gbps": _gbps(num_bytes, avg_ms),
            "bytes": num_bytes,
            "input_dtype": _dtype_name(input_dtype),
            "output_dtype": _dtype_name(input_dtype),
        }
    )


def _worker_main(args: argparse.Namespace) -> None:
    if not args.cache_dir:
        raise RuntimeError("--cache-dir is required in worker mode")

    cache_dir = Path(args.cache_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir / "triton")
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(cache_dir / "flydsl")

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if args.backend == "flydsl-rmsnorm":
        result = _bench_flydsl(args)
    elif args.backend == "aiter-triton-rmsnorm":
        result = _bench_aiter_triton(args)
    else:
        raise RuntimeError(f"unknown backend: {args.backend}")

    result["cache_dir"] = str(cache_dir)
    print("RESULT_JSON: " + json.dumps(result, sort_keys=True), flush=True)


def _run_worker(
    *,
    script_path: Path,
    base_args: argparse.Namespace,
    backend: str,
    shape: tuple[int, int],
    repeat: int,
    cache_root: Path,
) -> dict[str, Any]:
    m, n = shape
    cache_dir = cache_root / f"{backend}-{_format_shape(shape)}-r{repeat}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--backend",
        backend,
        "--m",
        str(m),
        "--n",
        str(n),
        "--dtype",
        base_args.dtype,
        "--eps",
        str(base_args.eps),
        "--warmup",
        str(base_args.warmup),
        "--iters",
        str(base_args.iters),
        "--seed",
        str(base_args.seed + repeat),
        "--cache-dir",
        str(cache_dir),
        "--atol",
        str(base_args.atol),
        "--rtol",
        str(base_args.rtol),
    ]
    if base_args.prewarm_runtime:
        cmd.append("--prewarm-runtime")
    else:
        cmd.append("--no-prewarm-runtime")
    if base_args.flydsl_source_locs:
        cmd.append("--flydsl-source-locs")
    if base_args.check:
        cmd.append("--check")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if not base_args.quiet or completed.returncode != 0:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed for {backend} {_format_shape(shape)} repeat {repeat}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("RESULT_JSON: "):
            return json.loads(line[len("RESULT_JSON: ") :])
    raise RuntimeError("worker completed but did not print RESULT_JSON")


def _summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault((result["shape"], result["backend"]), []).append(result)

    summary = []
    for (shape, backend), rows in sorted(grouped.items()):
        jit = sorted(row["jit_estimate_ms"] for row in rows)
        first = sorted(row["first_call_ms"] for row in rows)
        runtime = sorted(row["avg_ms"] for row in rows)
        gbps = sorted(row["gbps"] for row in rows)
        mid = len(rows) // 2
        summary.append(
            {
                "shape": shape,
                "backend": backend,
                "repeats": len(rows),
                "jit_ms_median": jit[mid],
                "first_call_ms_median": first[mid],
                "runtime_ms_median": runtime[mid],
                "gbps_median": gbps[mid],
                "input_dtype": rows[0]["input_dtype"],
                "output_dtype": rows[0]["output_dtype"],
            }
        )
    return summary


def _print_markdown(summary: list[dict[str, Any]]) -> None:
    baselines = {
        (row["shape"], row["input_dtype"]): row
        for row in summary
        if row["backend"] == "aiter-triton-rmsnorm"
    }

    def fmt_ms(value: float) -> str:
        if value >= 100:
            return f"{value:.0f}"
        if value >= 10:
            return f"{value:.1f}"
        return f"{value:.3f}"

    def fmt_x(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}x"

    print()
    print(
        "| shape | dtype | backend | JIT ms | runtime ms | GB/s | "
        "JIT vs Triton | runtime vs Triton | repeats |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        base = baselines.get((row["shape"], row["input_dtype"]))
        jit_ratio = None
        runtime_ratio = None
        if base is not None:
            jit_ratio = base["jit_ms_median"] / row["jit_ms_median"]
            runtime_ratio = base["runtime_ms_median"] / row["runtime_ms_median"]
        print(
            "| {shape} | {dtype} | {backend} | {jit} | {runtime} | "
            "{gbps:.1f} | {jit_ratio} | {runtime_ratio} | {repeats} |".format(
                shape=row["shape"],
                dtype=row["input_dtype"],
                backend=row["backend"],
                jit=fmt_ms(row["jit_ms_median"]),
                runtime=fmt_ms(row["runtime_ms_median"]),
                gbps=row["gbps_median"],
                jit_ratio=fmt_x(jit_ratio),
                runtime_ratio=fmt_x(runtime_ratio),
                repeats=row["repeats"],
            )
        )


def _parent_main(args: argparse.Namespace) -> None:
    script_path = Path(__file__).resolve()
    shapes = args.shape or [(args.m, args.n)]
    backends = args.backends.split(",")
    cache_root_is_temporary = args.cache_root is None
    cache_root = (
        Path(args.cache_root)
        if args.cache_root
        else Path(tempfile.mkdtemp(prefix="flydsl_rmsnorm_vs_triton_"))
    )
    if cache_root.exists() and args.clear_cache_root:
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"cache_root: {cache_root}")
    print(f"backends: {', '.join(backends)}")
    print(f"shapes: {', '.join(_format_shape(shape) for shape in shapes)}")
    print(f"dtype={args.dtype} warmup={args.warmup} iters={args.iters} repeats={args.repeats}")

    results = []
    try:
        for shape in shapes:
            for backend in backends:
                for repeat in range(args.repeats):
                    print(f"\n=== {backend} shape={_format_shape(shape)} repeat={repeat} ===", flush=True)
                    try:
                        results.append(
                            _run_worker(
                                script_path=script_path,
                                base_args=args,
                                backend=backend,
                                shape=shape,
                                repeat=repeat,
                                cache_root=cache_root,
                            )
                        )
                    except RuntimeError as exc:
                        if not args.keep_going:
                            raise
                        print(f"SKIP {backend} {_format_shape(shape)} repeat={repeat}: {exc}")

        summary = _summarize(results) if results else []
        if summary:
            _print_markdown(summary)
        else:
            print("\nNo successful benchmark results.")
        if args.json_out:
            payload = {"results": results, "summary": summary}
            Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"\nwrote JSON: {args.json_out}")
    finally:
        if cache_root_is_temporary and not args.keep_cache:
            shutil.rmtree(cache_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32768)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument(
        "--backends",
        default="flydsl-rmsnorm,aiter-triton-rmsnorm",
        help="comma separated: flydsl-rmsnorm,aiter-triton-rmsnorm",
    )
    parser.add_argument("--dtype", default="bf16", choices=("fp32", "float32", "fp16", "float16", "bf16", "bfloat16"))
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--cache-root")
    parser.add_argument("--clear-cache-root", action="store_true", default=True)
    parser.add_argument("--no-clear-cache-root", dest="clear_cache_root", action="store_false")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--json-out")
    parser.add_argument("--prewarm-runtime", dest="prewarm_runtime", action="store_true", default=True)
    parser.add_argument("--no-prewarm-runtime", dest="prewarm_runtime", action="store_false")
    parser.add_argument(
        "--flydsl-source-locs",
        action="store_true",
        help="keep full FlyDSL source locations during benchmark compilation",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", help=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        _worker_main(args)
    else:
        _parent_main(args)


if __name__ == "__main__":
    main()
