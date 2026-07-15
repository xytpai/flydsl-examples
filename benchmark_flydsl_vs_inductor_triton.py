#!/usr/bin/env python3
"""Compare standalone FlyDSL JIT with standalone Triton JIT for GEMM.

This script intentionally compares:

* ``kernels.hgemm_gfx950.layout_gemm`` from this repository.
* A standalone ``@triton.jit`` GEMM kernel equivalent to the one in
  ``test_hgemm_layout.py``.

It reports first-call JIT+run time and steady-state GPU runtime.  Each
backend/shape/repeat runs in a fresh subprocess with its own cache directory so
cold compile numbers are not affected by earlier runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any


_FLYDSL_PREWARM_FUNCS: tuple[Any, Any] | None = None


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"shape must be M,N,K or MxNxK, got {value!r}"
        )
    try:
        m, n, k = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}") from exc
    return m, n, k


def _format_shape(shape: tuple[int, int, int]) -> str:
    m, n, k = shape
    return f"{m}x{n}x{k}"


def _tflops(m: int, n: int, k: int, avg_ms: float) -> float:
    return (2.0 * m * n * k) / (avg_ms / 1e3) / 1e12


def _resolve_input_dtype(dtype_name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise RuntimeError(f"unsupported dtype {dtype_name!r}; use fp16 or bf16")
    return mapping[key]


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


def _prewarm_triton_runtime() -> None:
    import torch

    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("standalone-triton requires triton to be installed") from exc

    @triton.jit
    def _prewarm_kernel(x):
        pid = tl.program_id(0)
        val = tl.load(x + pid)
        tl.store(x + pid, val + 1.0)

    x = torch.zeros((1,), device="cuda", dtype=torch.float32)
    _prewarm_kernel[(1,)](x, num_warps=1)
    torch.cuda.synchronize()


def _prewarm_flydsl_runtime() -> None:
    import torch

    import flydsl.compiler  # noqa: F401
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    globals()["fx"] = fx
    global _FLYDSL_PREWARM_FUNCS
    if _FLYDSL_PREWARM_FUNCS is None:

        @flyc.kernel
        def _prewarm_vector_add_kernel(
            A: fx.Tensor,
            B: fx.Tensor,
            C: fx.Tensor,
        ):
            bid = fx.block_idx.x
            tid = fx.thread_idx.x
            block_dim = 64

            A = fx.rocdl.make_buffer_tensor(A)
            tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
            tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
            tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))

            tA = fx.slice(tA, (None, bid))
            tB = fx.slice(tB, (None, bid))
            tC = fx.slice(tC, (None, bid))
            tA = fx.logical_divide(tA, fx.make_layout(1, 1))
            tB = fx.logical_divide(tB, fx.make_layout(1, 1))
            tC = fx.logical_divide(tC, fx.make_layout(1, 1))

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


def _add_compile_estimate(result: dict[str, Any]) -> dict[str, Any]:
    # First call includes JIT compile plus one launch.  Subtracting steady-state
    # runtime gives a useful approximation while keeping measurement simple.
    result["compile_estimate_s"] = max(
        0.0, result["first_call_s"] - result["avg_ms"] / 1e3
    )
    result["compile_estimate_ms"] = result["compile_estimate_s"] * 1e3
    result["first_call_ms"] = result["first_call_s"] * 1e3
    return result


def _bench_flydsl_layout(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from kernels.hgemm_gfx950 import BLOCK_K, BLOCK_M, BLOCK_N, layout_gemm

    m, n, k = args.m, args.n, args.k
    if m % BLOCK_M or n % BLOCK_N or k % BLOCK_K:
        raise RuntimeError(
            "FlyDSL layout_gemm requires M/N/K multiples of "
            f"{(BLOCK_M, BLOCK_N, BLOCK_K)}, got {(m, n, k)}"
        )

    torch.manual_seed(args.seed)
    input_dtype = _resolve_input_dtype(args.dtype)
    a = torch.randn((m, k), dtype=input_dtype, device="cuda")
    b = torch.randn((n, k), dtype=input_dtype, device="cuda")
    c = torch.empty((m, n), dtype=torch.float32, device="cuda")

    if args.prewarm_runtime:
        _prewarm_cuda_runtime()
        _prewarm_flydsl_runtime()

    torch.cuda.synchronize()
    start = time.perf_counter()
    layout_gemm(
        a,
        b,
        c,
        block_m_warps=args.block_m_warps,
        block_n_warps=args.block_n_warps,
    )
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - start

    def run() -> None:
        layout_gemm(
            a,
            b,
            c,
            block_m_warps=args.block_m_warps,
            block_n_warps=args.block_n_warps,
        )

    avg_ms = _cuda_event_bench(run, warmup=args.warmup, iters=args.iters)

    return _add_compile_estimate({
        "backend": "flydsl-layout",
        "shape": _format_shape((m, n, k)),
        "first_call_s": first_call_s,
        "avg_ms": avg_ms,
        "tflops": _tflops(m, n, k, avg_ms),
        "input_dtype": _dtype_name(input_dtype),
        "output_dtype": "float32",
    })


def _bench_standalone_triton(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("standalone-triton requires triton to be installed") from exc

    @triton.jit
    def _triton_gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + offs_n[None, :] * stride_bn + k_idxs[:, None] * stride_bk,
                mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
                other=0.0,
            )
            acc += tl.dot(a, b, out_dtype=tl.float32)

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    m, n, k = args.m, args.n, args.k
    torch.manual_seed(args.seed)
    input_dtype = _resolve_input_dtype(args.dtype)
    a = torch.randn((m, k), dtype=input_dtype, device="cuda")
    b = torch.randn((n, k), dtype=input_dtype, device="cuda")
    c = torch.empty((m, n), dtype=torch.float32, device="cuda")

    grid = (
        triton.cdiv(m, args.triton_block_m) * triton.cdiv(n, args.triton_block_n),
    )

    if args.prewarm_runtime:
        _prewarm_cuda_runtime()
        _prewarm_triton_runtime()

    def launch() -> None:
        _triton_gemm_kernel[grid](
            a,
            b,
            c,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            m,
            n,
            k,
            BLOCK_M=args.triton_block_m,
            BLOCK_N=args.triton_block_n,
            BLOCK_K=args.triton_block_k,
            GROUP_M=args.triton_group_m,
            num_warps=args.triton_num_warps,
        )

    torch.cuda.synchronize()
    start = time.perf_counter()
    launch()
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - start

    avg_ms = _cuda_event_bench(launch, warmup=args.warmup, iters=args.iters)

    if args.check:
        ref = a.float() @ b.float().T
        torch.testing.assert_close(c, ref, atol=5e-2, rtol=5e-2)

    return _add_compile_estimate({
        "backend": "standalone-triton",
        "shape": _format_shape((m, n, k)),
        "first_call_s": first_call_s,
        "avg_ms": avg_ms,
        "tflops": _tflops(m, n, k, avg_ms),
        "input_dtype": _dtype_name(input_dtype),
        "output_dtype": "float32",
        "triton_block_m": args.triton_block_m,
        "triton_block_n": args.triton_block_n,
        "triton_block_k": args.triton_block_k,
    })


def _bench_inductor_triton(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch._inductor.config as inductor_config

    m, n, k = args.m, args.n, args.k

    # Keep the comparison focused on Inductor's Triton GEMM implementation.
    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm = True
    inductor_config.max_autotune_gemm_backends = "TRITON"

    torch.manual_seed(args.seed)
    input_dtype = _resolve_input_dtype(args.dtype)
    a = torch.randn((m, k), dtype=input_dtype, device="cuda")
    b = torch.randn((n, k), dtype=input_dtype, device="cuda")

    if args.prewarm_runtime:
        _prewarm_cuda_runtime()

    def gemm_fn(lhs: torch.Tensor, rhs_nk: torch.Tensor) -> torch.Tensor:
        return lhs @ rhs_nk.T

    compiled = torch.compile(
        gemm_fn,
        backend="inductor",
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
    )

    torch.cuda.synchronize()
    start = time.perf_counter()
    out = compiled(a, b)
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - start

    def run() -> None:
        compiled(a, b)

    avg_ms = _cuda_event_bench(run, warmup=args.warmup, iters=args.iters)

    if args.check:
        ref = a @ b.T
        torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)

    return _add_compile_estimate({
        "backend": "inductor-triton",
        "shape": _format_shape((m, n, k)),
        "first_call_s": first_call_s,
        "avg_ms": avg_ms,
        "tflops": _tflops(m, n, k, avg_ms),
        "input_dtype": _dtype_name(input_dtype),
        "output_dtype": str(out.dtype).replace("torch.", ""),
    })


def _worker_main(args: argparse.Namespace) -> None:
    if not args.cache_dir:
        raise RuntimeError("--cache-dir is required in worker mode")

    cache_dir = Path(args.cache_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir / "triton")
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(cache_dir / "flydsl")
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(args.compile_threads)

    if args.pytorch_root:
        sys.path.insert(0, args.pytorch_root)

    if args.backend == "flydsl-layout":
        result = _bench_flydsl_layout(args)
    elif args.backend == "standalone-triton":
        result = _bench_standalone_triton(args)
    elif args.backend == "inductor-triton":
        result = _bench_inductor_triton(args)
    else:
        raise RuntimeError(f"unknown backend: {args.backend}")

    result["cache_dir"] = str(cache_dir)
    result["compile_threads"] = args.compile_threads
    print("RESULT_JSON: " + json.dumps(result, sort_keys=True), flush=True)


def _run_worker(
    *,
    script_path: Path,
    base_args: argparse.Namespace,
    backend: str,
    shape: tuple[int, int, int],
    repeat: int,
    cache_root: Path,
) -> dict[str, Any]:
    m, n, k = shape
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
        "--k",
        str(k),
        "--warmup",
        str(base_args.warmup),
        "--iters",
        str(base_args.iters),
        "--seed",
        str(base_args.seed + repeat),
        "--compile-threads",
        str(base_args.compile_threads),
        "--dtype",
        base_args.dtype,
        "--cache-dir",
        str(cache_dir),
        "--block-m-warps",
        str(base_args.block_m_warps),
        "--block-n-warps",
        str(base_args.block_n_warps),
        "--triton-block-m",
        str(base_args.triton_block_m),
        "--triton-block-n",
        str(base_args.triton_block_n),
        "--triton-block-k",
        str(base_args.triton_block_k),
        "--triton-group-m",
        str(base_args.triton_group_m),
        "--triton-num-warps",
        str(base_args.triton_num_warps),
    ]
    if base_args.prewarm_runtime:
        cmd.append("--prewarm-runtime")
    else:
        cmd.append("--no-prewarm-runtime")
    if base_args.check:
        cmd.append("--check")
    if base_args.pytorch_root:
        cmd.extend(["--pytorch-root", base_args.pytorch_root])

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
        first_call = sorted(row["first_call_s"] for row in rows)
        compile_estimate = sorted(row["compile_estimate_s"] for row in rows)
        runtime = sorted(row["avg_ms"] for row in rows)
        tflops = sorted(row["tflops"] for row in rows)
        mid = len(rows) // 2
        summary.append(
            {
                "shape": shape,
                "backend": backend,
                "repeats": len(rows),
                "first_call_s_median": first_call[mid],
                "compile_estimate_s_median": compile_estimate[mid],
                "first_call_ms_median": first_call[mid] * 1e3,
                "compile_estimate_ms_median": compile_estimate[mid] * 1e3,
                "runtime_ms_median": runtime[mid],
                "tflops_median": tflops[mid],
                "input_dtype": rows[0]["input_dtype"],
                "output_dtype": rows[0]["output_dtype"],
            }
        )
    return summary


def _print_markdown(summary: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary:
        if row["backend"] == "standalone-triton":
            baselines[(row["shape"], row["input_dtype"])] = row

    def fmt_ms(value: float) -> str:
        if value >= 100:
            return f"{value:.0f}"
        if value >= 10:
            return f"{value:.1f}"
        return f"{value:.3f}"

    def fmt_x(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:.2f}x"

    def ratios(row: dict[str, Any]) -> tuple[float | None, float | None]:
        base = baselines.get((row["shape"], row["input_dtype"]))
        if base is None:
            return None, None
        jit_ratio = base["compile_estimate_ms_median"] / row[
            "compile_estimate_ms_median"
        ]
        runtime_ratio = base["runtime_ms_median"] / row["runtime_ms_median"]
        return jit_ratio, runtime_ratio

    print()
    print(
        "| shape | dtype | backend | JIT ms | runtime ms | TFLOP/s | "
        "JIT vs Triton | runtime vs Triton | repeats |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        jit_ratio, runtime_ratio = ratios(row)
        print(
            "| {shape} | {input_dtype} | {backend} | {compile_est} | "
            "{runtime} | {tflops:.1f} | {jit_ratio} | {runtime_ratio} | "
            "{repeats} |".format(
                shape=row["shape"],
                backend=row["backend"],
                input_dtype=row["input_dtype"],
                compile_est=fmt_ms(row["compile_estimate_ms_median"]),
                runtime=fmt_ms(row["runtime_ms_median"]),
                tflops=row["tflops_median"],
                jit_ratio=fmt_x(jit_ratio),
                runtime_ratio=fmt_x(runtime_ratio),
                repeats=row["repeats"],
            )
        )


def _parent_main(args: argparse.Namespace) -> None:
    script_path = Path(__file__).resolve()
    shapes = args.shape or [(args.m, args.n, args.k)]
    backends = args.backends.split(",")
    cache_root_is_temporary = args.cache_root is None
    cache_root = (
        Path(args.cache_root)
        if args.cache_root
        else Path(tempfile.mkdtemp(prefix="flydsl_vs_triton_"))
    )
    if cache_root.exists() and args.clear_cache_root:
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"cache_root: {cache_root}")
    print(f"backends: {', '.join(backends)}")
    print(f"shapes: {', '.join(_format_shape(shape) for shape in shapes)}")
    print(f"warmup={args.warmup} iters={args.iters} repeats={args.repeats}")

    results = []
    for shape in shapes:
        for backend in backends:
            for repeat in range(args.repeats):
                print(
                    f"\n=== {backend} shape={_format_shape(shape)} repeat={repeat} ===",
                    flush=True,
                )
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

    try:
        if results:
            summary = _summarize(results)
            _print_markdown(summary)
        else:
            summary = []
            print("\nNo successful benchmark results.")

        if args.json_out:
            payload = {"results": results, "summary": summary}
            Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"\nwrote JSON: {args.json_out}")
    finally:
        if cache_root_is_temporary and not args.keep_cache:
            shutil.rmtree(cache_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=textwrap.dedent(
            """\
            Examples:
              python benchmark_flydsl_vs_inductor_triton.py --m 8192 --n 8192 --k 8192
              python benchmark_flydsl_vs_inductor_triton.py --shape 4096,4096,4096 --shape 8192x8192x8192 --repeats 3

            Notes:
              * Default comparison is standalone FlyDSL JIT vs standalone Triton JIT.
              * Both default backends write fp32 output.
              * Use --dtype bf16 to compare bfloat16 inputs.
              * Runtime prewarm is enabled by default to reduce CUDA/Triton/FlyDSL
                cold-start noise before measuring the target GEMM.
              * JIT estimate subtracts one steady-state runtime from first measured call.
              * Each worker uses a fresh per-backend cache. Temporary cache roots
                are removed after the run unless --keep-cache is set.
              * Optional backend: --backends flydsl-layout,standalone-triton,inductor-triton
            """
        ),
    )
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument(
        "--backends",
        default="flydsl-layout,standalone-triton",
        help="comma separated: flydsl-layout,standalone-triton,inductor-triton",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float16", choices=("float16", "fp16", "bfloat16", "bf16"))
    parser.add_argument("--compile-threads", type=int, default=1)
    parser.add_argument("--block-m-warps", type=int, default=2)
    parser.add_argument("--block-n-warps", type=int, default=4)
    parser.add_argument("--triton-block-m", type=int, default=128)
    parser.add_argument("--triton-block-n", type=int, default=128)
    parser.add_argument("--triton-block-k", type=int, default=64)
    parser.add_argument("--triton-group-m", type=int, default=8)
    parser.add_argument("--triton-num-warps", type=int, default=8)
    parser.add_argument("--cache-root")
    parser.add_argument(
        "--clear-cache-root",
        action="store_true",
        default=True,
        help="clear --cache-root before running; enabled by default",
    )
    parser.add_argument(
        "--no-clear-cache-root",
        dest="clear_cache_root",
        action="store_false",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="keep temporary cache root after the run",
    )
    parser.add_argument("--json-out")
    parser.add_argument("--pytorch-root", default="/mnt/data/xiaobing/pytorch")
    parser.add_argument(
        "--prewarm-runtime",
        dest="prewarm_runtime",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-prewarm-runtime",
        dest="prewarm_runtime",
        action="store_false",
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
