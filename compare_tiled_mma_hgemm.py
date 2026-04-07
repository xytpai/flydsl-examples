"""Compare searched tiled-MMA GEMM against hgemm.

This script is intentionally narrower than `bench_gemm.py`:

- it benchmarks one GEMM shape at a time
- it searches the generic `(tile_m, tile_n, tile_k, split_k)` space for
  `tiled_mma_gemm.py`
- it searches the same generic space for both backends and reports a fair
  best-vs-best comparison
- the final selected configs are revalidated against `torch.matmul` before they
  are reported
"""

import argparse
import itertools
from dataclasses import dataclass

import torch
from tqdm import tqdm

from aiter.test_common import run_perftest
import hgemm
import tiled_mma_gemm as tiled_mma


CLI_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
}

SELECTIONS = {
    "tile_m": [16, 32, 48, 64, 96, 128],
    "tile_n": [64, 128, 256],
    "tile_k": [64, 128],
    "split_k": [1, 2, 4, 8, 16],
}

WEIGHT_LAYOUTS = {"preshuffle", "non-shuffle"}


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int
    dtype: torch.dtype


@dataclass(frozen=True)
class PreparedInputs:
    a: torch.Tensor
    b: torch.Tensor
    tiled_preshuffled_b: torch.Tensor
    hgemm_preshuffled_b: torch.Tensor
    ref: torch.Tensor


@dataclass(frozen=True)
class BenchResult:
    backend: str
    label: str
    config: dict
    us: float
    maxdiff: float
    tol: float


def _candidate_configs(limit=None):
    keys = list(SELECTIONS.keys())
    values = [SELECTIONS[key] for key in keys]
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    if limit is not None:
        configs = configs[:limit]
    return configs


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"unsupported dtype: {dtype!r}")


def _make_inputs(shape: Shape) -> PreparedInputs:
    a = torch.empty((shape.m, shape.k), dtype=shape.dtype, device="cuda")
    a.uniform_(-1, 1)
    b = torch.empty((shape.n, shape.k), dtype=shape.dtype, device="cuda")
    b.uniform_(-1, 1)
    return PreparedInputs(
        a=a,
        b=b,
        tiled_preshuffled_b=tiled_mma._shuffle_b(b),
        hgemm_preshuffled_b=hgemm.hgemm_shuffle_b(b),
        ref=torch.matmul(a, b.t()),
    )


def _get_tol(dtype: torch.dtype, split_k: int) -> float:
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    if split_k > 1:
        atol = max(atol, 0.125 if dtype == torch.float16 else 0.5)
    return atol


def _tflops(shape: Shape, us: float) -> float:
    return (2 * shape.m * shape.n * shape.k) / (us / 1e6) / 1e12


def _make_tiled_kernel_config(candidate: dict) -> tiled_mma.KernelConfig:
    return tiled_mma.KernelConfig(
        block_m=candidate["tile_m"],
        block_n=candidate["tile_n"],
        block_k=candidate["tile_k"],
    )


def _make_hgemm_kwargs(candidate: dict, *, weight_layout: str) -> dict:
    return {
        "TILE_M": candidate["tile_m"],
        "TILE_N": candidate["tile_n"],
        "TILE_K": candidate["tile_k"],
        "SPLIT_K": candidate["split_k"],
        "BLOCK_M_WARPS": 1,
        "BLOCK_N_WARPS": 4,
        "B_PRE_SHUFFLE": weight_layout == "preshuffle",
        "B_TO_LDS": False,
    }


def _make_tiled_runner(prepared: PreparedInputs, candidate: dict, *, weight_layout: str):
    config = _make_tiled_kernel_config(candidate)
    split_k = candidate["split_k"]
    tiled_mma._validate_shape(prepared.a.shape[0], prepared.b.shape[0], prepared.a.shape[1], config, split_k=split_k)

    def runner(out: torch.Tensor):
        if weight_layout == "preshuffle":
            tiled_mma._launch_preshuffled(
                prepared.a,
                prepared.tiled_preshuffled_b,
                out,
                config=config,
                split_k=split_k,
            )
        else:
            tiled_mma._launch_nonshuffled(
                prepared.a,
                prepared.b,
                out,
                config=config,
                split_k=split_k,
            )

    return runner


def _make_hgemm_runner(prepared: PreparedInputs, kwargs: dict):
    weight = prepared.hgemm_preshuffled_b if kwargs["B_PRE_SHUFFLE"] else prepared.b

    def runner(out: torch.Tensor):
        hgemm.hgemm_splitk_(
            out,
            prepared.a,
            weight,
            shuffle_b=False,
            hgemm_kwargs=kwargs,
        )

    return runner


def _validate_output(shape: Shape, out: torch.Tensor, ref: torch.Tensor, *, split_k: int):
    tol = _get_tol(shape.dtype, split_k)
    maxdiff = (out - ref).abs().max().item()
    if not torch.allclose(out, ref, atol=tol, rtol=tol):
        raise AssertionError(f"correctness failed: maxdiff={maxdiff}, tol={tol}")
    return maxdiff, tol


def _validate_and_measure(
    shape: Shape,
    runner,
    ref: torch.Tensor,
    *,
    split_k: int,
    warmup: int,
    niters: int,
) -> float:
    out = torch.empty_like(ref)
    runner(out)
    torch.cuda.synchronize()

    maxdiff, tol = _validate_output(shape, out, ref, split_k=split_k)

    _, us = run_perftest(
        runner,
        out,
        num_warmup=warmup,
        num_iters=niters,
    )
    return float(us), float(maxdiff), float(tol)


def _search_tiled_best(
    shape: Shape,
    prepared: PreparedInputs,
    candidates,
    *,
    weight_layout: str,
    warmup: int,
    niters: int,
):
    results = []
    pbar = tqdm(candidates, desc="search tiled_mma", leave=False)
    for candidate in pbar:
        try:
            runner = _make_tiled_runner(prepared, candidate, weight_layout=weight_layout)
            us, maxdiff, tol = _validate_and_measure(
                shape,
                runner,
                prepared.ref,
                split_k=candidate["split_k"],
                warmup=warmup,
                niters=niters,
            )
        except Exception:
            continue
        result = BenchResult("tiled_mma", "search", dict(candidate), us, maxdiff, tol)
        results.append(result)
        best_us = min(item.us for item in results)
        pbar.set_postfix_str(f"best={best_us:.4f} us")
    pbar.close()

    if not results:
        raise RuntimeError(
            "no valid tiled_mma candidate found; "
            "try a larger --candidate-limit or a shape divisible by more tile candidates"
        )

    results.sort(key=lambda item: item.us)
    return results


def _search_hgemm_best(
    shape: Shape,
    prepared: PreparedInputs,
    candidates,
    *,
    weight_layout: str,
    warmup: int,
    niters: int,
):
    results = []
    pbar = tqdm(candidates, desc="search hgemm", leave=False)
    for candidate in pbar:
        kwargs = _make_hgemm_kwargs(candidate, weight_layout=weight_layout)
        try:
            runner = _make_hgemm_runner(prepared, kwargs)
            us, maxdiff, tol = _validate_and_measure(
                shape,
                runner,
                prepared.ref,
                split_k=kwargs["SPLIT_K"],
                warmup=warmup,
                niters=niters,
            )
        except Exception:
            continue
        result = BenchResult("hgemm", "search", kwargs, us, maxdiff, tol)
        results.append(result)
        best_us = min(item.us for item in results)
        pbar.set_postfix_str(f"best={best_us:.4f} us")
    pbar.close()

    if not results:
        raise RuntimeError(
            "no valid hgemm candidate found; "
            "try a larger --candidate-limit or a shape divisible by more tile candidates"
        )

    results.sort(key=lambda item: item.us)
    return results


def _benchmark_tiled_candidate(
    shape: Shape,
    prepared: PreparedInputs,
    candidate: dict,
    *,
    weight_layout: str,
    warmup: int,
    niters: int,
) -> BenchResult:
    runner = _make_tiled_runner(prepared, candidate, weight_layout=weight_layout)
    us, maxdiff, tol = _validate_and_measure(
        shape,
        runner,
        prepared.ref,
        split_k=candidate["split_k"],
        warmup=warmup,
        niters=niters,
    )
    return BenchResult("tiled_mma", "best", dict(candidate), us, maxdiff, tol)


def _benchmark_hgemm_kwargs(
    shape: Shape,
    prepared: PreparedInputs,
    kwargs: dict,
    *,
    label: str,
    warmup: int,
    niters: int,
) -> BenchResult:
    runner = _make_hgemm_runner(prepared, kwargs)
    us, maxdiff, tol = _validate_and_measure(
        shape,
        runner,
        prepared.ref,
        split_k=kwargs["SPLIT_K"],
        warmup=warmup,
        niters=niters,
    )
    return BenchResult("hgemm", label, dict(kwargs), us, maxdiff, tol)


def _print_top_results(title: str, shape: Shape, results, top_k: int):
    count = min(top_k, len(results))
    print(f"## {title} top {count}")
    for index, result in enumerate(results[:count], start=1):
        print(
            f"{index}. {result.us:.4f} us, {_tflops(shape, result.us):.2f} TFLOPS, "
            f"maxdiff={result.maxdiff}, tol={result.tol}, config={result.config}"
        )


def _print_head_to_head(shape: Shape, tiled_result: BenchResult, other_result: BenchResult):
    tiled_vs_other = other_result.us / tiled_result.us
    if tiled_result.us <= other_result.us:
        verdict = f"tiled_mma is faster by {tiled_vs_other:.3f}x"
    else:
        verdict = f"tiled_mma reaches {tiled_vs_other:.3f}x of {other_result.backend} {other_result.label}"

    print(
        f"{tiled_result.backend} {tiled_result.label}: {tiled_result.us:.4f} us, "
        f"{_tflops(shape, tiled_result.us):.2f} TFLOPS, "
        f"maxdiff={tiled_result.maxdiff}, tol={tiled_result.tol}"
    )
    print(
        f"{other_result.backend} {other_result.label}: {other_result.us:.4f} us, "
        f"{_tflops(shape, other_result.us):.2f} TFLOPS, "
        f"maxdiff={other_result.maxdiff}, tol={other_result.tol}"
    )
    print(verdict)


def main():
    parser = argparse.ArgumentParser(
        description="Search tiled_mma_gemm configs and compare against hgemm.",
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", choices=sorted(CLI_DTYPE_MAP), default="bf16")
    parser.add_argument("--weight-layout", choices=sorted(WEIGHT_LAYOUTS), default="non-shuffle")
    parser.add_argument("--search-warmup", type=int, default=1)
    parser.add_argument("--search-niters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--niters", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        help="Only search the first N generic candidates, useful for smoke tests.",
    )
    parser.add_argument(
        "--show-top-only",
        action="store_true",
        help="Print searched top-k tables without running final end-to-end best-vs-best rebench.",
    )
    args = parser.parse_args()
    if args.search_niters <= 1:
        raise ValueError("--search-niters must be > 1 for run_perftest")
    if args.niters <= 1:
        raise ValueError("--niters must be > 1 for run_perftest")

    shape = Shape(
        m=args.m,
        n=args.n,
        k=args.k,
        dtype=CLI_DTYPE_MAP[args.dtype],
    )
    candidates = _candidate_configs(limit=args.candidate_limit)
    prepared = _make_inputs(shape)

    print(
        f"compare tiled_mma vs hgemm with M={shape.m}, N={shape.n}, K={shape.k}, "
        f"dtype={_dtype_name(shape.dtype)}, weight_layout={args.weight_layout}, "
        f"candidate_count={len(candidates)}"
    )

    tiled_search_results = _search_tiled_best(
        shape,
        prepared,
        candidates,
        weight_layout=args.weight_layout,
        warmup=args.search_warmup,
        niters=args.search_niters,
    )
    _print_top_results("tiled_mma", shape, tiled_search_results, args.top_k)
    hgemm_search_results = _search_hgemm_best(
        shape,
        prepared,
        candidates,
        weight_layout=args.weight_layout,
        warmup=args.search_warmup,
        niters=args.search_niters,
    )
    _print_top_results("hgemm", shape, hgemm_search_results, args.top_k)

    if args.show_top_only:
        return

    tiled_best = _benchmark_tiled_candidate(
        shape,
        prepared,
        tiled_search_results[0].config,
        weight_layout=args.weight_layout,
        warmup=args.warmup,
        niters=args.niters,
    )
    hgemm_best = _benchmark_hgemm_kwargs(
        shape,
        prepared,
        hgemm_search_results[0].config,
        label="best",
        warmup=args.warmup,
        niters=args.niters,
    )
    print("## Tiled Vs hgemm Best")
    _print_head_to_head(shape, tiled_best, hgemm_best)


if __name__ == "__main__":
    main()
