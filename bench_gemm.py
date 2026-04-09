"""Benchmark and tune FlyDSL GEMM kernels.

This script supports two workflows:
1. Benchmark a single GEMM shape against the `tgemm.mm` reference path.
2. Tune a CSV of GEMM shapes and emit only the shapes where FlyDSL wins.

The CSV tuning path currently assumes `bias=False`, `scaleAB=False`, and the
non-preshuffled-B wrapper path (`bpreshuffle=False`).
"""

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

from aiter.ops.flydsl.gemm_kernels import flydsl_kernel_name
from aiter.test_common import checkAllclose, run_perftest
from aiter.tuned_gemm import tgemm
from flydsl.runtime.device import get_rocm_arch
from gemm_kernal import flydsl_hgemm


@dataclass
class Args:
    dtype: torch.dtype
    m: int
    n: int
    k: int


CLI_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
}


FIXED_STAGE = 2
FIXED_B_TO_LDS = False
FIXED_C_TO_LDS = False
DEFAULT_BLOCK_M_WARPS = 1
DEFAULT_BLOCK_N_WARPS = 4
DEFAULT_B_PRESHUFFLE = False
GPU_ARCH = get_rocm_arch()
KERNEL_ASYNC_COPY = GPU_ARCH != "gfx942"

CONFIG_SELECTIONS = {
    "tile_k": [64, 128],
    "tile_m": [16, 32, 48, 64, 96, 128],
    "tile_n": [64, 128, 256],
    "split_k": [1, 2, 4, 8, 16],
}

TUNING_CONFIG_VARIANTS = (
    {
        "block_m_warps": DEFAULT_BLOCK_M_WARPS,
        "block_n_warps": DEFAULT_BLOCK_N_WARPS,
        "b_to_lds": FIXED_B_TO_LDS,
    },
    {
        "block_m_warps": 2,
        "block_n_warps": 2,
        "b_to_lds": True,
    },
)


def _kernel_uses_async_copy() -> bool:
    return KERNEL_ASYNC_COPY


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_config(config=None, *, default_b_preshuffle=DEFAULT_B_PRESHUFFLE):
    config = {} if config is None else dict(config)

    requested_stage = int(config.get("stages", FIXED_STAGE))
    if requested_stage not in (1, FIXED_STAGE):
        raise ValueError(
            f"Unsupported stages={requested_stage}; current `hgemm.py` only supports stage=2"
        )

    normalized = {
        "tile_m": int(config.get("tile_m", 128)),
        "tile_n": int(config.get("tile_n", 128)),
        "tile_k": int(config.get("tile_k", 64)),
        "split_k": int(config.get("split_k", 1)),
        "stages": FIXED_STAGE,
        "async_copy": _kernel_uses_async_copy(),
        "block_m_warps": int(config.get("block_m_warps", DEFAULT_BLOCK_M_WARPS)),
        "block_n_warps": int(config.get("block_n_warps", DEFAULT_BLOCK_N_WARPS)),
        "b_to_lds": _as_bool(config.get("b_to_lds", FIXED_B_TO_LDS)),
        "b_preshuffle": _as_bool(config.get("b_preshuffle", default_b_preshuffle)),
        "c_to_lds": _as_bool(config.get("c_to_lds", FIXED_C_TO_LDS)),
    }
    return normalized

CSV_COLUMNS = [
    "cu_num",
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
    "libtype",
    "solidx",
    "splitK",
    "us",
    "kernelName",
    "err_ratio",
    "tflops",
    "bw",
]


def _candidate_configs():
    keys = list(CONFIG_SELECTIONS.keys())
    values = [CONFIG_SELECTIONS[key] for key in keys]
    candidate_configs = []
    seen_configs = set()
    for combo in itertools.product(*values):
        base_config = dict(zip(keys, combo))
        for variant in TUNING_CONFIG_VARIANTS:
            config = _normalize_config({**base_config, **variant})
            config_key = tuple(sorted(config.items()))
            if config_key in seen_configs:
                continue
            seen_configs.add(config_key)
            candidate_configs.append(config)
    return candidate_configs


CANDIDATE_CONFIGS = _candidate_configs()


HELP_EPILOG = """Examples:
  Benchmark one shape:
    python3 bench_gemm.py --m 4096 --n 4096 --k 4096 --dtype bf16

  Search the best FlyDSL config for one shape, then benchmark it:
    python3 bench_gemm.py --m 4096 --n 4096 --k 4096 --dtype bf16 --search-best

  Tune a CSV and save only FlyDSL winners:
    python3 bench_gemm.py --input-csv /workdir/aiter/aiter/configs/model_configs/test/kimik2_untuned_gemm_bf16.csv \\
      --output-csv /workdir/flydsl-examples-xiaobing/kimik2_flydsl_bf16_tuned_gemm.csv \\
      --search-warmup 5 --search-niters 50 --warmup 20 --niters 100

  Smoke-test the first 10 shapes from a CSV:
    python3 bench_gemm.py --input-csv /path/to/shapes.csv --limit 10

Notes:
  - `--input-csv` mode only writes shapes where FlyDSL is faster than `tgemm.mm`.
  - `--output-csv` defaults to a name derived from `--input-csv`.
  - Use `ROCR_VISIBLE_DEVICES=<gpu_id>` to pin a run to one GPU.
"""


def create_inputs(args):
    a = torch.empty((args.m, args.k), dtype=args.dtype, device="cuda")
    a.uniform_(-1, 1)
    b = torch.empty((args.n, args.k), dtype=args.dtype, device="cuda")
    b.uniform_(-1, 1)
    return a, b


def create_outputs(args):
    return torch.empty((args.m, args.n), dtype=args.dtype, device="cuda")


def ref_func(a, b):
    return tgemm.mm(a, b, None, otype=a.dtype)


def func(a, b, c, config=None):
    config = _normalize_config(
        config,
        default_b_preshuffle=getattr(b, "is_shuffled", False),
    )
    return flydsl_hgemm(
        a,
        b,
        c,
        tile_m=config["tile_m"],
        tile_n=config["tile_n"],
        tile_k=config["tile_k"],
        split_k=config["split_k"],
        block_m_warps=config["block_m_warps"],
        block_n_warps=config["block_n_warps"],
        stages=config["stages"],
        async_copy=config["async_copy"],
        b_to_lds=config["b_to_lds"],
        b_preshuffle=config["b_preshuffle"],
        auto_shuffle_b=config["b_preshuffle"],
        c_to_lds=config["c_to_lds"],
    )


def _get_tol(k: int) -> float:
    return float(k) / 2048 * 6e-1


def _measure_flydsl_duration(a, b, c, c_ref, config=None, warmup=5, niters=50):
    config = {} if config is None else dict(config)
    for _ in range(warmup):
        func(a, b, c, config=config)
    torch.cuda.synchronize()
    tol = _get_tol(a.shape[-1])
    if not torch.allclose(c, c_ref, atol=tol, rtol=tol):
        maxdiff = (c - c_ref).abs().max().item()
        raise AssertionError(
            f"config {config} failed correctness check: maxdiff={maxdiff}, tol={tol}"
        )
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(niters):
            func(a, b, c, config=config)
    hgemm_durations = [
        event.device_time
        for event in prof.events()
        if event.name.startswith("hgemm_")
    ]
    if not hgemm_durations:
        raise RuntimeError(f"no hgemm profiler events captured for config {config}")
    return float(np.median(hgemm_durations))


def _create_tuning_context(args):
    a, b = create_inputs(args)
    c = create_outputs(args)
    c_ref = F.linear(a, b)
    return a, b, c, c_ref


def _config_to_kernel_name(config, dtype: str = "bf16", out_dtype: str = "bf16"):
    config = _normalize_config(config)
    return flydsl_kernel_name(
        stage=config["stages"],
        dtype=dtype,
        out_dtype=out_dtype,
        tile_m=config["tile_m"],
        tile_n=config["tile_n"],
        tile_k=config["tile_k"],
        split_k=config["split_k"],
        block_m_warp=config["block_m_warps"],
        block_n_warp=config["block_n_warps"],
        async_copy=config["async_copy"],
        b_to_lds=config["b_to_lds"],
        b_preshuffle=config["b_preshuffle"],
        c_to_lds=config["c_to_lds"],
    )


KERNEL_SOLIDX_MAP = {
    _config_to_kernel_name(config): idx for idx, config in enumerate(CANDIDATE_CONFIGS)
}


def _config_to_solidx(config):
    return KERNEL_SOLIDX_MAP[_config_to_kernel_name(config)]


def _dtype_to_csv_string(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "torch.bfloat16"
    if dtype == torch.float16:
        return "torch.float16"
    raise ValueError(f"unsupported dtype: {dtype!r}")


def _parse_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "bf16": torch.bfloat16,
        "f16": torch.float16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"unsupported dtype string: {dtype_str!r}")
    return mapping[dtype_str]


def _parse_bool(value):
    return _as_bool(value)


def _calculate_perf(m, n, k, us, dtype, out_dtype):
    if us <= 0:
        return -1.0, -1.0
    inbpe = torch.empty((), dtype=dtype).element_size()
    outbpe = torch.empty((), dtype=out_dtype).element_size()
    flops = m * n * k * 2
    tflops = round(flops / (us * 1_000_000), 2)
    bw = round(
        (m * k * inbpe + n * k * inbpe + m * n * outbpe) / (us * 1e-6) / 1e9,
        2,
    )
    return tflops, bw


def _measure_ref_duration(a, b, warmup=20, niters=100):
    _, ref_us = run_perftest(
        ref_func,
        a,
        b,
        num_warmup=warmup,
        num_iters=niters,
    )
    return float(ref_us)


def _measure_end_to_end_flydsl_duration(a, b, config=None, warmup=20, niters=100):
    _, flydsl_us = run_perftest(
        func,
        a,
        b,
        None,
        config=config,
        num_warmup=warmup,
        num_iters=niters,
    )
    return float(flydsl_us)


def _benchmark_shape(args, config, warmup=20, niters=100):
    a, b = create_inputs(args)
    ref_output = F.linear(a, b)
    output = func(a, b, None, config=config)
    tol = _get_tol(args.k)
    err_ratio = checkAllclose(
        ref_output,
        output,
        rtol=tol,
        atol=tol,
        tol_err_ratio=1.0,
        printLog=False,
    )
    flydsl_us = _measure_end_to_end_flydsl_duration(
        a,
        b,
        config=config,
        warmup=warmup,
        niters=niters,
    )
    ref_us = _measure_ref_duration(a, b, warmup=warmup, niters=niters)
    return flydsl_us, ref_us, float(err_ratio)


def _make_output_row(args, config, flydsl_us, err_ratio):
    config = _normalize_config(config)
    kernel_name = _config_to_kernel_name(config)
    tflops, bw = _calculate_perf(args.m, args.n, args.k, flydsl_us, args.dtype, args.dtype)
    return {
        "cu_num": torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "bias": False,
        "dtype": _dtype_to_csv_string(args.dtype),
        "outdtype": _dtype_to_csv_string(args.dtype),
        "scaleAB": False,
        "bpreshuffle": config["b_preshuffle"],
        "libtype": "flydsl",
        "solidx": _config_to_solidx(config),
        "splitK": config["split_k"],
        "us": round(flydsl_us, 4),
        "kernelName": kernel_name,
        "err_ratio": round(err_ratio, 4),
        "tflops": tflops,
        "bw": bw,
    }


def _default_output_csv(input_csv: str) -> str:
    path = Path(input_csv)
    name = path.name
    if name.endswith("_untuned_gemm_bf16.csv"):
        out_name = name.replace("_untuned_gemm_bf16.csv", "_flydsl_bf16_tuned_gemm.csv")
    else:
        out_name = f"{path.stem}_flydsl_tuned_gemm.csv"
    return str(path.with_name(out_name))


def tune_shapes_from_csv(
    input_csv: str,
    output_csv: str,
    *,
    search_warmup: int = 5,
    search_niters: int = 50,
    warmup: int = 20,
    niters: int = 100,
    limit: int | None = None,
):
    with open(input_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        shape_pbar = tqdm(rows, desc=f"tune->{output_path.name}")
        for row in shape_pbar:
            args = Args(
                dtype=_parse_torch_dtype(row["dtype"]),
                m=int(row["M"]),
                n=int(row["N"]),
                k=int(row["K"]),
            )
            shape_pbar.set_postfix_str(f"M={args.m},N={args.n},K={args.k}")
            if _parse_bool(row.get("bias", False)) or _parse_bool(row.get("scaleAB", False)):
                continue
            try:
                best_config, _ = get_best_config(
                    args,
                    warmup=search_warmup,
                    niters=search_niters,
                    leave=False,
                )
                flydsl_us, ref_us, err_ratio = _benchmark_shape(
                    args,
                    best_config,
                    warmup=warmup,
                    niters=niters,
                )
            except Exception as exc:
                print(
                    f"skip shape M={args.m}, N={args.n}, K={args.k}: {exc}",
                    flush=True,
                )
                continue
            if flydsl_us < ref_us:
                writer.writerow(_make_output_row(args, best_config, flydsl_us, err_ratio))
                f.flush()
                saved += 1
        shape_pbar.close()
    print(f"saved {saved} FlyDSL winners to {output_path}", flush=True)


def get_best_config(args, warmup=5, niters=50, leave=True):
    best_config = None
    best_time = float("inf")
    a, b, c, c_ref = _create_tuning_context(args)
    total_configs = len(CANDIDATE_CONFIGS)
    pbar = tqdm(total=total_configs, desc=f"{args}", leave=leave)
    try:
        for config in CANDIDATE_CONFIGS:
            try:
                duration = _measure_flydsl_duration(
                    a,
                    b,
                    c,
                    c_ref,
                    config=config,
                    warmup=warmup,
                    niters=niters,
                )
            except Exception:
                pbar.update(1)
                continue
            if duration < best_time:
                best_time = duration
                best_config = config
            pbar.update(1)
    finally:
        pbar.close()
    if best_config is None:
        raise RuntimeError("no valid FlyDSL config found")
    return best_config, best_time


def benchmark(args, func, ref_func, config=None, warmup=20, niters=100):
    a, b = create_inputs(args)
    output = func(a, b, None, config=config)
    ref_output = F.linear(a, b)
    maxdiff_out = (output - ref_output).abs().max().item()
    tol = _get_tol(args.k)
    print(f"config: {config or {}}")
    print(f"maxdiff_out: {maxdiff_out}")
    assert torch.allclose(output, ref_output, atol=tol, rtol=tol)

    print("===================== [FLYDSL] =====================")
    _, flydsl_us = run_perftest(
        func,
        a,
        b,
        None,
        config=config,
        num_warmup=warmup,
        num_iters=niters,
    )
    print(f"avg: {flydsl_us:.4f} us/iter")

    print("===================== [REF] =====================")
    _, ref_us = run_perftest(
        ref_func,
        a,
        b,
        num_warmup=warmup,
        num_iters=niters,
    )
    print(f"avg: {ref_us:.4f} us/iter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark or tune FlyDSL GEMM kernels against the tgemm.mm reference.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--m",
        type=int,
        help="Rows of matrix A / output C in single-shape mode.",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Rows of matrix B / columns of output C in single-shape mode.",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Reduction dimension in single-shape mode.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=sorted(CLI_DTYPE_MAP),
        help="Input/output dtype for single-shape mode.",
    )
    parser.add_argument(
        "--search-best",
        action="store_true",
        help="Search the FlyDSL candidate config space before benchmarking one shape.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations for the final end-to-end benchmark.",
    )
    parser.add_argument(
        "--niters",
        type=int,
        default=100,
        help="Measured iterations for the final end-to-end benchmark.",
    )
    parser.add_argument(
        "--search-warmup",
        type=int,
        default=5,
        help="Warmup iterations used while searching candidate FlyDSL configs.",
    )
    parser.add_argument(
        "--search-niters",
        type=int,
        default=50,
        help="Measured iterations used while searching candidate FlyDSL configs.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        help="Input CSV of GEMM shapes. When set, the script runs in batch tuning mode.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Output CSV path for batch tuning mode. Defaults to a name derived from the input CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N rows from --input-csv, useful for smoke tests.",
    )
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    warmup = args.warmup
    niters = args.niters
    search_best = args.search_best
    if args.input_csv:
        output_csv = args.output_csv or _default_output_csv(args.input_csv)
        tune_shapes_from_csv(
            args.input_csv,
            output_csv,
            search_warmup=args.search_warmup,
            search_niters=args.search_niters,
            warmup=warmup,
            niters=niters,
            limit=args.limit,
        )
    else:
        if args.m is None or args.n is None or args.k is None:
            parser.error("--m/--n/--k are required when --input-csv is not provided")
        run_args = Args(
            dtype=CLI_DTYPE_MAP[args.dtype],
            m=args.m,
            n=args.n,
            k=args.k,
        )

        best_config = None
        if search_best:
            best_config, best_us = get_best_config(
                run_args,
                warmup=args.search_warmup,
                niters=args.search_niters,
            )
            print(f"best_config: {best_config}, best_us: {best_us:.4f}")

        benchmark(
            run_args,
            func,
            ref_func,
            config=best_config,
            warmup=warmup,
            niters=niters,
        )