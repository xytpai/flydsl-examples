import argparse
import itertools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

from aiter.test_common import run_perftest
from aiter.tuned_gemm import tgemm
from gemm_kernal import flydsl_hgemm
from hgemm import hgemm_shuffle_b


@dataclass
class Args:
    dtype: torch.dtype
    m: int
    n: int
    k: int


CONFIG_SELECTIONS = {
    "tile_k": [64, 128],
    "tile_m": [16, 32, 48, 64, 96, 128],
    "tile_n": [64, 128, 256],
    "split_k": [1, 2, 4, 8],
    "stages": [1, 2],
    "async_copy": [False],
}


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
    config = {} if config is None else dict(config)
    return flydsl_hgemm(
        a,
        b,
        c,
        tile_m=config.get("tile_m", 128),
        tile_n=config.get("tile_n", 128),
        tile_k=config.get("tile_k", 64),
        split_k=config.get("split_k", 1),
        stages=config.get("stages", 2),
        async_copy=config.get("async_copy", False),
        b_preshuffle=getattr(b, "is_shuffled", False),
        auto_shuffle_b=False,
    )


def _measure_flydsl_duration(a, b_ref, b_flydsl, c, config=None, warmup=5, niters=50):
    config = {} if config is None else dict(config)
    c_ref = F.linear(a, b_ref)
    for _ in range(warmup):
        func(a, b_flydsl, c, config=config)
    torch.cuda.synchronize()
    tol = float(a.shape[-1]) / 2048 * 6e-1
    if not torch.allclose(c, c_ref, atol=tol, rtol=tol):
        maxdiff = (c - c_ref).abs().max().item()
        raise AssertionError(
            f"config {config} failed correctness check: maxdiff={maxdiff}, tol={tol}"
        )
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(niters):
            func(a, b_flydsl, c, config=config)
    hgemm_durations = [
        event.device_time
        for event in prof.events()
        if event.name.startswith("hgemm_")
    ]
    if not hgemm_durations:
        raise RuntimeError(f"no hgemm profiler events captured for config {config}")
    return float(np.median(hgemm_durations))


def tuning_benchmark(args, config=None, warmup=5, niters=50):
    a, b = create_inputs(args)
    c = create_outputs(args)
    return _measure_flydsl_duration(
        a,
        b,
        b,
        c,
        config=config,
        warmup=warmup,
        niters=niters,
    )


def get_best_config(args, warmup=5, niters=50):
    best_config = None
    best_time = float("inf")
    keys = list(CONFIG_SELECTIONS.keys())
    values = [CONFIG_SELECTIONS[key] for key in keys]
    total_configs = int(np.prod([len(v) for v in values], dtype=np.int64))
    pbar = tqdm(total=total_configs, desc=f"{args}")
    try:
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            try:
                duration = tuning_benchmark(
                    args,
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
    b_flydsl = hgemm_shuffle_b(b.clone())
    c = create_outputs(args)
    output = func(a, b_flydsl, c, config=config)
    ref_output = F.linear(a, b)
    maxdiff_out = (output - ref_output).abs().max().item()
    tol = float(args.k) / 2048 * 6e-1
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
    parser = argparse.ArgumentParser(description="Examples")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    parser.add_argument("--search-best", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--niters", type=int, default=100)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {"f16": torch.half, "bf16": torch.bfloat16}
    warmup = args.warmup
    niters = args.niters
    search_best = args.search_best
    run_args = Args(
        dtype=dtype_convert[args.dtype],
        m=args.m,
        n=args.n,
        k=args.k,
    )

    best_config = None
    if search_best:
        best_config, best_us = get_best_config(run_args)
        print(f"best_config: {best_config}, best_us: {best_us:.4f}")

    benchmark(
        run_args,
        func,
        ref_func,
        config=best_config,
        warmup=warmup,
        niters=niters,
    )