import json
import torch
import itertools
import argparse

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity
from pathlib import Path
from dataclasses import dataclass
from flydsl.runtime.device import get_rocm_arch

from kernels.hgemm_layout_gfx950 import hgemm, make_hgemm_gfx950_param

gpu_arch = get_rocm_arch()
base_dir = Path(__file__).resolve().parent
temp_dir = base_dir / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Args:
    dtype: torch.dtype
    m: int
    n: int
    k: int


@dataclass
class TunedArgs:
    arch: str
    dtype: str
    m: int
    n: int
    k: int
    config: dict
    duration: float
    tflops: float


def create_inputs(args):
    a = torch.empty((args.m, args.k), dtype=args.dtype, device="cuda")
    a.uniform_(-1, 1)
    b = torch.empty((args.n, args.k), dtype=args.dtype, device="cuda")
    b.uniform_(-1, 1)
    bias = torch.empty((args.n,), dtype=args.dtype, device="cuda")
    bias.uniform_(10, 20)
    return (a, b, bias)


def create_outputs(args):
    c = torch.randn((args.m, args.n), dtype=args.dtype, device="cuda")
    return (c,)


def tuning_benchmark(args, kwargs={}, niters=50):
    # correctness test
    a, b, bias = create_inputs(args)
    c = create_outputs(args)[0]
    c_ref = create_outputs(args)[0]
    F.linear(a, b, out=c_ref, bias=bias)
    hgemm(a, b, c, bias=bias, user_kwargs=kwargs)
    tol = float(args.k) / 2048 * 6e-1
    is_allclose = torch.allclose(c, c_ref, atol=tol, rtol=tol)
    assert is_allclose
    # performance bench
    inputs = [create_inputs(args) for i in range(niters)]
    outputs = [create_outputs(args) for i in range(niters)]
    with profile(
        activities=[ProfilerActivity.CUDA],
    ) as prof:
        for i in range(niters):
            hgemm(
                inputs[i][0],
                inputs[i][1],
                outputs[i][0],
                bias=inputs[i][2],
                user_kwargs=kwargs,
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    # table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    hgemm_durations = []
    for event in prof.events():
        if event.name.startswith("hgemm_"):
            hgemm_durations.append(event.device_time)
    duration = np.median(hgemm_durations)
    return duration


def hgemm_get_configs():
    selections = {
        "block_m": [16, 32, 48, 64, 80, 96, 128, 256],
        "block_n": [64, 80, 96, 128, 256],
        "block_k": [64, 128, 256],
        "stages": [i for i in range(2, 10)],
        "m_waves": [1, 2, 4],
        "n_waves": [1, 2, 4],
        "group_m": [0, 4],
        "use_half_tile_interleaved": [False, True],
    }
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    valid_configs = []
    for config in configs:
        if not config["use_half_tile_interleaved"]:
            mma_m_iters = config["block_m"] // config["m_waves"] // 16
            mma_n_iters = config["block_n"] // config["n_waves"] // 16
            if mma_m_iters > 4 or mma_n_iters > 4:
                continue
        try:
            make_hgemm_gfx950_param(**config)
            valid_configs.append(config)
        except Exception:
            pass
    return valid_configs


def tune_single(args):
    configs = hgemm_get_configs()
    best_duration = float(1e10)
    best_idx = 0
    pbar = tqdm(total=len(configs), desc=f"{args}")
    for i, config in enumerate(configs):
        try:
            dur = tuning_benchmark(args, kwargs=config)
        except Exception:
            dur = float(1e10)
        if dur < best_duration:
            best_duration = dur
            best_idx = i
        pbar.update(1)
    tflops = 2.0 * args.m * args.n * args.k / best_duration * 1e-6
    result = TunedArgs(
        arch=gpu_arch,
        dtype=str(args.dtype),
        m=args.m,
        n=args.n,
        k=args.k,
        config=configs[best_idx],
        duration=best_duration,
        tflops=tflops,
    )
    pbar.close()
    print(result, flush=True)
    return result


def tune_all(
    dtype,
    out_prefix,
):
    mnks = [
        (8, 4096, 4096),
        (16, 4096, 4096),
        (32, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
        (256, 4096, 4096),
        (512, 4096, 4096),
        (2048, 4096, 4096),
        (32, 14336, 4096),
        (16, 28672, 4096),
        (4096, 256, 4096),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 4096, 8192),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
        (8, 7168, 2048),
        (8, 7168, 2048),
        (32, 384, 7168),
    ]
    with open(f"{out_prefix}.jsonl", "w", encoding="utf-8") as f:
        for mnk in mnks:
            args = Args(dtype=dtype, m=mnk[0], n=mnk[1], k=mnk[2])
            result = tune_single(args)
            result = vars(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examples")
    parser.add_argument("--out", type=str, default="temp/hgemm_tuned")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--tune_all", action="store_true")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {"f16": torch.half, "bf16": torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    if args.single:
        tune_single(args)
    elif args.tune_all:
        tune_all(args.dtype, args.out)

    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 1024 --n 1024 --k 1024
    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 2048 --n 2048 --k 2048
    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 4096 --n 4096 --k 4096
    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 4096 --n 4096 --k 8192

    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 8192 --n 8192 --k 8192

    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 8 --n 7168 --k 2048
    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 4096 --n 256 --k 4096
    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 32 --n 384 --k 7168

    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --tune_all
