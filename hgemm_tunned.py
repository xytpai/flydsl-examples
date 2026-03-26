import time
import json
import torch
import argparse
import functools
import itertools
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass
from flydsl.runtime.device import get_rocm_arch
gpu_arch = get_rocm_arch()

from hgemm import hgemm_, selections


@dataclass
class Args:
    dtype: torch.dtype
    m: int
    n: int
    k: int


@dataclass
class TunnedArgs:
    arch: str
    dtype: str
    m: int
    n: int
    k: int
    config: dict
    duration: float
    tflops: float


def create_inputs(args):
    a = torch.empty((args.m, args.k), dtype=args.dtype, device='cuda')
    a.uniform_(-1, 1)
    b = torch.empty((args.n, args.k), dtype=args.dtype, device='cuda')
    b.uniform_(-1, 1)
    return (a, b)


def create_outputs(args):
    c = torch.randn((args.m, args.n), dtype=args.dtype, device='cuda')
    return (c,)


def benchmark(args, hgemm_kwargs={}, warmup=5, niters=50):
    a, b = create_inputs(args)
    c = create_outputs(args)[0]
    c_ref = create_outputs(args)[0]
    F.linear(a, b, out=c_ref)
    for i in range(warmup):
        hgemm_(a, b, c, hgemm_kwargs=hgemm_kwargs)
    is_allclose = torch.allclose(c, c_ref, atol=1e-1, rtol=1e-1)
    assert is_allclose == True
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        for i in range(niters):
            hgemm_(a, b, c, hgemm_kwargs=hgemm_kwargs)
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    hgemm_durations = []
    for event in prof.events():
        if event.name.startswith("hgemm_"):
            hgemm_durations.append(event.device_time)
    duration = np.median(hgemm_durations)
    return duration


def tune_single(args):
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    best_duration = float(1e10)
    best_idx = 0
    pbar = tqdm(total=len(configs), desc=f"{args}")
    for i, config in enumerate(configs):
        try:
            dur = benchmark(args, hgemm_kwargs=config)
        except:
            dur = float(1e10)
        if dur < best_duration:
            best_duration = dur
            best_idx = i
        pbar.update(1)
    tflops = 2.0 * args.m * args.n * args.k / best_duration * 1e-6
    result = TunnedArgs(
        arch = gpu_arch,
        dtype = str(args.dtype),
        m = args.m,
        n = args.n,
        k = args.k,
        config = configs[best_idx],
        duration = best_duration,
        tflops = tflops
    )
    print(result, flush=True)
    return result


def tune_all(
    dtype,
    out_prefix,
    ms = [2, 4, 8, 16, 32, 64, 128, 256],
    ns = [384, 1024, 2048, 4096, 5120, 6144, 7168, 8192],
    ks = [384, 1024, 2048, 4096, 5120, 6144, 7168, 8192],
):
    results = []
    args = Args(dtype=dtype, m=0, n=0, k=0)
    # ms = ms[:2]
    # ns = ns[:1]
    # ks = ks[:1]
    for m in ms:
        for n in ns:
            for k in ks:
                args.m = m
                args.n = n
                args.k = k
                result = tune_single(args)
                results.append(result)
    with open(f"{out_prefix}.jsonl", "w", encoding="utf-8") as f:
        for item in results:
            item = vars(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Examples")
    parser.add_argument("--out", type=str, default='hgemm_tunned')
    parser.add_argument("--dtype", type=str, default='bf16')
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    tune_all(args.dtype, args.out)
