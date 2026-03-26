import time
import json
import torch
import argparse
import functools
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass
from flydsl.runtime.device import get_rocm_arch
gpu_arch = get_rocm_arch()
base_dir = Path(__file__).resolve().parent
temp_dir = base_dir / 'temp'
temp_dir.mkdir(parents=True, exist_ok=True)

from hgemm import hgemm_, selections, benchmark, ref_func


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
    a = torch.empty((args.m, args.k), dtype=args.dtype, device='cuda')
    a.uniform_(-1, 1)
    b = torch.empty((args.n, args.k), dtype=args.dtype, device='cuda')
    b.uniform_(-1, 1)
    return (a, b)


def create_outputs(args):
    c = torch.randn((args.m, args.n), dtype=args.dtype, device='cuda')
    return (c,)


def tuning_benchmark(args, hgemm_kwargs={}, warmup=5, niters=50):
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
            dur = tuning_benchmark(args, hgemm_kwargs=config)
        except:
            dur = float(1e10)
        if dur < best_duration:
            best_duration = dur
            best_idx = i
        pbar.update(1)
    tflops = 2.0 * args.m * args.n * args.k / best_duration * 1e-6
    result = TunedArgs(
        arch = gpu_arch,
        dtype = str(args.dtype),
        m = args.m,
        n = args.n,
        k = args.k,
        config = configs[best_idx],
        duration = best_duration,
        tflops = tflops
    )
    pbar.close()
    print(result, flush=True)
    return result


def tune_all(
    dtype,
    out_prefix,
    ms = [2, 4, 8, 16, 24, 32, 48, 64, 72, 128, 256, 384, 448, 512],
    ns = [384, 512, 1024, 2048, 2112, 3072, 4096, 5120, 6144, 7168, 8192],
    ks = [384, 512, 1024, 1536, 2048, 4096, 5120, 6144, 7168, 8192],
):
    args = Args(dtype=dtype, m=0, n=0, k=0)
    # ms = ms[:2]
    # ns = ns[:2]
    # ks = ks[:1]
    for m in ms:
        with open(f"{out_prefix}_m_{m}.jsonl", "w", encoding="utf-8") as f:
            for n in ns:
                for k in ks:
                    args.m = m
                    args.n = n
                    args.k = k
                    result = tune_single(args)
                    result = vars(result)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()


TUNED_CONFIGS = None
MAP_CONFIGS = {}
SHOW_TUNED_LOG = False
def hgemm_tuned(a, b, c):
    global TUNED_CONFIGS
    global MAP_CONFIGS
    global SHOW_TUNED_LOG
    dtype = a.dtype
    k = a.shape[-1]
    a = a.view(-1, k)
    m = a.shape[0]
    n = b.shape[0]
    if TUNED_CONFIGS is None:
        with open('hgemm_tuned.jsonl', 'r', encoding='utf-8') as f:
            TUNED_CONFIGS = [json.loads(line) for line in f]
            for line in TUNED_CONFIGS:
                key = (line['arch'], line['dtype'], line['m'], line['n'], line['k'])
                MAP_CONFIGS[key] = line['config']
    if MAP_CONFIGS.get((gpu_arch, str(dtype), m, n, k), None) is not None:
        config = MAP_CONFIGS[(gpu_arch, str(dtype), m, n, k)]
        if SHOW_TUNED_LOG:
            print(f"Found tuned config for m={m}, n={n}, k={k}: {config}")
    else:
        config = {}
    hgemm_(a, b, c, hgemm_kwargs=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Examples")
    parser.add_argument("--out", type=str, default='temp/hgemm_tuned')
    parser.add_argument("--dtype", type=str, default='bf16')
    parser.add_argument("--single", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--tune_all", action='store_true')
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    if args.single:
        tune_single(args)
    elif args.eval:
        args = Args(dtype=args.dtype, m=args.m, n=args.n, k=args.k)
        benchmark(args, hgemm_tuned, ref_func)
    elif args.tune_all:
        tune_all(args.dtype, args.out)
    # rm -rf ~/.flydsl/ ; python3 hgemm_tuned.py --single --dtype bf16 --m 4096 --n 4096 --k 4096
    # rm -rf ~/.flydsl/ ; python3 hgemm_tuned.py --eval --dtype bf16 --m 256 --n 8192 --k 384
    # rm -rf ~/.flydsl/ ; python3 hgemm_tuned.py --tune_all
