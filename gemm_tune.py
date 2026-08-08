import os

os.environ.setdefault("KINETO_LOG_LEVEL", "6")

import json
import torch
import itertools
import argparse
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from torch.profiler import profile, ProfilerActivity
from pathlib import Path
from dataclasses import dataclass
from flydsl.runtime.device import get_rocm_arch

from kernels.hgemm_universal_gfx950 import (
    HGEMM_DTYPE_BF16,
    HGEMM_DTYPE_FP16,
    hgemm,
    make_hgemm_param_and_validate,
)

DEFAULT_COMPILE_WORKERS = 32
_COMPILE_TENSORS = None
_COMPILE_LAYOUT = "nt"

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
    layout: str
    enable_split_k: bool = False
    compile_workers: int = DEFAULT_COMPILE_WORKERS


@dataclass
class TunedArgs:
    arch: str
    dtype: str
    layout: str
    m: int
    n: int
    k: int
    config: dict
    duration: float
    tflops: float


@dataclass(frozen=True)
class GemmTileIoUPruner:
    m: int
    n: int
    k: int
    keep_ratio: float

    @staticmethod
    def _ceil_div(value, divisor):
        return (value + divisor - 1) // divisor

    def _split_k_padded(self, block_k, split_k):
        working_k = self._ceil_div(self.k, split_k)
        padded_k = 0
        for split_idx in range(split_k):
            remaining_k = max(self.k - split_idx * working_k, 0)
            part_k = min(working_k, remaining_k)
            if part_k > 0:
                padded_k += self._ceil_div(part_k, block_k) * block_k
        return padded_k

    def _config_iou(self, config):
        padded_m = self._ceil_div(self.m, config["block_m"]) * config["block_m"]
        padded_n = self._ceil_div(self.n, config["block_n"]) * config["block_n"]
        padded_k = self._split_k_padded(config["block_k"], config["split_k"])
        return (self.m * self.n * self.k) / (padded_m * padded_n * padded_k)

    def prune(self, configs):
        if not configs:
            return configs
        config_ious = [self._config_iou(config) for config in configs]
        threshold = max(config_ious) * self.keep_ratio
        return [config for config, iou in zip(configs, config_ious) if iou >= threshold]


def empty_layout_matrix(rows, cols, dtype, is_t, device="cuda"):
    if is_t:
        return torch.empty((cols, rows), dtype=dtype, device=device).t()
    return torch.empty((rows, cols), dtype=dtype, device=device)


def create_inputs(args):
    a = empty_layout_matrix(
        args.m,
        args.k,
        args.dtype,
        args.layout[0] == "t",
    )
    a.uniform_(-1, 1)
    b = empty_layout_matrix(
        args.k,
        args.n,
        args.dtype,
        args.layout[1] == "t",
    )
    b.uniform_(-1, 1)
    bias = torch.empty((args.n,), dtype=args.dtype, device="cuda")
    bias.uniform_(10, 20)
    return (a, b, bias)


def create_outputs(args):
    c = torch.randn((args.m, args.n), dtype=args.dtype, device="cuda")
    return (c,)


def _init_compile_worker(dtype, m, n, k, layout):
    global _COMPILE_TENSORS, _COMPILE_LAYOUT
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    _COMPILE_TENSORS = (
        empty_layout_matrix(m, k, dtype, layout[0] == "t"),
        empty_layout_matrix(k, n, dtype, layout[1] == "t"),
        torch.empty((m, n), dtype=dtype, device="cuda"),
        torch.empty(n, dtype=dtype, device="cuda"),
    )
    _COMPILE_LAYOUT = layout


def _compile_policy_worker(job):
    index, config = job
    try:
        assert _COMPILE_TENSORS is not None
        a, b, out, bias = _COMPILE_TENSORS
        hgemm(
            a,
            b,
            out,
            bias=bias,
            user_kwargs=config,
            layout=_COMPILE_LAYOUT,
        )
        torch.cuda.synchronize()
        return index, None
    except Exception as error:
        return index, str(error)


def parallel_compile_policies(args, configs, max_workers=DEFAULT_COMPILE_WORKERS):
    if not configs:
        return []

    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"
    os.environ.setdefault("ARCH", gpu_arch)
    workers = max(1, min(max_workers or DEFAULT_COMPILE_WORKERS, len(configs)))
    jobs = list(enumerate(configs))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_compile_worker,
        initargs=(args.dtype, args.m, args.n, args.k, args.layout),
    ) as executor:
        results = list(
            tqdm(
                executor.map(_compile_policy_worker, jobs),
                total=len(jobs),
                desc=f"compile policies ({workers} workers)",
            )
        )

    failures = {index: error for index, error in results if error}
    for index, error in list(failures.items())[:3]:
        print(f"compile failed: {configs[index]}: {error}", flush=True)
    return [config for i, config in enumerate(configs) if i not in failures]


def tuning_benchmark(args, kwargs={}, niters=50):
    # correctness test
    a, b, bias = create_inputs(args)
    c = create_outputs(args)[0]
    c_ref = create_outputs(args)[0]
    torch.addmm(bias, a, b, out=c_ref)
    hgemm(a, b, c, bias=bias, user_kwargs=kwargs, layout=args.layout)
    tol = (
        float(args.k)
        / 2048
        * 6e-1
        * kwargs.get("split_k", 1)
        * kwargs.get("k_waves", 1)
    )
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
                layout=args.layout,
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


def hgemm_get_configs(args):
    split_k_candidates = [1]
    if args.enable_split_k:
        split_k_candidates.extend(
            split_k for split_k in range(2, 10) if args.k % split_k == 0
        )
    selections = {
        "block_m": [16, 32, 48, 64, 80, 96, 128, 256],
        "block_n": [16, 32, 64, 80, 96, 128, 256],
        "block_k": [64, 128, 256],
        "stages": [i for i in range(2, 10)],
        "split_k": split_k_candidates,
        "m_waves": [1, 2, 4],
        "n_waves": [1, 2, 4],
        "k_waves": [1, 2],
        "group_m": [0, 4],
        "use_half_tile_interleaved": [False, True],
    }
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    keep_ratio = 0.75 if args.m <= 32 else 0.85 if args.m <= 128 else 0.95
    configs = GemmTileIoUPruner(
        args.m,
        args.n,
        args.k,
        keep_ratio,
    ).prune(configs)
    valid_configs = []
    is_large_gemm = args.m >= 4096 and args.n >= 4096 and args.k >= 4096
    in_dtype_id = HGEMM_DTYPE_FP16 if args.dtype is torch.float16 else HGEMM_DTYPE_BF16
    for config in configs:
        if is_large_gemm:
            if not (
                config["use_half_tile_interleaved"]
                and config["block_m"] == 256
                and config["block_n"] == 256
                and config["block_k"] == 64
                and config["stages"] == 2
                and config["split_k"] == 1
                and config["m_waves"] == 2
                and config["n_waves"] == 4
                and config["k_waves"] == 1
            ):
                continue
        else:
            if not config["use_half_tile_interleaved"]:
                mma_m_iters = config["block_m"] // config["m_waves"] // 16
                mma_n_iters = config["block_n"] // config["n_waves"] // 16
                if mma_m_iters > 4 or mma_n_iters > 4:
                    continue
        try:
            validation_config = {
                **config,
                "in_dtype_id": in_dtype_id,
                "out_dtype_id": in_dtype_id,
            }
            param = make_hgemm_param_and_validate(
                args.m,
                args.n,
                args.k,
                validation_config,
            )
            if param is not None:
                valid_configs.append(config)
        except Exception:
            pass
    return valid_configs


def tune_single(args):
    configs = hgemm_get_configs(args)
    configs = parallel_compile_policies(
        args,
        configs,
        max_workers=args.compile_workers,
    )
    if not configs:
        raise RuntimeError(f"no policies compiled successfully for {args}")
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
        layout=args.layout,
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
    layout,
    enable_split_k=False,
    compile_workers=DEFAULT_COMPILE_WORKERS,
):
    mnks = [
        # splitk-regresiion-test
        # (1, 5120, 2880),     # AITER FlyDSL: 7.7863 us
        # (2, 5120, 2880),     # AITER FlyDSL: 7.9572 us
        # (4, 5120, 2880),     # AITER FlyDSL: 7.9943 us
        # (8, 5120, 2880),     # AITER FlyDSL: 8.1986 us
        # (16, 5120, 2880),    # AITER FlyDSL: 8.6447 us
        # (32, 5120, 2880),    # AITER FlyDSL: 10.4444 us
        # (48, 5120, 2880),    # AITER FlyDSL: 12.2810 us
        # splitk
        # (32, 384, 7168),
        # (32, 384, 16384),
        # (800, 384, 7168),
        # (32, 7168, 2048),
        # (8, 7168, 2048),
        # (32, 2880, 2048),
        # normal
        (8, 4096, 4096),
        (16, 4096, 4096),
        (32, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
        (256, 4096, 4096),
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (2048, 4096, 4096),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 4096, 8192),
        (8192, 8192, 8192),
        (8, 7168, 2048),
        (32, 384, 7168),
        (32, 14336, 4096),
        (16, 28672, 4096),
        (4096, 256, 4096),
    ]
    with open(f"{out_prefix}.jsonl", "w", encoding="utf-8") as f:
        for mnk in mnks:
            args = Args(
                dtype=dtype,
                m=mnk[0],
                n=mnk[1],
                k=mnk[2],
                layout=layout,
                enable_split_k=enable_split_k,
                compile_workers=compile_workers,
            )
            result = tune_single(args)
            result = vars(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examples")
    parser.add_argument("--out", type=str, default="temp/hgemm_tuned")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument(
        "--layout",
        choices=("nn", "nt", "tn", "tt"),
        default="nt",
    )
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--tune_all", action="store_true")
    parser.add_argument(
        "--enable_split_k",
        "--enable-split-k",
        action="store_true",
        help="include valid split_k values greater than 1 in tuning",
    )
    parser.add_argument(
        "--compile-workers",
        type=int,
        default=DEFAULT_COMPILE_WORKERS,
        help="parallel policy compiler processes (default: 32)",
    )
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
        tune_all(
            args.dtype,
            args.out,
            args.layout,
            enable_split_k=args.enable_split_k,
            compile_workers=args.compile_workers,
        )

    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 1024 --n 1024 --k 1024
    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 2048 --n 2048 --k 2048
    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 4096 --n 4096 --k 4096
    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 4096 --n 4096 --k 8192

    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 8192 --n 8192 --k 8192

    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 8 --n 7168 --k 2048
    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 4096 --n 256 --k 4096
    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --single --dtype bf16 --m 32 --n 384 --k 7168

    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --tune_all
    # rm -rf ~/.flydsl/ ; python3 gemm_tune.py --tune_all --enable_split_k
