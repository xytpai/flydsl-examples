"""Benchmark and tune FlyDSL GEMM+TP-allreduce fusion kernels.

This follows the general workflow of `bench_gemm.py`, but benchmarks the
fused FlyDSL GEMM+all-reduce path against the unfused
`tgemm.mm + get_tp_group().all_reduce(...)` fallback.

The CSV tuning path assumes:
- `bias=False`
- `scaleAB=False`
- `bpreshuffle=False`
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata as importlib_metadata
import itertools
import math
from dataclasses import dataclass
from multiprocessing import Pool, freeze_support
from pathlib import Path
from typing import Any, Optional

_original_version = importlib_metadata.version


def _patched_version(package_name: str) -> str:
    if package_name == "flydsl":
        return "0.1.2"
    return _original_version(package_name)


importlib_metadata.version = _patched_version

import torch
import torch.distributed as dist
from tqdm import tqdm

from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.flydsl.gemm_kernels import (
    FIXED_C_TO_LDS,
    FIXED_STAGE,
    KERNEL_ASYNC_COPY,
    SPLIT_K_COUNTER_MAX_LEN,
    _to_kernel_dtype,
    _validate_hgemm_tiling,
)
from hgemm_ar_params import (
    TP_HGEMM_AR_CONFIG_SELECTIONS,
    TP_HGEMM_AR_CONFIG_VARIANTS,
    flydsl_tp_hgemm_ar_kernel_name,
)
from aiter.test_common import checkAllclose, ensure_spawn_method
from aiter.tuned_gemm import tgemm
from gemm_ar_kernal import flydsl_hgemm_ar

parallel_state = importlib.import_module("aiter.dist.parallel_state")


def _compile_hgemm_ar_kernel(*args, **kwargs):
    from hgemm_ar import compile_hgemm_ar_kernel

    return compile_hgemm_ar_kernel(*args, **kwargs)


@dataclass(frozen=True)
class Args:
    dtype: torch.dtype
    m: int
    n: int
    k: int
    world_size: int
    include_atomic_add: bool = False


CLI_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
}

HGEMM_AR_MAX_BLOCKS = 80

DEFAULT_BLOCK_M_WARPS = 2
DEFAULT_BLOCK_N_WARPS = 2
DEFAULT_B_TO_LDS = True
DEFAULT_B_PRESHUFFLE = False
SMALL_M_VALUES = {1, 2, 4, 8, 16, 32}
SMALL_M_TILE_K_OPTIONS = [32, 64, 128]

DEFAULT_CONFIG_SELECTIONS = {
    key: list(values) for key, values in TP_HGEMM_AR_CONFIG_SELECTIONS.items()
}
DEFAULT_TUNING_CONFIG_VARIANTS = tuple(
    dict(variant) for variant in TP_HGEMM_AR_CONFIG_VARIANTS
)

SEARCH_SKIP_SLOW_CANDIDATE_FACTOR = 1.5
SEARCH_SKIP_SLOW_CANDIDATE_SLACK_US = 50.0

CSV_COLUMNS = [
    "world_size",
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
    "kernelName",
    "us",
    "err_ratio",
    "tflops",
    "bw",
]

HELP_EPILOG = """Examples:
  Tune a CSV of GEMM+AR shapes on 2 GPUs:
    python3 bench_gemm_allreduce_fusion.py \\
      --tp 2 \\
      --input-csv /path/to/tp_gemm_ar_smoke_untuned_gemm_bf16.csv

  Tune a single shape:
    python3 bench_gemm_allreduce_fusion.py \\
      --tp 2 --m 32 --n 1024 --k 7168 --dtype bf16
"""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def _dtype_to_csv_string(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "torch.bfloat16"
    if dtype == torch.float16:
        return "torch.float16"
    raise ValueError(f"unsupported dtype: {dtype!r}")


def _normalize_config(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    config = {} if config is None else dict(config)
    split_k = int(config.get("split_k", 1))
    use_atomic_add = _as_bool(config.get("use_atomic_add", False)) or split_k > 1
    return {
        "tile_m": int(config.get("tile_m", 128)),
        "tile_n": int(config.get("tile_n", 128)),
        "tile_k": int(config.get("tile_k", 64)),
        "split_k": split_k,
        "stages": FIXED_STAGE,
        "async_copy": KERNEL_ASYNC_COPY,
        "block_m_warps": int(config.get("block_m_warps", DEFAULT_BLOCK_M_WARPS)),
        "block_n_warps": int(config.get("block_n_warps", DEFAULT_BLOCK_N_WARPS)),
        "b_to_lds": _as_bool(config.get("b_to_lds", DEFAULT_B_TO_LDS)),
        "b_preshuffle": _as_bool(config.get("b_preshuffle", DEFAULT_B_PRESHUFFLE)),
        "use_atomic_add": use_atomic_add,
        "c_to_lds": FIXED_C_TO_LDS,
    }


def _small_m_tile_m_options(m: int) -> list[int]:
    return [16] if m <= 4 else [16, 32]


def _small_m_supports_two_m_warps(m: int) -> bool:
    # `block_m_warps=2` needs at least one 16x16 MMA tile per warp.
    return any(tile_m >= 32 for tile_m in _small_m_tile_m_options(m))


def _small_m_tuning_variants(m: int) -> tuple[dict[str, Any], ...]:
    variants = [
        {"block_m_warps": 1, "block_n_warps": 1, "b_to_lds": False},
        {"block_m_warps": 1, "block_n_warps": 2, "b_to_lds": False},
        {"block_m_warps": 1, "block_n_warps": 4, "b_to_lds": False},
        {"block_m_warps": 1, "block_n_warps": 2, "b_to_lds": True},
        {"block_m_warps": 1, "block_n_warps": 4, "b_to_lds": True},
    ]
    if _small_m_supports_two_m_warps(m):
        variants.extend(
            [
                {"block_m_warps": 2, "block_n_warps": 1, "b_to_lds": False},
                {"block_m_warps": 2, "block_n_warps": 2, "b_to_lds": False},
                {"block_m_warps": 2, "block_n_warps": 2, "b_to_lds": True},
            ]
        )
    return tuple(variants)


def _iter_candidate_spaces(args: Args):
    yield DEFAULT_CONFIG_SELECTIONS, DEFAULT_TUNING_CONFIG_VARIANTS
    if args.m not in SMALL_M_VALUES:
        return

    tile_k_choices = [tk for tk in SMALL_M_TILE_K_OPTIONS if args.k % tk == 0]
    if not tile_k_choices:
        return

    yield (
        {
            "tile_k": tile_k_choices,
            "tile_m": _small_m_tile_m_options(args.m),
            "tile_n": [
                tn for tn in DEFAULT_CONFIG_SELECTIONS["tile_n"] if args.n % tn == 0
            ],
            "split_k": [
                sk for sk in DEFAULT_CONFIG_SELECTIONS["split_k"] if args.k % sk == 0
            ],
        },
        _small_m_tuning_variants(args.m),
    )


def _supports_hgemm_ar_block_capacity(args: Args, config: dict[str, Any]) -> bool:
    config = _normalize_config(config)
    bm = (args.m + config["tile_m"] - 1) // config["tile_m"]
    bn = args.n // config["tile_n"]
    required = bm * bn * config["split_k"]
    limit = min(HGEMM_AR_MAX_BLOCKS, SPLIT_K_COUNTER_MAX_LEN)
    return required <= limit


def _is_valid_config_for_shape(args: Args, config: dict[str, Any]) -> bool:
    config = _normalize_config(config)
    if args.n % config["tile_n"] != 0:
        return False
    if args.k % config["tile_k"] != 0:
        return False
    if args.k % config["split_k"] != 0:
        return False
    if not _supports_hgemm_ar_block_capacity(args, config):
        return False
    try:
        _validate_hgemm_tiling(
            args.m,
            args.n,
            args.k,
            dtype=_to_kernel_dtype(args.dtype),
            tile_m=config["tile_m"],
            tile_n=config["tile_n"],
            tile_k=config["tile_k"],
            pack_n=1,
            split_k=config["split_k"],
            stages=config["stages"],
            block_m_warps=config["block_m_warps"],
            block_n_warps=config["block_n_warps"],
            b_to_lds=config["b_to_lds"],
        )
    except Exception:
        return False
    return True


def _candidate_configs(args: Args) -> list[dict[str, Any]]:
    candidate_configs = []
    seen_configs = set()
    include_atomic_add = bool(args.include_atomic_add)
    use_atomic_add_values = (False, True) if include_atomic_add else (False,)
    for selections, variants in _iter_candidate_spaces(args):
        keys = list(selections.keys())
        values = [selections[key] for key in keys]
        if any(len(value) == 0 for value in values):
            continue
        for combo in itertools.product(*values):
            base_config = dict(zip(keys, combo))
            for variant in variants:
                for use_atomic_add in use_atomic_add_values:
                    config = _normalize_config(
                        {
                            **base_config,
                            **variant,
                            "use_atomic_add": use_atomic_add,
                        }
                    )
                    if not _is_valid_config_for_shape(args, config):
                        continue
                    config_key = tuple(sorted(config.items()))
                    if config_key in seen_configs:
                        continue
                    seen_configs.add(config_key)
                    candidate_configs.append(config)
    return candidate_configs


SUPPORTED_CONFIGS_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}


def _get_tol(k: int, world_size: int = 1) -> float:
    base_tol = float(k) / 2048 * 6e-1
    # Local HGEMM error compounds across TP ranks, but in practice it grows
    # sub-linearly with the reduction width.
    return base_tol * math.sqrt(max(int(world_size), 1))


def _candidate_search_budget_us(best_search_us: float) -> Optional[float]:
    if not math.isfinite(best_search_us):
        return None
    return max(
        float(best_search_us) * SEARCH_SKIP_SLOW_CANDIDATE_FACTOR,
        float(best_search_us) + SEARCH_SKIP_SLOW_CANDIDATE_SLACK_US,
    )


def _shape_seed(m: int, n: int, k: int) -> int:
    return int(m) * 1_000_000 + int(n) * 1_000 + int(k)


def _create_inputs(
    args: Args, rank_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    base_seed = _shape_seed(args.m, args.n, args.k)
    torch.manual_seed(base_seed)
    a = torch.empty((args.m, args.k), dtype=args.dtype, device=device)
    a.uniform_(-1, 1)
    torch.manual_seed(base_seed + 97 + rank_id)
    b = torch.empty((args.n, args.k), dtype=args.dtype, device=device)
    b.uniform_(-1, 1)
    return a, b


def _sync_group(device_group, device: torch.device) -> None:
    sync_token = torch.zeros(1, device=device)
    dist.all_reduce(sync_token, group=device_group)
    torch.cuda.synchronize(device)


def _initialize_tp_only_group(tp_size: int, backend: Optional[str] = None) -> None:
    backend = backend or torch.distributed.get_backend(
        parallel_state.get_world_group().device_group
    )
    world_size = torch.distributed.get_world_size()
    if world_size != tp_size:
        raise ValueError(
            f"TP-only init expects world_size == tp_size, got {world_size=} {tp_size=}"
        )
    assert parallel_state._TP is None, "tensor parallel group is already initialized"
    parallel_state._TP = parallel_state.init_model_parallel_group(
        [list(range(world_size))],
        parallel_state.get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )


def _measure_runtime_us(
    fn,
    *,
    warmup: int,
    niters: int,
    device_group,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    warmup = max(int(warmup), 0)
    niters = max(int(niters), 1)

    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize(device)
    _sync_group(device_group, device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(niters):
        out = fn()
    end_event.record()
    end_event.synchronize()
    elapsed_us = float(start_event.elapsed_time(end_event)) * 1e3 / niters

    elapsed_tensor = torch.tensor(elapsed_us, device=device, dtype=torch.float64)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX, group=device_group)
    _sync_group(device_group, device)
    return out, float(elapsed_tensor.item())


def _run_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = tgemm.mm(a, b, None, otype=a.dtype)
    return get_tp_group().all_reduce(out, ca_fp8_quant=False)


def _run_fused(a: torch.Tensor, b: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    config = _normalize_config(config)
    return flydsl_hgemm_ar(
        a,
        b,
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
        use_atomic_add=config["use_atomic_add"],
        auto_shuffle_b=config["b_preshuffle"],
        c_to_lds=config["c_to_lds"],
    )


def _is_candidate_supported(args: Args, config: dict[str, Any]) -> bool:
    config = _normalize_config(config)
    if not _supports_hgemm_ar_block_capacity(args, config):
        return False
    try:
        _validate_hgemm_tiling(
            args.m,
            args.n,
            args.k,
            dtype=_to_kernel_dtype(args.dtype),
            tile_m=config["tile_m"],
            tile_n=config["tile_n"],
            tile_k=config["tile_k"],
            pack_n=1,
            split_k=config["split_k"],
            stages=config["stages"],
            block_m_warps=config["block_m_warps"],
            block_n_warps=config["block_n_warps"],
            b_to_lds=config["b_to_lds"],
        )
        _compile_hgemm_ar_kernel(
            args.world_size,
            _to_kernel_dtype(args.dtype),
            args.n,
            args.k,
            TILE_M=config["tile_m"],
            TILE_N=config["tile_n"],
            TILE_K=config["tile_k"],
            SPLIT_K=config["split_k"],
            BLOCK_M_WARPS=config["block_m_warps"],
            BLOCK_N_WARPS=config["block_n_warps"],
            B_PRE_SHUFFLE=config["b_preshuffle"],
            B_TO_LDS=config["b_to_lds"],
            USE_CROSS_DEVICE_ATOMIC=config["use_atomic_add"],
        )
        return True
    except Exception:
        return False


def _get_supported_configs(
    args: Args,
    *,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    cache_key = (
        args.world_size,
        args.dtype,
        args.m,
        args.n,
        args.k,
        bool(args.include_atomic_add),
    )
    cached = SUPPORTED_CONFIGS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    candidate_configs = _candidate_configs(args)
    progress = None
    if show_progress:
        print(
            f"[rank0] checking supported configs for "
            f"tp={args.world_size}, M={args.m}, N={args.n}, K={args.k}: "
            f"{len(candidate_configs)} raw candidates",
            flush=True,
        )
        progress = tqdm(
            candidate_configs,
            total=len(candidate_configs),
            desc=f"support->tp{args.world_size},M={args.m},N={args.n},K={args.k}",
            mininterval=5.0,
            miniters=5,
        )

    iterator = candidate_configs if progress is None else progress
    supported: list[dict[str, Any]] = []
    total_candidates = len(candidate_configs)
    try:
        for idx, config in enumerate(iterator, start=1):
            if _is_candidate_supported(args, config):
                supported.append(config)
            if progress is not None and (
                idx == 1 or idx == total_candidates or idx % 5 == 0
            ):
                progress.set_postfix_str(f"valid={len(supported)}")
    finally:
        if progress is not None:
            progress.close()

    SUPPORTED_CONFIGS_CACHE[cache_key] = supported
    return supported


def _config_to_kernel_name(config: dict[str, Any], dtype: torch.dtype) -> str:
    config = _normalize_config(config)
    return flydsl_tp_hgemm_ar_kernel_name(
        dtype=_to_kernel_dtype(dtype),
        tile_m=config["tile_m"],
        tile_n=config["tile_n"],
        tile_k=config["tile_k"],
        split_k=config["split_k"],
        block_m_warps=config["block_m_warps"],
        block_n_warps=config["block_n_warps"],
        b_preshuffle=config["b_preshuffle"],
        b_to_lds=config["b_to_lds"],
        use_atomic_add=config["use_atomic_add"],
        stages=config["stages"],
        async_copy=config["async_copy"],
    )


def _calculate_perf(
    m: int,
    n: int,
    k: int,
    us: float,
    dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> tuple[float, float]:
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


def _make_output_row(
    args: Args,
    config: dict[str, Any],
    fused_us: float,
    err_ratio: float,
) -> dict[str, Any]:
    kernel_name = _config_to_kernel_name(config, args.dtype)
    tflops, bw = _calculate_perf(args.m, args.n, args.k, fused_us, args.dtype, args.dtype)
    return {
        "world_size": args.world_size,
        "cu_num": torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count,
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "bias": False,
        "dtype": _dtype_to_csv_string(args.dtype),
        "outdtype": _dtype_to_csv_string(args.dtype),
        "scaleAB": False,
        "bpreshuffle": False,
        "libtype": "flydsl_ar",
        "kernelName": kernel_name,
        "us": round(fused_us, 4),
        "err_ratio": round(err_ratio, 4),
        "tflops": tflops,
        "bw": bw,
    }


def _default_output_csv(input_csv: str) -> str:
    path = Path(input_csv)
    name = path.name
    if name.endswith("_untuned_gemm_bf16.csv"):
        out_name = name.replace("_untuned_gemm_bf16.csv", "_bf16_tuned_tp_gemm_ar.csv")
    else:
        out_name = f"{path.stem}_tuned_tp_gemm_ar.csv"
    return str(path.with_name(out_name))


def _worker(
    shape_args: list[Args],
    rank_id: int,
    distributed_init_method: Optional[str],
    search_warmup: int,
    search_niters: int,
    warmup: int,
    niters: int,
):
    device = torch.device(f"cuda:{rank_id}")
    torch.cuda.set_device(device)

    try:
        set_custom_all_reduce(True)
        init_distributed_environment(
            world_size=shape_args[0].world_size,
            rank=rank_id,
            distributed_init_method=distributed_init_method,
        )
        _initialize_tp_only_group(shape_args[0].world_size)

        tp_group = get_tp_group()
        _sync_group(tp_group.device_group, device)
        results = []
        for idx, args in enumerate(shape_args):
            if rank_id == 0:
                print(
                    f"[rank0] tuning shape {idx + 1}/{len(shape_args)}: "
                    f"tp={args.world_size}, M={args.m}, N={args.n}, K={args.k}",
                    flush=True,
                )
            try:
                a, b = _create_inputs(args, rank_id, device)
                ref_out = _run_reference(a, b)
                tol = _get_tol(args.k, args.world_size)

                valid_configs = _get_supported_configs(args, show_progress=rank_id == 0)
                if rank_id == 0:
                    print(
                        f"[rank0] candidate configs for M={args.m}, N={args.n}, K={args.k}: "
                        f"{len(valid_configs)}",
                        flush=True,
                    )

                best_config = None
                best_search_us = math.inf
                search_progress = None
                if rank_id == 0 and valid_configs:
                    search_progress = tqdm(
                        total=len(valid_configs),
                        desc=f"search->tp{args.world_size},M={args.m},N={args.n},K={args.k}",
                        mininterval=3.0,
                    )

                correct_configs = 0
                slow_skipped_configs = 0
                total_valid = len(valid_configs)

                try:
                    for idx_cfg, config in enumerate(valid_configs, start=1):
                        if search_progress is not None:
                            budget_us = _candidate_search_budget_us(best_search_us)
                            postfix = (
                                f"running={idx_cfg}/{total_valid},"
                                f"correct={correct_configs},"
                                f"cfg=t{config['tile_m']}x{config['tile_n']}x{config['tile_k']},"
                                f"sk={config['split_k']}"
                            )
                            if math.isfinite(best_search_us):
                                postfix += f",best_us={best_search_us:.2f}"
                            if budget_us is not None:
                                postfix += f",budget_us={budget_us:.2f}"
                            if slow_skipped_configs:
                                postfix += f",slow_skip={slow_skipped_configs}"
                            search_progress.set_postfix_str(postfix)
                            search_progress.refresh()

                        _run_fused(a, b, config)
                        torch.cuda.synchronize(device)
                        _sync_group(tp_group.device_group, device)
                        out, probe_us = _measure_runtime_us(
                            lambda cfg=config: _run_fused(a, b, cfg),
                            warmup=0,
                            niters=1,
                            device_group=tp_group.device_group,
                            device=device,
                        )
                        ok_tensor = torch.tensor(
                            1 if torch.allclose(out, ref_out, atol=tol, rtol=tol) else 0,
                            device=device,
                            dtype=torch.int32,
                        )
                        dist.all_reduce(
                            ok_tensor,
                            op=dist.ReduceOp.MIN,
                            group=tp_group.device_group,
                        )
                        if int(ok_tensor.item()) != 1:
                            _sync_group(tp_group.device_group, device)
                            if search_progress is not None:
                                search_progress.update(1)
                            continue

                        correct_configs += 1
                        budget_us = _candidate_search_budget_us(best_search_us)
                        if budget_us is not None and probe_us > budget_us:
                            slow_skipped_configs += 1
                            if search_progress is not None:
                                postfix = (
                                    f"running={idx_cfg}/{total_valid},"
                                    f"correct={correct_configs},"
                                    f"probe_us={probe_us:.2f},"
                                    f"budget_us={budget_us:.2f},"
                                    f"best_us={best_search_us:.2f},"
                                    f"slow_skip={slow_skipped_configs}"
                                )
                                search_progress.set_postfix_str(postfix)
                                search_progress.update(1)
                            continue

                        _, search_us = _measure_runtime_us(
                            lambda cfg=config: _run_fused(a, b, cfg),
                            warmup=search_warmup,
                            niters=search_niters,
                            device_group=tp_group.device_group,
                            device=device,
                        )
                        if search_us < best_search_us:
                            best_search_us = search_us
                            best_config = dict(config)
                        if search_progress is not None:
                            search_progress.update(1)
                finally:
                    if search_progress is not None:
                        search_progress.close()

                if best_config is None:
                    results.append(
                        {
                            "rank": rank_id,
                            "shape": args,
                            "best_config": None,
                            "search_us": None,
                            "fused_us": None,
                            "ref_us": None,
                            "err_ratio": None,
                            "winner": False,
                            "error": "no valid fusion config found",
                        }
                    )
                    continue

                fused_out, fused_us = _measure_runtime_us(
                    lambda: _run_fused(a, b, best_config),
                    warmup=warmup,
                    niters=niters,
                    device_group=tp_group.device_group,
                    device=device,
                )
                _, ref_us = _measure_runtime_us(
                    lambda: _run_reference(a, b),
                    warmup=warmup,
                    niters=niters,
                    device_group=tp_group.device_group,
                    device=device,
                )
                local_err_ratio = float(
                    checkAllclose(
                        ref_out,
                        fused_out,
                        rtol=tol,
                        atol=tol,
                        tol_err_ratio=1.0,
                        printLog=False,
                    )
                )
                err_tensor = torch.tensor(local_err_ratio, device=device, dtype=torch.float64)
                dist.all_reduce(
                    err_tensor, op=dist.ReduceOp.MAX, group=tp_group.device_group
                )

                results.append(
                    {
                        "rank": rank_id,
                        "shape": args,
                        "best_config": best_config,
                        "search_us": float(best_search_us),
                        "fused_us": float(fused_us),
                        "ref_us": float(ref_us),
                        "err_ratio": float(err_tensor.item()),
                        "winner": bool(fused_us < ref_us),
                        "error": None,
                    }
                )
                del fused_out, ref_out, a, b
                torch.cuda.empty_cache()
            except Exception as exc:
                results.append(
                    {
                        "rank": rank_id,
                        "shape": args,
                        "best_config": None,
                        "search_us": None,
                        "fused_us": None,
                        "ref_us": None,
                        "err_ratio": None,
                        "winner": False,
                        "error": str(exc),
                    }
                )
        return results
    finally:
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
        torch.cuda.empty_cache()


def _run_shapes_tuning(
    shape_args: list[Args],
    *,
    search_warmup: int,
    search_niters: int,
    warmup: int,
    niters: int,
) -> list[dict[str, Any]]:
    if not shape_args:
        return []
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=shape_args[0].world_size) as pool:
        results = [
            pool.apply_async(
                _worker,
                args=(
                    shape_args,
                    rank_id,
                    init_method,
                    search_warmup,
                    search_niters,
                    warmup,
                    niters,
                ),
            )
            for rank_id in range(shape_args[0].world_size)
        ]
        ret = [result.get() for result in results]

    rank0_results = ret[0]
    for shape_idx, rank0 in enumerate(rank0_results):
        for other_results in ret[1:]:
            other = other_results[shape_idx]
            if other["best_config"] != rank0["best_config"]:
                raise RuntimeError(
                    "rank-local best configs diverged during TP fusion tuning"
                )
            if other["error"] != rank0["error"]:
                raise RuntimeError("rank-local tuning failures diverged")
    return rank0_results


def tune_shapes_from_csv(
    input_csv: str,
    output_csv: str,
    *,
    tp: int,
    include_atomic_add: bool = False,
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
    shape_args = []
    for row in rows:
        if (
            _as_bool(row.get("bias", False))
            or _as_bool(row.get("scaleAB", False))
            or _as_bool(row.get("bpreshuffle", False))
        ):
            continue
        shape_args.append(
            Args(
                dtype=_parse_torch_dtype(row["dtype"]),
                m=int(row["M"]),
                n=int(row["N"]),
                k=int(row["K"]),
                world_size=int(tp),
                include_atomic_add=bool(include_atomic_add),
            )
        )

    results = _run_shapes_tuning(
        shape_args,
        search_warmup=search_warmup,
        search_niters=search_niters,
        warmup=warmup,
        niters=niters,
    )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        shape_pbar = tqdm(results, desc=f"tune->{output_path.name}")
        for result in shape_pbar:
            args = result["shape"]
            shape_pbar.set_postfix_str(f"tp={args.world_size},M={args.m},N={args.n},K={args.k}")
            if result["error"] is not None:
                print(
                    f"skip shape tp={args.world_size}, M={args.m}, N={args.n}, K={args.k}: {result['error']}",
                    flush=True,
                )
                continue
            if result["winner"]:
                writer.writerow(
                    _make_output_row(
                        args,
                        result["best_config"],
                        result["fused_us"],
                        result["err_ratio"],
                    )
                )
                f.flush()
                saved += 1
        shape_pbar.close()

    print(f"saved {saved} TP fusion winners to {output_path}", flush=True)


def tune_single_shape(
    args: Args,
    *,
    search_warmup: int = 5,
    search_niters: int = 50,
    warmup: int = 20,
    niters: int = 100,
) -> None:
    result = _run_shapes_tuning(
        [args],
        search_warmup=search_warmup,
        search_niters=search_niters,
        warmup=warmup,
        niters=niters,
    )[0]
    if result["error"] is not None:
        raise RuntimeError(result["error"])
    print(
        {
            "tp": args.world_size,
            "M": args.m,
            "N": args.n,
            "K": args.k,
            "best_config": result["best_config"],
            "search_us": result["search_us"],
            "fused_us": result["fused_us"],
            "ref_us": result["ref_us"],
            "err_ratio": result["err_ratio"],
            "winner": result["winner"],
        },
        flush=True,
    )


if __name__ == "__main__":
    freeze_support()
    ensure_spawn_method()

    parser = argparse.ArgumentParser(
        description="Benchmark or tune FlyDSL GEMM+TP-allreduce fusion kernels.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel world size.")
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
        "--search-warmup",
        type=int,
        default=5,
        help="Warmup iterations used while searching candidate FlyDSL fusion configs.",
    )
    parser.add_argument(
        "--search-niters",
        type=int,
        default=50,
        help="Measured iterations used while searching candidate FlyDSL fusion configs.",
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
    parser.add_argument(
        "--include-atomic-add",
        action="store_true",
        help=(
            "Also search USE_ATOMIC_ADD=True configs for split_k=1. "
            "The split_k>1 path always forces cross-device atomic."
        ),
    )
    cli_args = parser.parse_args()
    print(f"run: {__file__}, args: {cli_args}", flush=True)

    if cli_args.input_csv:
        output_csv = cli_args.output_csv or _default_output_csv(cli_args.input_csv)
        tune_shapes_from_csv(
            cli_args.input_csv,
            output_csv,
            tp=cli_args.tp,
            include_atomic_add=cli_args.include_atomic_add,
            search_warmup=cli_args.search_warmup,
            search_niters=cli_args.search_niters,
            warmup=cli_args.warmup,
            niters=cli_args.niters,
            limit=cli_args.limit,
        )
    else:
        if cli_args.m is None or cli_args.n is None or cli_args.k is None:
            parser.error("--m/--n/--k are required when --input-csv is not provided")

        tune_single_shape(
            Args(
                dtype=CLI_DTYPE_MAP[cli_args.dtype],
                m=cli_args.m,
                n=cli_args.n,
                k=cli_args.k,
                world_size=cli_args.tp,
                include_atomic_add=cli_args.include_atomic_add,
            ),
            search_warmup=cli_args.search_warmup,
            search_niters=cli_args.search_niters,
            warmup=cli_args.warmup,
            niters=cli_args.niters,
        )
