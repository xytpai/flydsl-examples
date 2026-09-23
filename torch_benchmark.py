import argparse
import ast
import json
import multiprocessing as mp
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest import mock

from tqdm import tqdm

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch.nn.functional as F


# Reference GEMM throughput in TFLOP/s on MI355X (gfx950). Empty cells were not measured.
# BF16: pytorch#190903 graph replay (ATen default, Triton/FlyDSL EXHAUSTIVE).
# MXFP8/MXFP4: pytorch#196719 (ATen vs FlyDSL).
# | M | N | K | BF16 ATen | BF16 Triton | BF16 FlyDSL | MXFP8 ATen | MXFP8 FlyDSL | MXFP4 ATen | MXFP4 FlyDSL |
# |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
# | 8 | 4096 | 4096 | 22.8 | 20.6 | 22.9 |  |  |  |  |
# | 16 | 4096 | 4096 | 43.6 | 39.2 | 45.9 |  |  |  |  |
# | 32 | 4096 | 4096 | 72.6 | 69.3 | 85.2 | 31.3 | 69.8 | 39.4 | 99.6 |
# | 64 | 4096 | 4096 | 112.7 | 120.3 | 160.3 | 70.4 | 135.7 | 103.1 | 190.2 |
# | 128 | 4096 | 4096 | 188.0 | 181.2 | 246.1 | 138.5 | 240.7 | 187.4 | 406.0 |
# | 256 | 4096 | 4096 | 364.0 | 310.1 | 434.9 | 293.3 | 412.2 | 468.1 | 698.0 |
# | 512 | 4096 | 4096 | 539.4 | 509.8 | 598.6 | 459.3 | 634.6 | 677.7 | 1127.5 |
# | 1024 | 4096 | 4096 | 747.8 | 782.7 | 842.2 | 729.3 | 1057.5 | 1075.6 | 1662.4 |
# | 2048 | 4096 | 4096 | 706.3 | 953.1 | 1126.5 | 1129.5 | 1559.7 | 1540.5 | 2281.6 |
# | 4096 | 4096 | 4096 | 1379.2 | 1161.8 | 1384.1 | 1321.9 | 2114.7 | 2244.2 | 3030.1 |
# | 4096 | 4096 | 8192 | 1514.9 | 1238.5 | 1491.5 | 1505.1 | 2484.4 | 2809.4 | 3679.7 |
# | 8192 | 8192 | 8192 | 1571.2 | 1240.6 | 1540.9 | 1369.7 | 2804.0 | 3110.0 | 4155.1 |
# | 32 | 14336 | 4096 | 148.0 | 141.5 | 166.7 | 94.7 | 192.7 | 111.3 | 329.6 |
# | 16 | 28672 | 4096 | 77.8 | 100.6 | 101.6 |  |  |  |  |
# | 32 | 28672 | 4096 |  |  |  | 180.8 | 271.1 | 204.6 | 487.7 |
# | 4096 | 256 | 4096 | 385.2 | 378.4 | 415.0 | 298.8 | 406.1 | 505.8 | 691.8 |
# | 4096 | 4096 | 256 |  |  |  | 304.0 | 400.6 | 343.4 | 566.6 |
# | 4096 | 4096 | 512 |  |  |  | 520.4 | 681.7 | 607.4 | 947.0 |
# | 4096 | 4096 | 1024 |  |  |  | 810.7 | 1111.2 | 1043.2 | 1499.2 |
# | 4096 | 4096 | 2048 |  |  |  | 1111.4 | 1637.7 | 1613.7 | 2266.2 |


SHAPES = [
    (8, 4096, 4096),
    (16, 4096, 4096),
    (32, 4096, 4096),
    (64, 4096, 4096),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
    (2048, 4096, 4096),
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    (8192, 8192, 8192),
    (32, 14336, 4096),
    (16, 28672, 4096),
    (4096, 256, 4096),
]


# pytorch#196719 MXFP8 and MXFP4 suite. Same shapes for both formats.
MXFP_SHAPES = [
    (32, 4096, 4096),
    (32, 14336, 4096),
    (32, 28672, 4096),
    (64, 4096, 4096),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
    (2048, 4096, 4096),
    (4096, 256, 4096),
    (4096, 4096, 256),
    (4096, 4096, 512),
    (4096, 4096, 1024),
    (4096, 4096, 2048),
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    (8192, 8192, 8192),
]


def shapes_for_dtype(dtype):
    return MXFP_SHAPES if dtype in ("mxfp8", "mxfp4") else SHAPES


BACKEND_PATCHES = {
    "aten": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN",
        "max_autotune_gemm_search_space": "DEFAULT",
    },
    "triton": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "TRITON",
        "max_autotune_gemm_search_space": "EXHAUSTIVE",
    },
    "flydsl": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "FLYDSL",
        "max_autotune_gemm_search_space": "EXHAUSTIVE",
        "flydsl_enable_autotuning": True,
    },
}


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "mxfp8": torch.float8_e4m3fn,
    "mxfp4": torch.float4_e2m1fn_x2,
}


def mm_nt(a, b):
    return torch.mm(a, b.t())


def scaled_mm_nt(a, b, scale_a, scale_b):
    # B is stored [N, storage_K], just as in mm_nt; scales are NOT transposed.
    from torch.nn.functional import ScalingType, SwizzleType

    return F.scaled_mm(
        a, b.t(),
        scale_a, ScalingType.BlockWise1x32,
        scale_b, ScalingType.BlockWise1x32,
        SwizzleType.NO_SWIZZLE, SwizzleType.NO_SWIZZLE,
        output_dtype=torch.bfloat16,
    )


def decode_mxfp(data, scale):
    if data.dtype == torch.float8_e4m3fn:
        values = data.float()
    else:
        # Low nibble is the first logical K element.
        packed = data.view(torch.uint8)
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        lut = torch.tensor(
            [0., .5, 1., 1.5, 2., 3., 4., 6.,
             -0., -.5, -1., -1.5, -2., -3., -4., -6.],
            device=data.device,
        )
        values = lut[codes.long()]
    exponents = scale.view(torch.uint8).float()
    factors = torch.exp2(exponents - 127).repeat_interleave(32, dim=-1)
    return values * factors


def create_mxfp_inputs(m, n, k, mxfp_format, device="cuda"):
    # The PR uses this quantizer. Import lazily so ordinary GEMM does not need
    # PyTorch's internal quantization test utilities.
    from torch.testing._internal.common_quantized import to_mxfp

    scale_a, a = to_mxfp(torch.randn(m, k, device=device), format=mxfp_format)
    scale_b, b = to_mxfp(torch.randn(n, k, device=device), format=mxfp_format)
    a, b = a.contiguous(), b.contiguous()
    scale_a, scale_b = scale_a.contiguous(), scale_b.contiguous()
    # Compare against the actual quantized operands, NOT the pre-quantized input.
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        reference = decode_mxfp(a, scale_a) @ decode_mxfp(b, scale_b).t()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    return (a, b, scale_a, scale_b), reference


def check_accuracy(result, reference, is_mxfp):
    if not torch.isfinite(result).all().item():
        raise AssertionError("GEMM output contains NaN or Inf")
    diff = result.float() - reference.float()
    max_diff = diff.abs().max().item()
    relative_rms = (
        diff.square().mean() / reference.float().square().mean().clamp_min(1e-20)
    ).sqrt().item()
    atol, rtol = (6e-2, 3e-2) if is_mxfp else (3e-2, 3e-2)
    torch.testing.assert_close(
        result.float(), reference.float(), atol=atol, rtol=rtol
    )
    if is_mxfp and relative_rms > .008:
        raise AssertionError(f"MXFP relative RMS error {relative_rms} exceeds .008")
    return max_diff, relative_rms


def run_padded_stride_regression(args):
    """Check aligned padded rows and reject unaligned FlyDSL row strides."""
    from torch._inductor.heuristics.template import flydsl as flydsl_heuristics
    from torch._inductor.utils import run_and_get_code

    dtype = DTYPES[args.dtype]
    common_config = {
        "TILE_M": 128,
        "TILE_N": 128,
        "TILE_K": 64,
        "STAGES": 2,
        "SPLIT_K": 1,
        "BLOCK_K_WARPS": 1,
        "GROUP_M": 0,
        "B_TO_LDS": True,
    }
    cases = (
        {
            "name": "full-tile-small",
            "shape": (64, 64, 128),
            "a_row_stride": 160,
            "b_row_stride": 160,
            "expect_flydsl": True,
            "kernel_config": {
                **common_config,
                "BLOCK_M_WARPS": 4,
                "BLOCK_N_WARPS": 4,
                "USE_HALF_TILE_INTERLEAVED": False,
            },
        },
        {
            "name": "hti-large",
            "shape": (1024, 1024, 1024),
            "a_row_stride": 1056,
            "b_row_stride": 1088,
            "expect_flydsl": True,
            "kernel_config": {
                **common_config,
                "BLOCK_M_WARPS": 2,
                "BLOCK_N_WARPS": 2,
                "USE_HALF_TILE_INTERLEAVED": True,
            },
        },
        {
            "name": "unaligned-row-stride",
            "shape": (64, 64, 128),
            "a_row_stride": 129,
            "b_row_stride": 129,
            "expect_flydsl": False,
            "kernel_config": {
                **common_config,
                "BLOCK_M_WARPS": 4,
                "BLOCK_N_WARPS": 4,
                "USE_HALF_TILE_INTERLEAVED": False,
            },
        },
    )

    torch.manual_seed(0)
    regression_config = {
        **BACKEND_PATCHES["flydsl"],
        "max_autotune_gemm_search_space": "DEFAULT",
        "flydsl_enable_autotuning": False,
    }
    for case in cases:
        name = case["name"]
        m, n, k = case["shape"]
        a_row_stride = case["a_row_stride"]
        b_row_stride = case["b_row_stride"]
        expect_flydsl = case["expect_flydsl"]

        torch._dynamo.reset()
        torch.cuda.empty_cache()

        a_storage = torch.randn(
            (m, a_row_stride), device="cuda", dtype=dtype
        )
        b_storage = torch.randn(
            (n, b_row_stride), device="cuda", dtype=dtype
        )
        a = a_storage[:, :k]
        b = b_storage[:, :k]

        assert a.stride() == (a_row_stride, 1)
        assert b.stride() == (b_row_stride, 1)
        assert b.t().stride() == (1, b_row_stride)

        case_config = regression_config
        if not expect_flydsl:
            case_config = {
                **regression_config,
                "max_autotune_gemm_backends": "ATEN,FLYDSL",
            }

        with (
            inductor_config.patch(**case_config),
            mock.patch.object(
                flydsl_heuristics,
                "get_gemm_configs",
                return_value=[case["kernel_config"]],
            ) as get_gemm_configs,
        ):
            compiled = torch.compile(mm_nt, backend="inductor")
            result, (code,) = run_and_get_code(compiled, a, b)

        uses_flydsl = "async_compile.flydsl" in code
        assert uses_flydsl == expect_flydsl
        if expect_flydsl:
            get_gemm_configs.assert_called()
        else:
            get_gemm_configs.assert_not_called()

        ref = mm_nt(a, b)
        abs_diff = (result.float() - ref.float()).abs()
        max_diff = abs_diff.max().item()
        mismatches = (abs_diff > 3e-2).sum().item()

        print(
            f"FlyDSL padded-stride regression [{name}]: "
            f"M={m} N={n} K={k}, "
            f"A.stride={a.stride()}, B.stride={b.stride()}, "
            f"B.T.stride={b.t().stride()}, "
            f"selected={'FlyDSL' if uses_flydsl else 'fallback'}"
        )
        print(f"max absolute error: {max_diff}")
        print(f"entries with absolute error > 0.03: {mismatches}/{m * n}")

        torch.testing.assert_close(result, ref, atol=3e-2, rtol=3e-2)


def tflops(m, n, k, ms):
    return 2.0 * m * n * k / (ms * 1.0e9)


def e2e_bench(fn, inputs, warmup, reps, rounds, check):
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            result = fn(*inputs)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / reps)
        # Validate each timing round outside its event interval.
        check(result)
    return statistics.median(samples)


def cuda_graph_bench(fn, inputs, warmup, reps, rounds, check):
    # One call per graph, one input allocation (hot); this bypasses the compiled
    # backend's Python launch path, but still measures graph replay launch gaps.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_output = fn(*inputs)

        # Guard against capture on the wrong stream / an empty graph.
        graph_output.fill_(float("nan"))
        graph.replay()
        check(graph_output)

        def replay(*_inputs):
            graph.replay()
            return graph_output

        ms = e2e_bench(replay, inputs, warmup, reps, rounds, check)
    torch.cuda.current_stream().wait_stream(stream)
    return ms


def get_kernel_info(codes):
    """Read called entries and static tuning config from the final wrappers."""
    kernels = {}
    config_keys = {
        "TILE_M", "TILE_N", "TILE_K", "STAGES", "M_WAVES", "N_WAVES",
        "GROUP_M", "USE_HALF_TILE_INTERLEAVED", "BLOCK_M", "BLOCK_N", "BLOCK_K",
        "matrix_instr_nonkdim", "waves_per_eu", "kpack",
    }
    error = None
    try:
        for code in codes:
            tree = ast.parse(code)
            definitions = {
                node.targets[0].id: node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) in (
                    "async_compile.flydsl", "async_compile.triton",
                )
            }
            calls = sorted(
                (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
                key=lambda node: (node.lineno, node.col_offset),
            )
            for call in calls:
                func = ast.unparse(call.func)
                if func.endswith(".run") and func[:-4] in definitions:
                    definition = definitions[func[:-4]]
                    name = ast.literal_eval(definition.args[0])
                    source = ast.parse(ast.literal_eval(definition.args[1]))
                    cfg = {}
                    for node in ast.walk(source):
                        if (isinstance(node, ast.AnnAssign)
                                and isinstance(node.target, ast.Name)
                                and node.target.id in config_keys
                                and isinstance(node.value, ast.Constant)):
                            cfg[node.target.id] = node.value.value
                        elif (isinstance(node, ast.keyword)
                              and node.arg in ("num_warps", "num_stages")
                              and isinstance(node.value, ast.Constant)):
                            cfg[node.arg] = node.value.value
                    kernels[name] = cfg
                elif func in (
                    "extern_kernels.mm", "extern_kernels.addmm",
                    "torch.ops.aten._scaled_mm_v2.default",
                    "aten._scaled_mm_v2.default",
                ):
                    # The library's internal GPU kernel is not exposed here.
                    kernels[func] = {}
        if not kernels:
            error = "No supported kernel calls found in generated wrapper"
    except (SyntaxError, ValueError, TypeError, IndexError) as exc:
        kernels.clear()
        error = repr(exc)
    return {
        "kernel_name": "; ".join(kernels) or None,
        "kernel_config": kernels,
        "kernel_name_source": "inductor_wrapper",
        "kernel_name_error": error,
    }


def _flydsl_compile_jobs(dtype, shapes, search_space):
    """One job per distinct FlyDSL config, using a shape that config accepts."""
    from torch._inductor.heuristics.template.flydsl import (
        get_gemm_configs,
        is_gemm_config_valid_for_shape,
        is_gemm_config_worth_tuning,
    )
    from torch._inductor.kernel.vendored_templates.flydsl.kernels.gemm_gfx950 import (
        GEMM_DTYPE_BF16,
        GEMM_DTYPE_FP16,
        GEMM_DTYPE_MXFP4,
        GEMM_DTYPE_MXFP8,
        infer_has_k_tail,
    )

    mxfp = dtype in ("mxfp8", "mxfp4")
    dtype_id = {
        "bfloat16": GEMM_DTYPE_BF16,
        "float16": GEMM_DTYPE_FP16,
        "mxfp8": GEMM_DTYPE_MXFP8,
        "mxfp4": GEMM_DTYPE_MXFP4,
    }[dtype]
    out_dtype_id = GEMM_DTYPE_BF16 if mxfp else None
    with inductor_config.patch(
        max_autotune_gemm=True,
        flydsl_enable_autotuning=True,
        max_autotune_gemm_search_space=search_space or "EXHAUSTIVE",
    ):
        configs = get_gemm_configs(dtype if mxfp else None)
    jobs = []
    seen = set()
    for m, n, k in shapes:
        for cfg in configs:
            if not is_gemm_config_worth_tuning(m, n, k, cfg):
                continue
            if not is_gemm_config_valid_for_shape(
                m,
                n,
                k,
                dtype_id,
                cfg,
                a_is_transposed=False,
                b_is_transposed=True,
                out_dtype_id=out_dtype_id,
                has_bias=False,
            ):
                continue
            tile_k = int(cfg["TILE_K"])
            stages = int(cfg["STAGES"])
            hti = bool(cfg.get("USE_HALF_TILE_INTERLEAVED", False))
            key = (
                dtype,
                int(cfg["TILE_M"]),
                int(cfg["TILE_N"]),
                tile_k,
                stages,
                int(cfg["M_WAVES"]),
                int(cfg["N_WAVES"]),
                int(cfg["GROUP_M"]),
                hti,
                infer_has_k_tail(k, tile_k, stages, hti),
            )
            if key in seen:
                continue
            seen.add(key)
            jobs.append((dtype, m, n, k, cfg))
    return jobs


def _compile_flydsl_job(job):
    """Compile one FlyDSL config. Runs in its own process."""
    dtype_name, m, n, k, cfg = job
    os.environ["COMPILE_ONLY"] = "1"
    os.environ["FLYDSL_COMPILE_ONLY"] = "1"
    try:
        import flydsl.compiler as flyc

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.gemm_gfx950 import (
            GEMM_DTYPE_BF16,
            GEMM_DTYPE_FP16,
            GEMM_DTYPE_MXFP4,
            GEMM_DTYPE_MXFP8,
            gemm_gfx950,
            gemm_mxfp_gfx950,
            infer_has_k_tail,
            make_gemm_param_and_validate,
        )

        torch.cuda.set_device(0)
        dtype_id = {
            "bfloat16": GEMM_DTYPE_BF16,
            "float16": GEMM_DTYPE_FP16,
            "mxfp8": GEMM_DTYPE_MXFP8,
            "mxfp4": GEMM_DTYPE_MXFP4,
        }[dtype_name]
        mxfp = dtype_name in ("mxfp8", "mxfp4")
        tile_k = int(cfg["TILE_K"])
        stages = int(cfg["STAGES"])
        hti = bool(cfg.get("USE_HALF_TILE_INTERLEAVED", False))
        param = make_gemm_param_and_validate(
            m,
            n,
            k,
            {
                "dtype_id": dtype_id,
                "out_dtype_id": GEMM_DTYPE_BF16 if mxfp else None,
                "tile_m": int(cfg["TILE_M"]),
                "tile_n": int(cfg["TILE_N"]),
                "tile_k": tile_k,
                "stages": stages,
                "m_waves": int(cfg["M_WAVES"]),
                "n_waves": int(cfg["N_WAVES"]),
                "group_m": int(cfg["GROUP_M"]),
                "use_half_tile_interleaved": hti,
                "a_is_transposed": False,
                "b_is_transposed": True,
                "has_bias": False,
                "has_k_tail": infer_has_k_tail(k, tile_k, stages, hti),
            },
        )
        if param is None:
            return f"rejected {cfg}"
        storage_k = k // 2 if dtype_name == "mxfp4" else k
        if mxfp:
            value_dtype = (
                torch.float4_e2m1fn_x2
                if dtype_name == "mxfp4"
                else torch.float8_e4m3fn
            )
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            # Values and e8m0 scales are uint8, matching flydsl_mm.py.jinja.
            # B is stored [N, K] and compiled as b.t(), so the stride-1 axis is K.
            # mark_layout_dynamic keeps that axis in the disk cache key.
            a = torch.empty((m, storage_k), device="cuda", dtype=value_dtype).view(
                torch.uint8
            )
            b = torch.empty((n, storage_k), device="cuda", dtype=value_dtype).t().view(
                torch.uint8
            )
            scale_a = torch.empty((m, k // 32), device="cuda", dtype=torch.uint8)
            scale_b = torch.empty((n, k // 32), device="cuda", dtype=torch.uint8)
            tensors = (out, a, b, scale_a, scale_b, out)
            kernel = gemm_mxfp_gfx950
        else:
            value_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
            out = torch.empty((m, n), device="cuda", dtype=value_dtype)
            a = torch.empty((m, k), device="cuda", dtype=value_dtype)
            # Same NT layout as mm_nt: B storage is [N, K], the kernel sees b.t().
            b = torch.empty((n, k), device="cuda", dtype=value_dtype).t()
            tensors = (out, a, b)
            kernel = gemm_gfx950
        flyc.compile(
            kernel,
            *(flyc.from_torch_tensor(t).mark_layout_dynamic() for t in tensors),
            param,
            0,
        )
        return None
    except Exception as exc:
        return f"{cfg}: {exc}"


_AUTOTUNE_DESC = ["autotune"]


def _install_autotune_progress():
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    if getattr(AlgorithmSelectorCache, "_benchmark_progress", False):
        return
    AlgorithmSelectorCache._benchmark_progress = True
    orig_choices = AlgorithmSelectorCache.benchmark_choices
    orig_choice = AlgorithmSelectorCache.benchmark_choice
    bar = {"tqdm": None}

    def benchmark_choices(cls, choices, autotune_args, is_collective=False):
        bar["tqdm"] = tqdm(
            total=len(choices),
            desc=_AUTOTUNE_DESC[0],
            unit="config",
            leave=False,
        )
        try:
            return orig_choices.__func__(cls, choices, autotune_args, is_collective)
        finally:
            bar["tqdm"].close()
            bar["tqdm"] = None

    def benchmark_choice(cls, choice, autotune_args):
        try:
            return orig_choice.__func__(cls, choice, autotune_args)
        finally:
            if bar["tqdm"] is not None:
                bar["tqdm"].update(1)

    AlgorithmSelectorCache.benchmark_choices = classmethod(benchmark_choices)
    AlgorithmSelectorCache.benchmark_choice = classmethod(benchmark_choice)


def aot_precompile_flydsl(dtype, shapes, search_space, workers):
    """Compile each FlyDSL config once, before the results table."""
    from torch._inductor.runtime.flydsl_cache import configure_flydsl_cache_dir

    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(workers)
    inductor_config.compile_threads = workers
    configure_flydsl_cache_dir()
    jobs = _flydsl_compile_jobs(dtype, shapes, search_space)
    if not jobs:
        return
    pool_size = min(workers, len(jobs))
    failures = []
    executor = ProcessPoolExecutor(
        max_workers=pool_size,
        mp_context=mp.get_context("spawn"),
    )
    try:
        for error in tqdm(
            executor.map(_compile_flydsl_job, jobs, chunksize=1),
            total=len(jobs),
            desc=f"AOT precompile ({pool_size} processes)",
            unit="config",
        ):
            if error:
                failures.append(error)
    finally:
        # Workers hold a CUDA context and do not exit when the pool joins.
        processes = list(getattr(executor, "_processes", {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.kill()
    if failures:
        print(f"AOT precompile failed {len(failures)} configs. First: {failures[0]}")


def compile_case(backend, m, n, k, args):
    # Do not reset Dynamo here. This loop compiles every shape before timing
    # any of them, and reset() drops the earlier specializations. The later
    # call then retraces mm_nt with automatic dynamic shapes (M/N become
    # symbols). FlyDSL only emits configs for static shapes, so Inductor
    # raises NoValidChoicesError. dynamic=False keeps each shape static;
    # isolate_recompiles keeps each shape's cache past the recompile limit.
    torch.cuda.empty_cache()
    # Reuse the same random values for every backend at a given shape.
    torch.manual_seed(args.seed)
    is_mxfp = args.dtype in ("mxfp8", "mxfp4")
    if is_mxfp:
        inputs, ref = create_mxfp_inputs(m, n, k, args.dtype)
        fn = scaled_mm_nt
    else:
        dtype = DTYPES[args.dtype]
        a = torch.randn((m, k), device="cuda", dtype=dtype)
        b = torch.randn((n, k), device="cuda", dtype=dtype)
        inputs = (a, b)
        fn = mm_nt
        ref = mm_nt(a, b)

    metrics = {"float_max_diff": 0., "relative_rms": 0., "accuracy_checks": 0}

    def check(result):
        expected_dtype = torch.bfloat16 if is_mxfp else DTYPES[args.dtype]
        if result.dtype != expected_dtype:
            raise AssertionError(f"expected {expected_dtype} output, got {result.dtype}")
        max_diff, relative_rms = check_accuracy(result, ref, is_mxfp)
        metrics["float_max_diff"] = max(metrics["float_max_diff"], max_diff)
        metrics["relative_rms"] = max(metrics["relative_rms"], relative_rms)
        metrics["accuracy_checks"] += 1

    config = dict(BACKEND_PATCHES[backend])
    if args.search_space is not None:
        config["max_autotune_gemm_search_space"] = args.search_space
    with inductor_config.patch(**config):
        compiled = torch.compile(
            fn, backend="inductor", dynamic=False, isolate_recompiles=True
        )
        from torch._inductor.utils import run_and_get_code

        result, codes = run_and_get_code(compiled, *inputs)
        selected_backend = backend
        if is_mxfp:
            code = "\n".join(codes)
            selected_backend = (
                "flydsl" if "async_compile.flydsl" in code
                else "triton" if "async_compile.triton" in code
                else "aten" if "_scaled_mm_v2" in code
                else "unknown"
            )
            if selected_backend != backend:
                raise RuntimeError(
                    f"requested {backend}, but MXFP lowered to {selected_backend}; "
                    "refusing to report fallback performance as the requested backend"
                )
        # A failed correctness check must not produce performance numbers.
        check(result)
        kernel_info = get_kernel_info(codes)
    return {
        "backend": backend,
        "selected_backend": selected_backend,
        "kernel_info": kernel_info,
        "m": m,
        "n": n,
        "k": k,
        "is_mxfp": is_mxfp,
        "config": config,
        "compiled": compiled,
        "inputs": inputs,
        "check": check,
        "metrics": metrics,
    }


def time_case(prepared, args):
    with inductor_config.patch(**prepared["config"]):
        e2e_ms = e2e_bench(
            prepared["compiled"],
            prepared["inputs"],
            args.warmup,
            args.reps,
            args.rounds,
            prepared["check"],
        )
        graph_error = None
        try:
            graph_ms = cuda_graph_bench(
                prepared["compiled"],
                prepared["inputs"],
                args.warmup,
                args.reps,
                args.rounds,
                prepared["check"],
            )
        except AssertionError:
            raise
        except Exception as exc:
            graph_ms = None
            graph_error = repr(exc)

    is_mxfp = prepared["is_mxfp"]
    return {
        "backend": prepared["backend"],
        "selected_backend": prepared["selected_backend"],
        **prepared["kernel_info"],
        "m": prepared["m"],
        "n": prepared["n"],
        "k": prepared["k"],
        "dtype": args.dtype,
        "output_dtype": str(torch.bfloat16 if is_mxfp else DTYPES[args.dtype]),
        "op": "scaled_mm" if is_mxfp else "mm",
        "layout": "nt",
        "bias": False,
        "seed": args.seed,
        "search_space": prepared["config"]["max_autotune_gemm_search_space"],
        "timing": "hot; CUDA events; one invocation per captured graph",
        "ok": True,
        **prepared["metrics"],
        "e2e_ms": e2e_ms,
        "graph_ms": graph_ms,
        "graph_error": graph_error,
        "e2e_tflops": tflops(prepared["m"], prepared["n"], prepared["k"], e2e_ms),
        "graph_tflops": (
            tflops(prepared["m"], prepared["n"], prepared["k"], graph_ms)
            if graph_ms is not None
            else None
        ),
    }


def parse_backends(value):
    backends = [name.strip() for name in value.split(",")]
    if backends == ["all"]:
        return list(BACKEND_PATCHES)
    if any(name not in BACKEND_PATCHES for name in backends):
        raise argparse.ArgumentTypeError(
            "expected aten, triton, flydsl, or a comma-separated combination; "
            "all must be used alone"
        )
    return list(dict.fromkeys(backends))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=parse_backends,
        default="all",
        metavar="BACKENDS",
        help="aten, triton, flydsl, a comma-separated combination (e.g. flydsl,aten), "
             "or all (default); runs in the given order, ignoring duplicates",
    )
    parser.add_argument("--output", default="./temp")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--shape-index", type=int, default=None,
        help="index in the shape list for the selected dtype",
    )
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compile-workers",
        type=int,
        default=16,
        help="FlyDSL config compile processes. Threads share one kernel and do not overlap.",
    )
    parser.add_argument(
        "--search-space", choices=["DEFAULT", "EXHAUSTIVE"], default=None,
        help="override the backend's default autotuning search space",
    )
    parser.add_argument(
        "--padded-stride-regression",
        action="store_true",
        help="run aligned padded-row and unaligned-stride FlyDSL regressions",
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.reps <= 0 or args.rounds <= 0:
        parser.error("warmup must be nonnegative; reps and rounds must be positive")
    if args.compile_workers <= 0:
        parser.error("--compile-workers must be positive")
    if args.padded_stride_regression:
        if args.dtype in ("mxfp8", "mxfp4"):
            parser.error("--padded-stride-regression only supports bfloat16/float16")
        run_padded_stride_regression(args)
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backends = args.backend
    shapes = shapes_for_dtype(args.dtype)
    if args.shape_index is not None and not 0 <= args.shape_index < len(shapes):
        parser.error(f"--shape-index must be between 0 and {len(shapes) - 1}")
    if args.shape_index is not None:
        shapes = [shapes[args.shape_index]]

    if "flydsl" in backends:
        aot_precompile_flydsl(args.dtype, shapes, args.search_space, args.compile_workers)
        print("AOT precompile finished.", flush=True)

    _install_autotune_progress()
    # One reset before the compile loop, not inside it. See compile_case.
    torch._dynamo.reset()
    prepared = []
    for backend in backends:
        for m, n, k in shapes:
            _AUTOTUNE_DESC[0] = f"autotune {backend} M={m} N={n} K={k}"
            try:
                prepared.append(compile_case(backend, m, n, k, args))
            except Exception as exc:
                prepared.append({
                    "error": repr(exc),
                    "backend": backend,
                    "m": m,
                    "n": n,
                    "k": k,
                })

    header = (
        f"{'Backend':<8} {'Shape (M/N/K)':<28} {'DType':<10} {'Accuracy':<8} "
        f"{'E2E ms':>10} {'E2E TFLOPS':>12} "
        f"{'Graph ms':>10} {'Graph TFLOPS':>13} {'Max diff':>11}  "
        "Config"
    )
    print(header)
    print("-" * len(header))

    with out_path.open("a", buffering=1) as f:
        for item in prepared:
            backend = item["backend"]
            m, n, k = item["m"], item["n"], item["k"]
            if item.get("error") and "compiled" not in item:
                row = {
                    "backend": backend,
                    "m": m,
                    "n": n,
                    "k": k,
                    "dtype": args.dtype,
                    "ok": False,
                    "kernel_name": None,
                    "error": item["error"],
                }
            else:
                try:
                    row = time_case(item, args)
                except Exception as exc:
                    row = {
                        "backend": backend,
                        "m": m,
                        "n": n,
                        "k": k,
                        "dtype": args.dtype,
                        "ok": False,
                        "kernel_name": None,
                        "error": repr(exc),
                    }
            f.write(json.dumps(row, sort_keys=True) + "\n")
            accuracy = (
                "ERROR"
                if row.get("error")
                else "PASS"
                if row["ok"]
                else "FAIL"
            )
            e2e_ms = (
                f"{row['e2e_ms']:.4f}" if row.get("e2e_ms") is not None else "-"
            )
            e2e_tflops = (
                f"{row['e2e_tflops']:.1f}"
                if row.get("e2e_tflops") is not None
                else "-"
            )
            graph_ms = (
                f"{row['graph_ms']:.4f}"
                if row.get("graph_ms") is not None
                else "-"
            )
            graph_tflops = (
                f"{row['graph_tflops']:.1f}"
                if row.get("graph_tflops") is not None
                else "-"
            )
            max_diff = (
                f"{row['float_max_diff']:.3e}"
                if row.get("float_max_diff") is not None
                else "-"
            )
            config_text = "; ".join(
                ", ".join(f"{k}={v}" for k, v in cfg.items()) or "-"
                for cfg in row.get("kernel_config", {}).values()
            ) or "-"
            shape = f"M={m} N={n} K={k}"
            print(
                f"{backend.upper():<8} {shape:<28} "
                f"{args.dtype:<10} {accuracy:<8} "
                f"{e2e_ms:>10} {e2e_tflops:>12} "
                f"{graph_ms:>10} {graph_tflops:>13} "
                f"{max_diff:>11}  {config_text}",
                flush=True,
            )
            for field, label in (
                ("error", "Error"),
                ("graph_error", "Graph warning"),
                ("kernel_name_error", "Config warning"),
            ):
                if row.get(field):
                    print(f"  {label}: {row[field]}", flush=True)


if __name__ == "__main__":
    main()
