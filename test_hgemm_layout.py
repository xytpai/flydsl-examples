import torch
import pytest
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass
from flydsl.runtime.device import get_rocm_arch

from kernels.hgemm_layout_gfx950 import hgemm

ROTARY_INPUTS_TARGET_BYTES = 8 * 1024**3


@dataclass
class _TestArgs:
    dtype: torch.dtype
    m: int
    n: int
    k: int
    block_m: int
    block_n: int
    block_k: int
    stages: int
    m_waves: int
    n_waves: int
    group_m: int
    has_bias: bool


def create_inputs(args: _TestArgs):
    a = torch.empty((args.m, args.k), dtype=args.dtype, device="cuda")
    a.uniform_(-1, 1)
    b = torch.empty((args.n, args.k), dtype=args.dtype, device="cuda")
    b.uniform_(-1, 1)
    if args.has_bias:
        bias = torch.empty((args.n,), dtype=args.dtype, device="cuda")
        bias.uniform_(10, 20)
    else:
        bias = None
    return (a, b, bias)


def create_outputs(args: _TestArgs):
    c = torch.randn((args.m, args.n), dtype=args.dtype, device="cuda")
    return (c,)


def ref_func(*args):
    a, b, bias, c = args
    F.linear(a, b, out=c, bias=bias)


def make_triton_maxautotune_func():
    import torch._inductor.config as inductor_config

    inductor_config.max_autotune_gemm_backends = "TRITON"

    def triton_maxautotune_func(a, b, bias, c):
        out = F.linear(a, b, bias=bias)
        c.copy_(out)

    return torch.compile(triton_maxautotune_func, mode="max-autotune", fullgraph=True)


def func(*args):
    a, b, bias, c, kwargs = args
    hgemm(a, b, c, bias=bias, user_kwargs=kwargs)


def tensor_nbytes(tensors: torch.Tensor):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def get_rotary_inputs(sample_inputs: torch.Tensor, sample_outputs: torch.Tensor):
    global ROTARY_INPUTS_TARGET_BYTES
    slot_bytes = 2 * (tensor_nbytes(sample_inputs) + tensor_nbytes(sample_outputs))
    rotary_inputs = ROTARY_INPUTS_TARGET_BYTES // slot_bytes
    return max(1, int(rotary_inputs))


def check_acc(args: _TestArgs):
    kwargs = {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "stages": args.stages,
        "m_waves": args.m_waves,
        "n_waves": args.n_waves,
        "group_m": args.group_m,
    }
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = inputs + outputs
    ref_inouts = inputs + ref_outputs
    maxdiff_out_ = []

    def get_tol(args):
        k_scale = (args.k / 8192) ** 0.5
        if args.dtype is torch.bfloat16:
            return 2e-1 * k_scale, 2e-1
        return 5e-2 * k_scale, 5e-2

    atol, rtol = get_tol(args)
    for _ in range(5):
        func(*(inouts + (kwargs,)))
        ref_func(*ref_inouts)
        for output, ref_output in zip(outputs, ref_outputs):
            maxdiff_out = (output - ref_output).abs().max().item()
            maxdiff_out_.append(maxdiff_out)
            print(maxdiff_out, flush=True)
            torch.testing.assert_close(
                output,
                ref_output,
                atol=atol,
                rtol=rtol,
                check_dtype=True,
            )
    print(f"\n{args}\nmaxdiff_out:{maxdiff_out_}")


def benchmark(args: _TestArgs, warmup: int = 500, niters: int = 600):
    kwargs = {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "stages": args.stages,
        "m_waves": args.m_waves,
        "n_waves": args.n_waves,
        "group_m": args.group_m,
    }
    sample_inputs = create_inputs(args)
    sample_outputs = create_outputs(args)
    rotary_inputs = get_rotary_inputs(sample_inputs, sample_outputs)
    inputs = [sample_inputs] + [create_inputs(args) for _ in range(rotary_inputs - 1)]
    ref_inputs = [create_inputs(args) for _ in range(rotary_inputs)]
    outputs = [sample_outputs] + [
        create_outputs(args) for _ in range(rotary_inputs - 1)
    ]
    ref_outputs = [create_outputs(args) for _ in range(rotary_inputs)]
    triton_maxautotune_func = make_triton_maxautotune_func()
    global ROTARY_INPUTS_TARGET_BYTES
    print(
        f"rotary_inputs:{rotary_inputs}, target_bytes:{ROTARY_INPUTS_TARGET_BYTES}, "
        f"warmup:{warmup}, niters:{niters}"
    )

    def run_ref(idx):
        ref_func(*(ref_inputs[idx] + ref_outputs[idx]))

    def run_triton_maxautotune(idx):
        triton_maxautotune_func(*(ref_inputs[idx] + ref_outputs[idx]))

    def run_flydsl(idx):
        func(*(inputs[idx] + outputs[idx] + (kwargs,)))

    print("===================== [INTERLEAVED] =====================")
    for i in range(warmup):
        idx = i % rotary_inputs
        if i % 2 == 0:
            run_ref(idx)
            run_triton_maxautotune(idx)
            run_flydsl(idx)
        else:
            run_flydsl(idx)
            run_triton_maxautotune(idx)
            run_ref(idx)
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CUDA],
    ) as prof:
        for i in range(warmup, niters):
            idx = i % rotary_inputs
            if i % 2 == 0:
                run_ref(idx)
                run_triton_maxautotune(idx)
                run_flydsl(idx)
            else:
                run_flydsl(idx)
                run_triton_maxautotune(idx)
                run_ref(idx)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


@pytest.mark.parametrize(
    "dtype",
    [
        "bf16",
    ],
)
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias",
    [
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, False),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 4, True),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, False),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, True),
        (8192, 8192, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, False),
        (8192, 8192, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, True),
        (8160, 8160, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, False),
        (8160, 8160, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 4, 0, False),
        (2048, 2048, 2048, 128, 128, 64, 4, 4, 4, 0, True),
        (2048, 2048, 2048 - 64, 128, 128, 64, 2, 4, 4, 0, False),
        (2048, 2048, 2048 - 64, 128, 128, 64, 4, 4, 4, 0, True),
    ],
)
def test_hgemm_acc_main_loop(
    dtype: str,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
):
    dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        group_m,
        has_bias,
    )
    check_acc(args)


@pytest.mark.parametrize(
    "dtype",
    [
        "bf16",
    ],
)
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias",
    [
        (3, 5120, 2880, 64, 64, 64, 5, 2, 2, 4, True),
        (3, 5120, 2880, 64, 64, 64, 5, 2, 2, 0, False),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, True),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, False),
    ],
)
def test_hgemm_acc_small_m(
    dtype: str,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
):
    dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        group_m,
        has_bias,
    )
    check_acc(args)


# =========================================== benchmark ===========================================


@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias",
    [
        (8192, 8192, 8192, 256, 256, 64, 2, 4, 4, 0, True),
        (4096, 4096, 4096, 256, 256, 64, 2, 2, 4, 4, True),
        (4096, 4096, 8192, 256, 256, 64, 2, 2, 4, 4, True),
        (2048, 2048, 2048, 128, 128, 64, 5, 2, 4, 0, True),
        (1024, 1024, 1024, 64, 64, 64, 6, 4, 4, 0, True),
        (8, 7168, 2048, 16, 32, 128, 6, 1, 2, 0, True),
    ],
)
def test_hgemm_benchmark_smoke(
    dtype: str,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
):
    dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        group_m,
        has_bias,
    )
    benchmark(args)
