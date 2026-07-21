import torch
import pytest
import itertools
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass

from kernels.hgemm_layout_gfx950 import hgemm, make_hgemm_gfx950_param

ROTARY_INPUTS_TARGET_BYTES = 8 * 1024**3


@dataclass
class _TestArgs:
    dtype: torch.dtype | str
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
    use_half_tile_interleaved: bool = False


def get_torch_fp8_dtype():
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    pytest.skip("This PyTorch build does not expose torch.float8_e4m3fn")


def quantize_ptpc_fp8(x: torch.Tensor):
    fp8_dtype = get_torch_fp8_dtype()
    fp8_max = float(torch.finfo(fp8_dtype).max)
    scale = x.float().abs().amax(dim=1, keepdim=True) / fp8_max
    scale[scale == 0] = 1
    x_fp8 = (x.float() / scale).to(fp8_dtype)
    scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).squeeze(1)
    return x_fp8, scale.float()


def create_inputs(args: _TestArgs):
    if args.dtype == "fp8_ptpc":
        a_f32 = torch.empty((args.m, args.k), dtype=torch.float32, device="cuda")
        b_f32 = torch.empty((args.n, args.k), dtype=torch.float32, device="cuda")
        a_f32.uniform_(-1, 1)
        b_f32.uniform_(-1, 1)
        a, scale_a = quantize_ptpc_fp8(a_f32)
        b, scale_b = quantize_ptpc_fp8(b_f32)
        if args.has_bias:
            bias = torch.empty((args.n,), dtype=torch.bfloat16, device="cuda")
            bias.uniform_(10, 20)
        else:
            bias = None
        return (a, b, scale_a, scale_b, bias)
    else:
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
    dtype = torch.bfloat16 if args.dtype == "fp8_ptpc" else args.dtype
    c = torch.randn((args.m, args.n), dtype=dtype, device="cuda")
    return (c,)


def ref_func(*args):
    if len(args) == 6:
        a, b, scale_a, scale_b, bias, c = args
        a_f32 = a.float() * scale_a[:, None]
        b_f32 = b.float() * scale_b[:, None]
        if bias is not None:
            ref = torch.addmm(bias.float(), a_f32, b_f32.t())
        else:
            ref = torch.mm(a_f32, b_f32.t())
        c.copy_(ref.to(c.dtype))
    else:
        a, b, bias, c = args
        F.linear(a, b, out=c, bias=bias)


def make_triton_maxautotune_func():
    import torch._inductor.config as inductor_config

    inductor_config.max_autotune_gemm_backends = "TRITON"
    inductor_config.max_autotune_gemm_search_space = "EXHAUSTIVE"

    torch._dynamo.reset()

    def triton_maxautotune_func(a, b, bias, c):
        out = F.linear(a, b, bias=bias)
        c.copy_(out)

    return torch.compile(triton_maxautotune_func, mode="max-autotune", fullgraph=True)


def func(*args):
    if len(args) == 7:
        a, b, scale_a, scale_b, bias, c, kwargs = args
        hgemm(a, b, c, bias=bias, scale_a=scale_a, scale_b=scale_b, user_kwargs=kwargs)
    else:
        a, b, bias, c, kwargs = args
        hgemm(a, b, c, bias=bias, user_kwargs=kwargs)


def make_test_args(
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
    use_half_tile_interleaved: bool,
):
    if dtype == "fp8_ptpc":
        block_k = 128
        resolved_dtype = dtype
    else:
        resolved_dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    return _TestArgs(
        resolved_dtype,
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
        use_half_tile_interleaved,
    )


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
        "use_half_tile_interleaved": args.use_half_tile_interleaved,
    }
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = inputs + outputs
    ref_inouts = inputs + ref_outputs
    maxdiff_out_ = []

    def get_tol(args):
        k_scale = (args.k / 8192) ** 0.5
        if args.dtype == "fp8_ptpc":
            return 2e-1 * k_scale, 2e-1
        elif args.dtype is torch.bfloat16:
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
        "use_half_tile_interleaved": args.use_half_tile_interleaved,
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
        if i % 3 == 0:
            run_ref(idx)
            run_triton_maxautotune(idx)
            run_flydsl(idx)
        if i % 3 == 1:
            run_flydsl(idx)
            run_ref(idx)
            run_triton_maxautotune(idx)
        elif i % 3 == 2:
            run_triton_maxautotune(idx)
            run_flydsl(idx)
            run_ref(idx)
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CUDA],
    ) as prof:
        for i in range(warmup, niters):
            idx = i % rotary_inputs
            if i % 3 == 0:
                run_ref(idx)
                run_triton_maxautotune(idx)
                run_flydsl(idx)
            if i % 3 == 1:
                run_flydsl(idx)
                run_ref(idx)
                run_triton_maxautotune(idx)
            elif i % 3 == 2:
                run_triton_maxautotune(idx)
                run_flydsl(idx)
                run_ref(idx)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


@pytest.mark.parametrize(
    "dtype",
    [
        "bf16",
        "fp16",
        # "fp8_ptpc",
    ],
)
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias, is_hti",
    [
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 4, True, False),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, True, False),
        (8192, 8192, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8192, 8192, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, True, False),
        (8192, 8192, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8192, 8192, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, True, False),
        (8160, 8160, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8160, 8160, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, True, False),
        (8160, 8160, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8160, 8160, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, True, False),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 4, 0, False, False),
        (2048, 2048, 2048, 128, 128, 64, 4, 4, 4, 0, True, False),
        (2048, 2048, 2048 - 64, 128, 128, 64, 2, 4, 4, 0, False, False),
        (2048, 2048, 2048 - 64, 128, 128, 64, 4, 4, 4, 0, True, False),
        # hti
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 4, True, True),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, True, True),
        (8192, 8192, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8192, 8192, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, True, True),
        (8192, 8192, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8192, 8192, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, True, True),
        (8160, 8160, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8160, 8160, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, True, True),
        (8160, 8160, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8160, 8160, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, True, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 2, 2, 0, False, True),
        (2048, 2048, 2048 - 64, 128, 128, 64, 2, 2, 2, 0, False, True),
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
    is_hti: bool,
):
    args = make_test_args(
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
        is_hti,
    )
    check_acc(args)


@pytest.mark.parametrize(
    "dtype",
    [
        "bf16",
        "fp16",
        # "fp8_ptpc",
    ],
)
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias, is_hti",
    [
        (3, 5120, 2880, 64, 64, 64, 5, 2, 2, 4, True, False),
        (3, 5120, 2880, 64, 64, 64, 5, 2, 2, 0, False, False),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, True, False),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, False, False),
        (3, 32, 128 + 64, 128, 128, 64, 3, 2, 2, 4, True, False),
        (3, 32, 128 - 64, 128, 128, 64, 3, 2, 2, 0, False, False),
        # hti
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 4, True, True),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, False, True),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, True, True),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, False, True),
        (3, 32, 128 + 64, 128, 128, 64, 2, 2, 2, 4, True, True),
        (3, 32, 128 + 64 * 3, 128, 128, 64, 2, 2, 2, 4, True, True),
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
    is_hti: bool,
):
    args = make_test_args(
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
        is_hti,
    )
    check_acc(args)


@pytest.mark.parametrize(
    "dtype",
    [
        "bf16",
        "fp16",
        # "fp8_ptpc",
    ],
)
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias, is_hti",
    [
        (8, 4096, 4096, 16, 16, 128, 8, 1, 1, 4, True, False),
        (16, 4096, 4096, 16, 16, 128, 8, 1, 1, 4, True, False),
        (32, 4096, 4096, 16, 16, 64, 8, 1, 1, 0, True, False),
        (64, 4096, 4096, 32, 32, 64, 8, 2, 2, 4, True, False),
        (128, 4096, 4096, 64, 32, 128, 4, 4, 2, 4, True, False),
        (256, 4096, 4096, 64, 64, 64, 7, 4, 2, 4, True, False),
        (512, 4096, 4096, 64, 128, 64, 6, 2, 4, 4, True, False),
        (1024, 4096, 4096, 128, 128, 64, 4, 2, 4, 4, True, False),
        (2048, 4096, 4096, 128, 256, 64, 3, 4, 4, 4, True, False),
        (1024, 1024, 1024, 64, 64, 64, 4, 2, 4, 0, True, False),
        (2048, 2048, 2048, 128, 128, 64, 3, 4, 2, 0, True, False),
        (4096, 4096, 4096, 256, 256, 64, 2, 2, 4, 4, True, True),
        (4096, 4096, 8192, 256, 256, 64, 2, 2, 4, 4, True, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, True, True),
        (16384, 16384, 16384, 256, 256, 64, 2, 2, 4, 4, True, True),
        (8, 7168, 2048, 16, 16, 128, 8, 1, 1, 4, True, False),
        (32, 384, 7168, 16, 16, 128, 8, 1, 1, 0, True, False),
        (32, 14336, 4096, 32, 64, 64, 8, 2, 2, 0, True, False),
        (16, 28672, 4096, 16, 64, 128, 3, 1, 4, 4, True, False),
        (4096, 256, 4096, 64, 64, 64, 6, 4, 2, 4, True, False),
    ],
)
def test_hgemm_acc_bench(
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
    is_hti: bool,
):
    args = make_test_args(
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
        is_hti,
    )
    check_acc(args)


# =========================================== benchmark ===========================================


@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias, is_hti",
    [
        (8, 4096, 4096, 16, 16, 128, 8, 1, 1, 4, True, False),
        (16, 4096, 4096, 16, 16, 128, 8, 1, 1, 4, True, False),
        (32, 4096, 4096, 16, 16, 64, 8, 1, 1, 0, True, False),
        (64, 4096, 4096, 32, 32, 64, 8, 2, 2, 4, True, False),
        (128, 4096, 4096, 64, 32, 128, 4, 4, 2, 4, True, False),
        (256, 4096, 4096, 64, 64, 64, 7, 4, 2, 4, True, False),
        (512, 4096, 4096, 64, 128, 64, 6, 2, 4, 4, True, False),
        (1024, 4096, 4096, 128, 128, 64, 4, 2, 4, 4, True, False),
        (2048, 4096, 4096, 128, 256, 64, 3, 4, 4, 4, True, False),
        (1024, 1024, 1024, 64, 64, 64, 4, 2, 4, 0, True, False),
        (2048, 2048, 2048, 128, 128, 64, 3, 4, 2, 0, True, False),
        (4096, 4096, 4096, 256, 256, 64, 2, 2, 4, 4, True, True),
        (4096, 4096, 8192, 256, 256, 64, 2, 2, 4, 4, True, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, True, True),
        (16384, 16384, 16384, 256, 256, 64, 2, 2, 4, 4, True, True),
        (8, 7168, 2048, 16, 16, 128, 8, 1, 1, 4, True, False),
        (32, 384, 7168, 16, 16, 128, 8, 1, 1, 0, True, False),
        (32, 14336, 4096, 32, 64, 64, 8, 2, 2, 0, True, False),
        (16, 28672, 4096, 16, 64, 128, 3, 1, 4, 4, True, False),
        (4096, 256, 4096, 64, 64, 64, 6, 4, 2, 4, True, False),
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
    is_hti: bool,
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
        is_hti,
    )
    benchmark(args)


def hgemm_get_configs():
    selections = {
        "block_m": [16, 32, 48, 64, 80, 96, 128, 256],
        "block_n": [16, 32, 64, 80, 96, 128, 256],
        "block_k": [64, 128, 256],
        "stages": [i for i in range(2, 10)],
        "m_waves": [1, 2, 4],
        "n_waves": [1, 2, 4],
        "group_m": [0, 4],
    }
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    valid_configs = []
    for config in configs:
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


if __name__ == "__main__":
    configs = hgemm_get_configs()
    print(len(configs))
