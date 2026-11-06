import torch
import pytest
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass

from kernels.gemm_a16w16_tdm_gfx1250 import (
    GEMM_DTYPE_BF16,
    gemm_a16w16_gfx1250,
    gemm_fp8_ptpc_gfx1250,
    validate_gemm_gfx1250_config,
)

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
    layout: str = "nt"
    split_k: int = 1


@dataclass
class _Fp8PtpcArgs:
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
    layout: str = "nt"
    split_k: int = 1


def _require_gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx1250":
        pytest.skip(f"requires gfx1250, got {arch}")


@pytest.fixture(autouse=True)
def gfx1250_device():
    _require_gfx1250()


def empty_layout_matrix(rows: int, cols: int, dtype: torch.dtype, is_t: bool):
    if is_t:
        return torch.empty((cols, rows), dtype=dtype, device="cuda").t()
    return torch.empty((rows, cols), dtype=dtype, device="cuda")


def create_inputs(args: _TestArgs):
    a = empty_layout_matrix(args.m, args.k, args.dtype, args.layout[0] == "t")
    a.uniform_(-1, 1)
    b = empty_layout_matrix(args.k, args.n, args.dtype, args.layout[1] == "t")
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
    a, b, bias, c, _layout = args
    # Round the accumulator to the output dtype before adding bias.
    product = (a.float() @ b.float()).to(a.dtype)
    if bias is not None:
        product = (product.float() + bias.float()).to(a.dtype)
    c.copy_(product)


def make_triton_maxautotune_func():
    import torch._inductor.config as inductor_config

    inductor_config.max_autotune_gemm_backends = "TRITON"
    inductor_config.max_autotune_gemm_search_space = "DEFAULT"

    torch._dynamo.reset()

    def triton_maxautotune_func(a, b, bias, c):
        out = torch.mm(a, b)
        if bias is not None:
            out = out + bias
        c.copy_(out)

    return torch.compile(triton_maxautotune_func, mode="max-autotune", fullgraph=True)


def _kernel_kwargs(args: _TestArgs):
    return {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "stages": args.stages,
        "m_waves": args.m_waves,
        "n_waves": args.n_waves,
        "group_m": args.group_m,
        "split_k": args.split_k,
    }


def func(*args):
    a, b, bias, c, kwargs, layout = args
    gemm_a16w16_gfx1250(a, b, c, bias=bias, layout=layout, **kwargs)


def tensor_nbytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def get_rotary_inputs(sample_inputs, sample_outputs):
    global ROTARY_INPUTS_TARGET_BYTES
    slot_bytes = 2 * (tensor_nbytes(sample_inputs) + tensor_nbytes(sample_outputs))
    rotary_inputs = ROTARY_INPUTS_TARGET_BYTES // slot_bytes
    return max(1, int(rotary_inputs))


def check_acc(args: _TestArgs):
    kwargs = _kernel_kwargs(args)
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = inputs + outputs
    ref_inouts = inputs + ref_outputs
    maxdiff_out_ = []

    def get_tol(args):
        k_scale = (args.k / 8192) ** 0.5
        k_scale *= args.split_k
        atol_scale = 1.5 if args.has_bias else 1.0
        if args.dtype is torch.bfloat16:
            return 2e-1 * k_scale * atol_scale, 2e-1
        return 5e-2 * k_scale * atol_scale, 5e-2

    atol, rtol = get_tol(args)
    for _ in range(5):
        func(*(inouts + (kwargs, args.layout)))
        ref_func(*(ref_inouts + (args.layout,)))
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
    kwargs = _kernel_kwargs(args)
    sample_inputs = create_inputs(args)
    sample_outputs = create_outputs(args)
    rotary_inputs = get_rotary_inputs(sample_inputs, sample_outputs)
    inputs = [sample_inputs] + [create_inputs(args) for _ in range(rotary_inputs - 1)]
    ref_inputs = [create_inputs(args) for _ in range(rotary_inputs)]
    outputs = [sample_outputs] + [create_outputs(args) for _ in range(rotary_inputs - 1)]
    ref_outputs = [create_outputs(args) for _ in range(rotary_inputs)]
    triton_maxautotune_func = make_triton_maxautotune_func()
    global ROTARY_INPUTS_TARGET_BYTES
    print(
        f"rotary_inputs:{rotary_inputs}, target_bytes:{ROTARY_INPUTS_TARGET_BYTES}, "
        f"warmup:{warmup}, niters:{niters}"
    )

    def run_ref(idx):
        ref_func(*(ref_inputs[idx] + ref_outputs[idx] + (args.layout,)))

    def run_triton_maxautotune(idx):
        triton_maxautotune_func(*(ref_inputs[idx] + ref_outputs[idx]))

    def run_flydsl(idx):
        func(*(inputs[idx] + outputs[idx] + (kwargs, args.layout)))

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

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
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


def _fp8_ptpc_kernel_kwargs(args: _Fp8PtpcArgs):
    return {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "stages": args.stages,
        "m_waves": args.m_waves,
        "n_waves": args.n_waves,
        "group_m": args.group_m,
        "split_k": args.split_k,
    }


def _fp8_layout_matrix(rows: int, cols: int, is_t: bool):
    shape = (cols, rows) if is_t else (rows, cols)
    storage = torch.empty(shape, dtype=torch.float16, device="cuda")
    storage.uniform_(-0.5, 0.5)
    matrix = storage.to(torch.float8_e4m3fn)
    return matrix.t() if is_t else matrix


def create_fp8_ptpc_inputs(args: _Fp8PtpcArgs):
    a = _fp8_layout_matrix(args.m, args.k, args.layout[0] == "t")
    b = _fp8_layout_matrix(args.k, args.n, args.layout[1] == "t")
    scale_a = torch.empty((args.m,), dtype=torch.float32, device="cuda")
    scale_b = torch.empty((args.n,), dtype=torch.float32, device="cuda")
    scale_a.uniform_(0.5, 1.5)
    scale_b.uniform_(0.5, 1.5)
    if args.has_bias:
        bias = torch.empty((args.n,), dtype=torch.bfloat16, device="cuda")
        bias.uniform_(-0.5, 0.5)
    else:
        bias = None
    return a, b, scale_a, scale_b, bias


def create_fp8_ptpc_output(args: _Fp8PtpcArgs):
    return torch.empty((args.m, args.n), dtype=torch.bfloat16, device="cuda")


def fp8_ptpc_ref_func(a, b, scale_a, scale_b, bias, split_k):
    if split_k == 1:
        product = a.float() @ b.float()
    else:
        part_k = a.shape[1] // split_k
        partials = [
            (
                a[:, part * part_k : (part + 1) * part_k].float()
                @ b[part * part_k : (part + 1) * part_k].float()
            )
            .to(torch.bfloat16)
            .float()
            for part in range(split_k)
        ]
        product = partials[0]
        for partial in partials[1:]:
            product = product + partial
    product = product * scale_a[:, None]
    product = product * scale_b[None, :]
    result = product.to(torch.bfloat16)
    if bias is not None:
        result = (result.float() + bias.float()).to(torch.bfloat16)
    return result


def benchmark_fp8_ptpc(
    args: _Fp8PtpcArgs, warmup: int = 500, niters: int = 600
):
    kwargs = _fp8_ptpc_kernel_kwargs(args)
    sample_inputs = create_fp8_ptpc_inputs(args)
    sample_output = create_fp8_ptpc_output(args)
    rotary_inputs = get_rotary_inputs(sample_inputs, (sample_output,))
    inputs = [sample_inputs] + [
        create_fp8_ptpc_inputs(args) for _ in range(rotary_inputs - 1)
    ]
    outputs = [sample_output] + [
        create_fp8_ptpc_output(args) for _ in range(rotary_inputs - 1)
    ]
    print(
        f"rotary_inputs:{rotary_inputs}, target_bytes:{ROTARY_INPUTS_TARGET_BYTES}, "
        f"warmup:{warmup}, niters:{niters}"
    )

    def run_flydsl(idx):
        a, b, scale_a, scale_b, bias = inputs[idx]
        gemm_fp8_ptpc_gfx1250(
            a,
            b,
            scale_a,
            scale_b,
            out=outputs[idx],
            bias=bias,
            layout=args.layout,
            **kwargs,
        )

    print("===================== [FP8 PTPC FLYDSL] =====================")
    for i in range(warmup):
        run_flydsl(i % rotary_inputs)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(warmup, niters):
            run_flydsl(i % rotary_inputs)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


def _dtype_from_name(dtype: str) -> torch.dtype:
    return torch.bfloat16 if "bf16" in dtype else torch.half


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
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
        (2048, 2048, 2048, 128, 128, 64, 2, 2, 2, 0, False),
    ],
)
def test_gemm_a16w16_gfx1250_acc_main_loop(
    layout: str,
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
    args = _TestArgs(
        _dtype_from_name(dtype),
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
        layout,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, split_k, "
    "m_waves, n_waves, has_bias, group_m",
    [
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, True, 0),
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, False, 0),
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, True, 4),
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, False, 4),
        (3, 5120, 2880, 64, 64, 64, 5, 3, 2, 2, True, 0),
        (3, 5120, 2880, 64, 64, 64, 5, 3, 2, 2, False, 0),
        (800, 384, 7168, 32, 64, 128, 6, 1, 1, 2, True, 0),
        (800, 384, 7168, 32, 64, 128, 6, 1, 1, 2, False, 0),
        (800, 384, 7168, 32, 64, 128, 6, 2, 1, 2, True, 0),
        (800, 384, 7168, 32, 64, 128, 6, 2, 1, 2, False, 0),
        (800, 384, 7168, 32, 64, 128, 6, 2, 1, 2, False, 4),
        (64, 64, 512, 64, 64, 64, 2, 2, 2, 2, True, 0),
        (64, 64, 512, 64, 64, 64, 2, 2, 2, 2, False, 0),
        (65, 96, 4096, 64, 64, 256, 2, 4, 2, 2, False, 0),
        (96, 64, 7168, 64, 64, 256, 2, 7, 2, 2, False, 0),
        (128, 64, 4096, 64, 64, 256, 2, 8, 2, 2, False, 0),
        (128, 128, 6144, 64, 64, 256, 2, 6, 2, 2, False, 0),
        (65, 136, 2048, 64, 128, 128, 2, 4, 2, 4, False, 0),
        (129, 72, 1024, 128, 64, 128, 2, 2, 2, 2, True, 0),
        (129, 136, 1536, 128, 128, 128, 2, 3, 2, 4, False, 4),
        (257, 264, 512, 256, 256, 64, 2, 2, 2, 4, False, 0),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, True, 0),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, False, 0),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, True, 4),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, False, 4),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, True, 0),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, False, 0),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, True, 4),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, False, 4),
        (4096, 128, 6144, 64, 64, 256, 2, 2, 2, 2, False, 0),
        (512, 2560, 6144, 128, 128, 128, 2, 3, 2, 4, False, 4),
        (576, 256, 4096, 64, 64, 256, 2, 4, 2, 2, True, 0),
        (1024, 128, 6144, 64, 64, 256, 2, 6, 2, 2, True, 0),
        (1216, 64, 7168, 64, 64, 256, 2, 7, 2, 2, False, 0),
        (1280, 64, 4096, 64, 64, 256, 2, 8, 2, 2, False, 0),
        (1024, 64, 4096, 64, 64, 256, 2, 8, 2, 2, False, 0),
        (1272, 64, 4096, 64, 64, 256, 2, 8, 2, 2, False, 0),
        (1280, 72, 4096, 64, 64, 256, 2, 8, 2, 2, False, 0),
        (1280, 128, 4096, 64, 64, 256, 2, 8, 2, 2, False, 0),
        (1280, 64, 4096, 64, 64, 256, 2, 8, 2, 2, True, 0),
        (1280, 64, 4096, 64, 64, 128, 2, 8, 2, 2, False, 0),
        (1280, 64, 4096, 64, 64, 256, 2, 4, 2, 2, False, 0),
    ],
)
def test_gemm_a16w16_gfx1250_acc_split_k(
    layout: str,
    dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    split_k: int,
    m_waves: int,
    n_waves: int,
    has_bias: bool,
    group_m: int,
):
    args = _TestArgs(
        dtype=dtype,
        m=m,
        n=n,
        k=k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        has_bias=has_bias,
        layout=layout,
        split_k=split_k,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, split_k, m_waves, n_waves, group_m",
    [
        # fmt: off
        (   8,  4096,  4096,  16,  16, 256, 4,  1,  1,  1,  4),
        (  16,  4096,  4096,  16,  16, 256, 4,  1,  1,  1,  4),
        (  32,  4096,  4096,  16,  32, 256, 4,  1,  1,  2,  4),
        (  64,  4096,  4096,  32,  32, 128, 6,  1,  2,  2,  4),
        ( 128,  4096,  4096,  32,  64, 128, 4,  1,  1,  4,  0),
        ( 256,  4096,  4096,  64,  64, 128, 4,  1,  4,  2,  0),
        ( 512,  4096,  4096,  64, 128,  64, 5,  1,  2,  4,  4),
        (1024,  4096,  4096, 128, 128,  64, 4,  1,  2,  4,  4),
        (2048,  4096,  4096, 256, 128,  64, 3,  1,  4,  2,  0),
        (1024,  1024,  1024,  64,  64,  64, 4,  1,  1,  4,  4),
        (2048,  2048,  2048, 128, 128,  64, 4,  1,  2,  4,  0),
        (4096,  4096,  4096, 256, 256,  64, 2,  1,  2,  4,  4),
        (4096,  4096,  8192, 256, 256,  64, 2,  1,  2,  4,  4),
        (8192,  8192,  8192, 256, 256,  64, 2,  1,  2,  4,  4),
        (   8,  7168,  2048,  16,  16,  64, 8,  1,  1,  1,  4),
        (  32,   384,  7168,  16,  16, 256, 4,  1,  1,  1,  0),
        (  32, 14336,  4096,  32,  64, 128, 5,  1,  2,  2,  4),
        (  16, 28672,  4096,  16,  64, 256, 2,  1,  1,  2,  4),
        (4096,   256,  4096,  64,  64, 128, 4,  1,  4,  2,  4),
        (   1,  5120,  2880,  16,  64,  64, 7,  3,  1,  2,  0),
        (   2,  5120,  2880,  16,  64,  64, 7,  3,  1,  2,  0),
        (   4,  5120,  2880,  16,  64,  64, 7,  3,  1,  2,  0),
        (   8,  5120,  2880,  16,  64,  64, 8,  3,  1,  2,  0),
        (  16,  5120,  2880,  16,  64,  64, 8,  3,  1,  2,  0),
        (  32,  5120,  2880,  16,  32,  64, 8,  1,  1,  2,  0),
        (  48,  5120,  2880,  16,  64,  64, 8,  1,  1,  2,  0),
        (  32,   384,  7168,  32,  32,  64, 8,  8,  2,  1,  0),
        (  32,   384, 16384,  32,  32,  64, 8,  8,  1,  2,  0),
        (  32,  7168,  2048,  32,  32,  64, 8,  1,  2,  1,  0),
        (  32,  2880,  2048,  32,  32,  64, 8,  2,  1,  2,  0),
        (  64,  4096,  4096,  32,  32,  64, 8,  1,  1,  2,  0),
        ( 128,  4096,  4096,  64,  32, 128, 4,  1,  4,  1,  4),
        ( 256,  4096,  4096,  64,  64,  64, 8,  1,  2,  2,  0),
        ( 512,  4096,  4096,  64, 128,  64, 6,  1,  2,  4,  4),
        (1024,  1024,  1024,  64,  64,  64, 4,  1,  2,  4,  4),
        (2048,  2048,  2048, 128, 128,  64, 4,  1,  2,  4,  4),
        (8192,  8192,  8192, 256, 256,  64, 2,  1,  2,  4,  0),
        (   8,  7168,  2048,  16,  16, 128, 6,  1,  1,  1,  4),
        (  32, 14336,  4096,  32,  64, 256, 3,  1,  1,  2,  4),
        (  16, 28672,  4096,  16,  32, 256, 3,  1,  1,  2,  4),
        (4096,   256,  4096,  64,  64,  64, 8,  1,  2,  2,  4),
        (   1,  5120,  2880,  16,  64,  64, 9,  3,  1,  2,  0),
        (   2,  5120,  2880,  16,  64,  64, 8,  3,  1,  2,  0),
        (   4,  5120,  2880,  16,  64,  64, 8,  3,  1,  2,  0),
        (   8,  5120,  2880,  16,  64,  64, 9,  3,  1,  2,  0),
        (  16,  5120,  2880,  16,  64,  64, 5,  3,  1,  2,  0),
        (  32,   384,  7168,  32,  32,  64, 8,  8,  1,  2,  0),
        (   8,  7168,  2048,  16,  16, 128, 7,  1,  1,  1,  4),
        (  32,  2880,  2048,  32,  64,  64, 8,  4,  2,  2,  0),
        (   8,  4096,  4096,  16, 128, 256, 2,  8,  1,  4,  0),
        (  16,  4096,  4096,  16,  64, 128, 5,  8,  1,  4,  0),
        (  32,  4096,  4096,  32,  64,  64, 4,  8,  2,  4,  0),
        (  64,  4096,  4096,  64,  64, 256, 3,  4,  4,  4,  0),
        ( 128,  4096,  4096, 128,  64, 128, 5,  4,  4,  2,  0),
        ( 256,  4096,  4096, 128,  64, 256, 3,  2,  4,  4,  0),
        ( 512,  4096,  4096, 256, 128, 128, 3,  4,  4,  4,  0),
        (1024,  4096,  4096, 256, 128, 128, 2,  2,  4,  4,  0),
        (2048,  4096,  4096, 128, 256, 128, 3,  1,  4,  4,  4),
        (1024,  1024,  1024,  64,  64, 128, 3,  1,  4,  4,  0),
        (2048,  2048,  2048, 128, 128, 256, 2,  1,  4,  4,  4),
        (4096,  4096,  4096, 256, 256, 128, 2,  1,  4,  2,  4),
        (4096,  4096,  8192, 256, 256, 128, 2,  1,  4,  2,  4),
        (8192,  8192,  8192, 256, 256, 128, 2,  1,  4,  2,  0),
        (   8,  7168,  2048,  16,  64, 128, 3,  4,  1,  2,  0),
        (  32,   384,  7168,  16,  32, 128, 7,  8,  1,  2,  0),
        (  32, 14336,  4096,  32, 128, 128, 3,  4,  2,  4,  0),
        (  16, 28672,  4096,  16, 128, 128, 6,  1,  1,  4,  0),
        (4096,   256,  4096, 128, 128, 128, 4,  4,  4,  4,  0),
        (   8,  4096,  4096,  16,  64, 128, 3,  8,  1,  2,  0),
        (  16,  4096,  4096,  16,  64, 128, 5,  8,  1,  4,  0),
        (  32,  4096,  4096,  32,  64, 128, 3,  8,  2,  4,  0),
        (  64,  4096,  4096,  32,  64, 128, 5,  4,  2,  4,  0),
        ( 128,  4096,  4096, 128,  64, 128, 3,  4,  4,  4,  0),
        ( 256,  4096,  4096, 128, 128, 128, 3,  4,  4,  4,  0),
        ( 512,  4096,  4096, 128, 128, 128, 3,  2,  4,  4,  0),
        (1024,  4096,  4096, 128, 256, 128, 3,  2,  4,  4,  0),
        (2048,  4096,  4096, 128, 256, 128, 3,  1,  4,  4,  4),
        (1024,  1024,  1024, 128,  64, 256, 3,  2,  4,  2,  0),
        (2048,  2048,  2048, 128, 128, 128, 3,  1,  4,  4,  4),
        (4096,  4096,  4096, 256, 256, 128, 2,  1,  4,  2,  0),
        (4096,  4096,  8192, 256, 256, 128, 2,  1,  4,  2,  4),
        (8192,  8192,  8192, 256, 256, 128, 2,  1,  4,  2,  0),
        (   8,  7168,  2048,  16,  64, 256, 3,  2,  1,  4,  0),
        (  32,   384,  7168,  16,  64, 512, 3,  7,  1,  4,  0),
        (  32, 14336,  4096,  32,  64, 128, 4,  2,  2,  2,  0),
        (  16, 28672,  4096,  16,  64, 256, 3,  1,  1,  4,  4),
        (4096,   256,  4096, 128, 128, 128, 3,  4,  4,  4,  0),
        # fmt: on
    ],
)
def test_gemm_a16w16_gfx1250_acc_tuned_policies(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    split_k: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    layout: str,
):
    args = _TestArgs(
        dtype=torch.bfloat16,
        m=m,
        n=n,
        k=k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        has_bias=True,
        layout=layout,
        split_k=split_k,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias",
    [
        (3, 5120, 2880, 64, 64, 64, 5, 2, 2, 4, True),
        (3, 5120, 2880, 64, 64, 64, 5, 2, 2, 0, False),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, True),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, False),
        (3, 32, 128 + 64, 128, 128, 64, 3, 2, 2, 4, True),
    ],
)
def test_gemm_a16w16_gfx1250_acc_small_m(
    layout: str,
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
    args = _TestArgs(
        _dtype_from_name(dtype),
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
        layout,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias",
    [
        (8, 4096, 4096, 16, 16, 128, 8, 1, 1, 4, True),
        (16, 4096, 4096, 16, 16, 128, 8, 1, 1, 4, True),
        (32, 4096, 4096, 16, 16, 64, 8, 1, 1, 0, True),
        (64, 4096, 4096, 32, 32, 64, 8, 2, 2, 4, True),
        (128, 4096, 4096, 64, 32, 128, 4, 4, 2, 4, True),
        (256, 4096, 4096, 64, 64, 64, 7, 4, 2, 4, True),
        (512, 4096, 4096, 64, 128, 64, 6, 2, 4, 4, True),
        (1024, 4096, 4096, 128, 128, 64, 4, 2, 4, 4, True),
        (2048, 4096, 4096, 128, 256, 64, 3, 4, 4, 4, True),
        (1024, 1024, 1024, 64, 64, 64, 4, 2, 4, 0, True),
        (2048, 2048, 2048, 128, 128, 64, 3, 4, 2, 0, True),
        (4096, 4096, 4096, 256, 256, 64, 2, 2, 4, 4, True),
        (4096, 4096, 8192, 256, 256, 64, 2, 2, 4, 4, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, True),
        (16384, 16384, 16384, 256, 256, 64, 2, 2, 4, 4, True),
        (8, 7168, 2048, 16, 16, 128, 8, 1, 1, 4, True),
        (32, 384, 7168, 16, 16, 128, 8, 1, 1, 0, True),
        (32, 14336, 4096, 32, 64, 64, 8, 2, 2, 0, True),
        (16, 28672, 4096, 16, 64, 128, 3, 1, 4, 4, True),
        (4096, 256, 4096, 64, 64, 64, 6, 4, 2, 4, True),
    ],
)
def test_gemm_a16w16_gfx1250_acc_bench(
    layout: str,
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
    args = _TestArgs(
        _dtype_from_name(dtype),
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
        layout,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("split_k", [1, 2])
def test_gemm_a16w16_gfx1250_padded_stride_and_storage_offset(layout: str, split_k: int):
    m = n = 64
    k = 256
    dtype = torch.bfloat16
    column_offset = 8

    def make_padded_matrix(rows: int, cols: int, padding: int, is_t: bool):
        if is_t:
            pitch = rows + padding
            storage = torch.empty((cols + 1, pitch), dtype=dtype, device="cuda")
            storage.uniform_(-1, 1)
            return storage[1:, column_offset : column_offset + rows].t()
        pitch = cols + padding
        storage = torch.empty((rows + 1, pitch), dtype=dtype, device="cuda")
        storage.uniform_(-1, 1)
        return storage[1:, column_offset : column_offset + cols]

    a = make_padded_matrix(m, k, 32, layout[0] == "t")
    b = make_padded_matrix(k, n, 48, layout[1] == "t")

    assert a.shape == (m, k)
    assert b.shape == (k, n)
    for tensor, is_t in zip((a, b), (layout[0] == "t", layout[1] == "t")):
        assert not tensor.is_contiguous()
        if is_t:
            assert tensor.stride(0) == 1
            assert tensor.stride(1) > tensor.shape[0]
            leading_stride = tensor.stride(1)
        else:
            assert tensor.stride(1) == 1
            assert tensor.stride(0) > tensor.shape[1]
            leading_stride = tensor.stride(0)
        assert tensor.storage_offset() > 0
        assert tensor.data_ptr() % 16 == 0
        assert leading_stride * tensor.element_size() % 16 == 0

    bias = torch.empty((n,), dtype=dtype, device="cuda").uniform_(-1, 1)
    out = torch.empty((m, n), dtype=dtype, device="cuda")
    ref = torch.empty_like(out)
    kwargs = {
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "stages": 2,
        "split_k": split_k,
        "m_waves": 2,
        "n_waves": 2,
        "group_m": 0,
    }

    result = gemm_a16w16_gfx1250(a, b, out, bias=bias, layout=layout, **kwargs)
    ref_func(a, b, bias, ref, layout)

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        out,
        ref,
        atol=1e-1 if split_k > 1 else 3e-2,
        rtol=2e-1,
        check_dtype=True,
    )


def test_gemm_a16w16_gfx1250_allocates_output_with_default_policy():
    m = n = 64
    k = 256
    dtype = torch.bfloat16
    a = torch.randn((m, k), dtype=dtype, device="cuda")
    b = empty_layout_matrix(k, n, dtype, is_t=True)
    b.normal_()

    out = gemm_a16w16_gfx1250(a, b)
    ref = torch.empty_like(out)
    ref_func(a, b, None, ref, "nt")

    assert out.shape == (m, n)
    assert out.dtype == dtype
    assert out.device == a.device
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=2e-1)


@pytest.mark.parametrize(
    ("operand", "layout"),
    [
        ("A", "nn"),
        ("B", "nn"),
        ("B", "nt"),
    ],
)
def test_gemm_a16w16_gfx1250_rejects_unsupported_input_strides(operand: str, layout: str):
    m = n = 64
    k = 256
    dtype = torch.bfloat16
    a = empty_layout_matrix(m, k, dtype, is_t=layout[0] == "t")
    b = empty_layout_matrix(k, n, dtype, is_t=layout[1] == "t")
    a.normal_()
    b.normal_()

    if operand == "A":
        a = torch.randn((m, k * 2), dtype=dtype, device="cuda")[:, ::2]
    else:
        if layout[1] == "t":
            b = torch.randn((k, n), dtype=dtype, device="cuda")
        else:
            b = torch.randn((k, n * 2), dtype=dtype, device="cuda")[:, ::2]

    with pytest.raises(ValueError, match=rf"^{operand} does not satisfy"):
        gemm_a16w16_gfx1250(a, b, layout=layout)


def test_gemm_a16w16_gfx1250_rejects_noncontiguous_bias():
    m = n = 64
    k = 256
    dtype = torch.bfloat16
    a = torch.randn((m, k), dtype=dtype, device="cuda")
    b = empty_layout_matrix(k, n, dtype, is_t=True)
    b.normal_()
    bias = torch.randn((n * 2,), dtype=dtype, device="cuda")[::2]
    out = torch.empty((m, n), dtype=dtype, device="cuda")
    kwargs = {
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "stages": 2,
        "m_waves": 2,
        "n_waves": 2,
        "group_m": 0,
    }

    assert not bias.is_contiguous()
    with pytest.raises(ValueError, match="bias must"):
        gemm_a16w16_gfx1250(a, b, out, bias=bias, layout="nt", **kwargs)


@pytest.mark.parametrize("n", [4096, 4032])
def test_gemm_a16w16_gfx1250_block_swizzle_boundary_paths(n: int):
    args = _TestArgs(
        dtype=torch.bfloat16,
        m=320,
        n=n,
        k=256,
        block_m=64,
        block_n=64,
        block_k=64,
        stages=2,
        m_waves=2,
        n_waves=2,
        group_m=4,
        has_bias=False,
        layout="nt",
    )
    check_acc(args)


def test_gemm_a16w16_gfx1250_wide_k_tile():
    args = _TestArgs(
        dtype=torch.bfloat16,
        m=64,
        n=128,
        k=1024,
        block_m=32,
        block_n=64,
        block_k=128,
        stages=4,
        m_waves=1,
        n_waves=2,
        group_m=0,
        has_bias=True,
        layout="nt",
    )
    check_acc(args)


def test_gemm_a16w16_gfx1250_split_k_uses_two_streams():
    args = _TestArgs(
        dtype=torch.bfloat16,
        m=64,
        n=64,
        k=512,
        block_m=64,
        block_n=64,
        block_k=64,
        stages=2,
        m_waves=2,
        n_waves=2,
        group_m=0,
        has_bias=True,
        layout="nt",
        split_k=2,
    )
    a, b, bias = create_inputs(args)
    out0 = create_outputs(args)[0]
    out1 = create_outputs(args)[0]
    ref = create_outputs(args)[0]
    kwargs = _kernel_kwargs(args)
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()

    gemm_a16w16_gfx1250(a, b, out0, bias=bias, layout=args.layout, **kwargs)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream0):
        gemm_a16w16_gfx1250(
            a, b, out0, bias=bias, layout=args.layout, stream=stream0, **kwargs
        )
    with torch.cuda.stream(stream1):
        gemm_a16w16_gfx1250(
            a, b, out1, bias=bias, layout=args.layout, stream=stream1, **kwargs
        )
    stream0.synchronize()
    stream1.synchronize()
    ref_func(a, b, bias, ref, args.layout)

    torch.testing.assert_close(out0, ref, atol=2e-1, rtol=2e-1)
    torch.testing.assert_close(out1, ref, atol=2e-1, rtol=2e-1)


def test_gemm_a16w16_gfx1250_rejects_unsupported_k_partitioning():
    m = n = 64
    k = 480
    dtype = torch.bfloat16
    a = torch.randn((m, k), dtype=dtype, device="cuda")
    b = empty_layout_matrix(k, n, dtype, is_t=True)
    b.normal_()
    kwargs = {
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "stages": 2,
        "m_waves": 2,
        "n_waves": 2,
        "group_m": 0,
    }

    with pytest.raises(ValueError, match="K must be divisible"):
        gemm_a16w16_gfx1250(a, b, layout="nt", **kwargs)


@pytest.mark.parametrize(
    "changed,message",
    [
        ({"stages": 1}, "stages"),
        ({"stages": 34}, "stages"),
        ({"k": 32, "stages": 3}, "prefetch"),
        ({"k": 40}, "K must"),
        ({"block_k": 48}, "power of two"),
        (
            {
                "block_m": 256,
                "block_n": 256,
                "block_k": 128,
                "stages": 3,
                "m_waves": 4,
                "n_waves": 4,
            },
            "LDS",
        ),
        ({"m_waves": 0}, "positive"),
        ({"group_m": -1}, "nonnegative"),
    ],
)
def test_config_rejections(changed, message):
    config = {
        "m": 128,
        "n": 256,
        "k": 256,
        "block_m": 64,
        "block_n": 128,
        "block_k": 32,
        "stages": 2,
        "m_waves": 2,
        "n_waves": 2,
        "group_m": 0,
        "dtype_id": GEMM_DTYPE_BF16,
        "mma_m": 16,
        "mma_n": 16,
        "mma_k": 32,
    }
    config.update(changed)
    with pytest.raises(ValueError, match=message):
        validate_gemm_gfx1250_config(**config)


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("split_k", [1, 2], ids=["no-split", "split-k"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.parametrize("edge", [False, True], ids=["full-tile", "edge-tile"])
def test_gemm_fp8_ptpc_gfx1250_all_paths(
    layout: str,
    split_k: int,
    has_bias: bool,
    edge: bool,
):
    # Covers vector/gather B loads, pre/post-reduction scaling, bias, interior/
    # edge stores, output allocation, and both paired/remainder pipeline loops.
    torch.manual_seed(0)
    m, n = (65, 80) if edge else (64, 64)
    args = _Fp8PtpcArgs(
        m=m,
        n=n,
        k=768,
        block_m=64,
        block_n=64,
        block_k=128,
        stages=3,
        m_waves=2,
        n_waves=2,
        group_m=4 if edge else 0,
        has_bias=has_bias,
        layout=layout,
        split_k=split_k,
    )
    a, b, scale_a, scale_b, bias = create_fp8_ptpc_inputs(args)
    out = create_fp8_ptpc_output(args) if edge else None
    result = gemm_fp8_ptpc_gfx1250(
        a,
        b,
        scale_a,
        scale_b,
        out=out,
        bias=bias,
        layout=layout,
        **_fp8_ptpc_kernel_kwargs(args),
    )
    reference = fp8_ptpc_ref_func(a, b, scale_a, scale_b, bias, split_k)

    if out is not None:
        assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        result,
        reference,
        atol=2.5e-1 if split_k > 1 else 1.25e-1,
        rtol=1e-1,
        check_dtype=True,
    )


# =========================================== benchmark ===========================================


@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, split_k, "
    "m_waves, n_waves, group_m, has_bias",
    [
        # fmt: off
        ( 4096,  128, 6144,  64,  64, 256, 2,  2,  2,  2,  0, False),
        (  512, 2560, 6144, 128, 128, 128, 2,  3,  2,  4,  4, False),
        (  576,  256, 4096,  64,  64, 256, 2,  4,  2,  2,  0,  True),
        ( 1024,  128, 6144,  64,  64, 256, 2,  6,  2,  2,  0,  True),
        ( 1216,   64, 7168,  64,  64, 256, 2,  7,  2,  2,  0, False),
        ( 1280,   64, 4096,  64,  64, 256, 2,  8,  2,  2,  0, False),
        # fmt: on
    ],
)
def test_gemm_a16w16_gfx1250_split_k_benchmark(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    split_k: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
):
    args = _TestArgs(
        dtype=torch.bfloat16,
        m=m,
        n=n,
        k=k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        has_bias=has_bias,
        layout="nt",
        split_k=split_k,
    )
    benchmark(args)


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, split_k, "
    "m_waves, n_waves, group_m, has_bias",
    [
        (8, 4096, 4096, 16, 128, 256, 2, 8, 1, 4, 0, True),
        (16, 4096, 4096, 16, 64, 128, 5, 8, 1, 4, 0, True),
        (32, 4096, 4096, 32, 64, 64, 4, 8, 2, 4, 0, True),
        (64, 4096, 4096, 64, 64, 256, 3, 4, 4, 4, 0, True),
        (128, 4096, 4096, 128, 64, 128, 5, 4, 4, 2, 0, True),
        (256, 4096, 4096, 128, 64, 256, 3, 2, 4, 4, 0, True),
        (512, 4096, 4096, 256, 128, 128, 3, 4, 4, 4, 0, True),
        (1024, 4096, 4096, 256, 128, 128, 2, 2, 4, 4, 0, True),
        (2048, 4096, 4096, 128, 256, 128, 3, 1, 4, 4, 4, True),
        (1024, 1024, 1024, 64, 64, 128, 3, 1, 4, 4, 0, True),
        (2048, 2048, 2048, 128, 128, 256, 2, 1, 4, 4, 4, True),
        (4096, 4096, 4096, 256, 256, 128, 2, 1, 4, 2, 4, True),
        (4096, 4096, 8192, 256, 256, 128, 2, 1, 4, 2, 4, True),
        (8192, 8192, 8192, 256, 256, 128, 2, 1, 4, 2, 0, True),
        (8, 7168, 2048, 16, 64, 128, 3, 4, 1, 2, 0, True),
        (32, 384, 7168, 16, 32, 128, 7, 8, 1, 2, 0, True),
        (32, 14336, 4096, 32, 128, 128, 3, 4, 2, 4, 0, True),
        (16, 28672, 4096, 16, 128, 128, 6, 1, 1, 4, 0, True),
        (4096, 256, 4096, 128, 128, 128, 4, 4, 4, 4, 0, True),
        (8, 4096, 4096, 16, 64, 128, 3, 8, 1, 2, 0, True),
        (16, 4096, 4096, 16, 64, 128, 5, 8, 1, 4, 0, True),
        (32, 4096, 4096, 32, 64, 128, 3, 8, 2, 4, 0, True),
        (64, 4096, 4096, 32, 64, 128, 5, 4, 2, 4, 0, True),
        (128, 4096, 4096, 128, 64, 128, 3, 4, 4, 4, 0, True),
        (256, 4096, 4096, 128, 128, 128, 3, 4, 4, 4, 0, True),
        (512, 4096, 4096, 128, 128, 128, 3, 2, 4, 4, 0, True),
        (1024, 4096, 4096, 128, 256, 128, 3, 2, 4, 4, 0, True),
        (2048, 4096, 4096, 128, 256, 128, 3, 1, 4, 4, 4, True),
        (1024, 1024, 1024, 128, 64, 256, 3, 2, 4, 2, 0, True),
        (2048, 2048, 2048, 128, 128, 128, 3, 1, 4, 4, 4, True),
        (4096, 4096, 4096, 256, 256, 128, 2, 1, 4, 2, 0, True),
        (4096, 4096, 8192, 256, 256, 128, 2, 1, 4, 2, 4, True),
        (8192, 8192, 8192, 256, 256, 128, 2, 1, 4, 2, 0, True),
        (8, 7168, 2048, 16, 64, 256, 3, 2, 1, 4, 0, True),
        (32, 384, 7168, 16, 64, 512, 3, 7, 1, 4, 0, True),
        (32, 14336, 4096, 32, 64, 128, 4, 2, 2, 2, 0, True),
        (16, 28672, 4096, 16, 64, 256, 3, 1, 1, 4, 4, True),
        (4096, 256, 4096, 128, 128, 128, 3, 4, 4, 4, 0, True),
    ],
)
def test_gemm_a16w16_gfx1250_benchmark_smoke(
    layout: str,
    dtype: str,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    split_k: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
):
    args = _TestArgs(
        _dtype_from_name(dtype),
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
        layout,
        split_k,
    )
    benchmark(args)


def test_gemm_fp8_ptpc_gfx1250_benchmark_smoke():
    args = _Fp8PtpcArgs(
        m=8192,
        n=8192,
        k=8192,
        block_m=256,
        block_n=256,
        block_k=128,
        stages=2,
        m_waves=4,
        n_waves=2,
        group_m=0,
        has_bias=False,
        layout="nt",
        split_k=1,
    )
    benchmark_fp8_ptpc(args)
