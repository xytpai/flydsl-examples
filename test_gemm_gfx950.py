import torch
import pytest
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass

from kernels.gemm_gfx950 import (
    GEMM_DTYPE_BF16,
    GEMM_DTYPE_FP32,
    GEMM_DTYPE_MXFP8,
    GEMM_DTYPE_MXFP4,
    gemm,
    make_gemm_param_and_validate,
)
from kernels.gemm_gfx950_utils import GFX950_DMA_BYTES

ROTARY_INPUTS_TARGET_BYTES = 8 * 1024**3
DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "mxfp8": torch.float8_e4m3fn,
    "mxfp4": torch.float4_e2m1fn_x2,
}
MXFP_DTYPES = (DTYPES["mxfp8"], DTYPES["mxfp4"])


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
    k_waves: int
    group_m: int
    has_bias: bool
    use_half_tile_interleaved: bool = False
    layout: str = "nt"
    split_k: int = 1
    out_dtype: torch.dtype | None = None

    @property
    def is_mxfp(self):
        return self.dtype in MXFP_DTYPES

    def __post_init__(self):
        if self.is_mxfp and self.out_dtype is None:
            self.out_dtype = torch.bfloat16


def a_dma_vec_size(dtype: torch.dtype) -> int:
    return GFX950_DMA_BYTES // torch.empty((), dtype=dtype).element_size()


def skip_unsupported_accuracy_layout(args: _TestArgs):
    if args.layout[0] == "t":
        a_vec_size = a_dma_vec_size(args.dtype)
        if args.m % a_vec_size != 0:
            pytest.skip(
                "column-major A requires M divisible by "
                f"{a_vec_size} for GFX950 DMA; got M={args.m}"
            )


def empty_layout_matrix(rows: int, cols: int, dtype: torch.dtype, is_t: bool):
    if is_t:
        return torch.empty((cols, rows), dtype=dtype, device="cuda").t()
    return torch.empty((rows, cols), dtype=dtype, device="cuda")


def create_mxfp_operand(rows, k, dtype, is_t):
    if dtype == torch.float8_e4m3fn:
        data = torch.empty((rows, k), device="cuda").uniform_(-1, 1).to(dtype)
    else:
        codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, device="cuda")
        packed = codes[:, ::2] | (codes[:, 1::2] << 4)
        data = packed.view(dtype)
    if is_t:
        data = data.t().contiguous().t()
    # Independent E8M0 exponents across rows and K groups expose scale indexing
    # and stage-reuse errors. Keep magnitudes comparable to the 16-bit tests.
    # FP8 samples span +/-1, whereas raw FP4 spans +/-6.
    exponent = 125 if dtype == torch.float8_e4m3fn else 123
    scale = torch.randint(exponent, exponent + 3, (rows, k // 32), dtype=torch.uint8, device="cuda")
    return data, scale.view(torch.float8_e8m0fnu)


def dequantize_mxfp(data, scale):
    if data.dtype == torch.float8_e4m3fn:
        values = data.float()
    else:
        packed = data.view(torch.uint8)
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
        lut = torch.tensor(
            [0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
            device=data.device,
        )
        values = lut[codes]
    scales = torch.exp2(scale.view(torch.uint8).float() - 127)
    return (values.unflatten(1, (-1, 32)) * scales.unsqueeze(-1)).flatten(1)


def reference_inputs(inputs):
    a, b, bias, *scales = inputs
    if scales:
        sa, sb = scales
        a = dequantize_mxfp(a, sa)
        b = dequantize_mxfp(b.t(), sb).t()
    return a, b, bias


def create_inputs(args: _TestArgs):
    if args.is_mxfp:
        a, sa = create_mxfp_operand(args.m, args.k, args.dtype, args.layout[0] == "t")
        bt, sb = create_mxfp_operand(args.n, args.k, args.dtype, args.layout[1] != "t")
        bias = (
            torch.empty((args.n,), dtype=args.out_dtype, device="cuda").uniform_(10, 20)
            if args.has_bias else None
        )
        return a, bt.t(), bias, sa, sb
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
    if args.has_bias:
        bias = torch.empty((args.n,), dtype=args.dtype, device="cuda")
        bias.uniform_(10, 20)
    else:
        bias = None
    return (a, b, bias)


def create_outputs(args: _TestArgs):
    dtype = args.dtype if args.out_dtype is None else args.out_dtype
    c = torch.randn((args.m, args.n), dtype=dtype, device="cuda")
    return (c,)


def ref_func(*args):
    c, _layout = args[-2:]
    a, b, bias = reference_inputs(args[:-2])
    if c.dtype == a.dtype:
        if bias is None:
            torch.mm(a, b, out=c)
        else:
            torch.addmm(bias, a, b, out=c)
    else:
        if bias is None:
            ref = torch.mm(a.float(), b.float())
        else:
            ref = torch.addmm(bias.float(), a.float(), b.float())
        c.copy_(ref)


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


def func(*args):
    a, b, bias, *scales, c, kwargs, layout = args
    scale_kwargs = dict(zip(("scale_a", "scale_b"), scales))
    gemm(a, b, c, bias=bias, user_kwargs=kwargs, layout=layout, **scale_kwargs)


def tensor_nbytes(tensors: torch.Tensor):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def get_rotary_inputs(sample_inputs, sample_outputs, sample_ref_inputs):
    global ROTARY_INPUTS_TARGET_BYTES
    slot_bytes = (
        tensor_nbytes(sample_inputs) + tensor_nbytes(sample_ref_inputs)
        + 2 * tensor_nbytes(sample_outputs)
    )
    rotary_inputs = ROTARY_INPUTS_TARGET_BYTES // slot_bytes
    return max(1, int(rotary_inputs))


def gemm_kwargs(args: _TestArgs):
    kwargs = {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "stages": args.stages,
        "m_waves": args.m_waves,
        "n_waves": args.n_waves,
        "k_waves": args.k_waves,
        "group_m": args.group_m,
        "use_half_tile_interleaved": args.use_half_tile_interleaved,
        "split_k": args.split_k,
    }
    if args.is_mxfp:
        min_block_k = 128 * args.k_waves
        if args.block_k < min_block_k:
            pytest.skip(f"MXFP requires block_k >= {min_block_k} for k_waves={args.k_waves}; got {args.block_k}")
        validation_kwargs = {
            **kwargs,
            "in_dtype_id": GEMM_DTYPE_MXFP8 if args.dtype == torch.float8_e4m3fn else GEMM_DTYPE_MXFP4,
            "out_dtype_id": GEMM_DTYPE_FP32 if args.out_dtype == torch.float32 else GEMM_DTYPE_BF16,
            "a_is_transposed": args.layout[0] == "t",
            "b_is_transposed": args.layout[1] == "t",
            "has_bias": args.has_bias,
        }
        if make_gemm_param_and_validate(args.m, args.n, args.k, validation_kwargs) is None:
            pytest.skip(f"MXFP shape/policy unsupported: {args}")
    return kwargs


def check_acc(args: _TestArgs):
    skip_unsupported_accuracy_layout(args)
    kwargs = gemm_kwargs(args)
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = inputs + outputs
    ref_inouts = reference_inputs(inputs) + ref_outputs
    maxdiff_out_ = []

    def get_tol(args):
        k_scale = (args.k / 8192) ** 0.5
        k_scale *= args.split_k * args.k_waves
        atol_scale = 1.5 if args.has_bias else 1.0
        if args.is_mxfp and args.out_dtype == torch.float32:
            return 2e-4 * max(1.0, k_scale), 2e-4
        if args.dtype is torch.bfloat16 or args.is_mxfp:
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
    kwargs = gemm_kwargs(args)
    sample_inputs = create_inputs(args)
    sample_outputs = create_outputs(args)
    sample_ref_inputs = reference_inputs(create_inputs(args))
    rotary_inputs = get_rotary_inputs(sample_inputs, sample_outputs, sample_ref_inputs)
    inputs = [sample_inputs] + [create_inputs(args) for _ in range(rotary_inputs - 1)]
    # MXFP benchmarks compare with decoded FP32 GEMM; decoding is NOT timed.
    ref_inputs = [sample_ref_inputs] + [
        reference_inputs(create_inputs(args)) for _ in range(rotary_inputs - 1)
    ]
    outputs = [sample_outputs] + [
        create_outputs(args) for _ in range(rotary_inputs - 1)
    ]
    ref_outputs = [create_outputs(args) for _ in range(rotary_inputs)]
    triton_maxautotune_func = make_triton_maxautotune_func()
    global ROTARY_INPUTS_TARGET_BYTES
    if args.is_mxfp:
        print(f"MXFP policy: {args}; torch/Triton baseline: decoded FP32 GEMM")
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


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias, is_hti",
    [
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 4, True, False),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, True, False),
        (8192, 8192, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8192, 8192, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, True, False),
        (8160, 8160, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8160, 8160, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, True, False),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 4, 0, False, False),
        (2048, 2048, 2048, 128, 128, 64, 4, 4, 4, 0, True, False),
        (2048, 2048, 2048 - 64, 128, 128, 64, 2, 4, 4, 0, False, False),
        (2048, 2048, 2048 - 64, 128, 128, 64, 4, 4, 4, 0, True, False),
        # hti
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 4, True, True),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, True, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 2, 2, 0, False, True),
    ],
)
def test_gemm_acc_main_loop(
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
    is_hti: bool,
):
    dtype = DTYPES[dtype]
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
        1,
        group_m,
        has_bias,
        is_hti,
        layout,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES)
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, split_k, "
    "m_waves, n_waves, k_waves, has_bias, group_m, "
    "use_half_tile_interleaved",
    [
        # test_gemm_acc_ft_stage_split_k
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, 1, True, 0, False),
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, 1, False, 0, False),
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, 1, True, 4, False),
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, 1, False, 4, False),
        # test_gemm_acc_small_m
        (3, 5120, 2880, 64, 64, 64, 5, 3, 2, 2, 1, True, 0, False),
        (3, 5120, 2880, 64, 64, 64, 5, 3, 2, 2, 1, False, 0, False),
        # test_gemm_acc_ft_slice_k
        (800, 384, 7168, 32, 64, 128, 6, 1, 1, 2, 2, True, 0, False),
        (800, 384, 7168, 32, 64, 128, 6, 1, 1, 2, 2, False, 0, False),
        (800, 384, 7168, 32, 64, 128, 6, 2, 1, 2, 2, True, 0, False),
        (800, 384, 7168, 32, 64, 128, 6, 2, 1, 2, 2, False, 0, False),
        (800, 384, 7168, 32, 64, 128, 6, 2, 1, 2, 2, False, 4, False),
        # test_gemm_acc_hti_split_k
        (64, 64, 512, 64, 64, 64, 2, 2, 2, 2, 1, True, 0, True),
        (64, 64, 512, 64, 64, 64, 2, 2, 2, 2, 1, False, 0, True),
        # Common PR-tuned HTI split-K policies.
        (65, 96, 4096, 64, 64, 256, 2, 4, 2, 2, 1, False, 0, True),
        (96, 64, 7168, 64, 64, 256, 2, 7, 2, 2, 1, False, 0, True),
        (128, 64, 4096, 64, 64, 256, 2, 8, 2, 2, 1, False, 0, True),
        (128, 128, 6144, 64, 64, 256, 2, 6, 2, 2, 1, False, 0, True),
        (65, 136, 2048, 64, 128, 128, 2, 4, 2, 4, 1, False, 0, True),
        (129, 72, 1024, 128, 64, 128, 2, 2, 2, 2, 1, True, 0, True),
        (129, 136, 1536, 128, 128, 128, 2, 3, 2, 4, 1, False, 4, True),
        (257, 264, 512, 256, 256, 64, 2, 2, 2, 4, 1, False, 0, True),
        # HTI split-K coverage mirrored from the layout tests.
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, 1, True, 0, True),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, 1, False, 0, True),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, 1, True, 4, True),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, 1, False, 4, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, 1, True, 0, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, 1, False, 0, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, 1, True, 4, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, 1, False, 4, True),
        # Exact HTI split-K winners
        (4096, 128, 6144, 64, 64, 256, 2, 2, 2, 2, 1, False, 0, True),
        (512, 2560, 6144, 128, 128, 128, 2, 3, 2, 4, 1, False, 4, True),
        (576, 256, 4096, 64, 64, 256, 2, 4, 2, 2, 1, True, 0, True),
        (1024, 128, 6144, 64, 64, 256, 2, 6, 2, 2, 1, True, 0, True),
        (1216, 64, 7168, 64, 64, 256, 2, 7, 2, 2, 1, False, 0, True),
        (1280, 64, 4096, 64, 64, 256, 2, 8, 2, 2, 1, False, 0, True),
        # Split-K stress variants for timing-sensitive and boundary paths.
        (1024, 64, 4096, 64, 64, 256, 2, 8, 2, 2, 1, False, 0, True),
        (1272, 64, 4096, 64, 64, 256, 2, 8, 2, 2, 1, False, 0, True),
        (1280, 72, 4096, 64, 64, 256, 2, 8, 2, 2, 1, False, 0, True),
        (1280, 128, 4096, 64, 64, 256, 2, 8, 2, 2, 1, False, 0, True),
        (1280, 64, 4096, 64, 64, 256, 2, 8, 2, 2, 1, True, 0, True),
        (1280, 64, 4096, 64, 64, 128, 2, 8, 2, 2, 1, False, 0, True),
        (1280, 64, 4096, 64, 64, 256, 2, 4, 2, 2, 1, False, 0, True),
        (1024, 64, 4096, 64, 64, 256, 2, 8, 2, 2, 1, False, 0, False),
        (1272, 64, 4096, 64, 64, 256, 2, 8, 2, 2, 1, False, 0, False),
        (1280, 72, 4096, 64, 64, 256, 2, 8, 2, 2, 1, False, 0, False),
        (1280, 128, 4096, 64, 64, 256, 2, 8, 2, 2, 1, False, 0, False),
        (1280, 64, 4096, 64, 64, 256, 2, 8, 2, 2, 1, True, 0, False),
        (1280, 64, 4096, 64, 64, 128, 2, 8, 2, 2, 1, False, 0, False),
        (1280, 64, 4096, 64, 64, 256, 2, 4, 2, 2, 1, False, 0, False),
    ],
)
def test_gemm_acc_split_k(
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
    k_waves: int,
    has_bias: bool,
    group_m: int,
    use_half_tile_interleaved: bool,
):
    assert k_waves > 0
    assert split_k > 1 or k_waves > 1
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
        k_waves=k_waves,
        group_m=group_m,
        has_bias=has_bias,
        use_half_tile_interleaved=use_half_tile_interleaved,
        layout=layout,
        split_k=split_k,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, split_k, "
    "m_waves, n_waves, k_waves, group_m, use_half_tile_interleaved",
    [
        # fmt: off
        #    M,     N,     K,  BM,  BN,  BK, S, SK, MW, NW, KW, GM,   HTI
        (   8,  4096,  4096,  16,  16, 256, 4,  1,  1,  1,  2,  4, False),
        (  16,  4096,  4096,  16,  16, 256, 4,  1,  1,  1,  2,  4, False),
        (  32,  4096,  4096,  16,  32, 256, 4,  1,  1,  2,  2,  4, False),
        (  64,  4096,  4096,  32,  32, 128, 6,  1,  2,  2,  1,  4, False),
        ( 128,  4096,  4096,  32,  64, 128, 4,  1,  1,  4,  1,  0, False),
        ( 256,  4096,  4096,  64,  64, 128, 4,  1,  4,  2,  1,  0, False),
        ( 512,  4096,  4096,  64, 128,  64, 5,  1,  2,  4,  1,  4, False),
        (1024,  4096,  4096, 128, 128,  64, 4,  1,  2,  4,  1,  4, False),
        (2048,  4096,  4096, 256, 128,  64, 3,  1,  4,  2,  2,  0, False),
        (1024,  1024,  1024,  64,  64,  64, 4,  1,  1,  4,  1,  4, False),
        (2048,  2048,  2048, 128, 128,  64, 4,  1,  2,  4,  1,  0, False),
        (4096,  4096,  4096, 256, 256,  64, 2,  1,  2,  4,  1,  4,  True),
        (4096,  4096,  8192, 256, 256,  64, 2,  1,  2,  4,  1,  4,  True),
        (8192,  8192,  8192, 256, 256,  64, 2,  1,  2,  4,  1,  4,  True),
        (   8,  7168,  2048,  16,  16,  64, 8,  1,  1,  1,  1,  4, False),
        (  32,   384,  7168,  16,  16, 256, 4,  1,  1,  1,  2,  0, False),
        (  32, 14336,  4096,  32,  64, 128, 5,  1,  2,  2,  1,  4, False),
        (  16, 28672,  4096,  16,  64, 256, 2,  1,  1,  2,  2,  4, False),
        (4096,   256,  4096,  64,  64, 128, 4,  1,  4,  2,  1,  4, False),
        (   1,  5120,  2880,  16,  64,  64, 7,  3,  1,  2,  1,  0, False),
        (   2,  5120,  2880,  16,  64,  64, 7,  3,  1,  2,  1,  0, False),
        (   4,  5120,  2880,  16,  64,  64, 7,  3,  1,  2,  1,  0, False),
        (   8,  5120,  2880,  16,  64,  64, 8,  3,  1,  2,  1,  0, False),
        (  16,  5120,  2880,  16,  64,  64, 8,  3,  1,  2,  1,  0, False),
        (  32,  5120,  2880,  16,  32,  64, 8,  1,  1,  2,  1,  0, False),
        (  48,  5120,  2880,  16,  64,  64, 8,  1,  1,  2,  1,  0, False),
        (  32,   384,  7168,  32,  32,  64, 8,  8,  2,  1,  1,  0, False),
        (  32,   384, 16384,  32,  32,  64, 8,  8,  1,  2,  1,  0, False),
        ( 800,   384,  7168,  64,  96,  64, 4,  4,  2,  2,  1,  0, False),
        (  32,  7168,  2048,  32,  32,  64, 8,  1,  2,  1,  1,  0, False),
        (   8,  7168,  2048,  16,  16,  64, 8,  1,  1,  1,  1,  4, False),
        (  32,  2880,  2048,  32,  32,  64, 8,  2,  1,  2,  1,  0, False),
        # Additional policies from the latest tuning run.
        (  64,  4096,  4096,  32,  32,  64, 8,  1,  1,  2,  1,  0, False),
        ( 128,  4096,  4096,  64,  32, 128, 4,  1,  4,  1,  1,  4, False),
        ( 256,  4096,  4096,  64,  64,  64, 8,  1,  2,  2,  1,  0, False),
        ( 512,  4096,  4096,  64, 128,  64, 6,  1,  2,  4,  1,  4, False),
        (1024,  1024,  1024,  64,  64,  64, 4,  1,  2,  4,  1,  4, False),
        (2048,  2048,  2048, 128, 128,  64, 4,  1,  2,  4,  1,  4, False),
        (8192,  8192,  8192, 256, 256,  64, 2,  1,  2,  4,  1,  0,  True),
        (   8,  7168,  2048,  16,  16, 128, 6,  1,  1,  1,  1,  4, False),
        (  32, 14336,  4096,  32,  64, 256, 3,  1,  1,  2,  2,  4, False),
        (  16, 28672,  4096,  16,  32, 256, 3,  1,  1,  2,  2,  4, False),
        (4096,   256,  4096,  64,  64,  64, 8,  1,  2,  2,  1,  4, False),
        (   1,  5120,  2880,  16,  64,  64, 9,  3,  1,  2,  1,  0, False),
        (   2,  5120,  2880,  16,  64,  64, 8,  3,  1,  2,  1,  0, False),
        (   4,  5120,  2880,  16,  64,  64, 8,  3,  1,  2,  1,  0, False),
        (   8,  5120,  2880,  16,  64,  64, 9,  3,  1,  2,  1,  0, False),
        (  16,  5120,  2880,  16,  64,  64, 5,  3,  1,  2,  1,  0, False),
        (  32,   384,  7168,  32,  32,  64, 8,  8,  1,  2,  1,  0, False),
        (   8,  7168,  2048,  16,  16, 128, 7,  1,  1,  1,  2,  4, False),
        (  32,  2880,  2048,  32,  64,  64, 8,  4,  2,  2,  1,  0, False),
        # fmt: on
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, *MXFP_DTYPES], ids=["bf16", "mxfp8", "mxfp4"])
def test_gemm_acc_tuned_policies(
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
    k_waves: int,
    group_m: int,
    use_half_tile_interleaved: bool,
    layout: str,
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
        k_waves=k_waves,
        group_m=group_m,
        has_bias=True,
        use_half_tile_interleaved=use_half_tile_interleaved,
        layout=layout,
        split_k=split_k,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize(
    "split_k, use_half_tile_interleaved",
    [(1, False), (3, False), (1, True), (3, True)],
)
def test_gemm_acc_fp32_output(
    layout: str,
    dtype: torch.dtype,
    has_bias: bool,
    split_k: int,
    use_half_tile_interleaved: bool,
):
    args = _TestArgs(
        dtype=dtype,
        m=64,
        n=64,
        k=768,
        block_m=64,
        block_n=64,
        block_k=64,
        stages=2,
        m_waves=2,
        n_waves=2,
        k_waves=1,
        group_m=0,
        has_bias=has_bias,
        use_half_tile_interleaved=use_half_tile_interleaved,
        layout=layout,
        split_k=split_k,
        out_dtype=torch.float32,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias, is_hti",
    [
        (3, 5120, 2880, 64, 64, 64, 5, 2, 2, 4, True, False),
        (3, 5120, 2880, 64, 64, 64, 5, 2, 2, 0, False, False),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, True, False),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, False, False),
        (3, 32, 128 + 64, 128, 128, 64, 3, 2, 2, 4, True, False),
    ],
)
def test_gemm_acc_small_m(
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
    is_hti: bool,
):
    dtype = DTYPES[dtype]
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
        1,
        group_m,
        has_bias,
        is_hti,
        layout,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("dtype", list(DTYPES))
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
def test_gemm_acc_bench(
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
    is_hti: bool,
):
    dtype = DTYPES[dtype]
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
        1,
        group_m,
        has_bias,
        is_hti,
        layout,
    )
    check_acc(args)



@pytest.mark.parametrize(
    "overrides, fence_before_load",
    [
        ({"in_dtype_id": GEMM_DTYPE_MXFP8, "block_k": 128}, True),
        ({}, False),
        ({"has_bias": True}, False),
        ({"group_m": 4}, False),
        ({"split_k": 2}, True),
        ({"use_half_tile_interleaved": False}, True),
        ({"block_m": 128}, True),
        ({"block_k": 128}, True),
        ({"n_waves": 2}, True),
        ({"a_is_transposed": True}, True),
        ({"b_is_transposed": False}, True),
    ],
)
def test_gemm_fence_before_load_policy(monkeypatch, overrides, fence_before_load):
    import kernels.gemm_gfx950 as kernel

    monkeypatch.setattr(kernel, "get_rocm_arch", lambda: "gfx950")
    param = kernel.make_gemm_gfx950_param(**{
        "in_dtype_id": GEMM_DTYPE_MXFP4,
        "out_dtype_id": GEMM_DTYPE_BF16,
        "block_m": 256, "block_n": 256, "block_k": 256,
        "m_waves": 2, "n_waves": 4, "use_half_tile_interleaved": True,
        **overrides,
    })
    assert param.fence_before_load is fence_before_load
    peer = kernel.GemmGfx950Param(**{
        **{name: getattr(param, name) for name in kernel.GemmGfx950Param.__annotations__},
        "fence_before_load": not fence_before_load,
    })
    assert peer.__cache_signature__() != param.__cache_signature__()


@pytest.mark.parametrize("rows", [32, 64, 96, 128, 160])
@pytest.mark.parametrize("storage_k", [32, 64, 96, 128, 192, 256, 320, 384, 448, 512])
@pytest.mark.parametrize("is_transposed", [False, True])
def test_gemm_mxfp_lds_swizzle_mapping(rows, storage_k, is_transposed):
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from kernels.gemm_gfx950 import make_mxfp_lds_layout

    # Check the actual layout, including non-power-of-two row/K extents.
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            layout = make_mxfp_lds_layout(rows, storage_k, is_transposed)
            extent = rows if is_transposed else storage_k

            def address(offset):
                row, col = (
                    (offset % rows, offset // rows) if is_transposed
                    else (offset // storage_k, offset % storage_k)
                )
                return fx.get_scalar(fx.crd2idx((row, col), layout))

            for offset in range(0, rows * storage_k, 16):
                mapped = address(offset)
                # Loader keeps the outer coordinate and inverts within the row.
                assert mapped // extent == offset // extent
                assert mapped % 16 == 0
                assert address(mapped) == offset
                assert address(offset + 15) == mapped + 15


@pytest.mark.parametrize("dtype", MXFP_DTYPES, ids=["mxfp8", "mxfp4"])
@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "block_m, block_n, block_k, m_waves, n_waves, k_waves, is_hti",
    [
        (96, 96, 384, 3, 1, 1, False),
        (160, 32, 640, 5, 1, 1, False),
        # Both contiguous/transpose paths, including HTI halves and slice-K.
        (128, 192, 384, 2, 3, 1, True),
        (96, 96, 384, 1, 1, 3, False),
    ],
)
def test_gemm_mxfp_non_power_of_two_k(
    dtype, layout, out_dtype, block_m, block_n, block_k,
    m_waves, n_waves, k_waves, is_hti,
):
    args = _TestArgs(
        dtype=dtype, m=block_m, n=block_n, k=block_k * 2,
        block_m=block_m, block_n=block_n, block_k=block_k, stages=2,
        m_waves=m_waves, n_waves=n_waves, k_waves=k_waves, group_m=0,
        has_bias=False, use_half_tile_interleaved=is_hti,
        layout=layout, out_dtype=out_dtype,
    )
    check_acc(args)


@pytest.mark.parametrize("dtype", MXFP_DTYPES, ids=["mxfp8", "mxfp4"])
@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "block_m, block_n, block_k, n_waves, k, split_k, has_bias",
    [
        # Short partition, exact chunk, chunk-slot reuse, and partial final chunk.
        (128, 128, 128, 2, 512, 2, False),
        (128, 128, 128, 2, 1024, 1, True),
        (64, 64, 256, 2, 7168, 1, False),
        (64, 64, 256, 2, 3072, 2, True),
        # Unequal A/B scale sizes and larger MMA repeats.
        (64, 128, 256, 2, 6144, 3, True),
        (64, 256, 256, 2, 6144, 3, False),
        (128, 128, 256, 4, 6144, 3, False),
        # FP4 large-tile pre-load fence policy, with and without bias.
        (256, 256, 256, 4, 3072, 1, False),
        (256, 256, 256, 4, 3072, 1, True),
    ],
)
def test_gemm_mxfp_hti_scales(
    dtype, layout, out_dtype, block_m, block_n, block_k, n_waves, k, split_k, has_bias,
):
    args = _TestArgs(
        dtype=dtype, m=80, n=112, k=k,
        block_m=block_m, block_n=block_n, block_k=block_k, stages=2,
        m_waves=2, n_waves=n_waves, k_waves=1, group_m=0,
        has_bias=has_bias, use_half_tile_interleaved=True,
        layout=layout, split_k=split_k, out_dtype=out_dtype,
    )
    kwargs = gemm_kwargs(args)
    assert kwargs["use_half_tile_interleaved"]  # Must not fall back to full-tile.
    a, b, bias, sa, sb = create_inputs(args)

    def pad(tensor, is_t, alignment):
        data = tensor.view(torch.uint8)
        data = data.t() if is_t else data
        storage = torch.empty(
            (data.shape[0] + 1, data.shape[1] + alignment), dtype=torch.uint8, device=data.device,
        )
        padded = storage[1:, alignment:]
        padded.copy_(data)
        return (padded.t() if is_t else padded).view(tensor.dtype)

    out = create_outputs(args)[0]
    ref = create_outputs(args)[0]
    ref_func(a, b, bias, sa, sb, ref, layout)
    # Reuse one compiled policy with dense and offset/padded dynamic strides.
    inputs = [
        (a, b, bias, sa, sb),
        (pad(a, layout[0] == "t", 16), pad(b, layout[1] == "t", 16), bias,
         pad(sa, False, 4), pad(sb, False, 4)),
    ]
    for tensors in inputs:
        for _ in range(3):
            out.fill_(float("nan"))
            func(*tensors, out, kwargs, layout)
            torch.testing.assert_close(
                out, ref,
                atol=2e-3 if out_dtype == torch.float32 else 2e-1 * split_k,
                rtol=2e-4 if out_dtype == torch.float32 else 2e-2,
            )

@pytest.mark.parametrize("dtype", MXFP_DTYPES, ids=["mxfp8", "mxfp4"])
@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "block_m, block_n, block_k, n_waves, k_waves, stages, k, split_k, has_bias",
    [
        (64, 64, 256, 2, 1, 2, 256, 1, False),
        (64, 64, 256, 2, 1, 2, 512, 1, True),
        (64, 64, 256, 2, 2, 2, 1536, 3, False),
        (128, 128, 128, 2, 1, 2, 1024, 2, False),
        (256, 256, 256, 4, 1, 2, 3072, 1, False),
        (256, 256, 256, 4, 1, 2, 3072, 1, True),
        # Inline DMA waits must include scales for every outstanding stage.
        (64, 64, 256, 2, 1, 3, 512, 1, False),
        (64, 64, 256, 2, 1, 4, 768, 1, False),
        (64, 64, 256, 2, 2, 3, 1536, 3, False),
        (64, 64, 256, 2, 2, 4, 6144, 2, False),
    ],
)
def test_gemm_mxfp_ft_waits(
    dtype, layout, out_dtype, block_m, block_n, block_k, n_waves, k_waves, stages, k, split_k, has_bias,
):
    args = _TestArgs(
        dtype=dtype, m=80, n=112, k=k,
        block_m=block_m, block_n=block_n, block_k=block_k, stages=stages,
        m_waves=2, n_waves=n_waves, k_waves=k_waves, group_m=0,
        has_bias=has_bias, use_half_tile_interleaved=False,
        layout=layout, split_k=split_k, out_dtype=out_dtype,
    )
    kwargs = gemm_kwargs(args)
    a, b, bias, sa, sb = create_inputs(args)

    def pad(tensor, is_t, alignment):
        data = tensor.view(torch.uint8)
        data = data.t() if is_t else data
        storage = torch.empty(
            (data.shape[0] + 1, data.shape[1] + alignment), dtype=torch.uint8, device=data.device,
        )
        padded = storage[1:, alignment:]
        padded.copy_(data)
        return (padded.t() if is_t else padded).view(tensor.dtype)

    out = create_outputs(args)[0]
    ref = create_outputs(args)[0]
    ref_func(a, b, bias, sa, sb, ref, layout)
    # Reuse one compiled policy with dense and offset/padded dynamic strides.
    inputs = [
        (a, b, bias, sa, sb),
        (pad(a, layout[0] == "t", 16), pad(b, layout[1] == "t", 16), bias,
         pad(sa, False, 4), pad(sb, False, 4)),
    ]
    for tensors in inputs:
        for _ in range(3):
            out.fill_(float("nan"))
            func(*tensors, out, kwargs, layout)
            torch.testing.assert_close(
                out, ref,
                atol=2e-3 if out_dtype == torch.float32 else 2e-1 * split_k,
                rtol=2e-4 if out_dtype == torch.float32 else 2e-2,
            )


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("split_k", [1, 2])
@pytest.mark.parametrize("use_half_tile_interleaved", [False, True])
def test_gemm_padded_stride_and_storage_offset(
    layout: str,
    split_k: int,
    use_half_tile_interleaved: bool,
):
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
    for tensor, is_t in zip(
        (a, b),
        (layout[0] == "t", layout[1] == "t"),
    ):
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
        "use_half_tile_interleaved": use_half_tile_interleaved,
    }

    result = gemm(
        a,
        b,
        out,
        bias=bias,
        user_kwargs=kwargs,
        layout=layout,
    )
    ref_func(a, b, bias, ref, layout)

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        out,
        ref,
        atol=1e-1 if split_k > 1 else 3e-2,
        rtol=2e-1,
        check_dtype=True,
    )


@pytest.mark.parametrize("split_k", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, *MXFP_DTYPES], ids=["bf16", "mxfp8", "mxfp4"])
def test_gemm_fp32_slice_k_without_bias(dtype: torch.dtype, split_k: int):
    args = _TestArgs(
        dtype=dtype,
        m=64,
        n=128,
        k=2048,
        block_m=32,
        block_n=64,
        block_k=128,
        stages=4,
        m_waves=1,
        n_waves=2,
        k_waves=2,
        group_m=0,
        has_bias=False,
        use_half_tile_interleaved=False,
        layout="nt",
        split_k=split_k,
        out_dtype=torch.float32,
    )
    check_acc(args)


def test_gemm_allocates_output_with_default_policy():
    m = n = 64
    k = 256
    dtype = torch.bfloat16
    a = torch.randn((m, k), dtype=dtype, device="cuda")
    b = empty_layout_matrix(k, n, dtype, is_t=True)
    b.normal_()

    out = gemm(a, b)
    ref = torch.mm(a, b)

    assert out.shape == (m, n)
    assert out.dtype == dtype
    assert out.device == a.device
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=2e-1)


@pytest.mark.parametrize(
    ("operand", "layout"),
    [
        ("A", "nn"),
        ("A", "tn"),
        ("B", "nn"),
        ("B", "nt"),
    ],
)
def test_gemm_rejects_unsupported_input_strides(operand: str, layout: str):
    m = n = 64
    k = 256
    dtype = torch.bfloat16
    a = empty_layout_matrix(m, k, dtype, is_t=layout[0] == "t")
    b = empty_layout_matrix(k, n, dtype, is_t=layout[1] == "t")
    a.normal_()
    b.normal_()

    if operand == "A":
        if layout[0] == "t":
            a = torch.randn((m, k), dtype=dtype, device="cuda")
        else:
            a = torch.randn((m, k * 2), dtype=dtype, device="cuda")[:, ::2]
    else:
        if layout[1] == "t":
            b = torch.randn((k, n), dtype=dtype, device="cuda")
        else:
            b = torch.randn((k, n * 2), dtype=dtype, device="cuda")[:, ::2]

    with pytest.raises(ValueError, match=rf"^{operand} does not satisfy"):
        gemm(a, b, layout=layout)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_rejects_unaligned_m_for_column_major_a(dtype: torch.dtype):
    a_vec_size = a_dma_vec_size(dtype)
    m = a_vec_size + 1
    n = 64
    k = 256
    padded_m = 2 * a_vec_size
    a_storage = torch.randn((k, padded_m), dtype=dtype, device="cuda")
    a = a_storage[:, :m].t()
    b = torch.randn((k, n), dtype=dtype, device="cuda")

    assert a.stride() == (1, padded_m)
    assert a.data_ptr() % GFX950_DMA_BYTES == 0
    assert a.stride(1) * a.element_size() % GFX950_DMA_BYTES == 0
    with pytest.raises(ValueError, match=rf"M divisible by {a_vec_size}"):
        gemm(a, b, layout="tn")


def test_gemm_host_bias_contiguity_fallback():
    m = n = 64
    k = 256
    dtype = torch.bfloat16
    a = torch.randn((m, k), dtype=dtype, device="cuda")
    b = empty_layout_matrix(k, n, dtype, is_t=True)
    b.normal_()
    bias_storage = torch.randn((n * 2,), dtype=dtype, device="cuda")
    bias = bias_storage[::2]
    out = torch.empty((m, n), dtype=dtype, device="cuda")
    ref = torch.empty_like(out)
    kwargs = {
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "stages": 2,
        "m_waves": 2,
        "n_waves": 2,
        "group_m": 0,
        "use_half_tile_interleaved": False,
    }

    gemm(a, b, out, bias=bias, user_kwargs=kwargs, layout="nt")
    ref_func(a, b, bias, ref, "nt")

    assert not bias.is_contiguous()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=2e-1)


@pytest.mark.parametrize("n", [4096, 4032])
@pytest.mark.parametrize("dtype", [torch.bfloat16, *MXFP_DTYPES], ids=["bf16", "mxfp8", "mxfp4"])
def test_gemm_block_swizzle_boundary_paths(dtype: torch.dtype, n: int):
    args = _TestArgs(
        dtype=dtype,
        m=320,
        n=n,
        k=256,
        block_m=64,
        block_n=64,
        block_k=64,
        stages=2,
        m_waves=2,
        n_waves=2,
        k_waves=1,
        group_m=4,
        has_bias=False,
        use_half_tile_interleaved=False,
        layout="nt",
    )
    check_acc(args)


@pytest.mark.parametrize("dtype", [torch.bfloat16, *MXFP_DTYPES], ids=["bf16", "mxfp8", "mxfp4"])
def test_gemm_slice_k_four_waves(dtype: torch.dtype):
    args = _TestArgs(
        dtype=dtype,
        m=64,
        n=128,
        k=1024,
        block_m=32,
        block_n=64,
        block_k=128,
        stages=4,
        m_waves=1,
        n_waves=2,
        k_waves=4,
        group_m=0,
        has_bias=True,
        use_half_tile_interleaved=False,
        layout="nt",
    )
    check_acc(args)


def test_gemm_split_k_uses_stream_local_sync_buffers():
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
        k_waves=1,
        group_m=0,
        has_bias=True,
        use_half_tile_interleaved=False,
        layout="nt",
        split_k=2,
    )
    a, b, bias = create_inputs(args)
    out0 = create_outputs(args)[0]
    out1 = create_outputs(args)[0]
    ref = create_outputs(args)[0]
    kwargs = {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "stages": args.stages,
        "split_k": args.split_k,
        "m_waves": args.m_waves,
        "n_waves": args.n_waves,
        "k_waves": args.k_waves,
        "group_m": args.group_m,
        "use_half_tile_interleaved": args.use_half_tile_interleaved,
    }
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()

    gemm(a, b, out0, bias=bias, user_kwargs=kwargs, layout=args.layout)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream0):
        gemm(
            a,
            b,
            out0,
            bias=bias,
            user_kwargs=kwargs,
            layout=args.layout,
            stream=stream0,
        )
    with torch.cuda.stream(stream1):
        gemm(
            a,
            b,
            out1,
            bias=bias,
            user_kwargs=kwargs,
            layout=args.layout,
            stream=stream1,
        )
    stream0.synchronize()
    stream1.synchronize()
    ref_func(a, b, bias, ref, args.layout)

    torch.testing.assert_close(out0, ref, atol=2e-1, rtol=2e-1)
    torch.testing.assert_close(out1, ref, atol=2e-1, rtol=2e-1)


@pytest.mark.parametrize(
    "k, use_half_tile_interleaved, message",
    [
        (480, False, "K-tail is unsupported"),
        (64, True, "HTI requires at least two"),
        (192, True, "HTI requires at least two"),
    ],
)
def test_gemm_rejects_unsupported_k_partitioning(
    k: int,
    use_half_tile_interleaved: bool,
    message: str,
):
    m = n = 64
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
        "use_half_tile_interleaved": use_half_tile_interleaved,
    }

    with pytest.raises(AssertionError, match=message):
        gemm(a, b, user_kwargs=kwargs, layout="nt")


# =========================================== benchmark ===========================================


@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, split_k, "
    "m_waves, n_waves, k_waves, group_m, has_bias",
    [
        # fmt: off
        #    M,    N,    K,  BM,  BN,  BK, S, SK, MW, NW, KW, GM,  Bias
        ( 4096,  128, 6144,  64,  64, 256, 2,  2,  2,  2,  1,  0, False),
        (  512, 2560, 6144, 128, 128, 128, 2,  3,  2,  4,  1,  4, False),
        (  576,  256, 4096,  64,  64, 256, 2,  4,  2,  2,  1,  0,  True),
        ( 1024,  128, 6144,  64,  64, 256, 2,  6,  2,  2,  1,  0,  True),
        ( 1216,   64, 7168,  64,  64, 256, 2,  7,  2,  2,  1,  0, False),
        ( 1280,   64, 4096,  64,  64, 256, 2,  8,  2,  2,  1,  0, False),
        # fmt: on
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, *MXFP_DTYPES], ids=["bf16", "mxfp8", "mxfp4"])
def test_gemm_hti_split_k_benchmark(
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
    k_waves: int,
    group_m: int,
    has_bias: bool,
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
        k_waves=k_waves,
        group_m=group_m,
        has_bias=has_bias,
        use_half_tile_interleaved=True,
        layout="nt",
        split_k=split_k,
    )
    benchmark(args)


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("dtype", ["bf16", "mxfp8", "mxfp4"])
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
def test_gemm_benchmark_smoke(
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
    is_hti: bool,
):
    dtype = DTYPES[dtype]
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
        1,
        group_m,
        has_bias,
        is_hti,
        layout,
    )
    benchmark(args)
