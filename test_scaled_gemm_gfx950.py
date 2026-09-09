import pytest
import torch
from dataclasses import dataclass
from torch.profiler import ProfilerActivity, profile

from kernels.gemm_a16w16_gfx950_utils import GFX950_DMA_BYTES
from kernels.scaled_gemm_gfx950 import scaled_gemm

ROTARY_INPUTS_TARGET_BYTES = 8 * 1024**3


@dataclass
class _TestArgs:
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
    layout: str = "nt"
    split_k: int = 1
    out_dtype: torch.dtype = torch.bfloat16
    mma_m: int = 16
    mma_n: int = 16
    mma_k: int = 128


def _skip_if_no_fp8():
    if not torch.cuda.is_available():
        pytest.skip("GPU is required")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    if arch.split(":")[0] != "gfx950":
        pytest.skip("FP8 PTPC full-tile requires gfx950")


def _a_dma_vec_size() -> int:
    return GFX950_DMA_BYTES


def _skip_unsupported_accuracy_layout(args: _TestArgs):
    if args.layout[0] == "t" and args.m % _a_dma_vec_size() != 0:
        pytest.skip(
            "column-major A requires M divisible by "
            f"{_a_dma_vec_size()} for GFX950 DMA; got M={args.m}"
        )


def empty_layout_matrix(rows: int, cols: int, dtype: torch.dtype, is_t: bool):
    if is_t:
        return torch.empty((cols, rows), dtype=dtype, device="cuda").t()
    return torch.empty((rows, cols), dtype=dtype, device="cuda")


def _fill_fp8(tensor):
    tmp = torch.empty(tensor.shape, dtype=torch.bfloat16, device=tensor.device)
    tmp.uniform_(-1, 1)
    tensor.copy_(tmp.to(torch.float8_e4m3fn))
    return tensor


def create_inputs(args: _TestArgs):
    a = _fill_fp8(
        empty_layout_matrix(args.m, args.k, torch.float8_e4m3fn, args.layout[0] == "t")
    )
    b = _fill_fp8(
        empty_layout_matrix(args.k, args.n, torch.float8_e4m3fn, args.layout[1] == "t")
    )
    scale_a = torch.rand(args.m, device="cuda") * 0.25 + 0.05
    scale_b = torch.rand(args.n, device="cuda") * 0.25 + 0.05
    if args.has_bias:
        bias = torch.empty((args.n,), dtype=args.out_dtype, device="cuda")
        bias.uniform_(0.1, 1.0)
    else:
        bias = None
    return a, b, scale_a, scale_b, bias


def create_outputs(args: _TestArgs):
    return (torch.empty((args.m, args.n), dtype=args.out_dtype, device="cuda"),)


def ref_func(a, b, scale_a, scale_b, bias, c):
    ref = torch.mm(a.float(), b.float()) * scale_a[:, None] * scale_b[None, :]
    if bias is not None:
        ref = ref + bias.float()
    c.copy_(ref.to(c.dtype))


def func(a, b, scale_a, scale_b, bias, c, kwargs, layout):
    scaled_gemm(
        a,
        b,
        scale_a,
        scale_b,
        out=c,
        bias=bias,
        user_kwargs=kwargs,
        layout=layout,
        out_dtype=c.dtype,
    )


def _ptpc_scale_views(scale_a, scale_b):
    return scale_a.view(-1, 1).contiguous(), scale_b.view(1, -1).contiguous()


def scaled_mm_func(a, b, scale_a_2d, scale_b_2d, c):
    torch._scaled_mm(
        a,
        b,
        scale_a=scale_a_2d,
        scale_b=scale_b_2d,
        out_dtype=c.dtype,
        out=c,
    )


def make_triton_maxautotune_func():
    import torch._inductor.config as inductor_config

    inductor_config.max_autotune_gemm_backends = "TRITON"
    inductor_config.max_autotune_gemm_search_space = "DEFAULT"
    torch._dynamo.reset()

    def triton_maxautotune_func(a, b, scale_a_2d, scale_b_2d, c):
        out = torch._scaled_mm(
            a,
            b,
            scale_a=scale_a_2d,
            scale_b=scale_b_2d,
            out_dtype=c.dtype,
        )
        c.copy_(out)

    return torch.compile(triton_maxautotune_func, mode="max-autotune", fullgraph=True)


def tensor_nbytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def get_rotary_inputs(sample_inputs, sample_outputs):
    slot_bytes = 2 * (tensor_nbytes(sample_inputs) + tensor_nbytes(sample_outputs))
    return max(1, ROTARY_INPUTS_TARGET_BYTES // slot_bytes)


def kernel_kwargs(args: _TestArgs):
    return {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "stages": args.stages,
        "m_waves": args.m_waves,
        "n_waves": args.n_waves,
        "k_waves": args.k_waves,
        "group_m": args.group_m,
        "use_half_tile_interleaved": False,
        "split_k": args.split_k,
        "mma_m": args.mma_m,
        "mma_n": args.mma_n,
        "mma_k": args.mma_k,
    }


def check_acc(args: _TestArgs):
    _skip_if_no_fp8()
    _skip_unsupported_accuracy_layout(args)
    kwargs = kernel_kwargs(args)
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    k_scale = (args.k / 8192) ** 0.5 * args.split_k * args.k_waves
    atol = 0.5 * k_scale * (1.5 if args.has_bias else 1.0)
    rtol = 0.5
    maxdiff_out_ = []
    for _ in range(3):
        func(*inputs, *outputs, kwargs, args.layout)
        ref_func(*inputs, *ref_outputs)
        for output, ref_output in zip(outputs, ref_outputs):
            maxdiff_out = (output.float() - ref_output.float()).abs().max().item()
            maxdiff_out_.append(maxdiff_out)
            print(maxdiff_out, flush=True)
            torch.testing.assert_close(
                output.float(),
                ref_output.float(),
                atol=atol,
                rtol=rtol,
            )
    print(f"\n{args}\nmaxdiff_out:{maxdiff_out_}")


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias",
    [
        (512, 512, 512, 256, 256, 128, 2, 2, 4, 0, False),
        (512, 512, 512, 256, 256, 128, 2, 2, 4, 4, True),
        (528, 528, 512, 256, 256, 128, 2, 2, 4, 0, False),
        (512, 512, 512 + 128, 256, 256, 128, 2, 2, 4, 0, False),
        (2048, 2048, 2048, 128, 128, 128, 2, 4, 4, 0, False),
    ],
)
def test_scaled_gemm_acc_main_loop(
    layout: str,
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
    check_acc(
        _TestArgs(
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
            layout,
        )
    )


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, split_k, "
    "m_waves, n_waves, k_waves, has_bias, group_m",
    [
        (32, 384, 7168, 32, 64, 128, 5, 8, 2, 2, 1, True, 0),
        (32, 384, 7168, 32, 64, 128, 5, 8, 2, 2, 1, False, 0),
        (800, 384, 7168, 32, 64, 256, 6, 1, 1, 2, 2, True, 0),
        (800, 384, 7168, 32, 64, 256, 6, 2, 1, 2, 2, False, 0),
    ],
)
def test_scaled_gemm_acc_split_k(
    layout: str,
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
):
    assert k_waves > 0
    assert split_k > 1 or k_waves > 1
    check_acc(
        _TestArgs(
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
            layout=layout,
            split_k=split_k,
        )
    )


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("block_k", [64, 128])
def test_scaled_gemm_acc_mma_32x32x64(layout: str, has_bias: bool, block_k: int):
    check_acc(
        _TestArgs(
            m=64,
            n=64,
            k=256,
            block_m=64,
            block_n=64,
            block_k=block_k,
            stages=2,
            m_waves=2,
            n_waves=2,
            k_waves=1,
            group_m=0,
            has_bias=has_bias,
            layout=layout,
            mma_m=32,
            mma_n=32,
            mma_k=64,
        )
    )


def _cuda_time_ms(run, rotary_inputs, niters=50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(niters):
        run(i % rotary_inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / niters


def benchmark(args: _TestArgs, warmup: int = 500, niters: int = 600):
    _skip_if_no_fp8()
    assert args.layout == "nt", "torch._scaled_mm comparison is NT-only"
    assert not args.has_bias, "benchmark compares the no-bias scaled-mm path"
    kwargs = kernel_kwargs(args)
    sample_inputs = create_inputs(args)
    sample_outputs = create_outputs(args)
    rotary_inputs = get_rotary_inputs(sample_inputs, sample_outputs)
    inputs = [sample_inputs] + [create_inputs(args) for _ in range(rotary_inputs - 1)]
    outputs = [sample_outputs] + [
        create_outputs(args) for _ in range(rotary_inputs - 1)
    ]
    sm_scales = [_ptpc_scale_views(inp[2], inp[3]) for inp in inputs]
    flops = 2.0 * args.m * args.n * args.k
    print(
        f"{args}\nrotary_inputs:{rotary_inputs}, target_bytes:{ROTARY_INPUTS_TARGET_BYTES}, "
        f"warmup:{warmup}, niters:{niters}, flops:{flops:.3e}"
    )

    def run_flydsl(idx):
        func(*inputs[idx], *outputs[idx], kwargs, args.layout)

    has_scaled_mm = hasattr(torch, "_scaled_mm")
    if has_scaled_mm:
        try:
            a, b, _sa, _sb, _bias = inputs[0]
            scale_a_2d, scale_b_2d = sm_scales[0]
            (c,) = outputs[0]
            scaled_mm_func(a, b, scale_a_2d, scale_b_2d, c)
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"torch._scaled_mm unavailable: {exc}")
            has_scaled_mm = False

    def run_scaled_mm(idx):
        a, b, _sa, _sb, _bias = inputs[idx]
        scale_a_2d, scale_b_2d = sm_scales[idx]
        (c,) = outputs[idx]
        scaled_mm_func(a, b, scale_a_2d, scale_b_2d, c)

    has_triton = has_scaled_mm
    triton_fn = None
    if has_triton:
        try:
            triton_fn = make_triton_maxautotune_func()
            a, b, _sa, _sb, _bias = inputs[0]
            scale_a_2d, scale_b_2d = sm_scales[0]
            (c,) = outputs[0]
            triton_fn(a, b, scale_a_2d, scale_b_2d, c)
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"triton max-autotune unavailable: {exc}")
            has_triton = False

    def run_triton(idx):
        a, b, _sa, _sb, _bias = inputs[idx]
        scale_a_2d, scale_b_2d = sm_scales[idx]
        (c,) = outputs[idx]
        triton_fn(a, b, scale_a_2d, scale_b_2d, c)

    runners = [("flydsl", run_flydsl)]
    if has_scaled_mm:
        runners.append(("scaled_mm", run_scaled_mm))
    if has_triton:
        runners.append(("triton", run_triton))

    def run_once(i, idx):
        n = len(runners)
        for off in range(n):
            runners[(i + off) % n][1](idx)

    print("===================== [INTERLEAVED] =====================")
    for i in range(warmup):
        run_once(i, i % rotary_inputs)
        torch.cuda.synchronize()

    print("===================== [CUDA EVENTS] =====================")
    for name, run in runners:
        ms = _cuda_time_ms(run, rotary_inputs)
        print(f"{name}: {ms:.3f} ms, {flops / ms / 1e9:.2f} TFLOPS")

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(warmup, niters):
            run_once(i, i % rotary_inputs)
            torch.cuda.synchronize()
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1, max_name_column_width=140
        )
    )


@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves",
    [
        (2048, 2048, 2048, 128, 128, 128, 2, 4, 4),
        (4096, 4096, 4096, 256, 256, 128, 2, 2, 4),
        (8192, 8192, 8192, 256, 256, 128, 2, 2, 4),
    ],
)
def test_scaled_gemm_benchmark_nt(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
):
    benchmark(
        _TestArgs(
            m=m,
            n=n,
            k=k,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            k_waves=1,
            group_m=0,
            has_bias=False,
            layout="nt",
        )
    )
