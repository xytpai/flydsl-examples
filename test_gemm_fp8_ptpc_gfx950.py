import torch
import pytest
from dataclasses import dataclass
from torch.profiler import profile, ProfilerActivity

from kernels.gemm_fp8_ptpc_gfx950 import gemm_fp8_ptpc
from kernels.gemm_a16w16_gfx950_utils import GFX950_DMA_BYTES

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
    use_half_tile_interleaved: bool = True
    layout: str = "nt"
    split_k: int = 1
    out_dtype: torch.dtype = torch.bfloat16


def empty_layout_matrix(rows: int, cols: int, dtype: torch.dtype, is_t: bool):
    if is_t:
        return torch.empty((cols, rows), dtype=dtype, device="cuda").t()
    return torch.empty((rows, cols), dtype=dtype, device="cuda")


def _fill_fp8(tensor: torch.Tensor):
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
    scale_a = torch.rand((args.m,), device="cuda", dtype=torch.float32) * 0.25 + 0.05
    scale_b = torch.rand((args.n,), device="cuda", dtype=torch.float32) * 0.25 + 0.05
    if args.has_bias:
        bias = torch.empty((args.n,), dtype=torch.bfloat16, device="cuda")
        bias.uniform_(-1, 1)
    else:
        bias = None
    return a, b, scale_a, scale_b, bias


def ref_fp8_ptpc(a, b, scale_a, scale_b, bias, out):
    acc = torch.mm(a.float(), b.float())
    ref = acc * scale_a[:, None] * scale_b[None, :]
    if bias is not None:
        ref = ref + bias.float()
    out.copy_(ref.to(out.dtype))


def _kwargs(args: _TestArgs):
    return {
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


def _skip_if_no_fp8():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is required")


def check_acc(args: _TestArgs):
    _skip_if_no_fp8()
    if args.layout[0] == "t":
        a_vec_size = GFX950_DMA_BYTES
        if args.m % a_vec_size != 0:
            pytest.skip(f"column-major A requires M divisible by {a_vec_size}")

    a, b, scale_a, scale_b, bias = create_inputs(args)
    out = torch.empty((args.m, args.n), dtype=args.out_dtype, device="cuda")
    ref = torch.empty_like(out)
    k_scale = (args.k / 8192) ** 0.5
    k_scale *= args.split_k * args.k_waves
    atol = 5e-1 * k_scale * (1.5 if args.has_bias else 1.0)
    rtol = 5e-1
    maxdiffs = []
    for _ in range(3):
        gemm_fp8_ptpc(
            a,
            b,
            scale_a,
            scale_b,
            out=out,
            bias=bias,
            user_kwargs=_kwargs(args),
            layout=args.layout,
            out_dtype=args.out_dtype,
        )
        ref_fp8_ptpc(a, b, scale_a, scale_b, bias, ref)
        maxdiff = (out.float() - ref.float()).abs().max().item()
        maxdiffs.append(maxdiff)
        print(maxdiff, flush=True)
        torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)
    print(f"\n{args}\nmaxdiff_out:{maxdiffs}")


@pytest.mark.parametrize("layout", ["nt", "nn"])
@pytest.mark.parametrize("has_bias", [False, True])
def test_fp8_ptpc_acc_hti_default(layout, has_bias):
    check_acc(
        _TestArgs(
            m=256,
            n=256,
            k=256,
            block_m=256,
            block_n=256,
            block_k=128,
            stages=2,
            m_waves=2,
            n_waves=4,
            k_waves=1,
            group_m=0,
            has_bias=has_bias,
            use_half_tile_interleaved=True,
            layout=layout,
        )
    )


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_fp8_ptpc_acc_full_tile(out_dtype):
    check_acc(
        _TestArgs(
            m=128,
            n=128,
            k=256,
            block_m=128,
            block_n=128,
            block_k=128,
            stages=2,
            m_waves=4,
            n_waves=4,
            k_waves=1,
            group_m=0,
            has_bias=False,
            use_half_tile_interleaved=False,
            layout="nt",
            out_dtype=out_dtype,
        )
    )


def test_fp8_ptpc_acc_full_tile_nn():
    check_acc(
        _TestArgs(
            m=128,
            n=128,
            k=256,
            block_m=128,
            block_n=128,
            block_k=128,
            stages=2,
            m_waves=4,
            n_waves=4,
            k_waves=1,
            group_m=0,
            has_bias=False,
            use_half_tile_interleaved=False,
            layout="nn",
        )
    )


def test_fp8_ptpc_acc_split_k():
    check_acc(
        _TestArgs(
            m=128,
            n=128,
            k=512,
            block_m=128,
            block_n=128,
            block_k=128,
            stages=2,
            m_waves=4,
            n_waves=4,
            k_waves=1,
            group_m=0,
            has_bias=True,
            use_half_tile_interleaved=False,
            layout="nt",
            split_k=2,
        )
    )


def test_fp8_ptpc_allocates_bf16_output():
    _skip_if_no_fp8()
    m, n, k = 256, 256, 256
    a = _fill_fp8(torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda"))
    b = _fill_fp8(torch.empty((n, k), dtype=torch.float8_e4m3fn, device="cuda")).t()
    scale_a = torch.ones((m,), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((n,), device="cuda", dtype=torch.float32)
    out = gemm_fp8_ptpc(a, b, scale_a, scale_b, layout="nt")
    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16


def test_fp8_ptpc_rejects_unsupported_k_partitioning():
    _skip_if_no_fp8()
    m, n, k = 256, 256, 128
    a = _fill_fp8(torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda"))
    b = _fill_fp8(torch.empty((n, k), dtype=torch.float8_e4m3fn, device="cuda")).t()
    scale_a = torch.ones((m,), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((n,), device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError, match="HTI requires at least two"):
        gemm_fp8_ptpc(
            a,
            b,
            scale_a,
            scale_b,
            layout="nt",
            user_kwargs={"use_half_tile_interleaved": True, "block_k": 128, "stages": 2},
        )


def create_outputs(args: _TestArgs):
    c = torch.empty((args.m, args.n), dtype=args.out_dtype, device="cuda")
    return (c,)


def tensor_nbytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def get_rotary_inputs(sample_inputs, sample_outputs):
    slot_bytes = 2 * (tensor_nbytes(sample_inputs) + tensor_nbytes(sample_outputs))
    rotary_inputs = ROTARY_INPUTS_TARGET_BYTES // slot_bytes
    return max(1, int(rotary_inputs))


def ref_func(a, b, scale_a, scale_b, out):
    torch._scaled_mm(
        a,
        b,
        scale_a=scale_a.view(-1, 1).contiguous(),
        scale_b=scale_b.view(1, -1).contiguous(),
        out_dtype=out.dtype,
        out=out,
    )


def benchmark(args: _TestArgs, warmup: int = 500, niters: int = 600):
    _skip_if_no_fp8()
    kwargs = _kwargs(args)
    sample_inputs = create_inputs(args)
    sample_outputs = create_outputs(args)
    rotary_inputs = get_rotary_inputs(sample_inputs, sample_outputs)
    inputs = [sample_inputs] + [create_inputs(args) for _ in range(rotary_inputs - 1)]
    outputs = [sample_outputs] + [
        create_outputs(args) for _ in range(rotary_inputs - 1)
    ]
    flops = 2.0 * args.m * args.n * args.k
    print(
        f"rotary_inputs:{rotary_inputs}, target_bytes:{ROTARY_INPUTS_TARGET_BYTES}, "
        f"warmup:{warmup}, niters:{niters}, flops:{flops:.3e}"
    )

    def run_flydsl(idx):
        a, b, scale_a, scale_b, bias = inputs[idx]
        (c,) = outputs[idx]
        gemm_fp8_ptpc(
            a,
            b,
            scale_a,
            scale_b,
            out=c,
            bias=bias,
            user_kwargs=kwargs,
            layout=args.layout,
            out_dtype=args.out_dtype,
        )

    has_scaled_mm = hasattr(torch, "_scaled_mm")
    if has_scaled_mm:
        try:
            a, b, scale_a, scale_b, _bias = inputs[0]
            (c,) = outputs[0]
            ref_func(a, b, scale_a, scale_b, c)
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"torch._scaled_mm unavailable: {exc}")
            has_scaled_mm = False

    def run_ref(idx):
        a, b, scale_a, scale_b, _bias = inputs[idx]
        (c,) = outputs[idx]
        ref_func(a, b, scale_a, scale_b, c)

    def run_once(i, idx):
        if not has_scaled_mm:
            run_flydsl(idx)
            return
        if i % 2 == 0:
            run_ref(idx)
            run_flydsl(idx)
        else:
            run_flydsl(idx)
            run_ref(idx)

    print("===================== [INTERLEAVED] =====================")
    for i in range(warmup):
        run_once(i, i % rotary_inputs)
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timed_iters = 50
    start.record()
    for i in range(timed_iters):
        run_flydsl(i % rotary_inputs)
    end.record()
    torch.cuda.synchronize()
    flydsl_ms = start.elapsed_time(end) / timed_iters
    print(
        f"flydsl_fp8_ptpc: {flydsl_ms:.3f} ms, "
        f"{flops / flydsl_ms / 1e9:.2f} TFLOPS"
    )
    if has_scaled_mm:
        start.record()
        for i in range(timed_iters):
            run_ref(i % rotary_inputs)
        end.record()
        torch.cuda.synchronize()
        scaled_ms = start.elapsed_time(end) / timed_iters
        print(
            f"torch._scaled_mm: {scaled_ms:.3f} ms, "
            f"{flops / scaled_ms / 1e9:.2f} TFLOPS"
        )

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(warmup, niters):
            run_once(i, i % rotary_inputs)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


def test_fp8_ptpc_benchmark_8192_hti_nt():
    benchmark(
        _TestArgs(
            m=8192,
            n=8192,
            k=8192,
            block_m=256,
            block_n=256,
            block_k=128,
            stages=2,
            m_waves=2,
            n_waves=4,
            k_waves=1,
            group_m=0,
            has_bias=False,
            use_half_tile_interleaved=True,
            layout="nt",
        )
    )
