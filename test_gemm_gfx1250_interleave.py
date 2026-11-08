# SPDX-License-Identifier: MIT
"""Optional quadrant schedule: exact parity with the existing GEMM schedule."""

import pytest
import torch

from kernels.gemm_a16w16_tdm_gfx1250 import (
    GEMM_DTYPE_BF16,
    gemm_a16w16_gfx1250,
    gemm_fp8_ptpc_gfx1250,
    make_gemm_a16w16_gfx1250_kernel_name,
    validate_gemm_gfx1250_config,
)


def require_gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx1250":
        pytest.skip(f"requires gfx1250, got {arch}")


def offset_matrix(rows, cols, dtype):
    alignment = 16 // torch.empty((), dtype=dtype).element_size()
    storage = torch.full(
        (rows + 2, (cols + alignment - 1) // alignment * alignment + 2 * alignment),
        17.0, device="cuda", dtype=dtype,
    )
    matrix = storage[1:rows + 1, alignment:cols + alignment]
    matrix.copy_((torch.randn(rows, cols, device="cuda") * 0.2).to(dtype))
    return matrix, storage


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("stages,k_tiles", [(2, 1), (2, 2), (2, 3), (3, 2), (3, 5), (4, 3), (4, 8)])
@pytest.mark.parametrize("split_k,has_bias", [(1, False), (1, True), (3, True)])
def test_interleaved_pipeline_replay(dtype, stages, k_tiles, split_k, has_bias):
    require_gfx1250()
    torch.manual_seed(129)
    torch.set_num_threads(1)
    fp8 = dtype == torch.float8_e4m3fn
    tile = 256 if fp8 else 128
    block_k = 128 if fp8 or stages < 4 else 64
    # FP8 tile256 needs at most four BK128 stages (278528B).
    m, n, k = 137, 144, split_k * block_k * k_tiles
    output_dtype = torch.bfloat16 if fp8 else dtype
    a, _ = offset_matrix(m, k, dtype)
    weight, _ = offset_matrix(n, k, dtype)
    b = weight.t()
    outputs = [offset_matrix(m, n, output_dtype) for _ in range(2)]
    scale_a = torch.rand(m, device="cuda") * 0.5 + 0.1
    scale_b = torch.rand(n, device="cuda") * 0.5 + 0.1
    bias = torch.randn(n, device="cuda", dtype=output_dtype) * 0.1 if has_bias else None
    config = dict(
        block_m=tile, block_n=tile, block_k=block_k, stages=stages,
        m_waves=2, n_waves=2, group_m=3, split_k=split_k,
    )

    def run(interleave):
        out = outputs[int(interleave)][0]
        if fp8:
            return gemm_fp8_ptpc_gfx1250(
                a, b, scale_a, scale_b, out, bias, interleave=interleave, **config,
            )
        return gemm_a16w16_gfx1250(a, b, out, bias, interleave=interleave, **config)

    for flag in (False, True):
        run(flag)
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs[1][0], outputs[0][0], atol=0, rtol=0)
    graphs = []
    try:
        for flag in (False, True):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(flag)
            graphs.append(graph)
        for rep in range(4):
            a.copy_((-a.float()).to(dtype))
            weight.copy_((-weight.float()).to(dtype))
            if rep == 1:
                a.zero_()
            elif rep == 2:
                a.copy_((torch.randn(m, k, device="cuda") * 0.2).to(dtype))
            if fp8:
                scale_a.mul_(0.9)
                scale_b.mul_(1.1)
            for (out, _), graph in zip(outputs, graphs):
                out.fill_(float("nan"))
                graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs[1][0], outputs[0][0], atol=0, rtol=0)
            assert torch.isfinite(outputs[1][0]).all()
            for _, storage in outputs:
                assert torch.all(storage[0] == 17) and torch.all(storage[-1] == 17)
                assert torch.all(storage[:, :8] == 17)
                assert torch.all(storage[:, n + 8:] == 17)
    finally:
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_m,block_n,m_waves,n_waves", [
    (256, 256, 2, 4), (128, 256, 2, 2), (256, 128, 2, 2),
])
@pytest.mark.parametrize("split_k", [1, 2, 5, 9])
def test_interleaved_rectangular_wave_tiles(dtype, block_m, block_n, m_waves, n_waves, split_k):
    require_gfx1250()
    torch.manual_seed(333)
    m, n, k = 257, 264, split_k * 384
    a = torch.randn(m, k, device="cuda", dtype=dtype) * 0.2
    b = (torch.randn(n, k, device="cuda", dtype=dtype) * 0.2).t()
    config = dict(
        block_m=block_m, block_n=block_n, m_waves=m_waves, n_waves=n_waves,
        block_k=128, stages=2, split_k=split_k, group_m=4,
    )
    original = gemm_a16w16_gfx1250(a, b, **config)
    interleaved = gemm_a16w16_gfx1250(a, b, **config, interleave=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(interleaved, original, rtol=0, atol=0)
    if split_k == 1:
        reference = (a.float() @ b.float()).to(dtype)
        torch.testing.assert_close(interleaved.float(), reference.float(), atol=0.03, rtol=0.03)


@pytest.mark.parametrize("changes", [
    {"b_k_contiguous": False}, {"block_m": 96}, {"m_waves": 4},
    {"interleave": "yes"},
])
def test_interleaved_configuration_rejected(changes):
    config = dict(
        m=128, n=128, k=512, block_m=128, block_n=128, block_k=128,
        stages=2, m_waves=2, n_waves=2, group_m=0,
        dtype_id=GEMM_DTYPE_BF16, mma_m=16, mma_n=16, mma_k=32,
        b_k_contiguous=True, split_k=1, interleave=True,
    )
    config.update(changes)
    with pytest.raises(ValueError):
        validate_gemm_gfx1250_config(**config)


def test_schedule_kernel_names_are_separate():
    args = (GEMM_DTYPE_BF16, 128, 128, 128, 2, 2, 2, 0, False, True, 16, 16, 32, 1)
    original = make_gemm_a16w16_gfx1250_kernel_name(*args)
    assert original == make_gemm_a16w16_gfx1250_kernel_name(*args, interleave=False)
    assert make_gemm_a16w16_gfx1250_kernel_name(*args, interleave=True) == original + "_interleave"



@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("has_bias", [False, True])
def test_interleaved_odd_n_and_independent_reference(dtype, has_bias):
    require_gfx1250()
    torch.manual_seed(736)
    fp8 = dtype == torch.float8_e4m3fn
    output_dtype = torch.bfloat16 if fp8 else dtype
    m, n, k = 131, 139, 512
    a, _ = offset_matrix(m, k, dtype)
    weight, _ = offset_matrix(n, k, dtype)
    out, storage = offset_matrix(m, n, output_dtype)
    scale_a = torch.rand(m, device="cuda") * 0.25 + 0.125
    scale_b = torch.rand(n, device="cuda") * 0.25 + 0.125
    bias = torch.randn(n, device="cuda", dtype=output_dtype) * 0.1 if has_bias else None
    config = dict(
        block_m=256, block_n=256, block_k=128, stages=2,
        m_waves=2, n_waves=2, interleave=True,
    )
    if fp8:
        gemm_fp8_ptpc_gfx1250(a, weight.t(), scale_a, scale_b, out, bias, **config)
    else:
        gemm_a16w16_gfx1250(a, weight.t(), out, bias, **config)
    torch.cuda.synchronize()
    # Small full CPU FP32 oracle; it does not reuse either kernel schedule.
    reference = a.cpu().float() @ weight.cpu().float().t()
    if fp8:
        reference = reference * scale_a.cpu()[:, None] * scale_b.cpu()[None, :]
    reference = reference.to(output_dtype)
    if bias is not None:
        reference = (reference.float() + bias.cpu().float()).to(output_dtype)
    torch.testing.assert_close(out.cpu(), reference, atol=0.015, rtol=0.02)
    assert torch.all(storage[:, :8] == 17) and torch.all(storage[:, n + 8:] == 17)
    assert torch.all(storage[0] == 17) and torch.all(storage[-1] == 17)


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
