import argparse

import pytest
import torch

from kernels.hgemm_gfx950 import benchmark_layout_gemm, layout_gemm

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA/ROCm is required for FlyDSL GEMM tests",
)


if triton is not None:

    @triton.jit
    def _triton_gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + offs_n[None, :] * stride_bn + k_idxs[:, None] * stride_bk,
                mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
                other=0.0,
            )
            acc += tl.dot(a, b, out_dtype=tl.float32)

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def benchmark_triton_gemm(
    m: int = 8192,
    n: int = 8192,
    k: int = 8192,
    *,
    warmup: int = 10,
    iters: int = 50,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    group_m: int = 8,
    num_warps: int = 8,
) -> dict[str, float] | None:
    if triton is None:
        print("triton_gemm skipped: triton is not installed")
        return None

    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((m, n), dtype=a.dtype, device="cuda")
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    def launch():
        _triton_gemm_kernel[grid](
            a,
            b,
            c,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            m,
            n,
            k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            num_warps=num_warps,
        )

    launch()
    torch.cuda.synchronize()

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iters):
        launch()
    end.record(stream)
    end.synchronize()

    avg_ms = start.elapsed_time(end) / iters
    tflops = (2.0 * m * n * k) / (avg_ms / 1e3) / 1e12
    result = {
        "m": float(m),
        "n": float(n),
        "k": float(k),
        "avg_ms": avg_ms,
        "tflops": tflops,
        "block_m": float(block_m),
        "block_n": float(block_n),
        "block_k": float(block_k),
    }
    print(
        "triton_gemm "
        f"tile={block_m}x{block_n}x{block_k} M={m} N={n} K={k}: "
        f"{avg_ms:.4f} ms, {tflops:.2f} TFLOP/s"
    )
    return result


def test_layout_gemm_correctness():
    torch.manual_seed(0)
    m, n, k = 256, 256, 128
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((m, n), dtype=a.dtype, device="cuda")

    out = layout_gemm(a, b, c)
    torch.cuda.synchronize()

    assert out.data_ptr() == c.data_ptr()
    ref = (a.float() @ b.float().T).to(out.dtype)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize(
    "m, n, k, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, HAS_BIAS, GROUP_M, USE_HALF_TILE_INTERLEAVED",
    [
        # (16384, 16384, 16384, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        # (8192, 8192, 8192, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, False),
        # (8160, 8160, 8160, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        # (4096, 4096, 8192, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        # (4096, 4096, 4096, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 1, 4, 4, 1, False, 0, False),
    ],
)
def test_layout_gemm_benchmark_smoke(
    dtype: str,
    m: int,
    n: int,
    k: int,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    STAGES: int,
    SPLIT_K: int,
    BLOCK_M_WARPS: int,
    BLOCK_N_WARPS: int,
    BLOCK_K_WARPS: int,
    HAS_BIAS: bool,
    GROUP_M: int,
    USE_HALF_TILE_INTERLEAVED: bool,
):
    if dtype != "bf16":
        pytest.skip("layout_gemm currently supports bf16 only")
    if SPLIT_K != 1 or BLOCK_K_WARPS != 1:
        pytest.skip(
            "layout_gemm benchmark path currently supports split_k=1 and block_k_warps=1"
        )
    if m % TILE_M or n % TILE_N or k % TILE_K:
        pytest.skip("layout_gemm benchmark path currently requires tile-aligned M/N/K")

    result = benchmark_layout_gemm(
        m,
        n,
        k,
        warmup=10,
        iters=20,
        block_m=TILE_M,
        block_n=TILE_N,
        block_k=TILE_K,
        stages=STAGES,
        m_waves=BLOCK_M_WARPS,
        n_waves=BLOCK_N_WARPS,
    )

    assert result["avg_ms"] > 0
    assert result["tflops"] > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark layout-system GEMM")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--profiler-row-limit", type=int, default=20)
    parser.add_argument("--block-m-warps", type=int, default=4)
    parser.add_argument("--block-n-warps", type=int, default=4)
    parser.add_argument("--skip-triton", action="store_true")
    parser.add_argument("--triton-block-m", type=int, default=128)
    parser.add_argument("--triton-block-n", type=int, default=128)
    parser.add_argument("--triton-block-k", type=int, default=64)
    parser.add_argument("--triton-group-m", type=int, default=8)
    parser.add_argument("--triton-num-warps", type=int, default=8)
    args = parser.parse_args()

    flydsl_result = benchmark_layout_gemm(
        args.m,
        args.n,
        args.k,
        warmup=args.warmup,
        iters=args.iters,
        profiler_row_limit=args.profiler_row_limit,
        m_waves=args.block_m_warps,
        n_waves=args.block_n_warps,
    )

    triton_result = None
    if not args.skip_triton:
        triton_result = benchmark_triton_gemm(
            args.m,
            args.n,
            args.k,
            warmup=args.warmup,
            iters=args.iters,
            block_m=args.triton_block_m,
            block_n=args.triton_block_n,
            block_k=args.triton_block_k,
            group_m=args.triton_group_m,
            num_warps=args.triton_num_warps,
        )

    if triton_result is not None:
        ratio = flydsl_result["tflops"] / triton_result["tflops"]
        print(f"FlyDSL / Triton TFLOP ratio: {ratio:.3f}x")
