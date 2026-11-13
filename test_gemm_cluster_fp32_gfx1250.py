"""FP32 cluster split-K precision, graph lifetime, and generated ISA."""

import inspect
import re
import subprocess
from pathlib import Path

import pytest
import torch

import kernels.gemm_a16w16_tdm_gfx1250 as g


@pytest.fixture(autouse=True)
def gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("requires gfx1250")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
        pytest.skip("requires gfx1250")
    torch.set_num_threads(1)


CFG = {
    "block_m": 16,
    "block_n": 16,
    "block_k": 256,
    "stages": 5,
    "m_waves": 1,
    "n_waves": 1,
    "split_k": 5,
}


def make_inputs(dtype=torch.bfloat16):
    a = torch.empty(6, 5120, device="cuda", dtype=dtype)
    b = torch.empty(32, 5120, device="cuda", dtype=dtype).t()
    c = torch.empty(6, 32, device="cuda", dtype=dtype)
    return a, b, c


def test_cluster_is_the_only_split_k_method():
    for fn in (
        g.gemm_a16w16_gfx1250,
        g.build_gemm_gfx1250,
        g.validate_gemm_gfx1250_config,
    ):
        assert "split_k_atomic" not in inspect.signature(fn).parameters
    args = (g.GEMM_DTYPE_BF16, 16, 16, 256, 5, 1, 1, 0, False, True, 16, 16, 32)
    assert g.make_gemm_a16w16_gfx1250_kernel_name(*args, 5).endswith("_fp32sk")
    assert not g.make_gemm_a16w16_gfx1250_kernel_name(*args, 1).endswith("_fp32sk")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [False, True])
def test_ref_shape_precision_and_replay(dtype, has_bias, monkeypatch):
    a, b, c = make_inputs(dtype)
    a.zero_()
    b.zero_()
    for part in range(5):
        a[:, part * 1024] = b[part * 1024, :] = 1
        a[:, part * 1024 + 1] = b[part * 1024 + 1, :] = 1 / 16
    bias = torch.full((32,), -0.5, device="cuda", dtype=dtype) if has_bias else None
    expected = torch.full_like(c, 5 * (1 + 1 / 256))
    if has_bias:
        expected = (expected.float() + bias.float()).to(dtype)
    allocations = []
    original = torch.empty

    def capture(*args, **kwargs):
        t = original(*args, **kwargs)
        allocations.append(t)
        return t

    with monkeypatch.context() as p:
        p.setattr(torch, "empty", capture)
        g.gemm_a16w16_gfx1250(a, b, c, bias, **CFG)
    torch.cuda.synchronize()
    ws = [t for t in allocations if t.dtype == torch.float32]
    assert len(ws) == 1 and ws[0].numel() == 2 * 5 * 16 * 16
    torch.testing.assert_close(c, expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g.gemm_a16w16_gfx1250(a, b, c, bias, **CFG)
    for i in range(20):
        c.fill_(float("nan") if i % 2 else 999)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(c, expected, atol=0, rtol=0)
    graph.reset()


def test_no_counter_allocations_and_split1_no_workspace(monkeypatch):
    a, b, c = make_inputs()
    a.fill_(0.25)
    b.fill_(0.125)
    for sk in (1, 5):
        g.gemm_a16w16_gfx1250(a, b, c, **dict(CFG, split_k=sk))
    torch.cuda.synchronize()
    orig_empty, _orig_zeros = torch.empty, torch.zeros
    calls = []

    def empty(*args, **kwargs):
        t = orig_empty(*args, **kwargs)
        calls.append(t)
        return t

    def zeros(*args, **kwargs):
        raise AssertionError("cluster split-K must not initialize counters/output")

    with monkeypatch.context() as p:
        p.setattr(torch, "empty", empty)
        p.setattr(torch, "zeros", zeros)
        g.gemm_a16w16_gfx1250(a, b, c, **dict(CFG, split_k=1))
        assert not calls
        g.gemm_a16w16_gfx1250(a, b, c, **CFG)
    torch.cuda.synchronize()
    assert len(calls) == 1 and calls[0].dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_two_graphs_captured_on_same_stream_replay_concurrently(dtype):
    slots = [make_inputs(dtype), make_inputs(dtype)]
    for i, (a, b, _) in enumerate(slots):
        a.fill_((i + 1) / 16)
        b.fill_(1 / 32)
        g.gemm_a16w16_gfx1250(a, b, slots[i][2], **CFG)
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    graphs = []
    for a, b, c in slots:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            junk = torch.empty(4096, device="cuda")
            junk.fill_(999)
            del junk
            g.gemm_a16w16_gfx1250(a, b, c, stream=capture_stream, **CFG)
        graphs.append(graph)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for _ in range(100):
        for stream, graph in zip(streams, graphs):
            with torch.cuda.stream(stream):
                graph.replay()
    torch.cuda.synchronize()
    for i, (_, _, c) in enumerate(slots):
        torch.testing.assert_close(c, torch.full_like(c, (i + 1) * 10), atol=0, rtol=0)
    # Eager calls on explicit streams must also retain independent scratch.
    for _ in range(20):
        for stream, (a, b, c) in zip(streams, slots):
            g.gemm_a16w16_gfx1250(a, b, c, stream=stream, **CFG)
    torch.cuda.synchronize()
    for i, (_, _, c) in enumerate(slots):
        torch.testing.assert_close(c, torch.full_like(c, (i + 1) * 10), atol=0, rtol=0)
    for graph in graphs:
        graph.reset()


def test_fp8_scale_after_fp32_reduction():
    a = torch.full((16, 768), 16.0, device="cuda").to(torch.float8_e4m3fn)
    b = torch.full((16, 768), 16.0, device="cuda").to(torch.float8_e4m3fn).t()
    sa = torch.full((16,), 0.1, device="cuda")
    sb = torch.full((16,), 0.25, device="cuda")
    bias = torch.full((16,), 0.5, device="cuda", dtype=torch.bfloat16)
    ref = ((a.float() @ b.float()) * sa[:, None] * sb[None, :]).bfloat16()
    ref = (ref.float() + bias.float()).bfloat16()
    out = g.gemm_fp8_ptpc_gfx1250(
        a,
        b,
        sa,
        sb,
        bias=bias,
        block_m=16,
        block_n=16,
        block_k=128,
        stages=2,
        m_waves=1,
        n_waves=1,
        split_k=3,
    )
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_isa_uses_cluster_and_fp32_vectors_no_atomics(monkeypatch, tmp_path):
    tools = list(
        Path("/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core").rglob(
            "llvm-objdump"
        )
    )
    if not tools:
        pytest.skip("llvm-objdump required")
    compiled = []
    original = g.run_cached

    def capture(*args, **kwargs):
        cf = original(*args, **kwargs)
        compiled.append(cf)
        return cf

    monkeypatch.setattr(g, "run_cached", capture)
    a, b, c = make_inputs()
    a.fill_(0.25)
    b.fill_(0.125)
    g.gemm_a16w16_gfx1250(a, b, c, **CFG)
    torch.cuda.synchronize()
    ir = compiled[-1]._keepalive._ir_text
    assert "_fp32sk" in ir
    raw = re.search(r'bin = "((?:\\.|[^"\\])*)"', ir)[1]
    data = bytearray()
    i = 0
    while i < len(raw):
        if raw[i] == "\\":
            if re.fullmatch("[0-9a-fA-F]{2}", raw[i + 1 : i + 3]):
                data.append(int(raw[i + 1 : i + 3], 16))
                i += 3
            else:
                data.extend(raw[i + 1].encode())
                i += 2
        else:
            data.extend(raw[i].encode())
            i += 1
    elf = tmp_path / "cluster.co"
    elf.write_bytes(data)
    asm = subprocess.check_output(
        [str(tools[0]), "-d", "--mcpu=gfx1250", str(elf)], text=True
    )
    ops = [
        line.split()[0] for line in asm.splitlines() if "//" in line and line.strip()
    ]
    assert not any("atomic" in op for op in ops)
    assert "s_barrier_signal" in ops and "s_barrier_wait" in ops
    assert any(
        op.startswith(("buffer_store_dwordx4", "buffer_store_b128")) for op in ops
    )
    assert any(op.startswith(("buffer_load_dwordx4", "buffer_load_b128")) for op in ops)
    (tmp_path / "cluster.s").write_text(asm)
