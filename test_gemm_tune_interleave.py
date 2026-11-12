# SPDX-License-Identifier: MIT
"""gfx1250 tuner integration for the optional quadrant schedule."""
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

import gemm_tune as tuner


@pytest.fixture
def device_props(monkeypatch):
    props = SimpleNamespace(
        multi_processor_count=256, warp_size=32,
        max_threads_per_multi_processor=2048,
        shared_memory_per_block=320 * 1024,
        shared_memory_per_multiprocessor=320 * 1024,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_: props)
    return props


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nt_candidates_include_valid_interleaving(dtype, device_props):
    args = tuner.Args(dtype, 8192, 8192, 8192, "nt", enable_split_k=True)
    configs = tuner.gemm_gfx1250_get_configs(args)
    plain = [c for c in configs if not c["interleave"]]
    interleaved = [c for c in configs if c["interleave"]]
    assert len(plain) == 650
    assert len(interleaved) == 308
    assert all(c["split_k"] == 1 for c in configs)  # existing large-grid pruner
    plain_tiles = {
        tuple((key, value) for key, value in c.items() if key != "interleave")
        for c in plain
    }
    for config in interleaved:
        tile = tuple((key, value) for key, value in config.items() if key != "interleave")
        assert tile in plain_tiles
        tuner.validate_gemm_gfx1250_config(
            args.m, args.n, args.k, config["block_m"], config["block_n"],
            config["block_k"], config["stages"], config["m_waves"],
            config["n_waves"], config["group_m"],
            tuner.GEMM_DTYPE_FP16 if dtype is torch.float16 else tuner.GEMM_DTYPE_BF16,
            config["mma_m"], config["mma_n"], config["mma_k"],
            True, config["split_k"], interleave=True,
        )



def test_split_k_interleaved_candidates(device_props):
    args = tuner.Args(torch.bfloat16, 128, 128, 3072, "nt", enable_split_k=True)
    configs = tuner.gemm_gfx1250_get_configs(args)
    interleaved = [c for c in configs if c["interleave"]]
    assert any(c["split_k"] == 3 for c in interleaved)
    assert any(c["split_k"] > 1 for c in interleaved)
    for config in interleaved:
        tuner.validate_gemm_gfx1250_config(
            args.m, args.n, args.k, config["block_m"], config["block_n"],
            config["block_k"], config["stages"], config["m_waves"],
            config["n_waves"], config["group_m"], tuner.GEMM_DTYPE_BF16,
            config["mma_m"], config["mma_n"], config["mma_k"],
            True, config["split_k"], interleave=True,
        )


def test_nn_does_not_enumerate_interleaving(device_props):
    args = tuner.Args(torch.bfloat16, 256, 256, 1024, "nn", enable_split_k=True)
    configs = tuner.gemm_gfx1250_get_configs(args)
    assert configs
    assert all(c["interleave"] is False for c in configs)


def test_pruner_preserves_both_schedules_and_counts_padding(device_props):
    config = dict(
        block_m=256, block_n=256, block_k=32, stages=2,
        m_waves=2, n_waves=2, split_k=1, group_m=0,
    )
    pruner = tuner.GemmConfigPruner(256, 256, 1024, device_props, 2, arch="gfx1250")
    plain = dict(config, interleave=False)
    interleaved = dict(config, interleave=True)
    assert pruner._lds_bytes(plain) == 256 * 256 * 2
    assert pruner._lds_bytes(interleaved) == 256 * (256 + 8) * 2
    assert pruner.prune([plain, interleaved]) == [plain, interleaved]


def test_winner_serialization_keeps_interleave(monkeypatch):
    # Test winner bookkeeping deterministically, separately from GPU timing.
    configs = [dict(interleave=False), dict(interleave=True)]
    monkeypatch.setattr(tuner, "gpu_arch", "gfx1250")
    monkeypatch.setattr(tuner, "gemm_a16w16_get_configs", lambda _: configs)
    monkeypatch.setattr(tuner, "parallel_compile_policies", lambda _, cs, **kw: cs)
    monkeypatch.setattr(
        tuner, "tuning_benchmark",
        lambda _, kwargs: 5.0 if kwargs["interleave"] else 10.0,
    )
    result = tuner.tune_single(tuner.Args(torch.bfloat16, 256, 256, 1024, "nt"))
    restored = json.loads(json.dumps(asdict(result)))
    assert restored["config"]["interleave"] is True
    assert restored["duration"] == 5.0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_compile_benchmark_and_replay_candidate(dtype):
    if not torch.cuda.is_available() or tuner.gpu_arch != "gfx1250":
        pytest.skip("requires gfx1250")
    torch.set_num_threads(1)
    args = tuner.Args(dtype, 128, 128, 512, "nt", compile_workers=2)
    configs = tuner.gemm_gfx1250_get_configs(args)
    pair = [
        c for c in configs
        if all(c[key] == value for key, value in dict(
            block_m=128, block_n=128, block_k=128, stages=2,
            m_waves=2, n_waves=2, split_k=1, group_m=0,
        ).items())
    ]
    assert len(pair) == 2 and {c["interleave"] for c in pair} == {False, True}
    compiled = tuner.parallel_compile_policies(args, pair, max_workers=2)
    assert len(compiled) == 2
    for config in compiled:
        # Actual existing profiler benchmark and input/bias generator.
        duration = tuner.tuning_benchmark(args, kwargs=config, niters=5)
        assert 0 < duration < float("inf")
        restored = json.loads(json.dumps(config))
        a, b, bias = tuner.create_inputs(args)
        out = tuner.create_outputs(args)[0]
        tuner.run_gemm(a, b, out, bias, restored, args.layout)
        torch.cuda.synchronize()
        reference = (a.float() @ b.float()).to(dtype)
        reference = (reference.float() + bias.float()).to(dtype)
        torch.testing.assert_close(out, reference, atol=.03, rtol=.02)


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
