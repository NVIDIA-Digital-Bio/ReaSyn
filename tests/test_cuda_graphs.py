"""Regression tests for CUDA-graph autoregressive decoding.

These tests require the real AR checkpoint and a CUDA device. They are skipped
cleanly on ordinary CPU-only developer machines.
"""

import gc
import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from cuda_graph_testlib import (  # noqa: E402
    AR_CKPT,
    bare_sampler,
    check_graph,
    check_sampler_contract_and_guards,
    load_model,
    make_sequences,
    production_sampler,
    run_lock_failure,
)
from reasyn.chem.mol import Molecule  # noqa: E402
from reasyn.sampler.parallel import Worker, WorkerPool  # noqa: E402
from reasyn.sampler.sampler import MAX_CUDAGRAPH_CACHE_SIZE, Sampler  # noqa: E402


@pytest.fixture(scope="module")
def ar_model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not hasattr(torch.cuda, "CUDAGraph") or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA graphs and BF16 are required")
    if not pathlib.Path(AR_CKPT).is_file():
        pytest.skip(f"AR checkpoint not found: {AR_CKPT}")
    torch.cuda.set_device(0)
    model = load_model(AR_CKPT)
    yield model
    del model
    gc.collect()
    torch.cuda.empty_cache()


def test_actual_env_graph_eager_parity_many_same_shape_cases(ar_model):
    """Exercise real Sampler init/env paths, replay, RNG, and pool ordering."""
    assert check_graph(ar_model)


def test_graph_guards_and_fp32_output_ownership(ar_model):
    class EditflowStub:
        model_type = "editflow"

    check_sampler_contract_and_guards(ar_model, EditflowStub())

    sampler = production_sampler(ar_model, True)
    static = torch.randn(
        (8, ar_model.vocab_size), dtype=torch.float32, device=sampler.device
    )

    def fake_graph(*_args, **_kwargs):
        return static

    sampler._run_ar_graph = fake_graph
    output = sampler._forward_ar_batched(
        torch.empty(0, device=sampler.device),
        torch.empty(0, device=sampler.device),
        make_sequences(917, 7),
    )
    snapshot = output.clone()
    static.add_(1)
    assert output.dtype == torch.float32
    assert output.data_ptr() != static.data_ptr()
    assert torch.equal(output, snapshot)


def test_cpu_model_never_enables_graphs_on_a_gpu_host(monkeypatch):
    class Dummy(torch.nn.Module):
        def __init__(self, model_type):
            super().__init__()
            self.model_type = model_type
            self.p = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("REASYN_CUDAGRAPH", "1")
    monkeypatch.setenv("REASYN_BATCHED_AR", "1")
    monkeypatch.setenv("REASYN_BUCKET", "1")
    monkeypatch.setenv("REASYN_BF16", "1")
    sampler = Sampler(
        None,
        None,
        Molecule("CCO"),
        [Dummy("autoregressive"), Dummy("editflow")],
    )
    assert sampler.device.type == "cpu"
    assert not sampler._cudagraph


def test_bounded_graph_cache_has_bounded_growth_and_releases(ar_model):
    if torch.cuda.get_device_properties(0).total_memory < 16 * 2**30:
        pytest.skip("large-shape graph memory regression requires at least 16 GiB")
    torch.manual_seed(2026)
    smiles = torch.randint(
        1, ar_model.encoder.smiles_emb.num_embeddings, (1, 128), device="cuda"
    )
    with torch.inference_mode():
        code, mask = ar_model.encoder(smiles)

    sampler = bare_sampler(ar_model, True)
    shapes = [
        (1, 64), (8, 64), (32, 64), (128, 64), (256, 64),
        (1, 256), (8, 256), (32, 256), (128, 256), (256, 256),
        (1, 512), (8, 512), (32, 512), (128, 512), (256, 512),
    ]
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_reserved()
    try:
        with torch.inference_mode():
            for batch_size, length in shapes:
                sampler._forward_ar_batched(
                    code,
                    mask,
                    make_sequences(batch_size * 1000 + length, batch_size, length),
                )
        torch.cuda.synchronize()
        growth = torch.cuda.memory_reserved() - baseline
        print(
            f"MEMORY_GROWTH keys={len(shapes)} reserved_delta_mib={growth / 2**20:.1f}"
        )
        assert len(sampler._graph_cache) == MAX_CUDAGRAPH_CACHE_SIZE
        assert len({entry["g"].pool() for entry in sampler._graph_cache.values()}) == (
            MAX_CUDAGRAPH_CACHE_SIZE
        )
        # This selected grid includes the largest legal key. Pool identity catches
        # accidental sharing; the LRU and bound catch runaway aggregate reservation.
        assert growth < 12 * 2**30, f"graph cache reserved {growth / 2**30:.2f} GiB"
    finally:
        sampler._clear_graph_cache()
    torch.cuda.synchronize()
    released = torch.cuda.memory_reserved()
    print(
        f"MEMORY_AFTER_CLEAR reserved_mib={released / 2**20:.1f} "
        f"baseline_mib={baseline / 2**20:.1f}"
    )
    assert released <= baseline + 512 * 2**20


def test_worker_device_binding_and_single_worker_guard(monkeypatch):
    class StopBeforeCheckpointLoad(Exception):
        pass

    selected = []

    def stop_after_device(device):
        selected.append(device)
        raise StopBeforeCheckpointLoad

    monkeypatch.setattr(os, "sched_setaffinity", lambda *_args: None, raising=False)
    monkeypatch.setattr(torch.cuda, "set_device", stop_after_device)
    worker = Worker(
        [pathlib.Path("ar.ckpt"), pathlib.Path("eb.ckpt")], None, None, "3", None
    )
    with pytest.raises(StopBeforeCheckpointLoad):
        worker.run()
    assert selected == [torch.device("cuda:3")]

    monkeypatch.setenv("REASYN_CUDAGRAPH", "1")
    monkeypatch.setenv("REASYN_BATCHED_AR", "1")
    monkeypatch.setenv("REASYN_BUCKET", "1")
    monkeypatch.setenv("REASYN_BF16", "1")
    with pytest.raises(ValueError, match="num_workers_per_gpu=1"):
        WorkerPool([0], 2, 1, 1)


def test_gpu_lock_is_released_after_failure():
    run_lock_failure()
