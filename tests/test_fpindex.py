# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reasyn.chem.fpindex import FingerprintIndex
from reasyn.chem.mol import FingerprintOption, Molecule
from reasyn.utils.sample_utils import get_reactants


def _index(fingerprints=None, smiles=None):
    if fingerprints is None:
        smiles = smiles or ["C", "CC", "CCC", "COC", "N#N", "C1CC1"]
        fingerprints = np.zeros((len(smiles), 8), dtype=np.uint8)
    elif smiles is None:
        smiles = [str(index) for index in range(len(fingerprints))]
    index = object.__new__(FingerprintIndex)
    index._molecules = tuple(smiles)
    index._smiles = smiles
    index._fp = np.asarray(fingerprints, dtype=np.uint8)
    index._fp_option = SimpleNamespace(dim=index._fp.shape[1])
    return index


def _pairs(results):
    return [
        [(result.index, float(result.distance)) for result in row]
        for row in results
    ]


@pytest.fixture(autouse=True)
def _clear_fp_cache():
    FingerprintIndex._fp_cuda.cache_clear()
    yield
    FingerprintIndex._fp_cuda.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_query_matches_legacy_cpu_topk_with_ties(monkeypatch):
    index = _index(np.zeros((100, 8), dtype=np.uint8))
    queries = torch.stack((torch.zeros((3, 8)), torch.ones((3, 8))))

    monkeypatch.setenv("REASYN_GPU_QUERY", "0")
    expected = _pairs(index.query_cuda(queries, k=3))
    monkeypatch.setenv("REASYN_GPU_QUERY", "1")
    actual = _pairs(
        index.query_cuda(
            queries,
            k=3,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
    )

    assert actual == expected
    assert len(actual) == 2
    assert all(len(row) == 9 for row in actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_existing_cuda_query_keeps_its_device(monkeypatch):
    index = _index(np.zeros((4, 8), dtype=np.uint8))
    query = torch.zeros((1, 8), device="cuda")
    seen = []
    fp_cuda = index.fp_cuda

    def record_device(device):
        seen.append(torch.device(device))
        return fp_cuda(device)

    monkeypatch.setattr(index, "fp_cuda", record_device)
    monkeypatch.setenv("REASYN_GPU_QUERY", "1")
    index.query_cuda(query, k=1, device=torch.device("cpu"))

    assert seen == [query.device]


def test_query_uses_explicit_indexed_device(monkeypatch):
    index = _index(np.zeros((4, 8), dtype=np.uint8))
    query = torch.zeros((1, 8))
    moves = []
    fp_devices = []

    def fake_to(tensor, device):
        moves.append(torch.device(device))
        return tensor

    def fake_fp_cuda(device):
        fp_devices.append(torch.device(device))
        return torch.tensor(index._fp, dtype=torch.float)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    monkeypatch.setattr(index, "fp_cuda", fake_fp_cuda)
    monkeypatch.setenv("REASYN_GPU_QUERY", "1")
    index.query_cuda(query, k=1, device=torch.device("cuda:7"))

    assert moves == [torch.device("cuda:7")]
    assert fp_devices == [torch.device("cuda:7")]


def test_fp_cache_canonicalizes_bare_cuda(monkeypatch):
    index = _index()
    allocations = []
    sentinel = object()

    def fake_tensor(*args, **kwargs):
        allocations.append(kwargs["device"])
        return sentinel

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(torch, "tensor", fake_tensor)

    assert index.fp_cuda(torch.device("cuda")) is sentinel
    assert index.fp_cuda(torch.device("cuda:7")) is sentinel
    assert allocations == [torch.device("cuda:7")]
    assert FingerprintIndex._fp_cuda.cache_info().hits == 1


def test_cpu_only_host_ignores_requested_cuda_device(monkeypatch):
    index = _index(np.zeros((4, 8), dtype=np.uint8))
    query = torch.ones((1, 8))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("REASYN_GPU_QUERY", "1")

    actual = _pairs(index.query_cuda(query, k=2, device=torch.device("cuda:7")))
    monkeypatch.setenv("REASYN_GPU_QUERY", "0")
    expected = _pairs(index.query_cuda(query, k=2))

    assert actual == expected


def test_get_reactants_forwards_model_device():
    class SpyIndex:
        _fp_option = FingerprintOption(morgan_n_bits=8)

        def query_cuda(self, q, k, device=None):
            self.device = device
            result = SimpleNamespace(molecule=Molecule("C"), index=0, distance=0.0)
            return [[result]]

    index = SpyIndex()
    device = torch.device("cuda:7")
    get_reactants("C", fpindex=index, device=device)

    assert index.device == device


def test_rapidfuzz_matches_editdistance_and_uses_one_worker(monkeypatch):
    from rapidfuzz import process as rf_process

    index = _index()
    real_cdist = rf_process.cdist
    workers = []

    def record_workers(*args, **kwargs):
        workers.append(kwargs["workers"])
        return real_cdist(*args, **kwargs)

    monkeypatch.setattr(rf_process, "cdist", record_workers)
    for query in ("", "not-a-smiles(", "C", "😀"):
        monkeypatch.setenv("REASYN_RAPIDFUZZ", "0")
        expected = _pairs(index.query_cuda(query, k=len(index._smiles)))
        monkeypatch.setenv("REASYN_RAPIDFUZZ", "1")
        actual = _pairs(index.query_cuda(query, k=len(index._smiles)))
        assert actual == expected

    assert workers == [1, 1, 1, 1]


def test_missing_rapidfuzz_falls_back_to_editdistance(monkeypatch):
    index = _index()
    query = "not-a-smiles("
    monkeypatch.setenv("REASYN_RAPIDFUZZ", "0")
    expected = _pairs(index.query_cuda(query, k=len(index._smiles)))
    real_import = builtins.__import__
    attempts = []

    def without_rapidfuzz(name, *args, **kwargs):
        if name == "rapidfuzz" or name.startswith("rapidfuzz."):
            attempts.append(name)
            raise ImportError("simulated missing RapidFuzz")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_rapidfuzz)
    monkeypatch.setenv("REASYN_RAPIDFUZZ", "1")
    actual = _pairs(index.query_cuda(query, k=len(index._smiles)))

    assert actual == expected
    assert attempts


def test_real_index_gpu_query_matches_legacy_cpu_topk(monkeypatch):
    path = os.environ.get("REASYN_REAL_FPINDEX")
    if not path:
        pytest.skip("set REASYN_REAL_FPINDEX to exercise the real index")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    with Path(path).open("rb") as handle:
        index = pickle.load(handle)
    query_ids = [0, len(index._fp) // 4, len(index._fp) // 2, len(index._fp) - 1]
    queries = torch.from_numpy(index._fp[query_ids].astype(np.float32, copy=True))

    monkeypatch.setenv("REASYN_GPU_QUERY", "0")
    expected = _pairs(index.query_cuda(queries, k=100))
    monkeypatch.setenv("REASYN_GPU_QUERY", "1")
    actual = _pairs(
        index.query_cuda(
            queries,
            k=100,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
    )

    assert actual == expected
