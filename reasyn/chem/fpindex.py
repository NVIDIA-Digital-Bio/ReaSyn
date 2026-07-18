# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import functools
import os
import pathlib
import pickle
import tempfile
import editdistance
from collections.abc import Iterable, Sequence

import joblib
import numpy as np
import torch
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm

from .mol import FingerprintOption, Molecule, read_mol_file


@dataclasses.dataclass
class _QueryResult:
    index: int
    molecule: Molecule
    fingerprint: np.ndarray
    distance: float


def _fill_fingerprint(
    fp: np.memmap,
    offset: int,
    molecules: Iterable[Molecule],
    fp_option: FingerprintOption,
):
    os.sched_setaffinity(0, range(os.cpu_count() or 1))
    for i, mol in enumerate(molecules):
        fp[offset + i] = mol.get_fingerprint(fp_option).astype(np.uint8)


def compute_fingerprints(
    molecules: Sequence[Molecule],
    fp_option: FingerprintOption,
    batch_size: int = 1024,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tempdir_s:
        temp_fname = pathlib.Path(tempdir_s) / "fingerprint"
        fp = np.memmap(
            str(temp_fname),
            dtype=np.uint8,
            mode="w+",
            shape=(len(molecules), fp_option.dim),
        )
        joblib.Parallel(n_jobs=joblib.cpu_count() // 2)(
            joblib.delayed(_fill_fingerprint)(
                fp=fp,
                offset=start,
                molecules=molecules[start : start + batch_size],
                fp_option=fp_option,
            )
            for start in tqdm(range(0, len(molecules), batch_size), desc="Fingerprint")
        )
        return np.array(fp)


class FingerprintIndex:
    def __init__(self, molecules: Iterable[Molecule], fp_option: FingerprintOption) -> None:
        super().__init__()
        self._molecules = tuple(molecules)
        self._smiles = [m.csmiles for m in molecules]
        self._fp_option = fp_option
        self._fp = self._init_fingerprint()
        self._tree = self._init_tree()

    @property
    def molecules(self) -> tuple[Molecule, ...]:
        return self._molecules

    @property
    def fp_option(self) -> FingerprintOption:
        return self._fp_option

    def _init_fingerprint(self, batch_size: int = 1024) -> np.ndarray:
        return compute_fingerprints(
            molecules=self._molecules,
            fp_option=self._fp_option,
            batch_size=batch_size,
        )

    def _init_tree(self) -> BallTree:
        tree = BallTree(self._fp, metric="manhattan")
        return tree

    def __getitem__(self, index: int) -> tuple[Molecule, np.ndarray]:
        return self._molecules[index], self._fp[index]

    def query(self, q: np.ndarray, k: int) -> list[list[_QueryResult]]:
        """
        Args:
            q: shape (bsz, ..., fp_dim)
        """
        bsz = q.shape[0]
        dist, idx = self._tree.query(q.reshape([-1, self._fp_option.dim]), k=k)
        dist = dist.reshape([bsz, -1])
        idx = idx.reshape([bsz, -1])
        results: list[list[_QueryResult]] = []
        for i in range(dist.shape[0]):
            res: list[_QueryResult] = []
            for j in range(dist.shape[1]):
                index = int(idx[i, j])
                res.append(
                    _QueryResult(
                        index=index,
                        molecule=self._molecules[index],
                        fingerprint=self._fp[index],
                        distance=dist[i, j],
                    )
                )
            results.append(res)
        return results

    def fp_cuda(self, device: torch.device) -> torch.Tensor:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return self._fp_cuda(device)

    @functools.cache
    def _fp_cuda(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self._fp, dtype=torch.float, device=device)

    @torch.no_grad()
    def query_cuda(
        self,
        q: torch.Tensor | str,
        k: int,
        device: torch.device | None = None,
    ) -> list[list[_QueryResult]]:
        if isinstance(q, str):  # if mol is invalid, use edit distance instead
            bsz = 1
            # RapidFuzz computes identical Levenshtein distances in one C++/SIMD
            # call. Falls back to editdistance if rapidfuzz is absent.
            # REASYN_RAPIDFUZZ=0 restores the original loop for A/B measurement.
            if os.environ.get('REASYN_RAPIDFUZZ', '1') != '0':
                try:
                    from rapidfuzz import process as _rf_process
                    from rapidfuzz.distance import Levenshtein as _rf_lev
                    dists = _rf_process.cdist([q], self._smiles,
                                              scorer=_rf_lev.distance, workers=1)[0]
                    pwdist = torch.Tensor(dists.astype('float32'))
                except ImportError:
                    pwdist = torch.Tensor([editdistance.eval(q, s) for s in self._smiles])
            else:
                pwdist = torch.Tensor([editdistance.eval(q, s) for s in self._smiles])
        else:
            bsz = q.size(0)
            q = q.reshape([-1, self._fp_option.dim])
            # Use the caller's model device; an existing CUDA query takes precedence.
            # REASYN_GPU_QUERY=0 restores the original path for A/B measurement.
            if os.environ.get('REASYN_GPU_QUERY', '1') != '0' and torch.cuda.is_available():
                if q.is_cuda:
                    dev = q.device
                elif device is not None:
                    dev = device
                else:
                    dev = torch.device("cuda", torch.cuda.current_device())
                dev = torch.device(dev)
                if dev.type == "cuda" and dev.index is None:
                    dev = torch.device("cuda", torch.cuda.current_device())
                q = q.to(dev)
            else:
                dev = q.device
            pwdist = torch.cdist(self.fp_cuda(dev), q, p=1)  # (n_mols, n_queries)
        # Keep legacy CPU topk tie-breaking while calculating distances on the GPU.
        dist_t, idx_t = torch.topk(pwdist.cpu(), k=k, dim=0, largest=False)  # (k, n_queries)
        dist = dist_t.t().reshape([bsz, -1]).numpy()
        idx = idx_t.t().reshape([bsz, -1]).numpy()

        results: list[list[_QueryResult]] = []
        for i in range(dist.shape[0]):
            res: list[_QueryResult] = []
            for j in range(dist.shape[1]):
                index = int(idx[i, j])
                res.append(
                    _QueryResult(
                        index=index,
                        molecule=self._molecules[index],
                        fingerprint=self._fp[index],
                        distance=dist[i, j],
                    )
                )
            results.append(res)
        return results


def create_fingerprint_index_cache(
    molecule_path: pathlib.Path,
    cache_path: pathlib.Path,
    fp_option: FingerprintOption,
):
    mols = list(read_mol_file(molecule_path))
    fpindex = FingerprintIndex(mols, fp_option=fp_option)
    with open(cache_path, "wb") as f:
        pickle.dump(fpindex, f)
    return fpindex
