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

import os
import pathlib
import pickle
import tempfile
from collections.abc import Iterable
from functools import cached_property

import joblib
import numpy as np
from tqdm.auto import tqdm

from .mol import Molecule, read_mol_file
from .reaction import Reaction, ReactionContainer, read_reaction_file


def _fill_matrix(matrix: np.memmap, offset: int, reactants: Iterable[Molecule], reactions: Iterable[Reaction]):
    for i, reactant in enumerate(reactants):
        for j, reaction in enumerate(reactions):
            flag = 0
            for t in reaction.match_reactant_templates(reactant):
                flag |= 1 << t
            matrix[offset + i, j] = flag


class ReactantReactionMatrix:
    def __init__(
        self,
        reactants: Iterable[Molecule],
        reactions: Iterable[Reaction],
        matrix: np.ndarray | os.PathLike | None = None,
    ) -> None:
        super().__init__()
        self._reactants = tuple(reactants)
        self._reactions = tuple(reactions)
        self._matrix = self._init_matrix(matrix)

    def _init_matrix(self, matrix: np.ndarray | os.PathLike | None, batch_size: int = 1024) -> np.ndarray:
        if isinstance(matrix, np.ndarray):
            return matrix
        elif isinstance(matrix, (os.PathLike, str)):
            return np.load(matrix)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tempdir_s:
            temp_fname = pathlib.Path(tempdir_s) / "matrix"
            matrix = np.memmap(
                str(temp_fname),
                dtype=np.uint8,
                mode="w+",
                shape=(len(self._reactants), len(self._reactions)),
            )
            joblib.Parallel(n_jobs=joblib.cpu_count() // 2)(
                joblib.delayed(_fill_matrix)(
                    matrix=matrix,
                    offset=start,
                    reactants=self._reactants[start : start + batch_size],
                    reactions=self._reactions,
                )
                for start in tqdm(range(0, len(self._reactants), batch_size), desc="Create matrix")
            )
            return np.array(matrix)

    @property
    def reactants(self) -> tuple[Molecule, ...]:
        return self._reactants

    @cached_property
    def reactions(self) -> ReactionContainer:
        return ReactionContainer(self._reactions)

    @cached_property
    def seed_reaction_indices(self) -> list[int]:
        full_flag = np.array([0b01 if rxn.num_reactants == 1 else 0b11 for rxn in self._reactions], dtype=np.uint8)
        return np.nonzero(full_flag == np.bitwise_or.reduce(self._matrix, axis=0))[0].tolist()

    @cached_property
    def reactant_count(self) -> np.ndarray:
        return (self._matrix != 0).astype(np.int32).sum(0)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


def create_reactant_reaction_matrix_cache(
    reactant_path: pathlib.Path,
    reaction_path: pathlib.Path,
    cache_path: pathlib.Path,
    excl_path: pathlib.Path | None = None,
):
    rxns = ReactionContainer(read_reaction_file(reaction_path))
    mols = list(read_mol_file(reactant_path))
    if excl_path is not None:
        excl_smiles = {m.smiles for m in read_mol_file(excl_path)}
        mols = [m for m in mols if m.smiles not in excl_smiles]
    m = ReactantReactionMatrix(mols, rxns)
    with open(cache_path, "wb") as f:
        pickle.dump(m, f)
    return m
