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

import copy
import dataclasses
import time
from collections.abc import Iterable
from functools import cached_property
from multiprocessing.synchronize import Lock

import pandas as pd
import torch
from tdc import Oracle

from reasyn.chem.fpindex import FingerprintIndex
from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.chem.mol import FingerprintOption, Molecule
from reasyn.chem.stack import Stack
from reasyn.data.common import featurize_stack
from reasyn.models.reasyn import ReaSyn


@dataclasses.dataclass
class State:
    stack: Stack = dataclasses.field(default_factory=Stack)
    scores: list[float] = dataclasses.field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(self.scores)


@dataclasses.dataclass
class _ProductInfo:
    molecule: Molecule
    stack: Stack


class TimeLimit:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._start = time.time()

    def exceeded(self) -> bool:
        if self._seconds <= 0:
            return False
        return time.time() - self._start > self._seconds

    def check(self):
        if self.exceeded():
            raise TimeoutError()


class StatePool:
    def __init__(
        self,
        fpindex: FingerprintIndex,
        rxn_matrix: ReactantReactionMatrix,
        mol: Molecule,
        model: ReaSyn,
        factor: int = 16,
        max_active_states: int = 256,
        sort_by_score: bool = True,
        mols_to_filter=None,
        filter_sim: float = 0.8
    ) -> None:
        super().__init__()
        self._fpindex = fpindex
        self._rxn_matrix = rxn_matrix

        self._model = model
        self.device = next(iter(model.parameters())).device
        
        self._mol = mol
        smiles = mol.tokenize_csmiles()
        self._smiles = smiles[None].to(self.device)
        
        self._factor = factor
        self._max_active_states = max_active_states
        self._sort_by_score = sort_by_score
        self._mols_to_filter = mols_to_filter
        self._filter_sim = filter_sim

        # input filtering
        if self._mols_to_filter is not None and \
            max([self._mol.sim(f) for f in self._mols_to_filter] or [-1]) > self._filter_sim:
            print(f'Input molecule {self._mol.csmiles} filtered')
            quit()

        self._active: list[State] = [State()]
        self._finished: list[State] = []
        self._aborted: list[State] = []

    @cached_property
    def code(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            return self._model.encoder({"smiles": self._smiles})
            
    def _sort_states(self) -> None:
        if self._sort_by_score:
            self._active.sort(key=lambda s: s.score, reverse=True)
        self._active = self._active[: self._max_active_states]

    def _predict_reward(self, reward_model, reactants):
        if isinstance(reactants, Molecule):
            reactants = [reactants]
        if isinstance(reward_model, Oracle):    # JNK3
            scores = reward_model([r.csmiles for r in reactants])
        else:                                   # PMO regressor
            scores = reward_model.predict([r._rdmol for r in reactants])
        return max(scores)

    def evolve(
        self,
        gpu_lock: Lock | None = None,
        time_limit: TimeLimit | None = None,
        reward_model=None,
    ) -> None:
        if reward_model and reward_model == 'jnk3':   # JNK3 hit expansion
            reward_model = Oracle('jnk3')

        if len(self._active) == 0:
            return
        
        _active = copy.deepcopy(self._active)
        feat_list = [
            featurize_stack(
                state.stack,
                end_token=False,
            )
            for state in _active
        ]
        
        if gpu_lock is not None:
            gpu_lock.acquire()

        code, code_padding_mask = self.code
        
        next: list[State] = []
        for feat, base_state in zip(feat_list, self._active):
            if time_limit is not None and time_limit.exceeded():
                break
            
            tokens = feat['tokens'].to(self.device)
            result = self._model.predict(
                code=code,
                code_padding_mask=code_padding_mask,
                tokens=tokens,
                fpindex=self._fpindex,
                rxn_matrix=self._rxn_matrix,
                topk=self._factor,
            )
            sampled_type, sampled_item = result.sampled_type, result.sampled_item
            
            for i in range(self._factor):
                if sampled_type == 'END':
                    self._finished.append(base_state)

                elif sampled_type == 'BB':
                    reactant, mol_idx, score = sampled_item[i]
                    new_state = copy.deepcopy(base_state)
                    new_state.stack.push_mol(reactant, mol_idx)
                    if reward_model is not None:
                        score = self._predict_reward(reward_model, reactant)
                    new_state.scores.append(score)
                    next.append(new_state)

                elif sampled_type == 'RXN':
                    if i >= len(sampled_item):
                        self._aborted.append(new_state)
                        continue
                    new_state = copy.deepcopy(base_state)
                    for j in range(i, len(sampled_item)):  # try until success
                        reaction, rxn_idx, score = result.sampled_item[j]
                        success = new_state.stack.push_rxn(reaction, rxn_idx, product_limit=None)
                        if success:
                            # intermediate filtering
                            filtered = False
                            if self._mols_to_filter is not None:
                                for m in new_state.stack.get_top():
                                    if max([m.sim(f) for f in self._mols_to_filter]) > self._filter_sim:
                                        filtered = True
                                        print(f'Product molecule {m.csmiles} filtered')
                                        break
                            if not filtered:
                                self._finished.append(new_state)
                                if reward_model is not None:
                                    score = self._predict_reward(reward_model, new_state.stack.get_top())
                                new_state.scores.append(score)
                                next.append(new_state)
                                break
                    else:
                        j += 1
                        self._aborted.append(new_state)
                    sampled_item = sampled_item[:i] + sampled_item[j:]  # remove failed RXNs

                else:
                    self._aborted.append(base_state)

        del self._active
        self._active = next
        self._sort_states()

        if gpu_lock is not None:
            gpu_lock.release()

    def get_products(self) -> Iterable[_ProductInfo]:
        visited: set[Molecule] = set()
        for state in self._finished:
            for mol in state.stack.get_top():
                if mol in visited:
                    continue
                yield _ProductInfo(mol, state.stack)
                visited.add(mol)
        yield from []
    
    def get_dataframe(self, num_calc_extra_metrics: int = 10) -> pd.DataFrame:
        rows: list[dict[str, str | float]] = []
        smiles_to_mol: dict[str, Molecule] = {}
        for product in self.get_products():
            rows.append(
                {
                    "target": self._mol.smiles,
                    "smiles": product.molecule.smiles,
                    "score": self._mol.sim(product.molecule, FingerprintOption.morgan_for_tanimoto_similarity()),
                    "synthesis": product.stack.get_action_string(),
                    "num_steps": product.stack.count_reactions(),
                }
            )
            smiles_to_mol[product.molecule.smiles] = product.molecule
        rows.sort(key=lambda r: r["score"], reverse=True)
        for row in rows[:num_calc_extra_metrics]:
            mol = smiles_to_mol[str(row["smiles"])]
            row["scf_sim"] = self._mol.scaffold.tanimoto_similarity(
                mol.scaffold,
                fp_option=FingerprintOption.morgan_for_tanimoto_similarity(),
            )
            row["pharm2d_sim"] = self._mol.dice_similarity(mol, fp_option=FingerprintOption.gobbi_pharm2d())
            row["rdkit_sim"] = self._mol.tanimoto_similarity(mol, fp_option=FingerprintOption.rdkit())

        df = pd.DataFrame(rows)
        return df
