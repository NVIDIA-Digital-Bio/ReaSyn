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

import torch
import numpy as np
import dataclasses
import time

from reasyn.chem.fpindex import FingerprintIndex
from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.chem.mol import Molecule
from reasyn.chem.reaction import Reaction
from reasyn.chem.stack import Stack


@dataclasses.dataclass
class _ReactantItem:
    reactant: Molecule
    index: int
    score: float

    def __iter__(self):
        return iter([self.reactant, self.index, self.score])


@dataclasses.dataclass
class _ReactionItem:
    reaction: Reaction
    index: int
    score: float

    def __iter__(self):
        return iter([self.reaction, self.index, self.score])


@dataclasses.dataclass
class PredictResult:
    sampled_type: str   # 'ABORTED', 'END', 'BB', 'RXN'
    sampled_item: None | _ReactantItem | _ReactionItem


@dataclasses.dataclass
class State:
    stack: Stack = dataclasses.field(default_factory=Stack)
    scores: list[float] = dataclasses.field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(self.scores)


@dataclasses.dataclass
class ProductInfo:
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
        

def get_reactants(
        mol: str | Molecule,
        fpindex: FingerprintIndex,
        topk=1,
        use_edit_distance: bool = False) -> list[list[_ReactantItem]]:
    if isinstance(mol, str):
        mol = Molecule(mol)
    
    if mol._rdmol is None:
        if not use_edit_distance:
            return None
        query_res = fpindex.query_cuda(q=mol.smiles, k=topk)[0]
    else:
        fp = torch.Tensor(mol.get_fingerprint(option=fpindex._fp_option))
        query_res = fpindex.query_cuda(q=fp[None, :], k=topk)[0]
    mols = np.array([q.molecule for q in query_res])
    mol_idxs = np.array([q.index for q in query_res])
    distances = np.array([q.distance for q in query_res])
    scores = 1.0 / (distances + 0.1)
    
    sorted_indices = (-scores).argsort()
    mols = mols[sorted_indices]
    mol_idxs = mol_idxs[sorted_indices]
    scores = scores[sorted_indices]
    return [_ReactantItem(reactant=mol, index=mol_idx, score=score)
            for mol, mol_idx, score in zip(mols, mol_idxs, scores)]


def get_reactions(reaction_logits: torch.Tensor, rxn_matrix: ReactantReactionMatrix, topk=None) -> list[list[_ReactionItem]]:
    if topk is None:
        topk = len(rxn_matrix.reactions)
    reaction_probs = reaction_logits.softmax(dim=-1)
    reaction_probs, sorted_indices = reaction_probs.topk(topk, dim=-1, largest=True)

    return [_ReactionItem(reaction=rxn_matrix.reactions[idx], index=idx, score=score)
            for idx, score in zip(sorted_indices, reaction_probs)]


# don't consider INT
def action_string_to_stack(pathway, rxn_matrix: ReactantReactionMatrix):
    mol_or_rxn_list = pathway.split(';')
    stack = Stack()
    for mol_or_rxn in mol_or_rxn_list:
        if mol_or_rxn.startswith('R'):  # RXN
            rxn_id = int(mol_or_rxn.lstrip('R'))
            rxn = rxn_matrix.reactions[rxn_id]
            stack.push_rxn(rxn, rxn_id)
        else:   # BB
            mol = Molecule(mol_or_rxn)
            stack.push_mol(mol, 0)
    return stack


def get_sub_stacks(stack: Stack, num_samples: int = 1, topdown: bool = False):
    num_samples = min(num_samples, len(stack._mols) - 1)
    sub_idxs = np.random.choice(range(1, len(stack._mols)), num_samples, replace=False)
    
    sub_stacks = []
    for idx in sub_idxs:
        if topdown:
            mols, mol_idxs = stack._mols[idx:], stack.get_mol_idx_seq()[idx:]
            rxns, rxn_idxs = stack._rxns[idx:], stack.get_rxn_idx_seq()[idx:]
        else:
            mols, mol_idxs = stack._mols[:idx], stack.get_mol_idx_seq()[:idx]
            rxns, rxn_idxs = stack._rxns[:idx], stack.get_rxn_idx_seq()[:idx]
        
        sub_stack = Stack()
        for mol, mol_idx, rxn, rxn_idx in zip(mols, mol_idxs, rxns, rxn_idxs):
            if rxn is not None:
                push_fn = sub_stack.push_topdown if topdown else sub_stack.push_rxn
                push_fn(rxn, rxn_idx)
            elif mol is not None:
                push_fn = sub_stack.push_topdown if topdown else sub_stack.push_mol
                push_fn(mol, mol_idx)
        sub_stacks.append(sub_stack)
    return sub_stacks
