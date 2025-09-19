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
import itertools
import random
from collections.abc import Iterable
from typing import TypeAlias

import numpy as np

from .matrix import ReactantReactionMatrix
from .mol import Molecule
from .reaction import Reaction


_NumReactants: TypeAlias = int
_MolOrRxnIndex: TypeAlias = int
_TokenType: TypeAlias = tuple[_NumReactants, _MolOrRxnIndex]


def _flatten(l):
    for el in l:
        if isinstance(el, list):
            yield from _flatten(el)
        else:
            yield el


@dataclasses.dataclass
class _Node:
    mol: Molecule
    rxn: Reaction | None
    token: _TokenType
    children: list["_Node"]

    def to_str(self, depth: int) -> str:
        pad = " " * depth * 2
        lines = [f"{pad}{self.mol.smiles}"]
        if self.rxn is not None:
            for c in self.children:
                lines.append(f"{c.to_str(depth + 1)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Node(\n{self.to_str(1)}\n)"


class Stack:
    def __init__(self) -> None:
        super().__init__()
        self._mols: list[Molecule] = []
        self._rxns: list[Reaction | None] = []
        self._tokens: list[_TokenType] = []
        self._stack: list[set[Molecule]] = []
        
    @property
    def mols(self) -> tuple[Molecule, ...]:
        return tuple(self._mols)

    @property
    def rxns(self) -> tuple[Reaction | None, ...]:
        return tuple(self._rxns)

    @property
    def tokens(self) -> tuple[_TokenType, ...]:
        return tuple(self._tokens)

    def get_top(self) -> set[Molecule]:
        return self._stack[-1] if self._stack else set()
        
    def get_one_top(self) -> Molecule:
        return next(iter(self.get_top()))

    def get_second_top(self) -> set[Molecule]:
        return self._stack[-2]

    def get_third_top(self) -> set[Molecule]:
        return self._stack[-3]

    def push_mol(self, mol: Molecule, index: int) -> None:
        self._mols.append(mol)
        self._rxns.append(None)
        self._tokens.append((0, index))
        self._stack.append({mol})
        
    def push_rxn(self, rxn: Reaction, index: int, product_limit: int | None = None) -> bool:
        if len(self._stack) < rxn.num_reactants:
            return False

        prods: list[Molecule] = []
        if rxn.num_reactants == 1:
            for r in self.get_top():
                prods += rxn([r])
        elif rxn.num_reactants == 2:
            for r1, r2 in itertools.product(self.get_top(), self.get_second_top()):
                if product_limit is not None and len(prods) >= product_limit:
                    break
                prods += rxn([r1, r2]) + rxn([r2, r1])
        elif rxn.num_reactants == 3:
            for r1, r2, r3 in itertools.product(self.get_top(), self.get_second_top(), self.get_third_top()):
                if product_limit is not None and len(prods) >= product_limit:
                    break
                prods += (
                    rxn([r1, r2, r3])
                    + rxn([r1, r3, r2])
                    + rxn([r2, r1, r3])
                    + rxn([r2, r3, r1])
                    + rxn([r3, r2, r1])
                    + rxn([r3, r1, r2])
                )
        else:
            return False

        if len(prods) == 0:
            return False
        if product_limit is not None:
            prods = prods[:product_limit]
        prod: Molecule = random.choice(prods)

        self._mols.append(prod)
        self._rxns.append(rxn)
        self._tokens.append((rxn.num_reactants, index))
        for _ in range(rxn.num_reactants):
            self._stack.pop()
        self._stack.append(set(prods))
        return True

    def __len__(self) -> int:
        return len(self._mols)

    def __getitem__(self, index: int) -> Molecule:
        return self._mols[index]

    def get_mol_idx_seq(self) -> list[int | None]:
        return [t[1] if t[0] == 0 else None for t in self.tokens]

    def get_rxn_idx_seq(self) -> list[int | None]:
        return [t[1] if t[0] > 0 else None for t in self.tokens]

    def count_reactions(self) -> int:
        cnt = 0
        for rxn in self._rxns:
            if rxn is not None:
                cnt += 1
        return cnt

    def get_action_string(self, delim: str = ";") -> str:
        tokens: list[str] = []
        for mol, (num_reactants, idx) in zip(self._mols, self._tokens):
            if num_reactants == 0:
                tokens.append(mol.smiles)
            else:
                tokens.append(f"R{idx}")
        return delim.join(tokens)

    def get_stack_depth(self) -> int:
        return len(self._stack)


def create_init_stack(matrix: ReactantReactionMatrix, weighted_ratio: float = 0.0) -> Stack:
    stack = Stack()

    if weighted_ratio != 0.0:
        prob_w = matrix.reactant_count[matrix.seed_reaction_indices].astype(np.float32)
        prob_w = prob_w / prob_w.sum()
        prob_u = np.ones_like(prob_w) / len(prob_w)
        prob = weighted_ratio * prob_w + (1 - weighted_ratio) * prob_u
        rxn_index: int = np.random.choice(matrix.seed_reaction_indices, p=prob)
    else:
        rxn_index = random.choice(matrix.seed_reaction_indices)
    rxn_col = matrix.matrix[:, rxn_index]
    rxn = matrix.reactions[rxn_index]

    if rxn.num_reactants == 2:
        m1 = np.random.choice(np.bitwise_and(rxn_col, 0b01).nonzero()[0])
        m2 = np.random.choice(np.bitwise_and(rxn_col, 0b10).nonzero()[0])
        if random.randint(0, 1) % 2 == 1:
            m1, m2 = m2, m1
        stack.push_mol(matrix.reactants[m1], m1)
        stack.push_mol(matrix.reactants[m2], m2)
    elif rxn.num_reactants == 1:
        m = np.random.choice(rxn_col.nonzero()[0])
        stack.push_mol(matrix.reactants[m], m)
    elif rxn.num_reactants == 3:
        m1 = np.random.choice(np.bitwise_and(rxn_col, 0b001).nonzero()[0])
        m2 = np.random.choice(np.bitwise_and(rxn_col, 0b010).nonzero()[0])
        m3 = np.random.choice(np.bitwise_and(rxn_col, 0b100).nonzero()[0])
        m1, m2, m3 = random.sample([m1, m2, m3], 3)
        stack.push_mol(matrix.reactants[m1], m1)
        stack.push_mol(matrix.reactants[m2], m2)
        stack.push_mol(matrix.reactants[m3], m3)

    stack.push_rxn(rxn, rxn_index)
    return stack


def expand_stack(stack: Stack, matrix: ReactantReactionMatrix):
    matches = matrix.reactions.match_reactions(random.choice(list(stack.get_top())))
    if len(matches) == 0:
        return stack, False
    rxn_index = random.choice(list(matches.keys()))
    reactant_flag = 1 << matches[rxn_index][0]

    rxn_col = matrix.matrix[:, rxn_index]
    if np.any(rxn_col >= 4):
        # Case of tri-mol reaction
        all_reactants = 0b111
        remaining_reactants = all_reactants ^ reactant_flag
        reactant_1 = remaining_reactants & 0b001  # Isolate the 001 bit
        reactant_2 = remaining_reactants & 0b010  # Isolate the 010 bit
        reactant_3 = remaining_reactants & 0b100  # Isolate the 100 bit
        valid_reactants = [reactant for reactant in [reactant_1, reactant_2, reactant_3] if reactant != 0]
        s_indices_1 = np.logical_and(rxn_col != 0, (rxn_col & valid_reactants[0]) == valid_reactants[0]).nonzero()[0]
        s_indices_2 = np.logical_and(rxn_col != 0, (rxn_col & valid_reactants[1]) == valid_reactants[1]).nonzero()[0]
        s_indices_1, s_indices_2 = random.sample([s_indices_1, s_indices_2], 2)
        s_index1 = np.random.choice(s_indices_1)
        stack.push_mol(matrix.reactants[s_index1], s_index1)
        s_index2 = np.random.choice(s_indices_2)
        stack.push_mol(matrix.reactants[s_index2], s_index2)
        rxn_success = stack.push_rxn(matrix.reactions[rxn_index], rxn_index)
    else:
        # case of uni- and bi-mol reaction
        s_indices = np.logical_and(rxn_col != 0, rxn_col != reactant_flag).nonzero()[0]
        
        # Case of uni-mol reaction
        if len(s_indices) == 0:
            return stack, False

        # Case of bi-mol reaction
        s_index = np.random.choice(s_indices)
        stack.push_mol(matrix.reactants[s_index], s_index)
        # NOTE: when using the comprehensive reaction template sets, the new reactions are not guaranteed suceed.
        #       This is a bug and should be fixed.
        #       To avoid wrong results temporarily, we check the reaction outcome here.
        rxn_success = stack.push_rxn(matrix.reactions[rxn_index], rxn_index)
    return stack, rxn_success


def create_stack(
    matrix: ReactantReactionMatrix,
    max_num_reactions: int = 5,
    max_num_atoms: int = 80,
    init_stack_weighted_ratio: float = 0.0,
) -> Stack:
    stack = create_init_stack(matrix, weighted_ratio=init_stack_weighted_ratio)
    for _ in range(1, max_num_reactions):
        stack, changed = expand_stack(stack, matrix)
        if not changed:
            break
        if max(map(lambda m: m.num_atoms, stack.get_top())) > max_num_atoms:
            break
    return stack


def create_stack_step_by_step(
    matrix: ReactantReactionMatrix,
    min_num_reactions: int = 1,
    max_num_reactions: int = 5,
    max_num_atoms: int = 80,
    init_stack_weighted_ratio: float = 0.0,
) -> Iterable[Stack]:
    stack = create_init_stack(matrix, weighted_ratio=init_stack_weighted_ratio)
    if min_num_reactions == 1:
        yield stack
    for i in range(1, max_num_reactions):
        stack, changed = expand_stack(stack, matrix)
        if changed:
            if i >= min_num_reactions - 1:
                yield stack
        else:
            break
        if max(map(lambda m: m.num_atoms, stack.get_top())) > max_num_atoms:
            break
