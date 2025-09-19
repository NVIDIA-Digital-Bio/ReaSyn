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

import random
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
import reasyn.utils.mutate as mu


def cut(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[*]-;!@[*]")):
        return None

    bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[*]-;!@[*]")))  # single bond not in ring
    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])
    try:
        return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
    except ValueError:
        return None


def cut_ring(mol):
    for i in range(10):
        if random.random() < 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")))
            bis = (
                (bis[0], bis[1]),
                (bis[2], bis[3]),
            )
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")))
            bis = (
                (bis[0], bis[1]),
                (bis[1], bis[2]),
            )

        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]

        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)])

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            if len(fragments) == 2:
                return fragments
        except ValueError:
            return None


def ring_ok(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts("[r3,r4]=[r3,r4]"))

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


average_size = 39.15
size_stdev = 3.50
# MIN_SIZE, MAX_SIZE = 20, 40


def mol_ok(mol):
    try:
        Chem.SanitizeMol(mol)
        target_size = size_stdev * np.random.randn() + average_size  # parameters set in GA_mol
        if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
        # if MIN_SIZE < mol.GetNumAtoms() < MAX_SIZE:
            return True
        else:
            return False
    except ValueError:
        return False


def crossover_ring(parent_A, parent_B):
    ring_smarts = Chem.MolFromSmarts("[R]")
    if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = ["[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]", "[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]"]
    rxn_smarts2 = ["([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]", "([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]"]

    for i in range(10):
        fragments_A = cut_ring(parent_A)
        fragments_B = cut_ring(parent_B)

        if fragments_A is None or fragments_B is None:
            return None

        new_mol_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            new_mol_trial = []
            for fa in fragments_A:
                for fb in fragments_B:
                    new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])

        new_mols = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_ok(m):
                    new_mols += list(rxn2.RunReactants((m,)))

        new_mols2 = []
        for m in new_mols:
            m = m[0]
            if mol_ok(m) and ring_ok(m):
                new_mols2.append(m)

        if len(new_mols2) > 0:
            return random.choice(new_mols2)

    return None


def crossover_non_ring(parent_A, parent_B):
    for i in range(10):
        fragments_A = cut(parent_A)
        fragments_B = cut(parent_B)
        if fragments_A is None or fragments_B is None:
            return None
        rxn = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
        new_mol_trial = []
        for fa in fragments_A:
            for fb in fragments_B:
                new_mol_trial.append(rxn.RunReactants((fa, fb))[0])

        new_mols = []
        for mol in new_mol_trial:
            mol = mol[0]
            if mol_ok(mol):
                new_mols.append(mol)

        if len(new_mols) > 0:
            return random.choice(new_mols)

    return None


def crossover(parent_A, parent_B):
    parent_smiles = [Chem.MolToSmiles(parent_A), Chem.MolToSmiles(parent_B)]
    try:
        Chem.Kekulize(parent_A, clearAromaticFlags=True)
        Chem.Kekulize(parent_B, clearAromaticFlags=True)

    except ValueError:
        pass

    for i in range(10):
        if random.random() <= 0.5:
            new_mol = crossover_non_ring(parent_A, parent_B)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol
        else:
            new_mol = crossover_ring(parent_A, parent_B)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol

    return None


def make_mating_pool(population_mol, population_scores, offspring_size: int, softmax_temp=None):
    if softmax_temp is not None:
        _population_scores = np.array(population_scores) / softmax_temp
        population_probs = np.exp(_population_scores - np.max(_population_scores))
        population_probs /= population_probs.sum()
    else:
        population_scores = [s + 1e-10 for s in population_scores]
        sum_scores = sum(population_scores)
        population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    try:
        new_child = crossover(parent_a, parent_b)
        if new_child is not None:
            new_child = mu.mutate(new_child, mutation_rate)
        return new_child
    except KeyboardInterrupt:
        quit()
    except:
        return parent_a
    