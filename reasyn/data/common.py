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

from collections.abc import Sequence
import random
import torch
from reasyn.chem.mol import Molecule
from reasyn.chem.stack import Stack
from reasyn.chem.featurize import TokenType


def featurize_stack_actions(
    mol_seq,
    rxn_idx_seq,
    end_token: bool,
    sampling_direction='bu',
    sampling=False,
) -> dict[str, torch.Tensor]:
    # smiles_mask_rev:  (True) START & END & PAD & RXN, (False) MOL
    # rxn_mask:         (True) RXN, (False) START & END & PAD & MOL

    # remove product molecule (the last one in mol_seq) during training
    if not sampling:
        # mol_seq is list or tuple
        mol_seq = mol_seq[:-1] + type(mol_seq)([None])
    
    # BU + TD
    if (not sampling and random.random() > 0.5) or \
        (sampling and sampling_direction == 'td'):
        # During training: randomly choose bottom up / top down
        # During sampling: designate sampling direction
        mol_seq, rxn_idx_seq = mol_seq[::-1], rxn_idx_seq[::-1]
    
    seq = [TokenType.START]
    smiles_mask_rev = [1] * len(seq)
    rxn_mask = [0] * len(seq)
    
    for mol, rxn_idx in zip(mol_seq, rxn_idx_seq):
        if rxn_idx is not None: # RXN
            seq.append(rxn_idx + TokenType.RXN_MIN) # 0~114 -> 157~271
            smiles_mask_rev.append(1)
            rxn_mask.append(1)
    
        elif mol is not None:   # BB
            smiles = mol.tokenize_csmiles()
            seq.append(TokenType.MOL_START)
            seq.extend(smiles + TokenType.MOL_END)  # 1~153 -> 4~156
            seq.append(TokenType.MOL_END)
            smiles_mask_rev.extend([0] * (2 + len(smiles)))
            rxn_mask.extend([0] * (2 + len(smiles)))

    if end_token:
        seq.append(0)
        smiles_mask_rev.append(1)
        rxn_mask.append(0)
    
    seq = torch.tensor(seq, dtype=torch.long)
    smiles_mask_rev = torch.tensor(smiles_mask_rev, dtype=torch.bool)
    rxn_mask = torch.tensor(rxn_mask, dtype=torch.bool)
    token_padding_mask = torch.zeros_like(seq).to(torch.bool)
    
    feats = {
        'tokens': seq,
        'smiles_mask_rev': smiles_mask_rev,
        'rxn_mask': rxn_mask,
        'token_padding_mask': token_padding_mask
    }
    return feats


def featurize_stack(
    stack: Stack,
    end_token: bool,
    sampling_direction: str = 'bu',
) -> dict[str, torch.Tensor]:
    return featurize_stack_actions(
        mol_seq=stack._mols,
        rxn_idx_seq=stack.get_rxn_idx_seq(),
        end_token=end_token,
        sampling_direction=sampling_direction,
        sampling=True,
    )


def create_data(
    product: Molecule,
    mol_seq: Sequence[Molecule],
    rxn_idx_seq: Sequence[int | None],
):
    stack_feats = featurize_stack_actions(
        rxn_idx_seq=rxn_idx_seq,
        mol_seq=mol_seq,
        end_token=True,
    )
    data = {
        "smiles": product.tokenize_csmiles(),
        "tokens": stack_feats["tokens"],
        "rxn_mask": stack_feats["rxn_mask"],
        "smiles_mask_rev": stack_feats["smiles_mask_rev"],
        "token_padding_mask": stack_feats["token_padding_mask"]
    }
    return data
