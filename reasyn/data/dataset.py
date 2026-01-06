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
import pickle
import random
from typing import cast

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

from reasyn.chem.fpindex import FingerprintIndex
from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.chem.stack import Stack, create_stack_step_by_step
from reasyn.utils.train_utils import worker_init_fn

from .collate import (
    apply_collate,
    collate_padding_masks,
    collate_tokens,
    collate_tokens_editflow
)
from .common import create_data


class Collater:
    def __init__(
        self,
        model_type,
        max_num_atoms: int = 96,
        max_smiles_len: int = 256,
        max_num_tokens: int = 24
    ):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_smiles_len = max_smiles_len
        self.max_num_tokens = max_num_tokens

        self.spec_smiles = {"smiles": collate_tokens}
        if model_type == 'editflow':
            self.spec_tokens = {
                "tokens": collate_tokens_editflow,
                "rxn_mask": collate_padding_masks,
                "token_padding_mask": collate_padding_masks,
            }
        else:
            self.spec_tokens = {
                "tokens": collate_tokens,
                "smiles_mask_rev": collate_padding_masks,
                "token_padding_mask": collate_padding_masks,
            }

    def __call__(self, data_list):
        data_list_t = cast(list[dict[str, torch.Tensor]], data_list)
        batch = {
            **apply_collate(self.spec_smiles, data_list_t, max_size=self.max_smiles_len),
            **apply_collate(self.spec_tokens, data_list_t, max_size=self.max_num_tokens),
        }
        return batch


class ProjectionDataset(IterableDataset):
    def __init__(
        self,
        reaction_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        max_num_atoms: int = 80,
        max_smiles_len: int = 192,
        min_num_reactions: int = 1,
        max_num_reactions: int = 5,
        init_stack_weighted_ratio: float = 0.0,
        virtual_length: int = 65536,
    ) -> None:
        super().__init__()
        self._reaction_matrix = reaction_matrix
        self._max_num_atoms = max_num_atoms
        self._max_smiles_len = max_smiles_len
        self._min_num_reactions = min_num_reactions
        self._max_num_reactions = max_num_reactions
        self._fpindex = fpindex
        self._init_stack_weighted_ratio = init_stack_weighted_ratio
        self._virtual_length = virtual_length

    def __len__(self) -> int:
        return self._virtual_length

    def __iter__(self):
        while True:
            for stack in create_stack_step_by_step(
                self._reaction_matrix,
                min_num_reactions=self._min_num_reactions,
                max_num_reactions=self._max_num_reactions,
                max_num_atoms=self._max_num_atoms,
                init_stack_weighted_ratio=self._init_stack_weighted_ratio,
            ):
                mol_seq_full = stack.mols
                rxn_idx_seq_full = stack.get_rxn_idx_seq()
                product = random.choice(list(stack.get_top()))
                data = create_data(
                    product=product,
                    mol_seq=mol_seq_full,
                    rxn_idx_seq=rxn_idx_seq_full,
                )
                data["smiles"] = data["smiles"][: self._max_smiles_len]
                yield data
            

class ProjectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        batch_size: int,
        num_workers: int = 4,
        data_path = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_options = kwargs
        self.collater_options = {'model_type': config.model.model_type,
                                 'max_num_tokens': config.model.decoder.pe_max_len}

    def setup(self, stage: str | None = None) -> None:
        if not os.path.exists(self.config.chem.rxn_matrix):
            raise FileNotFoundError(
                f"Reaction matrix not found: {self.config.chem.rxn_matrix}. "
                "Please generate the reaction matrix before training."
            )
        if not os.path.exists(self.config.chem.fpindex):
            raise FileNotFoundError(
                f"Fingerprint index not found: {self.config.chem.fpindex}. "
                "Please generate the fingerprint index before training."
            )

        with open(self.config.chem.rxn_matrix, "rb") as f:
            rxn_matrix: ReactantReactionMatrix = pickle.load(f)

        with open(self.config.chem.fpindex, "rb") as f:
            fpindex = pickle.load(f)

        self.train_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            fpindex=fpindex,
            virtual_length=self.config.train.val_freq * self.batch_size,
            **self.dataset_options,
        )
        self.val_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            fpindex=fpindex,
            virtual_length=self.batch_size,
            **self.dataset_options,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=Collater(**self.collater_options),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=Collater(**self.collater_options),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )


class EditFlowCollater:
    def __init__(self, collate_keys, max_num_tokens: int = 24):
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.spec_tokens = {k: collate_tokens_editflow for k in collate_keys}
        
    def __call__(self, data_list):
        batch = {**apply_collate(self.spec_tokens, data_list, max_size=self.max_num_tokens)}
        return batch
    

class EditFlowDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        batch_size: int,
        num_workers: int = 4,
        train_shuffle: bool = True,
        data_path: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.data_path = data_path
        
    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = load_dataset(self.data_path,
                                          split='train', streaming=True).with_format('torch')
        if self.train_shuffle:
            self.train_dataset = self.train_dataset.shuffle(buffer_size=100_000)
        self.val_dataset = load_dataset(self.data_path,
                                        split='train', streaming=True).with_format('torch')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=EditFlowCollater(collate_keys=next(iter(self.train_dataset)).keys(),
                                        max_num_tokens=self.config.model.decoder.pe_max_len),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=EditFlowCollater(collate_keys=next(iter(self.train_dataset)).keys(),
                                        max_num_tokens=self.config.model.decoder.pe_max_len),
            persistent_workers=True,
        )
