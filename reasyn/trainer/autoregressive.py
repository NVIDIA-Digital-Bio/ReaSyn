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

import pickle
from typing import Any
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.nn import functional as F

from reasyn.chem.fpindex import FingerprintIndex
from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.utils.train import get_optimizer, get_scheduler, sum_weighted_losses
from reasyn.models.reasyn import ReaSyn


class Wrapper(pl.LightningModule):
    def __init__(self, config, args: dict | None = None):
        super().__init__()
        self.save_hyperparameters(
            {
                "config": OmegaConf.to_container(config),
                "args": args or {},
            }
        )
        self.model = ReaSyn(config.model)
        
    @property
    def config(self):
        return OmegaConf.create(self.hparams["config"])

    @property
    def args(self):
        return OmegaConf.create(self.hparams.get("args", {}))

    def setup(self, stage: str) -> None:
        super().setup(stage)

        # Load chem data
        with open(self.config.chem.rxn_matrix, "rb") as f:
            self.rxn_matrix: ReactantReactionMatrix = pickle.load(f)

        with open(self.config.chem.fpindex, "rb") as f:
            self.fpindex: FingerprintIndex = pickle.load(f)
        
    def configure_optimizers(self):
        optimizer = get_optimizer(self.config.train.optimizer, self.model)
        if "scheduler" in self.config.train:
            scheduler = get_scheduler(self.config.train.scheduler, optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/loss",
            }
        return optimizer

    def get_loss(self, batch):
        logits = self.model(batch)[:, :-1].contiguous()
        target = batch["tokens"][:, 1:].contiguous()
        # smiles_mask_rev is False for MOL tokens
        smiles_mask_rev = batch["smiles_mask_rev"][:, 1:].contiguous()
        # token_padding_mask is True for PAD tokens
        pred_mask = ~(batch["token_padding_mask"][:, 1:].contiguous())
        
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        _loss = F.cross_entropy(logits_flat, target_flat, reduction="none")
        
        smiles_mask = (~smiles_mask_rev) * pred_mask
        smiles_mask_flat = smiles_mask.view(-1)
        total_smiles = smiles_mask.sum().to(logits_flat) + 1e-6
        loss_smiles = (_loss * smiles_mask_flat).sum() / total_smiles
        
        smiles_mask_rev = smiles_mask_rev * pred_mask
        smiles_mask_rev_flat = smiles_mask_rev.view(-1)
        total_other = smiles_mask_rev.sum().to(logits_flat) + 1e-6
        loss_other = (_loss * smiles_mask_rev_flat).sum() / total_other
        
        loss_dict = {'smiles': loss_smiles, 'other': loss_other}
        loss = sum_weighted_losses(loss_dict, self.config.train.loss_weights)
        return loss, loss_dict

    def training_step(self, batch, batch_idx: int):
        loss, loss_dict = self.get_loss(batch)
        self.log_dict({f"train/loss_{k}": v for k, v in loss_dict.items()}, on_step=True, logger=True)
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> Any:
        loss, loss_dict = self.get_loss(batch)
        self.log_dict({f"val/loss_{k}": v for k, v in loss_dict.items()}, on_step=False, logger=True, sync_dist=True)
        self.log("val/loss", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        return loss
