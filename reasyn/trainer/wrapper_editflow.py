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
import pickle
from typing import Any
import pytorch_lightning as pl
from omegaconf import OmegaConf

from reasyn.chem.fpindex import FingerprintIndex
from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.utils.train_utils import get_optimizer, get_scheduler
from reasyn.utils.editflow_utils import make_batch, fill_gap_tokens_with_repeats
from reasyn.models.reasyn import ReaSyn


class Wrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters({"config": OmegaConf.to_container(config)})
        self.model = ReaSyn(config.model)
        
    @property
    def config(self):
        return OmegaConf.create(self.hparams["config"])

    def setup(self, stage: str) -> None:
        super().setup(stage)
        
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
                "monitor": "lr-AdamW"
            }
        return optimizer

    def get_loss(self, batch, eps=1e-6):
        x1 = batch['tokens']
        z0 = batch.get('z0')
        z1 = batch.get('z1')
        if z0 is None or z1 is None:    # on-the-fly generation for EmptyCoupling
            z0, z1 = self.model.coupling.sample_and_align(x1)
            
        xt, x_pad_mask, t, uz_mask, z_gap_mask, z_pad_mask = make_batch(
            x1,
            z0=z0,
            z1=z1,
            vocab_size=self.model.vocab_size,
            scheduler=self.model.scheduler,
        )
        
        ut, ins_probs, sub_probs = self.model(
            smiles=batch['smiles'],
            tokens=xt,
            token_padding_mask=x_pad_mask,
            t=t
        )
        
        predict_mask = (~x_pad_mask).unsqueeze(-1)
        ut = ut * predict_mask
        ins_probs = ins_probs * predict_mask
        sub_probs = sub_probs * predict_mask

        lambda_ins = ut[:, :, 0]
        lambda_sub = ut[:, :, 1]
        lambda_del = ut[:, :, 2]
        u_tia_ins = lambda_ins.unsqueeze(-1) * ins_probs
        u_tia_sub = lambda_sub.unsqueeze(-1) * sub_probs
        u_tia_del = lambda_del.unsqueeze(-1)

        ux_cat = torch.cat([u_tia_ins, u_tia_sub, u_tia_del], dim=-1)   # (batch_size, z_seq_len, 2 * vocab_size + 1)
        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap_mask, z_pad_mask)
        
        # compute Bregman divergence loss
        schedule_coeff = (self.model.scheduler.derivative(t) / (1 - self.model.scheduler(t))).unsqueeze(-1)
        log_uz_cat = torch.log(uz_cat + eps)
        loss = ut.sum(dim=(1, 2)) - (log_uz_cat * uz_mask * schedule_coeff).sum(dim=(1, 2)) # (batch_size,)
        loss = loss.mean()

        u_tot = ut.sum(dim=(1, 2)).mean()
        u_con = (uz_cat * uz_mask).sum(dim=(1, 2)).mean()
        u_ins = lambda_ins.sum(dim=1).mean()
        u_sub = lambda_sub.sum(dim=1).mean()
        u_del = lambda_del.sum(dim=1).mean()
        aux_dict = {'u_tot': u_tot, 'u_con': u_con, 'u_ins': u_ins, 'u_sub': u_sub,  'u_del': u_del}
        return loss, aux_dict
        
    def training_step(self, batch, batch_idx: int):
        loss, aux_dict = self.get_loss(batch)
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log_dict({f"train/{k}": v for k, v in aux_dict.items()}, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> Any:
        loss, _ = self.get_loss(batch)
        self.log("val/loss", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        return loss
