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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reasyn.chem.fpindex import FingerprintIndex
from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.chem.mol import Molecule
from reasyn.chem.reaction import Reaction
from reasyn.chem.featurize import decode_smiles, TokenType

from .encoder import Encoder
from .decoder import Decoder
from .classifier_head import ClassifierHead
from reasyn.chem.featurize import TokenType


class ReaSyn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(**cfg.encoder)
        self.d_model: int = self.encoder.dim
        self.max_len = cfg.decoder.pe_max_len
        self.model_type = cfg.model_type
        self.vocab_size = int(max(TokenType))
        
        self.decoder = Decoder(**cfg.decoder,
                               vocab_size=self.vocab_size,
                               use_causal_mask=self.model_type == 'autoregressive',
                               use_time_embed=self.model_type == 'editflow')
        if self.model_type == 'editflow':
            from reasyn.utils.editflow_utils import get_coupling, KappaScheduler
            self.rates_out = ClassifierHead(self.d_model, 3)    # insert, substitute, delete
            self.ins_out = ClassifierHead(self.d_model, self.vocab_size)    # insert logits
            self.sub_out = ClassifierHead(self.d_model, self.vocab_size)    # substitute logits
            self.coupling = get_coupling(cfg.coupling_type, vocab_size=self.vocab_size)
            self.scheduler = KappaScheduler(cfg.scheduler_type)
        elif self.model_type == 'autoregressive':
            self.token_head = ClassifierHead(self.d_model, self.vocab_size)
    
    def forward(
        self,
        smiles: torch.Tensor,
        tokens: torch.Tensor,
        token_padding_mask: torch.Tensor | None,
        t: torch.Tensor | None = None   # for EditFlow
    ) -> torch.Tensor:
        code, code_padding_mask = self.encoder(smiles=smiles)
        h = self.decoder(
            code=code,
            code_padding_mask=code_padding_mask,
            tokens=tokens,
            token_padding_mask=token_padding_mask,
            t=t
        )
        if self.model_type == 'editflow':
            ut = self.rates_out(h)
            ins_logits = self.ins_out(h)
            sub_logits = self.sub_out(h)
            ut = F.softplus(ut) # ensure positive rates
            ins_probs = F.softmax(ins_logits, dim=-1)
            sub_probs = F.softmax(sub_logits, dim=-1)
            return ut, ins_probs, sub_probs
        logits = self.token_head(h)
        return logits
    
    def sample(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        tokens: torch.Tensor,
        token_padding_mask: torch.Tensor | None,
        t: torch.Tensor | None = None,  # for EditFlow
    ) -> torch.Tensor:
        h = self.decoder(
            code=code,
            code_padding_mask=code_padding_mask,
            tokens=tokens,
            token_padding_mask=token_padding_mask,
            t=t
        )
        if self.model_type == 'editflow':
            ut = self.rates_out(h)
            ins_logits = self.ins_out(h)
            sub_logits = self.sub_out(h)
            return ut, ins_logits, sub_logits
        if self.model_type == 'autoregressive':
            h = h[:, -1]    # (1, h_dim)
        logits = self.token_head(h)
        return logits
