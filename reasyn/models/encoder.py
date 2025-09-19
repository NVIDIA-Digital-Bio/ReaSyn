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

from torch import nn
from reasyn.models.positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self,
        num_token_types: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        pe_max_len: int,
    ):
        super().__init__()
        self._dim = d_model
        self.smiles_emb = nn.Embedding(num_token_types, d_model, padding_idx=0)
        self.pe_enc = PositionalEncoding(
            d_model=d_model,
            max_len=pe_max_len,
        )
        self.enc = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch):
        smiles = batch["smiles"]
        h = self.pe_enc(self.smiles_emb(smiles))
        padding_mask = smiles == 0  # the positions with the value of True will be ignored
        out = self.enc(h, src_key_padding_mask=padding_mask)
        return out, padding_mask
