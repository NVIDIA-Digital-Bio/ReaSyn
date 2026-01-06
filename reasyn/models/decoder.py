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
from torch import nn

from reasyn.models.positional_encoding import PositionalEncoding
from reasyn.models.time_embedder import TimestepEmbedder    # for EditFlow
from reasyn.models.transformer_rope import TransformerDecoderLayerRoPE  # for EditFlow


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 6,
        pe_max_len: int = 32,
        output_norm: bool = False,
        vocab_size: int = 272,
        use_causal_mask: bool = False,
        use_time_embed: bool = False,   # for EditFlow
    ) -> None:
        super().__init__()
        self.use_causal_mask = use_causal_mask
        self.use_time_embed = use_time_embed
        self.in_token = nn.Embedding(vocab_size, d_model)
        if use_time_embed:
            self.te_dec = TimestepEmbedder(d_model)
            layer = TransformerDecoderLayerRoPE
        else:
            self.pe_dec = PositionalEncoding(d_model=d_model, max_len=pe_max_len)
            layer = nn.TransformerDecoderLayer

        self.dec = nn.TransformerDecoder(
            decoder_layer=layer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if output_norm else None,
        )

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        emb_token = self.in_token(tokens)
        if not self.use_time_embed:   # no pe_dec (absolute PE) for EditFlow
            emb_token = self.pe_dec(emb_token)
        return emb_token

    def forward(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        tokens: torch.Tensor,
        token_padding_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None   # for EditFlow
    ) -> torch.Tensor:
        bsz, seqlen = tokens.size()

        x = self.embed(tokens)
        if t is not None and self.use_time_embed:
            x = x + self.te_dec(t)

        causal_mask = (
            nn.Transformer.generate_square_subsequent_mask(
                sz=x.size(1),
                dtype=x.dtype,
                device=x.device,
            )
            if self.use_causal_mask else None
        )
        tgt_key_padding_mask = (
            torch.zeros(
                [bsz, seqlen],
                dtype=x.dtype,
                device=x.device,
            ).masked_fill_(token_padding_mask, -torch.finfo(x.dtype).max)
            if token_padding_mask is not None else None
        )
        y = self.dec(
            tgt=x,
            memory=code,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=code_padding_mask,
        )   # (bsz, seq_len, d_model)
        return y
