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
from torch import nn

from reasyn.chem.fpindex import FingerprintIndex
from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.chem.mol import Molecule
from reasyn.chem.reaction import Reaction
from reasyn.chem.featurize import decode_smiles, TokenType

from .encoder import Encoder
from .decoder import Decoder
from .classifier_head import ClassifierHead


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


class ReaSyn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(**cfg.encoder)
        self.d_model: int = self.encoder.dim
        self.max_len = cfg.decoder.pe_max_len
        self.vocab_size = max(TokenType) + 1
        # self.decoder = Decoder(**cfg.decoder, vocab_size=self.vocab_size)
        self.decoder = Decoder(**cfg.decoder, vocab_size=self.vocab_size + 2)   # to match with trained ckpt
        self.token_head = ClassifierHead(self.d_model, self.vocab_size)
        
    def forward(self, batch):
        code, code_padding_mask = self.encoder(batch)
        h = self.decoder(
            code=code,
            code_padding_mask=code_padding_mask,
            tokens=batch["tokens"],
            token_padding_mask=batch["token_padding_mask"]
        )
        logits = self.token_head(h)
        return logits

    ### for finetuning
    @torch.no_grad()
    def sample_for_finetune(self, batch, group_size=None, softmax_temp=0.1):
        code, code_padding_mask = self.encoder(batch)
        
        if group_size is not None:
            code = code.repeat_interleave(group_size, dim=0)
            code_padding_mask = code_padding_mask.repeat_interleave(group_size, dim=0)
        bs = code.shape[0]
        tokens = torch.full((bs, 1), TokenType.START).to(batch['smiles'].device)
        
        finished = torch.zeros(bs).byte()
        for _ in range(self.max_len - 1):
            h = self.decoder(
                code=code,
                code_padding_mask=code_padding_mask,
                tokens=tokens,
                token_padding_mask=None,
            )
            h_next = h[:, -1]
            logits = self.token_head(h_next)
            prob = torch.nn.functional.softmax(logits / softmax_temp, dim=1)
            next_token = torch.multinomial(prob, num_samples=1)
            tokens = torch.hstack([tokens, next_token])
            
            is_eos = (next_token.squeeze(-1) == TokenType.END).cpu()
            finished = torch.ge(finished + is_eos, 1)
            if finished.prod() == 1: break
        return tokens

    def get_ll(self, batch):
        code, code_padding_mask = self.encoder(batch)

        group_size = batch['tokens'].shape[0] // code.shape[0]
        if group_size is not None:
            code = code.repeat_interleave(group_size, dim=0)
            code_padding_mask = code_padding_mask.repeat_interleave(group_size, dim=0)
        
        h = self.decoder(
            code=code,
            code_padding_mask=code_padding_mask,
            tokens=batch['tokens'],
            token_padding_mask=None,
        )[:, :-1]
        logits = self.token_head(h)
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        
        target = batch['tokens'][:, 1:]
        ll = torch.gather(log_prob, dim=-1, index=target[..., None]).squeeze(-1)
        return ll

    ### for sampling; cannot batchrize
    @torch.no_grad()
    def predict(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        tokens: torch.Tensor,
        fpindex: FingerprintIndex,
        rxn_matrix: ReactantReactionMatrix,
        topk: int = 4,
        temperature_token: float = 0.1,
    ):
        def sample_token(tokens):
            h = self.decoder(
                code=code,
                code_padding_mask=code_padding_mask,
                tokens=tokens,
                token_padding_mask=None,
            )
            h_next = h[:, -1]  # (1, h_dim)
            
            token_logits = self.token_head(h_next)
            token_sampled = torch.multinomial(
                torch.nn.functional.softmax(token_logits / temperature_token, dim=-1),
                num_samples=1,
            )
            return token_sampled, token_logits

        def get_reactants(smiles) -> list[list[_ReactantItem]]:
            mol = Molecule(smiles)
            if mol._rdmol is None:
                return
            
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
        
        def get_reactions(reaction_logits) -> list[list[_ReactionItem]]:
            reaction_probs = reaction_logits.softmax(dim=-1)
            sorted_indices = (-reaction_probs).argsort()
            reaction_probs = reaction_probs[sorted_indices]

            return [_ReactionItem(reaction=rxn_matrix.reactions[idx], index=idx, score=score)
                    for idx, score in zip(sorted_indices, reaction_probs)]
        
        assert len(tokens.shape) == 1, 'no batch allowed'
        
        if len(tokens) > self.max_len:
            sampled_type = 'ABORTED'
            sampled_item = None
        else:
            tokens = tokens[None, :]
            token_sampled, token_logits = sample_token(tokens)
        
        if token_sampled == TokenType.MOL_START:
            sampled_type = 'BB'
            token_sampled_bb = []
            while token_sampled != TokenType.MOL_END and tokens.shape[-1] < self.max_len - 2:
                tokens = torch.hstack([tokens, token_sampled])
                token_sampled, _ = sample_token(tokens)
                token_sampled_bb.append(token_sampled)
            token_sampled_bb = torch.tensor(token_sampled_bb)[:-1]  # exclude MOL_END
            smiles = decode_smiles(token_sampled_bb)
            sampled_item = get_reactants(smiles)
            if sampled_item is None:
                sampled_type = 'ABORTED'
        
        elif token_sampled >= TokenType.RXN_MIN:
            sampled_type = 'RXN'
            reaction_logits = token_logits[0, TokenType.RXN_MIN : TokenType.RXN_MAX + 1]    # (115,)
            sampled_item = get_reactions(reaction_logits)

        else:
            sampled_type = 'END'
            sampled_item = None

        return PredictResult(sampled_type, sampled_item)
