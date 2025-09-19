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

import re
import torch
from copy import deepcopy
from .autoregressive import Wrapper
from reasyn.chem.featurize import decode_tokens, TokenType
from reasyn.chem.stack import Stack
from reasyn.chem.mol import Molecule


class FinetuneWrapper(Wrapper):
    def __init__(self, pretrained_path, config, args):
        super().__init__(config, args)

        ckpt = torch.load(pretrained_path, weights_only=False)['state_dict']
        ckpt = {k: v for k, v in ckpt.items() if not k.startswith('prior')}
        self.model.load_state_dict({k[6:]: v for k, v in ckpt.items()}) # agent
        self.prior = deepcopy(self.model)   # prior
        for p in self.prior.parameters():
            p.requires_grad = False

        self.group_size = config.train.grpo.group_size
        self.softmax_temp = config.train.grpo.softmax_temp
        self.eps = config.train.grpo.eps
        self.beta = config.train.grpo.beta

    def get_reward(self, sequence, input_smiles):
        rxn_pattern = re.compile('R\d+')
        stack = Stack()
        is_next_int = False

        for mol_or_rxn in sequence.split(','):
            if rxn_pattern.match(mol_or_rxn):   # RXN
                rxn_idx = int(mol_or_rxn.lstrip('R'))
                success = stack.push_rxn(self.rxn_matrix.reactions[rxn_idx], rxn_idx)
                if not success:
                    return 0.
                is_next_int = True  # next mol_or_rxn is INT
            else:
                mol = Molecule(mol_or_rxn)
                if not mol.is_valid:
                    return 0.
                if not is_next_int: # BB
                    stack.push_mol(mol, 0)
                is_next_int = False
        
        input_mol = Molecule(input_smiles)
        max_sim = max([input_mol.sim(m) for m in stack.get_top()])
        reward = max_sim
        return reward

    def training_step(self, batch, batch_idx):
        input_smiles_list = [s for s in batch['smiles_raw']
                             for _ in range(self.group_size)]
        batch['tokens'] = self.model.sample_for_finetune(batch, group_size=self.group_size, softmax_temp=self.softmax_temp)
        sequences = [decode_tokens(token_ids) for token_ids in batch['tokens']]
        rewards = [self.get_reward(sequence, input_smiles)
                   for sequence, input_smiles in zip(sequences, input_smiles_list)]
        rewards = torch.Tensor(rewards).view(-1, self.group_size).to(self.device)   # bs, gs
        
        # ignore tokens after the first EOS token
        is_eos = batch['tokens'][:, 1:] == TokenType.END
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=self.device).expand(is_eos.size(0), -1)
        mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        rew_mean = rewards.mean(dim=1).unsqueeze(1)
        rew_std = rewards.std(dim=1).unsqueeze(1)
        advantages = (rewards - rew_mean) / (rew_std + 1e-4)
        advantages = advantages.view(-1).unsqueeze(1)
        
        agent_ll = self.model.get_ll(batch)
        prior_ll = self.prior.get_ll(batch)

        ll_diff = torch.clamp(agent_ll - prior_ll, -self.eps, self.eps)
        per_token_kl = torch.exp(ll_diff) - ll_diff - 1
        # x - x.detach() allows for preserving gradients from x
        advantages = torch.exp(agent_ll - agent_ll.detach()) * advantages
        per_token_loss = -(advantages - self.beta * per_token_kl)
        loss = ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)).mean()

        mean_kl = ((per_token_kl * mask).sum(dim=1) / mask.sum(dim=1)).mean()
        agent_ll = (agent_ll * mask).sum() / mask.sum()
        prior_ll = (prior_ll * mask).sum() / mask.sum()
        
        self.log('finetune/loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log('finetune/loss_kl', mean_kl, on_step=True, logger=True)
        self.log('finetune/prior_ll', prior_ll, on_step=True, logger=True)
        self.log('finetune/agent_ll', agent_ll, on_step=True, logger=True)
        self.log('finetune/reward', rewards.mean(), on_step=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        tokens = self.model.sample_for_finetune(batch)
        sequences = [decode_tokens(token_ids) for token_ids in tokens]
        rewards = [self.get_reward(sequence, input_smiles)
                   for sequence, input_smiles in zip(sequences, batch['smiles_raw'])]
        reward = torch.Tensor(rewards).mean().to(self.device)
        self.log("val/loss", reward, on_step=False, logger=True, sync_dist=True)

        return reward
