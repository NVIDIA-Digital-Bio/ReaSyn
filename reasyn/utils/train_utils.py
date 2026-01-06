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
import random

import numpy as np
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def get_optimizer(cfg, model: torch.nn.Module):
    param_groups = list(model.parameters())
    if cfg["type"] == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(
                cfg["beta1"],
                cfg["beta2"],
            ),
        )
    elif cfg["type"] == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(
                cfg["beta1"],
                cfg["beta2"],
            ),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg['type']}")


def get_scheduler(cfg, optimizer: torch.optim.Optimizer):
    if cfg["type"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg["factor"],
            patience=cfg["patience"],
            min_lr=cfg["min_lr"],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {cfg['type']}")


def sum_weighted_losses(losses: dict[str, torch.Tensor], weights: dict[str, float] | None):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss: torch.Tensor = torch.zeros_like(list(losses.values())[0])
    if weights is None:
        weights = {k: 1.0 for k in losses.keys()}
    for k in losses.keys():
        loss = loss + weights[k] * losses[k]
    return loss


def worker_init_fn(worker_id: int):
    os.sched_setaffinity(0, range(os.cpu_count() or 1))
    global_rank = rank_zero_only.rank
    process_seed = torch.initial_seed()

    base_seed = process_seed - worker_id
    print(
        f"Initializing random number generators of process {global_rank} worker {worker_id} with base seed {base_seed}"
    )
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)
