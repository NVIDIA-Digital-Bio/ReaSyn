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

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


def _SimpleMLP(dim_in: int, dim_out: int, dim_hidden: int) -> Callable[[torch.Tensor], torch.Tensor]:
    return nn.Sequential(
        nn.Linear(dim_in, dim_hidden),
        nn.ReLU(),
        nn.Linear(dim_hidden, dim_hidden),
        nn.ReLU(),
        nn.Linear(dim_hidden, dim_out),
    )


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dim_hidden: int | None = None):
        super().__init__()
        dim_hidden = dim_hidden or d_model * 2
        self.mlp = _SimpleMLP(d_model, num_classes, dim_hidden=dim_hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)
