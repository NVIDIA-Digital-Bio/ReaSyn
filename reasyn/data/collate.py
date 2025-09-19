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

from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F


def collate_tokens(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [F.pad(f, pad=[0, max_size - f.size(-1)], mode="constant", value=0) for f in features]
    return torch.stack(features_padded, dim=0)


def collate_2d_tokens(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [
        F.pad(f, pad=[0, max_size - f.size(-1), 0, max_size - f.size(-2)], mode="constant", value=0) for f in features
    ]
    return torch.stack(features_padded, dim=0)


def collate_1d_features(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [F.pad(f, pad=[0, 0, 0, max_size - f.size(-2)], mode="constant", value=0) for f in features]
    return torch.stack(features_padded, dim=0)


def collate_2d_features(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [
        F.pad(f, pad=[0, 0, 0, max_size - f.size(-2), 0, max_size - f.size(-3)], mode="constant", value=0)
        for f in features
    ]
    return torch.stack(features_padded, dim=0)


def collate_padding_masks(masks: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    masks_padded = [F.pad(m, pad=[0, max_size - m.size(-1)], mode="constant", value=True) for m in masks]
    return torch.stack(masks_padded, dim=0)


def apply_collate(
    spec: Mapping[str, Callable[[Sequence[torch.Tensor], int], torch.Tensor]],
    data_list: Sequence[dict[str, torch.Tensor]],
    max_size: int,
) -> dict[str, torch.Tensor]:
    transpose = {k: [d[k] for d in data_list] for k in spec.keys()}
    batch = {k: spec[k](transpose[k], max_size) for k in spec.keys()}
    return batch
