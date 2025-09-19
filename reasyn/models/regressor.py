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
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class Regressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, fps_tensor):
        return self.layer(fps_tensor)
    

class RegressorWrapper:
    def __init__(self, lr=1e-4, device='cuda'):
        super().__init__()
        self.model = Regressor().to(device)
        self.device = device
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def prepare_input(self, mol_list):
        fps_list = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in mol_list]
        fps_tensor = torch.Tensor(np.array(fps_list))
        return fps_tensor

    def train(self, mol_list, prop_list):
        fps_tensor = self.prepare_input(mol_list).to(self.device)
        preds = self.model(fps_tensor)
        loss = self.loss_fn(preds.squeeze(-1), torch.Tensor(prop_list).to(self.device))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def predict(self, mol_list):
        if mol_list and isinstance(mol_list[0], str):
            mol_list = [Chem.MolFromSmiles(s) for s in mol_list]
        fps_tensor = self.prepare_input(mol_list).to(self.device)
        return self.model(fps_tensor).squeeze(-1).tolist()
