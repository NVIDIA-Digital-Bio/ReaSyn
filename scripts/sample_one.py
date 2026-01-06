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

import sys
sys.path.append('.')
from time import time
from reasyn.sampler.parallel import run_sampling_one
from reasyn.chem.mol import Molecule


if __name__ == "__main__":
    # input = Molecule('Fc1cscc1CN1CCC(C2CCCOC2)CC1')
    input = Molecule('O=C(Nc1ccc(F)cc1)N(Cc1noc(C2CC2)n1)c1ccc(Cl)cc1Cl')
    
    t_start = time()
    df = run_sampling_one(
        input=input,
        model_path=['data/trained_model/NV-ReaSyn-AR-166M-v2.ckpt', 'data/trained_model/NV-ReaSyn-EB-174M-v2.ckpt'],
        exhaustiveness=4,
        search_width=2,
        num_cycles=2,
        num_editflow_samples=100,
        num_editflow_samples=10,
    )
    print(df)
    print(f'{time() - t_start:.2f} sec elapsed')
    