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
from reasyn.chem.mol import Molecule, read_mol_file


def _input_mols_option(p):
    return list(read_mol_file(p))


if __name__ == "__main__":
    # input = Molecule('Fc1cscc1CN1CCC(C2CCCOC2)CC1')
    # input = Molecule('CN(CCNC(=O)CC1CCC2(CC1)CCC2O)C(=O)c1cccc2ncnn12')
    # input = Molecule('CCCC[C@H](NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCNC(=N)N)NC(=O)c1ccc(/C=C2\SC(=O)N(c3ccc(C)cc3)C2=O)cc1)C(N)=O')
    input = Molecule('O=C(Nc1ccc(F)cc1)N(Cc1noc(C2CC2)n1)c1ccc(Cl)cc1Cl')
    
    t_start = time()
    df = run_sampling_one(
        input=input,
        model_path='data/trained_model/model.ckpt',
        exhaustiveness=1,
        search_width=1,
    )
    print(df)
    print(f'{time() - t_start:.2f} sec elapsed')
    