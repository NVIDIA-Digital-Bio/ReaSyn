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

import pickle
import sys
sys.path.append('.')
from reasyn.chem.mol import Molecule
from reasyn.chem.reaction import Reaction


fpindex = pickle.load(open('data/processed/comp_2048/fpindex.pkl', 'rb'))
rxn_matrix = pickle.load(open('data/processed/comp_2048/matrix.pkl', 'rb'))

mols = [Molecule(m.csmiles) for m in fpindex._molecules]
fp_option = (fpindex._fp_option.type, fpindex._fp_option.morgan_radius,
             fpindex._fp_option.morgan_n_bits, fpindex._fp_option.rdkit_fp_size)
reactions = [Reaction(r.smarts) for r in rxn_matrix._reactions]
pickle.dump((mols, fp_option, reactions),
            open('data/processed/comp_2048/tmp.pkl', 'wb'))
