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
import pickle
import sys
sys.path.append('.')
from reasyn.chem.fpindex import FingerprintOption, FingerprintIndex
from reasyn.chem.matrix import ReactantReactionMatrix


mols, fp_option, reactions = pickle.load(open('data/processed/comp_2048/tmp.pkl', 'rb'))
os.remove('data/processed/comp_2048/tmp.pkl')

fp_option = FingerprintOption(*fp_option)
fpindex = FingerprintIndex(mols, fp_option)
pickle.dump(fpindex, open('data/processed/comp_2048/fpindex.pkl', 'wb'))

rxn_matrix = ReactantReactionMatrix(mols, reactions)
pickle.dump(rxn_matrix, open('data/processed/comp_2048/matrix.pkl', 'wb'))
