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
import argparse
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from tdc import Evaluator
from reasyn.chem.mol import Molecule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    file = parser.parse_args().file

    df = pd.read_csv(file)
    total = 1000

    # canonicalize
    df['target'] = df['target'].apply(lambda s: Molecule(s).csmiles)
    df['smiles'] = df['smiles'].apply(lambda s: Molecule(s).csmiles)
    df = df.drop_duplicates()

    print(df.loc[df.groupby("target").idxmax()["score"]].select_dtypes(include="number").sum() / total)

    count_success = len(df["target"].unique())
    print(f"Success rate: {count_success}/{total} = {count_success / total:.3f}")

    df2 = df[df['target'] == df['smiles']]
    df2 = df2.drop_duplicates('target')
    count_recons = len(df2)
    print(f"Reconstruction rate: {count_recons}/{total} = {count_recons / total:.3f}")

    rxn_pattern = re.compile('R\d+')
    product_good_pathways = defaultdict(list)
    bb_good_pathways = defaultdict(set)
    for _, row in df.iterrows():
        if row["score"] >= 0.8:
            product_good_pathways[row["target"]].append(row['smiles'])
            for mol_or_rxn in row['synthesis'].split(';'):
                if not rxn_pattern.match(mol_or_rxn):
                    bb_good_pathways[row["target"]].add(Molecule(mol_or_rxn).csmiles)
    
    evaluator = Evaluator('diversity')
    product_diversity = [evaluator(products) for products in product_good_pathways.values() if len(products) > 1]
    product_diversity = [d for d in product_diversity if not np.isnan(d)]
    product_diversity += [0] * (total - len(product_diversity))
    print(f"Mean diversity of products in sim > 0.8 pathways: {np.mean(product_diversity):.3f}")
    bb_diversity = [evaluator(bbs) for bbs in bb_good_pathways.values() if len(bbs) > 1]
    bb_diversity = [d for d in bb_diversity if not np.isnan(d)]
    bb_diversity += [0] * (total - len(bb_diversity))
    print(f"Mean diversity of BBs in sim > 0.8 pathways: {np.mean(bb_diversity):.3f}")
