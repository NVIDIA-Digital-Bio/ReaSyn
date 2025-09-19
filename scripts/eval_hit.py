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
import pandas as pd
from tdc import Oracle
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


if __name__ == '__main__':
    oracle = Oracle('jnk3')

    hit = pd.read_csv('data/jnk3_hit.txt')
    hit['JNK3'] = oracle(hit['SMILES'].tolist())
    hit = hit.to_dict()
    thr = dict(zip(hit['SMILES'].values(), hit['JNK3'].values()))

    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    file = parser.parse_args().file

    df = pd.read_csv(file)
    total = len(df)
    print(total)

    if 'jnk3' not in df:
        df['jnk3'] = oracle(df['smiles'].tolist())
        df['thr'] = df['target'].apply(lambda s: thr[s])
        df.to_csv(file, index=False)

    df2 = df[df['jnk3'] > df['thr']]
    print(f'Improve rate:\t{len(df2) / total * 100:.2f} %')
    print(f'Similarity:\t{df2["score"].mean():.3f}')

    df2 = df[df['score'] >= 0.6]
    print(f'Analog rate:\t{len(df2) / total * 100:.2f} %')
    print(f'JNK3:\t\t{df2["jnk3"].mean():.3f}')

    df2 = df2[df2['jnk3'] > df2['thr']]
    print(f'Success rate:\t{len(df2) / total * 100:.2f} %')
