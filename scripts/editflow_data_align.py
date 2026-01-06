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
import os
import argparse
import joblib
from collections import defaultdict
from datasets import Dataset
from tqdm import tqdm
from omegaconf import OmegaConf

from reasyn.data.dataset import ProjectionDataModule, EditFlowDataModule
from reasyn.chem.featurize import TokenType
from reasyn.utils.editflow_utils import get_coupling, align_x2z


def get_aligned_batch(batch, coupling):
    try:
        x1 = batch['tokens']
        if 'x0' in batch:
            x0 = batch['x0']
        else:
            x0 = coupling.sample(x1)
        z0, z1 = align_x2z(x0, x1)
        return {'smiles': list(batch['smiles']),
                'tokens': list(batch['tokens']),
                'z0': list(z0),
                'z1': list(z1)}
    except:
        return {'smiles': [],
                'tokens': [],
                'z0': [],
                'z1': []}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    parser.add_argument('--data_path', '-d', type=str, default='data/edit_bridge')
    parser.add_argument('--batch-size', '-b', type=int, default=512)    # batch to parallelize
    parser.add_argument('--num-data', type=int, default=51_200_000)     # for UniformCoupling
    parser.add_argument('--num-data-per-shard', type=int, default=1_024_000)    # ~300MB per shard
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    assert config.model.model_type == 'editflow'
    num_workers = os.cpu_count()
    data_path = args.data_path
    print(f'\033[92mCoupling: {config.model.coupling_type} | '
          f'Save path: {data_path} | '
          f'Workers: {num_workers}\033[0m')
    os.mkdir(data_path)
    
    if config.model.coupling_type == 'bridge':
        # x0 already generated
        args.data_path += f'_x0'
        datamodule = EditFlowDataModule(
            config,
            batch_size=1,
            num_workers=num_workers,
            train_shuffle=False,
            data_path=args.data_path,
            **config.data,
        )
    else:
        datamodule = ProjectionDataModule(
            config,
            batch_size=1,
            num_workers=num_workers,
            **config.data,
        )
    datamodule.setup()
    dataloader = datamodule.train_dataloader()

    vocab_size = int(max(TokenType))   # exclude EPS token
    coupling = get_coupling(config.model.coupling_type, vocab_size=vocab_size)
    
    pool = joblib.Parallel(n_jobs=num_workers)

    batch = []
    aligned_all = defaultdict(list)
    # shard = shard_start
    # num_iters = args.num_data - shard_start * args.num_data_per_shard
    shard = 0
    num_data = 0        # to track the number of errors
    
    if config.model.coupling_type == 'bridge':
        pbar = tqdm(dataloader)
    else:
        pbar = tqdm(dataloader, total=args.num_data)
    for i, data in enumerate(pbar):
        if i == args.num_data: break
        batch.append(data)
        
        if len(batch) == args.batch_size:
            aligned_batch = pool(joblib.delayed(get_aligned_batch)(b, coupling) for b in batch)
            for aligned_data in aligned_batch:
                for k in aligned_data:
                    aligned_all[k].extend(aligned_data[k])
                if len(aligned_all[k]) >= args.num_data_per_shard:
                    dataset = Dataset.from_dict(aligned_all)
                    dataset.to_parquet(os.path.join(data_path, f'{shard}.parquet'))
                    del dataset
                    num_data += len(aligned_all[k])
                    print(f'\033[92mShard {shard} saved ({len(aligned_all[k])})\033[0m')
                    aligned_all = defaultdict(list)
                    shard += 1
            batch = []
    
    if len(aligned_all[k]):
        dataset = Dataset.from_dict(aligned_all)
        dataset.to_parquet(os.path.join(data_path, f'{shard}.parquet'))
        num_data += len(aligned_all[k])
        print(f'\033[92mShard {shard} saved ({len(aligned_all[k])})\033[0m')
    print(f'\033[92m{num_data} aligned data saved\033[0m')
