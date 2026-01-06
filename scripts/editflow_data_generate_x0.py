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
import math
import argparse
import torch
from collections import defaultdict
from datasets import Dataset
from tqdm import tqdm
from omegaconf import OmegaConf
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import subprocess
from multiprocessing import synchronize as sync
import warnings
warnings.filterwarnings('ignore')

from reasyn.models.reasyn import ReaSyn
from reasyn.data.dataset import ProjectionDataModule
from reasyn.chem.featurize import TokenType
from reasyn.utils.editflow_utils import get_coupling, KappaScheduler


class Worker(mp.Process):
    def __init__(
        self,
        task_queue,
        result_queue,
        gpu_id: str,
        gpu_lock: sync.Lock,
        coupling_type: str,
        scheduler_type: str,
        model_path: str | None = None,
    ):
        super().__init__()
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._gpu_id = gpu_id
        self._gpu_lock = gpu_lock
        self._coupling_type = coupling_type
        self._scheduler_type = scheduler_type
        self._model_path = model_path
        self._vocab_size = int(max(TokenType))
        
    def run(self) -> None:
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_id

        # load pretrained model
        assert self._model_path is not None
        ckpt = torch.load(self._model_path, weights_only=False)
        pretrained_config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        pretrained_model = ReaSyn(pretrained_config.model)
        pretrained_model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        pretrained_model.to('cuda').eval()
        for p in pretrained_model.parameters():
            p.requires_grad = False
        self.model = pretrained_model
        assert self._vocab_size == self.model.vocab_size
        
        # load coupling and scheduler for EditFlow
        self.coupling = get_coupling(self._coupling_type, vocab_size=self._vocab_size)
        self.scheduler = KappaScheduler(self._scheduler_type)

        while True:
            next_task = self._task_queue.get()
            if next_task is None:
                self._task_queue.task_done()
                break
            try:
                result = self.process(next_task)
            except KeyboardInterrupt:
                print(f"{self.name}: Exiting due to KeyboardInterrupt")
                return
            except:
                result = {'smiles': [], 'tokens': [], 'x0': []}
            self._task_queue.task_done()
            self._result_queue.put(result)
        
    def process(self, batch):
        self._gpu_lock.acquire()
        batch['smiles'] = batch['smiles'].to('cuda')
        _x0 = self.coupling.sample(self.model, batch['smiles']).cpu()
        batch_tokens = []
        # batch['tokens']: exclude padding tokens to save memory
        for tokens in batch['tokens']:
            pad_idx = torch.where(tokens == TokenType.END)[0][0] + 1    # + 1 for EOS
            batch_tokens.append(tokens[:pad_idx])
        x0 = []
        # x0: fill with [END] after the first [END]
        for tokens in _x0:
            pad_idx = torch.where(tokens == TokenType.END)[0][0] + 1    # + 1 for EOS
            x0.append(tokens[:pad_idx])
        self._gpu_lock.release()
        return {'smiles': list(batch['smiles'].cpu()),
                'tokens': batch_tokens,
                'x0': x0}
        

class WorkerPool:
    def __init__(
        self,
        gpu_ids: list[int | str],
        num_workers_per_gpu: int,
        task_qsize: int = 0,
        result_qsize: int = 0,
        **worker_opt,
    ) -> None:
        super().__init__()
        self._task_queue = mp.JoinableQueue(task_qsize)
        self._result_queue = mp.Queue(result_qsize)
        self._gpu_ids = [str(d) for d in gpu_ids]
        self._gpu_locks = [mp.Lock() for _ in gpu_ids]
        num_gpus = len(gpu_ids)
        num_workers = num_workers_per_gpu * num_gpus
        self._workers = [
            Worker(
                task_queue=self._task_queue,
                result_queue=self._result_queue,
                gpu_id=self._gpu_ids[i % num_gpus],
                gpu_lock=self._gpu_locks[i % num_gpus],
                **worker_opt,
            )
            for i in range(num_workers)
        ]

        for w in self._workers:
            w.start()

    def submit(self, batch, block: bool = True, timeout: float | None = None):
        self._task_queue.put(batch, block=block, timeout=timeout)

    def fetch(self, block: bool = True, timeout: float | None = None):
        return self._result_queue.get(block=block, timeout=timeout)

    def kill(self):
        for w in self._workers:
            w.kill()
        self._result_queue.close()
        self._task_queue.close()

    def end(self):
        for _ in self._workers:
            self._task_queue.put(None)
        self._task_queue.join()
        for w in self._workers:
            w.terminate()
        self._result_queue.close()
        self._task_queue.close()


def _count_gpus():
    return int(
        subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l", shell=True, text=True
        ).strip()
    )


def save_shard(shard, path, data_all):
    dataset = Dataset.from_dict(data_all)
    dataset.to_parquet(os.path.join(path, f'{shard}.parquet'))
    print(f'\033[92mShard {shard} saved ({len(data_all["smiles"])})\033[0m')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    parser.add_argument('--pretrained_path', '-m', type=str, default='data/trained_model/NV-ReaSyn-AR-166M-v2.ckpt')
    parser.add_argument('--data_path', '-d', type=str, default='data/edit_bridge')
    parser.add_argument('--batch-size', '-b', type=int, default=8)
    parser.add_argument('--num-workers-per-gpu', type=int, default=32)  # number of processes
    parser.add_argument('--num-data', type=int, default=1_024_000)      # number of data generation
    parser.add_argument('--num-data-per-shard', type=int, default=10240)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    assert config.model.model_type == 'editflow'
    assert config.model.coupling_type == 'bridge'
    args.data_path += f'_x0'
    num_workers = args.num_workers_per_gpu * _count_gpus()
    print(f'\033[92mCoupling: {config.model.coupling_type} | '
          f'Save path: {args.data_path} | '
          f'Workers: {num_workers}\033[0m')
    print(f'\033[92mPretrained model: {args.pretrained_path}\033[0m')
    
    if os.path.exists(args.data_path):  # resume data generation
        filenames = os.listdir(args.data_path)
        shard_start = max([int(os.path.splitext(f)[0]) for f in os.listdir(args.data_path)] or [-1]) + 1
    else:
        os.mkdir(args.data_path)
        shard_start = 0
    
    datamodule = ProjectionDataModule(
        config,
        batch_size=args.batch_size,
        num_workers=4, #os.cpu_count()
        **config.data,
    )
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    
    pool = WorkerPool(
        gpu_ids=list(range(_count_gpus())),
        num_workers_per_gpu=args.num_workers_per_gpu,
        model_path=args.pretrained_path,
        coupling_type=config.model.coupling_type,
        scheduler_type=config.model.scheduler_type,
    )
    
    data_all = defaultdict(list)
    shard = shard_start
    num_iters = math.ceil((args.num_data - shard_start * args.num_data_per_shard) / args.batch_size)
    num_data = 0
    
    for i, batch in enumerate(tqdm(dataloader, total=num_iters)):
        if i == num_iters: break
        pool.submit(batch)
        data = pool.fetch()
        torch.cuda.empty_cache()
        for k in data:
            data_all[k].extend(data[k])
        if len(data_all[k]) >= args.num_data_per_shard:
            save_shard(shard, args.data_path, data_all)
            num_data += len(data_all[k])
            data_all = defaultdict(list)
            shard += 1
        
    if len(data_all[k]):
        save_shard(shard, args.data_path, data_all)
        num_data += len(data_all[k])
    print(f'\033[92m{num_data} aligned data saved\033[0m')
    pool.end()
