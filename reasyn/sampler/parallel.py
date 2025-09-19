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

import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import os
import pathlib
import pickle
import subprocess
from multiprocessing import synchronize as sync
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from reasyn.chem.fpindex import FingerprintIndex
from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.chem.mol import FingerprintOption, Molecule
from reasyn.models.reasyn import ReaSyn
from reasyn.sampler.state_pool import StatePool, TimeLimit

import warnings
warnings.filterwarnings('ignore')


TaskQueueType: TypeAlias = "mp.JoinableQueue[Molecule | None]"
ResultQueueType: TypeAlias = "mp.Queue[tuple[Molecule, pd.DataFrame]]"


class Worker(mp.Process):
    def __init__(
        self,
        model_path: pathlib.Path,
        task_queue: TaskQueueType,
        result_queue: ResultQueueType,
        gpu_id: str,
        gpu_lock: sync.Lock,
        state_pool_opt: dict | None = None,
        max_evolve_steps: int = 8,
        max_results: int = 100,
        time_limit: int = 600,
        add_bb_path: str = None,
        verbose=True,
        exact_break=False,
        reward_model=None,
        mols_to_filter=None,
        filter_sim: float = 0.8
    ):
        super().__init__()
        self._model_path = model_path
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._gpu_id = gpu_id
        self._gpu_lock = gpu_lock

        self._state_pool_opt = state_pool_opt or {}
        self._max_evolve_steps = max_evolve_steps
        self._max_results = max_results
        self._time_limit = time_limit
        self.add_bb_path = add_bb_path
        self.verbose = verbose
        self.exact_break = exact_break
        self.reward_model = reward_model
        self.mols_to_filter = mols_to_filter
        self.filter_sim = filter_sim

    def run(self) -> None:
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_id

        ckpt = torch.load(self._model_path, map_location="cpu")
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        model = ReaSyn(config.model).to("cuda")
        model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        model.eval()
        self._model = model
        
        self._fpindex: FingerprintIndex = pickle.load(open(config.chem.fpindex, "rb"))
        self._rxn_matrix: ReactantReactionMatrix = pickle.load(open(config.chem.rxn_matrix, "rb"))
        if self.add_bb_path:
            _fpindex_add = pickle.load(open(self.add_bb_path, "rb"))
            num_orig_bb = len(self._fpindex._molecules)
            self._fpindex._molecules += _fpindex_add._molecules
            self._fpindex._fp = np.vstack([self._fpindex._fp, _fpindex_add._fp])
            if self.verbose:
                print(f'BB expanded: {num_orig_bb} -> {len(self._fpindex._molecules)}')
                
        try:
            while True:
                next_task = self._task_queue.get()
                if next_task is None:
                    self._task_queue.task_done()
                    break
                result_df = self.process(next_task)
                self._task_queue.task_done()
                self._result_queue.put((next_task, result_df))
        except KeyboardInterrupt:
            print(f"{self.name}: Exiting due to KeyboardInterrupt")
            return
        
    def process(self, mol: Molecule):
        sampler = StatePool(
            fpindex=self._fpindex,
            rxn_matrix=self._rxn_matrix,
            mol=mol,
            model=self._model,
            **self._state_pool_opt,
        )

        tl = TimeLimit(self._time_limit)
        for _ in range(self._max_evolve_steps):
            sampler.evolve(gpu_lock=self._gpu_lock, time_limit=tl, reward_model=self.reward_model)
            
            if self.exact_break:
                max_sim = max(
                    [
                        p.molecule.sim(mol, FingerprintOption.morgan_for_tanimoto_similarity())
                        for p in sampler.get_products()
                    ]
                    or [-1]
                )
                if max_sim == 1.0:
                    break

        df = sampler.get_dataframe()[: self._max_results]
        return df
        

class WorkerPool:
    def __init__(
        self,
        gpu_ids: list[int | str],
        num_workers_per_gpu: int,
        task_qsize: int,
        result_qsize: int,
        **worker_opt,
    ) -> None:
        super().__init__()
        self._task_queue: TaskQueueType = mp.JoinableQueue(task_qsize)
        self._result_queue: ResultQueueType = mp.Queue(result_qsize)
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

    def submit(self, task: Molecule, block: bool = True, timeout: float | None = None):
        self._task_queue.put(task, block=block, timeout=timeout)

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


def run_parallel_sampling(
    input: list[Molecule],
    output: pathlib.Path,
    model_path: pathlib.Path,
    search_width: int = 24,
    exhaustiveness: int = 64,
    num_gpus: int = -1,
    num_workers_per_gpu: int = 8,
    task_qsize: int = 0,
    result_qsize: int = 0,
    time_limit: int = 600,
    sort_by_scores: bool = True,
    add_bb_path: str = None,
    reward_model=None,
    mols_to_filter=None,
    filter_sim: float = 0.8
) -> None:
    num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
    
    pool = WorkerPool(
        gpu_ids=list(range(num_gpus)),
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        model_path=model_path,
        state_pool_opt={
            "factor": search_width,
            "max_active_states": exhaustiveness,
            "sort_by_score": sort_by_scores,
            "mols_to_filter": mols_to_filter,
            "filter_sim": filter_sim
        },
        time_limit=time_limit,
        add_bb_path=add_bb_path,
        reward_model=reward_model,
        mols_to_filter=mols_to_filter,
        filter_sim=filter_sim
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    total = len(input)
    for mol in input:
        pool.submit(mol)

    df_all: list[pd.DataFrame] = []
    with open(output, "w") as f:
        for _ in tqdm(range(total)):
            _, df = pool.fetch()
            if len(df) == 0:
                continue
            df.to_csv(f, float_format="%.3f", index=False, header=f.tell() == 0)
            df_all.append(df)
            
    df_merge = pd.concat(df_all, ignore_index=True)
    print(df_merge.loc[df_merge.groupby("target").idxmax()["score"]].select_dtypes(include="number").sum() / total)

    count_success = len(df_merge["target"].unique())
    print(f"Success rate: {count_success}/{total} = {count_success / total:.3f}")

    recons_targets: set[str] = set()
    for _, row in df_merge.iterrows():
        if row["score"] == 1.0:
            mol_target = Molecule(row["target"])
            mol_recons = Molecule(row["smiles"])
            if mol_recons.csmiles == mol_target.csmiles:
                recons_targets.add(row["target"])
    count_recons = len(recons_targets)
    print(f"Reconstruction rate: {count_recons}/{total} = {count_recons / total:.3f}")

    pool.end()


def run_parallel_sampling_return_smiles(
    input: list[Molecule],
    model_path: pathlib.Path,
    search_width: int = 24,
    exhaustiveness: int = 64,
    num_gpus: int = -1,
    num_workers_per_gpu: int = 6,
    task_qsize: int = 0,
    result_qsize: int = 0,
    time_limit: int = 600,
    sort_by_scores: bool = True,
    add_bb_path: str = None,
    reward_model=None,
    mols_to_filter=None,
    filter_sim: float = 0.8
) -> None:
    num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
    
    pool = WorkerPool(
        gpu_ids=list(range(num_gpus)),
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        model_path=model_path,
        state_pool_opt={
            "factor": search_width,
            "max_active_states": exhaustiveness,
            "sort_by_score": sort_by_scores,
            "mols_to_filter": mols_to_filter,
            "filter_sim": filter_sim
        },
        time_limit=time_limit,
        add_bb_path=add_bb_path,
        verbose=False,
        exact_break=True,
        reward_model=reward_model
    )

    total = len(input)
    for mol in input:
        pool.submit(mol)

    df_all: list[pd.DataFrame] = []

    for _ in tqdm(range(total)):
        _, df = pool.fetch()
        if len(df) == 0:
            continue
        df_all.append(df)

    df_merge = pd.concat(df_all, ignore_index=True)
    pool.end()

    return df_merge


def run_sampling_one(
    input: Molecule,
    model_path: pathlib.Path,
    sort_by_scores: bool = True,
    search_width: int = 24,
    exhaustiveness: int = 64,
    max_evolve_steps: int = 8,
    max_results: int = 100,
    time_limit: int = 600,
    add_bb_path: pathlib.Path = None,
    device='cuda',
    reward_model=None,
    mols_to_filter=None,
    filter_sim=0.8
) -> pd.DataFrame:

    ckpt = torch.load(model_path, map_location="cpu")
    config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
    model = ReaSyn(config.model).to(device)
    model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
    model.eval()
    _model = model
    
    state_pool_opt={
        "factor": search_width,
        "max_active_states": exhaustiveness,
        "sort_by_score": sort_by_scores,
        "mols_to_filter": mols_to_filter,
        "filter_sim": filter_sim
    }
    _fpindex: FingerprintIndex = pickle.load(open(config.chem.fpindex, "rb"))
    _rxn_matrix: ReactantReactionMatrix = pickle.load(open(config.chem.rxn_matrix, "rb"))
    if add_bb_path:
        _fpindex_add = pickle.load(open(add_bb_path, "rb"))
        num_orig_bb = len(_fpindex._molecules)
        _fpindex._molecules += _fpindex_add._molecules
        _fpindex._fp = np.vstack([_fpindex._fp, _fpindex_add._fp])
        print(f'BB expanded: {num_orig_bb} -> {len(_fpindex._molecules)}')

    sampler = StatePool(
        fpindex=_fpindex,
        rxn_matrix=_rxn_matrix,
        mol=input,
        model=_model,
        **state_pool_opt,
    )
    
    tl = TimeLimit(time_limit)
    for _ in range(max_evolve_steps):
        sampler.evolve(gpu_lock=None, time_limit=tl, reward_model=reward_model)
        
    df = sampler.get_dataframe()[: max_results]
    return df
