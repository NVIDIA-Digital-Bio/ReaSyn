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
import random
import torch
import joblib
import numpy as np
import tdc
from tdc.generation import MolGen
import yaml
from joblib import delayed
from rdkit import Chem
import reasyn.utils.crossover as ga
from omegaconf import OmegaConf
from reasyn.chem.mol import Molecule
from reasyn.sampler.parallel import run_parallel_sampling_return_smiles
from reasyn.models.regressor import RegressorWrapper


def sanitize(mol_list):
    new_mol_list = []
    smiles_set = set()
    for mol in mol_list:
        if mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol)
                if smiles is not None and smiles not in smiles_set:
                    smiles_set.add(smiles)
                    new_mol_list.append(mol)
            except KeyboardInterrupt:
                quit()
            except:
                print("bad smiles")
    return new_mol_list, list(smiles_set)


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


class Oracle:
    def __init__(self, name, mol_buffer={}):
        self.name = name
        self.evaluator = None
        self.max_oracle_calls = 10000
        self.freq_log = 100
        self.mol_buffer = mol_buffer
        self.sa_scorer = tdc.Oracle(name='sa')
        self.diversity_evaluator = tdc.Evaluator(name='diversity')
        self.last_log = 0
        self.output_dir = "results_pmo"
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self):
        output_file_path = os.path.join(self.output_dir, f"{self.name}.yaml")
        self.sort_buffer()
        with open(output_file_path, "w") as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False):
        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[: self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_top10_sa = np.mean(self.sa_scorer(smis[:10]))
        diversity_top100 = self.diversity_evaluator(smis)

        msg = (
            f'{n_calls}/{self.max_oracle_calls} | '
            f'avg_top1: {avg_top1:.3f} | '
            f'avg_top10: {avg_top10:.3f} | '
            f'avg_top100: {avg_top100:.3f} | '
            f'auc_top1: {top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls):.3f} | '
            f'auc_top10: {top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls):.3f} | '
            f'auc_top100: {top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls):.3f} | '
            f'avg_top10_sa: {avg_top10_sa:.3f} | '
            f'div: {diversity_top100:.3f}')
        print(msg)
        with open(os.path.join(self.output_dir, f"{self.name}.log"), "a") as f:
            f.write(msg + '\n')

    def __len__(self):
        return len(self.mol_buffer)

    def score_smi(self, smi):
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                pass
            else:
                self.mol_buffer[smi] = [float(self.evaluator(smi)), len(self.mol_buffer) + 1]
            return self.mol_buffer[smi][0]

    def __call__(self, smiles_lst):
        if isinstance(smiles_lst, list):
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result()
        else:
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result()
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


def projection(mol_list, args, regressor=None):
    _, smiles_list = sanitize(mol_list)
    input = [Molecule(s) for s in smiles_list]
    result_df = run_parallel_sampling_return_smiles(
        input=input,
        model_path=args.model_path,
        search_width=args.search_width,
        exhaustiveness=args.exhaustiveness,
        reward_model=regressor
    )
    result_df.drop_duplicates(subset="target", inplace=True, keep="first")
    smiles_list = result_df.smiles.to_list()
    return [Chem.MolFromSmiles(s) for s in smiles_list]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    

if __name__ == "__main__":
    import argparse
    from time import time

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', "--oracle",           type=str,   required=True, 
                        choices=['amlodipine_mpo',
                                 'celecoxib_rediscovery',
                                 'drd2',
                                 'fexofenadine_mpo',
                                 'gsk3b',
                                 'jnk3',
                                 'median1',
                                 'median2',
                                 'osimertinib_mpo',
                                 'perindopril_mpo',
                                 'ranolazine_mpo',
                                 'sitagliptin_mpo',
                                 'zaleplon_mpo'])
    parser.add_argument('-m', "--model_path",       type=str,   required=True)
    parser.add_argument("--search_width",           type=int,   default=2)
    parser.add_argument("--exhaustiveness",         type=int,   default=4)
    parser.add_argument("--population_size",        type=int,   default=100)
    parser.add_argument("--offspring_size",         type=int,   default=100)
    parser.add_argument("--mutation_rate",          type=float, default=0.1)
    parser.add_argument("--use_regressor",          action='store_true')
    parser.add_argument("--seed",                   type=int,   default=0)
    args = parser.parse_args()

    config = OmegaConf.load('reasyn/utils/hparams_tdc.yml')
    args.regressor_lr = config[args.oracle]
    
    set_seed(args.seed)
    fname = f'{args.oracle}_{args.seed}'
    print(f'\033[92m{fname}\033[0m')

    if args.use_regressor:
        regressor = RegressorWrapper(lr=args.regressor_lr)
        regressor_mol, regressor_prop = [], []
        regressor_max_data_size = 1000
    else:
        regressor = None

    oracle = Oracle(fname)
    oracle.assign_evaluator(tdc.Oracle(name=args.oracle))
    pool = joblib.Parallel(n_jobs=64)

    all_smiles = MolGen(name="ZINC").get_data().smiles.to_list()
    population_smiles = np.random.choice(all_smiles, args.population_size)
    population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
    del all_smiles, population_smiles
    population_mol = projection(population_mol, args)
    population_scores = oracle([Chem.MolToSmiles(mol) for mol in population_mol])

    patience = 0
    t_start = time()
    while True:
        if len(oracle) > 100:
            oracle.sort_buffer()
            old_score = np.mean([item[1][0] for item in list(oracle.mol_buffer.items())[:100]])
        else:
            old_score = 0

        mating_pool = ga.make_mating_pool(population_mol, population_scores, args.population_size)
        offspring_mol = pool(
            delayed(ga.reproduce)(mating_pool, args.mutation_rate) for _ in range(args.offspring_size)
        )
        offspring_mol = projection(offspring_mol, args, regressor)
        offspring_scores = oracle([Chem.MolToSmiles(mol) for mol in offspring_mol])

        population_mol += offspring_mol
        population_scores += offspring_scores
        population_tuples = list(zip(population_scores, population_mol))
        population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[: args.population_size]
        population_mol = [t[1] for t in population_tuples]
        population_scores = [t[0] for t in population_tuples]

        # early stopping
        if len(oracle) > 100:
            oracle.sort_buffer()
            new_score = np.mean([item[1][0] for item in list(oracle.mol_buffer.items())[:100]])
            if (new_score - old_score) < 1e-3:
                patience += 1
                if patience >= 5:
                    oracle.log_intermediate(finish=True)
                    print("convergence criteria met, abort ...... ")
                    break
            else:
                patience = 0
            old_score = new_score

        # regressor training
        if args.use_regressor:
            regressor_mol += offspring_mol
            regressor_prop += offspring_scores
            if len(regressor_mol) > regressor_max_data_size:
                regressor_mol = regressor_mol[-regressor_max_data_size:]
                regressor_prop = regressor_prop[-regressor_max_data_size:]
            regressor.train(regressor_mol, regressor_prop)
        
        if oracle.finish:
            break
        oracle.save_result()
    print(f'\033[92m{fname} | {time() - t_start:.2f} sec elapsed\033[0m')
