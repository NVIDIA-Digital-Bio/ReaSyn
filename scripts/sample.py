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
import pathlib
import click
from reasyn.sampler.parallel import run_parallel_sampling
from reasyn.chem.mol import Molecule, read_mol_file


def _input_mols_option(p):
    return list(read_mol_file(p))


@click.command()
@click.option("--input", "-i", type=_input_mols_option, required=True)
@click.option("--output", "-o", type=click.Path(exists=False, path_type=pathlib.Path), required=True)
@click.option("--model_path", "-m", type=str, required=True)
@click.option("--search_width", type=int, default=4)
@click.option("--exhaustiveness", type=int, default=8)
@click.option("--num_gpus", type=int, default=-1)
@click.option("--num_workers_per_gpu", type=int, default=8)
@click.option("--task_qsize", type=int, default=0)
@click.option("--result_qsize", type=int, default=0)
@click.option("--time_limit", type=int, default=10000)
@click.option("--add_bb_path", type=str, default=None)
@click.option("--no_exact_break", is_flag=True)
@click.option("--num_cycles", type=int, default=1)
@click.option("--num_editflow_samples", type=int, default=100)
@click.option("--num_editflow_steps", type=int, default=100)
@click.option("--mols_to_filter", type=str, default=None)
@click.option("--filter_sim", type=float, default=0.8)
def main(
    input: list[Molecule],
    output: pathlib.Path,
    model_path: str,
    search_width: int,
    exhaustiveness: int,
    num_gpus: int,
    num_workers_per_gpu: int,
    task_qsize: int,
    result_qsize: int,
    time_limit: int,
    add_bb_path: str | None,
    no_exact_break: bool,
    num_cycles: int,
    num_editflow_samples: int,
    num_editflow_steps: int,
    mols_to_filter: str,
    filter_sim: float
):
    model_path = [pathlib.Path(path) for path in model_path.split(',')]
    assert all([path.exists() for path in model_path])
    
    run_parallel_sampling(
        input=input,
        output=output,
        model_path=model_path,
        search_width=search_width,
        exhaustiveness=exhaustiveness,
        num_gpus=num_gpus,
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        time_limit=time_limit,
        add_bb_path=add_bb_path,
        exact_break=not no_exact_break,
        num_cycles=num_cycles,
        num_editflow_samples=num_editflow_samples,
        num_editflow_steps=num_editflow_steps,
        mols_to_filter=_input_mols_option(mols_to_filter)
                       if mols_to_filter is not None else None,
        filter_sim=filter_sim
    )


if __name__ == "__main__":
    main()
