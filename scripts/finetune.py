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
import click
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import callbacks, loggers, strategies
from reasyn.data.dataset import ProjectionDataModule
from reasyn.trainer.finetune import FinetuneWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


@click.command()
@click.option("--config_path", "-c", type=str, default="configs/finetune.yml")
@click.option("--pretrained_path", "-m", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--exp_name", "-n", type=str, default='debug_finetune')
@click.option("--seed", type=int, default=42)
@click.option("--batch-size", "-b", type=int, default=2)    # batch size per GPU
@click.option("--num-workers", type=int, default=8)
@click.option("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", 1)))
@click.option("--num-sanity-val-steps", type=int, default=1)
@click.option("--log-dir", type=click.Path(dir_okay=True, file_okay=False), default="./logs")
def main(
    config_path: str,
    pretrained_path: str,
    exp_name: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    num_nodes: int,
    num_sanity_val_steps: int,
    log_dir: str,
):
    os.makedirs(log_dir, exist_ok=True)
    pl.seed_everything(seed)

    config = OmegaConf.load(config_path)
    os.environ["WANDB_RUN_ID"] = exp_name
    
    resume, ckpt_path = pretrained_path, None
    logger = loggers.WandbLogger(project='reasyn', name=exp_name, save_dir=log_dir)
    save_dir = os.path.join(logger.save_dir, logger.name, logger._name, 'checkpoints')
    if os.path.exists(save_dir):
        filenames = os.listdir(save_dir)
        if filenames:
            last_filename = sorted(filenames, key=lambda x: int(x[5:-5]))[-1]
            resume = ckpt_path = os.path.join(save_dir, last_filename)
    
    datamodule = ProjectionDataModule(
        config,
        batch_size=batch_size,
        num_workers=num_workers,
        **config.data,
    )
    
    model = FinetuneWrapper(
        pretrained_path=pretrained_path,
        config=config,
        args={
            "config_path": config_path,
            "seed": seed,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "resume": resume,
        },
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        num_nodes=num_nodes,
        strategy=strategies.DDPStrategy(static_graph=True) if config.train.accumulate_grad_batches == 1
            else strategies.DDPStrategy(static_graph=False),
        num_sanity_val_steps=num_sanity_val_steps,
        gradient_clip_val=config.train.max_grad_norm,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        log_every_n_steps=1,
        max_steps=config.train.max_iters,
        callbacks=[
            callbacks.ModelCheckpoint(save_top_k=-1, filename='{step}'),
            callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=logger,
        val_check_interval=config.train.val_freq,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
