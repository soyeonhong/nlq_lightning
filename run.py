import re
import math
from argparse import ArgumentParser, Namespace

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities.distributed import rank_zero_only
from model.ours.trainer import get_trainer
from model.ours.dataset import JointDataModule
from model.ours.lightning_module import LightningModule
from omegaconf import OmegaConf, DictConfig

import os
import subprocess
from pathlib import Path

@rank_zero_only
def log_to_console(msg):
    print(msg)


@rank_zero_only
def write_batch_script(jid, default_root_dir):
    p_script = Path(default_root_dir) / f"slurm-{jid}.sh"
    command = f"scontrol write batch_script {jid} {p_script}"
    print(f'Writing batch script to {p_script}')
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    
@rank_zero_only
def log_checkpoint_info(checkpoint_path, missing_keys, unexpected_keys):
    print(f'Load checkpoint: {checkpoint_path}')
    print(f'Missing Keys: {missing_keys}')
    print(f'Unexpected Keys: {unexpected_keys}')
    
def within_slurm_batch():
    command = (
        "scontrol show jobid " + os.environ.get("SLURM_JOB_ID", "") +
        " | grep -oP '(?<=BatchFlag=)([0-1])'"
    )
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    batch_flag = int(result.stdout.strip())
    return batch_flag == 1

@hydra.main(config_path='config', config_name='base', version_base='1.3')
def train(config: DictConfig):
    pl.seed_everything(config.trainer.random_seed, workers=True)
    default_root_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    jid = os.environ.get("SLURM_JOB_ID")
    

    data = JointDataModule(config)
    data.setup()
    model = LightningModule(config)
    
    trainer, ckpt_callback = get_trainer(config, jid, enable_progress_bar=not within_slurm_batch())
    
    log_to_console('\n' + "="*80 + '\n')
    log_to_console(OmegaConf.to_yaml(config, resolve=True))
    log_to_console("="*80 + '\n')
        
    # write job script
    if within_slurm_batch():
        write_batch_script(jid, default_root_dir)


    if config.run_type == 'eval':  # evaluation
        trainer, _ = get_trainer(config, jid, enable_progress_bar=not within_slurm_batch(),
                                 enable_checkpointing=False)
        model = LightningModule.load_from_checkpoint(config.checkpoint_path)
        trainer.predict(
            model, [data.val_dataloader(), data.train_dataloader()],
        )
        
    else:  # training
        trainer.fit(
            model, data.train_dataloader(), [data.val_dataloader(), data.train_dataloader()], 
        )
        
        # evaluation
        p_ckpt = '/data/soyeonhong/nlq/nlq_lightning/outputs/debug/2024-12-09/154606/epoch=0-val_R1_03=0.000.ckpt'
        p_ckpt = p_ckpt if config.get('debug') else ckpt_callback.best_model_path

        model = LightningModule.load_from_checkpoint(p_ckpt)
        
        trainer.predict(
            model, [data.val_dataloader(), data.train_dataloader()],
        )

    
if __name__ == '__main__':
    if int(os.environ.get('SLURM_JOB_NUM_NODES', 1)) == 1 or int(os.environ.get('SLURM_NNODES', 1)) == 1:
        os.environ["SLURM_JOB_NAME"] = "bash"  # https://github.com/Lightning-AI/pytorch-lightning/issues/16236#issuecomment-1690552495
    OmegaConf.register_new_resolver("job_type", lambda : 'batch' if within_slurm_batch() else 'debug')
    OmegaConf.register_new_resolver('runtime_outdir', lambda : hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))
    
    train()
