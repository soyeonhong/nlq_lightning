import re
import math
from argparse import ArgumentParser, Namespace

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, open_dict
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import pytorch_lightning.utilities as L_utils

from model.ours.dataset import JointDataModule
from model.ours.lightning_module import LightningModule
from omegaconf import OmegaConf, DictConfig

import os
import subprocess
from pathlib import Path

@L_utils.rank_zero_only
def log_to_console(msg):
    print(msg)


@L_utils.rank_zero_only
def write_batch_script(jid, default_root_dir):
    p_script = Path(default_root_dir) / f"slurm-{jid}.sh"
    command = f"scontrol write batch_script {jid} {p_script}"
    print(f'Writing batch script to {p_script}')
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    
def within_slurm_batch():
    command = (
        "scontrol show jobid " + os.environ.get("SLURM_JOB_ID", "") +
        " | grep -oP '(?<=BatchFlag=)([0-1])'"
    )
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    batch_flag = int(result.stdout.strip())
    return batch_flag == 1


def _adjust_ddp_config(trainer_cfg):
    trainer_cfg = dict(trainer_cfg)
    strategy = trainer_cfg.get('strategy', None)
    if trainer_cfg['gpus'] > 1 and strategy is None:
        strategy = 'ddp'  # Select ddp by default
    if strategy == 'ddp':
        trainer_cfg['strategy'] = DDPPlugin(
            find_unused_parameters=trainer_cfg['find_unused_parameters'], 
            gradient_as_bucket_view=True)
    return trainer_cfg


@hydra.main(config_path='config', config_name='base')
def train(config: DictConfig):
    pl.seed_everything(config.trainer.random_seed, workers=True)
    trainer_cfg = Namespace(**_adjust_ddp_config(config.trainer))
    default_root_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    jid = os.environ.get("SLURM_JOB_ID")
    
    log_to_console('\n' + "="*80 + '\n')
    log_to_console(OmegaConf.to_yaml(config, resolve=True))
    log_to_console("="*80 + '\n')

    data = JointDataModule(config.dataset)
    data.setup()

    total_steps = trainer_cfg.max_epochs * math.floor(len(data.train_dataset) / trainer_cfg.gpus / config.dataset.batch_size)
    model = LightningModule(config, total_steps)
    if trainer_cfg.checkpoint_path:
        state_dict = torch.load(trainer_cfg.checkpoint_path, map_location='cpu')['state_dict']
        if not trainer_cfg.load_nlq_head:
            print('Train NLQ head from scratch')
            state_dict = {k: v for k, v in state_dict.items() if not "nlq_head" in k}
        if not trainer_cfg.load_decoder:
            print('Train LM decoder head from scratch')
            state_dict = {k: v for k, v in state_dict.items() if not ("decoder" in k or "lm_head" in k)}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f'Load checkpoint: {trainer_cfg.checkpoint_path}')
        print(f'Missing Keys: {missing_keys}')
        print(f'Unexpected Keys: {unexpected_keys}')
        
    # write job script
    if within_slurm_batch():
        write_batch_script(jid, default_root_dir)


    if trainer_cfg.test_only:  # evaluation
        trainer = pl.Trainer.from_argparse_args(
            trainer_cfg, 
            enable_checkpointing=False, 
            logger=False
        )
        if trainer_cfg.val:
            trainer.validate(
                model, data.val_dataloader(),
            )
        else:
            trainer.test(
                model, [data.val_dataloader(), data.train_dataloader()],
            )
    else:  # training
        model_checkpoint = []
        if 'QaEgo4D_test' in config.dataset.test_splits:
            model_checkpoint.append(
                ModelCheckpoint(
                    save_last=False, 
                    monitor='val_ROUGE', 
                    mode='max',
                    save_top_k=1, 
                    filename='{step}-{' + 'val_ROUGE' + ':.3f}')
            )
        if 'QaEgo4D_test_close' in config.dataset.test_splits:
            model_checkpoint.append(
                ModelCheckpoint(
                    save_last=False, 
                    monitor='val_close_acc', 
                    mode='max',
                    save_top_k=1, 
                    filename='{step}-{' + 'val_close_acc' + ':.3f}')
            )
        if 'NLQ_val' in config.dataset.test_splits:
            model_checkpoint.append(
                ModelCheckpoint(
                    dirpath=default_root_dir,
                    save_last=False, 
                    monitor='val_R1_03', 
                    mode='max',
                    save_top_k=1, 
                    filename='{epoch}-{' + 'val_R1_03' + ':.3f}')
            )
        trainer = pl.Trainer.from_argparse_args(trainer_cfg, 
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            # StochasticWeightAveraging(swa_lrs=1e-2),
            *model_checkpoint
        ],
        logger=TensorBoardLogger(
                save_dir=default_root_dir,
                version=os.environ.get("SLURM_JOB_ID"),
                name="lit",
                default_hp_metric=False
            ))
        trainer.fit(
            model, data.train_dataloader(), [data.val_dataloader(), data.train_dataloader()], 
        )

    
if __name__ == '__main__':
    if int(os.environ.get('SLURM_JOB_NUM_NODES', 1)) == 1 or int(os.environ.get('SLURM_NNODES', 1)) == 1:
        os.environ["SLURM_JOB_NAME"] = "bash"  # https://github.com/Lightning-AI/pytorch-lightning/issues/16236#issuecomment-1690552495
    OmegaConf.register_new_resolver("job_type", lambda : 'batch' if within_slurm_batch() else 'debug')
    OmegaConf.register_new_resolver('runtime_outdir', lambda : hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))
    
    train()
