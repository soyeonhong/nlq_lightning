import hydra
import torch
import lightning as L
import pytorch_lightning.utilities as L_utils

from model.ours.dataset import JointDataModule
from model.ours.lightning_module import LightningModule
from omegaconf import OmegaConf, DictConfig

import os
import subprocess
from pathlib import Path

from model.ours.trainer import get_trainer

@L_utils.rank_zero_only
def log_to_console(msg):
    print(msg)


@L_utils.rank_zero_only
def write_batch_script(jid, default_root_dir):
    p_script = Path(default_root_dir) / f"slurm-{jid}.sh"
    command = f"scontrol write batch_script {jid} {p_script}"
    print(f'Writing batch script to {p_script}')
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    
@L_utils.rank_zero_only
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
    L.seed_everything(config.random_seed, workers=True)
    torch.set_float32_matmul_precision('highest')
    default_root_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    jid = os.environ.get("SLURM_JOB_ID")
    checkpoint_path = config.checkpoint_path
    test_only = config.test_only
    
    log_to_console('\n' + "="*80 + '\n')
    log_to_console(OmegaConf.to_yaml(config, resolve=True))
    log_to_console("="*80 + '\n')

    data = JointDataModule(config.dataset)
    data.setup()
    model = LightningModule(config)
    
    log_to_console('\n' + "="*80 + '\n')
    log_to_console(model)
    log_to_console('\n' + "="*80 + '\n')
    
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)['state_dict']
        if not config.load_nlq_head:
            print('Train NLQ head from scratch')
            state_dict = {k: v for k, v in state_dict.items() if not "nlq_head" in k}
        if not config.load_decoder:
            print('Train LM decoder head from scratch')
            state_dict = {k: v for k, v in state_dict.items() if not ("decoder" in k or "lm_head" in k)}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        # Use the rank_zero_only function for printing
        log_checkpoint_info(checkpoint_path, missing_keys, unexpected_keys)
        
    trainer, ckpt_callback = get_trainer(config, jid, enable_progress_bar=not within_slurm_batch())
        
    # write job script
    if within_slurm_batch():
        write_batch_script(jid, default_root_dir)

    if test_only:  # evaluation
        trainer.predict(
            model, [data.val_dataloader(), data.train_dataloader()],
        )
    else:  
        # training
        trainer.fit(
            model, data.train_dataloader(), [data.val_dataloader(), data.train_dataloader()], 
        )
        
        # evaluation
        # p_ckpt = 'outputs/batch/2024-10-13/130884/epoch=105-iou=0.4454.ckpt'
        # p_ckpt = p_ckpt if config.get('debug') else ckpt_callback.best_model_path
        # eval_config = hydra.compose(config_name=config.get('eval_config','eval'), overrides=[
        #     f'ckpt={str(p_ckpt).replace('=', '\\=')}',
        #     f'batch_size={config.batch_size}',
        #     f'num_workers={config.num_workers}',
        #     f'prefetch_factor={config.prefetch_factor}'
        # ])
        # model = LightningModule.load_from_checkpoint(p_ckpt)
        # data = JointDataModule(eval_config)

    
if __name__ == '__main__':
    if int(os.environ.get('SLURM_JOB_NUM_NODES', 1)) == 1 or int(os.environ.get('SLURM_NNODES', 1)) == 1:
        os.environ["SLURM_JOB_NAME"] = "bash"  # https://github.com/Lightning-AI/pytorch-lightning/issues/16236#issuecomment-1690552495
    OmegaConf.register_new_resolver("job_type", lambda : 'batch' if within_slurm_batch() else 'debug')
    OmegaConf.register_new_resolver('runtime_outdir', lambda : hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))
    
    train()
