#!/bin/bash

#SBATCH --job-name eval
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --partition batch_grad
#SBATCH -x ariel-k[1,2],ariel-m1
#SBATCH -t 3-0
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err

python run.py \
    'dataset.batch_size=32' \
    'run_type=eval' \
    'checkpoint_path="/data/soyeonhong/nlq/nlq_lightning/outputs/batch/2024-12-10/154714/epoch=24-val_R1_03=12.873.ckpt"'
