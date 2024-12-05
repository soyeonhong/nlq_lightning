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

set -e  # exit on error

hostname

python run.py \
    'dataset.batch_size=32' \
    'test_only=True' \
    'checkpoint_path="/data/soyeonhong/nlq/nlq_lightning/checkpoints/GroundVQA_B-NLQ-VLG-val_R1_03=15.5.ckpt"' \
    'load_nlq_head=True'