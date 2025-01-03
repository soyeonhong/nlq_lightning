#!/bin/bash

#SBATCH --job-name feedback_loss_both
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
    "+experiment=feedback" \
    "model.lm_enc_feed=True" \
    "model.lm_proj_feed=True" \
    "model.feed_last=True" \
    "model.feedback_type=rmse"


