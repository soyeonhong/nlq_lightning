#!/bin/bash

#SBATCH --job-name object_aug_epoch_thr
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

# object_qa
python run.py \
    "+experiment=object_qa" \
    "dataset.batch_size=3"

# object_aug
# python run.py \
#     "+experiment=object_aug" \
#     "dataset.from_query=True" \
#     "model.epoch_thr=15"

