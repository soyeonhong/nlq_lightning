#!/bin/bash

#SBATCH --job-name gvqa_reproduce_nlq_lit
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

python run.py

