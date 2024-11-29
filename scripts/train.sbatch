#!/bin/bash

#SBATCH --job-name gvqa_no_decoder
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

NUMBER_Of_GPUS=$SLURM_GPUS_ON_NODE
export base_lr=0.0001
export base_bsz=$(( $NUMBER_Of_GPUS * 16 ))
export bsz_per_gpu=6
export bsz=$(( $NUMBER_Of_GPUS * $bsz_per_gpu ))
export lr=$(python -c "print(f'{""$base_lr / $base_bsz * $bsz"":.5e}')")
echo "Base LR: $base_lr, Base BSZ: $base_bsz, LR: $lr, BSZ: $bsz"

python run.py \
    "model=groundvqa_b" \
    "dataset.nlq_train_splits=[NLQ_train]" \
    "dataset.test_splits=[NLQ_val]" \
    "dataset.batch_size=${bsz_per_gpu}" \
    "trainer.gpus=${NUMBER_Of_GPUS}" \
    "optim.optimizer.lr=${lr}" \
    "trainer.find_unused_parameters=True"


# NLQv2 train
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py \
#     model=groundvqa_b \
#     'dataset.nlq_train_splits=[NLQ_train]' \
#     'dataset.test_splits=[NLQ_val]' \
#     dataset.batch_size=16 \
#     trainer.find_unused_parameters=True \
#     trainer.gpus=8


