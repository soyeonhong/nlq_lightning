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
    'model=groundvqa_b' \
    'dataset.nlq_train_splits=[NLQ_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=32 \
    trainer.test_only=True \
    'trainer.checkpoint_path="/data/soyeonhong/nlq/GroundVQA/checkpoints/GroundVQA_B-NLQ-VLG-val_R1_03=15.5.ckpt"' \
    'trainer.load_nlq_head=True'

# NLQv2 val
# CUDA_VISIBLE_DEVICES=1 python run.py \
#     model=groundvqa_b \
#     'dataset.qa_train_splits=[QaEgo4D_train]' \
#     'dataset.test_splits=[NLQ_val]' \
#     dataset.batch_size=32 \
#     +trainer.test_only=True \
#     '+trainer.checkpoint_path="lightning_logs/version_0/checkpoints/step=1403-val_R1_03=15.444.ckpt"' \
#     trainer.load_nlq_head=True