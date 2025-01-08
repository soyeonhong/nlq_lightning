#!/bin/bash

#SBATCH --job-name=baseline_eval
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=43G
#SBATCH -x ariel-k[1,2],ariel-m1

set -e  # exit on error

hostname

sshopt="-o UserKnownHostsFile=/data/$USER/.ssh/known_hosts -i /data/$USER/.ssh/id_rsa"

################# get batch host info and network interfaces #################
batchhost=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
batchhostip=$(getent hosts $batchhost | head -n1 | awk '{print $1}')
batchhostport=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
interfaces=()
for host in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    echo $host
    hostip=$(ssh $sshopt $host hostname -i | awk '{print $1}')
    interfaces+=($(ssh $sshopt $host bash -c "ifconfig | grep -B1 $hostip | head -n1 | awk '{print \$1}' | sed 's/:\$//'"))
done
interfaces=$(echo "${interfaces[@]}" | tr ' ' ',')  # string join

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Batch host: $batchhost"
echo "Batch host IP: $batchhostip"
echo "Batch host port: $batchhostport"
echo "Network interfaces: ${interfaces[@]}"


MASTER_ADDR=$batchhostip MASTER_PORT=$batchhostport NCCL_SOCKET_IFNAME=$interfaces \
    srun -N $SLURM_NNODES --exclusive --open-mode=append --cpus-per-task=8 \
ppython run.py \
    'dataset.batch_size=32' \
    'run_type=eval' \
    'checkpoint_path="/data/soyeonhong/nlq/nlq_lightning/outputs/debug/2024-12-09/154606/epoch=0-val_R1_03=0.000.ckpt"'