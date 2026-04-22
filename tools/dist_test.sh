#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=gpu
#SBATCH -J j_job
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

source /home/wangshiying_group/wangshiying/data/miniconda3/etc/profile.d/conda.sh
conda activate changeclip

export PYTHONPATH=/online1/wangshiying_group/wangshiying/gjx/clip3-2:$PYTHONPATH

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=$(shuf -i 10000-30000 -n 1)
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch 
