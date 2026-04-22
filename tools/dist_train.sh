#!/bin/bash
# 【重要】删除了所有 #SBATCH 开头的行

# 环境变量可以在这里再次确认，或者依赖调用者(4.sh)传递
# 建议保留 Python 路径设置，防止子进程找不到代码
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=$(shuf -i 10000-30000 -n 1)
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 使用 torchrun 启动
# 当 GPUS=1 时，torchrun 会启动单进程，不会报错
PYTHONPATH="$PYTHONPATH" \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_changeclip.py \
    $CONFIG \
    --launcher pytorch


