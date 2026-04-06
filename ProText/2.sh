#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a800x4

#SBATCH -J g_job_1
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
source /home/dc001/miniconda3/bin/activate changeclip
sh scripts/protext/fully_supervised_and_dg.sh levir_cd output/levir_cd_protext --no-dg