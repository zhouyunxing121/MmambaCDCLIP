#!/bin/bash
#SBATCH -o job_install.%j.out
#SBATCH --partition=gpu
#SBATCH -J install_job
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=01:00:00

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8


/home/wangshiying_group/wangshiying/data/miniconda3/bin/conda activate base


conda activate changeclip

python tools/general/write_path.py