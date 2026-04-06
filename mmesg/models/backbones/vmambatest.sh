#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a800x4
#SBATCH --nodelist=gpu_03 
#SBATCH -J pytorch_job_1
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --qos=normal

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

source /home/dc001/miniconda3/bin/activate changeclip
python vmambatest.py