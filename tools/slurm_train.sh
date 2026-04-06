#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a800x4
#SBATCH --nodelist=gpu_03  
#SBATCH -J pytorch_job_1
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
source activate rscd_mamba

python -u tools/train.py configs/0cd_ce/changeclip_sysu.py --work-dir work_dirs/changeclip_sysu --launcher="slurm" 
python -u tools/test.sh SYSU configs/0cd_ce/changeclip_sysu.py 2 work_dirs/changeclip_sysu