#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a800x4

#SBATCH -J g_job_1
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --qos=normal

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

source /home/dc001/miniconda3/bin/activate changeclip
export PYTHONPATH="/home/dc001/clip3-2:$PYTHONPATH"

JOB_ID=${SLURM_JOB_ID}

bash /home/dc001/clip3-2/tools/dist_train.sh configs/0cd_ce/pureclip_clcd.py 2 --work-dir work_dirs/pureclip_clcd > job.${JOB_ID}.train.log 2>&1


if [ $? -eq 0 ]; then
    
    bash /home/dc001/clip3-2/tools/test.sh CLCD configs/0cd_ce/pureclip_clcd.py 2 work_dirs/pureclip_clcd --work-dir work_dirs/pureclip_clcd > job.${JOB_ID}.test.log 2>&1
else
    echo "Training failed, skipping testing. See job.${JOB_ID}.train.log for details."
fi