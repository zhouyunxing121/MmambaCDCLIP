#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=gpu
#SBATCH -J g_job
#SBATCH -N 1
#SBATCH --ntasks-per-node=1      
#SBATCH --gres=gpu:1             
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8        


export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8


source /online1/wangshiying_group/wangshiying/miniconda3/etc/profile.d/conda.sh
conda activate changeclip


export PYTHONPATH=/online1/wangshiying_group/wangshiying/gjx/clip3-2:$PYTHONPATH


# 解决 GDAL 与 PyTorch/OpenCV 的 GLIBCXX 动态库冲突
# 强制系统优先使用 conda 环境内的高版本 C++ 库
export LD_PRELOAD=/home/wangshiying_group/wangshiying/data/miniconda3/envs/changeclip/lib/libstdc++.so.6


JOB_ID=${SLURM_JOB_ID}



bash /online1/wangshiying_group/wangshiying/gjx/clip3-2/tools/test.sh \
    SYSU configs/0cd_ce/changeclip_sysu.py 1 work_dirs/changeclip_sysu \
    --tta \
    --work-dir work_dirs/changeclip_sysu \
    > job.${JOB_ID}.test.log 2>&1
