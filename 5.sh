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


# 注意：这里调用 dist_train.sh，并传入 GPU 数量为 1
bash /online1/wangshiying_group/wangshiying/gjx/clip3-2/tools/dist_train.sh \
    configs/0cd_ce/changeclip_clcd.py 1 \
    --work-dir work_dirs/changeclip_clcd \
    > job.${JOB_ID}.train.log 2>&1

# 5. 测试阶段（如果训练成功）
if [ $? -eq 0 ]; then
    echo "Training finished successfully. Starting testing..."
    # GPU数为1
    bash /online1/wangshiying_group/wangshiying/gjx/clip3-2/tools/test.sh \
        CLCD configs/0cd_ce/changeclip_clcd.py 1 work_dirs/changeclip_clcd \
        --work-dir work_dirs/changeclip_clcd \
        > job.${JOB_ID}.test.log 2>&1
else
    echo "Training failed, skipping testing."
fi