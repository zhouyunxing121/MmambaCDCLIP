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

source /online1/wangshiying_group/wangshiying/miniconda3/etc/profile.d/conda.sh
conda activate changeclip


export PYTHONPATH=/online1/wangshiying_group/wangshiying/gjx/clip3-2:$PYTHONPATH



# bash tools/dist_train.sh configs/0cd_ce/changeclip_sysu.py 2 --work-dir work_dirs/changeclip_sysu
# bash tools/test.sh SYSU configs/0cd_ce/changeclip_sysu.py 2 work_dirs/changeclip_sysu

# bash tools/dist_train.sh configs/0cd_ce/changeclip_sysu_vit.py 2 --work-dir work_dirs/changeclip_sysu_vit
# bash tools/test.sh SYSU configs/0cd_ce/changeclip_sysu_vit.py 2 work_dirs/changeclip_sysu_vit

# bash tools/dist_train.sh configs/0cd_ce/changeclip_whucd.py 2 --work-dir work_dirs/changeclip_whucd
# bash tools/test.sh WHUCD configs/0cd_ce/changeclip_whucd.py 2 work_dirs/changeclip_whucd

# bash tools/dist_train.sh configs/0cd_ce/changeclip_whucd_vit.py 2 --work-dir work_dirs/changeclip_whucd_vit
# bash tools/test.sh WHUCD configs/0cd_ce/changeclip_whucd_vit.py 2 work_dirs/changeclip_whucd_vit


bash tools/dist_train.sh configs/0cd_ce/changeclip_levir.py 4 --work-dir work_dirs/changeclip_levir2
bash tools/test.sh LEVIR configs/0cd_ce/changeclip_levir.py 4 work_dirs/changeclip_levir2

bash tools/dist_train.sh configs/0cd_ce/changeclip_levir_vit.py 4 --work-dir work_dirs/changeclip_levir_vit2
bash tools/test.sh LEVIR configs/0cd_ce/changeclip_levir_vit.py 4 work_dirs/changeclip_levir_vit2

# bash tools/dist_train.sh configs/0cd_ce/changeclip_levirplus.py 2 --work-dir work_dirs/changeclip_levirplus
# bash tools/test.sh LEVIRPLUS configs/0cd_ce/changeclip_levirplus.py 2 work_dirs/changeclip_levirplus

# bash tools/dist_train.sh configs/0cd_ce/changeclip_levirplus_vit.py 2 --work-dir work_dirs/changeclip_levirplus_vit
# bash tools/test.sh LEVIRPLUS configs/0cd_ce/changeclip_levirplus_vit.py 2 work_dirs/changeclip_levirplus_vit

# bash tools/dist_train.sh configs/0cd_ce/changeclip_cdd.py 2 --work-dir work_dirs/changeclip_cdd
# bash tools/test.sh CDD configs/0cd_ce/changeclip_cdd.py 2 work_dirs/changeclip_cdd

# bash tools/dist_train.sh configs/0cd_ce/changeclip_cdd_vit.py 2 --work-dir work_dirs/changeclip_cdd_vit
# bash tools/test.sh CDD configs/0cd_ce/changeclip_cdd_vit.py 2 work_dirs/changeclip_cdd_vit