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
python /home/wangshiying/gjx/ChangeCLIP/tools/clip_inference.py --src_path /home/wangshiying/gjx/rschange/data/DSIFN \
                        --split train val test \
                        --img_split t1 t2 \
                        --model_name ViT-B/16 \
                        --class_names_path /home/wangshiying/gjx/ChangeCLIP/tools/rscls.txt \
                        --device cuda:0 \
                        --tag 56_vit16
