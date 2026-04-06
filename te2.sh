#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a800x4  
#SBATCH -J pytorch_job_1
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --qos=normal

conda activate changeclip


which nvcc
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi

python -V  

python -c "import torch; print(torch.__version__)"

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.version.cuda)"

python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

echo "Anaconda environment: changeclip——pip list"
pip list

echo "Anaconda environment: changeclip——conda list"
conda list