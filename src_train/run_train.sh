#!/bin/bash
#SBATCH --account=asignal
#SBATCH --job-name=s_res_pytorch
#SBATCH --output=output_train.txt
#SBATCH --error=error_train.txt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:v100:1

module load pytorch
#module load cuda/11.0  # or the appropriate CUDA version
#module load cudnn/8.0  # if required
#module load scipy

cd /scratch/asignal/silaevmi/supervised_upsampling_pytorch

python3 -m venv --system-site-packages venv

source venv/bin/activate

pip install librosa
pip install matplotlib

cd /scratch/asignal/silaevmi/supervised_upsampling_pytorch/src

make run_training

deactivate