#!/bin/bash
#SBATCH --job-name=cnn_train_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/gpu_train_%j.out
#SBATCH --error=logs/gpu_train_%j.err

# Load modules
module load class/default
module load cs137/2026spring

# Activate the virtual environment
source /cluster/tufts/c26sp1cs0137/cgolem01/venv/bin/activate

# Run code
python /cluster/tufts/c26sp1cs0137/cgolem01/train.py --use_residual