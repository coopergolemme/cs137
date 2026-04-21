#!/bin/bash
#SBATCH --job-name=cnn_train_cpu
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/cpu_train_%j.out
#SBATCH --error=logs/cpu_train_%j.err

# Load modules
module load class/default
module load cs137/2026spring

# Activate the virtual environment
source /cluster/tufts/c26sp1cs0137/cgolem01/venv/bin/activate

# Run code
python /cluster/tufts/c26sp1cs0137/cgolem01/train.py