#!/bin/bash
#SBATCH --job-name=CNN-Transformer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --output=/cluster/tufts/c26sp1cs0137/cgolem01/energy_forecasting/logs/train_%j.out
#SBATCH --error=/cluster/tufts/c26sp1cs0137/cgolem01/energy_forecasting/logs/train_%j.err

# Load modules
module load class/default
module load cs137/2026spring


# Activate the virtual environment
source /cluster/tufts/c26sp1cs0137/cgolem01/venv/bin/activate

# Run code
python /cluster/tufts/c26sp1cs0137/cgolem01/energy_forecasting/train.py

