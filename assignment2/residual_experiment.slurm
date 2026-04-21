#!/bin/bash
#SBATCH --job-name=reslink_study
#SBATCH --output=logs/reslink_gpu_%j.out
#SBATCH --error=logs/reslink_gpu_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

module load class/default
module load cs137/2026spring

source /cluster/tufts/c26sp1cs0137/cgolem01/venv/bin/activate

python run_residual_experiment.py
