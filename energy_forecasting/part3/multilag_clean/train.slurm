#!/bin/bash
#SBATCH --job-name=multilag_clean
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/cluster/tufts/c26sp1cs0137/kkapli01/logs/%j.out
#SBATCH --error=/cluster/tufts/c26sp1cs0137/kkapli01/logs/%j.err

cd /cluster/tufts/c26sp1cs0137/kkapli01/multilag_clean
$HOME/.conda/envs/kai_env/bin/python -u train.py
