#!/bin/bash
#SBATCH --job-name=saliency
#SBATCH --output=/cluster/tufts/c26sp1cs0137/cgolem01/logs/saliency_%j.out
#SBATCH --error=/cluster/tufts/c26sp1cs0137/cgolem01/logs/saliency_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

cd /cluster/tufts/c26sp1cs0137/cgolem01
source venv/bin/activate
python3 saliency_analysis.py
