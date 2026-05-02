#!/bin/bash
#SBATCH --job-name=part3_prep
#SBATCH --partition=batch
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/cluster/tufts/c26sp1cs0137/kkapli01/logs/%j.out
#SBATCH --error=/cluster/tufts/c26sp1cs0137/kkapli01/logs/%j.err

cd /cluster/tufts/c26sp1cs0137/kkapli01/part3
$HOME/.conda/envs/kai_env/bin/python prep_cluster.py
