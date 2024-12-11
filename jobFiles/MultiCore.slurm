#!/bin/bash
#SBATCH --job-name=SingleCore
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=output-%j.out
#SBATCH --error=error-%j.out
#SBATCH -A class-dsci2022

# Load modules
module load python/3.10.8
ml miniconda3/24.3.0
# Activate your Python environment if needed
source activate lab_env

# Run the Python script
python reggression2.py
