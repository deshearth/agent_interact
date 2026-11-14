#!/bin/bash -l
#SBATCH --job-name=agent-replay
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=benchmark-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc3/container.sh
export HPC3_USE_GPU=1
hpc3_container /workspace/.hpc3/venv/bin/python scripts/hpc3_breadth.py benchmark
