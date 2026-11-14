#!/bin/bash -l
#SBATCH --job-name=ag-auto-pool
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=auto-pool-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc3/container.sh
export HPC3_USE_GPU=1
hpc3_container /workspace/.hpc3/venv/bin/python scripts/hpc3_expansion_pool.py --slot "$1" --hours 7.5
