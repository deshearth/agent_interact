#!/bin/bash -l
#SBATCH --job-name=agent-unseen
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=unseen-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch hpc3/unseen_graph.sbatch WORKER_INDEX" >&2
    exit 2
fi
source hpc3/container.sh
export HPC3_USE_GPU=1
hpc3_container /workspace/.hpc3/venv/bin/python scripts/hpc3_unseen_graph.py worker \
    --worker "$1" --hours 47.5
