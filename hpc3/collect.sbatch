#!/bin/bash -l
#SBATCH --job-name=agent-collect
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=collect-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch hpc3/collect.sbatch train|validation|assessment hpc3/results/benchmark_JOB/report.json" >&2
    exit 2
fi
source hpc3/container.sh
export HPC3_USE_GPU=1
hpc3_container /workspace/.hpc3/venv/bin/python scripts/hpc3_breadth.py collect \
    --split "$1" --benchmark-report "$2" --hours 23.5
