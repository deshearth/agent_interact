#!/bin/bash -l
#SBATCH --job-name=agent-conversations
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=conversations-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch --dependency=afterok:BENCHMARK_JOB hpc3/automatic_collect.sbatch hpc3/results/benchmark_JOB/report.json" >&2
    exit 2
fi
source hpc3/container.sh
export HPC3_USE_GPU=1
hpc3_container /workspace/.hpc3/venv/bin/python scripts/hpc3_auto_collect.py \
    --benchmark-report "$1" --hours 47.5
