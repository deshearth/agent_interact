#!/bin/bash -l
#SBATCH --job-name=agent-expand-cpu
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=expand-cpu-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch hpc3/breadth_expand_finish.sbatch select|finalize" >&2
    exit 2
fi
source hpc3/container.sh
export HPC3_USE_GPU=0
hpc3_container /workspace/.hpc3/venv/bin/python scripts/breadth_expand.py "$1"
