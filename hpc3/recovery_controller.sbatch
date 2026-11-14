#!/bin/bash -l
#SBATCH --job-name=ag-auto-control
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=auto-control-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ "$1" == "controller" ]]; then
    # Slurm commands must use the host installation and credentials.
    /usr/bin/python3 scripts/hpc3_recovery.py controller
else
    source hpc3/container.sh
    export HPC3_USE_GPU=0
    hpc3_container /workspace/.hpc3/venv/bin/python scripts/hpc3_recovery.py "$1"
fi
