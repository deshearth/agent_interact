#!/bin/bash -l
#SBATCH --job-name=ag3200-control
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=scale-control-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ "$1" == "controller" ]]; then
    /usr/bin/python3 scripts/hpc3_scale_controller.py controller
elif [[ "$1" == "prepare" ]]; then
    source hpc3/container.sh
    export HPC3_USE_GPU=0
    hpc3_container /workspace/.hpc3/venv/bin/python scripts/hpc3_scale_worker.py prepare
    /usr/bin/python3 scripts/hpc3_scale_controller.py prepare --cron-host "$HPC3_CRON_HOST"
else
    source hpc3/container.sh
    export HPC3_USE_GPU=0
    hpc3_container /workspace/.hpc3/venv/bin/python scripts/breadth_scale.py "$1"
fi
