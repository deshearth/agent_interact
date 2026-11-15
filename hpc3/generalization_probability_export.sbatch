#!/bin/bash -l
#SBATCH --job-name=agent-probability-audit
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=runs/generalization_investigation_320_v1/export/export-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
/usr/bin/python3 runs/generalization_investigation_320_v1/export/export_generalization_probabilities.py \
  --base . --output runs/generalization_investigation_320_v1/source_probabilities.json.gz
