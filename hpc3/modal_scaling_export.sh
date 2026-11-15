#!/bin/bash
#SBATCH --partition=free
#SBATCH --account=yifengc7
#SBATCH --job-name=modal512_export
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=runs/modal_scaling_512_v1/transfer/export-%j.log
set -euo pipefail
cd /data/homezvol3/yifengc7/agent_interact_hpc3_20261109/hpc3_benchmark_full
python3 runs/modal_scaling_512_v1/transfer/export_modal_scaling.py \
  --base . \
  --manifest runs/modal_scaling_512_v1/transfer/first192.json \
  --output runs/modal_scaling_512_v1/export
