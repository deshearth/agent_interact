#!/bin/bash -l
#SBATCH --job-name=agent-handoff
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=handoff-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ $# -ne 1 ]]; then
    echo "Pass the expected SHA-256 of hpc3/handoff_snapshot.tar.gz" >&2
    exit 2
fi
python3 - "$1" <<'PY'
import hashlib, sys
from pathlib import Path
h = hashlib.sha256()
with Path('hpc3/handoff_snapshot.tar.gz').open('rb') as stream:
    for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
if h.hexdigest() != sys.argv[1]: raise SystemExit('Handoff archive checksum mismatch')
PY
mkdir -p .hpc3/handoff_staging
tar -xzf hpc3/handoff_snapshot.tar.gz -C .hpc3/handoff_staging
source hpc3/container.sh
hpc3_container /workspace/.hpc3/venv/bin/python scripts/hpc3_promote_handoff.py \
    --source .hpc3/handoff_staging/hpc3_handoff --target /workspace
hpc3_container /workspace/.hpc3/venv/bin/python scripts/hpc3_breadth.py verify
