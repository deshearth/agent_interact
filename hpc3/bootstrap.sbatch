#!/bin/bash -l
#SBATCH --job-name=agent-setup
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=setup-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc3/container.sh

if [[ ! -f "$HPC3_IMAGE" ]]; then
    # An immutable amd64 manifest for the official Python image; no root build.
    singularity pull --force "$HPC3_IMAGE.tmp" \
        docker://python@sha256:06bb5357fbe87fed6fd7793c8c525ba09417bb23361b161511bb2993e8f0a7e3
    mv "$HPC3_IMAGE.tmp" "$HPC3_IMAGE"
fi
hpc3_container python -m venv /workspace/.hpc3/venv
hpc3_container /workspace/.hpc3/venv/bin/python -m pip install \
    --disable-pip-version-check --only-binary=:all: --require-hashes -r hpc3/requirements-cuda12.txt
hpc3_container /workspace/.hpc3/venv/bin/python -m pip check
hpc3_container /workspace/.hpc3/venv/bin/python - <<'PY'
import hashlib, importlib.metadata, json, platform
from pathlib import Path
image = Path('.hpc3/python.sif')
h = hashlib.sha256()
with image.open('rb') as stream:
    for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
value = {'container_sha256': h.hexdigest(), 'python': platform.python_version(),
         'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
Path('.hpc3/environment.json').write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
PY
hpc3_container /workspace/.hpc3/venv/bin/python scripts/hpc3_breadth.py download-model
