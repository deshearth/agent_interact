#!/bin/bash
# Source from a Slurm script after changing to the exported package directory.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run setup and computation through Slurm, not on a login node." >&2
    exit 2
fi
module load singularity/3.11.3
HPC3_WORKSPACE=$(pwd -P)
HPC3_IMAGE="$HPC3_WORKSPACE/.hpc3/python.sif"
export SINGULARITY_CACHEDIR="$HPC3_WORKSPACE/.hpc3/image-cache"
mkdir -p "$SINGULARITY_CACHEDIR" "$HPC3_WORKSPACE/.hpc3/cache"

hpc3_container() {
    local gpu_args=()
    if [[ "${HPC3_USE_GPU:-0}" == "1" ]]; then
        if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            echo "Slurm has not exposed a GPU to this job." >&2
            return 2
        fi
        gpu_args=(--nv --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
    fi
    singularity exec --cleanenv "${gpu_args[@]}" \
        --bind "$HPC3_WORKSPACE:/workspace" --pwd /workspace \
        --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
        --env MLX_ENABLE_TF32=0 --env MLX_PTX_CACHE_DIR=/workspace/.hpc3/cache/mlx \
        --env HF_HOME=/workspace/.hpc3/cache/huggingface \
        --env PIP_CACHE_DIR=/workspace/.hpc3/cache/pip \
        --env HF_HUB_DISABLE_TELEMETRY=1 --env TOKENIZERS_PARALLELISM=false \
        --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
        --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
        "$HPC3_IMAGE" "$@"
}
