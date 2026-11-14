# Four-worker source collection

This is an orchestration extension to the frozen HPC3 handoff. It uses the same
MLX source, Llama weights, prompts, graph assignments, sample IDs, and A30 replay
report. The original source and submission files remain unchanged.

The fixed 80 training and 20 validation questions are assigned round-robin within
each split: four workers, each with 20 training and 5 validation questions. Each
worker has one A30 GPU, 4 CPU cores, 32 GB host memory, a 48-hour Slurm limit, and
a 47.5-hour collection budget. All jobs use `free-gpu`. The 40 assessment questions
are excluded. Predictor training is a later step on the Mac.

`hpc3/parallel_runs/<run-id>/plan.json` records question ownership, preserved file
hashes, and the orchestration/source/replay hashes. Worker state, phase checkpoints,
compilation cache, and per-allocation replay results live under `worker_00/` through
`worker_03/`. Each allocation must match the validated runtime and replay all five
reference groups exactly before collecting. A runtime or replay mismatch stops it.

Saved request chunks and complete episodes are published into the original
`runs/question_breadth_v1/chunks/` and `runs/question_breadth_v1/episodes/` directories
using atomic, no-overwrite hard links. An existing different file is an error.
Model weights are shared; they are not copied per worker. New inference never runs
on the login node.

The old serial collector must stop before preparing a plan. Shared exclusion locks
prevent it from running alongside the four workers. Preparing a new plan requires
all current workers to be stopped. A separate exclusive lock prevents two jobs from
running the same worker simultaneously.

To check job state after logging in:

```bash
squeue -u "$USER" -o '%.12i %.20j %.10T %.10M %R'
```

From the recorded cluster workspace, inspect saved progress without loading a model:

```bash
python3 scripts/hpc3_parallel_collect.py status --plan hpc3/parallel_runs/RUN_ID/plan.json
```

The status command reports the last saved worker state; Slurm is authoritative for
whether a job is currently running. If a job disappears from `squeue`, use `sacct`
to distinguish completion from preemption, timeout, or failure:

```bash
sacct -j JOB_IDS --format=JobID,JobName,State,Elapsed,ExitCode
```

After a preemption or time limit, resubmit only the affected worker with the same
plan and worker index. It reuses every saved group, including partial conversations:

```bash
sbatch hpc3/parallel_collect.sbatch hpc3/parallel_runs/RUN_ID/plan.json WORKER_INDEX
```

There is no automatic retry or automatic predictor training. The Mac generator stays
paused. Training/validation collection is complete when all four workers have
finished their assignments and the canonical episode counts are 80/20/0.

Preparation, before releasing held workers:

```bash
python3 scripts/hpc3_parallel_collect.py prepare --run-id RUN_ID --workers 4 \
  --benchmark-report hpc3/results/benchmark_59409953/report.json
```

The preparation command verifies the stopped handoff, existing cached outputs and
replay report, preserves cached conversation groups, and refuses existing assessment
interaction outputs. It does not load a model or generate responses.
