# HPC3 submission: Llama conversation generation

Prepared for `yifengc7`, initially using the **free** CPU and GPU queues. The first
GPU job is a benchmark and numerical replay, not production collection. The Mac
collector continues during preparation. No API calls or paid queue submissions
are part of these scripts.

The scientific comparison remains 80 training / 20 validation / 40 assessment
questions, 818 transformer parameters versus the 19-coefficient Ising baseline.
HPC3 is for source generation; predictor fitting can remain on the Mac.

## Connect and transfer

Connect this Mac to UCI VPN, or use the campus network. Log in from Terminal and
complete the password/Duo prompts there:

```sh
ssh yifengc7@hpc3.rcic.uci.edu
```

Check `sbank`, `sinfo -p free-gpu -o '%P %a %l %D %G'`, and
`module avail singularity`. A positive personal account balance is needed for
free-queue access. The actual available GPU, driver and module versions must be
checked remotely; a local package check cannot establish them.

From a local Terminal, transfer the prepared archive using UCI's transfer host:

```sh
scp runs/hpc3_submission_v1/hpc3_benchmark.tar.gz yifengc7@access-hpc3.rcic.uci.edu:/pub/yifengc7/
```

On HPC3, extract into the private `/pub/yifengc7` area (or a confirmed private DFS
directory with enough room), not a scratch directory that is erased after each
job. Allow about 15 GB initially for model weights, CUDA libraries, image, caches
and outputs; verify actual quota before setup.

```sh
cd /pub/yifengc7
tar -xzf hpc3_benchmark.tar.gz
cd hpc3_benchmark
```

The archive excludes credentials, home-directory files, virtual environments and
the 4.53 GB model. Setup downloads the exact pinned public checkpoint and verifies
every recorded model-file SHA-256. Keep `hpc3/manifest.json` with the package.

## First submissions

```sh
HPC3_SETUP_JOB=$(sbatch --parsable hpc3/bootstrap.sbatch)
sbatch --dependency=afterok:${HPC3_SETUP_JOB%%;*} hpc3/benchmark.sbatch
squeue -u yifengc7
```

Setup runs on four CPU cores in `free`, with 16 GB RAM and a two-hour limit. It
pulls an immutable amd64 official Python container, installs the pinned CUDA
environment outside the read-only image, and downloads/verifies the model. No
root or fakeroot setup is needed. Re-submit setup if it was preempted. A GPU job
held by `afterok` after failed setup will need cancelling and resubmission with
the successful setup job ID.

The benchmark requests one **A100**, four CPU cores, 32 GB host memory and 45
minutes in `free-gpu`. V100 is intentionally excluded: MLX requires compute
capability at least 7.5. An L40S alternative, after checking queue availability,
is `sbatch -p free-gpu32 --gres=gpu:L40S:1 hpc3/benchmark.sbatch`.

Use `sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode` to inspect completion.
Read `setup-JOB_ID.log`, `benchmark-JOB_ID.log` and
`hpc3/results/benchmark_JOB_ID/report.json`. No background monitor is installed.

## What is held fixed and checked

- Same Llama-3.1-8B MLX 4-bit checkpoint and tokenizer, with float32 floating
  parameters/activations; same MLX 0.32.4 and MLX-LM 0.32.2.
- Same prompts, chat-template date, independent per-request/per-token RNG keys,
  temperature 0.7, full-vocabulary sampling, EOS handling and parse retries.
- Original collection, formatter, message delivery and signed-graph code remain
  byte-identical. Old absolute paths are resolved through the export manifest;
  original provenance files are not rewritten.
- `MLX_ENABLE_TF32=0` disables reduced-precision float32 CUDA operations. CUDA
  library and container hashes, GPU type and driver are recorded.
- Replay tests batches 4, 8 and 16. It requires exact prompt tokenization,
  **entire generated token sequences**, termination reasons, reproducibility on
  repeat, and first-token A/B/other probabilities within 0.001 of the Mac.
  The fastest passing tested batch is selected using warmed timings.
- Production additionally requires actual train-conversation prompts containing
  neighbor arguments for both message and opinion generation. An early package
  without those references can benchmark setup but cannot unlock collection.
  Re-export after those Mac chunks exist; do not weaken the gate.

Passing this finite regression check does not prove identical distributions on
every future prompt. Every new CUDA chunk carries the execution provenance; the
hardware migration must be disclosed in the final study report. If replay fails,
retain the result and investigate the backend. Do not mix a different checkpoint,
quantization, dtype, serving engine or failed-replay output into the frozen study.

## Production handoff after a successful benchmark

1. Review benchmark correctness, warmed speed, memory and queue conditions.
   Do not stop a useful local collector before this evidence exists.
2. Stop the local background supervisor, its pipeline and source worker in a
   controlled handoff, preserving every completed atomic request group. The
   exporter refuses handoff while any of the three collection locks is held.
   Do not run both collectors on the same study after ownership changes.
3. From the Mac repository, make an authoritative snapshot:

   ```sh
   .venv/bin/python scripts/hpc3_prepare.py --mode handoff --output runs/hpc3_submission_v1/hpc3_handoff
   ```

4. Transfer/extract that new package. It includes frozen selection, completed
   screening, raw request chunks and completed episodes. Reuse the existing
   `.hpc3` environment/model files at the same container paths, or run setup
   again. Run the benchmark in the new package so all hashes and fixtures match.
5. Once that package's report says `numerical_gate_passed: true`, submit:

   ```sh
   sbatch hpc3/collect.sbatch train hpc3/results/benchmark_JOB_ID/report.json
   ```

   On preemption or a time-limit exit (75), re-submit the same command. Completed
   groups are reused with identity/hash checks; only an interrupted group is
   recomputed. Submit validation after train completion with `validation` as the
   first argument. There is no automatic paid-queue fallback or unlimited retry.
6. Transfer results back into a staging directory. Verify protocol/chunk hashes,
   retain the HPC3 execution record, and merge only missing identical-identity
   artifacts while the Mac collector is stopped. Fit/freeze predictors using the
   existing Mac pipeline. It needs the new CUDA chunks and episodes, not the
   container or remote environment.
7. Assessment remains locked until `models_frozen.json` and every hashed final
   predictor artifact exist. A later handoff can carry them for assessment
   generation. Never open assessment early just to keep a GPU busy.

The continuation script refuses a benchmark-only export, missing inbox coverage,
a failed replay, a changed GPU/software identity, changed frozen files, or an
unfrozen assessment split. This package does not itself stop local processes or
merge remote results.

## Sources checked during preparation

- [HPC3 login and VPN](https://rcic.uci.edu/account/login.html)
- [HPC3 data transfers](https://rcic.uci.edu/storage/data-transfer.html)
- [Free queues and GPU accounts](https://rcic.uci.edu/slurm/slurm.html)
- [GPU submission examples](https://rcic.uci.edu/slurm/examples.html)
- [HPC3 software and OS](https://rcic.uci.edu/software/software.html)
- [Container setup on compute nodes](https://rcic.uci.edu/software/user-installed.html#create-singularity-containers)
- [Pinned MLX CUDA installation requirements](https://github.com/ml-explore/mlx/blob/v0.32.4/docs/src/install.rst)
- [MLX float32 precision controls](https://github.com/ml-explore/mlx/blob/v0.32.4/docs/src/usage/precision.rst)
- [MLX lookup of wheel-provided CUDA headers](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/cuda/jit_module.cpp)
