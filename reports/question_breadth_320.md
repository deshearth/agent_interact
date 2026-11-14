# Expansion to 320 training / 80 validation questions

Submitted to HPC3 on **November 10, 2026, at 4:53 p.m. Central**. This expands the number of distinct questions, preserving the existing 80 training / 20 validation questions and adding **240 training / 60 validation conversations**. The original 40 assessment questions remain reserved, ungenerated and unscored. The separate 20-question unseen-graph experiment continues unchanged.

## Selection and comparability

The previously rejected 420 candidates have mean pooled initial-response entropy of 0.760 bits, compared with 0.992 for the existing development cohort. Reusing 300 of them would change the question population substantially. Instead, the expansion constructs **1,200 fresh candidates**, excluding all previous 560 candidates, the 40 released objective questions, and the 32 persona examples.

The original rational-answer eligibility and distractor rules are retained. Number-masked templates and character-ngram cosine similarity of at least 0.90 exclude approximate duplicates across the prior and new candidate banks. These text checks do not guarantee semantic independence. The verified MATH source contains 8,114 eligible fresh candidates after initial filters; the final candidate bank has 1,200 unique IDs and family keys. The maximum retained similarity to previously accepted/excluded material is 0.89777.

Correct A/B placement and topic quotas are assigned before screening. Each topic/answer cell contains exactly four fresh candidates per added selected question. Screening uses 32 personas with eight independent initial votes each; pooled response entropy ranks candidates within each cell, with question ID breaking ties. Conversation votes are separate fresh draws. Interaction outcomes and predictor errors never influence selection.

All original split assignments are retained. The added 240/60 assignments balance difficulty level and screening entropy through a fixed, predeclared search over 2,000 seeded splits. The final development cohort has equal A/B correct labels, near-equal topic representation, and balanced familiar graphs:

| Split | Retained | Added | Final | Questions per graph, J0–J7 |
| --- | ---: | ---: | ---: | ---: |
| Training | 80 | 240 | 320 | 40 |
| Validation | 20 | 60 | 80 | 10 |
| Reserved assessment | 40 | 0 | 40 | Not generated |

Before conversations start, the newly selected mean pooled entropy must be at least the old development mean minus 0.02 bits, and its mean within-agent entropy must be at least the old mean minus 0.10 bits. A failed quality gate stops dependent collection for diagnosis. The same selection fraction reduces an easier-question confound but does not guarantee matched observed difficulty or entropy.

Each conversation uses the unchanged pinned Llama 3.1-8B MLX source, float32 floating parameters, temperature 0.7, 32 agents, eight updates, and five fresh votes per opinion. Every GPU allocation must pass the original five-group numerical replay. All source samples retain explicit per-request RNG identities.

## Execution

Eight GPU workers are requested per phase. Each uses one A30 in the `free-gpu` partition with a 48-hour allocation and a 47.5-hour generation budget. Screening workers each own 150 candidates; collection workers each own 30 new training questions and seven or eight new validation questions. Phase dependencies prevent collection GPUs from sitting idle while screening finishes.

| Phase | Jobs | Dependency |
| --- | --- | --- |
| Screen fresh candidates | 59434102–59434109 | None |
| Select and assign splits, free CPU | 59434110 | All eight screening workers succeed |
| Generate added conversations | 59434111–59434118 | Selection and entropy gate succeed |
| Audit complete dataset, free CPU | 59434119 | All eight collection workers succeed |

At the 4:54 p.m. Central snapshot, screening worker 0 was running its numerical replay and the other seven were queued for resources. Later stages were pending dependencies. The existing four unseen-graph workers were still running. This is a submission snapshot, not a completion claim; see the saved status or live scheduler for current progress.

Saved request groups support manual restart after preemption or a time limit. Automatic resubmission and chat notifications are not configured. An unsuccessful upstream job cancels its dependent jobs, so resuming a stopped phase requires rebuilding the remaining dependencies. Predictor retraining is not part of this collection chain.

Six focused checks passed for retained split assignments, topic/label/graph balancing, disjoint worker ownership, construction-cache reuse, source-runtime identity on restart, and mixed old/new episode provenance. Both Slurm scripts pass shell syntax checks. Preparation also revalidated the previous frozen model/learning-curve artifacts.

## Locations and status commands

Configuration: `configs/question_breadth_320.yaml`. Local metadata: `runs/question_breadth_320_v1/`. Remote data:

```text
/data/homezvol3/yifengc7/agent_interact_hpc3_20261109/hpc3_benchmark_full/runs/question_breadth_320_v1/
```

After logging in to HPC3:

```bash
squeue -u yifengc7 -o '%.12i %.22j %.10T %.10M %R'
cd /data/homezvol3/yifengc7/agent_interact_hpc3_20261109/hpc3_benchmark_full
python3 scripts/hpc3_breadth_expand.py status
```

Completion requires `collection_complete.json`, which is written only after all 400 development episodes pass the integrity audit. It records zero assessment use. `episode_index.json` records the correct source-protocol checksum separately for retained and new episodes; the old single-protocol training loader must not be used unchanged on this mixed-origin corpus.

The future comparison remains the 818-parameter transformer versus the 19-coefficient Ising baseline. Report the existing 20 validation questions as a common comparison anchor, the 60 added validation questions separately, and all 80 together. A question-count comparison must evaluate all training sizes on a common validation cohort. No new predictive accuracy result exists yet.

The precollection protocol SHA-256 is `87a36085b1fdcbc0a1cde5d29db2ffeb0e7d52874be191117b3355872ae2ccbc`. The upload manifest, deployment, submission record, and test report retain the implementation and data checksums.
