# Unseen-graph generalization: paired Llama experiment

**Complete.** The original 80-question predictors were scored automatically on November 13, 2026, at 4:01 p.m. Central. The new, already-frozen **320-question predictors** were evaluated on the same completed conversations on November 14 at 1:33 a.m.; independent verification finished at 1:36 a.m. No predictor or PCA was refitted for this follow-up, and no additional source conversations were generated.

## Results on matched questions

These comparisons use **the same 20 validation questions**, each with one familiar-graph and one unseen-graph conversation. There are **5,120 agent transitions per graph regime**. Transformer scores average seeds 142/143/144; no predictions are ensembled. The familiar-graph figures below cover these 20 questions, so they differ from the preceding expanded experiment's overall 80-question validation scores.

| Training questions | Predictor | Familiar-graph accuracy | Unseen-graph accuracy | Unseen NLL |
| ---: | --- | ---: | ---: | ---: |
| 80 | Ising, 19 coefficients | 83.32% | 81.23% | 0.39232 |
| 80 | Transformer, 818 parameters | 84.89% | 83.91% | 0.35999 |
| 320 | Ising, 19 coefficients | 83.05% | 81.37% | 0.39111 |
| 320 | Transformer, 818 parameters | 85.31% | **84.35%** | **0.35486** |

For the **320-question models**, the transformer has a **2.98 percentage-point unseen-graph accuracy advantage** over Ising, with a descriptive paired question-bootstrap interval of **+1.18 to +4.98 points**. Its NLL advantage is **-0.03626**, interval **-0.04470 to -0.02786**. Both intervals are conditional on these four graphs and this question assignment.

The transformer's familiar-to-unseen accuracy change is **-0.96 pp** (interval **-3.55 to +1.50**); Ising's is **-1.68 pp** (**-5.08 to +1.37**). Although the observed drop is smaller for the transformer, the paired difference between those drops is **+0.72 pp**, interval **-0.71 to +2.43**. This small study does not establish a difference in the magnitude of degradation.

Increasing training from 80 to 320 questions raises unseen-graph transformer accuracy by **+0.44 pp**, interval **-0.15 to +1.00**. Its NLL improves by **0.00513**, interval **-0.01015 to -0.00017** for new-minus-old loss. Ising's accuracy changes by **+0.14 pp**, interval **-0.21 to +0.64**. The larger-data accuracy benefit remains uncertain on this 20-question cohort; model selection and training-only PCA also differ between the two training studies.

## Differences across the four unseen graphs

| Graph | Questions | Ising320 accuracy | Transformer320 accuracy | Transformer advantage |
| --- | ---: | ---: | ---: | ---: |
| Jf0 | 5 | 85.31% | 85.78% | +0.47 pp |
| Jf1 | 5 | 82.73% | 85.47% | +2.73 pp |
| Jf2 | 5 | 84.14% | 85.16% | +1.02 pp |
| Jf3 | 5 | 73.28% | 80.99% | +7.71 pp |

Jf3 contributes about **65% of the overall accuracy advantage**; on the other three graphs combined, the advantage is **1.41 pp**. Each graph is assigned a different set of five questions, so these differences cannot be attributed to a graph property alone. This is an exploratory breakdown. The same limitation applies to reading the overall result as evidence about arbitrary future graphs.

The logged full-prompt sampling-reference error averages **8.58% on familiar graphs and 8.52% on unseen graphs**. Its average does not increase alongside the observed model errors. This is an approximate source-probability diagnostic that includes message information unavailable to the predictors; it is not an attainable error floor for their input state.

## Saved results and verification

Original frozen source/score artifacts remain in `runs/unseen_graph_v1/`. The new follow-up is separate: [evaluation protocol](../runs/unseen_graph_320_v1/protocol.json), [results, intervals and graph breakdown](../runs/unseen_graph_320_v1/results.json), and [completion audit](../runs/unseen_graph_320_v1/completion_audit.json). It uses [a separate evaluator](../scripts/evaluate_expanded_unseen_graph.py); run or verify it with `.venv/bin/python scripts/evaluate_expanded_unseen_graph.py`.

All 20 source episodes and three original scoring artifacts match their recorded cluster checksums. Three focused checks verify seed aggregation, question pairing and the graph-effect contrast. Independent calculations reproduce all aggregate new scores. PyTorch and the independent NumPy adapter agree on every predicted class across both graph regimes and all three seeds; their maximum probability difference is **4.91e-7**. Original graph-study scores and both models' familiar-question scores reproduce. All **95 original and 93 expanded frozen model artifacts** are unchanged. All **40 reserved assessment conversations remain ungenerated and unscored**.

The source experiment originally specified the **80-question frozen 818-parameter transformer** and **19-coefficient Ising** on four random graphs absent from training. The new comparison adds the 320-question models with each model set's own weights, context transform and decision thresholds held fixed. Probability-target variants are excluded.

## Paired design

- Use the same 20 validation questions as the completed familiar-graph comparison. Each receives one additional conversation, assigned by a fixed seed to Jf0–Jf3, five questions per graph. Each question therefore has its original familiar-graph conversation and one fresh-graph conversation.
- Reuse all five original round-zero votes for each of the 32 agents, retaining the original raw responses, token/probability records, sample identities and random-seed provenance. Prompt hashes must match. The new identifiers explicitly record this reuse; these are not newly generated initial responses.
- Generate eight subsequent message-exchange/opinion-update rounds with the same pinned Llama 3.1-8B MLX weights, float32 execution, prompts, temperature 0.7 and five-vote majority rule. Later random draws have new graph-specific identities. Thus initial opinions are paired, while later stochastic trajectories differ.
- Source generation uses the existing validated container/runtime and numerical replay gate. Every GPU allocation must reproduce all five reference groups before generating. Original source code, data and predictor artifacts are preserved.
- Generate **20 new episodes / 5,120 agent transitions**. All **40 reserved assessment questions remain ungenerated and unscored**. No paid model API is used, and the Mac source generator stays paused.

Each fresh graph has 32 nodes and 56 undirected signed edges, as do the training random graphs:

| Graph | Friendly edges | Unfriendly edges | Isolated agents | Questions |
| --- | ---: | ---: | ---: | ---: |
| Jf0 | 30 | 26 | 2 | 5 |
| Jf1 | 27 | 29 | 1 | 5 |
| Jf2 | 33 | 23 | 1 | 5 |
| Jf3 | 21 | 35 | 1 | 5 |

This tests new random graph instances within the familiar graph family. It does not test lattices, different agent counts or new persona populations. The questions have already been used for development evaluation, so this is fresh graph evidence rather than an untouched-question confirmation. One fresh graph and one stochastic trajectory per question cannot identify a deterministic causal graph effect or fully separate graph-instance difficulty from novelty.

## Scoring frozen in advance

Score every valid next opinion on the observed trajectories. Average within question and then equally across the 20 questions. Average the three transformer's model scores, not their predictions. Report:

1. Familiar versus unseen graph accuracy and NLL for each model.
2. Transformer versus Ising on exactly the same unseen episodes.
3. Per-question and per-graph breakdowns.
4. Expected error/NLL under logged source probabilities and error above the full-prompt sampling reference, to help distinguish model degradation from changed source-response uncertainty.

For probability metrics, normalize first bare A/B-token probabilities and compute the five-vote majority probability. This approximates the full parsed-response/retry process. The output records malformed answers, replacements and outside-A/B token mass. These full-prompt probabilities include messages unavailable at the original forecasting boundary and do not define a universal attainable error floor for the predictors.

Paired question-bootstrap intervals use 5,000 fixed-seed draws after averaging transformer seeds within question. Intervals are descriptive and conditional on these four graphs, the chosen assignment, and the reused validation questions. They do not treat individual agents or rounds as independent samples and do not estimate uncertainty over all possible graph populations.

## Execution and checks

Six checks passed for graph assignment/split isolation, NumPy/PyTorch inference agreement including isolated nodes, faithful initial-response reuse and prompt checks, paired-state/graph validation, the complete-collection scoring gate, and next-round probability extraction including retries. All three exported transformers reproduce every original development class prediction over 25,600 transitions; the maximum probability difference is **7.94e−7**. Ising probabilities reproduce within 1e−12 log-probability tolerance.

The cluster scorer uses the verified NumPy implementation so it needs no new ML installation and does not change the validated source runtime. After all 20 episodes were published, it automatically computed `results.json`, `predictions.npz`, `by_question.json`, and `by_seed_question.json`. No partial-cohort accuracy was reported. Source chunks were saved atomically. Automatic recovery, added on November 13 in the separate operational workflow, resumed interrupted workers; collection and scoring are now complete.

## Historical submission and status commands

The original November 10, 4:23 p.m. Central submission used four free-A30 jobs: **59433505–59433508**, five conversations each. Later recovery jobs superseded these identifiers. The commands below document that initial submission and do not describe currently running work.

Local protocol and deployment records: `runs/unseen_graph_v1/`. Cluster directory:

```text
/data/homezvol3/yifengc7/agent_interact_hpc3_20261109/hpc3_benchmark_full/runs/unseen_graph_v1/
```

After logging in to HPC3, check the scheduler:

```bash
squeue -j 59433505,59433506,59433507,59433508 -o '%.12i %.20j %.10T %.10M %R'
```

For collection/scoring progress, from the cluster workspace:

```bash
cd /data/homezvol3/yifengc7/agent_interact_hpc3_20261109/hpc3_benchmark_full
python3 scripts/hpc3_unseen_graph.py status
```

After jobs disappear from `squeue`, distinguish completion from interruption with:

```bash
sacct -j 59433505,59433506,59433507,59433508 --format=JobID,State,Elapsed,ExitCode
```

An interrupted worker can be resubmitted with the same index, preserving its saved requests:

```bash
sbatch hpc3/unseen_graph.sbatch WORKER_INDEX
```

Preparation, source orchestration and evaluation are in `scripts/unseen_graph_experiment.py` and `scripts/hpc3_unseen_graph.py`; the configuration is `configs/unseen_graph.yaml`. The original `question_breadth_v1` and `llama_probability_curve_v1` experiments remain separate.
