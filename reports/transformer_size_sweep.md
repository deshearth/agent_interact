# Transformer size comparison

Completed two new transformer sizes on the existing twenty-question GPT development pool. The 34,978-parameter reference and strengthened Ising baseline are reused after identity/checksum checks. All three neural sizes have the same inputs, four attention heads, six tuning candidates, inner stopping rule and final seeds. Only attention/hidden/feedforward widths change.

These are exploratory nested whole-question results on previously studied questions, not fresh confirmation. NLL is primary. Scores average seeds within question, then weight questions equally; predictions are not ensembled.

| Model | Neural parameters | Other fitted coefficients | Training error | Outer error | Outer NLL | Outer Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ising | 0 | 19 | 9.79% | 10.95% | 0.22755 | 0.13506 |
| Ising + current-opinion correction | 0 | 19 baseline + 3 correction | 9.53% | 10.11% | 0.22474 | 0.13226 |
| Transformer 7,370 | 7,370 | 0 | 7.72% | 9.17% | 0.21472 | 0.12745 |
| Transformer 16,842 | 16,842 | 0 | 7.70% | 9.32% | 0.22024 | 0.13067 |
| Transformer 34,978 | 34,978 | 0 | 7.61% | 9.48% | 0.22321 | 0.13270 |

## Paired comparisons

Differences are first minus reference, so negative favors the first model. Intervals resample twenty paired question differences 3,000 times. They are pointwise and descriptive: only four overlapping outer training folds, prior question exposure and multiple contrasts limit inference. An interval crossing zero does not establish equivalence or noninferiority.

| Comparison | Error difference, pp [95% interval] | NLL difference [95% interval] | Questions with lower NLL |
| --- | ---: | ---: | ---: | ---: |
| Transformer 7,370 − Transformer 34,978 | -0.31 [-0.63, +0.02] | -0.00849 [-0.01463, -0.00194] | 13/20 |
| Transformer 16,842 − Transformer 34,978 | -0.16 [-0.48, +0.17] | -0.00298 [-0.00774, +0.00151] | 12/20 |
| Transformer 7,370 − Transformer 16,842 | -0.15 [-0.47, +0.18] | -0.00551 [-0.01121, +0.00027] | 15/20 |
| Transformer 7,370 − Ising + current-opinion correction | -0.94 [-1.69, -0.17] | -0.01002 [-0.02386, +0.00333] | 12/20 |
| Transformer 16,842 − Ising + current-opinion correction | -0.79 [-1.59, +0.05] | -0.00450 [-0.02099, +0.01171] | 11/20 |
| Transformer 34,978 − Ising + current-opinion correction | -0.62 [-1.43, +0.23] | -0.00153 [-0.02040, +0.01663] | 10/20 |

## Matched procedure

Four outer folds each hold out five questions; the other fifteen support three inner folds of ten fitting/five selection questions. Every question's eight graphs, four replicas, eight updates and 32 agents stay together. All sixteen training-fold question PCA and Ising fits are exactly reused from the verified earlier study. No official test question enters the new fits or scoring.

Inputs remain current opinions, 16-dimensional persona/question context, signed graph and degree features. No arguments, raw text, additional history or question IDs are added. Architectures:

| Parameters | Hidden width | Heads | Key/head | Value/head | Feedforward |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7,370 | 28 | 4 | 4 | 7 | 56 |
| 16,842 | 44 | 4 | 5 | 11 | 88 |
| 34,978 | 64 | 4 | 8 | 16 | 128 |

Every size starts with a zero output head and receives learning rates 0.0001/0.0003 crossed with logit penalties 0/0.001/0.01, dropout 0.1, AdamW decay 0.001, batch32 and clipping1. The unchanged training loop evaluates inner NLL every 128 updates, allows initialization selection, uses six-check patience and a 3072-update cap. Within each outer fold/size, average inner NLL chooses a candidate; the median selected data passes sets the outer refit length, scaled from ten to fifteen questions. Refits use seeds 142/143/144. There is no outer-label early stopping or best-seed selection.

Settings are separately selected for each size. This compares equally tuned procedures rather than holding selected optimizer settings or update counts identical. Smaller models can receive different selected training durations, within the same search policy.

The strengthened Ising benchmark uses three correction coefficients (intercept, prior-logit scaling and current own opinion) atop its nineteen fold-fitted baseline coefficients. Its earlier nested penalty grid has three choices versus six for each neural model; it is a stronger benchmark, not a matched architecture/compute ablation. Its parameterization includes intercept/scale redundancy, so this is not a minimal coefficient count.

Executed 144 new inner fits and 24 new refits; reused 72 reference inner fits and 12 reference refits. New recorded fitting time: 43.6 minutes. Inner stopping counts: {'early_stopping': 102, 'step_cap': 42}. Reused fits are not counted as newly executed work.

## Selected settings

| Outer fold | Parameters | Learning rate | Output penalty | Refit updates | Origin |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 7,370 | 0.0003 | 0.001 | 2304 | new |
| 0 | 16,842 | 0.0003 | 0.001 | 2304 | new |
| 0 | 34,978 | 0.0001 | 0.001 | 3264 | reused |
| 1 | 7,370 | 0.0003 | 0.001 | 1344 | new |
| 1 | 16,842 | 0.0003 | 0.001 | 768 | new |
| 1 | 34,978 | 0.0001 | 0.001 | 1920 | reused |
| 2 | 7,370 | 0.0003 | 0.001 | 1152 | new |
| 2 | 16,842 | 0.0003 | 0.001 | 1536 | new |
| 2 | 34,978 | 0.0003 | 0.001 | 1536 | reused |
| 3 | 7,370 | 0.0003 | 0.001 | 1536 | new |
| 3 | 16,842 | 0.0003 | 0.001 | 1536 | new |
| 3 | 34,978 | 0.0003 | 0.001 | 1344 | reused |

## Training-cap diagnostic

| Parameters | Inner fits reaching update cap | Best checkpoint at cap | Selected candidate's inner checkpoints at cap |
| ---: | ---: | ---: | ---: |
| 7,370 | 26/72 | 4/72 | 0/12 |
| 16,842 | 16/72 | 2/72 | 0/12 |
| 34,978 | 6/72 | 0/72 | 0/12 |

A selected checkpoint at the update cap indicates that the search may be budget-limited; it does not establish convergence. The cap and candidate learning rates are matched to the reference, so this study estimates performance under that common search policy rather than each architecture's fully optimized potential. Full inner curves are saved in `tuning_curves.csv`.

## Verification and limits

Thirteen focused tests passed, including exact stochastic resume, fold/input separation, objective equivalence, learning and agent-permutation checks at the new widths, selection isolation and exact reference initialization. GPU smoke tests pass for both new sizes; replaying the reference's first 128 updates reproduces its original validation metrics exactly. All 36 final checkpoints, including the reused 12, were independently reloaded on CPU on 64 training and 64 assessment snapshots each. Maximum probability difference: 2.62e-06. The strengthened baseline's per-question scores reproduce independently.

All 1,923 tracked prior immutable artifacts remain unchanged. Input arrays and all 16 preprocessing folds are identical to their originals. All 24 new final models were frozen before new outer assessment. No new source-model calls, official test scoring or autonomous rollouts occurred. No claims are made about new graph families, personas, source LLMs or a universal interaction law.

The size sweep tests whether less capacity improves transfer with these inputs and this finite search. It cannot determine an irreducible accuracy ceiling, repair missing state or establish generalization on new independent questions. A final model choice still requires fresh question assessment.

![Transformer size comparison](/Users/melancholymirth/deshearthGit/github/agent_interact/reports/transformer_size_sweep.png)

Artifacts: `runs/transformer_size_sweep_v1/`; exact driver commands:

```sh
.venv/bin/python scripts/transformer_size_sweep.py prepare
.venv/bin/python scripts/transformer_size_sweep.py fit --device mps
.venv/bin/python scripts/report_transformer_size_sweep.py evaluate --device mps
.venv/bin/python scripts/report_transformer_size_sweep.py summarize
```
