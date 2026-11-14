# Transformer comparison below 7,370 parameters

Completed three new transformer sizes on the existing twenty-question GPT development pool. The 7,370-parameter reference and strengthened Ising baseline are reused after identity/checksum checks. All four neural sizes have the same inputs, four attention heads, six tuning candidates, inner stopping rule and final seeds. Only attention/hidden/feedforward widths change.

These are exploratory nested whole-question results on previously studied questions, not fresh confirmation. NLL is primary. Scores average seeds within question, then weight questions equally; predictions are not ensembled.

| Model | Neural parameters | Other fitted coefficients | Training error | Outer error | Outer NLL | Outer Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ising | 0 | 19 | 9.79% | 10.95% | 0.22755 | 0.13506 |
| Ising + current-opinion correction | 0 | 19 baseline + 3 correction | 9.53% | 10.11% | 0.22474 | 0.13226 |
| Transformer 314 | 314 | 0 | 8.19% | 9.22% | 0.21622 | 0.12655 |
| Transformer 818 | 818 | 0 | 8.02% | 9.12% | 0.21258 | 0.12493 |
| Transformer 2,626 | 2,626 | 0 | 7.66% | 9.17% | 0.22036 | 0.12792 |
| Transformer 7,370 | 7,370 | 0 | 7.72% | 9.17% | 0.21472 | 0.12745 |

## Paired comparisons

Differences are first minus reference, so negative favors the first model. Intervals resample twenty paired question differences 3,000 times. They are pointwise and descriptive: only four overlapping outer training folds, prior question exposure and multiple contrasts limit inference. An interval crossing zero does not establish equivalence or noninferiority.

| Comparison | Error difference, pp [95% interval] | NLL difference [95% interval] | Questions with lower NLL |
| --- | ---: | ---: | ---: | ---: |
| Transformer 2,626 − Transformer 7,370 | +0.00 [-0.24, +0.21] | +0.00564 [-0.00044, +0.01268] | 9/20 |
| Transformer 818 − Transformer 7,370 | -0.05 [-0.23, +0.11] | -0.00214 [-0.00975, +0.00545] | 9/20 |
| Transformer 314 − Transformer 7,370 | +0.05 [-0.24, +0.36] | +0.00150 [-0.00427, +0.00697] | 8/20 |
| Transformer 314 − Transformer 818 | +0.10 [-0.18, +0.34] | +0.00363 [-0.00126, +0.00824] | 6/20 |
| Transformer 818 − Transformer 2,626 | -0.05 [-0.36, +0.27] | -0.00778 [-0.01502, -0.00051] | 14/20 |
| Transformer 314 − Ising + current-opinion correction | -0.88 [-1.56, -0.16] | -0.00852 [-0.01920, +0.00199] | 13/20 |
| Transformer 818 − Ising + current-opinion correction | -0.99 [-1.69, -0.26] | -0.01216 [-0.02077, -0.00327] | 13/20 |
| Transformer 2,626 − Ising + current-opinion correction | -0.93 [-1.65, -0.15] | -0.00438 [-0.01789, +0.00923] | 12/20 |
| Transformer 7,370 − Ising + current-opinion correction | -0.94 [-1.69, -0.17] | -0.01002 [-0.02386, +0.00333] | 12/20 |

## Matched procedure

Four outer folds each hold out five questions; the other fifteen support three inner folds of ten fitting/five selection questions. Every question's eight graphs, four replicas, eight updates and 32 agents stay together. All sixteen training-fold question PCA and Ising fits are exactly reused from the verified earlier study. No official test question enters the new fits or scoring.

Inputs remain current opinions, 16-dimensional persona/question context, signed graph and degree features. No arguments, raw text, additional history or question IDs are added. Architectures:

| Parameters | Hidden width | Heads | Key/head | Value/head | Feedforward |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 314 | 4 | 4 | 1 | 1 | 8 |
| 818 | 8 | 4 | 1 | 2 | 16 |
| 2,626 | 16 | 4 | 2 | 4 | 32 |
| 7,370 | 28 | 4 | 4 | 7 | 56 |

Every size starts with a zero output head and receives learning rates 0.0001/0.0003 crossed with logit penalties 0/0.001/0.01, dropout 0.1, AdamW decay 0.001, batch32 and clipping1. The unchanged training loop evaluates inner NLL every 128 updates, allows initialization selection, uses six-check patience and a 3072-update cap. Within each outer fold/size, average inner NLL chooses a candidate; the median selected data passes sets the outer refit length, scaled from ten to fifteen questions. Refits use seeds 142/143/144. There is no outer-label early stopping or best-seed selection.

Settings are separately selected for each size. This compares equally tuned procedures rather than holding selected optimizer settings or update counts identical. Smaller models can receive different selected training durations, within the same search policy.

The strengthened Ising benchmark uses three correction coefficients (intercept, prior-logit scaling and current own opinion) atop its nineteen fold-fitted baseline coefficients. Its earlier nested penalty grid has three choices versus six for each neural model; it is a stronger benchmark, not a matched architecture/compute ablation. Its parameterization includes intercept/scale redundancy, so this is not a minimal coefficient count.

Executed 216 new inner fits and 36 new refits; reused 72 reference inner fits and 12 reference refits. New recorded fitting time: 83.9 minutes. Inner stopping counts: {'step_cap': 177, 'early_stopping': 39}. Reused fits are not counted as newly executed work.

## Selected settings

| Outer fold | Parameters | Learning rate | Output penalty | Refit updates | Origin |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 314 | 0.0003 | 0.0 | 4608 | new |
| 0 | 818 | 0.0003 | 0.0 | 4224 | new |
| 0 | 2,626 | 0.0003 | 0.0 | 4416 | new |
| 0 | 7,370 | 0.0003 | 0.001 | 2304 | reused |
| 1 | 314 | 0.0003 | 0.001 | 4224 | new |
| 1 | 818 | 0.0003 | 0.001 | 3840 | new |
| 1 | 2,626 | 0.0003 | 0.001 | 3072 | new |
| 1 | 7,370 | 0.0003 | 0.001 | 1344 | reused |
| 2 | 314 | 0.0003 | 0.001 | 4032 | new |
| 2 | 818 | 0.0003 | 0.001 | 2688 | new |
| 2 | 2,626 | 0.0003 | 0.001 | 3840 | new |
| 2 | 7,370 | 0.0003 | 0.001 | 1152 | reused |
| 3 | 314 | 0.0003 | 0.0 | 4416 | new |
| 3 | 818 | 0.0003 | 0.0 | 2496 | new |
| 3 | 2,626 | 0.0003 | 0.001 | 2688 | new |
| 3 | 7,370 | 0.0003 | 0.001 | 1536 | reused |

## Training-cap diagnostic

| Parameters | Inner fits reaching update cap | Best checkpoint at cap | Selected candidate's inner checkpoints at cap |
| ---: | ---: | ---: | ---: |
| 314 | 72/72 | 44/72 | 3/12 |
| 818 | 54/72 | 31/72 | 1/12 |
| 2,626 | 51/72 | 20/72 | 2/12 |
| 7,370 | 26/72 | 4/72 | 0/12 |

A selected checkpoint at the update cap indicates that the search may be budget-limited; it does not establish convergence. The cap and candidate learning rates are matched to the reference, so this study estimates performance under that common search policy rather than each architecture's fully optimized potential. Full inner curves are saved in `tuning_curves.csv`.

## Verification and limits

Eighteen focused tests passed, including exact stochastic resume, fold/input separation, objective equivalence, learning and agent-permutation checks at the new widths, selection isolation and exact reference initialization. GPU smoke tests pass for all three new sizes; replaying the reference's first 128 updates reproduces its original validation metrics exactly. All 48 final checkpoints, including the reused 12, were independently reloaded on CPU on 64 training and 64 assessment snapshots each. Maximum probability difference: 9.48e-06. The strengthened baseline's per-question scores reproduce independently.

All 3,226 tracked prior immutable artifacts remain unchanged. Input arrays and all 16 preprocessing folds are identical to their originals. All 36 new final models were frozen before new outer assessment. No new source-model calls, official test scoring or autonomous rollouts occurred. No claims are made about new graph families, personas, source LLMs or a universal interaction law.

The size sweep tests whether less capacity improves transfer with these inputs and this finite search. It cannot determine an irreducible accuracy ceiling, repair missing state or establish generalization on new independent questions. A final model choice still requires fresh question assessment.

![Transformer size comparison](/Users/melancholymirth/deshearthGit/github/agent_interact/reports/transformer_micro_sweep.png)

Artifacts: `runs/transformer_micro_sweep_v1/`; exact driver commands:

```sh
.venv/bin/python scripts/transformer_micro_sweep.py prepare
.venv/bin/python scripts/transformer_micro_sweep.py fit --device mps
.venv/bin/python scripts/report_transformer_micro_sweep.py evaluate --device mps
.venv/bin/python scripts/report_transformer_micro_sweep.py summarize
```
