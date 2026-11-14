# Transformer generalization: nested whole-question comparison

Completed **216 tuning fits, 36 refits with new initialization seeds and 16 fold-specific Ising fits** on the existing GPT development pool. Every neural setting is selected within the outer training questions. All outer outcomes are scored only after selection and all refits are frozen.

## Outer-fold results

Each of 20 questions is assessed once, in one of four outer folds. Each outer refit uses the other 15 questions. Neural results average three initialization seeds within each question, then give every question equal weight. These are fresh fits on previously studied questions, not a new confirmatory dataset. The numbers cannot be compared directly with earlier scores on only four validation questions.

| Model | Neural parameters | Other fitted coefficients | Refit training error | Outer error | Outer NLL | Outer Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ising | 0 | 19 | 9.79% | 10.95% | 0.22755 | 0.13506 |
| Full transformer | 135,458 | 0 | 7.77% | 9.81% | 0.23282 | 0.13760 |
| Small transformer | 34,978 | 0 | 7.61% | 9.48% | 0.22321 | 0.13270 |
| Ising + small transformer | 34,978 | 19 | 7.80% | 9.41% | 0.22354 | 0.13037 |

## Paired comparisons

Differences are first minus second; negative favors the first model. Intervals resample the 20 question-level differences after averaging seeds, with 3,000 paired bootstrap draws. They are descriptive, pointwise intervals: only four outer folds, overlapping training sets, previously examined questions and multiple contrasts limit inferential claims. Question/fold win counts use a 0.000001-nat tolerance to avoid counting floating-point ties.

| Comparison | NLL difference [descriptive 95% interval] | Error difference, pp [interval] | Questions with lower NLL |
| --- | ---: | ---: | ---: |
| Full transformer − Ising | +0.00527 [-0.02066, +0.03220] | -1.14 [-2.40, +0.01] | 12/20 |
| Small transformer − Full transformer | -0.00960 [-0.02165, +0.00012] | -0.32 [-0.74, +0.03] | 13/20 |
| Ising + small transformer − Small transformer | +0.00032 [-0.01163, +0.01270] | -0.07 [-0.58, +0.50] | 10/20 |
| Ising + small transformer − Full transformer | -0.00928 [-0.02697, +0.00818] | -0.40 [-0.95, +0.16] | 11/20 |
| Ising + small transformer − Ising | -0.00401 [-0.01504, +0.00740] | -1.54 [-2.58, -0.47] | 11/20 |
| Small transformer − Ising | -0.00433 [-0.02421, +0.01453] | -1.47 [-2.66, -0.31] | 10/20 |

## What was matched and selected

All models use the same current opinions, 16-dimensional persona/question context, signed graph and node degrees. No arguments, opinion history, question IDs or extra source responses were added. The full transformer has the original A2 architecture. The smaller and hybrid models have the same 34,978-parameter architecture; their initial neural weights and minibatch orders match. All neural output layers start at zero. Plain models initially predict one half; the hybrid initially reproduces its fold's fitted Ising probabilities.

Every neural model receives the same six candidate settings: learning rate 0.0001 or 0.0003, crossed with an output penalty of 0, 0.001 or 0.01. Dropout is 0.1 and AdamW weight decay is 0.001. The added loss is lambda times the mean squared neural logit difference. Thus ordinary models shrink toward an uncertain prediction and the hybrid shrinks toward Ising. This tests the choice of prior against generic confidence regularization at the same size and search budget. It does not isolate a residual connection while keeping every selected hyperparameter identical, because each procedure selects its own candidate.

Within each outer training set, three inner folds fit ten questions and select on five. Question PCA3, persona PCA3 and Ising's 19 coefficients are refitted inside each fitting set. All 32 personas occur in training in every fold. Held-out question embeddings are only transformed; their labels never fit PCA or Ising. The signed social channels use the original author's pinned two-stage fit.

Inner tuning uses seed 42, validation every 128 updates (including initialization), six-check patience, a 0.0001 minimum improvement and a 3,072-update cap. NLL excludes the training penalty. Each model's candidate minimizes mean NLL across the three inner folds. Refits use seeds 142/143/144 and the median selected inner number of data passes, scaled from ten to fifteen training questions. Refitting has no validation stopping or outer-label scoring. Initialization is a valid selection; a zero-duration hybrid is exactly Ising.

Inner stopping reasons: {'early_stopping': 209, 'step_cap': 7}. Zero-update outer refits: 0/36. Total recorded neural fit time: 45.1 minutes. A fixed compute ceiling remains a search limitation; this is not an exhaustive regularization or architecture search.

## Selected settings

| Outer fold | Model | Learning rate | Output penalty | Refit updates |
| --- | --- | ---: | ---: | ---: |
| 0 | Full transformer | 0.0003 | 0.001 | 384 |
| 0 | Small transformer | 0.0001 | 0.001 | 3264 |
| 0 | Ising + small transformer | 0.0003 | 0.01 | 384 |
| 1 | Full transformer | 0.0003 | 0.001 | 1152 |
| 1 | Small transformer | 0.0001 | 0.001 | 1920 |
| 1 | Ising + small transformer | 0.0003 | 0.01 | 768 |
| 2 | Full transformer | 0.0003 | 0.001 | 384 |
| 2 | Small transformer | 0.0003 | 0.001 | 1536 |
| 2 | Ising + small transformer | 0.0003 | 0.001 | 192 |
| 3 | Full transformer | 0.0001 | 0.001 | 1536 |
| 3 | Small transformer | 0.0003 | 0.001 | 1344 |
| 3 | Ising + small transformer | 0.0001 | 0.01 | 576 |

## Interpretation and scope

The matched small-hybrid comparison changes outer NLL by +0.00032; reducing the ordinary model's size changes NLL by -0.00960. Assess these paired differences, their question/fold variation and their intervals rather than choosing a method from training error alone. The hybrid can preserve errors in Ising as well as transferable structure. Better training fit alone does not establish better generalization.

The hybrid has lower NLL than the small ordinary model in 2/4 outer folds; the small ordinary model beats the full model in 3/4 folds. Exact fold differences are saved in `fold_contrasts.csv`.

Nested separation prevents the new tuning procedure from choosing settings on each outer fold's labels. It does not undo earlier human/model-development exposure to these 20 questions. No official test questions, unfamiliar graph tests, source-model calls or autonomous rollouts are added. These results concern next majority-of-five agent opinions, not mathematical correctness. They do not settle argument representation or establish a universal interaction law.

Generalization here means holding out question IDs within the same question bank, GPT source, 32 personas and eight familiar graphs. Semantic template families are not independently grouped or held out; exact duplicate question texts were already rejected by the source preparation.

Broader question coverage remains a separate proposed experiment. A later data study should hold the source LLM and generation protocol fixed, compare question breadth at a controlled transition budget, and retain genuinely untouched assessment questions. This run does not generate that new dataset.

## Verification and artifacts

Eight focused tests passed: nested question isolation, held-out PCA perturbation, baseline/target input boundaries, matched zero-output initialization, objective equivalence, equal-question scoring, attention learning after initialization and exact stochastic resume. All 36 refitted checkpoints were independently reloaded on CPU on 64 training and 64 assessment snapshots each. Maximum probability difference from MPS is 3.28e-06. All 581 recorded files from earlier studies are unchanged. Saved training metrics, fold identities, selected durations, initialization hashes and cohort checksums were verified.

![Generalization comparison](/Users/melancholymirth/deshearthGit/github/agent_interact/reports/transformer_generalization.png)

The first panel shows equal-question mean NLL with four outer-fold means overlaid. The second shows paired question-level NLL differences after averaging seeds, colored by outer fold. The plotted points are not independent repeated datasets.

```sh
.venv/bin/python -m pytest tests/test_transformer_generalization.py
.venv/bin/python scripts/transformer_generalization.py prepare
.venv/bin/python scripts/transformer_generalization.py tune --device mps
.venv/bin/python scripts/transformer_generalization.py select
.venv/bin/python scripts/transformer_generalization.py refit --device mps
.venv/bin/python scripts/report_transformer_generalization.py evaluate --device mps
.venv/bin/python scripts/report_transformer_generalization.py summarize
```

The frozen protocol, folds, PCA/Ising fits, 252 neural run directories, tuning curves, selected settings, predictions, per-question scores and integrity records are under `runs/transformer_generalization_v1/`. Completed runs are reused only after identity and artifact checks.
