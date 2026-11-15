# Training-question augmentation: 320 versus 512

Completed 2026-11-15T08:15:52.043443+00:00.

Best observed 512-question validation accuracy: **88.677%** (tree_semantic8_512_fixed). The 95% target remains unmet.

The new cohort adds the first 192 training units from the larger collection's frozen ordering to the old 320. Every selected unit was complete before export. The old 80 validation questions are the common anchor; new validation conversations and all 40 reserved assessment conversations remain unused. The outcome is the most probable next source A/B response, using all 256 agent updates per question on existing stochastic histories.

## Fixed comparison

| Model | 320 validation | 512, same updates | 512, same data passes |
| --- | ---: | ---: | ---: |
| Ising · 22 | 85.439% | 85.469% (fixed recipe) | — |
| Ordered choice · 978 | 88.066% | 88.078% | 88.143% |
| Choice + history · 2,306 | 88.441% | 88.452% | 88.421% |
| Arguments + history · 2,306 | 88.371% | 88.464% | 88.514% |
| Joint question/arguments · 26,210 | 87.479% | 87.642% | 87.718% |
| Argument-feature tree | 88.579% | 88.677% (fixed recipe) | — |

Ising and tree are refitted once with their fixed original recipes. Neural results average the same three initialization seeds; no ensembling or best-seed selection. Same updates holds optimizer steps fixed. Same data passes increases steps by approximately 512/320; both checkpoints were declared before fitting. All 320 endpoints replay unchanged saved models. Settings were originally selected inside the old training set. There is no new validation-based retuning.

The argument/history model gains **0.143 pp [0.013, 0.277]** from more questions at equal data passes. Other accuracy-change intervals include zero. This is a small descriptive improvement, with no adjustment for the many comparisons or earlier validation reuse. The wider argument model gains 0.239 pp at equal passes, but its interval includes zero and it still trails the compact models.

| Model / 512 budget | Training accuracy | Validation change | Paired 95% interval |
| --- | ---: | ---: | ---: |
| Ising · 22 / fixed | 85.792% | +0.029 pp | [-0.054, +0.117] |
| Ordered choice · 978 / same updates | 88.523% | +0.011 pp | [-0.116, +0.142] |
| Ordered choice · 978 / same passes | 88.630% | +0.076 pp | [-0.055, +0.208] |
| Choice + history · 2,306 / same updates | 88.999% | +0.011 pp | [-0.130, +0.156] |
| Choice + history · 2,306 / same passes | 89.029% | -0.020 pp | [-0.153, +0.112] |
| Arguments + history · 2,306 / same updates | 88.960% | +0.093 pp | [-0.080, +0.267] |
| Arguments + history · 2,306 / same passes | 89.063% | +0.143 pp | [+0.013, +0.277] |
| Joint question/arguments · 26,210 / same updates | 89.009% | +0.163 pp | [-0.111, +0.454] |
| Joint question/arguments · 26,210 / same passes | 89.527% | +0.239 pp | [-0.026, +0.526] |
| Argument-feature tree / fixed | 90.055% | +0.098 pp | [-0.107, +0.303] |

At 512 questions, the history model still has **no clear argument advantage** over its choice counterpart: the post-fit difference is +0.011 pp [−0.138, +0.160] at equal updates and +0.093 pp [−0.067, +0.254] at equal data passes. Their original internally selected training durations differ; these are comparisons of the fitted pipelines.

## Representation and data controls

The old question/persona PCA and message PCA remain frozen at their 320-question fits. Old inputs are byte-identical. Added questions and messages use the same frozen MiniLM encoder and complete token pooling. The full-vector argument model jointly encodes the question, options and message. All models use current probabilities; none receives next probabilities, answer keys, worked solutions or screening scores as features. The compact history pair uses the same numerical history and differs in choice-only versus actual argument tokens. Frozen MiniLM adds 22,713,216 pretrained weights to the update-rule counts.

Encoding added 78,508 distinct texts and retained all 10,181,391 tokens in 99.8 seconds. A six-text CPU replay differs by at most 1.7e-07 per vector entry. No source-model calls were made for this study.

## The added questions have a different topic mix

| Topic | Original 320 | Added 192 |
| --- | ---: | ---: |
| Algebra | 45 | 53 |
| Counting & Probability | 46 | 20 |
| Geometry | 46 | 9 |
| Intermediate Algebra | 46 | 39 |
| Number Theory | 46 | 26 |
| Prealgebra | 45 | 37 |
| Precalculus | 46 | 8 |

The larger MATH bank follows the user-approved availability-proportional topic quotas and revised selection fractions. Consequently this experiment measures augmentation from that selected pool, rather than a pure sample-size change under an identical question distribution. Graph, label and difficulty counts are retained in the numerical composition file.

## Verification

All 55,296 new full prompt hashes and 276,480 votes reproduce. The 192 added conversations comprise 153 A30 and 39 L40S runs. Maximum repeated-prompt normalized probability spread is 5.06e-05; none changes the preferred answer. Targets consistently use the first saved response's conditional A/B distribution. All cases are retained.

Six focused checks pass. The independent audit verifies all 80 prediction arrays and 30 paired contrasts, replays all 36 neural and 4 baseline validation outputs, and confirms 28 old prediction arrays are unchanged. Maximum independent metric difference is 0. All final checkpoints froze before validation scoring. A tiny lexical-scaling precision mismatch was fixed during the pre-fit input-equivalence check; the unscored protocol revision is archived.

The 80 validation questions have been reused extensively. Bootstrap intervals resample whole questions and condition on the fitted models; they do not account for the earlier search over architectures, repeated validation, or new training-cohort sampling. This remains familiar-graph, familiar-persona, one-step prediction after message delivery. It does not establish a universal accuracy ceiling or autonomous rollout performance.

Artifacts: [results](../runs/modal_scaling_512_v1/evaluation/results.json), [audit](../runs/modal_scaling_512_v1/evaluation/completion_audit.json), [protocol](../runs/modal_scaling_512_v1/protocol.json), [figure](modal_scaling_512.png).
