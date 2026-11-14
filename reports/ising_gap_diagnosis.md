# Why are the transformer and Ising scores close?

The new diagnosis points to a large easy subset, partial cancellation of the transformer's corrections, and question-specific response biases. A compact persistence correction recovers part of the transformer's advantage. These explain aspects of the measured gap; they do not identify an irreducible prediction ceiling or a unique causal mechanism.

All analyses use the same twenty previously studied GPT development questions, four outer folds and 163,840 agent transitions. Original predictors and preprocessing remain frozen. Scores average neural seeds rather than ensemble their probabilities. Accuracy means predicting the agent's next majority-of-five opinion, not answering the mathematical question correctly.

## Where the overall score comes from

Subgroups below are chosen after earlier overall results and are descriptive. Ising confidence is the larger of its two predicted probabilities. The pooled conditional rates weight transitions within each subgroup; full-cohort rates weight all twenty equally sized questions equally.

| Subset | Share of updates | Ising error | Small transformer error |
| --- | ---: | ---: | ---: |
| Ising confidence < 0.80 | 20.37% | 45.27% | 37.78% |
| 0.80–0.95 | 14.16% | 2.57% | 2.73% |
| At least 0.95 | 65.47% | 2.08% | 2.13% |

The difficult 20.37% contains 84.23% of Ising errors. The small transformer improves classification there but has worse NLL: 0.71237 versus 0.68887. Its probability estimates on difficult cases still need work; classification gain alone is insufficient.

Across all updates the small transformer corrects 4.61 percentage points of Ising errors, introduces 3.15 points, and shares 6.33 points of errors. The net gain is 1.47 points. These categories describe paired predictions, not a deployable oracle that knows which prediction is correct.

## Sampling variation does not explain everything

| Target vote agreement | Share of updates | Small transformer error | Share of its errors |
| --- | ---: | ---: | ---: |
| 3/5 | 3.64% | 48.85% | 18.74% |
| 4/5 | 4.07% | 42.46% | 18.23% |
| 5/5 | 92.29% | 6.48% | 63.03% |

Votes are used retrospectively here, never as next-outcome predictor inputs. Five agreeing samples do not establish a deterministic response or a Bayes accuracy ceiling. However, attributing all residual errors to split votes is inconsistent with this breakdown.

## Question-specific adaptation diagnostic

For each question, fit offsets on complete conversations in replicas 0/1 and evaluate replicas 2/3, then reverse. All eight graphs and all eight updates of each conversation stay together. Base predictors were originally fitted on the other fifteen questions. Offsets minimize summed binary cross-entropy plus one half the sum of squared offsets; this penalty was fixed before the adaptation fits. No hyperparameter search is performed.

This intentionally supplies outcomes from other conversations about the same question. It measures familiar-question adaptation and must not be presented as new-question generalization or a better model using the original information budget.

| Predictor | Original error | One offset per question | One offset per question and persona |
| --- | ---: | ---: | ---: |
| Ising | 10.95% | 9.10% | 8.82% |
| Small transformer | 9.48% | 8.45% | 8.31% |

A single question offset accounts for most of the adaptation improvement. This demonstrates a stable question-dependent prediction bias across recorded conversations. It does not by itself identify whether that bias comes from question representation, insufficient question diversity, or unrepresented message content.

## A new, inexpensive state experiment

Three small corrections were tested with the existing nested whole-question folds. Each uses the same frozen, fold-fitted Ising logit as an offset. The calibration control adds an intercept and a global logit coefficient (two correction coefficients). The current-state control adds the receiver's current spin (three). The initialization model adds two more inputs: smoothed log-odds from that receiver's five time-zero responses and from all 32 agents' time-zero responses in the same episode (five). Pooling the initial network is explicitly an additional global input. No later votes, future opinions or other same-question episodes construct these features.

All three receive the same penalty grid [0.0001, 0.001, 0.01], applied to standardized correction coefficients. Standardization uses fitting questions only. Three inner folds choose mean validation NLL for each outer fold/model; twelve outer refits are all frozen before their outcomes are scored. This adds 108 inner fits and 12 outer fits, with no new transformer training or source generation.

| Predictor | Outer error | Outer NLL |
| --- | ---: | ---: |
| Original Ising | 10.95% | 0.22755 |
| Ising + global calibration | 10.95% | 0.22754 |
| Ising + calibration + current own opinion | 10.11% | 0.22474 |
| Same + initial response tendencies | 10.20% | 0.22701 |
| Small transformer (previous frozen fits) | 9.48% | 0.22321 |

The simple current-opinion correction recovers about 57% of the small transformer's net accuracy advantage over Ising. Its coefficient is positive in all four folds (0.214–0.294 log-odds per signed spin). The remaining transformer advantage is 0.62 percentage points. Initial response features do not improve this tested linear correction; this does not rule out other estimators or nonlinear uses of initialization.

The current-opinion coefficient measures a predictive persistence association. The released spin prompt does not explicitly feed back the receiver's own previous answer, so it should not be interpreted as a proven causal self-influence mechanism. A compact candidate expression is:

```text
logit P(next majority = A) = b + a * Ising_score + c * current_spin
```

The original 19 Ising coefficients are fitted separately within each training fold; three correction coefficients are then fitted. This parameterization contains redundancy with the original intercept/scale and is not claimed to be a minimal parameter count.

Paired differences below use 3,000 question-bootstrap draws (seed 447), with neural seeds averaged within questions. Intervals are descriptive and pointwise: repeated exposure to this development pool, overlapping training folds and multiple comparisons prevent treating them as fresh confirmation.

| Comparison | Error difference, pp [descriptive 95% interval] | NLL difference [interval] |
| --- | ---: | ---: |
| Current-state correction − Ising | -0.84 [-1.54, -0.21] | -0.00281 [-0.00481, -0.00107] |
| Initialization − current-state correction | +0.09 [-0.09, +0.28] | +0.00227 [+0.00008, +0.00581] |
| Small transformer − current-state correction | -0.62 [-1.45, +0.21] | -0.00153 [-0.02016, +0.01737] |

## Interpretation and next discriminating experiment

The previous capacity study already showed that the same transformer can fit observed training labels much better than Ising (4.41% versus 9.15% error), while transferring worse. The present evidence therefore favors a transfer/information problem over a demonstrated inability to express the Ising rule. It remains possible that the transferable rule conditional on the supplied opinions and compressed context is relatively simple.

The source protocol explicitly asks for an A/B snap judgment without working through the problem, and asks each sender to justify its already selected option. This makes a strong opinion-aggregation baseline plausible, but the protocol alone does not prove that arguments are irrelevant or that influence is linear. Earlier full-question and argument-encoding experiments also failed to establish a transferable gain; simply repeating those changes is not a new explanation.

The next useful experiment should distinguish stochastic source responses, omitted argument information and across-question transfer. On a broader fixed-source question bank, retain representative natural updates and a separately reported low-confidence subset. For identical complete prompts, collect disjoint repeated groups of five responses to estimate reproducibility of the actual majority target. For matched opinions/personas/signs, vary the incoming argument text and compare the source's response distribution. Fit both the compact current-state correction and the small transformer on matched data, with opinion-only versus argument-aware inputs. Vary question breadth at a controlled transition budget. Freeze all procedures before fresh question assessment; keep question/template families together. Hard-case sampling should be reported separately or weighted back to the target distribution.

This would test whether useful information exists beyond the compact law and whether the predictor can learn it. Repeated prompts alone cannot establish causal argument effects; argument interventions alone cannot establish predictive generalization. If an open-source LLM is used, regenerate a matched training/assessment study for that source rather than combining it with the GPT results.

## Verification

Eight focused tests passed, covering independent numerical optimization, objective gradients, finite single-class offsets, error bookkeeping, initial/future vote separation, training-only normalization and exclusion of each assessed replica's labels from its calibration. All old aggregate scores were reproduced. Saved offsets and all twelve new outer predictions were independently reconstructed. A real-data assessment-label perturbation left its own assessed predictions unchanged. Selection/refit timestamps and source checksums were verified. No official test evaluation, source-model calls, paid API usage or autonomous rollout was added.

Artifacts: `runs/ising_gap_diagnosis_v1/`, `runs/ising_initial_state_probe_v1/`, and `reports/ising_gap_diagnosis_checks.xml`. Drivers: `scripts/diagnose_ising_gap.py`, `scripts/probe_initial_opinion_state.py`, and `scripts/report_ising_gap_diagnosis.py`.
