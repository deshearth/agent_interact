# Investigating approximately 95% source-mode accuracy

Completed November 15, 2026, 02:03 AM Central.

**Best observed validation accuracy: 88.579%**, from Argument PCA 8 tree. The 95% target remains 6.421 percentage points away. No subset filtering or ensemble is used.

Change versus the previous 978-parameter ordered-choice model: **+0.734 pp**, paired 95% interval [+0.400, +1.077]. This is an exploratory comparison after repeated use of the same 80 questions.

The best neural model is the **2,306-parameter choice transformer with causal history: 88.441% validation accuracy**. Observed history adds **0.337 pp [0.094, 0.588]** over its identical-size round-only control. Replacing choice tokens with arguments changes accuracy by **−0.070 pp [−0.225, +0.081]** in this history model. Thus history helps modestly; an argument advantage remains unsupported in this comparison.

## What was predicted

For every agent and transition, the target is A iff the source's recorded next single-response conditional A/B probability exceeds 0.5. The source probability comes from the fixed Llama prompt at temperature 0.7. The source's old sampled-majority outcome is not the label here. The earlier 91.4% sampling reference is therefore not a ceiling for this target. These are one-step predictions on existing stochastic histories, not newly generated deterministic-agent trajectories.

There are 320 training and 80 reused validation questions, each with 256 agent updates. Candidate selection uses only the internal 256/64 split. All final models and diagnostic promotion decisions froze before outer scoring. The 40 assessment questions remain unused. No new source calls were made.

## Final model comparison

| Model | Learned update parameters | Training accuracy | Validation accuracy |
| --- | ---: | ---: | ---: |
| Original Ising / soft target | 19 | 84.917% | 84.727% |
| Ising + current probability / mode target | 22 | 85.350% | 85.439% |
| Previous ordered choice / soft target | 978 | 88.044% | 87.845% |
| Direct-mode ordered choice | 978 | 88.344% | 88.066% |
| Richer arguments | 14,946 | 88.886% | 88.076% |
| Joint question/arguments · full vectors | 26,210 | 89.200% | 87.479% |
| Choice + observed history | 2,306 | 88.793% | 88.441% |
| Arguments + observed history | 2,306 | 88.757% | 88.371% |
| Learned token attention / arguments | 4,738 | 88.368% | 88.213% |
| Argument PCA 8 tree | 6,405 nodes | 90.267% | 88.579% |
| Normalized token attention / arguments | 4,738 | 88.362% | 88.211% |
| Normalized attention / message words | 4,738 | 88.288% | 88.073% |
| Normalized token attention / choice | 4,738 | 88.463% | 88.192% |
| Question-conditioned choice token | 4,706 | 88.392% | 88.110% |
| Arguments + round-only recurrent control | 2,306 | 88.310% | 88.065% |
| Choice + round-only recurrent control | 2,306 | 88.292% | 88.104% |
| ising current soft | 22 | 85.159% | 85.210% |
| ising initial mode | 25 | 85.874% | 85.615% |
| ising initial soft | 25 | 85.598% | 85.322% |
| nano current choice soft | 978 | 88.049% | 87.804% |
| Direct-mode choice + initial probability | 986 | 88.290% | 88.035% |
| nano initial choice soft | 986 | 88.045% | 87.894% |
| Joint question/choice · full vectors | 26,210 | 89.480% | 87.354% |
| Joint question/arguments · PCA 32 | 14,946 | 88.837% | 88.162% |
| Joint question/choice · PCA 32 | 14,946 | 88.767% | 88.022% |
| reference graph opinion | 818 | 86.621% | 86.502% |
| Graph transformer + probability / soft target | 826 | 87.408% | 87.254% |
| Ising + current probability / soft target | 22 | 85.159% | 85.210% |
| reference ordered arguments opinion | 970 | 87.089% | 87.012% |
| Previous ordered arguments / soft target | 978 | 88.081% | 87.887% |
| reference ordered choice opinion | 970 | 87.223% | 86.969% |
| rich arguments soft | 14,946 | 88.828% | 87.964% |
| Richer choice | 14,946 | 88.708% | 88.123% |
| rich choice soft | 14,946 | 88.739% | 88.091% |
| Token projection + mean / arguments | 4,706 | 88.087% | 87.970% |
| Learned token attention / choice | 4,738 | 88.457% | 88.188% |
| Token projection + mean / choice | 4,706 | 88.457% | 88.185% |
| Current-state tree | 4,636 nodes | 89.415% | 88.389% |
| Initial-state tree | 8,784 nodes | 90.566% | 88.311% |
| Lexical tree | 5,795 nodes | 90.071% | 88.501% |
| Rich-state tree | 4,026 nodes | 89.534% | 88.452% |
| Argument PCA 32 tree | 3,172 nodes | 89.252% | 88.457% |

Neural values average three individual refits, not a best seed or an ensemble. Frozen MiniLM has 22,713,216 pretrained parameters in addition to these update counts. Trees are larger diagnostic models and are reported by node count rather than treated as parameter-matched transformers. Soft-target and modal-target losses optimize different quantities; modal NLL in the numerical results is not a source-distribution calibration score.

## Hypothesis tests

1. **Missing initial state or simple history:** Six tree feature sets test current distributions, initial probabilities, richer state, lexical cues and wider argument PCA. A training-only residual-memory correction was also tested; all fixed correction weights reduced training accuracy.
2. **Wrong objective or insufficient width:** Direct modal classification is compared with source-probability supervision, with matched accuracy-based checkpoint selection. The richer models add question features, known-persona identity and the initial population state.
3. **Arguments lose their connection to the question:** Question, answer options and delivered argument are encoded jointly, with matched question-plus-choice controls. Both PCA 32 and full 384 vectors are tested. All 13,629,561 tokens are retained; four texts span multiple independent 510-token windows. The full vectors improve fitting but do not automatically improve held-out prediction.
4. **The frozen text encoder is unsuitable:** A three-epoch internal-only pilot adapts its last transformer layer at two rates, alongside a matched head-only continuation control. First five layers and embeddings remain frozen. This is a limited last-layer experiment, not an exhaustive test of end-to-end language-model training.
5. **Past behavior contains additional state:** A 16-unit recurrent component receives earlier opinions/probabilities and friendly/unfriendly population means, only through the current round. Identical round-only controls isolate the observed-history contribution.
6. **Selected training errors hide fitting capacity:** A 215,426-parameter model is trained without dropout or weight decay on 8 and 256 training questions. The small subset tests memorization; the larger subset exposes the training/generalization tradeoff.

7. **Sentence averaging discards useful words:** Frozen contextual token vectors are reduced using training-only PCA 32 and a learned token projection. Compare uniform pooling with four learned attention queries, for both choice-only and actual arguments. Every token is retained; five fixed token features mark the message region, uppercase A/B, digits and position. Attention adds 32 parameters to the matched mean-pooling model. These runs use 16-snapshot batches and the same two learning rates; their comparison with earlier models changes representation and batch size, while within-family controls match both.

8. **Population observations reveal shared question context:** Append all 32 current opinions and current/previous/cumulative probabilities to the state trees. On internal 64, the current-state tree gains 0.000 pp and the lexical tree gains 0.079 pp. Neither crosses the +0.5 pp promotion gate, so neither is scored on outer 80. Nonneighbor states are tested as predictive proxies, not asserted causal influences.

9. **The new token selector remains too uniform:** A training-only probe on two questions found normalized attention entropy 0.999997. A follow-up normalizes each attention key to unit variance, without additional fitted parameters, and compares all-token pooling with message-only pooling. Each message token still sees its question/options in the frozen encoder. The one-token A/B control removes 32 query parameters because no token choice is possible.

## Encoder adaptation: internal 64 only

| Encoder rate | Trainable encoder weights | Selected epoch | Training accuracy | Internal accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 2 | 90.013% | 86.359% |
| 1e-05 | 1,774,464 | 2 | 90.018% | 86.438% |
| 3e-05 | 1,774,464 | 2 | 89.935% | 86.499% |

Starting frozen-head accuracy was 86.279%. The prespecified +0.5 pp promotion gate was not reached; no encoder-adapted model was scored on outer 80.

## Capacity diagnostic: training-only development

| Input | Fitting questions | Terminal training accuracy | Terminal internal accuracy | Best internal accuracy |
| --- | ---: | ---: | ---: | ---: |
| arguments | 256 | 99.049% | 80.365% | 85.547% |
| arguments | 8 | 100.000% | — | — |
| choice | 256 | 98.302% | 82.373% | 85.229% |
| choice | 8 | 99.951% | — | — |

The 8-question diagnostic uses preprocessing fitted on the inner 256; it is a memorization check, not an 8-question learning-curve estimate. No capacity model reached the 89% internal promotion gate. Their terminal weights are not used as final predictors.

## Remaining errors by source probability margin

These are descriptive groups for the best observed model. Every group remains in the primary accuracy; the source's next probability is never used to select which cases to predict.

| Margin from 0.5 | Share of updates | Accuracy | Share of errors |
| --- | ---: | ---: | ---: |
| 0–0.025 | 2.59% | 57.36% | 9.66% |
| 0.025–0.05 | 3.07% | 65.02% | 9.41% |
| 0.05–0.1 | 5.43% | 67.90% | 15.26% |
| 0.1–0.2 | 10.17% | 72.73% | 24.28% |
| 0.2–0.3 | 11.58% | 82.21% | 18.04% |
| 0.3–0.4 | 15.77% | 90.52% | 13.08% |
| 0.4–0.5 | 51.39% | 97.72% | 10.26% |

## Interpretation and practical next experiment

The experiments distinguish the ability to fit existing questions from the ability to transfer to new questions. Increasing capacity can raise training accuracy substantially while reducing held-out accuracy. This rules out the simple claim that the architecture cannot fit the labels at all. It does not prove that 95% is unattainable, nor that arguments contain no predictive information.

The remaining task is to learn transferable relations between the question, the actual reasoning in the messages, and the receiver's response. The next data test compares 320 versus 512 training questions, adding the first 192 training units from the larger collection's preexisting frozen order. The old 80 validation questions provide a fixed anchor, while newly collected validation conversations remain unused in this interim test. The larger bank uses revised topic sampling, so this measures data augmentation under that sampling policy, not a pure count effect with an identical question distribution. More steps or parameters on the same 320 questions are not supported as a route to 95% by these results. A larger corpus may help; these experiments do not guarantee that it will close the gap.

## Verification and limitations

The independent audit reproduces 204 prediction files and 40,800 seed/question records, checks 234 paired contrasts, and replays all 82 new final models with zero class disagreements. Maximum independent metric discrepancy is 3.7e-09. Target perturbations leave inference unchanged. History tests verify that future observations cannot change current inputs.

Intervals use 5,000 paired whole-question bootstrap draws, conditional on fitted models. They do not account for repeated validation, searching many models, training resampling or multiple comparisons. The best observed score is exploratory. Source probabilities are conditional first-token A/B probabilities; the checked small non-A/B mass and rare formatting exceptions from the previous audit still apply. Graphs and personas are familiar; this study does not measure transfer to new graphs or agents. Actual delivered arguments are supplied, so this is not autonomous rollout before future messages are generated.

Failed logging, dtype and GPU-memory attempts are preserved separately and excluded from results. Encoder GPU settings were revised before any complete adaptation comparison or outer scoring. The independent HPC3 collection was not modified. After the shared SSH session was reconnected, the November 15 07:49 UTC controller snapshot showed 515 training and 130 validation conversations, with 28 GPUs running. All first 192 new training units in the frozen collection order were complete. Their export is a separate follow-up.

Artifacts: [numerical results](../runs/modal_accuracy_investigation_320_v1/evaluation_v1/results.json), [audit](../runs/modal_accuracy_investigation_320_v1/evaluation_v1/completion_audit.json), [hypothesis journal](../experiments/modal_accuracy_investigation_320_v1/README.md), [figure](modal_accuracy_investigation_320.png).
