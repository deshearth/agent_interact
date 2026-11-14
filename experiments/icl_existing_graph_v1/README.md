# Completed assistant in-context prediction pilot

The assistant studied **256 labeled agent updates from eight existing training questions**, then manually authored **128 probability forecasts** for four newly completed conversations. All forecasts were committed before either their outcomes or competing predictor probabilities were displayed. No optimizer was run and no new neural or statistical model weights were fitted.

The prediction model is the assistant conditioned on the training examples, source-generation protocol and existing conversation context. Its saved artifacts are the examples, the pre-test forecasting strategy, and the actual probability estimates. This is not a newly trained 818-parameter network or a standalone weight checkpoint.

## Matched results

Every model is scored on the same 32 agents in each of four questions, predicting update 3→4. The transformer row averages individual-model scores across the three original frozen refits, rather than ensembling their probabilities. Log loss and Brier are lower when better.

| Predictor | Accuracy | Errors | Log loss | Binary Brier |
| --- | ---: | ---: | ---: | ---: |
| Assistant, in-context | **90.625%** | **12/128** | **0.23551** | **0.06699** |
| Frozen 818-parameter transformer | 87.760% | 15.67/128, seed average | 0.23668 | 0.07068 |
| Frozen Ising, 19 coefficients | 85.938% | 18/128 | 0.26831 | 0.08253 |
| Keep the current opinion | 80.469% | 25/128 | — | — |

The assistant correctly predicted **116/128** next opinions. Its accuracy is 2.86 percentage points above the transformer, but their probability losses are almost identical. Against Ising, its accuracy is 4.69 points higher. Four question groups are too few to establish a reliable advantage: descriptive paired question-bootstrap accuracy intervals are **−2.60 to +9.64 points** against the transformer and **−1.56 to +14.06 points** against Ising. These small-group intervals are not strong inferential evidence.

| Question group | Assistant | Transformer, seed average | Ising |
| --- | ---: | ---: | ---: |
| Quadratic maximum, test_00 | 29/32, 90.625% | 78.125% | 71.875% |
| Two linear equations, test_01 | 24/32, 75.000% | 79.167% | 78.125% |
| Defined arithmetic operation, test_02 | 32/32, 100.000% | 98.958% | 96.875% |
| Exponential equation, test_03 | 31/32, 96.875% | 94.792% | 96.875% |

The observed gain is concentrated in the quadratic question. The assistant does worse on the linear-equation question. Many updates in the final two questions are easy B predictions, so the headline accuracy is not comparable to the earlier 80-question-training / 20-question-validation aggregate.

## What the assistant learned from the examples

Friendly-neighbor majority was the strongest observed pattern. Across the 256 demonstrations, 56/68 agents with an A-friendly majority next chose A; only 10/138 with a B-friendly majority chose A. A friendly tie gave 31/50 A. Stronger friendly majorities were more consistent. The assistant softened those small, correlated counts when assigning probabilities, rather than assigning probabilities of zero or one.

Unfriendly effects were inconsistent in the demonstrations, so the assistant did not mechanically invert every unfriendly neighbor's opinion. Current opinion was used as weaker evidence about an agent's tendencies, especially when friendly neighbors were absent. Raw question/persona text guided ambiguous cases cautiously: the source is explicitly prompted to give a snap judgment without computing, so mathematical correctness is not its next-opinion target.

This qualitative strategy is recorded in `runs/icl_existing_graph_v1/strategy_before_test.md` before the test inputs were displayed. The 128 final probabilities were entered manually in context. The counting scripts only summarized training examples and rendered the available test state; they did not fit or generate the forecasts.

## Blinding and interpretation

At the saved 5:13 p.m. Central cluster snapshot on November 10, four fresh-graph conversations had completed. All four were selected before any of their new outcomes were displayed. Each contributes one fixed boundary, all 32 agents. Eight original training questions were chosen by a fixed ID hash; none overlaps the four prediction questions. The packet exposed only current opinions, persona/question/choice text, graph and round. It excluded test next opinions, all other test times, messages, raw votes, correct labels, solutions and source probabilities.

Forecasts were sealed at **5:23:34 p.m. Central**; evaluation first revealed outcomes at **5:23:39 p.m.** The committed prediction SHA-256 is `f9a67ca3c36b9502f7bc1036bde43aa2947223a178fdd70477cdbef018dae79f`. The scorer rejects revisions or input changes. Seven checks passed for whitelisted input export, future-outcome independence, commitment gates and metric definitions. An independent recomputation from the saved per-case records reproduces all model scores and the persistence baseline.

This is **fresh-trajectory prediction on familiar development questions**, not an untouched-question test. The question texts and old aggregate findings already belong to the research context; public MATH problems may also occur in model pretraining. The new trajectory outcomes used here had not been displayed during forecasting. Blinding is procedural, not an operating-system access barrier.

The assistant has extensive pretraining and raw text access; the small predictors use compressed MiniLM/PCA context and different amounts of labeled fitting data. The comparison therefore cannot isolate model size, language understanding, context compression or training method as the reason for any difference. It does not establish a law or a general advantage.

The original full 20-episode unseen-graph study remains unchanged, with its complete-cohort scoring gate. This separate pilot does not write or select its official results. The 320/80 collection continues, all 40 assessment questions remain untouched, and the separately frozen 20-fresh-question / 160-forecast ICL study remains pending.

## Artifacts

- `runs/icl_existing_graph_v1/predictions.json`: the immutable assistant P(A) values, keyed by case ID.
- `runs/icl_existing_graph_v1/prediction_results.md`: all 128 estimates with observed labels after disclosure.
- `runs/icl_existing_graph_v1/case_results.json`: predictions, labels and matched small-model probabilities.
- `runs/icl_existing_graph_v1/results.json` and `by_question.json`: summary scores and per-question/seed detail.
- `runs/icl_existing_graph_v1/inputs/`: complete training examples and input-only test packet.
- `runs/icl_existing_graph_v1/ready.json`, `committed.json`, and `completion_audit.json`: file hashes, timestamps and validation.
- `runs/icl_existing_graph_v1/icl_prediction_package.zip`: portable context, strategy, predictions and result package; no newly trained weights.

The driver is `experiments/icl_existing_graph_v1/pilot.py`. Its `score` command verifies the original commitment and returns the saved result without revising the predictions. Source conversations, original model files and the future fresh-question protocol remain unchanged.
