# Ising versus transformer: training-capacity diagnostics

Completed 18 full-data diagnostic fits on the original GPT-4o-mini training cohort. The architecture is the unchanged **135,458-parameter opinion-based A2 graph transformer**, with dropout disabled. This is separate from the newer 282,146-parameter Llama/argument study.

All nine probability-imitation fits met the predeclared closeness criterion.
Selecting the learning rate separately for each seed using training objective only, mean absolute difference from the Ising probability is **0.022 percentage points**, with KL 1.24614e-06 nats. The observed-outcome transformer reaches **4.41% training error**, versus Ising 9.15%, the original validation-selected transformer 8.14%, and its old epoch-eight checkpoint 7.47%.

## What changed

The same 16 training questions, 512 episodes, 4,096 snapshots and 131,072 agent transitions are used throughout. Inputs are current opinions, the original 16-dimensional persona/question context, signed graph and node degrees. The original PCA and data are frozen. No arguments, new text representations, generator calls or test outcomes enter these fits.

There are two primary objectives. Ising imitation uses the frozen 19-coefficient Ising model's P(A) as a soft target at every existing input state. Observed-outcome fitting uses exactly the original next majority-of-five opinion. The soft-target loss has a nonzero entropy floor; KL to the teacher, not raw cross-entropy, measures imitation error.

Both stages use seeds 42/43/44, initial learning rates 0.0001/0.0003/0.001, AdamW with zero weight decay, zero dropout, gradient clipping 1, and 128 epochs (16,384 optimizer updates) per fit. Learning rate decays by a predeclared cosine schedule to 1% of its initial value. There is no validation-based stopping. Checkpoints are chosen by full-training objective at initialization, epochs 1/2 and every 4 epochs; a separate checkpoint records the lowest observed training error. Initial weights match across the primary objectives and learning rates within each seed.

The earlier selected models used dropout 0.1, weight decay 0.001 and validation selection at epoch 2, with optimization ending at epoch 8. The new comparison changes budget, regularization and learning-rate schedule together. It diagnoses whether stronger fitting is possible under this architecture; it does not isolate which individual training change causes the gain.

## Ising probability-imitation control

Means over three seeds. 'Training' uses observed training states; 'validation' uses the existing four development questions after all fitting is complete. This control evaluates reproduction of a known input-output function, not additional source-agent accuracy.

| Initial learning rate | Training probability MAE | Training KL | Validation probability MAE | Validation KL |
| --- | ---: | ---: | ---: | ---: |
| 0.0001 | 0.119 pp | 3.0383e-05 | 0.120 pp | 3.18696e-05 |
| 0.0003 | 0.051 pp | 5.68874e-06 | 0.050 pp | 5.4848e-06 |
| 0.001 | 0.022 pp | 1.24614e-06 | 0.021 pp | 1.11878e-06 |

Closeness was declared before fitting as training probability MAE ≤0.5 pp and KL ≤0.0005 nats. Reported continuous scores and validation transfer matter beyond the binary criterion. Successful imitation on sampled states does not prove exact representational inclusion for arbitrary inputs.

## Fit to observed opinions

| Model / initial learning rate | Training error | Training NLL | Existing validation error | Existing validation NLL |
| --- | ---: | ---: | ---: | ---: |
| Ising, frozen reference | 9.15% | 0.19539 | 11.91% | 0.28469 |
| Original transformer, validation-selected epoch 2 | 8.14% | 0.17877 | 10.59% | 0.27943 |
| Unregularized transformer, 0.0001 | 5.71% | 0.12027 | 10.49% | 0.35925 |
| Unregularized transformer, 0.0003 | 5.08% | 0.10735 | 11.83% | 0.42105 |
| Unregularized transformer, 0.001 | 4.41% | 0.09249 | 12.22% | 0.48890 |
| Unregularized transformer, rate selected on training NLL per seed | 4.41% | 0.09249 | 12.22% | 0.48890 |

The rate-selection summary is a capacity diagnostic, not a new generalization-optimized model. All nine rates/seeds are reported above or in the saved tables. Validation is scored only after fitting and never chooses checkpoint, rate, budget or stopping. Its four questions have been repeatedly used in previous exploratory work; no fresh confirmatory claim follows from these scores.

## Input conflicts and remaining optimization uncertainty

Grouping actual accessible receiver/sender features up to neighbor permutation gives 37,288 distinct groups, with 2,756 groups containing opposite outcomes. At least 4,466 of 131,072 labels must be misclassified by a deterministic predictor assigning one answer to each group: **3.41%**. The corresponding empirical cross-entropy lower bound is **0.06371** nats. The equality key excludes arbitrary graph/persona/question IDs, next outcomes and unobserved history; it includes the degree features actually supplied to each sender and receiver.

These are in-sample lower bounds. They are not a guaranteed attainable optimum for the architecture, an estimate of source-model Bayes error, or proof of sufficient state. Distinct feature groups may still be indistinguishable after the architecture's aggregation. No unique example identifiers were given to the learner.

The predeclared plateau description checks whether the best training objective improves by less than 0.0001 over the final 32 epochs. Plateau counts out of nine fits per stage are {'imitation': 9, 'observed': 0}. Every run used the fixed 128-epoch budget. Runs still improving at the end are budget-limited; none establishes a global optimization/capacity limit. The schedule itself also reduces late updates, so an observed plateau is not proof of convergence to a global optimum.

For the observed-outcome model with training-selected rates, its 4.41% total training error decomposes into 3.41 pp forced by conflicting labels, 0.26 pp additional errors on conflicting-input groups, and 0.74 pp on groups with consistent observed labels. Predicted classes agree within every identical-input group. This bookkeeping separates an established information bound from the remaining empirical fitting gap; it does not say that every remaining error is removable by this architecture.

The conditional Ising-plus-neural-correction stage was not run. Training-selected error 4.41% is at or below the predeclared 5.44% trigger; the main fitting runs closed at least half the earlier gap above the identical-input bound. Its training-only trigger and decision are preserved in `residual_trigger_plan.json` and `residual_decision.json`.

## Interpretation

Successful probability imitation shows that this architecture and optimization setup can closely reproduce the simple Ising mapping on the observed training states. It rules out an inability to learn that reference function on those states as a sufficient explanation for the earlier small fitting gain; it does not certify every possible function or state.

For the training-selected imitation rates, probability MAE on the separate development questions is 0.021 pp and KL is 1.11878e-06. Compare this with the training scores above when assessing whether the known rule transferred beyond the fitted question contexts. A small training imitation error alone does not establish such transfer.

The larger training-fit improvement shows that the earlier selected error was not the transformer's maximum demonstrated fitting ability. The stronger fit does not improve existing-validation NLL over the original selected model. This is consistent with overfitting the available training questions and outcomes; it does not identify whether missing state, limited independent questions, or other data properties dominate.

This diagnosis concerns the opinion/context graph predictor on the original GPT cohort. It does not resolve why adding argument vectors helps little in the separate Llama pilot, and it does not measure causal influence or an autonomous multi-round law.

## Checks and artifacts

Seven focused checks cover hard/soft objective equivalence, teacher entropy versus KL, count-weighted input bounds, no next-opinion leakage into teacher/features, neighborhood/permutation equality, exact Ising inclusion of a zero correction, and active learning/equivalence of the unregularized architecture. Every selected checkpoint was reloaded on CPU for 64 fixed training snapshots; maximum probability difference from saved MPS outputs is 3.87e-06. All 378 recorded files from the original study remain unchanged. The total recorded fit time is 33.2 minutes. No new LLM generation, paid API calls, consumed-test scoring or autonomous rollout occurred.

![Training diagnostics](/Users/melancholymirth/deshearthGit/github/agent_interact/reports/ising_transformer_capacity.png)

Curves show three-seed means for each fixed initial learning rate. The observed-error and NLL axes zoom into the fitted region, excluding the higher initialization values; full numerical histories include initialization.

```sh
.venv/bin/python -m pytest tests/test_ising_transformer_capacity.py
.venv/bin/python scripts/ising_transformer_capacity.py prepare
.venv/bin/python scripts/ising_transformer_capacity.py imitation --device mps
.venv/bin/python scripts/ising_transformer_capacity.py observed --device mps
.venv/bin/python scripts/report_ising_transformer_capacity.py decide-residual
.venv/bin/python scripts/report_ising_transformer_capacity.py evaluate --device mps
.venv/bin/python scripts/report_ising_transformer_capacity.py summarize
```

Completed fits and evaluations are reused only after provenance/hash checks. Models, curves, grouped inputs, all-rate scores, training-selected rates and validation outputs are under `runs/ising_transformer_capacity_v1/`. The conditional residual stage requires its decision file and is not launched by the primary commands.
