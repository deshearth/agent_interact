# Majority voting baselines for next opinion prediction

The strongest of the six simple rules tested is majority voting among friendly neighbors, with ties preserving the receiver’s current opinion. It reaches 82.66% validation accuracy, close to the fitted Ising baseline at 82.82%. The 818-parameter transformer reaches 84.09%. This gives a useful zero-weight baseline, while similar aggregate accuracy does not imply identical predictions or mechanisms.

All results use the existing MATH-based Llama-3.1-8B corpus: 320 training questions and 80 validation questions on eight familiar signed graphs. Validation contains 20,480 agent updates. Each prediction uses the observed opinions at time t to predict the receiver’s opinion at t+1. The target is the recorded majority of five source responses, not the mathematically correct option. The reserved 40 assessment conversations are not loaded or scored.

| Predictor | Training accuracy | Validation accuracy |
| --- | ---: | ---: |
| Always predict the training majority class | 62.584% | 60.615% |
| Keep own current opinion | 60.842% | 62.373% |
| Majority of all 32 agents | 68.693% | 70.059% |
| Majority of all communicating neighbors | 75.900% | 75.967% |
| Signed majority of communicating neighbors | 68.068% | 67.715% |
| Majority of friendly neighbors only | 82.830% | 82.656% |
| Ising baseline (19 fitted coefficients) | 83.081% | 82.817% |
| Opinion transformer (818 trainable parameters) | 84.422% | 84.092% |

## Rule definitions

- Plain neighbor majority counts only communicating neighbors, ignores relationship signs and excludes the receiver itself.
- Signed neighbor majority counts friendly neighbors normally and reverses every unfriendly neighbor’s vote: sign(sum_j J_ij s_j). It fixes both magnitudes at one.
- Friendly-neighbor majority counts only positive edges and ignores negative edges.
- Global majority counts all 32 current opinions, including the receiver and nonneighbors. It uses more information than a local inbox.
- For every vote-based rule, an exactly tied vote or empty voting set preserves the receiving agent’s current opinion. This policy is fixed before scoring.
- The constant baseline chooses B from the training set’s most frequent target class. No validation outcome chooses its label.

The voting rules have no learned weights and use no text, question embedding or persona embedding. Ising and the transformer receive their original context inputs and fitted weights. Neural accuracy averages the three frozen training seeds; it is not an ensemble. No likelihood scores are assigned to deterministic voting rules because no probability calibration was fitted.

## Comparison with friendly neighbor majority

Intervals resample the 80 validation questions in pairs, 5,000 times. They are pointwise, exploratory intervals on reused validation data, conditional on the fitted predictors.

| Fitted model minus friendly majority | Accuracy gain in percentage points | 95% interval |
| --- | ---: | --- |
| Ising baseline (19 fitted coefficients) | +0.161 | [-0.474, +0.767] |
| Opinion transformer (818 trainable parameters) | +1.436 | [+0.975, +1.911] |

The Ising accuracy advantage is small and uncertain in this comparison. The transformer retains an approximately 1.44-point advantage over friendly majority. Strong friendly-majority performance suggests that a simple aggregation rule captures much of the predictable behavior in this dataset. It does not establish a causal influence law or show that unfriendly messages never matter; the poor signed-rule result only rejects this specific equal-strength reversal heuristic.

## Ties and verification

| Voting rule | Validation updates requiring the tie fallback |
| --- | ---: |
| Majority of all communicating neighbors | 17.090% |
| Signed majority of communicating neighbors | 22.651% |
| Majority of friendly neighbors only | 22.236% |
| Majority of all 32 agents | 6.562% |

All six rules were independently reproduced by direct loops over every receiver and sender for all 102,400 development updates. Explicit empty-inbox, negative-edge and tied-vote cases pass. Saved predictions reproduce all 12 train/validation accuracy totals through independent confusion counts. Eight frozen reference prediction arrays reproduce the earlier Ising/transformer scores, and source checksums remain unchanged. There are no new neural fits or source-model calls.

Artifacts: `runs/majority_baselines_320_v1/protocol.json`, `results.json`, `predictions.npz`, `by_question.npz` and `completion_audit.json`. Reproduce with `.venv/bin/python scripts/majority_baselines_320.py`.
