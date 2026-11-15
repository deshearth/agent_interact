# Complete receiver-prompt representation: internal pilot

**No candidate passed the promotion gate.** On two fixed transition rounds within the old training questions, full-prompt encodings do not improve the matched state model. These are internal 256/64 results on two rounds, not scores on all 80 validation questions and not a claim about an attainable accuracy ceiling.

| Input | Training at selected step | Internal accuracy |
| --- | ---: | ---: |
| state | 92.786% | 84.302% |
| state_prompt32 | 91.241% | 84.180% |
| state_prompt384 | 91.113% | 84.155% |
| state_truth_oracle | 88.708% | 84.277% |

The baseline already includes observed probabilities, history, question/persona context, ordered states and argument PCA8. Prompt variants add the exact receiver-facing persona/question/options/friendly/unfriendly message prompt, encoded with frozen MiniLM. The `state_truth_oracle` column supplies the correct A/B answer only as a privileged diagnostic; it is never eligible to satisfy the prediction goal.

All 20,480 prompts are source-hash verified. All 21,639,598 tokens were retained. However, 19,973 prompts exceed510 tokens (maximum 4,406); MiniLM attends only within independent windows. This does not test a model with attention across the complete long prompt.

PCA32 fits only the256 fitting questions. Every candidate uses the same booster recipe and accuracy/NLL checkpoint selection. Promotion required at least+0.5percentage points over the matched state-only pilot. No outer validation or new source calls followed this failed gate.

Independent verification replays all4 models and6 full prompt encodings on CPU; maximum vector discrepancy is 1.4e-07. The initial oracle implementation mistakenly read the numeric correct-answer value instead of the A/B answer label, producing a constant diagnostic column. That invalid oracle fit is archived and excluded; the corrected A/B oracle was refitted. All legitimate features, targets, encodings and three non-oracle fits are unchanged.

The earlier best all-round validation results remain88.51% for a neural model and88.68% for the tree on512 training questions. The next separate hypothesis tests a shared function of new and previous inboxes to predict changes in log odds.
