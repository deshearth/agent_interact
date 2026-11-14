# Compact argument models on 320 training questions

The data are the existing MATH-based Llama-3.1-8B conversations: 320 training
questions and 80 validation questions. They are separate from the original
paper's GPT cohort.

This study extends the already tested width-four (314 parameters) and width-eight
(818 parameters) graph transformers with a bias-free projection of the delivered
sender argument into attention keys and values. Receiver queries, signed graph,
current opinions and the frozen 16-dimensional persona/question context remain
the same. The source agents broadcast one message per sender each round; this
broadcast property is verified for every delivered edge.

| Hidden width | Argument features | Trainable parameters | Execution |
| --- | ---: | ---: | --- |
| 4 | Training-only PCA8 | 346 | New |
| 4 | Full MiniLM384 | 1,850 | New |
| 8 | Training-only PCA8 | 882 | New |
| 8 | Full MiniLM384 | 3,890 | Reused from the separate, concurrently running argument study |

Width four is the smallest previously tested backbone, not a proven minimum
dimension for useful argument reasoning. The counts exclude the frozen
22,713,216-parameter MiniLM encoder and frozen PCA/context transforms. Argument
PCA uses 64,067 unique training messages and retains 26.893% of their embedding
variance. Its mean, components and scalar normalization are fitted using training
inputs only; no validation targets or vectors enter that fit.
The frozen message PCA itself stores 3,457 values (384 means, 8×384 components
and one scalar scale); these are additional to the reported trainable update-rule
parameters. The count is not a claim that the whole text-processing system has
only 346 parameters.

Every architecture has a separately trained conclusion-only control receiving
the fixed text “I favor option A.” or “I favor option B.” according to the verified
current sender stance. The two conditions have identical parameter counts,
initialization and optimizer/search policy. There are nine hyperparameter
candidates per model/condition and three fresh refit seeds after all choices
freeze. This adds 54 tuning fits and 18 refits, reusing six companion argument /
conclusion refits and six prior opinion-only refits plus the Ising baseline.

The 320 training and 80 validation questions, their 8 familiar signed graphs,
and every next-opinion target match the prior size sweep. There are 81,920 training
and 20,480 validation agent transitions. No source LLM calls or reserved
assessment conversations are used. This is a forecast after the message for the
update is delivered, before the next opinion is sampled. It is not an autonomous
rollout from opinions alone.

Primary contrasts compare argument and conclusion models by mean question NLL;
accuracy and Brier score are secondary. Question bootstraps average scores over
seeds first. Intervals are exploratory, pointwise, conditional on the fitted
models and based on the reused validation set. Gains would show usefulness of the
additional message representation, not necessarily mathematical reasoning or
causal influence.

A frozen-model sensitivity check substitutes each message with a different round
of the same question/sender/current stance. It preserves question, graph,
persona, prescribed stance and broadcast structure, while changing 96.287% of
message texts. It does not match receiver opinions, all other senders, round or
history, and is not a causal source-model intervention. This diagnostic is never
used to train, tune or choose the models.
Donor messages can come from later rounds, so the substitution procedure is an
offline diagnostic and cannot be used as a prospective forecast from state t.

All code and inputs freeze in `runs/compact_arguments_320_v1/protocol.json`.
The large message vectors are shared read-only from
`runs/breadth_arguments_320_v1/data/vectors.npy`. New transforms, models and
results live only under `runs/compact_arguments_320_v1/`; the companion study,
older studies and HPC3 collection are not modified.

Commands from the repository root:

```sh
.venv/bin/python -m pytest tests/test_compact_arguments_320.py
.venv/bin/python scripts/compact_argument_study.py prepare
.venv/bin/python scripts/compact_argument_study.py fit --device cpu
.venv/bin/python scripts/report_compact_arguments.py --device cpu
.venv/bin/python scripts/audit_compact_arguments.py
```

The fit command saves resumable checkpoints and has two-hour execution windows.
Repeat it after a budget exit (75) until all fits freeze. Reporting also requires
the companion study to have frozen its models. Actual results and the completion
audit are written only after the corresponding work completes.

## Completed result

All 54 tuning fits and 18 fresh refits completed November 14, 2026, followed by
automatic scoring and an independent audit. The audit passed at 22:23 UTC;
fit time was 78.4 minutes. Three-seed validation accuracies with arguments are
83.916% (346 parameters), 83.971% (882), 83.483% (1,850), and 83.524% (3,890).
The equal-parameter choice-only controls are 83.905%, 83.996%, 83.926% and
83.942%, respectively. Compressed inputs do not show a clear accuracy gain;
full vectors improve fitting but hurt validation. The 882 NLL improvement is
small and uncertain. All 40 assessment questions remain untouched.

See [the report](../../reports/compact_arguments_320.md) for training scores,
paired intervals and substitution diagnostics, and [the figure](../../reports/compact_arguments_320.png).
The final audit reproduced all 74 prediction arrays and 57 paired contrasts,
verified training-only PCA and actual message-branch learning, and preserved
all prior and companion artifacts. Both local training/postprocessing processes
completed successfully.
