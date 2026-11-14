# CoevolveSim compact-predictor pilot

Source: [carlomarxdk/coevolve-sim](https://github.com/carlomarxdk/coevolve-sim), pinned
at `b5ca480ccf292d0fa6a531475abbf47433f3a342`. The cohort contains 20 factual
medical-indication statements, matched specialist roles (`experts`), and two
prespecified graph realizations (`erdos-renyi-01`, `watts-strogatz-01`). Each of
the 40 runs has 48 agents and ten updates. This is a bounded pilot, not the full
paper's graph cohort.

Whole statements are split into 12 training / 4 validation / 4 reserved assessment
groups, stratified by the source true/false label. The previously inspected
`false_0` schema sample is forced into training. Native next-belief labels have
three categories: true, false, neither. The graph is unsigned. Inputs include
current beliefs, model/role identities, visible neighbor composition, round, and
degree; they exclude question identity/text, ground truth, source confidence, and
future state. No argument text or pretrained text encoder is used.

## Completed pilot

Four baseline families are recorded: Markov transitions, local-opinion logistic,
local opinions plus receiver identity, and full role-by-belief aggregate logistic.
The pooled MLP and attention model each have **795 trainable parameters**, with
the same visible information. Both are trained from scratch at two learning rates
and three seeds, with validation NLL selecting checkpoints and learning rates.
See [training.json](training.json) for the fixed budget and [models.py](models.py)
for the native three-category adaptation of the original small predictor.

There are 11,520 fitting and 3,840 validation updates. Validation error is **2.52%
for the MLP** and **2.58% for attention**; NLL is **0.08359 and 0.08102**. Four
validation statements and model selection on those statements prevent a strong
architecture ranking. These rates are not directly comparable to the original
signed binary Ising study. All assessment groups remain unscored.

## Artifacts

- [Data](../../../data/external/coevolve/v1/): pinned metadata, 120 trajectory
  files with provenance, derived graph arrays, and identity vocabulary. The
  aggregate parquet files and schema sample are retained for provenance and
  were not additional training examples.
- [Run records](../../../runs/external/coevolve_pilot_v1/): split, preparation
  hashes, frozen training protocol, `baselines/`, `fits/`, selection, predictions,
  and independent reconstruction checks in `verification.json`.
- [Results and limitations](../../../reports/external/coevolve_pilot_v1/results.md),
  [data audit](../../../reports/external/coevolve_pilot_v1/audit.md),
  [metrics](../../../reports/external/coevolve_pilot_v1/metrics.csv), and
  [per-statement scores](../../../reports/external/coevolve_pilot_v1/validation_by_question.csv).
- `superseded/unmatched_budget_895_795/` preserves a preliminary fit with an
  accidentally oversized MLP, its exact source, and its results. The failed
  parameter-count check prompted the corrected 795/795 refits. Main reports use
  only the corrected fits; no assessment results motivated the correction.

## Commands

From the repository root, with the retained metadata and environment:

```bash
.venv/bin/python -m experiments.external.acquire coevolve
.venv/bin/python -m experiments.external.coevolve_pilot_v1.prepare
.venv/bin/python -m pytest experiments/external/test_external.py -q
.venv/bin/python -u -m experiments.external.coevolve_pilot_v1.train
MPLCONFIGDIR=/private/tmp/agent-interact-mpl-cache .venv/bin/python -m experiments.external.coevolve_pilot_v1.report
```

Only the final command is needed to reproduce the report from completed artifacts;
it does not fit or score assessment examples. The training driver verifies frozen
code/data identities and reuses completed neural fits. Further model searches,
cohorts, or split changes belong in a new study directory. Assessment execution
has intentionally not been implemented in this development pilot.
