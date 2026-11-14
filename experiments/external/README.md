# External dataset experiments

These exploratory studies test prediction of an LLM agent's next answer using
publicly released trajectories. They are separate from the original-paper
experiments and the frozen 80/20/40 HPC3 source-generation study.

## Folder layout

| Location | Contents |
| --- | --- |
| `experiments/external/<study>/` | Study instructions, configuration, and adapters |
| `experiments/external/common.py` | Shared provenance and file utilities |
| `data/external/<dataset>/<version>/metadata/` | Pinned upstream listings, licenses, acquisition manifests |
| `data/external/<dataset>/<version>/raw/` | Downloaded source files; never edited |
| `data/external/<dataset>/<version>/derived/` | Rebuildable examples and feature caches |
| `runs/external/<study>/` | Frozen split/protocol, training logs, checkpoints, predictions |
| `reports/external/<study>/` | Data audit, results, and figures |

Raw data and run artifacts are excluded by the repository's existing `.gitignore`.
Source data may be losslessly compressed; manifests record the upstream bytes'
SHA-256 as well as the local artifact SHA-256. Raw, derived, and result files must
not overwrite earlier versions. Future changes to cohort, labels, or split use a
new version or study name. Do not move or rewrite older project artifacts.

## Study index — 2026-11-09

| Study | Source and question split | Completed work | Report |
| --- | --- | --- | --- |
| [mallm_pilot_v1](mallm_pilot_v1/README.md) | DEBATE, simple responses / no personas; 60 train, 20 validation, 20 reserved | Data audit; the supplied draft predicts all 1,440 development updates | [Audit](../../reports/external/mallm_pilot_v1/audit.md) |
| [mallm_critical_pilot_v1](mallm_critical_pilot_v1/README.md) | DEBATE, critical responses / expert personas; 60/20/20 | Data audit; only two of 360 validation updates depart from the supplied draft | [Audit](../../reports/external/mallm_critical_pilot_v1/audit.md) |
| [coevolve_pilot_v1](coevolve_pilot_v1/README.md) | CoevolveSim, matched specialists / two graphs; 12/4/4 | Four baseline families and 12 matched-budget neural fits; validation only | [Results](../../reports/external/coevolve_pilot_v1/results.md) |

Neither MALLM cohort passes the recorded feasibility gate for an argument-model
comparison, so no argument predictors were trained on them. This conclusion is
limited to the two audited configurations. Their question pools share only six
questions; all six keep the same partition across studies. Different question
pools prevent a causal comparison of the discussion configurations.

The CoevolveSim attention model and pooled MLP each have **795 trainable
parameters**. Their validation errors are **2.58% and 2.52%**, respectively;
attention has slightly lower NLL. These are selected development scores on only
four statements, not an independent assessment or a strong architecture ranking.
Both models are trained from scratch. This dataset provides belief/role summaries,
so this pilot does not test full argument text.

All reserved assessment groups remain **unscored**. Schema and consistency checks
read source files, including reserved examples; do not describe the files as
unopened. No source LLM calls were made. The original experiment and frozen HPC3
study remain separate; 51 protected implementation/protocol files are unchanged.

## Finding the artifacts

- [Data index](../../data/external/README.md): source revisions, downloaded files,
  derived examples, and licenses. Each downloaded file has a provenance manifest.
- [Run index](../../runs/external/README.md): question splits, protocols,
  checkpoints, predictions, and superseded attempts.
- [Report index](../../reports/external/README.md): audits, results, and figures.
- Earlier literature reconnaissance remains in `runs/dataset_scout_20261109/`;
  it was not moved into these new experiment directories.

Run the focused checks from the repository root:

```bash
.venv/bin/python -m pytest experiments/external/test_external.py -q
MPLCONFIGDIR=/private/tmp/agent-interact-mpl-cache .venv/bin/python -m experiments.external.coevolve_pilot_v1.report
```

The report command reloads all 12 corrected neural checkpoints and reconstructs
the four baselines without fitting or assessment scoring. Per-study instructions
below the index record acquisition and preparation commands. Retain existing
source/split/protocol artifacts; use a new study name for further scientific changes.

## Scientific boundaries

- Predict the next agent answer, not the benchmark's correct answer.
- Never include the target agent's new response or later messages in its input.
- Group every trajectory of the same question into one partition. Preserve
  configuration/model/graph metadata rather than pooling protocols silently.
- Keep assessment partitions unscored until model selection is frozen.
  Data validation may read their schemas and metadata; outcome diagnostics are
  restricted to fitting and validation questions after the split exists.
- Native labels and interaction rules can require architecture and baseline
  changes. Record actual parameter counts; do not call an adapted network the
  exact 818-parameter original model or impose fictitious friendship signs.
- All results are external exploratory evidence. A null result does not prove
  that arguments or nonlinear update rules are intrinsically unhelpful.
