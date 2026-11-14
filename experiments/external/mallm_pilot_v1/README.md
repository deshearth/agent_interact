# MALLM simple-response feasibility audit

Source: [Multi-Agent-LLMs/DEBATE](https://huggingface.co/datasets/Multi-Agent-LLMs/DEBATE),
revision `8a16e9fdd59412887ba270e87189500272a53ee3`, configuration
`simple_nopersona_memory_simple_voting`. The exact source path, size, and SHA-256
are recorded in [config.json](config.json).

This is an audit of next recorded answers, not answer correctness. There are
100 distinct question texts, three agents, and seven turns. Entire questions are
assigned to 60 training, 20 validation, and 20 reserved assessment groups before
outcome diagnostics. The next-answer label is the released extracted `solution`;
the explanation's agreement flag is not a yes/no opinion label.

Inputs are reconstructed from pre-response `memory_ids` and the supplied current
draft. Every memory ID must precede the target. First responses without a previous
own answer are excluded. The new target response and future messages are never
features. The target is not an independent post-discussion opinion probe.

## Result and decision

Copying the current draft makes **zero errors on 1,080 training and 360 validation
updates**. There is one training change from the receiver's own last answer and
none in validation. An argument predictor cannot improve on the observed
copy-draft result, so no such model was trained.

The pre-diagnostic feasibility policy requires at least 20 training draft
deviations across five questions and five validation deviations. This practical
screen does not guarantee statistical power. Assessment outcomes remain unscored.
The follow-up [critical configuration](../mallm_critical_pilot_v1/README.md) has its
own data and run directories.

## Artifacts and commands

- [Source and prepared examples](../../../data/external/mallm/v1/): `raw/`,
  `metadata/`, and `derived/` (question metadata, transitions, visible messages).
- [Run records](../../../runs/external/mallm_pilot_v1/): split, feasibility policy,
  preparation identity and diagnostics.
- [Data audit](../../../reports/external/mallm_pilot_v1/audit.md).

From the repository root, acquisition verifies or reuses the pinned download;
preparation validates the saved split and regenerates the derived development audit:

```bash
.venv/bin/python -m experiments.external.acquire mallm
.venv/bin/python -m experiments.external.mallm_pilot_v1.prepare
```

Repository metadata from the initial literature audit is retained locally in
`runs/dataset_scout_20261109/` and copied into this dataset's metadata directory.
Do not change this study's configuration or split for a new experiment.
