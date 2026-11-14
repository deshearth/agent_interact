# MALLM critical-response feasibility audit

Source: [Multi-Agent-LLMs/DEBATE](https://huggingface.co/datasets/Multi-Agent-LLMs/DEBATE),
revision `8a16e9fdd59412887ba270e87189500272a53ee3`, configuration
`critical_expert_memory_simple_voting`. This follow-up was chosen after the
simple/no-persona cohort had no development departures from its supplied draft.

The [shared MALLM adapter](../mallm_pilot_v1/prepare.py) applies the same timing,
label, and feasibility checks. This cohort has 100 question texts, only six shared
with the first cohort. Those six inherit their original partitions; new questions
fill the 60 training / 20 validation / 20 reserved assessment allocation using
metadata only. These are different question pools, so the audit does not isolate
the effect of critical prompts or expert personas.

## Result and decision

There are 16 departures from the supplied draft among **1,076 valid training
updates**, across three questions. In **360 validation updates**, there are only
two departures, both on one question: **0.56% copy-draft error**. Four otherwise
eligible training transitions are excluded: three have an unparsed target, own
answer, or draft, and one lacks a complete set of visible agent states.
Keeping the receiver's own last answer gives 1.39% training and 1.67% validation
error.

This cohort also fails the recorded argument-comparison feasibility gate. No
argument predictor was trained, and assessment outcomes remain unscored. This
does not establish that other DEBATE configurations are uninformative.

## Artifacts and commands

- [Source and prepared examples](../../../data/external/mallm_critical/v1/).
- [Run records](../../../runs/external/mallm_critical_pilot_v1/), including the
  final inherited split and pre-diagnostic feasibility policy.
- [Audit](../../../reports/external/mallm_critical_pilot_v1/audit.md).
- An initial candidate split with one shared-question conflict was rejected
  before outcome diagnostics and retained under `superseded/`; it was not used.

From the repository root:

```bash
.venv/bin/python -m experiments.external.acquire mallm --config experiments/external/mallm_critical_pilot_v1/config.json
.venv/bin/python -m experiments.external.mallm_pilot_v1.prepare --config experiments/external/mallm_critical_pilot_v1/config.json
```

The primary MALLM split must already exist. Do not change either split or merge
these configurations into a new outcome-selected cohort under the same study name.
