# Arguments on the expanded Llama corpus

**Completed November 14, 2026.** All 36 tuning fits and 12 fresh-seed refits
completed in 61.4 minutes of fitting. Adding delivered arguments improved
training fit but lowered mean validation accuracy: **83.94% → 83.52%** at
3,890 parameters and **83.99% → 83.76%** at 184,610 parameters. Validation NLL
also worsened at both widths. The matched accuracy differences have descriptive
paired 95% intervals of **-0.81 to -0.03 pp** and **-0.55 to +0.09 pp**.
These are reused-validation results, not untouched-assessment confirmation.
The independent completion audit passed, including all 38 primary prediction
arrays, six replacement diagnostics, and 21 paired contrasts.

[Report and interpretation](../../reports/breadth_arguments_320.md) ·
[Figure](../../reports/breadth_arguments_320.png) ·
[Numerical results](../../runs/breadth_arguments_320_v1/results/summary.json) ·
[Completion audit](../../runs/breadth_arguments_320_v1/completion_audit.json)

This experiment tests actual delivered arguments on the existing 320 training and
80 validation questions. It is separate from the earlier 818-parameter
opinion/context/graph study, and from the eight-question Llama argument pilot.
The 40 reserved assessment conversations remain excluded. No new source LLM
calls or encoder training are required. The larger HPC3 source collection is
independent of this experiment.

The compact model extends the 818-parameter graph transformer with a learned
384-to-8 text projection, giving **3,890 parameters**. A larger counterpart
extends the 135,458-parameter model with a 384-to-128 projection, giving
**184,610 parameters**. Both keep the exact training-only question PCA, shared
known-persona features, current opinions and prescribed signed graph from the
completed comparison. Argument vectors receive no PCA. Frozen MiniLM weights
are additional to these trainable counts.

Each width is trained in two conditions: actual arguments and fixed
conclusion-only text determined by the sender's current A/B stance. The latter
uses the same architecture and optimizer without additional information beyond
current opinions. Both predict the original outcomes from full-message source
conversations; no source messages or outcomes are regenerated. Nominal parameter matching does not equalize input rank or
effective statistical capacity. Earlier opinion-only and Ising predictors are
reproduced as references.

The forecast boundary is **after the messages for the target update have been
delivered, before target votes**. This is a conditional response model; it does
not generate future arguments or support autonomous text-state rollout. The
source broadcasts one message per connected sender; all 358,400 delivered
edges were checked against that broadcast and their prescribed current stance.
There are 100,000 sender messages and 80,191 unique encoding strings including
two controls. Of those sender messages, 96,983 explicitly state the prescribed
choice using the audited phrase, 3,015 lack that standard phrase, and two
explicitly conflict with the prescribed choice. Language benefits can include
wording and stance cues, not only logical quality.

Nine learning-rate/output-penalty candidates per width/condition are tuned with
validation-NLL stopping. Selected settings and durations freeze before three
fresh-seed refits. The same 80 validation questions are used throughout, so
paired question-bootstrap intervals are descriptive development evidence.
The primary comparisons are arguments versus conclusion-only within each width.

Artifacts:

- Configuration: `configs/breadth_arguments_320.yaml`
- Preparation and resumable encoding: `scripts/breadth_argument_features.py`
- Models, resumable training, and evaluation: `scripts/breadth_argument_study.py`
- Independent score/checkpoint audit: `scripts/audit_breadth_arguments.py`
- Scientific checks: `tests/test_breadth_arguments_320.py`
- Data cache, protocol, checkpoints and results: `runs/breadth_arguments_320_v1/`
- Logs and report: `reports/breadth_arguments_320*`

Commands from the project root:

```sh
.venv/bin/python scripts/breadth_argument_features.py prepare
.venv/bin/python scripts/breadth_argument_features.py encode --device mps
.venv/bin/python scripts/breadth_argument_study.py fit --device mps
.venv/bin/python scripts/breadth_argument_study.py evaluate --device mps
.venv/bin/python scripts/audit_breadth_arguments.py
```

Encoding checkpoints each block of complete texts. Training saves optimizer and
random-number state. A bounded training session exits with code 75 and can be
resumed with the same command and backend. Existing scientific artifacts remain
immutable.
