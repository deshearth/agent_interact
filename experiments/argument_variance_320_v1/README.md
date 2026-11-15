# Accuracy and argument representation diagnosis

Completed November 14, 2026. This follow-up diagnoses the frozen 320-training/80-validation comparison and tests one specific explanation: ordinary PCA may preserve differences between questions at the expense of differences among arguments about the same question.

## Controlled experiment

The 882-parameter graph transformer and frozen MiniLM vectors are unchanged. Eight replacement text axes maximize within-question covariance on the 64,067 unique training texts. If a text occurs in multiple training questions, its unit weight is divided across them, preserving the ordinary PCA's weighting. All messages use a single global training mean and scale at inference; no held-out question means or future-message inputs are needed.

The two input conditions are actual delivered arguments and the sender's A/B choice alone. Each reuses its own original settings and duration: conclusion candidate c6 for 11,264 updates; argument candidate c7 for 18,432. Three paired seeds, 142/143/144, begin from exactly the original weights. There is no new hyperparameter selection. Six new refits took 4.69 fit minutes on CPU.

## Results

| Representation and input | Training accuracy | Validation accuracy | Validation NLL |
| --- | ---: | ---: | ---: |
| Ordinary PCA8, choice only | 84.375% | 83.996% | 0.35951 |
| Ordinary PCA8, arguments | 84.498% | 83.971% | 0.35764 |
| Within-question PCA8, choice only | 84.420% | 84.089% | 0.35872 |
| Within-question PCA8, arguments | 84.447% | 84.038% | 0.35611 |

The compression change improves argument accuracy by +0.067 percentage points, with a paired descriptive 95% interval of [-0.093, +0.226]. Under the new compression, arguments reduce NLL by 0.00261 [0.00031, 0.00501] versus choice alone, but accuracy changes by -0.050 points [-0.208, +0.106]. The improvement over ordinary argument PCA remains uncertain. This is exploratory reused-validation evidence, not a fresh assessment result.

The diagnosis records error overlap, corrections and introduced errors, vote-agreement breakdowns, probability margins and message variance. Logged source probabilities are available for the retained 80 training and 20 validation questions only, and are used exclusively for retrospective expected-error diagnostics. All 100 source episodes match the earlier probability audit.

## Artifacts and reproduction

- Configuration: [argument_variance_320.yaml](../../configs/argument_variance_320.yaml)
- Scientific driver: [argument_variance_diagnosis.py](../../scripts/argument_variance_diagnosis.py)
- Independent audit: [audit_argument_variance.py](../../scripts/audit_argument_variance.py)
- Report: [argument_variance_320.md](../../reports/argument_variance_320.md)
- Figure: [argument_variance_320.png](../../reports/argument_variance_320.png)
- Data transforms, protocol, six checkpoints, prediction arrays and audit: `runs/argument_variance_320_v1/`

```sh
.venv/bin/python -m pytest -q tests/test_argument_variance_diagnosis.py
.venv/bin/python scripts/argument_variance_diagnosis.py all
.venv/bin/python scripts/audit_argument_variance.py
.venv/bin/python scripts/report_argument_variance.py
```

The scientific driver returns without retraining once frozen artifacts exist, and verifies their identities. Four focused tests pass. Independent checks reproduce 68 prediction arrays and 30 paired contrasts, verify the covariance directly, compare global transforms, confirm identical initializations and durations, and replay model predictions. Previous models, source data, background HPC3 collection and all 40 reserved assessment conversations remain unchanged.
