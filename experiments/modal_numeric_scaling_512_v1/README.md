# Numeric transformer: 320 to 512 training questions

The positive numeric-feature follow-up motivated this separately recorded exploratory replication. No new tuning followed the old320 validation result. Artifacts and frozen protocols live in `runs/modal_numeric_scaling_512_v1/`; [report](../../reports/modal_numeric_scaling_512.md).

Two2370-parameter history transformers retain the same frozen argumentPCA8, current distributions, observed history, actual message order and signed graph. The numeric arm appends eight exact rational-match fields; the control appends zeros. Six512 fits start from the exact original320 seeds and retain optimizer settings, transforms and batch sizes. Each saves two predeclared checkpoints: equal updates and approximately equal data passes. All12 checkpoints freeze before outer80 scoring.

Results, averaging three individual seeds:

| Model | 320 | 512 equal updates | 512 equal passes |
| --- | ---: | ---: | ---: |
| Zero-field control | 88.405% | 88.376% | 88.426% |
| Numeric argument fields | 88.672% | 88.924% | 88.960% |

At equal passes, numeric features improve accuracy by0.534pp [0.317,0.744] over the matched control. Increasing questions improves the numeric model by0.288pp [0.161,0.422]; the control changes by0.021pp [−0.101,0.146]. These are descriptive question-bootstrap intervals on reused validation, with no multiple-comparison correction. Revised topic proportions in the192 added questions limit a pure sample-count interpretation. No ensemble or validation-selected budget is used.

All36 prediction arrays/checkpoint outputs reproduce exactly;12 original320 prediction arrays are unchanged. Source targets, question order, initialization and21 paired contrasts verify. Two focused checks pass. New validation and reserved assessment remain unused. The95% target remains unmet; the larger HPC3 collection continues unchanged.

```sh
.venv/bin/python -m pytest tests/test_modal_numeric_scaling.py
.venv/bin/python scripts/modal_numeric_scaling.py run
.venv/bin/python scripts/modal_numeric_scaling.py evaluate
.venv/bin/python scripts/audit_modal_numeric_scaling.py
```
