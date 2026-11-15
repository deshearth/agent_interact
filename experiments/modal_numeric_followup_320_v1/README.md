# Numeric argument features: frozen transfer comparison

The [internal numeric screen](../modal_option_values_320_v1/README.md) passed its promotion gate. This separate experiment tests transfer without changing old models, source conversations or the collection workflow. Artifacts live in `runs/modal_numeric_followup_320_v1/`.

The promoted144 ordered numeric fields are appended to the state tree, fitted on320 and512 questions at the internally selected155 iterations. Matching state-only155-iteration controls distinguish numeric information from additional boosting iterations. Original105-iteration selected trees remain references.

Two2370-parameter history transformers have identical initial weights and dimensions. Both retain argumentPCA8. One also receives eight fixed numeric fields; the other receives zeros in those dimensions. Scaling preserves the prior semantic channel. Two learning rates are selected only on internal256/64, then three fresh seeds refit on all320. All four trees and six neural refits freeze before any new outer80 scoring. The new720 validation set and40 assessment questions remain unused.

Feature construction reproduces the internal numeric arrays exactly. Two focused checks verify matched initialization/semantic scaling and a live numeric gradient with target-independent inference. No correct-answer key, worked solution or future observation is supplied as a predictor feature.

```sh
.venv/bin/python scripts/modal_numeric_followup.py features
.venv/bin/python -m pytest tests/test_modal_numeric_followup.py
.venv/bin/python scripts/modal_numeric_followup.py run
.venv/bin/python scripts/evaluate_modal_numeric_followup.py
```

Completed and audited: all36 prediction files,20 new full-data outputs and8 tuning outputs reproduce; matched initializations and24 contrasts verify. The2370-parameter numeric transformer reaches88.672% versus88.405% for its matched control, +0.267pp [0.057,0.467]. The numeric tree reaches88.853% on320 and89.097% on512. Its512 gain over a duration-matched tree is+0.405pp [0.151,0.664]. The95% target remains unmet. The [report](../../reports/modal_numeric_followup_320.md) and [figure](../../reports/modal_numeric_followup_320.png) are complete; the figure was visually inspected. A [count-only control](../../reports/modal_numeric_identity_control_320.md) supports preserving the option-linked fields. A separate frozen512 neural replication follows.
