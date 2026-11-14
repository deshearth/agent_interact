# History availability control

This follow-up was triggered by the validation screen, before any controlled test evaluation. It holds the six-feature history architecture, training examples, optimization protocol, and seeds fixed. The preceding opinion's two one-hot values are set to zero; its availability flag and all current-opinion features remain. Transform checks verify that changing past opinions cannot affect the control input and that the initialization input is unchanged. This isolates the incremental value of past opinion values within the tested architecture.

| Model | State | NLL, mean ± seed SD | Accuracy % |
| --- | --- | ---: | ---: |
| A0 | opinion | 0.28356 ± 0.00234 | 89.31 |
| A0 | availability_only | 0.27678 ± 0.00510 | 89.35 |
| A0 | history | 0.26856 ± 0.00474 | 89.53 |
| A2 | opinion | 0.27617 ± 0.00155 | 89.34 |
| A2 | availability_only | 0.27056 ± 0.00575 | 89.16 |
| A2 | history | 0.26566 ± 0.00425 | 89.49 |

Full history minus availability-only NLL (negative favors past opinion values), with descriptive paired four-question bootstrap intervals after averaging seeds:

- A0: -0.00822, interval [-0.02917, +0.00486].
- A2: -0.00490, interval [-0.00965, -0.00120].

For A2, 93.8% of the original history-versus-opinion NLL gap occurs at initialization, where no past opinion exists. The availability-only control recovers 53.4% of that aggregate gap. These are different descriptive comparisons, not an additive causal decomposition. The remaining full-history advantage over the matched control needs more independent questions; training-seed repeats alone cannot establish state sufficiency.

All six controls use the same validation keys and targets as the original matrix, have matching data/checkpoint hashes, and stop by the original training protocol. Run manifests are named `control_manifest.json` and stored under `runs/history_availability_control/`; standard CLI evaluation does not apply this adapter and is deliberately unsupported for these artifacts. The adapter and base-code checksums are retained in the plan and manifests. No test outcomes were used.

Reproduce: `.venv/bin/python -u scripts/history_availability_control.py`, then `.venv/bin/python scripts/report_validation.py`.
