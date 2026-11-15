# Exact numeric option values

Training-only all-round comparison, internal 256/64 within the old 320. Data and immutable protocols live in `runs/modal_option_values_320_v1/`; [results](../../reports/modal_option_values_320.md).

Hypothesis: matching rational values in an incoming argument to the supplied A/B values retains useful information lost by generic embeddings. No correctness key, worked solution, future probability or future message is used.

The parser's focused tests found a sentence-final fraction boundary error before feature preparation or fitting. After correction all three checks pass. Both selected tree checkpoints and 768 raw messages replay. Ordered numeric fields increase internal accuracy from87.952% to88.483%, +0.531pp [0.226,0.830], passing the fixed +0.5pp promotion gate. Grouped fields reach88.428%. Intervals are descriptive after internal selection.

This triggered the separately frozen [full-data follow-up](../modal_numeric_followup_320_v1/README.md). No outer score was used to make that decision.

```sh
.venv/bin/python -m pytest tests/test_modal_option_values.py
.venv/bin/python scripts/modal_option_values.py run
.venv/bin/python scripts/audit_modal_option_values.py
```
