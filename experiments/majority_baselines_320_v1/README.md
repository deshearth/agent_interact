# Majority voting baselines

Completed November 14, 2026. Separate evaluation of six simple rules on the existing 320/80 development split. No source generation or new neural training. The 40 reserved assessment questions remain untouched.

See the [report](../../reports/majority_baselines_320.md), [driver](../../scripts/majority_baselines_320.py), and `runs/majority_baselines_320_v1/` for the frozen protocol, predictions, question-level scores and independent audit.

Reproduce with:

```sh
.venv/bin/python scripts/majority_baselines_320.py
```

Friendly-neighbor majority scores 82.656% validation accuracy; Ising 82.817%; the 818 transformer 84.092%. Ties and empty voting sets preserve the receiver’s current opinion. The global rule includes nonneighbors and therefore has a different information budget.
