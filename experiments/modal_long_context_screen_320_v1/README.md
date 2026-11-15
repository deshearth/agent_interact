# Native full-context encoder pilot

Run data, pinned checkpoint metadata and progress live in `runs/modal_long_context_screen_320_v1/`. This reuses the exact20,480 source receiver prompts from the MiniLM internal pilot: 256 fitting/64 internal questions and fixed transitions0,4 only. No new validation or assessment episodes are loaded.

The public [ModernBERT-base checkpoint](https://huggingface.co/answerdotai/ModernBERT-base) at revision `8949b909ec900327062f0ebf497f51aef5e6f0c8` has149,014,272 frozen encoder parameters. Mean and first-token vectors provide1536 features, with PCA and standardization fitted only on the256 fitting questions. All20,440,255 tokens are retained; maximum prompt length3722. Native full-context attention replaces independent510-token windows. This does not fine-tune the encoder.

Predeclared heads: state+PCA32/128 trees; linear corrections to the frozen state-tree logit using PCA32/128/full1536 at four regularizations; calibration-only control. Promotion requires at least+0.5pp over the matched reference, including calibration control for linear heads. Passing only triggers an all-round study, not a95% claim. The old80 validation questions are not scored here.

The MPS benchmark passed its CPU check. Encoding is resumable in fixed shape buckets. The first download stalled and was restarted with the HTTP transfer path; no model fit or source generation occurred in that failed download.

```sh
# Encoding requires access to the Mac GPU.
.venv/bin/python scripts/modal_long_context_screen.py encode
.venv/bin/python scripts/modal_long_context_screen.py screen
.venv/bin/python scripts/audit_modal_long_context_screen.py
```

Completed and audited: all15 heads replay; six CPU prompt encodings, including the three longest, differ by at most5.90e-6. Best internal accuracy84.351% versus84.302% for the matched state tree, below the+0.5pp promotion gate. No outer scoring follows. [Report](../../reports/modal_long_context_screen_320.md). The encoding took1675 seconds; fitting uses a frozen encoder and does not test end-to-end adaptation.
