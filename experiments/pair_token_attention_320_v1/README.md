# Ordinary attention with pair-specific friendship inputs

Completed November 14, 2026, at 7:31 p.m. Central. All 72 tuning fits and 24 fresh-seed refits are complete, in 45.8 elapsed fitting minutes (136.5 worker-minutes). Each incoming sender has a token specific to its relationship with the receiving agent, and the attention computation itself is ordinary scaled dot-product attention.

## Results

The replacement does not provide a clear accuracy improvement. All rows below include arguments and average three fresh training seeds.

| Hidden width / argument features | Parameters, previous → new | Previous validation accuracy | New validation accuracy | New choice-only control |
| --- | ---: | ---: | ---: | ---: |
| 4 / PCA8 | 346 → 346 | 83.916% | 83.923% | 83.947% |
| 8 / PCA8 | 882 → 946 | 83.971% | 83.986% | 83.965% |
| 4 / full384 | 1,850 → 1,850 | 83.483% | 83.158% | 83.983% |
| 8 / full384 | 3,890 → 3,954 | 83.524% | 83.385% | 84.025% |

The original 818-parameter opinion/context model remains at 84.092% and Ising19 at 82.817%. Compressed arguments have no clear benefit over the new matched controls; full vectors improve training fit but hurt validation accuracy and NLL. The architecture-change accuracy intervals all span zero. The 346-parameter argument model also has worse NLL than its previous counterpart. The [report](../../reports/pair_token_attention_320.md) includes every training/validation score, paired intervals, selected settings and the exact equations.

Five focused tests pass. The independent final audit reproduces 134 prediction arrays and 84 paired contrasts exactly, verifies 400 source episodes and all old/new frozen artifacts, confirms learned text/relationship branches, and finds zero replay class disagreements. The figure was visually inspected. Results are exploratory on reused validation; the 40 assessment conversations remain unused.

For receiver i and sender j, the token is `LayerNorm(GELU(Linear([u_j, tau_ij])))`. The shared sender state `u_j` contains its normalized state and projected frozen argument embedding. Receiver queries use its own state. One shared query, key, value and token encoder is used throughout; there are no learned pair IDs, edge-type logit biases, signed value gains or degree-dependent social gains. Existing graph-degree state features remain. This changes a bundle of architectural mechanisms, so it cannot isolate a single mechanism's effect.

## Frozen comparison

- The exact prior 320 training /80 validation Llama-generated MATH conversations, eight familiar signed graphs, 32 personas and eight updates. The 40 reserved assessment conversations remain unused.
- Four models: hidden width 4 or 8, with 8 PCA-compressed or all 384 frozen MiniLM message coordinates. Counts: 346, 1,850, 946 and 3,954 trainable parameters, respectively. These exclude the pretrained encoder and frozen transforms.
- Each model has an actual-argument condition and an equally sized choice-only control. Neither condition changes the source conversations or targets.
- Same nine learning-rate/regularization candidates, stopping policy and training settings as the prior study. One tuning seed; settings and duration then freeze before fresh refits at seeds 142, 143 and 144. All four models are reported. Total: 72 tuning fits and 24 refits.
- Exact old predictions are reused for the signed-attention counterparts, original opinion/context transformers and Ising19. Width-four counts match the old counterparts; width-eight counts increase by 64.
- Primary comparison: equal-question validation NLL, followed by accuracy and Brier score. Paired question bootstrap intervals average seed scores first; they are exploratory and conditional on fitted models. The reused validation set is not an untouched assessment set.
- The same post-fit, same-question/sender/stance message substitution is retained as an offline sensitivity diagnostic, with no causal or prospective interpretation.

The old models and the separate HPC3 collection are unchanged. Four local CPU workers each handle one model size; all tuning choices freeze together before refits begin. The training controller resumes checkpointed two-hour execution windows until this finite study finishes. Live phase and worker identities are in `controller_state.json`, and per-model progress is in `worker_progress/`. The initial serial fits and checkpoint are reused; `parallel_execution.json` records the execution-only change.

## Reproduction

```sh
.venv/bin/python -m pytest -q tests/test_pair_token_attention.py
.venv/bin/python scripts/pair_token_attention_study.py prepare
.venv/bin/python scripts/finish_pair_token_attention.py
```

Artifacts use `runs/pair_token_attention_320_v1/`; the final report is `reports/pair_token_attention_320.md`. The frozen protocol hashes code, input data and the previous predictions. Tests check ordinary-attention equivalence, receiver-dependent relationship encoding, locality, permutation equivariance, isolated agents, active text/sign gradients, target isolation and exact optimizer resumption.
