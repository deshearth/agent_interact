# Predicting changes in belief: internal test

**No candidate passed the promotion gate.**

Prediction rule: $\widehat{\ell}_{t+1}=\alpha\ell_t+f(X_t)-f(X_{t-1})$, with $\ell_t=\log[p_t/(1-p_t)]$.

The shared compact attention function sees current and previous inbox information: sender states, choice or argument tokens, friendship signs, exact positions and question/persona context. The previous inbox is empty before discussion. Current probability is an observed input; next probability supplies only the modal training label. Alpha is either fixed at1 or learned. The fixed version preserves current probability when both model inbox inputs are identical.

| Input / gain | Parameters | Training accuracy | Internal accuracy | Learned gain |
| --- | ---: | ---: | ---: | ---: |
| arguments_fixed | 968 | 85.518% | 84.442% | 1.0000 |
| arguments_learned | 969 | 85.959% | 84.778% | 0.6052 |
| choice_fixed | 968 | 85.339% | 84.332% | 1.0000 |
| choice_learned | 969 | 85.933% | 84.979% | 0.6042 |

The existing history-choice reference reaches 87.939% on the same internal64 questions; promotion required 88.439%. All8 rounds and32 agents are included. Two learning rates per condition and accuracy/NLL checkpoint selection were fixed in advance. These are internal tuning results, not outer-validation estimates.

This constrains the update through one shared potential difference. A poor result rejects this tested formulation as an improvement; it does not rule out every model of belief changes or prove an accuracy ceiling. Earlier ad hoc residual corrections also did not improve training accuracy.

All8 selected checkpoints and16 prediction arrays replay with 0 class disagreements. Maximum log-probability difference is 0; independent metric discrepancy is 5.6e-17. Three focused checks pass: causal previous-inbox alignment, fixed-gain identity/target independence, and nonzero gradients through both text branches.

An initial arithmetic identity test caught cancellation from evaluating `(current+new)-old`; the difference is now grouped as `current+(new-old)`. The interrupted initial fits are archived separately and excluded. No outer validation, new source calls or assessment scoring was performed.
