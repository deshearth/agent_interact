# Iterative generalization investigation

Started November 14, 2026. The user requests a hypothesis → test → revised hypothesis loop aimed at approximately 95% validation accuracy. This means predicting every next majority-of-five agent opinion on the existing population, not solving the underlying mathematics questions. The target is aspirational, not a promised achievable score.

Primary development split: existing 320 training /80 validation questions. These validation outcomes have been reused and do not provide a fresh confirmatory assessment. The 40 reserved assessment conversations remain ungenerated and unused. Keep all old predictors, data and the independently running 3,200/800 expansion unchanged. Do not achieve the target by selecting only easy validation cases, exposing future observations, replaying source random seeds or silently changing labels.

## Initial hypotheses, fixed before new results

1. **The requested accuracy exceeds the predictable part of the target.** Export all existing source probabilities, verify correspondence to the saved votes and reconstruct prompt hashes. Compare the full-prompt majority-probability reference across all 80 validation questions, check calibration and the small first-token approximation. The earlier 20-question anchor is insufficient for a full-cohort claim.
2. **The compact state omits useful confidence and history.** Compare current signed opinion summaries with a declared richer set using only observations available before the target update: current five-vote counts, past opinions/votes, round and existing context. Future votes and source probabilities cannot be input features.
3. **Attention and compressed context impose an unhelpful fitting bias.** Fit nonlinear tabular predictors on the same and enriched states, with whole-question internal tuning inside the 320 training questions. Freeze models before comparing on the reused 80 validation questions. Compare original frozen Ising and transformer predictions.

Later hypotheses will be recorded before their own experiments, with the evidence that motivated each revision. Source probabilities may be training-only labels in a separately declared distillation experiment; the validation source probabilities are diagnostic references and never learned-model inputs.

Artifacts use `runs/generalization_investigation_320_v1/`. Each cycle has its own configuration, input/code hashes, predictions and results. A final report will distinguish observed accuracy, probability loss, irreducible sampling uncertainty under the checked source approximation, and any secondary tasks with altered targets.

## Cycle 1: state and source audit

The source uses a different, recorded message permutation for each recipient and round. This was absent from the previous permutation-invariant predictors. The first cycle added an actual-order / independent-order control before running six internally tuned boosted-tree variants. All selections used a 256/64 question split inside the 320 training questions; all final fits used 320. The external 80 were scored only after the six models froze.

Verified results: actual order 84.595% versus independent order 83.809%; paired gain 0.786 percentage points, question-bootstrap 95% interval 0.508–1.064. Richer state 84.849% versus the original 818 transformer 84.092%; gain 0.757 points, interval 0.400–1.131. Argument summaries 84.751%, so no additional accuracy gain over richer state. These are larger tree models, not similarly sized transformers. Prediction replay and independent metric/selection audits passed.

The complete source export contains 400 episodes and 115,200 agent prompts. Every reconstructed prompt hash and 576,000 saved votes matched. For all 80 validation questions, the approximate full-prompt Bayes reference is 91.415% expected accuracy (question interval 90.617–92.191%); its realized accuracy is 91.548%. It uses first-token A/B probabilities to approximate parsed responses: validation outside-A/B first-token mass averages 1.54e-7 and only 2 of 102,400 initial update responses are not bare A/B. This strongly questions95% as an attainable expected score on this stochastic target; it is not a universal impossibility theorem or an achieved small-model score.

## Cycle 2: revised hypothesis, fixed before outcomes

The remaining gap contains both sampling uncertainty and a predictable component. Test whether replacing noisy training-majority labels by precise source-majority probabilities improves learning on all 320 training questions. Compare hard and soft labels for the actual-order, rich-state and argument-feature trees. Both use exactly the same duplicate-row weighted loss representation, six-recipe search, inner-question selection by observed-label NLL, and model-freeze barrier. Teacher probabilities never become inference inputs. Keep all cases: training-only labels with nontrivial outside-A/B mass, non-bare responses or retries fall back to observed labels (38 of 81,920 labels). Validation probabilities are diagnostic only.

An initial unscored attempt stopped on a floating-point guard because the majority-probability polynomial exceeded 1 by about 1e-15. Its artifacts and code are retained under `failed_attempts/distillation_before_probability_clamp`; no outer scoring occurred and no fit is reused. The rerun clamps only numerical roundoff to the valid probability interval, with a regression check.

## Cycle 3: revised hypothesis, fixed before outcomes

Transfer the verified order signal into a compact ordinary-attention model, rather than relying exclusively on a much larger tree predictor. Use a shared receiver–sender pair encoder containing relationship sign and three positional features (within-sign rank, first, last). Compare actual versus independent random positions, each with choice-only versus PCA8 argument inputs. All four conditions have 970 learned parameters, 24 more than the previous 946-parameter ordinary-attention model. Reuse the 256/64 inner-question split and training-only transforms; select training duration/rate internally, refit three seeds on 320, then score all frozen models on 80. This cycle tests whether a specific missing state variable can improve a compact neural rule; 95% is not presumed achievable.

## Cycle 4: revised hypothesis, fixed before outcomes

The PCA8 message representation may blur option-specific wording. Training-only inspection by fixed patterns found explicit option mentions in 79,962 of 80,000 delivered sender messages; the last such mention differs from the sender's current majority in 1,831 messages. This is a textual-mention diagnostic, not proof that those messages recommend a contrary answer: messages can discuss both options.

Replace each message's PCA8 vector with eight transparent features: counts of uppercase standalone A/B, first/last mention, explicit option-A/B counts, last explicit option mention, and word count. Use exactly the same 64 receiver summaries as the prior argument model, keeping the total feature dimension 255. Also test the combined semantic-plus-lexical representation (319 features). All features come from already-delivered messages. The same six-recipe, inner-question-only tree procedure is fixed before fitting; both variants retain every update and use the original hard labels. This tests missing option-specific information without assuming the entire frozen text encoder is inadequate.

## Completed outcomes and revised conclusion

All four cycles finished with **96 tuning fits and 26 refits**. Every model was frozen before its outer-validation scoring; all four independent audits passed. The source audit verified all 400 episodes, 115,200 full prompt reconstructions and 576,000 raw votes. No source calls or assessment outcomes were used.

- **Probability supervision:** accuracy changes versus matched hard-label fits are −0.210, −0.024 and +0.010 percentage points for order/rich/argument features; every paired interval includes zero. Rich and argument NLL improve. Precise labels alone do not solve the generalization gap.
- **Compact ordered transformer:** the 970-parameter argument model reaches **84.526%**, versus **84.357%** for its matched ordered choice-only control and **83.874%** for randomized positions with arguments. Argument gain is +0.169 pp [0.023, 0.314]; order gain is +0.653 pp [0.439, 0.866]. The original 818 model remains 84.092%. Thus actual order exposes a small positive argument effect in this setup.
- **Explicit option wording:** the best observed accuracy is **84.980%**, versus 84.751% with the equally wide PCA8 argument summaries. Gain: +0.229 pp [0.015, 0.454]. This model has 283 trees /8,207 nodes, so it is not a similarly sized transformer. Combining both message representations gives 84.854%.

**The 95% target was not reached.** The approximately 91.415% full-prompt expected-accuracy reference, with a 90.617–92.191% question interval and checked formatting/calibration, makes 95% unsupported for the current sampled-majority task. Sampling explains about 53.5% of the original transformer's expected error. The best new model still has 6.486 percentage points of error beyond that reference; these experiments explain and reduce only part of the remaining predictable gap. Further work should assess stochastic response probabilities and this excess error, retaining observed accuracy as a complementary metric. No label changes or easy-case filtering were used to approach the requested number.

The final audit's first pass was stopped because repeated compressed-array reads made it unnecessarily slow. Materializing the same arrays once completed the identical edge-position checks: all 358,400 directed positions and 12 neural checkpoint replays pass, with zero replay discrepancies. This changed audit execution only, not any training or prediction artifact.

All results are exploratory on reused validation; the 40 assessment questions remain reserved. [Full report](../../reports/generalization_investigation_320.md), [figure](../../reports/generalization_investigation_320.png), [consolidated numerical results](../../runs/generalization_investigation_320_v1/report_v1/summary.json).
