# Prediction of agent opinion distributions

The user proposes replacing sampled next-opinion labels with each source agent's next opinion distribution, and explicitly requests both input comparisons: current opinions alone versus opinions plus current distributions. This experiment uses the existing 320 training /80 validation conversations. The reserved 40 assessment conversations and the independent HPC3 collection remain untouched.

## Target and state

Primary target is the single-response probability `p(A)` at round `t+1`, with `p(B)=1-p(A)`. It comes from the recorded first-token A/B probabilities, normalized within A/B at the actual source temperature 0.7. This conditional distribution is fully recorded; its interpretation as a complete parsed-opinion distribution is approximate. All observations are retained, including rare high non-A/B-mass training cases; quality counts and mass are reported. Do not replace probability labels by hard outcomes. The majority-of-five probability is a separate secondary quantity, not the primary target.

Compare two available states: (1) current sampled majority opinions, (2) those same opinions plus current source probabilities. Next probabilities are labels only. Both states retain signed graphs and the original question/persona context. Argument models receive already-delivered messages with verified actual inbox order; their matched controls receive only current sender choices. Thus this is conditional prediction after message delivery, not an autonomous probability rollout before new messages are generated.

## Models fixed before fitting

| Family | Opinions as state | Opinions + current distributions |
| --- | ---: | ---: |
| Paper Ising / explicitly enriched Ising | 19 coefficients | 22 coefficients |
| Opinion/context graph transformer | 818 parameters | 826 parameters |
| Ordered transformer, choice-only control | 970 parameters | 978 parameters |
| Ordered transformer, actual arguments | 970 parameters | 978 parameters |

The richer Ising adds the receiver's current mean spin `2p_i-1` and friendly/unfriendly signed sums of neighbors' mean spins. Its original opinion drives remain. The pinned two-stage optimizer and regularization are unchanged. This enriched 22-coefficient model is distinguished from the original 19-coefficient baseline.

Each richer neural model adds one input channel per agent, making eight additional learned weights. Its new weights start at zero while common initial weights and RNG states match the opinion-only version. Argument and choice controls have identical parameter counts within each state. Frozen MiniLM (22,713,216 pretrained parameters) and training-only PCA8 are additional to these learned parameter counts; neither encoder weights nor validation-derived transforms are fitted.

## Selection and scoring

All models train against probability targets. Primary score is equal-question mean KL(source distribution || predicted distribution); zero means exact matching. Secondary scores are probability MAE, MSE, fraction of predictions within 0.05 probability units, and agreement with the source's most probable choice. These are distinct from accuracy on random majority labels; the old 91.4% expected-accuracy reference does not cap distribution agreement. No 95% success claim will be made by renaming the metric.

Reuse the prior graph/origin-stratified internal 256/64 split inside the 320 training questions. Fit preprocessing only on 256 for tuning and 320 for final fitting. Test nine learning-rate/output-penalty recipes per neural condition, seed42, selecting soft cross-entropy (equivalently KL because target entropy is fixed) and duration on the internal64. Refit each selected model with seeds142/143/144 on all320. Ising's fitting procedure is pinned. Freeze all18 neural refits and both Ising fits before outer80 scoring. There are54 neural tuning fits in this protocol.

Report average individual-seed scores, not ensemble or best-seed predictions. Use 5,000 paired whole-question bootstrap draws. The80 validation questions have been reused in prior experiments; comparisons remain exploratory. This protocol and all inputs/code are hashed before fitting. Artifacts live separately in `runs/distribution_prediction_320_v1/`; previous experiment files are read-only.

## Fixed descriptive references

Before any outer scoring, a separate `reference_protocol.json` was frozen at 2026-11-15 04:57 UTC (November 14, 10:57 p.m. Central). It adds two references without changing model selection: persistence predicts `p(t+1)=p(t)`, and a shared constant predicts the mean target probability calculated on the training 320 only. Paired KL/MAE differences compare the three enriched learned families (Ising, graph, ordered arguments) with persistence. These are descriptive controls for whether a model learns the change beyond carrying the current distribution forward. Their six contrasts are audited separately from the 60 primary contrasts.

## Completion and interpretation

Completed November 14, 2026 at 11:16 p.m. Central. All 54 tuning fits, 18 neural refits and two Ising fits finished in 37.8 elapsed fitting minutes. The independent audit reproduces 40 prediction files, 8,000 seed-question records, 60 primary contrasts and six fixed-reference contrasts, with at most 2.22e-16 numerical difference. All 18 neural models and both Ising models replay; six focused pre-training checks pass. The scientific figure was visually checked.

Current probabilities improve KL and MAE in all four model families. Validation MAE changes from 15.960 to 15.451 pp for Ising, 13.989 to 13.303 for the graph transformer, 13.530 to 12.772 for ordered choice, and 13.464 to 12.918 for ordered arguments. The graph transformers outperform their matched Ising baselines in KL and MAE. Arguments have no clear additional KL benefit; with probabilities they worsen MAE by 0.145 pp [0.058, 0.232]. This is evidence for enriching the numerical state, not for a new argument benefit in the frozen MiniLM/PCA8 representation.

The enriched ordered choice model has 12.492 pp training MAE and 12.772 pp validation MAE. The remaining discrepancy is visible on training data as well; these selected fits do not show a large validation-only gap. No prediction-accuracy ceiling is inferred from that observation. The reused 80 remain exploratory, the reserved 40 stay unused, and the independent source collection is unchanged.

See [full report](../../reports/distribution_prediction_320.md), [figure](../../reports/distribution_prediction_320.png), [numerical results](../../runs/distribution_prediction_320_v1/results.json), and [completion audit](../../runs/distribution_prediction_320_v1/completion_audit.json).
