# Interim training-question expansion: 320 to 512

Status: completed, independently audited and visually verified.

Target: all-case most probable next source A/B response on existing stochastic histories, after message delivery. This is not the sampled-majority target and not autonomous deterministic-agent rollout.

Use the first 192 new training units from the preexisting frozen collection order, plus the original 320 training questions. All selected units were complete. Keep the original 80 validation questions as the reused exploratory anchor. Newly collected validation and reserved 40 assessment conversations remain unopened.

Frozen representations: retain the original question/persona PCA and message PCA fitted on 320 training questions. Old inputs are byte-identical. Only added training texts are encoded with the same frozen 22.7M MiniLM, without truncation. No solutions, answer keys, screening statistics or future observations become inputs.

All 55,296 new prompt hashes and 276,480 votes were independently reconstructed locally. Maximum within-prompt probability variation is 0.0000506; no modes disagree across saved repetitions. The first original response defines the conditional A/B probability, matching the old dataset.

## Models and fixed budgets

| Model | Update parameters | Learning rate | Original / equal updates | Equal data passes |
| --- | ---: | ---: | ---: | ---: |
| choice978 | 978 | 0.0003 | 18,944 | 30,311 |
| history_arguments2306 | 2,306 | 0.001 | 4,608 | 7,373 |
| history_choice2306 | 2,306 | 0.001 | 6,144 | 9,831 |
| joint_arguments26210 | 26,210 | 0.0003 | 512 | 820 |

Three seeds (142, 143, 144) start from the original initialization for each neural condition. Each 512 run saves two predeclared checkpoints on one trajectory. No validation-based retuning. Ising22 and the semantic8 tree use their original fixed fitting recipes; the tree uses 105 iterations. All endpoints freeze before scoring the 80 validation questions.

Adding questions changes the topic mix under the approved larger-bank sampling policy. The original 320 are nearly equal across seven topics; the added 192 have 53 algebra, 39 intermediate algebra, 37 prealgebra, 26 number theory, 20 counting/probability, 9 geometry and 8 precalculus. This is an augmentation experiment, not a pure count effect in an identical population.

Six focused checks pass, including exact preservation of old model inputs and predictions, causal state alignment, split exclusion and unchanged initialization. A pre-fit lexical multiplication roundoff of 2.98e-8 was corrected to match old NumPy scalar precision; its unscored protocol/code are archived under `protocol_revisions/pre_fit_lexical_precision/`. No training or new outer scoring preceded the correction.

Artifacts live under `runs/modal_scaling_512_v1/`; source exports, derived features, fits, evaluation and protocol revisions are separated. The source export used a short free CPU job. GPU conversation generation and its automatic recovery were not modified.

## Results

Best neural accuracy is **88.514%**, from the argument/history model with equal data passes; best overall is **88.677%** from the tree. The 95% target remains unmet. The argument/history model improves by **0.143 pp [0.013, 0.277]** over its 320-question fit. Other accuracy-change intervals include zero. At 512, arguments do not clearly beat the history-choice control: +0.093 pp [−0.067, +0.254] at equal passes.

All 12 new neural fits (24 predeclared endpoints) and two baselines are complete. The audit exactly replays 36 neural and four baseline validation outputs, confirms 28 old prediction arrays unchanged, reproduces 80 prediction files and 30 paired contrasts, and verifies all 592 source episodes. Six focused checks pass. The figure is visually verified. See [report](../../reports/modal_scaling_512.md) and [figure](../../reports/modal_scaling_512.png).

Further training-only follow-ups are isolated in `runs/modal_inbox_screen_320_v1/` and `runs/modal_change_rule_320_v1/`. Their diagnostic scores do not enter the 512-question comparison.
