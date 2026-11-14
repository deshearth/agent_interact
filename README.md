# Compact neural models of agent interaction

This project tests whether a small neural update rule and a compact state can reproduce the opinion dynamics of much larger language-model agents. The initial study uses the released GPT-4o-mini objective episodes from [Physics of Agents](https://github.com/batu-el/physics-of-agents), pinned to commit `62bf0e7ddb94e6be936bed6a35aefea94e8d54a9`.

The research specification is [mac_experiment_work_mode.md](mac_experiment_work_mode.md). [STATUS.md](STATUS.md) records execution progress and limitations. The latest completed result is the [compact argument comparison](#compact-argument-comparison), extending the [delivered-argument comparison](#delivered-argument-comparison) and following the [question-count by model-size learning curve](#fresh-question-comparison) and wide size sweep on 320 training questions. The earlier [comparison below 7,370 parameters](reports/transformer_micro_sweep.md) follows the [earlier transformer size comparison](reports/transformer_size_sweep.md), [Ising-gap diagnosis](reports/ising_gap_diagnosis.md), [nested generalization comparison](reports/transformer_generalization.md), [Ising/transformer capacity diagnostics](reports/ising_transformer_capacity.md), [new Llama-data transformer training](reports/local_llama_transformer.md), [local open-model pilot](reports/local_open_model_pilot.md), [argument-state investigation](reports/argument_investigation.md), [fuller question-context comparison](reports/question_context.md), [argument fusion and transfer diagnosis](reports/argument_mechanisms.md), and [question-count learning curve](reports/question_count_learning_curve.md). Results also include the completed [frozen training-only PCA test study](reports/inductive_test_results.md), its [validation/preprocessing comparison](reports/inductive_validation_results.md), and the development-only [argument-aware response diagnostic](reports/argument_response_results.md). The earlier author-context screen remains in [paper_binary_results.md](reports/paper_binary_results.md).

## Compact argument comparison

**Completed November 14:** The smallest tested argument-aware update rule has **346 trainable parameters**, four hidden dimensions and eight compressed message features. Counts exclude the frozen 22.7M-parameter MiniLM encoder and frozen transforms; this is not an absolute minimum. The comparison uses the existing **320 training / 80 validation MATH-based Llama questions**, three fresh training seeds per condition and matched choice-only controls.

| Parameters | Hidden width / text features | Choice-only validation accuracy | Argument validation accuracy |
| ---: | --- | ---: | ---: |
| 346 | 4 / 8 | 83.905% | 83.916% |
| 882 | 8 / 8 | 83.996% | 83.971% |
| 1,850 | 4 / 384 | 83.926% | 83.483% |
| 3,890 | 8 / 384 | 83.942% | 83.524% |

Compressed arguments show no clear accuracy gain. The 882-parameter model has a small, uncertain NLL improvement. Full message vectors improve training fit but worsen validation accuracy and NLL. The original 818-parameter opinion/context model remains at 84.092% and Ising at 82.817%. Active text gradients and offline message substitutions verify that the predictors use the messages; these results concern generalization under this representation and training procedure, not universal argument irrelevance.

All 54 new tuning fits and 18 refits completed; six companion refits were reused. The independent audit reproduced 74 prediction arrays and 57 paired contrasts, with all source/checkpoint and training-only PCA checks passing. The 40 assessment questions remain untouched. Results are exploratory because validation was reused for selection. See the [full report with intervals](reports/compact_arguments_320.md), [comparison figure](reports/compact_arguments_320.png), [experiment protocol](experiments/compact_arguments_320_v1/README.md), and [completion audit](runs/compact_arguments_320_v1/completion_audit.json).

## Delivered-argument comparison

**Completed November 14:** Actual delivered arguments were added to the graph transformer on the existing **320 training /80 validation questions**. The 818-parameter backbone plus a learned 384-to-8 message projection has **3,890 trainable parameters**; the larger counterpart has **184,610**. Frozen MiniLM message vectors retain all 384 components, without argument PCA. Both widths have a conclusion-only control with the same parameter shapes, receiving only “I favor option A/B” from the current opinion. All conditions predict the original full-message source outcomes.

| Predictor | Trainable parameters | Training accuracy | Validation accuracy | Validation NLL |
| --- | ---: | ---: | ---: | ---: |
| Ising reference | 19 | 83.08% | 82.82% | 0.38751 |
| Original opinion/context transformer | 818 | 84.42% | 84.09% | 0.35884 |
| Compact, conclusion-only control | 3,890 | 84.32% | 83.94% | 0.35894 |
| Compact, delivered arguments | 3,890 | 85.47% | 83.52% | 0.37338 |
| Larger, conclusion-only control | 184,610 | 84.70% | 83.99% | 0.36189 |
| Larger, delivered arguments | 184,610 | 85.57% | 83.76% | 0.37050 |

Neural scores average three fresh-seed models, without ensembling. Arguments improve training accuracy by **1.15/0.87 percentage points**, but validation accuracy changes by **-0.42/-0.23 pp** at the compact/larger widths (descriptive paired 95% intervals **-0.81 to -0.03 / -0.55 to +0.09 pp**). Validation NLL worsens at both widths. This supports overfitting under this representation and training procedure, not a general conclusion that argument information cannot help. Equal parameter count does not equalize effective capacity: the conclusion-only inputs have much lower rank.

All **36 tuning fits and 12 refits** completed in **61.4 minutes** of recorded fitting. Eight focused checks passed; the completion audit independently reproduced **38 prediction arrays, six replacement diagnostics and 21 paired contrasts**, and verified all **400 source episodes, 515 prior artifacts and 328 new frozen artifacts**. The text branches learn and affect predictions. MiniLM stays frozen; its 22.7M parameters are additional to the trainable counts. The same 80 validation questions select settings, and all **40 reserved assessment conversations remain untouched**. This is prediction after current messages are delivered, not autonomous rollout. Earlier models, source data and the separate HPC3 collection are unchanged.

[Full report](reports/breadth_arguments_320.md) · [Figure](reports/breadth_arguments_320.png) · [Design and commands](experiments/breadth_arguments_320_v1/README.md) · [Numerical results](runs/breadth_arguments_320_v1/results/summary.json) · [Completion audit](runs/breadth_arguments_320_v1/completion_audit.json).

## External datasets: first exploration

External work now has a separate [experiment index](experiments/external/README.md), with per-study code/configuration, raw and derived data, run artifacts, and reports. Two MALLM/DEBATE configurations were audited; copying the supplied current draft makes zero validation errors in the simple/no-persona cohort and only two errors in 360 updates in the critical/expert cohort. Neither passes the recorded feasibility gate for an argument comparison, so argument predictors were not trained on them.

The first [CoevolveSim pilot](reports/external/coevolve_pilot_v1/results.md) is complete: 12 matched-budget neural fits plus four baseline families, using 12 training and four validation statements. A 795-parameter attention model gives **2.58% validation error / 0.08102 NLL**, versus **2.52% / 0.08359** for the equally sized pooled MLP. This small validation-selected comparison does not establish an architecture winner or test argument text. Four assessment statement groups remain unscored, as do both MALLM assessment partitions. No new source LLM calls were made, and the original/HPC3 study remains separate.

## Fresh-question comparison

**Larger source collection launched November 14, 2:36 p.m. Central:** The user approved **3,200 training /800 validation conversations**, including the existing 320/80, and revised MATH sampling. The addition is **2,880 training /720 validation conversations** on fresh questions; the 40 assessment questions stay reserved. New data is isolated in `runs/question_breadth_3200_v1/`, with scheduling/recovery in `runs/hpc3_scale_3200_v1/` on HPC3. This is collection only; no automatic predictor training is scheduled.

The same single-rational-answer eligibility, source model, personas, signed graphs, eight updates and five-vote targets are preserved. All **6,243 fresh eligible, deduplicated MATH questions** will be screened. Because the source cannot provide 14,400 fresh eligible candidates or enough geometry/precalculus questions for the previous equal-topic quotas, the user approved **topic quotas proportional to availability** and selection of the highest pre-interaction entropy within each topic/A/B cell. A/B labels and final graph counts remain balanced. Selection fractions and topic composition change, so this is not an unchanged-population replication; entropy distributions will be recorded before collection, and the original 80 validation questions remain a common comparison anchor. No previous train/validation/assessment assignments are changed.

The operational limits are **24 GPUs in `free-gpu` plus 8 in `free-gpu32`**, accounting for other GPU jobs. The A30 runtime is reused; the L40S passed all original numerical replay checks and is admitted with batch 16. The RTX6000 probe remains separate until it passes. Initial submissions are 24 A30 workers, 7 L40S workers and one pending RTX6000 compatibility probe; actual running counts depend on Slurm. Each worker repeats the numerical replay, and a partially completed unit stays bound to its initial runtime and batch configuration. A five-minute cluster-side scheduler replaces preempted/time-limited jobs, advances screening→selection→collection→integrity audit, and removes its cron entry when finished or when attention is needed. Your Mac and SSH session are not needed after launch. Saved request groups and immutable outputs prevent repeated work or duplicate conversations.

All **nine preparation/recovery checks passed**, including an end-to-end synthetic selection/collection, unchanged requests across interruptions, exclusion locks, runtime binding, separate queue limits, stage barriers and cron idempotency. The independent cron invocation and a real preempted-worker replacement were verified; initial screening responses are saved. The prior two completed predictor studies' frozen artifacts remain unchanged. Protocol: [question_breadth_3200_v1/protocol.json](runs/question_breadth_3200_v1/protocol.json); configuration: [question_breadth_3200.yaml](configs/question_breadth_3200.yaml); local checks: [preparation audit](runs/hpc3_scale_3200_v1/local_preparation_audit.json); [launch snapshot](runs/hpc3_scale_3200_v1/launch_snapshot.json). The earlier completed prediction results below remain unchanged.

**Question-count by model-size study completed November 14, 6:45 a.m. Central, including independent verification:** More training questions improve the neural predictors, especially the largest model's probability estimates, but **818 parameters retains the highest mean validation accuracy and lowest mean validation NLL at every question count**. The largest model benefits more from extra data over 40–320 questions, without overtaking the smaller model. Ising changes very little.

All rows use the same **80 validation questions /20,480 next-opinion transitions**. At 40/80/160 questions, scores average three nested subset paths and three fresh training seeds per neural model; at 320, the existing three-seed endpoint is reused exactly once. These are mean individual-model scores, not ensemble predictions. Ising is fitted once per subset.

| Training questions | Ising (19 coefficients) | Transformer (818) | Transformer (7,370) | Transformer (533,026) |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 82.88% | **83.80%** | 83.70% | 82.84% |
| 80 | 82.82% | **83.81%** | 83.74% | 83.52% |
| 160 | 82.81% | **83.95%** | 83.84% | 83.82% |
| 320 | 82.82% | **84.09%** | 83.91% | 83.81% |

Validation probability loss (NLL; lower is better):

| Training questions | Ising (19 coefficients) | Transformer (818) | Transformer (7,370) | Transformer (533,026) |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 0.38757 | **0.36784** | 0.37384 | 0.39111 |
| 80 | 0.38770 | **0.36436** | 0.36884 | 0.37670 |
| 160 | 0.38755 | **0.36048** | 0.36334 | 0.36841 |
| 320 | 0.38751 | **0.35884** | 0.35893 | 0.36414 |

From **40 to 320 questions**, validation accuracy increases by **0.29 percentage points** for 818 parameters (descriptive paired 95% interval **0.10 to 0.50**), **0.21 pp** for 7,370 parameters (**-0.01 to 0.43**), and **0.97 pp** for 533,026 parameters (**0.70 to 1.26**). The largest model gains **0.68 pp more than 818** over that range (**0.33 to 1.03**), largely by recovering its poorer performance at 40. Its deficit to 818 narrows from **0.97 pp** at 40 to **0.28 pp** at 320. Ising's change is **-0.06 pp** (**-0.24 to 0.12**).

The **160→320 accuracy gains are small and uncertain**: **+0.14 pp** for 818 (**-0.01 to +0.29**), **+0.07 pp** for 7,370 (**-0.09 to +0.21**), and **-0.01 pp** for 533,026 (**-0.17 to +0.15**). Nevertheless, NLL improves by **0.00441** for 7,370 (loss difference **-0.00601 to -0.00295**) and **0.00428** for 533,026 (**-0.00601 to -0.00261**). More data can improve predicted probabilities without changing many predicted opinions. These observations do not establish saturation beyond 320 or show that another architecture or input representation cannot improve accuracy.

Small-data instability matters. At 40, the largest model's three subset-path mean accuracies range from **82.00% to 83.32%**, and its widest within-subset seed range is **1.54 pp**. The corresponding subset means at 160 range from **83.76% to 83.87%**. The [learning-curve figure](reports/breadth_320_learning_curve.png) shows subset ranges after seed averaging; these bands are not confidence intervals. Training scores and all seed/subset values are saved in the [complete results](runs/breadth_320_learning_curve_v1/results/summary.json). Training subsets change with count, so their training accuracies are not measurements on one fixed sample.

The three nested paths balance all eight graphs, seven topics, correct A/B labels, difficulty and retained/new-source origin. Every subset contains 25% retained and 75% added questions. The new 80 subsets are sampled from the expanded 320 pool and **are not the earlier retained 80 cohort**. Each question contributes one conversation with 256 agent transitions; this study jointly increases question diversity and transition count. It does not isolate additional distinct questions from additional conversations on fixed questions.

All conditions use the same current opinions, signed graph and 16-dimensional question/persona context. Arguments are excluded; MiniLM is frozen and excluded from the predictor parameter counts. Question PCA is fitted only on the selected training subset; known-persona PCA is shared. Each model/subset receives the same nine learning-rate/output-penalty choices, validation-NLL selection rule and stopping policy. Maximum updates and validation cadence scale with training count to preserve data passes, rather than fixing total compute. All settings/durations froze before fresh seeds 142/143/144 were fitted, and all weights froze before final scoring. No selected checkpoint reached its update cap. This tests the tuned pipeline, including subset-fitted PCA, rather than changing data count while holding preprocessing and settings fixed.

The intervals use 5,000 paired question-bootstrap draws after averaging fitted seeds and subsets. They are **pointwise, exploratory and conditional on these selected subsets/models**, without correction for repeated validation use or multiple comparisons. They do not include the full uncertainty of resampling training questions or retraining. The same 80 validation questions helped choose settings; the 40 reserved assessment conversations remain ungenerated, unloaded and unscored. No new unseen-graph scores, source conversations, argument models or encoder training were added.

All **243 new tuning fits, 81 fresh-seed refits and nine new Ising fits completed**; the full 320 endpoint was reused once. Recorded neural fitting took **125.1 minutes**. Six focused tests passed, including a full synthetic execution at all three actual architectures. Independent verification reproduced **all 200 saved prediction arrays and 44 paired contrasts**, with maximum metric/interval discrepancy below **1.12e-16**. All 150 GPU/CPU comparisons had zero class disagreements (maximum probability difference **5.67e-6**). Checksums verified **2,226 new frozen artifacts, 1,137 prior artifacts and 202 result artifacts**. The empirical figure was visually checked; prior results and pending ICL protocols remain unchanged.

Artifacts are isolated in `runs/breadth_320_learning_curve_v1/`: [results](runs/breadth_320_learning_curve_v1/results/summary.json), [frozen models](runs/breadth_320_learning_curve_v1/models_frozen.json), [independent audit](runs/breadth_320_learning_curve_v1/completion_audit.json). Configuration: [breadth_320_learning_curve.yaml](configs/breadth_320_learning_curve.yaml). Figure and provenance: [PNG](reports/breadth_320_learning_curve.png), [JSON](reports/breadth_320_learning_curve.json). Reproduce the audit with `.venv/bin/python scripts/audit_breadth_320_learning_curve.py`. The earlier completed size sweep below remains unchanged.

**Wide model-size sweep completed November 14, 3:19 a.m. Central; independent audit 3:20 a.m.:** Six transformer sizes from **314 to 533,026 trainable parameters** were compared on the exact same **320 training /80 validation questions**. Increasing size does not improve the mean validation score in this search. The **818-parameter model retains the highest mean validation accuracy and lowest validation NLL**. Scores below average three independently fitted models, without ensembling; NLL is better when lower.

| Predictor (trainable parameters) | Training accuracy | Validation accuracy | Validation NLL | Unseen-graph accuracy | Unseen-graph NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ising (19) | 83.08% | 82.82% | 0.38751 | 81.37% | 0.39111 |
| Transformer (314) | 84.26% | 83.94% | 0.35992 | 84.51% | 0.35240 |
| Transformer (818) | 84.42% | 84.09% | 0.35884 | 84.35% | 0.35486 |
| Transformer (7,370) | 84.59% | 83.91% | 0.35893 | 84.21% | 0.35593 |
| Transformer (34,978) | 84.75% | 84.02% | 0.36120 | 83.95% | 0.35770 |
| Transformer (135,458) | 84.73% | 83.92% | 0.36208 | 84.24% | 0.35804 |
| Transformer (533,026) | 84.28% | 83.81% | 0.36414 | 83.68% | 0.36334 |

Validation contains 20,480 next-opinion transitions on the eight familiar graphs. The unseen-graph columns use the previously completed **20 paired validation questions, four new graphs and 5,120 transitions**; these are a different question subset from the full validation column. All predictors use the same current opinions, signed graph and frozen 16-dimensional question/persona context. **Arguments are not inputs.** Parameter counts exclude the shared frozen 22,713,216-parameter MiniLM encoder. Every transformer retains one attention block and four heads; only its widths change.

Relative to 818 parameters, validation accuracy changes are **-0.15, -0.18, -0.07, -0.17 and -0.28 percentage points** for 314, 7,370, 34,978, 135,458 and 533,026 parameters. The first four descriptive paired intervals include zero. The largest model's interval is **-0.53 to -0.04 pp**, and its validation NLL worsens by **0.00530** (interval **0.00348 to 0.00718**). The 7,370-parameter model's validation NLL is essentially unchanged: difference **+0.00009**, interval **-0.00227 to +0.00252**. All six transformers still outperform Ising in mean accuracy and NLL; widening does not extend that advantage here.

On the reused unseen graphs, 314 parameters has the best observed mean accuracy, but its **+0.16 pp** advantage over 818 is uncertain (interval **-0.15 to +0.52**). The largest model is **0.67 pp worse** than 818 (interval **-1.27 to -0.11**) and has **0.00849 higher NLL**. These are 5,000-draw, paired question-bootstrap, pointwise exploratory intervals, conditional on the fitted models and this graph cohort. They are not adjusted for the model search or multiple comparisons. Four graphs from the same family do not establish broader graph-population performance, and close scores do not establish equivalence. The **40 reserved assessment conversations remain ungenerated, unloaded and unscored**.

The stopping diagnostic helps explain the similar selected training scores. For the **533,026-parameter model, tuning seed42**, continuing from its selected **1,536** updates to its terminal **6,656** updates increases training accuracy from **84.12% to 86.17%**, while validation accuracy falls from **83.75% to 83.55%**. Training NLL improves **0.35426 → 0.30955**, while validation NLL worsens **0.36225 → 0.38430**. The 34,978- and 135,458-parameter models also fit training data more closely at their later checkpoints while worsening validation loss. This supports overfitting under the current state and training procedure. It is one selected tuning run per size, not a maximum-capacity or unregularized memorization test, and does not establish an information ceiling or a universal interaction law. The primary table uses the three fresh-seed refits, not these diagnostic checkpoints.

All sizes received the same nine learning-rate/output-penalty choices, optimizer, dropout, weight decay, batch size, update cap and validation-stopping rule. Each size's validation NLL fixed its settings and duration before seeds142/143/144 were refitted; all final weights froze before unseen-graph scoring. The existing 818 fits, Ising coefficients, prepared data and PCA were reused exactly. **All 45 new tuning fits and 15 refits completed**, taking **78.8 minutes** of recorded fitting time. Selected refit durations are 20,992 /18,432 /15,360 /3,584 /3,584 /1,536 updates in increasing parameter order; no selected checkpoint reached the 32,768-update cap. This is a matched search within one architecture family, not proof that another architecture or optimization procedure cannot improve prediction.

Four focused checks passed, including a full synthetic rehearsal at every size. **All 76 saved prediction arrays independently reproduce the reported metrics** (maximum aggregate difference below 8.4e-17). CPU predictions match saved GPU probabilities within **2.63e-6**, with zero class disagreements in the 30 checked arrays. Checksums verify all **416 new, 95 original and 93 expanded frozen model artifacts**, plus the pinned prior results. Previous reference scores reproduce. No source LLM calls or encoder fitting occurred.

[Figure](reports/breadth_320_size_sweep.png) · [scores, intervals, seeds and stopping diagnostics](runs/breadth_320_size_sweep_v1/results/summary.json) · [frozen checkpoints and provenance](runs/breadth_320_size_sweep_v1/models_frozen.json) · [independent completion audit](runs/breadth_320_size_sweep_v1/completion_audit.json). New artifacts are isolated in `runs/breadth_320_size_sweep_v1/`. Configuration: `configs/breadth_320_size_sweep.yaml`; training/scoring driver: `scripts/breadth_320_size_sweep.py`; independent check: `.venv/bin/python scripts/audit_breadth_320_size_sweep.py`. The earlier 818 expansion results below remain unchanged.

**Expanded predictor comparison completed November 14, 1:20 a.m. Central; independently verified at 1:21 a.m.** Both models were fitted on **320 training questions** (81,920 agent transitions) and evaluated on the same **80 validation questions** (20,480 transitions). These are next-opinion predictions, not mathematical-answer correctness. The transformer scores average three fresh-seed models, without ensembling.

| Model fitted on 320 questions | Training accuracy | Validation accuracy | Validation NLL (lower is better) |
| --- | ---: | ---: | ---: |
| Ising, 19 fitted coefficients | 83.08% | 82.82% | 0.38751 |
| Transformer, 818 trainable parameters | 84.42% | 84.09% | 0.35884 |

The transformer's validation accuracy advantage is **1.27 percentage points**, with a descriptive paired question-bootstrap interval of **+0.73 to +1.87 points**. Its NLL is lower on 73/80 questions. The accuracy advantage remains modest.

The original frozen models were also scored on these exact same 80 validation conversations, with their original PCA unchanged:

| Model | Trained on 80 questions | Trained on 320 questions | Accuracy change |
| --- | ---: | ---: | ---: |
| Transformer, 818 parameters | 83.75% | 84.09% | +0.34 pp |
| Ising, 19 coefficients | 82.91% | 82.82% | -0.09 pp |

The transformer's paired accuracy-change interval is **+0.05 to +0.65 points**; its NLL improves from **0.36446 to 0.35884** (difference -0.00562, interval -0.00854 to -0.00286). On the **60 newly added validation questions alone**, its accuracy gain is **+0.32 points** with interval **-0.01 to +0.66**, while NLL improves by 0.00555. Thus the expanded run gives a small improvement, not evidence of a large accuracy jump. Ising's small change is consistent with sampling variation. These comparisons include refitting training-only PCA and retuning model settings/duration, not solely increasing the training count. Validation selected settings and duration; these bootstrap intervals are descriptive, not untouched-assessment confirmation. All **40 reserved assessment conversations remain ungenerated and unscored**.

Both models still receive current opinions, the signed graph and the shared 16-dimensional question/persona context; **argument text is not an input**. MiniLM's 22,713,216 pretrained parameters remain frozen and are excluded from both predictor parameter counts. Cached vectors for the original 100 questions and 32 personas are preserved; the 300 added questions use the same complete-window encoder. Question PCA uses the 320 training questions only. Each of the eight familiar graphs contributes 40 training and 10 validation conversations, so this is new-question generalization on familiar graphs.

Nine LR/penalty settings selected learning rate **0.001**, output penalty **0.001** and **18,432 steps**; seeds **142/143/144** were then fitted for that fixed duration. The step cap and validation interval were scaled fourfold to preserve the old maximum number of data passes. The 12 neural fits took **27.4 minutes** on the Mac GPU. Seven focused checks passed. The selected tuning checkpoint and three refits reproduce saved probabilities on CPU within **7.16e-7**, with zero class disagreements. Independent calculations reproduce every aggregate metric, the original validation scores reproduce on their retained 20 questions, and all 95 original frozen artifacts remain unchanged. One unparseable individual source vote did not invalidate any majority state or transition.

Artifacts are isolated in `runs/question_breadth_320_train_v1/`: [scores and paired intervals](runs/question_breadth_320_train_v1/development_results/summary.json), [frozen model manifest](runs/question_breadth_320_train_v1/models_frozen.json), and [completion audit](runs/question_breadth_320_train_v1/completion_audit.json). Saved transformer checkpoints: [seed 142](runs/question_breadth_320_train_v1/predictors/refit/main/nano-c7-s142/best.pt), [seed 143](runs/question_breadth_320_train_v1/predictors/refit/main/nano-c7-s143/best.pt), [seed 144](runs/question_breadth_320_train_v1/predictors/refit/main/nano-c7-s144/best.pt); [Ising coefficients](runs/question_breadth_320_train_v1/predictors/folds/main/ising.json). The [separate driver](scripts/breadth_expand_train.py) and [configuration](configs/breadth_expand_train.yaml) preserve all original scientific code. Reproduce development scoring with `.venv/bin/python scripts/breadth_expand_train.py evaluate --device cpu`; no source generation or assessment scoring occurs.

The separate [unseen-graph study](reports/unseen_graph_generalization.md) is also complete, including evaluation of the new 320-question predictors. On four unseen random graphs and the same 20 paired validation questions, the new transformer scores **84.35%**, versus **81.37% for Ising**: **+2.98 pp**, descriptive interval **+1.18 to +4.98**. Its familiar-graph score on those same 20 questions is **85.31%**; that is a different question subset from the 80-question table above. Compared with the old 80-question transformer on the exact same unseen conversations, accuracy improves from **83.91% to 84.35%**, **+0.44 pp** (interval **-0.15 to +1.00**). Most of the transformer–Ising accuracy gap is concentrated in one graph, and the graphs share the training graph family. This does not establish generalization to arbitrary topologies or a clear difference in graph-induced degradation. All weights remain fixed and the 40 assessment questions remain untouched. Follow-up artifacts are isolated in `runs/unseen_graph_320_v1/`. Earlier collection and original-model progress below are historical; see [STATUS.md](STATUS.md) for the audit trail.

A separate [assistant in-context pilot](experiments/icl_existing_graph_v1/README.md) is complete. After reading 256 training examples from eight questions, the assistant manually committed 128 forecasts before seeing their outcomes. Accuracy is **90.625%**, versus **87.760%** for the frozen818 transformer and **85.938%** for Ising on those same cases; assistant and transformer log losses are nearly identical. Only four fresh-graph conversations on familiar questions are included, so this does not establish a general advantage. The saved forecasting strategy, context and all probabilities are available in the report. The original full graph-study scoring protocol remains unchanged.

**The corpus expansion to 320 training /80 validation questions is complete.** The [expansion protocol and job guide](reports/question_breadth_320.md) retain the existing 80/20 split, screen 1,200 fresh candidates, and add 240/60 conversations under the same selection fraction and source-generation rules. Initial jobs were submitted to HPC3 on November 10; automatic recovery and a 24-worker pool completed collection on November 14. Source artifacts remain in `runs/question_breadth_320_v1/`; the completed original 80/20 comparison below remains frozen.

**Automatic recovery enabled November 13:** A cluster-side schedule checks every five minutes and runs a small free-CPU controller. It resumes preempted/timed-out workers and advances screening, selection, conversation generation and final validation without an open SSH session. The expansion now uses a shared pool capped at **24 GPU jobs total**, including unfinished graph-study workers; actual concurrency depends on free A30 availability. Original construction batches, prompts, sample IDs, numerical replay gates, question selection and all saved data are preserved. The operational plan, job history and current status are in `runs/hpc3_recovery_v1/`. At handoff, 670/1,200 candidates were screened and the separate graph study had 19/20 conversations. Thirty-two focused checks passed and all 1,352 pre-handoff question/screen files were verified unchanged. No new conversations or predictor training were complete at activation. Earlier static job schedules are historical; see [STATUS.md](STATUS.md) for the latest verification.

After logging into HPC3, run `python3 scripts/hpc3_recovery.py status` from the remote project folder for the latest controller snapshot, or `squeue -u yifengc7` for the live queue. `runs/hpc3_recovery_v1/cron_heartbeat.json` records independent scheduled checks. Creating `runs/hpc3_recovery_v1/STOP` disables new submissions and makes pool workers stop after their current request; an explicit cancelled worker is also respected. Ordinary preemption resumes automatically; code, data-integrity or repeated model-load failures require attention. The cron entry removes itself after both workflows finish or require attention. Collection completion does not trigger predictor retraining or assessment generation.

The original study uses **80 training, 20 validation and 40 untouched assessment questions**, comparing the **818-parameter transformer with the original 19-coefficient Ising baseline**. Both predictors receive current opinions, the signed graph, and the same question/persona context from frozen MiniLM with training-only question PCA. Arguments are generated during the source conversations but are not predictor inputs. This is a matched comparison on new Llama data, not a direct question-count comparison with the older GPT dataset.

All 560 MATH candidates were constructed and screened before the frozen 80/20/40 selection, covering all seven topics with balanced correct A/B labels. This cohort is limited to rational numeric answers. Each selected question has one 32-agent conversation with eight updates and fresh five-vote majority targets. Development collection finished on HPC3 at **11:40 a.m. Central on November 10**; all 80 training and 20 validation episodes were copied locally and verified by checksum. The pinned MLX model and sampling implementation passed full numerical replay on each final A30 allocation. See the [HPC3 package](hpc3/README.md), [parallel worker instructions](hpc3/PARALLEL.md), and [status history](STATUS.md).

**Predictor training completed at 2:49 p.m. Central on November 10.** Nine tuning settings selected learning rate `0.001`, output penalty `0.001`, and 1,280 steps. Three fresh-seed models were fitted on the same 80 training questions and frozen. Their mean scores are:

| Model | Training error | Validation error | Validation NLL |
| --- | ---: | ---: | ---: |
| Ising, 19 coefficients | 16.74% | 16.68% | 0.37556 |
| Transformer, 818 parameters | 15.41% | 15.11% | 0.35420 |

These are equal-question averages of individual-model scores, not ensemble predictions. **The 1.57 percentage-point validation error reduction is preliminary:** validation was used to select the settings. All **40 assessment conversations remain ungenerated and unscored**; the next planned stage is their collection on HPC3 with the predictor weights fixed. Targets are next agent opinions, not mathematical answer correctness. All 12 focused checks passed, and the selected checkpoint plus three refits reproduced on CPU within `8.35e-7` probability difference. Results: [development summary](runs/question_breadth_v1/development_results/summary.json); frozen weights manifest: `runs/question_breadth_v1/models_frozen.json`. The Mac source generator remains paused.

The [development error investigation](reports/question_breadth_error_diagnosis.md) finds substantial sampling variability and information in the incoming messages that neither predictor receives. Split-vote targets account for **81.3% of transformer errors**. Logged source probabilities give an approximate **8.58% full-prompt sampling-error reference**, leaving considerable error beyond sampling alone. A verified natural example has identical own/neighbor opinions and relationship signs but source P(A) changing from **0.66% to 98.69%** as message text changes. The transformer's gain comes mainly from opinions staying unchanged, and corrections partly cancel: **4.75 points of Ising errors fixed, 3.18 points of new errors introduced**. These are exploratory development findings; no new source calls or assessment scoring occurred, and the frozen primary comparison is unchanged.

The new [Llama question-count and probability-target learning curve](reports/llama_probability_learning_curve.md) keeps this source model, architecture and validation set fixed. With **10 / 20 / 40 / 80 training questions**, the original majority-label transformer's validation accuracy is **83.52% / 83.95% / 84.68% / 84.89%**, versus Ising's **82.71% / 83.29% / 83.19% / 83.32%**. Three balanced nested subset paths and three optimizer seeds test question-selection and initialization variability; all models receive the original selected settings and 1,280 updates. Training on logged majority probabilities gives **83.46% / 84.38% / 84.59% / 85.14%** accuracy. Probability loss improves more clearly at small sample sizes; the full-80 accuracy gain is small and its descriptive interval includes zero. The separate experiment completed 60 transformer fits and 20 Ising fits, with seven focused checks and CPU reproduction of all predicted classes. Artifacts are in `runs/llama_probability_curve_v1/`. The 40 assessment conversations and original frozen comparison remain unchanged.

A [paired unseen-graph experiment](reports/unseen_graph_generalization.md) is now submitted to HPC3's free GPU queue. It reuses the original frozen predictors and evaluates the same 20 validation questions on Jf0–Jf3, preserving each question's original initial votes. Four workers generate five new conversations each; the complete cohort is scored automatically after collection. This tests fresh random graph instances within the existing graph family, with no predictor refitting and no assessment-question use. Artifacts are isolated in `runs/unseen_graph_v1/`.

## Interpreting the 818-parameter predictor

A completed [pooled-state comparison](reports/pooled_state_comparison.md) tests whether individual-neighbor attention is necessary for its gain. A network using only the receiver context, own opinion, signed-neighbor opinion totals and degrees reaches **9.57%** older held-out error with **809 parameters**, versus **9.12%** for the graph transformer and **10.95%** for Ising. It captures about 75% of the graph model's observed accuracy gain. Greater capacity or twice the training duration improves fitting without clearly improving held-out classification. On the difficult fifth of updates, Ising/pooled/graph errors are 45.27%/38.43%/36.28%; the other four-fifths have about 2.17% Ising/graph error. These reused-development results support a strong simple baseline plus limited transfer of additional flexibility, while leaving the specific value of individual-neighbor features unresolved. No new assessment data were used.

A completed [explicit-rule investigation](reports/transformer_rule_distillation.md) runs alongside source collection. Forty-eight small rules imitate the saved transformer probabilities using only the older GPT development data. Even a 42-coefficient rule with context-dependent and quadratic influence terms leaves about 4.9 percentage points of probability error, on both fitting and older held-out questions. Its next-opinion error is 10.10%, versus the transformer's 9.12%.

Identical pooled inputs sometimes produce different transformer outputs, showing that aggregate neighbor counts lose information. However, the observed minimum probability error from those exact collisions is only 0.14 pp on older held-out states. This does not establish pooling as the main cause of the much larger formula gap. The tested formulas do not fully explain the predictor, and no claim of a universal law follows. Ten checks and CPU reproduction passed; the fresh 80/20/40 experiment remains separate and unchanged.

## Transformer comparison below 7,370 parameters

The new sweep reduces the transformer to **2,626, 818 and 314 trainable parameters**, preserving the same inputs and nested tuning policy. Even 314 parameters gives an outer-question error close to the 7,370-parameter reference:

| Transformer parameters | Training error | Outer error | Outer NLL |
| ---: | ---: | ---: | ---: |
| 7,370 (reference) | 7.72% | 9.17% | 0.21472 |
| 2,626 | 7.66% | 9.17% | 0.22036 |
| 818 | 8.02% | **9.12%** | **0.21258** |
| 314 | 8.19% | 9.22% | 0.21622 |

The 818-parameter model has the best mean scores, but its advantage over the reference is uncertain: -0.05 percentage points of error, descriptive paired interval [-0.23, +0.11], and -0.00214 NLL [-0.00975, +0.00545]. At 314 parameters, the model is 95.7% smaller and its error difference is +0.05 pp [-0.24, +0.36]. Close observed means do not establish equivalence. Original Ising and its current-opinion correction have 10.95% and 10.11% error, respectively.

All 252 new fits and 18 focused checks are complete. All 48 final checkpoints, including the reused reference, reproduce on CPU; prior artifacts remain unchanged. Some selected inner checkpoints reach the training cap, so this is a comparison under the common finite search rather than proof of a capacity limit. These are exploratory results on the same twenty previously studied development questions. The [full report](reports/transformer_micro_sweep.md) contains paired intervals, architecture details, cap diagnostics and reproduction commands. The 818-parameter model is a compact candidate for fresh assessment; no new assessment questions were generated here.

## Smaller transformer comparison

The completed size sweep retains the same inputs, question folds and six-candidate tuning policy. The 7,370-parameter model has **9.17% outer error / 0.21472 NLL**, compared with **9.32% / 0.22024** at 16,842 parameters and **9.48% / 0.22321** at 34,978. Original Ising has 10.95% error; adding a current-opinion correction gives 10.11%.

Shrinking from 34,978 to 7,370 parameters improves NLL on 13/20 questions and 3/4 folds. The paired NLL interval favors the smaller model; its 0.31-point accuracy gain remains uncertain. These are exploratory results on reused development questions. All 168 new fits and 13 focused checks are complete, and all 36 final checkpoints, including the reused reference, were independently reproduced on CPU. See the [full report](reports/transformer_size_sweep.md) for training errors, paired intervals and cap diagnostics. Regenerate its final figure with `.venv/bin/python scripts/plot_transformer_size_sweep.py` after the report driver; this keeps formatting changes separate from frozen numerical assessment.

## Nested question comparison for generalization

The matched study is complete: **216 tuning fits, 36 refits and 16 fold-specific Ising fits**. Four outer folds assess all 20 existing GPT development questions once; inner folds choose neural settings and stopping duration. PCA and Ising are fitted only within each training fold.

| Model | Trainable neural parameters | Outer-question error | Outer NLL |
| --- | ---: | ---: | ---: |
| Ising | 0 (19 fitted coefficients) | 10.95% | 0.22755 |
| Full transformer | 135,458 | 9.81% | 0.23282 |
| Small transformer | 34,978 | 9.48% | 0.22321 |
| Ising + small transformer | 34,978, plus the fitted Ising coefficients | 9.41% | 0.22354 |

The small transformer has a modest advantage over the full model, but its descriptive NLL and error intervals include zero. The Ising correction gives essentially the same NLL as the ordinary small model and only 0.07 pp lower error; an additional benefit is unresolved. NLL was the declared primary metric. These means cover a different question pool than the earlier four-question validation scores, and the development questions have been studied before. The [full report](reports/transformer_generalization.md) includes paired intervals, all four outer folds, model selection, reproduction checks and limitations. Fresh question coverage remains the next proposed data study; no new source responses or official test scores were added.

## Training capacity on the original GPT data

All 18 diagnostic fits are complete. The unchanged **135,458-parameter** graph transformer can reproduce the frozen Ising probabilities closely: mean probability error is **0.022 percentage points** on training states and **0.021** on development validation states.

| Model | Training error | Existing validation error | Validation NLL |
| --- | ---: | ---: | ---: |
| Frozen Ising reference | 9.15% | 11.91% | 0.28469 |
| Original selected transformer | 8.14% | 10.59% | 0.27943 |
| Transformer fitted for 128 epochs without regularization | 4.41% | 12.22% | 0.48890 |

The new row averages three seeds, choosing each learning rate on training loss alone. The larger fitting advantage demonstrates capacity that the earlier training and validation selection did not expose, while the validation deterioration shows overfitting. Identical-input label conflicts impose a 3.41% training-error lower bound for these features; the observed fits have not established the architecture's minimum error. The [full report](reports/ising_transformer_capacity.md) includes all rates, controls, checks, curves and limitations. This study uses the original GPT data and is separate from the 282,146-parameter Llama argument comparison below. Earlier model and test artifacts remain unchanged.

## Newly trained Llama-data transformers

All **24 new fits are complete**: opinion/history versus arguments, four whole-question folds and three seeds. The 512 natural updates span eight local Llama questions. Each fold uses four questions for fitting, two for checkpoint selection and two for assessment. The 282,146-parameter graph transformer is trained from scratch; the text encoder stays frozen.

| Inputs | Selected training error | Heldout question accuracy |
| --- | ---: | ---: |
| Opinion/history + context | 20.18% | 70.64% |
| Same inputs + arguments | 20.12% | 70.90% |

The argument gain is **0.26 percentage points**, with a descriptive question-bootstrap interval of -0.07 to +0.72. This exploratory pilot does not establish a reliable argument benefit. The target is the agent's next majority-of-five opinion, not mathematical correctness. Every checkpoint was independently reloaded and verified; earlier studies remain unchanged. See the [training report](reports/local_llama_transformer.md) for probability scores, paired argument-removal predictions, exact splits, checks and limitations. Artifacts are in `runs/local_llama_transformer_v1/`.

## Local open-model source pilot

The paper includes Llama-3.1-8B-Instruct, Qwen3.5-9B and Gemma-3n-E4B-it. A pinned local 4-bit Llama-3.1-8B model now runs in a separate MLX environment on the Mac; no paid API is used. All 1,920 controlled responses are complete. Holding a message's conclusion fixed, the mean absolute first-token answer-probability change is 2.96 percentage points for a minimal mathematical error, 6.69 for a valid paraphrase, and 15.73 for deleting the rationale. These are local-source sensitivity measurements, not predictor accuracy or a demonstration of logical understanding.

Eight fully local conversations are complete: **512 agent updates**, 3,840 opinion responses and 496 broadcast messages. All 512 updates were rerun with conclusions only. Among 489 nonempty inboxes with explicit matching stances, removing rationales changes first-token answer probabilities by **22.40 percentage points** on average and changes the more probable A/B choice in 130 cases. The 16 empty-inbox controls have exactly unchanged probabilities. A stricter post-hoc subset of 414 complete, consistently stated inboxes still shows a 21.35-point mean absolute change.

Direct validity grading is unreliable on the four constructed problems: balanced accuracy is 51.56% with counterbalanced A/B grading labels and 49.90% with VALID/FLAWED words. Thus text dependence and recognition of logical validity must be distinguished. The complete [report](reports/local_open_model_pilot.md) records the controls, artifacts and limits. A 512-pair teacher dataset is exported in `runs/local_llama_intervention_v1/teacher_pairs/`; the source-collection pilot did not train predictors. The subsequent authorized transformer training is reported above in a separate namespace. The signed graph and five-vote opinion rule remain intact. Questions are reused development questions, and the quantized Llama source differs from the earlier GPT teacher.

## Previous investigation: arguments and question identity

The trained graph transformer receives each delivered message separately through a frozen 384-dimensional MiniLM encoding, alongside opinions/history and prescribed relationship labels. It processes the vectors; the frozen encoder processes the raw tokens. Computational traces and a small memorization test verify that the individual-message inputs are connected.

The earlier familiar-question text benefit can be reproduced without individual argument language:

| Original PCA3 model input | Error on held-out conversations, same 16 questions |
| --- | ---: |
| No argument vector | 6.47% |
| Full individual arguments | 5.93% |
| One average vector per question | 6.01% |
| Question ID only | 5.92% |
| Question/stance ID only | 5.85% |

These are three-seed means with identical replica splits and stopping rules. IDs and centroids are familiar-question mechanism controls, not models for unseen questions. Their success shows that the added branch can help fit question-specific behavior without using message language; it does not establish that the source LLM ignores arguments.

Across 54 new comparison fits, full frozen question-conditioned argument encoding did not improve unseen-question prediction over its matched message-only control (10.41% versus 10.34% error). Isolating individual argument residuals improved training fit with full question context, but not held-out probability quality. Reconstructed inbox order did not provide a transferable remedy. Six small capacity fits reached zero training error on conflicting state-matched examples; 24 already-saved terminal checkpoints confirm that selected training error is not a capacity limit. See the [report](reports/argument_investigation.md) for subgroup effects and limitations.

That investigation used existing development data, with no new source-LLM calls or consumed-test scores. It prepared a refined 72-prompt argument-quality pilot in `runs/argument_investigation_20261108/controlled_probe_uncertain/`, selecting mixed-opinion inboxes where a frozen no-text predictor is uncertain. The draft stays frozen. The local study above executes a separately identified 96-prompt extension with a conclusion-only control, retaining the distinction between raw source responses and the predictor's majority-of-five target. Emergent credibility remains deferred.

```bash
.venv/bin/python scripts/report_argument_investigation.py all --require-complete
.venv/bin/python scripts/write_argument_investigation_report.py
```

## Local setup

The prepared environment is native macOS ARM64, Python 3.11.15, PyTorch 2.15.0. The package also runs on CPU. Commands below assume the repository is the current directory.

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m mac_agents.cli inspect-environment
```

`requirements-lock.txt` records versions actually installed and checked on this Mac. Other platforms may need their own compatible resolution from `pyproject.toml`. All weights, datasets, feature caches, and run checkpoints stay outside source control. No language-model generator or paid API is needed for the released-data experiment.

## Data and state conditions

```bash
.venv/bin/python -m mac_agents.cli prepare-data --download --raw
.venv/bin/python -m mac_agents.cli prepare-data --config configs/paper_binary_inductive.yaml
.venv/bin/python -m mac_agents.cli encode-messages --partition development --device mps
```

Acquisition resumes by verifying existing files against Git blob checksums. `data/data_manifest.json` records pinned URLs, sizes, SHA-256 checksums, and source licenses. The raw dataset includes 1,600 main episodes and 120 supplemental low-rank episodes; the latter cover ten test questions.

The original split is 16 training / 4 validation / 20 held-out test questions, seed 1729. Those test questions have now been scored in the frozen test study and are not fresh assessment data. Initial fitting uses 512 episodes on eight random graphs; validation uses 128 episodes on those graphs. Later mechanism studies split the 16 fitting questions by replica, as described above and in their protocols. Known training questions on unseen graph families retain `question_split=train` and have a separate evaluation role. Complete community snapshots stay together.

In the initial state comparison, every model receives the same compact question/persona context. The first configuration uses the author's PCA artifacts, whose question-fitting provenance is not established. The separate inductive configuration fits PCA to the supplied raw question-plus-options embeddings using only the 16 training questions.

| State | Information at the forecast boundary | Autonomous rollout |
| --- | --- | --- |
| `opinion` | Current majority opinions | Yes |
| `vote_counts` | Opinions plus current positive/negative/valid vote counts | Not implemented; requires a count-state update |
| `history` | Opinions plus a masked two-snapshot local history | Yes, using generated history |
| `existing_messages` | Opinions plus frozen encodings of prior inboxes | Not implemented; requires a message-state or latent-memory update |

The common forecast boundary is after opinion state `t` and before generating the next messages. Existing inboxes use `messages_history[t-1]`; initialization has an empty inbox. `messages_history[t]` is excluded because it contains new messages used to produce the target state. The parser preserves recipient-specific records. The initial message encoder pools each receiver's prior inbox by relationship sign; this is a deliberately limited representation, and failure of this encoder would not establish message irrelevance.

Message features use frozen `all-MiniLM-L6-v2`, with the exact encoder revision, normalization, and truncation statistics recorded in the cache. Its 22.7 million frozen parameters are separate from the trainable predictor. `encode-messages --partition all` prepares held-out features when required for a frozen message-model evaluation; it performs no fitting or scoring.

## Argument-aware response diagnostic

The separate argument experiment asks whether individual delivered messages add predictive information beyond communicated stances, opinions, and recent opinion history. It uses `messages_history[t]` to predict `s[t+1]`, **after message delivery**. It is a response predictor requiring observed messages, not an autonomous simulator. Sender/recipient associations and the existing prescribed relationship labels are retained.

`configs/argument_response.yaml` defines 18 development fits: fixed aggregation and attention, each with stance-only, full-text, and matched-substitution inputs, at three seeds. The substitutions exchange messages between different episodes with matching question, round, sender, recipient, relationship, communicated stance and current sender/receiver opinions. Training and validation stay separate; unmatched messages remain unchanged. Text uses the same frozen encoder, now without pre-averaging the inbox. All conditions retain the same two-snapshot history and training-only-PCA context.

```bash
.venv/bin/python -m pytest tests/test_argument_response.py
.venv/bin/python -u scripts/argument_response.py prepare --device mps
.venv/bin/python -u scripts/argument_response.py smoke --device mps
.venv/bin/python -u scripts/argument_response.py validation --device mps
.venv/bin/python scripts/report_argument_response.py
```

The study uses the existing 16 fitting and four validation questions, with validation also choosing stopping time. Its findings are exploratory; the preceding controlled test has already been inspected. This driver deliberately has no test or rollout command. See [the response diagnostic report](reports/argument_response_results.md) for results, matching coverage, question-level uncertainty and limitations. Protocol, optimizer/RNG checkpoints and predictions live in `runs/argument_response_v1/`; new features live in `data/processed/argument_response_v1/`. The implementation is isolated from the completed study, and code/config/data hashes guard reuse and resume. The default cohort cap is 60 minutes, excluding feature preparation and reporting.

The subsequent [training diagnosis](reports/argument_training_diagnosis.md) confirms the graph transformer was trained while MiniLM stayed frozen. Saved checkpoints show overfitting with continued training. A separate six-fit diagnostic multiplying text inputs by 32 did not rescue validation performance. The report also examines gradients, branch magnitudes, message substitutions, and target-vote disagreement. These analyses remain development-only; the precise representation/generalization cause is unresolved.

```bash
.venv/bin/python -u scripts/diagnose_argument_response.py --device mps
.venv/bin/python -m pytest tests/test_argument_scale_diagnostic.py
.venv/bin/python -u scripts/argument_scale_diagnostic.py --device mps
.venv/bin/python scripts/report_argument_diagnostics.py
```

## Question-count learning curve

The approved learning curve compares 4, 8, 12 and 16 distinct training questions while holding the four validation questions fixed. Four nested subset paths balance question composition at each size; three training seeds compare stance/history and full-argument A2 models. Each selected question retains all graph/replica/round records. Question PCA is refitted only on the selected training questions. Model size, text encoder and optimizer remain fixed.

Every subset receives a validation check after 128 optimizer updates, with six-check patience. This holds training opportunity constant despite different data-pass lengths. The 16-question preprocessing exactly matches the original cache, and a first-128-update GPU replay matches its original score; six existing endpoint models are reused once. The other sizes require 72 new fits, isolated in `runs/question_count_curve_v1/`.

```bash
.venv/bin/python -m pytest tests/test_question_count_curve.py
.venv/bin/python -u scripts/question_count_curve.py prepare --device cpu
.venv/bin/python -u scripts/question_count_curve.py smoke --device mps
.venv/bin/python -u scripts/question_count_curve.py run --device mps
.venv/bin/python scripts/report_question_count_curve.py
```

The [completed learning curve](reports/question_count_learning_curve.md) includes training and validation **error rates**, NLL, question-level contrasts, subset/seed variability and provenance checks. Argument-model validation error falls from 12.40% at four questions to 10.67% at sixteen, with most classification improvement from four to eight; the stance/history control remains better on average at every count. Mean NLL improves through sixteen questions for both inputs. The study is development-only and cannot establish a scaling law beyond the available training questions. It measures the full pipeline as both training examples and fitted question features change. All 72 new fits stopped by early stopping; the six reused endpoint models are counted once. Consult `STATUS.md` for details.

## Argument fusion and transfer diagnosis

The [mechanism diagnostic](reports/argument_mechanisms.md) separates held-out conversations about familiar questions from unseen-question evaluation. It fits replicas 0/1 of all 16 training questions, selects stopping on replica 2, and evaluates replica 3 plus the four existing unseen development questions. Entire conversations are disjoint. All inputs retain the original prescribed signed graph and message timing.

Fifteen fits compare the original additive text fusion with an independently normalized text branch in the attention keys and values. Both include full and matched-shuffled text at seeds 42–44. They share a stance-only control whose function, initialization and optimizer updates are verified identical without text. The alternative has 210,466 parameters. Shuffles are rebuilt within each role; familiar-question matching coverage is limited to about 14.5% of messages. A separate read-only probe replaces every familiar-question message with a fitting-only question or question/stance centroid, checking dependence on individual-message variation.

```bash
.venv/bin/python -m pytest tests/test_argument_mechanisms.py --junitxml=reports/argument_mechanism_checks.xml
.venv/bin/python -m pytest tests/test_argument_centroid_probe.py --junitxml=reports/argument_centroid_checks.xml
.venv/bin/python -u scripts/argument_mechanisms.py prepare --device cpu
.venv/bin/python -u scripts/argument_mechanisms.py smoke --device mps
.venv/bin/python -u scripts/argument_mechanisms.py run --device mps
.venv/bin/python -u scripts/probe_argument_centroids.py
.venv/bin/python scripts/report_argument_mechanisms.py
```

All 15 fits and seven focused checks completed. Added text improves familiar-question error from 6.47% to 5.93%, but worsens unseen-question error from 9.84% to 10.84%. Replacing every familiar-question message with its training-derived question/stance average changes error only to 5.99%. The alternative text branch performs worse than additive fusion. The findings point toward shared question content and limited transfer, while leaving encoder quality and meaningful argument variation unresolved.

These are exploratory diagnostics with frozen MiniLM embeddings, no new LLM calls and no test scoring. Neither evaluation cohort selects checkpoints in this study, but both have been used in previous research. They contain different questions and are not paired with each other. The new split halves the earlier fitting data and changes stopping selection; comparisons to earlier full-data scores do not isolate architecture. The mechanism and centroid protocols, inputs and predictions occupy separate namespaces under `runs/`.

## Fuller question-and-choice context

The question-context study tests whether the earlier three-component question representation hides information useful for interpreting arguments. It adds the complete cached 1,536-dimensional question/choice embedding to both no-text and argument predictors. An equally sized control reconstructs only the original three principal components into the same input width. Both new branches have 382,498 parameters and the same initialization and training settings. Nine unchanged 185,890-parameter models from the preceding study provide the original reference.

All 18 new fits keep the preceding conversation split: fit replicas 0/1 of 16 questions, select stopping on replica 2, and evaluate replica 3 plus four unseen development questions. Centering, PCA and normalization use fitting questions only. The complete cached input strings are verified against each problem and its A/B choices. Message encodings, opinion history, prescribed graph signs, and donor substitutions remain fixed. No new API calls are needed.

```bash
.venv/bin/python -m pytest tests/test_question_context.py --junitxml=reports/question_context_checks.xml
.venv/bin/python -u scripts/question_context.py prepare --device cpu
.venv/bin/python -u scripts/question_context.py smoke --device mps
.venv/bin/python -u scripts/question_context.py run --device mps
.venv/bin/python scripts/report_question_context.py
```

All 18 new fits and six focused checks completed. Richer question features modestly improve training fit, but do not improve argument-model accuracy over the matched control: training error changes from 5.74% to 5.55%, familiar-question error from 6.01% to 6.02%, and unseen-question error from 10.35% to 10.36%. Paired uncertainty leaves the unseen effect unresolved; no clear argument benefit is unlocked. The [question-context report](reports/question_context.md) includes all input controls, seeds, question-level comparisons, losses and provenance checks.

This tests compression of frozen question embeddings. It does not train a token-level reasoning encoder or jointly encode a question and argument. All results remain exploratory on reused development data; the existing test set is not rescored.

## Preparation checks and baseline reproduction

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/benchmark_predictor.py
.venv/bin/python -m mac_agents.cli reproduce-baseline
.venv/bin/python scripts/preparation_smoke.py
.venv/bin/python -m mac_agents.cli report --experiment preparation
```

The tests cover graph locality, one-hop timing, permutation equivariance, isolation, routing/context separation, dimensional contracts, forecast timing, train-only feature fitting, test gating, and exact CPU training resume. The CPU/MPS agreement test runs when MPS is visible. A sandbox can hide the GPU even when the machine supports it; device availability is recorded for each execution.

Author reproduction imports the pinned fitting utilities and uses the original float64 preprocessing, 20 official training questions, two-stage signed coupling fit, masks, and metrics. Its official test evaluation is a separate reproduction check. Controlled comparison models keep their test partitions locked until selection is frozen.

Smoke runs use small real training/validation subsets and are explicitly ineligible for final selection. They validate execution, not research accuracy. The predictor microbenchmark measures a real batch on CPU and MPS; it is not an LLM speedup measurement.

## Run the controlled comparison

The one-seed screen includes P0 persistence, P1 smoothed Markov, I0 author Ising, A0 fixed aggregation, A1 attention without context in routing, and A2 contextual attention. A0 and A2 are also compared with each enriched state. State/architecture combinations and their inputs are recorded separately.

```bash
.venv/bin/python -m mac_agents.cli screen --minutes 120 --device mps
```

This applies an aggregate two-hour wall-time cap and saves progress to `runs/screen_plan_paper_binary.json`. It does not select models or evaluate the controlled test set. If the budget expires within a run, rerunning the screen resumes that checkpoint. An abrupt process termination is reported for manual checkpoint recovery to avoid silently duplicating a run.

After the one-seed screen, repeat the complete neural matrix with seeds 43 and 44 and evaluate autonomous validation rollouts. The follow-up command has a separate conservative cap of 60 minutes; only run it within the original remaining predictor budget.

```bash
.venv/bin/python -u scripts/validation_repeats.py --minutes 60
.venv/bin/python scripts/report_validation.py
```

The follow-up plan is saved to `runs/validation_repeats_plan.json`. It includes all state controls, rather than selecting from one seed, and remains validation-only. The report script verifies identical prediction keys and targets, recomputes saved scores, checks checkpoint hashes, and produces seed summaries, paired question-level intervals, and rollout plots. Use this script to regenerate `reports/paper_binary_results.md`; the older CLI `report` command is a run inventory and should use a separate experiment name.

The round breakdown motivated a further matched control: remove previous-opinion values while retaining the history-availability flag and the same architecture. Its six validation fits and results are in [history_availability_control.md](reports/history_availability_control.md).

```bash
.venv/bin/python -u scripts/history_availability_control.py
.venv/bin/python scripts/report_validation.py
```

This standalone adapter records its own checksum and stores `control_manifest.json` under `runs/history_availability_control/`. Standard CLI evaluation does not implement that feature transform; these control artifacts must not be evaluated with the ordinary CLI. The report includes them automatically after the control plan completes.

Individual runs are also available:

```bash
.venv/bin/python -m mac_agents.cli train --model A2 --state history --seed 42 --device mps
.venv/bin/python -m mac_agents.cli train --model A0 --state existing_messages --seed 42 --device mps
.venv/bin/python -m mac_agents.cli train --config configs/paper_binary_inductive.yaml --model A2 --state opinion --seed 42 --device mps
```

Use the same state and matched examples for architectural comparisons. For limited message-cache experiments, use `--smoke --limit-episodes 2`; `--match-state existing_messages` restricts an opinion comparator to the same cached episodes. Full message training refuses incomplete development caches.

Each run retains its resolved configuration, data/split/code hashes, actual device, checkpoints, RNG and optimizer state, validation predictions, metrics, and stopping reason. To resume an interrupted neural run, substitute its actual path:

```bash
.venv/bin/python -m mac_agents.cli train --resume runs/ACTUAL_RUN --max-steps 5000 --wall-minutes 120
```

Resume requires unchanged data, code, state condition, and backend. A step limit may be increased explicitly; original configuration and continuation caps are recorded. CPU resume is checked for bitwise equivalence; exact MPS training continuation is not claimed solely from CPU tests.

## Freeze and evaluate

The training-only PCA study has a dedicated entry point covering all three baselines and seven neural conditions at three seeds, including matched availability-only controls. Its validation report is [inductive_validation_results.md](reports/inductive_validation_results.md). Run stages sequentially:

```bash
.venv/bin/python -u scripts/inductive_study.py validation
.venv/bin/python scripts/report_inductive_study.py --partition validation
.venv/bin/python -u scripts/inductive_study.py freeze
.venv/bin/python -u scripts/inductive_study.py test
.venv/bin/python scripts/report_inductive_study.py --partition test
```

The plan and immutable protocol are under `runs/inductive_study/`; cumulative compute is capped at 90 minutes for this follow-up. Its source identity includes both core code and the study adapter, so **use this driver for these runs**: ordinary CLI evaluation deliberately rejects them. The adapter applies the initialization-only control consistently during fitting and autonomous rollout. Test execution requires the saved selection and protocol hashes, reuses completed evaluations, and performs no retraining. Report new-question and known-question transfer separately. Once scored, these test outcomes are available for analysis and cannot serve as an unseen selection set for later development.

For standard runs from the original author-context screen, the generic CLI workflow below applies. The dedicated study above is the current controlled test protocol.

Review validation results, repeat selected comparisons and matched controls with seeds 42, 43, and 44 where budget permits, and freeze before looking at controlled test scores. Replace the run placeholders below with actual non-smoke run directories.

```bash
.venv/bin/python -m mac_agents.cli freeze --runs runs/ACTUAL_RUN --rationale 'Selection based on the saved validation comparisons'
.venv/bin/python -m mac_agents.cli evaluate --run runs/ACTUAL_RUN --partition test --mode one_step --selection runs/selection.json
.venv/bin/python -m mac_agents.cli evaluate --run runs/ACTUAL_RUN --partition test --mode closed_loop --selection runs/selection.json
```

Only opinion/history models currently have autonomous state updates. Closed-loop evaluation rejects vote-count or observed-message models. Repeated test commands return the saved result rather than rescoring a changed selection. Full test report synthesis and targeted intervention work remain research-stage tasks after the selected runs.

One-step metrics include question-averaged NLL, two-class Brier score, and flip/class-balanced accuracy. Persistence has no artificial likelihood score. Rollout artifacts contain deterministic and sampled trajectories, horizon errors, switching counts, final outcome distributions, and a declared unanimity criterion. `report` adds paired question bootstraps only for matching validation examples and preprocessing; seed variation should be assessed separately.

## Limits of the initial implementation

- The source subset and binary target are deliberately fixed to GPT-4o-mini/objective.
- The initial message representation pools frozen embeddings; learned recurrent message-state evolution is future work.
- Generic text embeddings may miss important mathematical distinctions. Four validation questions also limit architecture selection precision.
- Validation screening, locked test evaluation, intervention experiments, the local MLX generator, and five-category data generation are separate milestones. Consult `STATUS.md` for which have actually run.
- The released author code is Apache-2.0; its license and immutable revision are retained with the acquired files. The frozen encoder's model metadata remains in its pinned Hugging Face cache.
