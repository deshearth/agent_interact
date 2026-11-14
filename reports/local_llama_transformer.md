# New transformer training on the local Llama data

Completed 24 newly initialized graph-transformer fits on the completed local Llama-3.1-8B-Instruct 4-bit conversations. Both conditions use exactly the same folds, initial weights, architecture and training settings. The only input difference is the incoming message vectors.

The target is the Llama agent's next **sampled majority-of-five A/B opinion**, not the correct mathematical answer and not the teacher's first-token probability. The table averages three training seeds; predictions are not ensembled.

| Predictor | Selected training error | Heldout question error | Heldout accuracy | Heldout NLL | Two-class Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opinion/history + context | 20.18% | 29.36% | 70.64% | 0.6290 | 0.4134 |
| Same inputs + arguments | 20.12% | 29.10% | 70.90% | 0.6179 | 0.4074 |

Arguments change heldout accuracy by **+0.26 percentage points** (descriptive question-bootstrap 95% interval -0.07 to +0.72) and NLL by -0.0112 (-0.0277 to +0.0007; lower is better). This small pilot does not establish a reliable argument benefit. These intervals summarize eight reused questions and are not confirmatory significance tests; the cross-validation training sets overlap.

The training-fold global-majority baseline has 59.96% heldout accuracy and NLL 0.6889. Simply retaining the receiver's current opinion has 55.27% accuracy.

## Data and split

Eight objective mathematics questions, one 32-agent conversation per question, two synchronous updates and the prescribed signed J0 graph give 512 natural transitions. In each of four fixed folds, four whole questions (256 transitions) train the predictor, two (128) select its checkpoint, and two (128) are scored only after selection. Each question is assessed exactly once per seed. Both rounds, every agent and all paired interventions remain in the same question partition.

All eight questions were previously inspected in exploratory source-model experiments. They are held out from these particular fits, not a fresh untouched research test. No previous GPT test set was scored. The much smaller number of training questions, different teacher and full-context input representation prevent direct attribution of differences from earlier GPT error rates to source-model identity.

| Fold | Train questions | Selection questions | Heldout questions |
| --- | --- | --- | --- |
| 0 | 146, 162, 316, 497 | 359, 309 | 133, 331 |
| 1 | 133, 162, 331, 497 | 146, 316 | 359, 309 |
| 2 | 133, 309, 331, 359 | 162, 497 | 146, 316 |
| 3 | 146, 309, 316, 359 | 133, 331 | 162, 497 |

## Model and actual training

Each predictor has **282,146 trainable parameters**, with four attention heads and a 128-dimensional hidden state. The frozen MiniLM encoder has 22,713,216 parameters and is additional to that predictor count. The no-argument control has 49,152 inactive text-projection parameters. Full persona and question/choice vectors, current opinions, two-snapshot histories, sender stances, node degrees and the original friendly/unfriendly graph cues are available to both models. Communicated stances were checked to equal sender opinions. No question-ID embeddings or target-answer keys are inputs.

The encoder processes each message separately. All input tokens were retained in 530 windows for 515 unique input strings; 9 long strings needed multiple windows (maximum 3276 tokens). Window pooling and frozen sentence vectors do not ensure preservation of mathematical reasoning or long-range token interactions. Incoming-message order is not represented by this graph attention rule.

Training used Apple MPS, AdamW at 0.0003, weight decay 0.001, batches of 32 receiver transitions, at most 50 epochs, selection-NLL early stopping with patience 6, gradient clipping 1 and seeds 42/43/44. The 24-fit training invocation took 19.4 seconds, excluding feature preparation and audit. Every fit changed its parameters; all stopped early. Selected epochs range from 1 to 7, and terminal epochs from 7 to 13. This is actual training, not inference with an old model.

| Condition | Selected training error | Terminal training error |
| --- | ---: | ---: |
| Opinion/history + context | 20.18% | 12.37% |
| Same inputs + arguments | 20.12% | 11.91% |

Terminal errors use the final optimization checkpoint, after the checkpoint selected on separate questions. Lower terminal training error does not justify choosing it on the assessed questions.

## Argument-removal check

The selected models were also evaluated on heldout paired inboxes in which each rationale was replaced by the sender's recorded conclusion. No model trained on these conclusion-only targets. This therefore tests transfer to a changed input distribution, not a model trained to fit the intervention distribution.

| Condition | Conclusion-only error | Error across both variants | Mean absolute predicted majority-probability change | Predicted label changes |
| --- | ---: | ---: | ---: | ---: |
| Opinion/history + context | 31.64% | 30.50% | 0.00 pp | 0.00% |
| Same inputs + arguments | 31.05% | 30.08% | 0.41 pp | 0.72% |

The observed sampled majority changes in 35.35% of the 512 pairs, including sampling variability. The argument model's probability-shift direction agrees with the observed majority-shift direction in 43.83% of changed pairs. This is a descriptive check: independently sampled five-vote groups can disagree even when their underlying distributions are identical. All empty-inbox controls and all no-argument predictions have exactly zero model response to argument removal.

The previous pilot's 22.40 pp teacher effect concerned **first-token probability**, whereas these predictors output **majority-of-five probability**. Those magnitudes are not interchangeable, and this report does not subtract them or call the teacher effect a predictor accuracy improvement.

## Verification and interpretation

Six focused tests passed: question separation; target-independent inputs and pre-update histories; equivalence to the earlier signed attention implementation; padding/empty-inbox isolation; individual-message gradient flow; and sensitivity to graph signs. Every saved selected checkpoint was reloaded on CPU and reproduced its MPS original and intervention probabilities (maximum absolute difference 1.07e-06). Selection epochs were independently reconstructed from selection NLL. All paired initial weights match, all trainable models updated, and all 54 earlier fit manifests and 342 declared artifacts remain unchanged. Source-data hashes match. No new LLM generation or paid API calls were made.

These completed fits show a small average improvement from adding argument vectors, with substantial uncertainty across questions. They do not establish that arguments are irrelevant or that opinions alone form a sufficient state. There are only four training questions per fit; generic frozen vectors, limited coverage and stochastic targets remain plausible constraints. The next useful data expansion would add independent questions and repeated conversations under a split fixed before generation. The current model predicts one update given observed messages/history; it does not generate messages or autonomously simulate a full conversation.

![Heldout performance by question](/Users/melancholymirth/deshearthGit/github/agent_interact/reports/local_llama_transformer.png)

## Reproduce

```sh
.venv/bin/python -m pytest tests/test_local_llama_transformer.py --junitxml=reports/local_llama_transformer_checks.xml
.venv/bin/python scripts/local_llama_transformer.py prepare --device mps
.venv/bin/python scripts/local_llama_transformer.py train --device mps
.venv/bin/python scripts/report_local_llama_transformer.py
```

Completed fits are reused only after hash checks. Protocol, data, checkpoints, optimization histories, seed/question scores, paired outputs and audit records are stored under `runs/local_llama_transformer_v1/`. This directory is separate from the frozen local source pilot and previous predictor studies.
