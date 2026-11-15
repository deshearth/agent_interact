# Scaling the numeric-argument transformer to 512 questions

**Numeric transformer validation accuracy reaches 88.960% among the two predeclared 512-question budgets.** Both budgets are reported below; this maximum is not used to select another checkpoint. The 95% target remains unmet.

| Model | 320 questions | 512, same updates | 512, same data passes |
| --- | ---: | ---: | ---: |
| Argument/history + zero-field control | 88.405% | 88.376% | 88.426% |
| Argument/history + numeric features | 88.672% | 88.924% | 88.960% |

Both models have 2,370 learned parameters and use the same frozen MiniLM/PCA argument vectors, observed history, actual inbox order, current probabilities and signed graph. The numeric arm additionally matches literal rational values in messages to the two supplied answer choices. The control receives zeros in those eight fields. No correct-answer key or future observation enters the inputs. The frozen 22,713,216-parameter encoder is additional to the update-rule count.

The 512 cohort adds the same first192 new training questions in the collection’s frozen ordering. All original features and PCA transforms remain unchanged. Each seed starts from exactly its original320 initialization, with the same optimizer and regularization. One checkpoint holds optimizer steps fixed; another multiplies steps by512/320. There is no new tuning. All12 checkpoints froze before outer scoring. Values average three individual seeds rather than an ensemble.

| Accuracy contrast | Change | Descriptive 95% question interval |
| --- | ---: | ---: |
| numeric / 512 / same_updates vs numeric / 320 / original | +0.252 pp | [+0.122, +0.394] |
| numeric / 512 / same_passes vs numeric / 320 / original | +0.288 pp | [+0.161, +0.422] |
| blank / 512 / same_updates vs blank / 320 / original | -0.029 pp | [-0.148, +0.096] |
| blank / 512 / same_passes vs blank / 320 / original | +0.021 pp | [-0.101, +0.146] |
| numeric / 320 / original vs blank / 320 / original | +0.267 pp | [+0.057, +0.479] |
| numeric / 512 / same_updates vs blank / 512 / same_updates | +0.549 pp | [+0.321, +0.770] |
| numeric / 512 / same_passes vs blank / 512 / same_passes | +0.534 pp | [+0.317, +0.744] |

| Model / cohort / budget | Training accuracy | Validation accuracy |
| --- | ---: | ---: |
| blank_320_original | 88.632% | 88.405% |
| blank_512_same_passes | 88.913% | 88.426% |
| blank_512_same_updates | 88.843% | 88.376% |
| numeric_320_original | 89.244% | 88.672% |
| numeric_512_same_passes | 89.490% | 88.960% |
| numeric_512_same_updates | 89.403% | 88.924% |

This follow-up was motivated after seeing the audited320 result and is exploratory. The same80 validation questions have been repeatedly reused; new validation and the40 reserved assessment questions remain unused. Revised topic sampling in the added questions limits a pure sample-count interpretation. These are all-case next-source-mode predictions on existing stochastic histories, with known graph families and personas.

All36 prediction files and checkpoint outputs replay exactly; all12 original320 prediction arrays remain unchanged. Target/question ordering, seed initialization, feature identity and all21 paired contrasts pass verification. Two focused checks pass. Maximum metric discrepancy is 2.8e-17. Six new fits yielded12 fixed endpoints. No source calls were made.
