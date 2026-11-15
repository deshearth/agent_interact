# Numeric arguments: full-data transfer comparison

**Best validation accuracy in this comparison: 89.097%.** The95% target remains unmet.

The experiment followed a positive internal-only screen: exact rational matching between argument numbers and supplied answer options improved internal accuracy by0.531percentage points. All four tree refits and six neural refits froze before these80-question validation scores were calculated. All eight rounds and32agents are retained.

| Model / training questions | Training accuracy | Validation accuracy |
| --- | ---: | ---: |
| Original tree /320 | 90.267% | 88.579% |
| Original tree /512 | 90.055% | 88.677% |
| Duration-matched tree /320 | 91.024% | 88.618% |
| Duration-matched tree /512 | 90.565% | 88.691% |
| Numeric argument tree /320 | 91.357% | 88.853% |
| Numeric argument tree /512 | 90.884% | 89.097% |
| History transformer, blank extra fields /320 | 88.632% | 88.405% |
| History transformer, numeric arguments /320 | 89.244% | 88.672% |
| Previous history/argument transformer /320 | 88.757% | 88.371% |
| Previous history/choice transformer /320 | 88.793% | 88.441% |

| Contrast | Accuracy change | Descriptive95% interval |
| --- | ---: | ---: |
| Numeric argument tree /320 vs Duration-matched tree /320 | +0.234pp | [-0.020, +0.483] |
| Numeric argument tree /512 vs Duration-matched tree /512 | +0.405pp | [+0.151, +0.664] |
| Numeric argument tree /320 vs Original tree /320 | +0.273pp | [+0.034, +0.513] |
| Numeric argument tree /512 vs Original tree /512 | +0.420pp | [+0.156, +0.679] |
| Numeric argument tree /512 vs Numeric argument tree /320 | +0.244pp | [+0.029, +0.454] |
| History transformer, numeric arguments /320 vs History transformer, blank extra fields /320 | +0.267pp | [+0.057, +0.467] |
| History transformer, numeric arguments /320 vs Previous history/argument transformer /320 | +0.301pp | [+0.089, +0.513] |
| History transformer, numeric arguments /320 vs Previous history/choice transformer /320 | +0.231pp | [+0.007, +0.456] |

The numeric fields use only the two supplied option values and actual sender message text: counts of exact rational matches, their order/tail locations and a final boxed value. Equivalent fractions/decimals are recognized; algebra is not solved. The correct-answer key and future probabilities never enter predictor inputs.

Both neural arms have2370 learned parameters, identical initial weights at each seed and the same history architecture. Both retain PCA8 argument vectors; only the eight numeric fields differ from zeros. Each arm selects its own learning rate and duration on the internal256/64 split, with the same search budget, then refits three fresh seeds. Reported neural scores average those individual models, not an ensemble. The frozen MiniLM encoder remains additional22,713,216parameters.

Tree conditions append144 ordered numeric features to the original622 state/argument features. Both numeric and duration-matched controls use155 iterations, selected solely in the internal numeric pilot; original trees use their independently selected105 iterations. The512 corpus adds the same frozen first192new training questions used in the earlier scaling study. Its revised topic proportions limit a pure sample-count interpretation.

| Validation subset, diagnostic only | Updates | Numeric512 accuracy | Duration control accuracy | Errors corrected | Errors introduced |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 20,480 | 89.097% | 88.691% | 1.514% | 1.108% |
| an option value appears | 19,291 | 88.938% | 88.492% | 1.560% | 1.115% |
| no option value appears | 1,189 | 91.674% | 91.926% | 0.757% | 1.009% |

Verification checks all36 prediction files, independently replays20 new training/validation outputs from10 models and all8 tuning outputs, confirms0 class disagreements, and reproduces all24 paired contrasts. Matched seed initializations and internal checkpoint selection also reproduce. Maximum log-probability discrepancy is0. Two model-input checks and the earlier three numeric-parser checks pass. Validation remains exploratory after repeated reuse; new validation and reserved assessment conversations remain unused.
