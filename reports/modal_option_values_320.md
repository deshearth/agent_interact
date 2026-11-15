# Exact numeric values in arguments: internal screen

**The ordered numeric features pass the prespecified +0.5 percentage-point promotion gate.** This is an internal256/64 comparison on all eight rounds, not outer validation. The inputs match literal integer, decimal and rational values in each actual message to the two supplied answer options. They never use the correct-answer key, worked solution or future state.

| Input | Training accuracy | Internal accuracy | Change vs reference |
| --- | ---: | ---: | ---: |
| Existing semantic8 state | — | 87.952% | — |
| summary numeric fields | 91.394% | 88.428% | +0.476pp [+0.079, +0.848] |
| ordered numeric fields | 91.856% | 88.483% | +0.531pp [+0.226, +0.830] |

Eight fields summarize each message: optionA/B occurrence counts, last matched option, last numeric literal, A/B mentions in the last third, final boxed-value match, and numeric-literal count. The grouped model adds64 features; the ordered model adds144 features in actual friend/enemy inbox slots. All original622 features remain.

The parser is deliberately limited to literal rational values. It does not solve algebra or decide whether an argument is correct; intermediate calculations can also mention an option value. The focused checks caught a sentence-final fraction boundary bug before any data preparation or fitting. The corrected parser passes all three checks.

Both models, their selected iterations, target order and 768 raw message feature vectors replay. Question-bootstrap intervals are descriptive. The next separate experiment freezes a320/512 full-data comparison and a controlled neural augmentation before outer scoring.
