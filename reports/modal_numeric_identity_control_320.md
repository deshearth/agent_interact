# Do numeric mention counts explain the improvement?

On the internal64 questions, option-linked numeric fields reach88.483%. A same-duration refit retaining numeric counts but removing the option-specific signs reaches87.689%. This supports keeping the connection between values and answer options in the representation. It does not establish a causal mechanism in the source LLM.

| Internal condition | Accuracy |
| --- | ---: |
| original | 88.483% |
| counts_only_refit | 87.689% |
| identity_removed_input | 87.830% |
| identity_swapped_input | 86.316% |

The counts-only model uses the same144 extra slots and155 boosting iterations. A/B occurrence and tail counts are symmetrized; last matched-option, last numeric-literal and boxed-value signs are zeroed. Thus this control also removes the presence flags implicit in the latter two signs; it is a count-only comparison, not a perfectly isolated intervention on label identity alone.

The last two rows change inputs of the already trained identity-aware tree without retraining. Their drops can reflect both removal of useful information and distribution shift. Swapping A/B associations lowers internal accuracy to86.316%; this is diagnostic only and is not scored on outer validation.

All four prediction arrays, metrics and paired question intervals were independently recomputed. The serialized count-only model and original155-step model reproduce their predictions exactly. No outer validation, new validation, source calls or assessment data are used.
