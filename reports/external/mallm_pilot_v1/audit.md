# MALLM pilot: data audit

Configuration: `simple_nopersona_memory_simple_voting`.

100 distinct questions, split into 60 fitting, 20 validation and 20 reserved assessment questions.
Assessment outcomes were not summarized or scored. Every visible-message ID is checked to precede its target.

| Partition | Updates | Own-answer changes | Changes from supplied draft | Questions with draft change |
| --- | ---: | ---: | ---: | ---: |
| train | 1080 | 1 | 0 | 0 |
| validation | 360 | 0 | 0 | 0 |

Argument-comparison feasibility gate: **not passed**.
The threshold was recorded before computing these outcome counts. It is a practical screen, not proof of statistical power.

The next-answer target is the author-extracted solution field. Agreement with an explanation is not used as a yes/no answer.
All agents see a shared current draft. Copying that draft is therefore a necessary baseline alongside keeping the agent’s own last answer.

Source: https://huggingface.co/datasets/Multi-Agent-LLMs/DEBATE
