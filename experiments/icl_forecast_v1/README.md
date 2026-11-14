# Blind in-context opinion-forecast pilot

**Prepared; waiting for fresh conversations. No predictions or accuracy results yet.** This separate experiment responds to the user's request for the assistant's own estimates without reusing previously inspected test outcomes. It adds no source-generation jobs, API calls, or model-weight training.

The target is the probability that a Llama agent's **next five-vote majority opinion is A**, not the probability that A solves the mathematics problem correctly. The assistant receives eight labeled updates from eight original training questions as in-context demonstrations. It then forecasts eight agents on each of 20 newly added validation questions: 160 forecasts, one boundary per question, spread across rounds 0–7. IDs and boundaries are selected by a fixed rule before test outcomes are inspected.

The packet contains the question/options, personas, current majority opinions, signed communication graph and requested agent IDs. It excludes correct-answer labels, mathematical solutions, screening statistics, raw vote counts, source probabilities, all message text, and future opinions. In particular, messages produced during the update being predicted are future information. An argument-aware follow-up would require its own declared information boundary.

Only the newly added validation cohort is eligible. The original 20 validation questions, other previously studied datasets, and all 40 reserved assessment questions cannot supply test cases. After scoring, these 20 questions become used development data; they must not subsequently be described as untouched confirmation. Existing public MATH questions may have appeared in model pretraining, but the newly sampled agent outcomes are the held-out targets.

## Procedure

1. `freeze` records checksums of the prediction protocol, exporter, existing frozen predictors, and corpus-expansion protocol before the new validation conversations are ready.
2. `prepare` waits for the fixed fresh cohort, then exports only whitelisted inputs and training demonstrations to `runs/icl_forecast_v1/inputs/`. The label file and panel mapping go to `sealed/`, outside the forecasting packet. Tool output contains counts and readiness only.
3. While forecasting, the assistant reads **only** `instructions.txt`, `demonstrations.json` and `test_inputs.json` in the input directory. It must not inspect the source episodes/chunks, selection, `sealed/` files, source probabilities or competing model outputs. Previous aggregate findings in this chat cannot be erased; this is a prospective test on fresh cases with that prior background.
4. The assistant writes all 160 probabilities to one JSON object and runs `commit`. This checks completeness, records a timestamp and hashes the predictions and input packet. Overwriting committed predictions is forbidden.
5. Only then may `score` open the labels. Report equal-question accuracy, log loss, binary Brier score and descriptive question-bootstrap intervals. Compare the original frozen 80-question transformer and Ising on exactly these same cases, separately from the assistant's forecasts. Their predictions must not enter the assistant's forecasting context.

Blinding is procedural, enforced by the workflow and tests, rather than an operating-system security barrier. The exporter reads target labels internally without displaying them. This is an exploratory manual pilot; a large pretrained assistant and an 818-parameter fitted predictor have different pretraining, representations and costs. Better assistant predictions would motivate further study of usable state information and representations, not establish that adding parameters alone improves the small model.

```bash
.venv/bin/python experiments/icl_forecast_v1/blind_pilot.py freeze
.venv/bin/python experiments/icl_forecast_v1/blind_pilot.py prepare
# Read only the allowed input packet and save forecasts once it is ready.
.venv/bin/python experiments/icl_forecast_v1/blind_pilot.py commit --predictions PATH_TO_FORECAST_JSON
.venv/bin/python experiments/icl_forecast_v1/blind_pilot.py score
```

The comparison with frozen small models requires feature preparation for the fresh questions and is not yet automated by this pilot. Do not report their whole-validation aggregate scores as if they measured the same 160 cases. No chat wakeup, automatic assistant forecasting, or automatic scoring is configured.
