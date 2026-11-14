# Predictor microbenchmark

Real training batch; A2/opinion; 3 warm-up and 20 timed forward/backward steps.

Timing excludes the optimizer, data loading, and text encoding. This is not an LLM speedup measurement.

- cpu: 0.005849 seconds/step; maximum CPU logit difference 0.
- mps: 0.003962 seconds/step; maximum CPU logit difference 9.54e-07.
