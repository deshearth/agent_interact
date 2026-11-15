# Full-context ModernBERT: training-only pilot

**No candidate passed the internal promotion gate.** These scores use two fixed rounds of the internal 256/64 question split. They do not measure all-round accuracy on the 80-question validation anchor and cannot establish 95% performance.

| Model | Training accuracy | Internal accuracy | Learned head size |
| --- | ---: | ---: | ---: |
| Frozen state-tree reference | — | 84.302% | 622 input features |
| tree_prompt32 | 89.496% | 83.569% | 3,172 tree nodes |
| tree_prompt128 | 91.949% | 84.058% | 5,124 tree nodes |
| linear_prompt0_l20.001 | 92.700% | 84.277% | 2 coefficients |
| linear_prompt32_l20.001 | 92.932% | 84.302% | 34 coefficients |
| linear_prompt32_l20.01 | 92.883% | 84.351% | 34 coefficients |
| linear_prompt32_l20.1 | 92.761% | 84.277% | 34 coefficients |
| linear_prompt32_l21 | 92.694% | 84.277% | 34 coefficients |
| linear_prompt128_l20.001 | 93.250% | 84.009% | 130 coefficients |
| linear_prompt128_l20.01 | 93.213% | 84.009% | 130 coefficients |
| linear_prompt128_l20.1 | 92.920% | 84.180% | 130 coefficients |
| linear_prompt128_l21 | 92.731% | 84.229% | 130 coefficients |
| linear_prompt1536_l20.001 | 94.574% | 83.179% | 1,538 coefficients |
| linear_prompt1536_l20.01 | 93.793% | 83.618% | 1,538 coefficients |
| linear_prompt1536_l20.1 | 93.170% | 83.960% | 1,538 coefficients |
| linear_prompt1536_l21 | 92.859% | 84.302% | 1,538 coefficients |

The [ModernBERT-base checkpoint](https://huggingface.co/answerdotai/ModernBERT-base) has 149,014,272 frozen encoder parameters in addition to the fitted prediction heads. This is a test of frozen representations, not end-to-end fine-tuning. Its native full-context attention sees every prompt token together; 20,440,255 tokens were retained, with maximum length 3,722. No independent windows or truncation were used.

The inputs are exactly the source receiver prompts: persona, question, supplied answer options, and actual ordered friendly/unfriendly messages. No correct-answer key, solution, future probability or future message enters these models. Mean and first-token vectors are concatenated; PCA and feature standardization use only the 256 fitting questions.

Tree heads append PCA32 or PCA128 to the existing state features. Linear heads correct the frozen state-tree logit using PCA32, PCA128 or all1536 encoder features at four fixed regularization strengths. A calibration-only control prevents mistaking improved logit calibration for useful argument information. Promotion requires at least0.5 percentage points above the matched reference; linear text corrections must also beat the calibration control.

The audit reproduces all15 fitted heads, training-only transforms and exact source-target alignment, and re-encodes 6 prompts on CPU including the three longest. Maximum CPU/GPU vector difference is 5.9e-06. Question-bootstrap intervals are descriptive after internal model selection. No new validation or reserved assessment data was used.

A separate all-round [numeric argument follow-up](modal_numeric_followup_320.md) subsequently reached88.672% for its neural model and89.097% for its512-question tree. Those results use different features and are not outcomes of this two-round full-context pilot.
