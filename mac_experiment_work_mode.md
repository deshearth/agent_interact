# Mac-first experiment handoff for Work mode

**Current authorized study, running:** The user specified 80 training, 20 validation and 40 untouched assessment questions and selected the 818-parameter transformer versus original Ising (19 coefficients). The user subsequently authorized background execution to completion, replacing the initial two-hour collection-session limit for this study. New source generation uses the installed pinned local 4-bit Llama, with float32 floating parameters/activations after a numerical batching check. The frozen settings are in `configs/question_breadth.yaml` and `runs/question_breadth_v1/source_protocol.json`. All 560 binary-choice candidates are constructed; screening is in progress. No new predictor scores exist yet. The detached `scripts/breadth_background.py` controller waits for the current bounded session, then resumes automatically through train/validation collection, fitting, model freezing, assessment collection and reporting. Do not start competing collectors. Check `runs/question_breadth_v1/background_session.json`, the source progress file and `STATUS.md` before acting.

The selection design uses rational-answer MATH questions, equal seven-topic quotas, balanced correct labels and pooled entropy from 32 personas times eight initial responses. Record within-persona variability separately from between-persona disagreement. All source interactions use prescribed signed graphs. Each question has one graph/conversation and eight updates, so broader coverage does not hold the prior replication/transition budget constant. Failed strict-format distractor selections use a predefined mathematically wrong option; 249/560 candidates used that fallback, whose plausibility is not independently certified. This is a separate matched Llama comparison, not an isolated estimate of the effect of question count on GPT results. Both predictors use current opinions and the same 16 context features; MiniLM is pretrained and frozen, PCA fits training questions, and screening statistics and arguments are excluded from predictor inputs. Assessment interaction outcomes stay ungenerated until all weights freeze. See `STATUS.md` for checks, counts, caveats and recovery commands. Earlier experimental code and artifacts remain frozen.

**Concurrent investigation, complete:** The user asked for other investigation while generation runs. A CPU-only study fitted 48 explicit rules to the twelve frozen 818-parameter GPT-data transformers. Context-dependent and quadratic formulas with up to 42 coefficients leave about 4.9 pp probability MAE even on their fitting questions; older held-out next-opinion error remains about 10.1% versus 9.12% for the teacher. A separately recorded follow-up found some differing teacher outputs under exactly identical pooled inputs, but the resulting finite-sample minimum probability MAE is only 0.141 pp on the older held-out states. This does not establish aggregation as the dominant cause of the formula gap. No universal law or fresh-data conclusion follows. Ten tests, CPU teacher reproduction, and prior/active-protocol integrity checks passed. See `reports/transformer_rule_distillation.md` and `runs/transformer_rule_distillation_v1/`. The active Llama collector and its 80/20/40 protocol were not modified.

**Earlier matched generalization study, complete:** The matched generalization comparison completed 216 inner tuning fits, 36 outer refits and 16 fold-specific PCA/Ising preparations. It uses four outer question folds and three inner folds on the existing twenty GPT development questions. Outer error/NLL are Ising 10.95%/0.22755, full transformer 9.81%/0.23282, small transformer 9.48%/0.22321, and Ising-plus-small-transformer 9.41%/0.22354. The small model uses 34,978 neural parameters versus 135,458. Its modest improvement over the full model remains uncertain, and added benefit of the Ising prior over the small model is unresolved. All models received the same six tuning choices, and PCA/Ising were refitted inside each training fold. Configurations and durations were frozen before refits at seeds 142/143/144 and before any outer assessment. Eight tests, real-data isolation checks and CPU reconstruction of all 36 checkpoints passed; 581 recorded old artifacts are unchanged. Neural fits took 45.1 minutes. No generation, paid calls, consumed official test scores or rollouts were added in that study. This is exploratory evidence on previously studied questions. See `reports/transformer_generalization.md`, `runs/transformer_generalization_v1/` and `STATUS.md`.

**Previous capacity investigation, complete:** The original GPT-data Ising/transformer capacity diagnosis completed 18 fits: imitate the frozen Ising probabilities, then fit observed opinions with no dropout/weight decay or validation stopping. The unchanged 135,458-parameter A2 model uses the original 16 questions and 131,072 transitions, three seeds, three learning rates and 128 epochs per fit. All imitation fits pass; training-selected probability MAE is 0.022 pp on train and 0.021 pp on development validation. Observed training error falls to 4.41%, versus 9.15% Ising and 8.14% for the earlier selected transformer. Validation error worsens from 10.59% to 12.22%, and NLL from 0.27943 to 0.48890: stronger fitting is possible, but it does not transfer. Exact-input conflicts force at least 3.41% empirical training error; this is not an architecture optimum or a Bayes-error estimate. The conditional residual stage was skipped under the training-only trigger recorded before observed-outcome fitting. Seven focused tests and independent CPU reconstruction of all selected checkpoints pass; 378 recorded old-study files remain unchanged. Fits took 33.2 minutes. No new generation, API calls, consumed-test scoring or rollout occurred. See `reports/ising_transformer_capacity.md`, `runs/ising_transformer_capacity_v1/` and `STATUS.md`. This does not resolve the separate Llama argument comparison.

**Previous authorized follow-up, complete:** The user requested actual training on the newly generated Llama data. All 24 matched graph-transformer fits are complete in `runs/local_llama_transformer_v1/`: four whole-question folds, three seeds and two input conditions, using 512 natural transitions across eight questions. Selected training errors are 20.18% (opinions/history) and 20.12% (arguments); heldout question accuracies are 70.64% and 70.90%. The +0.26 pp accuracy change has descriptive question-bootstrap interval [-0.07, +0.72]. Every model was trained from scratch with the same matched initialization and frozen text encoder, and every saved checkpoint was independently reproduced. These are exploratory predictions of majority-of-five agent opinions, distinct from teacher first-token sensitivity. Earlier studies remain frozen. See `reports/local_llama_transformer.md` and `STATUS.md` for splits, paired ablation results and exact commands.

**Project:** Multi-agent system  
**Target machine:** Mac Studio, Apple M2 Max, 32 GB unified memory  
**Prepared:** 2026-11-08  
**Reference paper:** *Physics of Agents: Statistical Mechanics Predicts Collective Behavior of AI Agents*, arXiv:2608.16578v2  
**Status:** Preparation, the author-context validation screen, and the training-only PCA study through frozen test evaluation are complete, including matched history-availability controls and autonomous rollouts. The 24-run cohort was frozen before scoring 1,048 test episodes. See `reports/inductive_test_results.md`, `STATUS.md`, and `README.md`. The test outcomes are now observed; future model selection needs a fresh held-out assessment. The user subsequently authorized local open-model generation; its separate bounded pilot is described below. Emergent credibility is explicitly deferred.

**Argument follow-up:** The separate development-only delivered-message diagnostic is also complete: 18 A0/A2 fits across stance-only, full-text and matched-substitution conditions at three seeds. Full text did not improve mean accuracy or NLL over stance alone in either architecture. A2 accuracy was 89.33% with text versus 89.53% without it. This result concerns the tested frozen-embedding representation on four validation questions and does not establish argument irrelevance. See `reports/argument_response_results.md`. No fresh test outcomes, source-LLM calls or autonomous message-state rollouts were added.

**Training investigation:** The graph transformer was trained from scratch; MiniLM alone remained frozen. A completed checkpoint/gradient audit shows active text/attention learning and overfitting during continued training. Six additional development fits with text inputs scaled by 32 did not rescue performance (88.32% accuracy). Original artifacts remain unchanged. See `reports/argument_training_diagnosis.md` for the scale control, representation limitations and vote-disagreement audit. Token-level fine-tuning and coverage beyond the 16 fitting questions remain untested; frozen question-conditioned encoding is covered by the later investigation below.

**Approved question-count follow-up:** Fit A2 stance/history and full-argument models with 4, 8, 12 and 16 training questions. Keep the same four validation questions. Four nested subset paths balance question composition and three model seeds separate initialization variation. Fit question PCA inside each selected training subset, retain all within-question trajectories, and validate every 128 optimizer updates with six-check patience. Reuse the six original 16-question endpoints after exact preprocessing and optimizer-replay checks. The independent `question_count_curve_v1` study contains 72 new fits and reports training/validation error and NLL, with descriptive uncertainty. No consumed test data or new source-LLM calls are included. See `STATUS.md` for execution progress.

**Learning-curve result:** This follow-up is complete. Argument-model validation error is 12.40%, 11.02%, 10.85% and 10.67% at 4, 8, 12 and 16 training questions; stance/history gives 12.17%, 10.72%, 10.31% and 10.47%. Most classification improvement occurs from four to eight, while mean NLL improves through sixteen. Added argument text remains worse on average at every size. The curve supports a benefit from broader training coverage within this pipeline, not a conclusion that more questions alone will fix argument modeling or a prediction beyond sixteen. All 72 new fits and audits completed; see `reports/question_count_learning_curve.md`.

**Bounded argument-state investigation:** The subsequent two-hour local investigation completed 54 controlled fits, six small capacity fits and read-only terminal/subgroup/input audits. In the original PCA3 architecture, familiar-question error is 6.47% with no argument vector, 5.93% with individual arguments, 6.01% with one question-average vector, and 5.92% with question IDs alone. The familiar gain can therefore arise without individual message language. This is a within-question mechanism result, not a model for unseen questions. Joint question-conditioned frozen encoding and reconstructed inbox order did not provide a transferable remedy. Isolating individual residuals improved fitting with full question context but not held-out probability quality. See `reports/argument_investigation.md` and the current `STATUS.md`.

**Completed local open-model follow-up:** The user explicitly requested generation with an open model. The paper uses Llama-3.1-8B-Instruct, Qwen3.5-9B and Gemma-3n-E4B-it as well as GPT-4o-mini. A pinned local MLX 4-bit Llama conversion completed 1,920 single-message intervention responses, eight fully new conversations (512 updates, 3,840 opinion responses and 496 messages), and 2,560 matched rationale-deletion responses, with zero paid API calls. Deletion changes first-token probabilities by 22.40 pp on average in the 489 nonempty contexts with explicit matching stances; 130 dominant A/B choices change. All 16 empty-inbox probability controls pass. A stricter post-hoc subset retains a 21.35 pp effect. Two direct validity-grading tasks, each with 1,440 responses, give balanced accuracies of 51.56% and 49.90%; the short-response self-grader is unreliable on these examples. These results distinguish text dependence from reliable validity recognition and do not demonstrate improved predictor accuracy or explain the original GPT result conclusively. All five collections are complete under separate frozen protocols in `runs/local_llama_intervention_v1/`; the final audit records completion within the session's 21:17:35 UTC bound. Raw probabilities, first-token diagnostics and five-vote outcomes remain distinct. The 512 teacher pairs are exported for follow-up, with no new predictor training or consumed-test scoring. Questions are reused development questions, graph cues remain prescribed, and previous studies stay frozen. See `reports/local_open_model_pilot.md` and `STATUS.md`.

## 0. Start here

### Instruction to the Work-mode assistant

Use this document as the working research specification. The objective is to find a compact state representation and a shared neural update rule that reproduce interacting language-model agents' dynamics. The paper's Ising-style predictor is a reference, and a small contextual graph transformer is an expressive candidate. Preparation has been authorized and executed; consult `STATUS.md` to distinguish completed checks from pending research runs.

Begin by inspecting the actual execution environment and project directory. The target is the user's Mac; a cloud Linux workspace is not that Mac. If direct Mac execution is unavailable, implement the portable project, run small CPU correctness checks where possible, and provide exact Mac commands. Clearly distinguish checks actually executed from commands awaiting Mac execution.

**The first milestone is the binary experiment on the released data, including an early comparison of state representations.** Audit raw votes, histories, and message availability alongside the original baseline. Implement the comparison models, evaluation, and reports before generating a new dataset. Use bounded runs and save resumable progress. A scientifically sound negative result is a successful experiment.

For the initial execution, work through Phases 0–3 in Section 12 within the configured compute budget. Prioritize state-representation comparisons over attention-width tuning; report pending work if the budget expires. Then implement and, if running on the Mac, benchmark the small local generator. Targeted influence experiments, the 96-community ordinal pilot, and the larger study require separately specified follow-up runs; do not start them automatically.

Make routine reversible implementation choices autonomously and record them. Respect the actual session's permissions and existing repository instructions. This file is a project specification, not a substitute for access permissions. Use the current project if the user selects one; otherwise create a separate `mac-agent-dynamics` project directory without overwriting unrelated work.

### Scope and working assumptions

- Work locally first. UCI HPC3, commercial GPUs, and paid model APIs are outside the initial execution.
- The Mac has 32 GB shared CPU/GPU memory.
- Train only the small dynamics predictor. Keep any text encoder and language-model generator frozen.
- Start with one source model and one task regime from the released data.
- Compare one-step prediction across declared information conditions, and autonomous multi-step rollout for models that can update every part of their own state.
- Examine vote counts, recent history, and existing message content in the initial state study. Keep non-binary outcomes as a later extension.
- Treat compact, transferable neural dynamics as a meaningful result; seek simpler mathematical expressions as a future interpretation objective.
- Keep a `STATUS.md` file with completed work, commands run, results, remaining work, and the next resumable command.

## 1. Research objective and questions

The central question is:

> Can a compact state representation and a shared neural update rule reproduce and generalize the interaction dynamics of much larger language-model agents, including how those dynamics respond to controlled changes?

There are two independently valuable outcomes: empirical evidence for a compact rule of interaction, and an efficient simulator of the measured behavioral dynamics. A neural rule with strong generalization and rollout fidelity is already interesting before it can be expressed as a simple equation. Its evidence applies to the tested questions, networks, source models, and generation protocols; predictive success does not establish a universal law or reproduce the LLM's full language capabilities.

**The sufficiency of binary opinions as state is a hypothesis to test.** Enriching the state may be essential even with an expressive update model. A failure can reflect omitted information, model limitations, optimization, or insufficient data. Interpret the state representation and update rule jointly.

Use attention as one candidate mechanism for context-dependent interaction. Small query/key widths are a secondary efficiency question. The paper already includes persona/question information in its intrinsic field and evaluates rollouts; adding context or rollouts alone is not the contribution. Separate gains from state information, routing, memory, and capacity through controlled comparisons. See [S1] and [S4].

The initial questions are:

| Question | Comparison that addresses it |
| --- | --- |
| What information must the state retain? | Current opinions versus additions of current vote counts, recent opinion history, or existing messages, with a common forecast boundary |
| Does memory help a small simulator? | Current-snapshot and history-based models, with generated history during autonomous rollout |
| Does learned routing improve prediction? | Contextual attention versus a contextual model with fixed signed aggregation |
| Does context need to affect routing? | Context in the local/value branch only versus context in queries and keys as well |
| Does a better one-step predictor also simulate useful dynamics? | One-step probability scores plus closed-loop trajectory and distribution checks |
| How much computation can be saved? | Measured predictor size, memory, and complete-rollout cost, including encoding and any remaining LLM calls |
| Does the rule predict responses to controlled changes? | A follow-up experiment varying arguments or source cues while holding other receiver inputs fixed |

Attention-width tuning and five-category outcomes remain secondary or later studies. The initial released-data comparison can establish predictive evidence; the intervention question needs a separately executed experiment.

Do not interpret an attention heatmap as proof of causal influence. If prediction improves, report the improvement and the tested information conditions. Causal claims require additional interventions and assumptions.

## 2. Mac execution and resource strategy

### Software choices

Use **PyTorch** for the small predictor, with MPS when available and explicit CPU execution when needed. MPS availability depends on the installed PyTorch/macOS combination; inspect it instead of assuming support. Start in float32. NumPy float64 arrays must be converted before moving tensors to MPS. See [S8].

Use **MLX-LM** for later local language-model generation. It supports Apple Silicon inference and quantized models. Pin the installed version after a successful smoke test. See [S10].

Recommended initial packages:

- Core: `torch`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `pyarrow`, `datasets`, `huggingface_hub`, `pyyaml`, `tqdm`, `matplotlib`.
- Checks: `pytest` for the scientific invariants in Section 11.
- Optional semantic features: `sentence-transformers`.
- Generation: `mlx-lm`, preferably in a separate virtual environment.

Use a compatible native ARM64 Python installation, preferably an existing supported Python 3.11 or 3.12. Resolve compatible versions on the machine, then write a lock file or a frozen package list. Do not invent package versions.

### Environment audit

Record:

- Operating system, macOS version if applicable, architecture, Python executable/version.
- Reported memory and available disk space.
- PyTorch version, `torch.backends.mps.is_built()`, and `torch.backends.mps.is_available()`.
- MLX/MLX-LM versions when generation is introduced.
- Actual device and dtype for every measured run.

A minimal diagnostic is:

```python
import platform
import sys
import torch

print({
    "system": platform.system(),
    "architecture": platform.machine(),
    "python": sys.version,
    "torch": torch.__version__,
    "mps_built": torch.backends.mps.is_built(),
    "mps_available": torch.backends.mps.is_available(),
})
```

Choose MPS only if it is available. Compare a small CPU and MPS batch before scaling. If an operation requires CPU execution, log that fact; do not silently characterize mixed execution as an all-GPU benchmark.

### Memory and compute defaults

These are conservative project defaults, not measured hardware limits:

| Item | Initial policy |
| --- | --- |
| Trainable dynamics model | Below 1 million parameters; aim for about 0.1–0.5 million |
| Precision | float32 for predictor training |
| Training batch | 32 complete community snapshots |
| Data loading | CPU-resident dataset; move only the current batch to MPS |
| DataLoader workers | 0 initially |
| Generator processes | 1 |
| Resident generator copies | 1 |
| Generator batching | Sequential first; test 2–4 only after profiling |
| Predictor/generator scheduling | Separate processes and stages, not concurrent GPU workloads |
| First predictor session | 2 hours aggregate wall-time budget, checkpointing if exhausted |
| Generator benchmark | 30-minute wall-time budget, checkpointing if exhausted |

Leave headroom for macOS and other applications; monitor memory pressure and swap. If pressure grows, reduce batch/context usage or end the current bounded run. Do not disable memory protections to force an experiment to fit.

Download the compact data and metadata first. Download language-model weights only when reaching the generation stage and after checking free disk space.

## 3. Sources, data acquisition, and provenance

### Authoritative starting points

- Paper and pinned version: [S1].
- Official repository: [S2].
- Released dataset: [S3].
- Author implementation guide: [S4].
- Loading and evaluation helpers: [S5].
- Baseline notebook: [S6].
- Reproduction caveats: [S7].

The released Hugging Face collection currently advertises **9,600 episode rows, 60 statements, and about 217 MB of files**. Its sole split is named **`train`**, but that collection includes the paper's experimental test questions. The table contains states and raw votes, not the full messages. [S3]

Fetch one canonical consolidated source, rather than both the repository copy and the Hugging Face copy unnecessarily. Also acquire the selected task's question/persona metadata, relevant author analysis code, and matching raw episode files needed for the early message-state audit. Inspect remote file listings before choosing download paths; do not assume parquet shard names. Record any differences between consolidated and raw records.

The implementation guide identifies question banks under `data/obj/` and `data/subj/`, persona metadata alongside them, consolidated repository records at `data/clean_runs.json`, and separate message records under `data/models/<model>/<mode>_energy/`. [S4]

Record repository commit SHA, dataset revision, downloaded-file checksums, model revisions, acquisition commands, and licenses in `data_manifest.json`. Repository paths in the source list point to moving branches; resolve immutable revisions during implementation.

### First subset

Use:

- `model = gpt-4o-mini`.
- `mode = objective`.
- All 32 agents and all eight updates present in those episodes.
- The original binary majority-state target.

This is a proposed manageable starting subset. It is distinct from the smaller eight-agent communities proposed for newly generated data.

### Dataset contract

Validate the observed schema before implementing assumptions:

| Field | Expected interpretation |
| --- | --- |
| `model` | Source language model |
| `mode` | Objective or subjective task |
| `statement` | Question text used for the metadata join |
| `J` | Signed adjacency, usually shape `[32,32]` in the first subset |
| `spins_history` | Nine states including initialization, usually `[9,32]` |
| `spins_raw_history` | Five raw votes per state/agent, usually `[9,32,5]` |
| `replica` | Episode repetition identifier |
| `ground_truth` | Evaluation-only objective answer |
| `political_lean` | Evaluation-only subjective annotation |

These fields and collection sizes are source observations, not independent sample counts. [S3]

Build an episode identifier from source model, task, question, graph identity, and replica. Hash the graph values and preserve its orientation. Never identify an episode by a dataframe row number alone.

The author loader joins `statement` to question-bank `question`, recovers `qid`/split, maps objective A to +1 and B to −1, and orders persona embeddings by `idx`. Preserve those alignments. [S5]

Raw episodes additionally document `personas`, `choices`, and `messages_history`. Message keys identify sender and recipient. The documented entry `messages_history[t]` contains messages generated after state `s^t` and used to produce `s^{t+1}`. Audit this alignment against actual records and the pinned generation code; these messages are not available at a forecast boundary immediately after observing `s^t`. At that boundary, the existing inbox comes from `messages_history[t-1]`, or is empty at initialization. [S4, S14]

### Required audit output

Produce `reports/data_audit.md` and machine-readable counts covering:

- Unique questions, graphs/families, episodes, agents, rounds, and valid transitions.
- Metadata join coverage, duplicates, shape mismatches, and label values.
- Exact mapping from the stored graph to sender/recipient orientation.
- Invalid or missing states/votes and the exclusion policy.
- Whether graph-family membership can be reconstructed from author metadata/code.
- Presence or absence of source embeddings and raw message text, with episode/edge/round coverage and raw-to-consolidated join checks.
- Current vote-count availability, valid sample counts, history lengths, and eligible examples for each state condition.
- Message timing relative to the forecast boundary, recipient-specific versus broadcast records, and common comparison subsets.

**Zero in the released binary records is a parse-failure value, not a neutral opinion.** The official audit also describes historical per-recipient message sampling and discrepancies between reproduced and printed metrics. Preserve and explain these distinctions. [S7]

For author reproduction, use the author's masks. For our controlled comparison, use one declared valid-example policy for every comparator. Do not replace invalid labels with a valid class. Any missing-neighbor treatment must be identical across compared input pipelines and explicitly represented.

If the data or metadata cannot be retrieved, report the exact blocker. Synthetic data may check code mechanics, but it cannot substitute for the paper-data experiment.

## 4. Splits and fair comparison

Maintain two clearly separated result sets.

### A. Author-reproduction check

Run the released analysis for the chosen subset using the author's published split, preprocessing, fitting procedure, and metrics. This checks that we understand the pipeline; it is not the model-selection set for our proposed architecture.

The objective task has 20 training and 20 test questions. Fitting uses eight random graphs with four replicas; other graph families test transfer. The paper's intrinsic field uses three-dimensional persona/question representations and their interactions. [S1]

Use source preprocessing exactly for this check and document it. If a published preprocessing artifact was built using a wider question pool, disclose that property rather than silently describing this check as fully inductive.

### B. Controlled predictor comparison

For the first objective subset:

1. Keep the official 20 test questions locked.
2. Divide the official 20 training questions into **16 training and 4 validation questions**, with fixed split seed 1729.
3. Fit on the training questions' random-graph episodes.
4. Validate on the validation questions' random-graph episodes.
5. Do not use held-out graph families to select architecture or training settings.
6. After selection, score:
   - New questions on the familiar random-graph family.
   - Known training questions on unseen graph families.
   - New questions on unseen graph families.

With complete expected records, the training pool is `16 × 8 × 4 = 512` episodes and `512 × 8 × 32 = 131,072` agent transitions before invalid-example filtering. Validation has 128 episodes and 32,768 transitions. Reconcile these calculations with the data audit rather than forcing counts to match.

Store separate `question_split` and `graph_role` fields. All agents, rounds, and replicas for the same question retain one `question_split`. The known-question/unseen-graph evaluation deliberately uses training question IDs with held-out graphs; it is an evaluation role, not reassignment of those questions to the new-question test split. Enforce episode and graph holdout for that evaluation. Group paraphrases or shared templates together if the bank contains them. Never randomly split individual transitions.

Fit scaling, PCA, and learned feature transformations using training data only. Frozen pretrained text encoders may independently encode held-out inputs; any PCA or normalization subsequently estimated from those embeddings still uses training data only.

For the first comparison, use the author's compact context representation across models. If the available question PCA is only a precomputed artifact with unverifiable fitting provenance, label that comparison accordingly; do not claim inductive preprocessing. Add a strict train-only feature experiment using raw embeddings or a frozen local encoder when those are available.

Refit every comparison model on the same 16-question pool. Keep the author-reproduction result using all 20 official training questions in a separate table.

Ground-truth answer keys, correctness labels, political-lean annotations, future opinions, and future messages must never enter predictor inputs. They may be used for evaluation or clearly documented split stratification.

Compare state variants on the same eligible transitions and forecast boundary. If a history or message condition has less coverage, rerun its opinion-only comparator on that matched subset and report the full-data reference separately. Keep static context and the binary target fixed during the state comparison. Use comparable update architectures and training budgets, record added feature-encoder capacity, and repeat promising state comparisons with both A0 and A2. Results conditioned on newly delivered messages belong in a separate table because the forecast starts later.

## 5. Candidate update model: a small contextual graph transformer

### State and context

Use a first-order opinion model as the minimal reference:

\[
p_\theta(s^{t+1}\mid s^t,J,c).
\]

Here `c` contains static question/options and persona information. This reference recomputes agent representations at every step and has no recurrent hidden memory. Do not assume its observed state is sufficient for the process.

### Initial state comparisons

Define the common forecast boundary immediately after observing `s^t`, before generating the next round's messages. Compare these additions to the same opinion/static-context inputs independently before testing combinations:

| State condition | Added information available at the boundary | Autonomous rollout status |
| --- | --- | --- |
| `opinion` | None; current categorical opinions | Closed state: feed back generated opinions |
| `vote_counts` | Current positive/negative vote counts and valid-sample count | One-step initially; requires a consistent vote-count transition or observation model for rollout |
| `history` | A short window of each agent's opinion history, with availability masks | Closed state: append generated opinions and shift the window |
| `existing_messages` | Encodings of each agent's existing inbox, preserving sender, recipient, relationship, and time | One-step initially; requires a message-state or learned-memory update for rollout |

Vote counts distinguish a 3–2 sample split from a 5–0 split. They are noisy response samples, not verified subjective confidence. They enrich inputs while the primary target remains the next majority-state label. Never use target-round votes as inputs.

Write enriched states as `z^t` and compare `p_theta(s^{t+1} | z^t, J, c)`. The longer-term simulator must also update every evolving component of `z^t` using its own outputs. A compact learned memory may eventually preserve useful message information without generating full text; demonstrating that ability is an additional modeling result, not an assumption.

Evaluate predictive gains on held-out questions with matched examples and model controls. Improvement from history or messages is evidence of useful additional information for the tested predictor. An absent gain does not prove sufficiency: encoder quality, model capacity, optimization, and sample size can limit the comparison.

For agent `i`, a small nonlinear encoder combines:

- Current observed or simulated state, using a categorical representation.
- Persona features.
- Question features, including answer options where available.
- Structural features such as positive/negative degree and total edge magnitude.
- The declared state's vote features, local opinion history, or existing-inbox representation, when that condition is active.

Do not include trainable embeddings for arbitrary agent indices or question IDs. Those IDs identify records, not reusable semantic features.

A question shared by every agent must interact with agent-specific information through a nonlinear encoder or explicit query/key modulation. A constant score offset shared across all senders cancels in a row softmax and does not produce contextual routing.

### Dimension choices

The "tokens" in the first dynamics model are **agents**, not every word in the conversation.

| Quantity | Initial choice |
| --- | --- |
| Agents `N` | 32 for released data; 8 for new local pilot |
| Agent representation `d_model` | 128 |
| Attention heads `H` | 4 |
| Query/key width per head `d_k` | 16 |
| Value width per head `d_v` | 32 |
| Local feed-forward width | 256 |
| Social exchanges per modeled round | 1 |
| Output categories | 2 initially; 5 for ordinal extension |

There is no requirement that `d_k` equal `d_model`, a language model's vocabulary size, or a text encoder's output width.

For batch `B`:

| Tensor | Shape |
| --- | --- |
| Agent representations | `[B,N,128]` |
| Query and key projections | `128 → 64`, reshaped to `[B,4,N,16]` |
| Value projection | `128 → 128`, reshaped to `[B,4,N,32]` |
| Attention score matrix | `[B,4,N,N]` |
| Concatenated head output | `[B,N,128]` |
| Output projection | `128 → 128` |
| Local FFN | `128 → 256 → 128` |
| Class logits | `[B,N,K]` |

Use score scaling `sqrt(d_k)`, which is 4 initially.

Use pre-LayerNorm residual processing and a pointwise GELU feed-forward network. Normalize each agent over its feature dimension; do not introduce cross-agent statistics that bypass the graph mask.

Implement explicit Q/K/V projections. Standard `nn.MultiheadAttention(embed_dim=128, num_heads=4)` gives head width 32; its `kdim`/`vdim` arguments change input widths rather than directly specifying projected head widths. [S9]

Dense masked attention is adequate for these small graphs. Reducing `d_k` reduces scoring/projection work, but it does not change the `N × N` attention matrix size or the value-aggregation dimension.

### Graph direction, signs, and timing

Adopt and document the convention:

\[
J_{ij}\ne 0 \quad\Longrightarrow\quad
\text{sender }j\text{ can communicate to receiver }i.
\]

Translate source data into that convention only after checking the author code.

Compute attention over actual incoming neighbors:

\[
a_{ij}^{h}
=\frac{\langle q_i^h,k_j^h\rangle}{\sqrt{d_k}}+b_h(r_{ij}),
\qquad
\alpha_{ij}^{h}
=\operatorname{softmax}_{j\in\mathcal N(i)}a_{ij}^{h}.
\]

Use relationship-dependent, learned signed value multipliers:

\[
m_i^h=\rho_i^h
\sum_{j\in\mathcal N(i)}
\alpha_{ij}^h\,\gamma_h(r_{ij})\,v_j^h.
\]

Here `r_ij` is the observed edge type. The learned scalar `gamma` may be positive or negative. Do not require every nominally discordant relationship to have a negative final effect. Let a modest degree-dependent, nonnegative strength `rho`, implemented with softplus, plus degree features, distinguish one identical neighbor from several identical neighbors. Keep magnitude in `rho` and signed response in `gamma`.

Keep the agent's own state in a residual/local branch. For an isolated agent, define the social message as zero; do not run softmax on an all-masked row and produce NaNs. If implementation uses a dummy self entry for numerical stability, ensure it contributes zero to the social aggregate.

Use exactly **one new social attention exchange per transition**. Encode each agent's state, including any already-existing local history or inbox, before that exchange. Stored information may reflect earlier rounds; do not feed a neighbor's newly updated memory back into the same round. Stacking new social layers would allow two-hop information within one physical round. Additional pointwise FFN depth is permitted. A global forecaster using broader graph access is a separate information condition.

Agent order is not time. Do not apply a triangular causal mask across agent indices.

### Deferred future direction: agents establish credibility through interaction

The user clarified that the intended future question is how agents establish mutual credibility themselves, without an externally prescribed communication graph or friendly/unfriendly assignments. This would study the evolution of relationships alongside opinions. Its interaction opportunities and credibility measurements would need a new experimental protocol; those choices remain open.

The user explicitly deferred this direction beyond the current experiment. The previously proposed predictor-only label-hiding experiment is withdrawn and is not part of the execution plan. No implementation, generation, or training runs for that proposal were started. The current released-data study conditions on the supplied signed graph because the source agents actually received agreement/disagreement relationship cues in their prompts.

### Binary output and training target

Use two output logits and categorical cross-entropy for the next valid majority-state label. Store the predicted probability of each binary state.

The next majority-sign state, the distribution of an individual raw vote, and the mean of five raw votes are different targets. Do not silently interchange them. The `vote_counts` input comparison keeps the same majority target. A separately fitted vote-count output model is an additional experiment, needed if that state condition is to evolve autonomously; generated counts and majority labels must be consistent.

For closed-loop simulation, feed back sampled or modal categorical states using the same transition function. Update history or learned memory from those generated states and its declared update rule. Feeding back probability vectors or expected states is a different approximation and must have its own label. A model that requires fresh observed votes or messages at every step is not yet an autonomous simulator.

The initial stochastic transition factorizes across agents conditional on the declared current state, graph, and static context: sample each next-agent state independently from its predicted categorical distribution. This is a modeling assumption; it does not capture residual simultaneous dependence from shared unobserved shocks. Check collective outcome distributions as well as marginal predictions.

### History and learned memory

Include the short-history comparison in the initial screen. A proposed starting window is two snapshots including the current one; represent unavailable past entries with an explicit mask rather than invented observations. Select any larger window using validation only and preserve the same temporal cutoff for every comparator.

A small local GRU or another recurrent state is a further candidate within the memory investigation, subject to the same budget. Train recurrent variants on complete trajectories or valid prefixes, maintain hidden-state continuity, and initialize them consistently. Random isolated transitions with a reset hidden state do not train the same recurrent model. Every rollout must use generated history after initialization. If a model needs an observed warm-up prefix, declare its length and compare all relevant models from that same forecast origin; report it separately from rollout starting at `s^0`.

## 6. Model ladder and training budget

### Models to implement

| ID | Model | Purpose |
| --- | --- | --- |
| P0 | Hard persistence | Accuracy/tracking reference |
| P1 | Smoothed own-state Markov transition | Probabilistic no-interaction reference |
| I0 | Author's Ising-style transition | Paper baseline |
| A0 | Contextual fixed-aggregation network | Nonlinear capacity/context control |
| A1 | Graph attention, context excluded from Q/K | Context acts locally and through values |
| A2 | Graph attention, context included in Q/K | Expressive interaction candidate |

Record architecture and state condition separately, for example `A2/history`. P0/P1/I0 remain reference models with their original inputs. Compare A0 and A2 with identical enriched inputs to assess routing; comparing an enriched A2 directly with I0 tests their combined differences in information and architecture.

For P1, estimate transition probabilities from training data with a declared smoothing rule. Avoid assigning arbitrary near-one probabilities to hard persistence just to give it a likelihood score.

For A0, pool the same per-sender value representations used by A2, including each sender's state/persona/question features, in the same relationship channels. Replace learned attention weights with fixed sums or means plus counts, and retain the same receiver-local inputs. Do not pool scalar spins first and discard sender-context associations. Match local encoder depth and approximate parameter scale to A2.

For A1, keep the Q/K branch separate from context-conditioned features. In the primary `opinion` comparison, routing sees current state and structural features; question/persona context can enter values and the local prediction branch. Do not accidentally feed context back into Q/K through shared hidden features. An A1 ablation with enriched state needs an explicit definition of allowed routing inputs; semantic message features or context-conditioned memory must not silently enter a branch labeled context-free.

For I0, adapt the authors' fitting utilities rather than constructing a similarly named substitute. The loader's 19-column tensor and the three-coupling fitting design are not interchangeable. Preserve the signed negative channel and the two-stage fit; the three coupling columns are linearly dependent. [S5]

Report trainable parameter counts and frozen-model parameters separately. An apparent routing gain caused by extra context or a much larger local encoder is not a clean attention result.

### Starting hyperparameters

```yaml
project: mac_agent_dynamics
device: auto_mps_else_cpu
dtype: float32

data:
  source: physics-of-agents/agent-opinions
  model: gpt-4o-mini
  mode: objective
  target: next_majority_state
  split_seed: 1729
  official_train_question_count: 20
  fit_question_count: 16
  validation_question_count: 4
  official_test_question_count: 20
  training_graph_family: random

state:
  condition: opinion
  forecast_boundary: after_opinion_before_messages
  history_window: 2  # Used by history condition; includes the current snapshot.
  message_timing: existing_inbox  # Used by existing_messages condition.

model:
  d_model: 128
  heads: 4
  query_key_dim_per_head: 16
  value_dim_per_head: 32
  ffn_dim: 256
  social_layers_per_step: 1
  recurrent_memory: false
  num_classes: 2
  dropout: 0.1
  max_trainable_parameters: 1000000

training:
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.001
  batch_community_snapshots: 32
  max_epochs: 50
  max_optimizer_steps: 5000
  gradient_clip_norm: 1.0
  early_stopping_patience_epochs: 6
  early_stopping_min_delta: 0.0001
  validation_metric: question_mean_nll
  initial_seed: 42
  final_seeds: [42, 43, 44]
  dataloader_workers: 0
  first_session_wall_minutes: 120

evaluation:
  stochastic_rollouts_per_initial_state: 32
  question_bootstrap_replicates: 1000
  test_locked_during_selection: true
```

These are starting settings for the `opinion` neural reference. Save an explicit state configuration for each variant. The finite-window history model can remain nonrecurrent; any recurrent variant needs its own sequence batching and initialization settings. Preserve the authors' separate optimizer settings for I0 and its reproduction check.

Do not rebalance the training loss by class initially: use likelihood training and report flip-sensitive metrics separately. If weighting is later tested, document its calibration implications.

### Bounded selection procedure

1. Audit state coverage and freeze a prioritized run matrix before fitting. Run model types and state pipelines on a small training-only subset to validate behavior.
2. Fit P0/P1/I0 and train A0/A1/A2 with `opinion` state and `d_k=16` where relevant, one seed.
3. Screen A2 with `vote_counts`, `history`, and `existing_messages` separately, holding the target, static context, forecast boundary, and training budget fixed. Include matched-subset `opinion` runs. Message auditing and a bounded message-state run belong in this phase even if opinion-only performance is good; report source or budget blockers explicitly.
4. Compare promising state additions with matched A0 controls. Use question-averaged validation NLL, uncertainty, parameter counts, and cost to select configurations. Save both the selected one-step predictor and the selected model eligible for autonomous rollout; they may differ. Do not select solely for improvement over I0.
5. Repeat selected state comparisons and their controls with three seeds where budget permits. Keep the primary A1/A2 routing comparison at the same width and state condition. Use validation only for any state combinations or recurrent alternatives.
6. If budget remains, screen A2 widths `d_k ∈ {8,16,32}` with fixed value width and state condition. Treat this as a separate efficiency study; do not attribute a width change solely to contextual routing. Prefer the smaller model when validation differences are unresolved.
7. Freeze state/architecture configurations, preprocessing, forecast origins, and rollout eligibility before evaluating test partitions once for the final comparison.

Keep the original 120-minute aggregate predictor budget; richer state comparisons replace lower-priority tuning rather than silently expanding it. Checkpoints, encoder preprocessing costs, and the pending run list must make partial completion resumable.

Save optimizer state, epoch/step, best validation checkpoint, RNG state, config, and manifest references. If the session budget expires, resume the pending run later; do not drop inconvenient runs or silently change hyperparameters.

## 7. State enrichment and message information

### Compact static context first

Use the same compact persona/question features available to the baseline. This tests whether interaction modeling helps without simultaneously changing the semantic encoder.

### Richer static text features second

A reasonable local encoder candidate is `sentence-transformers/all-MiniLM-L6-v2`, which produces 384-dimensional vectors. Its default text limit is 256 word pieces, so count and report truncation. [S12]

Encode question plus options and persona text separately, once per unique text. Cache embeddings on disk using a key containing the text hash, encoder revision, tokenizer revision, and preprocessing configuration.

Project frozen vectors to small trainable features, for example 32 dimensions for persona and 32 for question. Retrain A0 and A2 with the same cached embeddings. Keep this result separate from the compact-context comparison.

### Existing messages in the initial state study

Audit and screen message information early. For the common forecast boundary after `s^t`, use each agent's existing inbox from `messages_history[t-1]`; at initialization this inbox is empty. Encode that inbox locally before the next modeled social exchange. The purpose is to test whether the information already carried between rounds improves the state description.

Preserve sender identity, recipient identity, edge type, message time, and any recoverable presentation order. Use frozen text encodings with training-only fitted projections, record truncation and cache provenance, and compare attention against aggregation over exactly the same encoded inputs. Generic text embeddings may omit distinctions important to mathematical arguments; a failed embedding variant does not establish that message content is irrelevant.

The historical messages can be recipient-specific. Preserve that distinction when parsing the released raw records; do not collapse them to one message per sender. [S7]

Audit time alignment against the source implementation. For each example, store both the forecast boundary and which information existed there. A message written after the target answer is known is not a valid predictor of that answer.

### Response prediction after new messages arrive

A separate diagnostic may use `messages_history[t]` to predict `s^{t+1}` after those messages have been generated and delivered. This conditions on the next inbox at a later forecast boundary. Label it as response prediction with observed incoming messages and keep it separate from the initial-boundary state comparison. It may help assess message information, but is not an autonomous rollout or a mathematical upper bound on achievable accuracy.

Either message condition is a one-step predictor while it needs real messages supplied at successive evaluation times. Autonomous simulation requires a learned update for message representations or sufficient latent memory, or an explicitly included generator. Record any remaining LLM calls and their cost. Do not replace missing future messages with zeros and call that an equivalent test.

**Approved follow-up, November 2026:** Move the main effort toward argument content while keeping opinion/history predictors as references. The implemented `argument_response_v1` development diagnostic retains each delivered message's sender and recipient, current graph labels, and two-snapshot opinion history. Compare stance only, full individual text embeddings, and matched substitutions with both A0 and A2 at seeds 42–44. Use the same frozen encoder as the earlier study. Substitutions match partition, question, round, sender, receiver, relationship, communicated stance and both agents' current opinions; they do not match complete history or graph identity. They are predictive dependence controls, not source-LLM interventions. No newly generated conversations or emergent-credibility experiment is included.

This follow-up uses the existing 16/4 development split and a separately frozen protocol, implementation and feature namespace. Its four validation questions also select stopping time, so its uncertainty intervals are descriptive. The completed test set has already been inspected and is not reused as fresh confirmation. Record all 18 fits, match/change coverage, per-question and per-round results, and the unchanged earlier history-model reference. Full text includes reasoning and other linguistic cues; a gain does not isolate logical reasoning, and a null result cannot establish argument irrelevance. See `reports/argument_response_results.md` and `STATUS.md` for execution results.

**Further mechanism diagnosis, November 2026:** Following the question-count curve, investigate transfer and text fusion in `argument_mechanisms_v1`. Keep all 16 fitting questions but split entire conversation replicas: 0/1 fit, 2 selects checkpoints, 3 evaluates familiar questions. Evaluate the four existing unseen development questions separately, without using them for stopping in this study. Compare additive fusion with a separate normalized text K/V branch, full and matched-shuffled text, and a shared stance control at three seeds. Rebuild substitutions within the four roles to prevent held-out text entering fitting. Report reduced matching coverage, input-only embedding geometry, all seeds, per-question scores and unchanged source hashes. This fixed split changes fitting quantity and selection relative to previous studies; cross-study score changes do not isolate architecture.

The accompanying `argument_centroid_probe_v1` is read-only: replace every familiar-question incoming vector with a fitting-only question or question/stance average, normalized to preserve unit embedding magnitude. It removes individual-message variation while retaining average content, and never fits unseen-question centroids or changes predictor weights. Treat it as an input-dependence probe with possible distribution shift, not a trained baseline or source-LLM intervention. These analyses may narrow hypotheses but cannot establish argument irrelevance or a causal influence law. See `reports/argument_mechanisms.md`.

**Approved richer-question comparison, November 2026:** The user confirmed that current tasks are objective mathematical A/B questions and approved testing fuller question context in both no-argument and argument predictors. `question_context_v1` adds the complete cached 1,536-dimensional question/choice embedding through a learned projection before the original local encoder nonlinearity. A matched-size control uses only the original three principal components reconstructed in the same coordinate space; both have 382,498 parameters. Zero initialization preserves the original starting function and shared initialization. Keep original context features, message encodings, history, graph signs, conversation splits, donor substitutions and training settings unchanged. Fit all centering/PCA/normalization using only fitting questions.

Run the two new question representations with stance/history, full arguments and matched-shuffled arguments at three seeds (18 new fits), retaining nine frozen original-PCA references. Report full-versus-reduced contrasts and the change in argument contribution relative to each no-text control. Different input ranks are intentional; equal parameter counts do not equalize effective statistical capacity or optimizer geometry. This is a PCA-compression test on cached embeddings, not raw-token question reasoning or a jointly encoded question/argument pair. No new model API calls, test scoring or graph changes are included. See `reports/question_context.md` and the frozen protocol under `runs/question_context_v1/`.

### Targeted influence experiments

Design a follow-up test that holds the receiver's question, persona, other inbox contents, communicated stance labels, and relationship cues fixed while changing one supporting argument. Separate tests can change source cues or exposure counts. Randomize controlled conditions, repeat response sampling, and estimate differences in output probabilities with uncertainty. Freeze model predictions before obtaining held-out intervention outcomes.

A reproducible response difference under the argument intervention shows that stance labels omit information relevant to that intervention. Test whether the candidate rule predicts the difference. Model-only perturbations generate hypotheses; empirical influence claims need responses from the actual generator under the specified intervention.

Specify a bounded intervention protocol after the released-data study and local benchmark. A local-generator result applies to that frozen model/protocol and does not validate causal claims about the original source LLM. This experiment is separate from the later ordinal dataset and is not automatically launched by the initial execution plan.

## 8. Evaluation protocol

### One-step prediction

At round `t`, supply the observed declared state and permitted context, and predict `t+1`. Report the state condition, forecast boundary, eligible-example counts, and matched opinion-only comparator. Keep prediction after newly delivered messages in its own result set.

Primary measures:

- Question-averaged negative log likelihood.
- Brier score.
- The author's flip/class-balanced accuracy, implemented from its actual definition.
- Ordinary accuracy and a stay-versus-switch breakdown.

The author's flip/class-balanced metric groups transitions by observed current/next states. It is not simply sklearn's class-balanced accuracy. Preserve its masks and grouping semantics in the reproduction check. [S7]

Use stable log probabilities. Any numerical clipping used for reporting must be recorded and applied consistently.

### Deterministic rollout tracking

Initialize from the observed `s^0` and static inputs once, with masked unavailable history and explicitly initialized memory. At each step, select the most probable next opinion, feed it back, and update every required state component. Never read real future states, vote counts, or messages. Reject rollout for a state condition without a complete autonomous update. Report observed-prefix forecasts separately with their forecast origin and matched controls.

Report per-horizon state error, group mean-state error, and switching behavior. This is a trajectory-tracking evaluation, not a claim that a stochastic process must reproduce a particular realized path.

### Stochastic rollout distributions

Draw 32 rollouts per held-out initial state initially. Keep random-seed schedules fixed across comparisons.

Compare:

- Final positive-state fraction or category shares.
- Distribution of group mean state.
- Per-agent and group switching counts.
- A declared consensus criterion and time-to-consensus distribution.
- Outcome distributions within question/graph conditions where enough reference episodes exist.

Report reference and simulated sample counts. Additional Monte Carlo samples cannot compensate for having only a few observed episodes.

The binary majority-state model does not directly predict the paper's five-vote conviction statistic. Do not claim to reproduce that quantity without an explicit observation/vote model.

### Size and computational cost

Report trainable predictor parameters, frozen encoder/generator parameters, memory use, encoding and initialization cost, per-step latency, and complete-rollout time under stated batch sizes and devices. Separate one-time preprocessing and training from repeated simulation cost. Include any residual LLM calls rather than presenting the neural forward pass alone as the full simulator cost.

Claim measured speedup only against a measured reference performing the corresponding behavioral simulation under a stated protocol. If source-LLM timing is unavailable, report absolute predictor costs and leave that speedup unmeasured. The later local ordinal benchmark changes both model and protocol, so it is not a matched runtime baseline for the released GPT-4o-mini experiment. The compression claim concerns the specified behavioral outputs.

### Uncertainty and reporting

Compute paired model differences at the **question level**, preserving the associated graphs, episodes, agents, and rounds. Use a paired question bootstrap, initially 1,000 replicates, and show seed variation separately.

Report graph-family results separately. Do not count thousands of correlated agent-round records as thousands of independent experimental units.

A valid outcome may be:

- History or existing messages improve prediction over opinions alone on matched examples.
- Vote counts improve one-step prediction, while an autonomous count-state model remains unresolved.
- An enriched one-step predictor performs well, but a closed state update is not yet available.
- A compact history or memory model reproduces held-out trajectory and outcome distributions at low measured cost.
- A2 improves likelihood and rollout fidelity.
- A2 improves one-step likelihood but not rollout behavior.
- A2 matches I0 with greater cost.
- A2 underperforms I0.
- The available questions are too few to distinguish small differences.

Publish the observed result and unresolved alternatives. Do not keep selecting models on the test set until attention wins, or interpret an unsuccessful state/model pair as evidence that no compact rule exists. Report the tested domain and distinguish predictive evidence from intervention evidence.

## 9. Local generator benchmark and later non-binary extension

The local benchmark follows the released-data comparison. New ordinal data is a separate later experiment; richer predictive state is already studied with binary outcomes. The targeted influence experiment in Section 7 needs its own bounded protocol and may be a more useful next step than expanding the outcome labels.

### Generator choice

Start with the frozen non-thinking model `Qwen/Qwen3-4B-Instruct-2507`. The original model card identifies it as a 4B model. [S11]

A verified MLX conversion is:

`mlx-community/Qwen3-4B-Instruct-2507-4bit`

Record the conversion repository revision and quantization settings as well as the original model identity. A quantized conversion is a specific generator configuration, not an interchangeable copy of every other precision/backend. [S13]

Use one loaded model for all eight agents. Maintain independent prompts and conversation states. Start sequentially, then benchmark small batches if the installed MLX-LM version supports the required workflow.

### Opinion semantics

For the first non-binary pilot, use five ordered stance categories:

| Stored class | Stance |
| --- | --- |
| 0 | Strongly disagree |
| 1 | Disagree |
| 2 | Neutral |
| 3 | Agree |
| 4 | Strongly agree |

An invalid response has a separate error flag and no valid class. Neutrality is a substantive response, not a parser fallback.

Use five-way softmax first. For evaluation, compare with a simple cumulative-link ordinal baseline or an ordinal-output variant.

If extending objective questions to several answer options instead, treat options as **nominal categories** unless a real ordering exists. Do not impose a distance between arbitrary answer labels.

Distinguish:

- Stance extremity.
- Uncertainty about the response.
- Disagreement between agents.

A confidently neutral response differs from an uncertain mixture over opposite extremes. Preserve the full predicted category distribution.

### Fixed generation protocol

Proposed initial settings:

| Item | Setting |
| --- | --- |
| Agents | 8 |
| Personas | Fixed set for the first study; assignments recorded |
| Graphs | Sparse, about 2–3 neighbors per agent |
| State update | Synchronous |
| Prompt budget | At most 2,048 tokens initially, measured with the actual tokenizer |
| Output | One category plus a short supporting message |
| Output token cap | 96 new tokens, including formatting |
| Sampling | Temperature 0.7, top-p 0.9 initially; verify runtime support and record all settings |
| History | Current own state and latest permitted incoming messages |
| Retry policy | At most one format-repair retry; retain original response and mark repaired records |
| Checkpoint | Every completed round and episode |

Treat these as initial settings to freeze after the workflow benchmark. If format validity is poor, fix prompts/parser on benchmark cases and regenerate affected pilot data under one recorded protocol.

If the allowed repair retry also fails, mark the episode incomplete, preserve the previous valid snapshot and all failed outputs, and stop that episode. Do not invent a state to advance a synchronous round. Report incomplete-episode rates and apply the same exclusion policy to every predictor comparison.

A compact response format can be:

```json
{"choice": 3, "message": "A brief explanation of the selected stance."}
```

Use actual model chat templates. Do not assume a numerical label is one tokenizer token. Constrained decoding or direct choice probabilities are optional later refinements; validate their tokenization and interpretation before use.

Request a short communicative explanation, not an unrestricted hidden reasoning trace. Include any system prompt, graph-relationship wording, truncation rule, and output schema in the protocol version.

### Synchronous timing

1. Generate each agent's initial choice/message from the question and persona.
2. Freeze the complete round-`t` snapshot.
3. Construct every agent's next prompt from that snapshot only.
4. Generate all round-`t+1` outputs.
5. Commit the new snapshot only when the round is complete.

Sequential execution must not let later agents see already updated same-round answers.

Make stochastic generation resumable: save/restore the MLX RNG state at checkpoint boundaries, or derive stable per-request seeds from the protocol seed plus question, graph, episode, agent, round, and retry attempt. Use a stable digest rather than Python's process-dependent `hash()`. Persist completed responses and reuse them on resume; never silently regenerate or overwrite them. Freeze the batching configuration as well. If a backend cannot preserve the required per-request random streams when batching, retain sequential generation or label the changed execution condition explicitly.

Broadcast one message per sender in the new pilot. If a future experiment generates a separate message for each recipient, give it a new protocol identifier and recalculate the cost.

This efficient joint choice/message protocol differs from the original repeated-vote protocol. Results on it are a new experiment, not a faithful regeneration of the paper.

### Dataset stages and call counts

Let `Q` be questions, `G` graphs per question, `E` repeated episodes, `N` agents, and `R` update rounds after initialization.

\[
\text{communities}=QGE,\qquad
\text{agent transitions}=QGENR,
\]
\[
\text{joint choice/message responses}=QGEN(R+1).
\]

| Stage | Q | G | E | N | R | Communities | Transitions | Joint responses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-episode microbenchmark | 2 | 1 | 1 | 8 | 4 | 2 | 64 | 80 |
| Workflow pilot | 24 | 2 | 2 | 8 | 4 | 96 | 3,072 | 3,840 |
| Later initial study | 150 | 3 | 2 | 8 | 6 | 900 | 43,200 | 50,400 |

Counts assume one response per agent at every state, including initialization and the terminal state's message. They exclude retries, question-bank preparation, and extra votes. If answers and messages use separate calls, recalculate the count.

For the workflow pilot, use a fixed question split such as 16/4/4 and describe it as a pipeline check; four test questions are not enough for strong generalization claims. For the larger study, use 90/30/30 question groups and reserve a graph family for topology transfer. Preserve degree/sign statistics as far as practical so topology changes are interpretable.

Prepare a deduplicated stance-question bank appropriate for the five labels. Do not convert binary mathematics records into ordinal agreement labels. Record question provenance and keep related variants in one split.

Measure complete-episode wall time, input/output token counts, model-load time, peak memory, format failures, and retries. Estimate larger runs from these measurements. Do not promise a completion time from model size or a headline tokens-per-second figure.

### Non-binary objectives and metrics

Train the five-way predictor with categorical cross-entropy first. Report NLL, Brier score, stay/switch behavior, and ranked probability score:

\[
\mathrm{RPS}(p,y)
=\frac{1}{4}\sum_{k=0}^{3}
\left(\sum_{\ell=0}^{k}p_\ell-\mathbf 1[y\le k]\right)^2.
\]

Compare against persistence, an own-state Markov model, and a contextual signed-aggregation model with the same five-category output. A categorical/ordinal analogue of the Ising approach must be named and specified as an extension; the binary Ising baseline cannot be scored directly on five labels.

Self-reported numerical confidence is optional metadata, not a verified probability target. One observed response per context also does not identify that context's full generator distribution.

## 10. Project files and command contract

Create a small Python package with scripts as the main execution path. Notebooks may explore results, but the experiment must be reproducible without manually executing notebook cells.

Suggested structure:

| Path | Responsibility |
| --- | --- |
| `README.md` | Setup, real commands, and study overview |
| `STATUS.md` | Progress, limitations, next command |
| `pyproject.toml` | Package and dependency declaration |
| `requirements-lock.txt` | Resolved environment record |
| `configs/paper_binary.yaml` | First subset, state conditions, forecast boundaries, and training settings |
| `configs/ordinal_pilot.yaml` | Later generation/predictor protocol |
| `src/mac_agents/data/` | Acquisition, joins, audits, splits, vote/history/message features |
| `src/mac_agents/models/` | Ising adapter, controls, graph attention, state encoders and memory |
| `src/mac_agents/train.py` | Training and checkpoint/resume |
| `src/mac_agents/evaluate.py` | One-step and closed-loop evaluation |
| `src/mac_agents/generate.py` | Local MLX generation and resume |
| `src/mac_agents/report.py` | Tables, figures, and summaries |
| `tests/` | Scientific invariants |
| `data/raw/` | Immutable downloaded inputs |
| `data/processed/` | Versioned splits/features |
| `runs/<run_id>/` | Config, checkpoints, logs, predictions |
| `reports/` | Audit, benchmark, and findings |

Keep downloaded weights, large data, caches, and run artifacts out of source control. Preserve references/checksums and acquisition scripts.

Implement a CLI with operations equivalent to the following. **These are desired interfaces to create, not commands claimed to exist already.**

```bash
python -m mac_agents.cli inspect-environment
python -m mac_agents.cli prepare-data --config configs/paper_binary.yaml
python -m mac_agents.cli audit-data --config configs/paper_binary.yaml
python -m mac_agents.cli reproduce-baseline --config configs/paper_binary.yaml
python -m mac_agents.cli train --config configs/paper_binary.yaml --model contextual_attention --seed 42
python -m mac_agents.cli evaluate --run runs/SELECTED_RUN --mode one_step
python -m mac_agents.cli evaluate --run runs/SELECTED_RUN --mode closed_loop
python -m mac_agents.cli report --experiment paper_binary
python -m mac_agents.cli benchmark-generation --config configs/ordinal_pilot.yaml
```

The `SELECTED_RUN` placeholder must be replaced with an actual completed run directory. Test-set evaluation should require a saved selection/freeze record for architecture, state condition, preprocessing, and forecast origin. It must not happen automatically during screening. Closed-loop evaluation must check that the run has an autonomous update for every required state component.

Every run should retain:

- Immutable run ID and resolved configuration.
- State condition, permitted information at the forecast boundary, history/memory initialization, and rollout eligibility.
- Data/model/code revisions and split-manifest hashes.
- Text-encoder and feature-cache revisions, truncation counts, and matched comparison subset identifiers.
- Actual device, dtype, package versions, parameter counts, and RNG seeds.
- Training/validation history, stopping reason, and timing.
- Evaluation predictions with question/graph/episode/agent/round keys.
- Checkpoints and resume state.
- A list of scientific protocol deviations.

## 11. Meaningful verification

Implement focused checks for risks that would invalidate the experiment:

1. **Split integrity:** question IDs and grouped paraphrases have disjoint train/validation/new-question-test assignments. The known-question/unseen-graph evaluation retains `question_split=train` and separately verifies held-out graph/episode membership; no evaluation episode enters fitting.
2. **Alignment:** persona order, question IDs, graph axes, and target times match source records.
3. **Masking:** changing a non-neighbor's current state at the forecast boundary cannot affect a receiver in one modeled step when already-existing history, inboxes, and memory are held fixed.
4. **One-hop timing:** on a chain, a new perturbation two hops away cannot reach the receiver in the same modeled exchange. Information propagated in earlier rounds may already be present in stored state.
5. **Permutation equivariance:** consistently permuting agents, both graph axes, message identities, histories, and memory permutes predictions correspondingly.
6. **Isolation:** zero-neighbor nodes yield finite outputs and zero social aggregate.
7. **Dimension contract:** Q/K widths are 16 and value widths are 32 in the default model.
8. **Temporal integrity:** changing information after a run's declared forecast boundary cannot change its prediction. For initial-boundary forecasts, the new messages for the target update are excluded even though they precede the target label. Rollout code never reads held-out future observations.
9. **Probability validity:** finite nonnegative probabilities summing to one.
10. **Backend agreement:** small CPU and MPS forward/loss values agree within declared numerical tolerance.
11. **Resume behavior:** a stopped deterministic training fixture and a seeded stochastic generation fixture resume without duplicate episodes, regenerated completed responses, or skipped trajectory states. Verify RNG restoration or stable request-seed assignment.
12. **Generation timing:** serial calls respect the frozen previous-round snapshot.
13. **State closure:** rollout rejects models requiring unavailable observed message or vote features; valid history/memory rollouts update every component from their own generated state.
14. **History and count alignment:** unavailable history is masked, current vote counts exclude parse failures, target-round counts cannot enter inputs, and any generated count/majority outputs are mutually consistent.

Run stochastic/equivariance checks in evaluation mode or control the randomness. Passing these checks validates implementation properties, not the scientific hypothesis.

A tiny synthetic graph fixture is useful for these checks. Label it as synthetic and keep it out of reported research results.

## 12. Milestones and completion criteria

### Phase 0 — Environment and project

Deliver the environment audit, reproducible package skeleton, config, and actual execution-location statement.

**Complete when:** CPU execution works, MPS availability is recorded, output locations are clear, and no assumption about Mac access remains hidden.

### Phase 1 — Data and baseline

Acquire the selected consolidated and raw episode data, reconstruct splits, audit schema/alignment and vote/history/message coverage, and run the author baseline check. Create the separate 16/4/20 comparison split and matched state-comparison subsets. Document forecast boundaries before creating features.

**Complete when:** join coverage, state availability, message timing, and invalid-data policy are documented; fitting/evaluation roles are clear; baseline predictions and metrics are available or a concrete source blocker is reported.

### Phase 2 — Controlled state and model comparison

Implement P0/P1/I0/A0/A1/A2 and the initial opinion, vote-count, history, and existing-message state comparisons. Run the scientific checks, follow the bounded validation screen, and save selected configurations separately for one-step prediction and autonomous simulation. Record any additional recurrent or latent-state variant as a distinct model.

**Complete when:** the required state comparisons and matched controls are evaluated, training is stable, parameter/runtime figures are measured, and selection uses validation only. A missing source or exhausted budget must leave affected comparisons explicitly blocked or pending, with the phase marked partial. Improvement over I0 and an interpretable equation are not completion requirements.

### Phase 3 — Locked evaluation and report

Run selected comparisons with three seeds where budget permits. Evaluate one-step prediction for each selected information condition, and deterministic rollout and sampled outcomes for models with autonomous state updates. Produce `reports/paper_binary_results.md` with compact tables and plots.

**Required report content:**

- Dataset/split counts and missing-data handling.
- Author reproduction separated from the controlled comparison.
- State additions, matched-example comparisons, forecast boundaries, and observed-message diagnostics separated from autonomous simulation.
- Per-model NLL, Brier, flip/class-balanced accuracy, parameter count, and wall time.
- Encoding/initialization cost, rollout eligibility, memory use, and any remaining LLM calls; measured speedup only where a matched reference exists.
- Performance by horizon and graph family.
- Paired question-level uncertainty and seed variation.
- Interpretation of whether gains reflect state information, memory, routing, capacity, or remain unresolved.
- Scope of the evidence for a compact neural rule, and a proposed targeted intervention test for any influence hypothesis.
- Honest limitations and the next experiment justified by the results.

If a budget limit prevents completion, list completed and pending runs and provide the exact resume command.

### Phase 4 — Local generator benchmark

Resolve and pin the MLX model, validate the structured output, and run the two-community benchmark within its wall-time budget.

**Complete when:** measured generation throughput, token counts, memory, and format validity support a realistic pilot estimate.

### Phase 5 — Ordinal workflow pilot

If selected as a follow-up, generate the 96-community dataset with checkpoint/resume and fit the categorical controls/predictor. Use it to test richer output categories and the generation pipeline. Its priority relative to the targeted influence experiment should follow the binary study's findings.

**Complete when:** labels, timing, splits, response counts, and ordinal metrics are validated. This pilot is not sufficient evidence for broad performance claims.

### Phase 6 — Larger study

Plan the 900-community study from measured costs, observed variability, and the remaining scientific question. This phase is outside the initial execution target.

## 13. Sources and verification notes

Links below were checked while preparing this handoff. Resolve and record immutable revisions when implementing. Performance and runtime predictions in this document are design expectations, not benchmarks.

- **[S1] Paper, version 2:** [Physics of Agents](https://arxiv.org/html/2608.16578v2), with [abstract/version history](https://arxiv.org/abs/2608.16578). Primary source for the scientific setup and reported method.
- **[S2] Official repository:** [batu-el/physics-of-agents](https://github.com/batu-el/physics-of-agents).
- **[S3] Released collection:** [physics-of-agents/agent-opinions](https://huggingface.co/datasets/physics-of-agents/agent-opinions). Primary source for the visible schema and collection metadata.
- **[S4] Implementation guide:** [implementation_details.md](https://github.com/batu-el/physics-of-agents/blob/main/implementation_details.md).
- **[S5] Author helpers:** [res/utils.py](https://github.com/batu-el/physics-of-agents/blob/main/res/utils.py).
- **[S6] Prediction analysis:** [res/4_prediction.ipynb](https://github.com/batu-el/physics-of-agents/blob/main/res/4_prediction.ipynb).
- **[S7] Reproduction caveats:** [reproducibility_audit.md](https://github.com/batu-el/physics-of-agents/blob/main/reproducibility_audit.md).
- **[S8] PyTorch MPS:** [official backend documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).
- **[S9] PyTorch attention API:** [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
- **[S10] MLX-LM:** [official repository and usage documentation](https://github.com/ml-explore/mlx-lm).
- **[S11] Generator's original model card:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
- **[S12] Frozen text-encoder candidate:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- **[S13] Quantized Mac generator:** [MLX Community Qwen3-4B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit).
- **[S14] Author generation and record timing:** [lib/datagen/dynamics.py](https://github.com/batu-el/physics-of-agents/blob/main/lib/datagen/dynamics.py). Check the pinned version and historical protocol when aligning message and opinion records.

The original paper, repository, and dataset establish the baseline problem. The compact-state research objective, neural candidates, state comparisons, intervention proposals, compute limits, and acceptance criteria are research design choices for this project. These design revisions are not claims that any experiments or data audits have been executed.

