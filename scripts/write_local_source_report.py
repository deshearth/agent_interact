"""Write the reviewable report and scientific figure for local Llama pilots."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import argument_intervention_pilot as io
from scripts.local_argument_source import ROOT


def pp(x):
    return f"{100 * x:.2f}"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    model = read(ROOT / "model_provenance.json")
    audit = read(ROOT / "intervention/analysis/audit.json")
    summary = pd.read_csv(ROOT / "intervention/analysis/summary.csv")
    pairs = pd.read_csv(ROOT / "intervention/analysis/paired_contexts.csv")
    labels = [("valid_minus_invalid", "Valid vs minimally flawed rationale"),
              ("valid_paraphrase_minus_valid", "Two valid phrasings"),
              ("valid_minus_conclusion_only", "Valid rationale vs conclusion only")]
    conv_path = ROOT / "conversations/analysis/audit.json"
    ablation_path = ROOT / "rationale_ablation/analysis/audit.json"
    conv = read(conv_path) if conv_path.exists() else None
    ablation = read(ablation_path) if ablation_path.exists() else None
    grading_path = ROOT / "direct_argument_grading/analysis/audit.json"
    grading = read(grading_path) if grading_path.exists() else None
    word_path = ROOT / "word_argument_grading/analysis/audit.json"
    word_grading = read(word_path) if word_path.exists() else None
    quality_path = ROOT / "quality_audit/audit.json"
    quality = read(quality_path) if quality_path.exists() else None
    table = summary[summary.relationship_sign == 0].set_index("contrast")
    text = ["# Local open-weight source-model pilot", "",
        "The paper does include open-weight source models: **Llama-3.1-8B-Instruct, Qwen3.5-9B and Gemma-3n-E4B-it**, alongside GPT-4o-mini. See [Appendix B.3](https://arxiv.org/html/2608.16578v2#A2.SS3). The released metadata agrees; older backend-registry names and folder names are not the authoritative model versions.", "",
        "**New local inference has been executed.** This study uses a frozen 4-bit MLX conversion of Llama-3.1-8B-Instruct on the Mac Studio M2 Max (32 GB). It needs no API credential and made **zero paid API calls**. It is a separate local source model: quantization, serving backend and seeded sampling differ from the paper's hosted run. These measurements do not establish why the earlier GPT-4o-mini predictor failed to benefit from argument features.", "",
        "## Single-message intervention: complete", "",
        "There are 24 fixed receiver contexts across four reused development math questions, with three receivers for each question/relationship-sign combination. The background inboxes are from the earlier GPT-generated data. Their selection used mixed opinions and uncertainty of a frozen no-text predictor before observing these Llama responses. Each context has four versions: valid rationale, a minimal mathematical error with the same conclusion, a valid paraphrase, and the conclusion alone. Only one target message changes; expertise, question, choices, all other messages, graph cue and order stay fixed.", "",
        "All **1,920 independent full responses** completed: 20 draws for each of 96 prompts, yielding **384 preassigned disjoint five-vote outcomes**. All responses were exactly A or B and stopped normally; no parsing failures or replacement samples occurred. Temperature is 0.7, top-p is 1, and the output cap is 256 tokens. The released instruction explicitly asks for a snap judgment without working through the problem.", "",
        "| Intervention | Mean absolute probability change (pp) | Mean signed change toward target stance (pp) | Observed raw-response difference (pp) | Observed five-vote difference (pp) |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for key, title in labels:
        r = table.loc[key]
        text.append(f"| {title} | {pp(r.first_literal_mean_absolute_change)} | {pp(r.first_literal_mean_signed_change)} | {pp(r.raw_mean_change)} | {pp(r.majority_mean_change)} |")
    text += ["", "The probability columns use the model's computed **first literal-A/B token distribution**, with the full vocabulary retained. They are distinct from sampled complete-response fractions and majority-of-five outcomes. The maximum other-token probability was " + f"{audit['max_other_first_token_mass']:.3g}" + "; all observed responses contained one A/B token followed by EOS. No probability was renormalized onto A/B. Signed changes use valid minus flawed, paraphrase minus valid, and valid minus conclusion-only, respectively.", "",
        "The local source is sensitive to message content. Deleting a rationale has a substantially larger effect than the minimal validity edit, and rewording a valid argument also has a larger average effect than that edit. This is evidence of **text dependence**, not a clean demonstration that the model recognizes mathematical validity. A valid rationale can sometimes reduce support for its own conclusion. The deletion and paraphrase contrasts also change wording and sometimes length. Friendly and unfriendly contexts are different contexts, so comparing their averages does not estimate the causal effect of changing a graph sign.", "",
        "The validity contrast has a limited interpretation: its flawed rationale still asserts the correct final choice. A solver could retain that choice after independently solving the question or consulting the other messages, even if it notices the flaw. A weak effect here therefore does not show an inability to recognize invalid reasoning.", "",
        "| Intervention | Friendly: mean absolute change (pp) | Unfriendly: mean absolute change (pp) |",
        "| --- | ---: | ---: |"]
    for key, title in labels:
        s = summary[summary.contrast == key].set_index("relationship_sign")
        text.append(f"| {title} | {pp(s.loc[1, 'first_literal_mean_absolute_change'])} | {pp(s.loc[-1, 'first_literal_mean_absolute_change'])} |")
    text += ["", "Only 20 raw draws and four five-vote groups are available per prompt. The raw and majority differences are noisy pilot estimates; in particular, their direction can differ for a small effect. Per-prompt marginal Wilson intervals are saved in the data. The computed first-token probabilities avoid this Monte Carlo uncertainty but remain conditional on the chosen contexts and local model. There is no fresh-question confidence claim and no multiplicity-adjusted hypothesis test.", "",
             "## Fully new Llama conversations", ""]
    if conv:
        status = read(ROOT / "conversations/collection_status.json")
        text += [f"**Complete:** {conv['episodes']} new conversations, **{conv['transitions']} agent transitions**, {conv['opinion_response_attempts']:,} opinion-response attempts and {conv['message_responses']} generated broadcast messages. Every opinion and message comes from the local Llama model. There are 32 fixed expert personas, two synchronous update rounds, one replica per question, and signed graph J0 (112 directed nonzero edges: 70 friendly and 42 unfriendly). One agent is isolated, leaving 31 message senders per round. The paper's Llama protocol also uses one message per sender, broadcast to its neighbors; see [Appendix B.5](https://arxiv.org/html/2608.16578v2#A2.SS5).", "",
            f"The eight questions are four intervention questions plus the first four other development question IDs in lexicographic order, selected before viewing local outcomes. They remain reused objective math questions. Across the {conv['transitions']} updates, **{conv['changed_transitions']} ({pp(conv['change_fraction'])}%)** changed majority opinion; **{pp(conv['unanimous_update_vote_fraction'])}%** of update vote groups were unanimous. These describe dynamics, not prediction accuracy or mathematical correctness.", "",
            f"Opinion parsing failures: **{conv['opinion_parse_failures']}**. Messages hitting the 256-token cap: **{conv['truncated_messages']}/{conv['message_responses']}**. All messages, including truncated or mathematically inconsistent ones, are retained. The generator is asked for a rationale supporting its current choice; the resulting rationale is not guaranteed to be correct or internally consistent. Collection took {status['elapsed_seconds_this_invocation']/60:.1f} minutes in its final invocation, with peak MLX memory {status['peak_memory_gb']:.2f} GB.", "",
            "The audit reconstructs every opinion prompt and message prompt, all raw-vote majorities, signed inbox ordering and the shared broadcasts from the saved records. Exact user prompts, rendered chat token IDs, sampled token IDs, seeds, output termination, model identity and source-code hashes are retained. Parse retries follow the released three-attempt limit and are fully logged; persistent failures would become zero. The isolated agent supplies repeated identical-prompt controls.", ""]
    else:
        text += ["Generation is in progress. The frozen plan contains eight questions × 32 agents × two updates = 512 transitions, with five raw votes per opinion. No completed-cohort result is reported until collection and reconstruction audits finish.", ""]
    text += ["## Rationale deletion within the new conversations", ""]
    ablation_pairs = None
    if ablation:
        abl = pd.read_csv(ROOT / "rationale_ablation/analysis/summary.csv").set_index("subset")
        ablation_pairs = pd.read_csv(ROOT / "rationale_ablation/analysis/paired_contexts.csv")
        text += [f"**Complete:** reran all {ablation['contexts']} original update contexts with every incoming message replaced by its sender's recorded prompted choice, e.g. `I favor option A.` The question, expertise, prior states, graph signs, sender association and inbox order are fixed. This adds {ablation['response_attempts']:,} response attempts. The protocol selected all updates before observing their effects.", "",
            "Original messages were audited for an explicit stance near their start. A missing match is an unknown, not automatically an instruction violation. A matched stance can still accompany an incorrect or contradictory reason. The explicit-consistent subset below excludes inboxes with any unmatched or conflicting stance, so the core comparison does not rely solely on assuming that the message writer obeyed its prompt.", "",
            "| Contexts | Count | Mean absolute first-A probability change (pp) | Dominant first-token choice changes | Sampled majority changes |",
            "| --- | ---: | ---: | ---: | ---: |"]
        for key, name in [("all", "All updates"), ("nonempty", "Nonempty inboxes"),
                          ("explicit_consistent_stances_nonempty", "Nonempty, explicit matching stances"),
                          ("empty_inbox_negative_controls", "Empty-inbox controls")]:
            r = abl.loc[key]
            text.append(f"| {name} | {int(r.contexts)} | {pp(r.mean_absolute_first_A_change)} | {int(r.first_A_dominant_token_flips)} | {int(r.sampled_majority_flips)} |")
        text += ["", "A dominant first-token change compares which of A and B has greater first-token probability (ties choose A). No probability mass is renormalized. A sampled-majority change compares two independent groups of only five draws and therefore also includes ordinary sampling variation. Neither is a neural predictor error rate. Empty-inbox prompts are byte-identical across conditions; their first-token probabilities must agree within 1e-6, even when sampled majorities differ.", "",
                 "For the explicit-matching-stance pairs, an opinion/history-only predictor has identical inputs in both conditions and must return the same probability. If the teacher probabilities are p and q, its minimum average squared error against those two probabilities is (p−q)²/4. The saved summary reports the average of this bound. This is a structural limitation on the deliberately balanced paired benchmark, not an estimated error floor on natural conversations or the historical majority-vote task.", "",
                 "This intervention tests information in the delivered text beyond the stated choices. It changes content, wording and length together, and deletion prompts need not occur naturally in the original message-generation process. It does not isolate logical validity or estimate a natural-distribution lower bound on opinion-only prediction error.", ""]
        if quality:
            text += [f"A **post-hoc quality check** excludes truncated incoming messages and conservatively screens explicit stance mentions throughout each message. In the resulting {quality['strict_untruncated_consistent_contexts']} contexts, the mean absolute probability change remains **{pp(quality['strict_subset_mean_absolute_first_A_change'])} pp**, with **{quality['strict_subset_dominant_A_B_choice_changes']}** dominant-choice changes. None of the primary matching-stance contexts failed an additional guard against mistaking the article 'a' for label A near the message start. These regex checks do not certify mathematical correctness or full semantic consistency. The original primary cohort remains unchanged; details are in `quality_audit/`.", ""]
    else:
        text += ["A frozen follow-up selects all 512 original update contexts. It will preserve every graph cue and prompted stance while replacing each rationale with the conclusion alone. It also audits whether the original message explicitly stated the instructed stance. The analysis will separately report the matching-stance subset and empty-inbox negative controls. No result is reported until that collection finishes.", ""]
    text += ["## Direct check of argument-validity recognition", ""]
    if grading:
        grades = pd.read_csv(ROOT / "direct_argument_grading/analysis/summary.csv").set_index("condition")
        text += [f"A separate grading task is complete: **{grading['responses']:,} responses** to 72 prompts covering the same four questions, three argument versions and the first three canonical personas. The valid/invalid meanings of grading labels A and B are reversed in half the prompts. This changes the task to explicit mathematical grading, not an agent interaction; no communication graph was removed from the conversation experiment.", "",
            f"The primary comparison gives equal weight to valid and flawed arguments (48 prompts). Sampled grading accuracy is **{pp(grading['primary_raw_grading_accuracy'])}%**, against a 50% balanced baseline. Mean first-token probability assigned to the correct grading label is **{pp(grading['primary_mean_correct_first_literal_probability'])}%**. Choosing the more probable of the two grading labels is correct on **{pp(grading['primary_literal_pair_choice_accuracy'])}%** of these prompts. Valid paraphrases are a separate secondary condition; pooling all three would give an always-valid strategy a 2/3 baseline.", "",
            "| Argument | P(first token is the valid grading label) | Sampled grading accuracy |",
            "| --- | ---: | ---: |"]
        for key, name in (("valid", "Valid"), ("invalid", "Minimally flawed"), ("valid_paraphrase", "Valid paraphrase")):
            r = grades.loc[key]
            text.append(f"| {name} | {pp(r.p_valid_first_literal)}% | {pp(r.raw_correct_fraction)}% |")
        text += ["", f"Reversing grading labels changes the valid-label probability by {pp(grading['mean_absolute_label_swap_probability_change'])} percentage points on average. There are {grading['parse_failures']} parsing failures; maximum other-first-token mass is {grading['max_other_first_token_mass']:.3g}. Full per-question and label-swap results are saved. This is a small diagnostic on four constructed examples, not a general proof-verification benchmark. It assesses the edited reasoning when explicitly requested; it does not causally identify why an ordinary opinion update accepts or rejects a message.", ""]
    else:
        text += ["A 72-prompt direct-grading check is prepared, with three argument variants, three canonical personas and both A/B grading-label assignments for each of the four questions. It is a separate capability/manipulation diagnostic, not a change to the agent graph. No grading result is reported until collection completes.", ""]
    if word_grading:
        words = pd.read_csv(ROOT / "word_argument_grading/analysis/summary.csv").set_index("condition")
        text += ["### Follow-up using VALID/FLAWED verdict words", "",
            f"After observing the A/B grader's label sensitivity, a separate response-format follow-up collected **{word_grading['responses']:,} responses** to the same arguments and personas. It asks for VALID or FLAWED and reverses the definition order in half the prompts. The primary balanced valid-versus-flawed accuracy is **{pp(word_grading['primary_raw_grading_accuracy'])}%**, compared with {pp(grading['primary_raw_grading_accuracy'])}% for the earlier A/B grading task.", "",
            "| Argument | Fraction of sampled VALID verdicts | Sampled grading accuracy |",
            "| --- | ---: | ---: |"]
        for key, name in (("valid", "Valid"), ("invalid", "Minimally flawed"), ("valid_paraphrase", "Valid paraphrase")):
            r = words.loc[key]
            text.append(f"| {name} | {pp(r.raw_valid_verdict_fraction)}% | {pp(r.raw_correct_fraction)}% |")
        text += ["", f"Verdict parsing failures: {word_grading['verdict_parse_failures']}; strict-format failures: {word_grading['verdict_strict_format_failures']}. Definition order changes the sampled VALID fraction by {pp(word_grading['mean_absolute_definition_order_sampled_fraction_change'])} pp on average. This number includes 20-draw sampling noise and is a different contrast from the first study's exact A/B label-swap probability change. The common source logger's first-A/B probabilities are deliberately not used for word verdicts.", "",
            f"Changing the response vocabulary did not rescue validity discrimination: the model called {pp(words.loc['valid', 'raw_valid_verdict_fraction'])}% of valid arguments and {pp(words.loc['invalid', 'raw_valid_verdict_fraction'])}% of flawed arguments VALID. The two unparsed outputs were INVALID on valid-paraphrase prompts; they do not affect the primary valid-versus-flawed comparison.", "",
            "This follow-up is adaptive and uses the same four problems. It tests robustness to response format, preserving the first result; it is not an independent confirmation of general reasoning ability. Both grading tasks request short final responses, so failure in this setting does not establish what the model could do with an explicit reasoning trace or another prompt regime.", ""]
    else:
        text += ["An adaptive follow-up using direct VALID/FLAWED words is prepared after the A/B-label sensitivity finding. It preserves the original arguments and personas, reverses definition order, and will score complete raw verdicts rather than A/B token diagnostics. No result is reported until it completes.", ""]
    text += ["## Implication for the compact update rule", "",
        "The earlier GPT-trained transformer receives individual message embeddings, but its familiar-question gain could be reproduced with question IDs alone. These new measurements establish a workable local teacher and a direct way to distinguish message sensitivity from what the compact predictor learns. They do not demonstrate improved neural prediction: **no predictor has been trained or scored on this new Llama cohort**. The existing GPT-trained predictors are not evaluated as if they were matched Llama predictors.", "",
        "The next training study should use one pinned source model, a substantially broader bank of questions, and a question-level split fixed before fitting. Save matched argument variants as well as ordinary trajectories, compare stance/history-only and argument-aware models, and score both their ordinary predictions and their response to the paired changes. Exact first-token probabilities can supply a low-noise teacher diagnostic, but using them as targets would be a different task from the historical majority-of-five target and must be declared. A direct explanation of the earlier GPT result still requires a matching GPT-source intervention or a clearly labeled cross-model comparison.", "",
        "For further validity experiments, use hand-checked or independently verified labels: the local model's short-response self-grades are unreliable on these examples. Reasoning-permitted prompts or another source model would be separate follow-up regimes. The present result supports studying the information carried by message text, while leaving logical-quality recognition and the earlier GPT predictor's limitation unresolved.", "",
        "The prescribed friendly/unfriendly graph remains part of the experiment. Its sign places each message under the released agree/disagree heading; it is not a learned credibility weight. Emergent relationships remain deferred.", "",
        "## Read-only probe of the existing predictors", "",
        "Before local source collection, the nine frozen GPT-trained predictors (three model families × three seeds) were queried on the four-condition prompts: 864 predictions and 648 paired comparisons. The 648 predictions shared with the previous three-condition probe reproduced within 4.77e-7. There were zero optimizer steps and no counterfactual accuracy evaluation.", "",
        "| Frozen predictor | Minimal-error mean absolute change (pp) | Rationale-deletion mean absolute change (pp) | Valid-paraphrase mean absolute change (pp) |",
        "| --- | ---: | ---: | ---: |",
        "| Original PCA3 | 0.90 | 4.65 | 5.13 |",
        "| Full question context | 0.34 | 3.76 | 2.28 |",
        "| Joint question-conditioned embedding | 0.16 | 2.47 | 0.54 |", "",
        "These are sensitivity measurements of predictors trained on another teacher, not errors against the new Llama source. Their target is a majority-of-five outcome, which also differs from the first-token diagnostic above. The new conclusion-only probe is saved in `runs/argument_intervention_v1/predictor_probe/`; prior checkpoints and caches remain frozen.", "",
        "## Model and checks", "",
        f"- Conversion: [`{model['repository']}`](https://huggingface.co/{model['repository']}), pinned revision `{model['revision']}`. All downloaded files have saved SHA-256 hashes.",
        f"- Runtime: MLX {model['packages']['mlx']}, MLX-LM {model['packages']['mlx-lm']}, Transformers {model['packages']['transformers']}; separate `.venv-source`, recorded in `requirements-source-lock.txt`. Existing PyTorch environment and prior models remain separate.",
        "- Cache benchmark: cached and uncached first-token probabilities agreed exactly on four prompts; repeated cached draws with the same seed reproduced tokens. Only deterministic prompt KV state is copied between independent responses.",
        "- Focused local checks: 14 passing cases covering independent sample identities, five-vote groups, parsing/retry bookkeeping, broadcast/sign handling, synchronous phases, exact released prompts, resume identity, rationale-ablation controls, counterbalanced grading labels and word-verdict parsing/order controls.",
        f"- Sampling diagnostic: observed {audit['sampling_calibration_check']['observed_A']} A responses versus {audit['sampling_calibration_check']['expected_A_from_logits']:.2f} expected from saved probabilities; pooled residual {audit['sampling_calibration_check']['pooled_standardized_residual']:.2f} standard deviations. A parametric binomial calibration simulation gives tail fraction {audit['sampling_calibration_check']['parametric_monte_carlo_tail_fraction']:.3f}; this check is consistent with the intended sampler, not proof of independence.",
        "- Prior API-pilot preparation has 16 passing checks and five parent checks. It remains unexecuted; no API credential was used. Its 96 prompts were reused as inputs for the separately identified local intervention.", "",
        "## Artifacts and reproduction", "",
        "- `runs/local_llama_intervention_v1/model_provenance.json`: pinned local model and file hashes.",
        "- `runs/local_llama_intervention_v1/intervention/`: four-condition prompts, 1,920 raw samples, five-vote groups, paired contrasts and per-question summaries.",
        "- `runs/local_llama_intervention_v1/conversations/`: complete episode files, every source prompt/response, phase checkpoints and reconstruction audit.",
        "- `runs/local_llama_intervention_v1/rationale_ablation/`: all-context plan, controlled responses and paired analysis.",
        "- `runs/local_llama_intervention_v1/direct_argument_grading/`: counterbalanced validity-grading prompts, outputs and per-question analysis.",
        "- `runs/local_llama_intervention_v1/word_argument_grading/`: adaptive VALID/FLAWED response-format follow-up, kept separate from A/B grading.",
        "- `runs/local_llama_intervention_v1/teacher_pairs/`: 512 exported paired examples with pre-update inputs separated from raw-vote and first-token outcomes. This export declares no train/test split.",
        "- `reports/local_source_checks.xml`, `reports/local_intervention_analysis.log`, `reports/local_conversation_analysis.log`, `reports/local_ablation_analysis.log`: validation records.", "",
        "```bash",
        "# Read-only analysis of completed local data; no source generation:",
        ".venv/bin/python scripts/analyze_local_source.py intervention",
        ".venv/bin/python scripts/analyze_local_source.py conversations",
        ".venv/bin/python scripts/analyze_local_source.py ablation",
        ".venv/bin/python scripts/analyze_local_source.py grading",
        ".venv/bin/python scripts/analyze_local_source.py word_grading",
        ".venv/bin/python scripts/export_local_teacher_pairs.py",
        ".venv/bin/python scripts/local_source_quality_audit.py",
        ".venv/bin/python scripts/write_local_source_report.py",
        ".venv/bin/python -m pytest tests/test_local_source_pilot.py",
        "```", "",
        "Generation entry points are `local_argument_source.py benchmark`, `local_argument_intervention.py`, `local_conversation_pilot.py collect`, `local_rationale_ablation.py collect`, `local_argument_grading.py collect`, and `local_word_grading.py collect`, using `.venv-source/bin/python` with `HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false`. Collectors check frozen protocols, reuse logged sample identities on resume and honor the session deadline. Use collector commands only to resume unfinished runs: rerunning a completed collector can rewrite its invocation timestamps. Use the read-only analysis commands for completed cohorts. A future expanded cohort should get a new namespace and explicit bounded protocol; do not rewrite this cohort to change model versions or sampling settings.", "",
        "![Local source sensitivity](local_open_model_pilot.png)", "",
        f"Report generated {datetime.now(timezone.utc).isoformat()}. These are exploratory source-model experiments on existing development questions; the previously consumed test set is unchanged."]
    Path("reports/local_open_model_pilot.md").write_text("\n".join(text) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2 if ablation_pairs is not None else 1, figsize=(12 if ablation_pairs is not None else 7, 5), squeeze=False)
    ax = axes[0, 0]
    positions = np.arange(3)
    for sign, offset, label, color in [(1, -.18, "Friendly", "#3274a1"), (-1, .18, "Unfriendly", "#d28c3f")]:
        values = [summary[(summary.contrast == key) & (summary.relationship_sign == sign)].iloc[0].first_literal_mean_absolute_change * 100 for key, _ in labels]
        bars = ax.bar(positions + offset, values, width=.34, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_xticks(positions, ["Minimal error", "Valid paraphrase", "Remove rationale"])
    ax.set_ylabel("Mean absolute first-token probability change (pp)")
    ax.set_title("One message changes; conclusion held fixed\n24 fixed inboxes / 4 reused questions")
    ax.set_ylim(0, max(summary.first_literal_mean_absolute_change) * 125)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.15)
    if ablation_pairs is not None:
        ax = axes[0, 1]
        good = ablation_pairs.all_incoming_stances_explicit_and_consistent & ~ablation_pairs.empty_inbox
        ax.scatter(ablation_pairs.loc[~good, "full_p_first_A"], ablation_pairs.loc[~good, "conclusion_p_first_A"], s=12, alpha=.35, color="#999999", label="Other contexts")
        ax.scatter(ablation_pairs.loc[good, "full_p_first_A"], ablation_pairs.loc[good, "conclusion_p_first_A"], s=16, alpha=.6, color="#3274a1", label="Explicit matching stances")
        ax.plot([0, 1], [0, 1], "--", color="#666666", lw=1)
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel="P(first token A), original inbox", ylabel="P(first token A), conclusions only",
               title="All rationales removed in new Llama conversations\n512 paired updates / 8 reused questions")
        ax.set_aspect("equal")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("reports/local_open_model_pilot.png", dpi=180)
    fig.savefig("reports/local_open_model_pilot.pdf")
    plt.close(fig)
    print("Wrote reports/local_open_model_pilot.md, .png and .pdf")


if __name__ == "__main__":
    main()
