"""Render the development-only breadth error investigation."""
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
from mac_agents.common import digest, json_write
from scripts import breadth_train as study
from scripts.local_conversation_pilot import deliver


def main():
    root = study.ROOT / "error_diagnosis_v1"
    summary = json.loads((root / "summary.json").read_text())
    groups = json.loads((root / "groups.json").read_text())
    source = json.loads((root / "source_probability_diagnosis.json").read_text())
    controls = json.loads((root / "current_state_probe/results.json").read_text())["models"]
    frame = pd.read_csv(root / "transitions_with_source_probabilities.csv")
    val = frame[frame.split == "validation"]
    v = summary["summary"]["validation"]
    votes = [r for r in groups["target_agreement"] if r["split"] == "validation"]
    changes = [r for r in groups["switch"] if r["split"] == "validation"]
    source_val = source["results"]["validation"]
    qrows = json.loads((root / "by_question.json").read_text())
    # Preserve one natural matched-state example, including the exact inbox order.
    example = json.loads((root / "same_state_probability_examples.json").read_text())[0]
    qid, agent = example["updates"][0]["qid"], example["updates"][0]["agent"]
    q = next(q for q in json.loads((study.ROOT / "selection.json").read_text())["questions"] if q["qid"] == qid)
    episode = json.loads((study.ROOT / "episodes" / q["split"] / f"{qid}__{q['graph_id']}__r00.json").read_text())
    example["question"] = q["question"]
    example["choices"] = q["choices"]
    example["correct_option"] = q["answer"]
    for update in example["updates"]:
        step = update["round"]
        senders = {int(k.split(",")[0]): text for k, text in episode["messages_history"][step-1].items()}
        agree, disagree, _ = deliver(episode["J"], senders, episode["meta"]["name"], step)
        update["messages_from_friendly_neighbors"] = agree[agent]
        update["messages_from_unfriendly_neighbors"] = disagree[agent]
        update["neighbors"] = [{"agent": i, "sign": episode["J"][agent][i],
                                "current_spin": episode["spins_history"][step-1][i]}
                               for i in range(32) if episode["J"][agent][i]]
        update["target_votes"] = episode["spins_raw_history"][step][agent]
    assert example["updates"][0]["neighbors"] == example["updates"][1]["neighbors"]
    json_write(root / "matched_message_example.json", example)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    colors = ["#64748b", "#6d28d9"]
    for ax, rows, labels, title in [
        (axes[0], votes, [f"{r['group']}/5 agree\n{100*r['update_share']:.0f}% of data" for r in votes], "A  Target vote agreement"),
        (axes[1], changes, ["Opinion stays\n66% of updates", "Opinion changes\n34% of updates"], "B  Stay versus change")]:
        x = np.arange(len(rows))
        for j, (field, label) in enumerate([("ising_error", "Ising"), ("nano_error", "Transformer")]):
            bars = ax.bar(x + (j-.5)*.34, [100*r[field] for r in rows], width=.34, color=colors[j], label=label)
            ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 46)
        ax.set_ylabel("Prediction error (%)")
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    bars = axes[2].bar([0, 1, 2], [100*v["fixed"], 100*v["introduced"], 100*v["net_error_gain"]],
                       color=["#16856b", "#c35b56", "#6d28d9"], width=.6)
    axes[2].bar_label(bars, fmt="%.2f", padding=4)
    axes[2].set_xticks([0, 1, 2], ["Ising errors\ncorrected", "New errors\nintroduced", "Net error\nreduction"])
    axes[2].set_ylabel("Percentage points of all updates")
    axes[2].set_ylim(0, 6)
    axes[2].set_title("C  Why the net gain is modest", loc="left", fontsize=11, weight="bold")
    axes[2].grid(axis="y", alpha=.15)
    axes[2].set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.055, .9), frameon=False, ncol=2)
    fig.suptitle("Where the new predictors make mistakes", x=.055, ha="left", weight="bold", fontsize=15)
    fig.text(.055, .035, "20 validation questions · 5,120 agent updates · mean of three transformer seeds · exploratory, not untouched assessment", fontsize=9, color="#475569")
    fig.subplots_adjust(left=.055, right=.985, bottom=.23, top=.73, wspace=.36)
    figure = Path("reports/question_breadth_error_diagnosis.png")
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    lines = ["# Why is the transformer advantage modest on the new Llama data?", "",
        "The diagnosis finds two important limits: substantial stochastic target variation and message information absent from both predictors. The transformer also corrects many Ising errors while introducing others, and its net gain is concentrated on opinions that stay unchanged. These findings explain parts of the gap; they do not establish an irreducible ceiling or show that an argument-aware model will generalize.", "",
        "Only the 80 training and 20 validation questions were inspected. All 40 assessment conversations remain ungenerated. The frozen 818-parameter transformer and 19-coefficient Ising comparison is unchanged. Metrics predict agents' next majority-of-five opinions, not mathematical answer correctness. Neural scores average three seeds, not an ensemble.", "",
        "## 1. Most errors occur on split-vote targets", "",
        "| Next-state vote agreement | Share of validation updates | Ising error | Transformer error | Share of transformer errors |",
        "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {r['group']}/5 | {100*r['update_share']:.2f}% | {100*r['ising_error']:.2f}% | {100*r['nano_error']:.2f}% | {100*r['nano_error_share']:.2f}% |" for r in votes]
    lines += ["", "Split votes account for 42.97% of validation updates and 81.30% of transformer errors. The old GPT development data had only 7.71% split-vote targets. The new predictor is slightly worse than Ising on the 3/5 subset; it improves both 4/5 and unanimous targets. Vote agreement is a retrospective outcome diagnostic, not a predictor feature or a direct noise ceiling.", "",
        "The screening criterion also needs interpretation: pooled validation screening entropy averaged 0.994 bits, comprising 0.873 bits of within-agent entropy and 0.121 bits of the estimated between-agent component. High pooled entropy here mostly reflects variable answers from individual agents. These finite-eight-vote estimates are descriptive; this is not evidence that stable disagreement was separately optimized by selection.", "",
        "## 2. Logged source probabilities separate sampling variation", "",
        "Recovered all 144,000 development responses and verified all 28,800 reconstructed prompt hashes against the source logs. The five samples of each prompt have identical logged A/B probabilities. There were no parse retries; 143,988 responses were single bare A/B tokens. Average probability mass outside those two first tokens is 2.18e-7 (maximum 0.001345).", "",
        "Normalize the first bare-token probabilities to obtain p=P(A), then compute the majority-of-five probability q=p³(10−15p+6p²). This is a very close first-token approximation to the full parsed-response process, not an exact population Bayes bound. The complete next prompt includes newly generated messages and is unavailable at the original pre-message forecasting boundary.", "",
        "| Validation diagnostic | Error |", "| --- | ---: |",
        f"| Full-prompt sampling reference, mean min(q,1−q) | {100*source_val['models']['I0']['sampling_reference_error']:.2f}% |",
        f"| Full-prompt most-likely-label reference, observed error | {100*source_val['full_prompt_reference_observed_error']:.2f}% |",
        f"| Ising expected error under logged source probabilities | {100*source_val['models']['I0']['expected_error']:.2f}% |",
        f"| Transformer expected error under logged source probabilities | {100*source_val['models']['nano']['expected_error']:.2f}% |", "",
        "Thus sampling variability explains a substantial portion, but not all, of the error. The transformer still has 7.02 percentage points of expected error above the full-prompt reference. Some of this difference requires information unavailable at its prediction boundary; it is not all guaranteed recoverable by a larger model using the same inputs. The aggregate predicted and observed validation A rates agree (40.69% versus 40.80%; aggregate residual z=0.33).", "",
        "For an independent check, the 15 isolated validation question/agent contexts receive the same prompt at all nine states. Every one produces both majority labels; 49.63% of within-context pairs disagree. Isolates are only 2.34% of validation updates, so this alone cannot explain the cohort-wide error.", "",
        "## 3. Identical supplied opinions can hide different message information", "",
        "Natural example: question `breadth_math_04123`, receiver 26, rounds 1 and 5. The problem asks for the remainder of `6x^4−14x^3−4x^2+2x−26` divided by `2x−6`; A=52 and B=51. The receiver's opinion, each neighbor's opinion, personas, and signed relationships are identical. The implemented one-layer graph predictor therefore has identical effective inputs, which was checked numerically.", "",
        "| Quantity | Round 1 | Round 5 |", "| --- | ---: | ---: |",
        "| Llama single-response P(A), from logged logits | 0.66% | 98.69% |",
        "| Observed five-vote outcome | B, B, B, B, B | A, A, A, A, A |",
        "| Ising P(next majority=A) | 12.36% | 12.36% |",
        "| Transformer mean P(next majority=A) | 12.27% | 12.27% |", "",
        "The friendly neighbor endorses B in both rounds. Initially its message claims substitution gives 51; later it says the calculation gives 52 and explicitly supports A, while still professing support for B. Other incoming text and ordering also change. This demonstrates sensitivity to message information beyond stated stance; it does not isolate logical quality from wording/order, nor show that the existing text encoder can exploit it. These messages are newly produced between the original current state and target, so using them would move prediction to after message generation.", "",
        "Across validation, 771/5,120 updates (15.06%) belong to repeated connected-agent input states whose logged majority probability varies by more than 25 percentage points. There are 479 updates in repeated states whose probabilities cross one half. Repeated-state comparisons are descriptive and correlated. Full inboxes and exact probabilities are saved in `matched_message_example.json`.", "",
        "An optimistic empirical reference that memorizes mean source probability for each observed local state has 10.05% expected error, compared with the 8.58% full-prompt reference. This 1.48-point difference reflects collapsing the recorded prompts into those states; it is not a population estimate of the cost of omitting arguments. Many states have few observations, and this calculation uses validation probabilities retrospectively.", "",
        "## 4. The gain is mostly persistence, with partial cancellation", "",
        "| Observed transition | Share | Ising error | Transformer error |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {'Opinion changes' if r['group']=='True' else 'Opinion stays'} | {100*r['update_share']:.2f}% | {100*r['ising_error']:.2f}% | {100*r['nano_error']:.2f}% |" for r in changes]
    lines += ["", "Of all validation updates, the transformer corrects 4.75 percentage points of Ising errors but introduces 3.18 points of new errors, leaving a 1.57-point net gain. They share 11.93 points of errors. It improves error on 12/20 validation questions and NLL on 15/20, so the gain is not uniform across questions. The first update is particularly difficult: 24.01% transformer error, versus 11.20% at round eight.", "",
        "A separate compact diagnostic fitted only the 80 training questions, with fixed penalty 0.001 and training-only standardization; no validation tuning was performed. The original Ising weights stay fixed, and the learned correction is added to their logit. These diagnostic models do not replace either frozen primary predictor.", "",
        "| Diagnostic | Additional correction coefficients | Training error | Validation error | Validation NLL |", "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {r['model']} | {r['additional_coefficients']} | {100*r['train']['error']:.2f}% | {100*r['validation']['error']:.2f}% | {r['validation']['nll']:.5f} |" for r in controls]
    lines += ["", "Calibration alone changes essentially nothing. Adding current own opinion reduces error from 16.68% to 16.21%, recovering about 30% of the transformer's net improvement. Adding the current five-vote confidence gives 16.13%. These terms explain part of the gain, not all. The fitted current-opinion coefficient is positive; because the source prompt does not explicitly insert the receiver's prior opinion, this is predictive persistence, not proof of causal self-influence.", "",
        "## Additional checks and limits", "",
        "The conservative message parser finds an explicit, unambiguous self-endorsement in 98.16% of validation sender messages; none of those declarations contradict the recorded sender opinion. This rules against widespread simple stance-label corruption in the parsed subset, but not contradictory arithmetic or reasoning within a message, as the matched example illustrates. Topic, difficulty, graph, persona, round, and question breakdowns are saved. Graph effects cannot be separated cleanly from question effects with one graph/conversation per question.", "",
        "All three new focused checks pass, all model/source integrity gates remain intact, and every one of the 28,800 prompts was reconstructed from the saved episode data and matched to the source-log hash. The 40 assessment episodes remain absent. Subgroup thresholds and follow-up probes are exploratory; validation has already selected the primary training settings.", "",
        "## Next discriminating experiment", "",
        "Use the saved source probabilities as soft training targets in a separate diagnostic, keeping the question split and architecture fixed. This can separate noisy majority-label training from difficulties learning the response distribution, without new LLM generation. Then compare an argument-aware receiver predictor at an explicit after-message boundary. A pre-message dynamics predictor instead needs to represent or average over uncertain future messages and history. Preserve the current frozen pair for the promised untouched assessment.", "",
        f"![Validation error breakdown]({figure.resolve()})", "",
        "Artifacts: `runs/question_breadth_v1/error_diagnosis_v1/`. Drivers: `scripts/diagnose_breadth_errors.py`, `scripts/diagnose_breadth_probabilities.py`, `scripts/probe_breadth_current_state.py`, and `scripts/report_breadth_errors.py`."]
    report = Path("reports/question_breadth_error_diagnosis.md")
    report.write_text("\n".join(lines) + "\n")
    study.validate_prepared()
    study.authorize_split(study.ROOT, "assessment")
    assert not list((study.ROOT / "episodes/assessment").glob("*.json"))
    json_write(root / "completion_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "models_frozen_sha256": digest(study.ROOT / "models_frozen.json"),
        "assessment_generated_or_scored": False, "source_llm_calls": 0,
        "artifacts": {str(p): digest(p) for p in list(root.rglob("*")) + [report, figure]
                      if p.is_file() and p.name != "completion_audit.json"},
        "scripts": {str(p): digest(p) for p in [Path("scripts/diagnose_breadth_errors.py"), Path("scripts/diagnose_breadth_probabilities.py"), Path("scripts/probe_breadth_current_state.py"), Path(__file__)]}})
    print(json.dumps({"report": str(report), "figure": str(figure), "frozen_models_verified": True, "assessment_untouched": True}))


if __name__ == "__main__":
    main()
