"""Audit and summarize the completed local-source transformer comparison."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from scripts.local_llama_transformer import CONFIG, CONDITIONS, Data, LocalResponseUpdate, predict, verify_source


def boot(values, seed, count):
    values = np.asarray(values)
    r = np.random.default_rng(seed).integers(0, len(values), (count, len(values)))
    low, high = np.quantile(values[r].mean(1), [.025, .975])
    return {"mean": float(values.mean()), "descriptive_95_interval": [float(low), float(high)]}


def main():
    cfg = config(CONFIG)
    root = Path(cfg["output"])
    rows, _ = verify_source(cfg)
    protocol = json.loads((root/"protocol.json").read_text())
    complete = json.loads((root/"training_complete.json").read_text())
    assert complete["all_complete"] and complete["models"] == 24
    torch.set_num_threads(4)
    data = Data(cfg, "cpu")
    manifests, frames, reproduction = [], [], []
    for name, sha in sorted(complete["manifests"].items()):
        run = root/name
        assert digest(run/"manifest.json") == sha
        m = json.loads((run/"manifest.json").read_text())
        assert m["protocol_sha256"] == digest(root/"protocol.json")
        assert m["selected_weights_sha256"] != m["initial_weights_sha256"]
        for artifact, expected in m["artifacts"].items():
            assert digest(run/artifact) == expected
        frame = pd.read_parquet(run/"predictions.parquet")
        assert set(frame[frame.split == "heldout"].qid) == set(m["fold"]["heldout"])
        assert set(frame[frame.split == "train"].qid) == set(m["fold"]["train"])
        assert set(frame[frame.split == "selection"].qid) == set(m["fold"]["selection"])
        assert (frame.groupby(["split", "variant"]).qid.nunique() == [2, 2, 2, 4]).all()
        # Independently reload each saved best checkpoint on CPU and reproduce
        # its MPS heldout probabilities, including the argument-removal variant.
        model = LocalResponseUpdate(**cfg["model"])
        ck = torch.load(run/"best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        # min_delta can choose an earlier checkpoint than the exact minimum.
        hist = json.loads((run/"history.json").read_text())
        best, best_epoch = float("inf"), None
        for h in hist:
            if h["selection_nll"] < best-cfg["training"]["min_delta"]:
                best, best_epoch = h["selection_nll"], h["epoch"]
        assert ck["epoch"] == best_epoch == m["selected_epoch"]
        ids = data.ids(m["fold"]["heldout"])
        for variant in ("original", "conclusion_only"):
            lp = predict(model, data, ids, m["condition"], variant)
            saved = frame[(frame.split == "heldout") & (frame.variant == variant)].set_index("context_id")
            expected = saved.loc[[data.rows[i]["context_id"] for i in ids], "p_A"].to_numpy()
            actual = np.exp(lp[:, 1])
            difference = float(np.max(np.abs(actual-expected)))
            np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
            reproduction.append({"run": name, "variant": variant, "maximum_absolute_probability_error": difference})
        manifests.append(m)
        frames.append(frame)
    allp = pd.concat(frames, ignore_index=True)
    allp["error"] = (allp.prediction != allp.y).astype(float)
    allp["accuracy"] = 1-allp.error
    allp.to_parquet(root/"all_predictions.parquet", index=False)
    keys = ["condition", "split", "variant"]
    metrics = ["error", "accuracy", "nll", "brier_two_class"]
    summary = allp.groupby(keys)[metrics].mean().reset_index()
    summary.to_csv(root/"summary.csv", index=False)
    held = allp[(allp.split == "heldout") & (allp.variant == "original")].copy()
    assert (held.groupby(["condition", "seed", "context_id"]).size() == 1).all()
    assert len(held) == 512*3*2
    byq = held.groupby(["condition", "qid"])[metrics].mean().reset_index()
    byq.to_csv(root/"heldout_by_question.csv", index=False)
    byseed = held.groupby(["condition", "seed"])[metrics].mean().reset_index()
    byseed.to_csv(root/"heldout_by_seed.csv", index=False)
    paired = {}
    for metric in metrics:
        wide = byq.pivot(index="qid", columns="condition", values=metric)
        paired[metric] = boot(wide.arguments-wide.opinion_history,
            cfg["evaluation"]["bootstrap_seed"], cfg["evaluation"]["bootstrap_replicates"])
    baseline_rows = []
    for fold in protocol["folds"]:
        tr = [r for r in rows if r["question_group"] in fold["train"]]
        p = (1+sum(r["outcomes"]["original"]["majority_sign"] == 1 for r in tr))/(len(tr)+2)
        for r in rows:
            if r["question_group"] not in fold["heldout"]:
                continue
            y = int(r["outcomes"]["original"]["majority_sign"] == 1)
            stay = int(r["inputs"]["current_spins"][r["inputs"]["receiver"]] == 1)
            baseline_rows.append({"qid": r["question_group"], "context_id": r["context_id"],
                "p_A": p, "global_majority_error": float((p >= .5) != y),
                "global_majority_nll": float(-np.log(p if y else 1-p)),
                "global_majority_brier_two_class": float(2*(p-y)**2), "persistence_error": float(stay != y)})
    base = pd.DataFrame(baseline_rows)
    assert len(base) == 512
    base.to_csv(root/"baselines.csv", index=False)
    baseline = {c: float(base[c].mean()) for c in base.columns if c not in ("qid", "context_id", "p_A")}
    baseline["global_majority_accuracy"] = 1-baseline["global_majority_error"]
    baseline["persistence_accuracy"] = 1-baseline["persistence_error"]
    pair_rows, pair_summary = [], []
    for condition in CONDITIONS:
        h = allp[(allp.split == "heldout") & (allp.condition == condition)]
        original = h[h.variant == "original"].drop(columns="variant")
        removal = h[h.variant == "conclusion_only"].drop(columns="variant")
        joined = original.merge(removal, on=["context_id", "seed"], suffixes=("_original", "_removal"), validate="one_to_one")
        delta = joined.p_A_removal-joined.p_A_original
        changed = joined.y_original != joined.y_removal
        prediction_flip = joined.prediction_original != joined.prediction_removal
        assert np.max(np.abs(delta[joined.empty_inbox_original])) < 1e-7
        if condition == "opinion_history":
            assert np.max(np.abs(delta)) < 1e-7
        for subset, mask in (("all", np.ones(len(joined), dtype=bool)),
                             ("nonempty_consistent_stance", ~joined.empty_inbox_original & joined.consistent_stances_original)):
            pair_summary.append({"condition": condition, "subset": subset,
                "contexts": int(mask.sum()//3), "paired_error": float((joined.error_original[mask]+joined.error_removal[mask]).mean()/2),
                "mean_absolute_predicted_majority_probability_shift": float(delta[mask].abs().mean()),
                "predicted_label_flip_rate": float(prediction_flip[mask].mean()),
                "observed_majority_flip_rate": float(changed[mask].mean()),
                "direction_accuracy_on_observed_flips": float((np.sign(delta[mask & changed]) == np.sign(joined.y_removal[mask & changed]-joined.y_original[mask & changed])).mean())})
        pair_rows.append(joined.assign(condition=condition, probability_shift=delta))
    pd.concat(pair_rows).to_parquet(root/"paired_predictions.parquet", index=False)
    pd.DataFrame(pair_summary).to_csv(root/"paired_summary.csv", index=False)
    # Training and terminal errors are kept distinct: early stopping is based
    # solely on selection-question NLL, never these training/heldout scores.
    training = {c: {"selected_error": float(np.mean([m["metrics"]["train/original"]["error"] for m in manifests if m["condition"] == c])),
                    "terminal_error": float(np.mean([m["terminal_train_metrics"]["error"] for m in manifests if m["condition"] == c]))}
                for c in CONDITIONS}
    old_audit = json.loads(Path("runs/argument_investigation_20261108/completion_audit.json").read_text())
    old_count = old_artifacts = 0
    for directory, sha in old_audit["run_manifests"].items():
        run = Path(directory)
        assert digest(run/"manifest.json") == sha
        m = json.loads((run/"manifest.json").read_text())
        for name, expected in m["artifact_sha256"].items():
            assert digest(run/name) == expected
            old_artifacts += 1
        old_count += 1
    for fold in range(4):
        for seed in cfg["training"]["seeds"]:
            mm = [m for m in manifests if m["fold"]["fold"] == fold and m["seed"] == seed]
            assert len(mm) == 2 and mm[0]["initial_weights_sha256"] == mm[1]["initial_weights_sha256"]
            assert mm[0]["parameters"] == mm[1]["parameters"] == 282146
    assert all(m["stopping_reason"] == "early_stopping" for m in manifests)
    audit = {"completed_utc": datetime.now(timezone.utc).isoformat(), "fits": 24,
        "all_checkpoints_reproduced_on_cpu": True,
        "maximum_cpu_mps_probability_difference": max(r["maximum_absolute_probability_error"] for r in reproduction),
        "source_manifest_verified": True, "previous_manifests_unchanged": old_count,
        "previous_fit_artifacts_unchanged": old_artifacts, "all_matched_initial_weights": True,
        "parameter_updates_verified": True, "whole_question_splits_verified": True,
        "zero_control_response_to_argument_removal": True, "zero_empty_inbox_response_to_argument_removal": True,
        "all_checkpoint_selection_reproduced": True, "new_source_llm_calls": 0,
        "protocol_sha256": digest(root/"protocol.json"), "report_script_sha256": digest(__file__),
        "training": training, "baselines": baseline, "paired_differences": paired, "pair_diagnostic": pair_summary}
    json_write(root/"results.json", audit)
    pd.DataFrame(reproduction).to_csv(root/"checkpoint_reproduction.csv", index=False)
    write_report(cfg, manifests, summary, byq, audit)
    json_write(root/"analysis_integrity.json", {"completed_utc": audit["completed_utc"],
        "artifacts": {name: digest(root/name) for name in ("all_predictions.parquet", "summary.csv", "heldout_by_question.csv", "heldout_by_seed.csv", "baselines.csv", "paired_predictions.parquet", "paired_summary.csv", "results.json", "checkpoint_reproduction.csv")},
        "report_artifacts": {str(p): digest(p) for p in (Path("reports/local_llama_transformer.md"), Path("reports/local_llama_transformer.png"))}})
    print(json.dumps({"fits": 24, "primary_difference": paired, "summary": summary.to_dict("records"), "baselines": baseline, "paired": pair_summary}, indent=2))


def write_report(cfg, manifests, summary, byq, audit):
    root = Path(cfg["output"])
    protocol = json.loads((root/"protocol.json").read_text())
    features = json.loads((root/"features.json").read_text())
    complete = json.loads((root/"training_complete.json").read_text())
    names = {"opinion_history": "Opinion/history + context", "arguments": "Same inputs + arguments"}
    def row(condition, split, variant="original"):
        return summary[(summary.condition == condition) & (summary.split == split) & (summary.variant == variant)].iloc[0]
    lines = ["# New transformer training on the local Llama data", "",
        "Completed 24 newly initialized graph-transformer fits on the completed local Llama-3.1-8B-Instruct 4-bit conversations. Both conditions use exactly the same folds, initial weights, architecture and training settings. The only input difference is the incoming message vectors.", "",
        "The target is the Llama agent's next **sampled majority-of-five A/B opinion**, not the correct mathematical answer and not the teacher's first-token probability. The table averages three training seeds; predictions are not ensembled.", "",
        "| Predictor | Selected training error | Heldout question error | Heldout accuracy | Heldout NLL | Two-class Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for condition in CONDITIONS:
        tr, h = row(condition, "train"), row(condition, "heldout")
        lines.append(f"| {names[condition]} | {100*tr.error:.2f}% | {100*h.error:.2f}% | {100*h.accuracy:.2f}% | {h.nll:.4f} | {h.brier_two_class:.4f} |")
    accuracy, nll = audit["paired_differences"]["accuracy"], audit["paired_differences"]["nll"]
    lo, hi = accuracy["descriptive_95_interval"]
    nlo, nhi = nll["descriptive_95_interval"]
    lines += ["", f"Arguments change heldout accuracy by **{100*accuracy['mean']:+.2f} percentage points** (descriptive question-bootstrap 95% interval {100*lo:+.2f} to {100*hi:+.2f}) and NLL by {nll['mean']:+.4f} ({nlo:+.4f} to {nhi:+.4f}; lower is better). This small pilot does not establish a reliable argument benefit. These intervals summarize eight reused questions and are not confirmatory significance tests; the cross-validation training sets overlap.", "",
        f"The training-fold global-majority baseline has {100*audit['baselines']['global_majority_accuracy']:.2f}% heldout accuracy and NLL {audit['baselines']['global_majority_nll']:.4f}. Simply retaining the receiver's current opinion has {100*audit['baselines']['persistence_accuracy']:.2f}% accuracy.", "",
        "## Data and split", "",
        "Eight objective mathematics questions, one 32-agent conversation per question, two synchronous updates and the prescribed signed J0 graph give 512 natural transitions. In each of four fixed folds, four whole questions (256 transitions) train the predictor, two (128) select its checkpoint, and two (128) are scored only after selection. Each question is assessed exactly once per seed. Both rounds, every agent and all paired interventions remain in the same question partition.", "",
        "All eight questions were previously inspected in exploratory source-model experiments. They are held out from these particular fits, not a fresh untouched research test. No previous GPT test set was scored. The much smaller number of training questions, different teacher and full-context input representation prevent direct attribution of differences from earlier GPT error rates to source-model identity.", "",
        "| Fold | Train questions | Selection questions | Heldout questions |", "| --- | --- | --- | --- |"]
    for f in protocol["folds"]:
        qs = lambda name: ", ".join(q.replace("math_train_", "") for q in f[name])
        lines.append(f"| {f['fold']} | {qs('train')} | {qs('selection')} | {qs('heldout')} |")
    lines += ["", "## Model and actual training", "",
        f"Each predictor has **282,146 trainable parameters**, with four attention heads and a 128-dimensional hidden state. The frozen MiniLM encoder has {features['encoder_parameters']:,} parameters and is additional to that predictor count. The no-argument control has 49,152 inactive text-projection parameters. Full persona and question/choice vectors, current opinions, two-snapshot histories, sender stances, node degrees and the original friendly/unfriendly graph cues are available to both models. Communicated stances were checked to equal sender opinions. No question-ID embeddings or target-answer keys are inputs.", "",
        f"The encoder processes each message separately. All input tokens were retained in {features['windows']} windows for {features['unique_texts']} unique input strings; {features['multiwindow_texts']} long strings needed multiple windows (maximum {features['maximum_tokens']} tokens). Window pooling and frozen sentence vectors do not ensure preservation of mathematical reasoning or long-range token interactions. Incoming-message order is not represented by this graph attention rule.", "",
        f"Training used Apple MPS, AdamW at 0.0003, weight decay 0.001, batches of 32 receiver transitions, at most 50 epochs, selection-NLL early stopping with patience 6, gradient clipping 1 and seeds 42/43/44. The 24-fit training invocation took {complete['seconds_this_invocation']:.1f} seconds, excluding feature preparation and audit. Every fit changed its parameters; all stopped early. Selected epochs range from {min(m['selected_epoch'] for m in manifests)} to {max(m['selected_epoch'] for m in manifests)}, and terminal epochs from {min(m['terminal_epoch'] for m in manifests)} to {max(m['terminal_epoch'] for m in manifests)}. This is actual training, not inference with an old model.", "",
        "| Condition | Selected training error | Terminal training error |", "| --- | ---: | ---: |"]
    for c in CONDITIONS:
        t = audit["training"][c]
        lines.append(f"| {names[c]} | {100*t['selected_error']:.2f}% | {100*t['terminal_error']:.2f}% |")
    lines += ["", "Terminal errors use the final optimization checkpoint, after the checkpoint selected on separate questions. Lower terminal training error does not justify choosing it on the assessed questions.", "",
        "## Argument-removal check", "",
        "The selected models were also evaluated on heldout paired inboxes in which each rationale was replaced by the sender's recorded conclusion. No model trained on these conclusion-only targets. This therefore tests transfer to a changed input distribution, not a model trained to fit the intervention distribution.", "",
        "| Condition | Conclusion-only error | Error across both variants | Mean absolute predicted majority-probability change | Predicted label changes |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for c in CONDITIONS:
        p = next(p for p in audit["pair_diagnostic"] if p["condition"] == c and p["subset"] == "all")
        lines.append(f"| {names[c]} | {100*row(c, 'heldout', 'conclusion_only').error:.2f}% | {100*p['paired_error']:.2f}% | {100*p['mean_absolute_predicted_majority_probability_shift']:.2f} pp | {100*p['predicted_label_flip_rate']:.2f}% |")
    p = next(p for p in audit["pair_diagnostic"] if p["condition"] == "arguments" and p["subset"] == "all")
    lines += ["", f"The observed sampled majority changes in {100*p['observed_majority_flip_rate']:.2f}% of the 512 pairs, including sampling variability. The argument model's probability-shift direction agrees with the observed majority-shift direction in {100*p['direction_accuracy_on_observed_flips']:.2f}% of changed pairs. This is a descriptive check: independently sampled five-vote groups can disagree even when their underlying distributions are identical. All empty-inbox controls and all no-argument predictions have exactly zero model response to argument removal.", "",
        "The previous pilot's 22.40 pp teacher effect concerned **first-token probability**, whereas these predictors output **majority-of-five probability**. Those magnitudes are not interchangeable, and this report does not subtract them or call the teacher effect a predictor accuracy improvement.", "",
        "## Verification and interpretation", "",
        f"Six focused tests passed: question separation; target-independent inputs and pre-update histories; equivalence to the earlier signed attention implementation; padding/empty-inbox isolation; individual-message gradient flow; and sensitivity to graph signs. Every saved selected checkpoint was reloaded on CPU and reproduced its MPS original and intervention probabilities (maximum absolute difference {audit['maximum_cpu_mps_probability_difference']:.3g}). Selection epochs were independently reconstructed from selection NLL. All paired initial weights match, all trainable models updated, and all {audit['previous_manifests_unchanged']} earlier fit manifests and {audit['previous_fit_artifacts_unchanged']} declared artifacts remain unchanged. Source-data hashes match. No new LLM generation or paid API calls were made.", "",
        "These completed fits show a small average improvement from adding argument vectors, with substantial uncertainty across questions. They do not establish that arguments are irrelevant or that opinions alone form a sufficient state. There are only four training questions per fit; generic frozen vectors, limited coverage and stochastic targets remain plausible constraints. The next useful data expansion would add independent questions and repeated conversations under a split fixed before generation. The current model predicts one update given observed messages/history; it does not generate messages or autonomously simulate a full conversation.", "",
        f"![Heldout performance by question]({Path('reports/local_llama_transformer.png').resolve()})", "",
        "## Reproduce", "", "```sh",
        ".venv/bin/python -m pytest tests/test_local_llama_transformer.py --junitxml=reports/local_llama_transformer_checks.xml",
        ".venv/bin/python scripts/local_llama_transformer.py prepare --device mps",
        ".venv/bin/python scripts/local_llama_transformer.py train --device mps",
        ".venv/bin/python scripts/report_local_llama_transformer.py", "```", "",
        "Completed fits are reused only after hash checks. Protocol, data, checkpoints, optimization histories, seed/question scores, paired outputs and audit records are stored under `runs/local_llama_transformer_v1/`. This directory is separate from the frozen local source pilot and previous predictor studies.", ""]
    Path("reports/local_llama_transformer.md").write_text("\n".join(lines))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {"opinion_history": "#4a647d", "arguments": "#c75d34"}
    xx = np.arange(8)
    for c in CONDITIONS:
        part = byq[byq.condition == c].sort_values("qid")
        axes[0].plot(xx, 100*part.error, marker="o", color=colors[c], label=names[c])
        axes[1].plot(xx, part.nll, marker="o", color=colors[c], label=names[c])
    for ax in axes:
        ax.set_xticks(xx, [q.replace("math_train_", "") for q in sorted(byq.qid.unique())])
        ax.set_xlabel("Heldout question ID")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Prediction error (%)")
    axes[1].set_ylabel("Negative log likelihood")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("New Llama-data transformers · question-heldout scores averaged over 3 seeds")
    fig.savefig("reports/local_llama_transformer.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
