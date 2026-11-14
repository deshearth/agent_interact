"""Audit and report the isolated argument-fusion/transfer diagnostic."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from mac_agents.metrics import paired_bootstrap, score
from scripts.argument_features import load_cache
from scripts.argument_mechanisms import (CONFIG, ROOT, ROLES, audit_donors, cohort,
                                        implementation, split_index)
from scripts import probe_argument_centroids as centroid_probe

PREFIX = Path("reports/argument_mechanisms")
KEYS = ["episode_id", "round", "agent"]
METRICS = ("nll", "accuracy", "brier", "fcba", "flip_accuracy", "stay_accuracy")
LABELS = {("A2", "stance_only"): "Stance/history",
          ("A2", "full_arguments"): "Additive: full text",
          ("A2", "shuffled_arguments"): "Additive: shuffled text",
          ("separate", "full_arguments"): "Separate: full text",
          ("separate", "shuffled_arguments"): "Separate: shuffled text"}


def interval(row, metric):
    scale, places = (100, 2) if metric == "accuracy" else (1, 5)
    return f"{scale*row['difference']:+.{places}f} [{scale*row['ci95'][0]:+.{places}f}, {scale*row['ci95'][1]:+.{places}f}]"


def main():
    cfg = config(CONFIG)
    arrays, index, _, _ = load_cache(cfg, verify=True)
    index = split_index(index, cfg)
    protocol = json.loads((ROOT/"protocol.json").read_text())
    assert protocol["implementation"] == implementation()
    assert protocol["config_sha256"] == digest(CONFIG)
    for name, sha in protocol["prepared_sha256"].items():
        assert digest(ROOT/name) == sha
    for seed in cfg["training"]["seeds"]:
        got = audit_donors(arrays, index, np.load(ROOT/f"donors_s{seed}.npy", allow_pickle=False))
        assert got == protocol["shuffle_audit"][str(seed)]
    base = json.loads(Path("runs/argument_response_v1/protocol.json").read_text())
    for name, sha in base["implementation"].items():
        assert implementation()[name] == sha
    checks = ET.parse("reports/argument_mechanism_checks.xml").getroot()
    assert not any(checks.findall(f".//{tag}") for tag in ("failure", "error", "skipped"))
    centroid_checks = ET.parse("reports/argument_centroid_checks.xml").getroot()
    assert not any(centroid_checks.findall(f".//{tag}") for tag in ("failure", "error", "skipped"))
    test_count = len(checks.findall(".//testcase"))+len(centroid_checks.findall(".//testcase"))
    plan = json.loads((ROOT/"plan.json").read_text())
    assert plan["complete"] and plan["protocol_sha256"] == digest(ROOT/"protocol.json")
    expected = {(r["kind"], r["condition"], r["seed"]) for r in cohort(cfg)}
    assert {(r["kind"], r["condition"], r["seed"]) for r in plan["runs"]} == expected
    ep_map = {r["episode_id"]: e for e, r in enumerate(index)}
    references, rows, question_rows, round_rows, sensitivities, hashes = {}, [], [], [], [], {}
    for item in cohort(cfg):
        name = f"{item['kind']}-{item['condition']}-s{item['seed']}"
        run, out = ROOT/name, ROOT/"evaluation"/name
        manifest = json.loads((run/"manifest.json").read_text())
        assert manifest["status"] == "complete" and not manifest["smoke"]
        assert manifest["protocol_sha256"] == digest(ROOT/"protocol.json")
        assert manifest["train_snapshots"] == 2048 and manifest["validation_snapshots"] == 1024
        assert manifest["train_questions"] == protocol["roles"]["train"]["questions"]
        assert manifest["validation_questions"] == protocol["roles"]["validation"]["questions"]
        for file, sha in manifest["artifact_sha256"].items():
            assert digest(run/file) == sha
        evaluation = json.loads((out/"identity.json").read_text())
        assert evaluation["run_manifest_sha256"] == digest(run/"manifest.json")
        assert evaluation["best_sha256"] == digest(run/"best.pt")
        assert evaluation["protocol_sha256"] == digest(ROOT/"protocol.json")
        for file, sha in evaluation["artifacts"].items():
            assert digest(out/file) == sha
        hashes[name] = {"manifest_sha256": digest(run/"manifest.json"), "evaluation_sha256": digest(out/"identity.json")}
        for role in ROLES:
            location = run if role == "validation" else out
            frame = pd.read_parquet(location/f"{role}_predictions.parquet").sort_values(KEYS).reset_index(drop=True)
            assert not frame.duplicated(KEYS).any()
            truth = frame[KEYS+["qid", "current_state", "target"]]
            if role not in references:
                references[role] = truth
                es = frame.episode_id.map(ep_map).to_numpy()
                ts, agents = frame["round"].to_numpy(), frame.agent.to_numpy()
                assert {index[e]["diagnostic_role"] for e in es} == {role}
                np.testing.assert_array_equal(frame.current_state, arrays["spins"][es, ts, agents])
                np.testing.assert_array_equal(frame.target, arrays["spins"][es, ts+1, agents])
                assert set(frame.episode_id) == set(protocol["roles"][role]["episode_ids"])
            else:
                pd.testing.assert_frame_equal(references[role], truth)
            metrics, questions = score(frame)
            saved = json.loads((location/f"{role}_metrics.json").read_text())
            for metric in METRICS:
                np.testing.assert_allclose(metrics[metric], saved[metric], rtol=0, atol=1e-12)
            rows.append({**item, "role": role, **metrics, "error": 1-metrics["accuracy"],
                         "parameters": manifest["parameters"], "steps": manifest["steps"],
                         "best_step": manifest["best_step"], "reason": manifest["stopping_reason"]})
            question_rows.extend(questions.assign(**item, role=role).to_dict("records"))
            for round_id, subset in frame.groupby("round", sort=True):
                _, qs = score(subset)
                round_rows.extend(qs.assign(**item, role=role, round=round_id).to_dict("records"))
            if item["condition"] == "full_arguments" and role in ("familiar", "unseen"):
                swapped = pd.read_parquet(out/f"{role}_substitution_predictions.parquet").sort_values(KEYS).reset_index(drop=True)
                pd.testing.assert_frame_equal(truth, swapped[truth.columns])
                for qid, group in frame.groupby("qid", sort=True):
                    other = swapped.loc[group.index]
                    a, _ = score(group)
                    b, _ = score(other)
                    sensitivities.append({**item, "role": role, "qid": qid,
                        "nll_change": b["nll"]-a["nll"], "accuracy_change": b["accuracy"]-a["accuracy"],
                        "mean_absolute_probability_change": float((other.p_positive-group.p_positive).abs().mean()),
                        "changed_prediction_fraction": float((other.prediction != group.prediction).mean()),
                        "different_text_update_fraction": float(group.has_different_text.mean())})
    runs, questions = pd.DataFrame(rows), pd.DataFrame(question_rows)
    runs.to_csv(f"{PREFIX}_runs.csv", index=False)
    questions.to_csv(f"{PREFIX}_by_question.csv", index=False)
    pd.DataFrame(round_rows).to_csv(f"{PREFIX}_by_question_round.csv", index=False)
    sens = pd.DataFrame(sensitivities)
    sens.to_csv(f"{PREFIX}_sensitivity.csv", index=False)
    means = runs.groupby(["role", "kind", "condition"], sort=True)[list(METRICS)+["error"]].mean().reset_index()
    means.to_csv(f"{PREFIX}_summary.csv", index=False)
    qmeans = questions.groupby(["role", "kind", "condition", "qid"])[list(METRICS)].mean()
    paired = {}
    for role in ROLES:
        for kind in ("A2", "separate"):
            for control in ("stance_only", "shuffled_arguments"):
                control_kind = "A2" if control == "stance_only" else kind
                a = qmeans.loc[(role, kind, "full_arguments")]
                b = qmeans.loc[(role, control_kind, control)]
                label = f"{role}/{kind}: full minus {control}"
                paired[label] = {metric: paired_bootstrap(a[metric], b[metric],
                    cfg["evaluation"]["bootstrap_replicates"], cfg["evaluation"]["bootstrap_seed"])
                    for metric in ("nll", "accuracy")}
        for condition in ("full_arguments", "shuffled_arguments"):
            a = qmeans.loc[(role, "separate", condition)]
            b = qmeans.loc[(role, "A2", condition)]
            paired[f"{role}/{condition}: separate minus additive"] = {
                metric: paired_bootstrap(a[metric], b[metric], cfg["evaluation"]["bootstrap_replicates"],
                    cfg["evaluation"]["bootstrap_seed"]) for metric in ("nll", "accuracy")}
    json_write(f"{PREFIX}_paired.json", paired)
    geom = json.loads((ROOT/"geometry.json").read_text())
    pd.DataFrame(geom["rows"]).to_csv(f"{PREFIX}_geometry.csv", index=False)

    cp = json.loads((centroid_probe.OUT/"protocol.json").read_text())
    ci = json.loads((centroid_probe.OUT/"integrity.json").read_text())
    assert cp["code_sha256"] == digest(centroid_probe.__file__)
    assert cp["source_protocol_sha256"] == digest(ROOT/"protocol.json")
    assert ci["protocol_sha256"] == digest(centroid_probe.OUT/"protocol.json")
    assert digest(centroid_probe.OUT/"centroids.npz") == cp["centroids_sha256"]
    centroid_rows, centroid_qrows = [], []
    for key, files in ci["outputs"].items():
        name, variant = key.split("/")
        kind, _, seedstr = name.split("-")
        for suffix, sha in files.items():
            assert digest(centroid_probe.OUT/name/f"{variant}_{suffix}") == sha
        assert digest(ROOT/name/"best.pt") == cp["references"][name]["best_sha256"]
        first = pd.read_parquet(ROOT/"evaluation"/name/"familiar_predictions.parquet").sort_values(KEYS).reset_index(drop=True)
        second = pd.read_parquet(centroid_probe.OUT/name/f"{variant}_predictions.parquet").sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(first[references["familiar"].columns], second[references["familiar"].columns])
        a, aq = score(first)
        b, bq = score(second)
        saved = json.loads((centroid_probe.OUT/name/f"{variant}_metrics.json").read_text())
        for metric in METRICS:
            np.testing.assert_allclose(b[metric], saved[metric], rtol=0, atol=1e-12)
        centroid_rows.append({"kind": kind, "seed": int(seedstr[1:]), "variant": variant,
            "error": 1-b["accuracy"], "nll": b["nll"], "accuracy_change": b["accuracy"]-a["accuracy"],
            "nll_change": b["nll"]-a["nll"],
            "choice_change_fraction": float((first.prediction != second.prediction).mean()),
            "absolute_probability_change": float((first.p_positive-second.p_positive).abs().mean())})
        for qid, group in second.groupby("qid", sort=True):
            baseline = first.loc[group.index]
            before, _ = score(baseline)
            after, _ = score(group)
            centroid_qrows.append({"kind": kind, "seed": int(seedstr[1:]), "variant": variant, "qid": qid,
                "accuracy_change": after["accuracy"]-before["accuracy"], "nll_change": after["nll"]-before["nll"]})
    centroids = pd.DataFrame(centroid_rows)
    centroids.to_csv(f"{PREFIX}_centroid_probe.csv", index=False)
    cq = pd.DataFrame(centroid_qrows)
    cq.to_csv(f"{PREFIX}_centroid_probe_by_question.csv", index=False)
    centroid_intervals = {}
    for (kind, variant), group in cq.groupby(["kind", "variant"]):
        means_q = group.groupby("qid")[["accuracy_change", "nll_change"]].mean()
        centroid_intervals[f"{kind}/{variant}"] = {m: paired_bootstrap(means_q[m], means_q[m]*0,
            cfg["evaluation"]["bootstrap_replicates"], cfg["evaluation"]["bootstrap_seed"])
            for m in ("accuracy_change", "nll_change")}
    json_write(f"{PREFIX}_centroid_probe_paired.json", centroid_intervals)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = ["#424a54", "#337a9c", "#96bbcb", "#178278", "#88bfb6"]
    labels = ["Stance /\nhistory", "Additive\nfull", "Additive\nshuffled", "Separate\nfull", "Separate\nshuffled"]
    for col, role in enumerate(("familiar", "unseen")):
        for row, metric in enumerate(("error", "nll")):
            ax = axes[row, col]
            scale = 100 if metric == "error" else 1
            for x, ((kind, condition), color) in enumerate(zip(LABELS, colors)):
                vals = runs[(runs.role == role)&(runs.kind == kind)&(runs.condition == condition)][metric].to_numpy()*scale
                ax.scatter(x+np.linspace(-.12, .12, len(vals)), vals, c=color, s=28, alpha=.75)
                ax.plot([x-.21, x+.21], [vals.mean()]*2, color=color, linewidth=3)
            baseline = runs[(runs.role == role)&(runs.condition == "stance_only")][metric].mean()*scale
            ax.axhline(baseline, color=colors[0], linestyle="--", linewidth=1, alpha=.5)
            ax.set_xticks(range(5), labels)
            ax.set_ylabel("Error (%)" if metric == "error" else "Negative log-likelihood")
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title("Held-out conversations, 16 familiar questions" if role == "familiar" else "4 unseen development questions")
    fig.suptitle("Argument fusion and transfer across questions", fontsize=16, y=.98)
    fig.text(.5, .925, "Dots: three initialization seeds. Bars: means. Lower is better. Neither cohort selects checkpoints.", ha="center", fontsize=10)
    fig.text(.5, .015, "Exploratory reused development data; fit replicas 0/1, select on replica 2, familiar evaluation on replica 3.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .91])
    fig.savefig(f"{PREFIX}.png", dpi=180)
    fig.savefig(f"{PREFIX}.pdf")
    plt.close(fig)

    lines = ["# Why does adding argument text fail to improve prediction?", "",
        "This follow-up tests two possible limitations: transfer from familiar to unseen questions, and how frozen text vectors are combined with opinion features. It is exploratory development research, not fresh confirmation.", "",
        "The strongest clue is reliance on shared question content with limited transfer. Original additive text reduces error on held-out conversations about familiar questions from 6.47% to 5.93%, but raises error on the four unseen questions from 9.84% to 10.84%. Replacing every familiar-question message with a fitting-only question/stance average changes error only from 5.93% to 5.99%. Most of the familiar-question benefit therefore survives removal of individual-message variation. Common reasons may remain in those averages, so this does not mean the model uses only superficial topic labels.", "",
        "The alternative normalized text branch fits training data more closely but performs worse than additive fusion on both evaluation cohorts. This tested architectural change does not rescue the argument benefit. Full text nevertheless beats the matched-shuffled control slightly on unseen questions (+0.33 accuracy points for additive fusion), so the evidence is not an absence of all content dependence. The remaining question is whether useful reasoning differences are missing from the frozen representation, scarce in these conversations, or difficult to learn across questions. These causes are not identified separately by this study.", "",
        "## Design", "",
        "All 16 fitting questions remain represented. For every question and graph, replicas 0/1 fit the model (256 conversations; 65,536 agent transitions), replica 2 selects stopping time (128; 32,768), and replica 3 evaluates new conversations about familiar questions (128; 32,768). All 128 conversations on the four unseen development questions form a separate evaluation cohort. Entire conversations are disjoint across roles; neither evaluation cohort enters fitting or checkpoint selection in this study.", "",
        "The familiar and unseen cohorts contain different questions and are not paired. The original question PCA still fits exactly the same 16 fitting questions. These runs use half the original fitting conversations and a different selection cohort, so direct comparisons with earlier full-data results do not isolate architecture.", "",
        "Original additive fusion sums text, sender features and stance before a shared normalization. The alternative gives text its own normalization and key/value maps, combining the two branches in attention. It changes fusion and normalization together. It adds 24,576 parameters (210,466 total versus 185,890). With zero text, shared weights, outputs and optimizer updates agree exactly on CPU and GPU, allowing a shared stance-only control.", "",
        "Fifteen fits cover three initializations, the shared stance control and full/matched-shuffled text under both fusion methods. Donor substitutions are rebuilt separately inside each of the four roles; no fitting input borrows held-out text. Other matching fields and their limitations are unchanged. The original encoder, experiment files and checkpoints remain untouched.", "",
        "Splitting by replica reduces available swap partners: about 53.3% of fitting messages, 14.5% of selection/familiar-evaluation messages and 77.3% of unseen-question messages actually change. Full-versus-shuffled comparisons are consequently a weak test of message specificity on familiar questions. The all-message centroid probe below addresses this coverage limitation, with a different input-distribution limitation of its own.", "",
        "## Results", "", "Values average three seeds; errors are 100% minus accuracy. NLL is the stopping metric.", "",
        "| Inputs | Training error | Familiar-question error | Unseen-question error | Familiar NLL | Unseen NLL |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    mi = means.set_index(["role", "kind", "condition"])
    for (kind, condition), label in LABELS.items():
        train, fam, unseen = [mi.loc[(role, kind, condition)] for role in ("train", "familiar", "unseen")]
        lines.append(f"| {label} | {100*train.error:.2f}% | {100*fam.error:.2f}% | {100*unseen.error:.2f}% | {fam.nll:.5f} | {unseen.nll:.5f} |")
    lines += ["", "![Fusion and transfer comparison](argument_mechanisms.png)", "",
        "| Cohort / comparison | Accuracy difference, percentage points [95% interval] | NLL difference [95% interval] |",
        "| --- | ---: | ---: |"]
    for label, comparison in paired.items():
        if label.startswith(("familiar/", "unseen/")):
            lines.append(f"| {label} | {interval(comparison['accuracy'], 'accuracy')} | {interval(comparison['nll'], 'nll')} |")
    lines += ["", "Intervals average seeds within question and use 1,000 paired question bootstrap draws. They are descriptive, pointwise intervals on 16 familiar or four unseen questions, after prior development work. They do not adjust for all comparisons. The fixed replica split does not measure split-to-split variability.", "",
        "## What is represented in the frozen text vectors?", "",
        "Centroids below are fitted using input messages from fitting replicas only; no response targets are used. Reconstruction is evaluated on distinct familiar-question conversations. Variance uses message-occurrence weights, including repeated strings.", "",
        "| Cohort | Variance explained by question centroid | By question + stated stance | Question identified by nearest centroid | Exact text previously seen in fit |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in geom["rows"]:
        if row["role"] == "unseen":
            continue
        lines.append(f"| {row['role']} | {100*row['question_centroid_explained_fraction']:.2f}% | {100*row['question_stance_centroid_explained_fraction']:.2f}% | {100*row['nearest_question_centroid_accuracy']:.2f}% | {100*row['exact_text_seen_in_fit_fraction']:.2f}% |")
    lines += ["", "Most Euclidean variation is associated with which problem is discussed. This quantifies a strong question signal; it is not a measure of mutual information, proof that reasoning is absent, or proof that a downstream network cannot use small residual directions. Some messages share exact wording across conversations, as shown above; none of the unseen-question messages exactly matches fitting text. Merely holding out conversations does not remove question, persona or graph familiarity.", "",
        "## Sensitivity to matched text substitutions", "",
        "The full-text predictor is also evaluated after matched substitutions without retraining. Differences concern this predictor, not a causal change in the original LLM's response. Substitutions may exchange paraphrases and leave singleton strata unchanged.", "",
        "| Fusion | Cohort | Choice changes | Mean absolute probability change, points | NLL change |",
        "| --- | --- | ---: | ---: | ---: |"]
    for (kind, role), group in sens.groupby(["kind", "role"], sort=True):
        lines.append(f"| {kind} | {role} | {100*group.changed_prediction_fraction.mean():.2f}% | {100*group.mean_absolute_probability_change.mean():.3f} | {group.nll_change.mean():+.5f} |")
    lines += ["", "## Removing individual-message variation", "",
        "A separate read-only probe replaces every real message vector with a fitting-only question average or question/stance average, normalized to unit length like the original embeddings. Padding, graph, states, history and all model weights are unchanged. It evaluates the six selected full-text models on familiar-question replica 3; no unseen-question centroids are fitted. No response targets enter the averages.", "",
        "| Fusion | Replacement | Familiar error | Accuracy change, points [95% interval] | NLL change [95% interval] | Choices changed |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for (kind, variant), group in centroids.groupby(["kind", "variant"]):
        contrasts = centroid_intervals[f"{kind}/{variant}"]
        lines.append(f"| {kind} | {variant} | {100*group.error.mean():.2f}% | {interval(contrasts['accuracy_change'], 'accuracy')} | {interval(contrasts['nll_change'], 'nll')} | {100*group.choice_change_fraction.mean():.2f}% |")
    lines += ["", "These replacements remove variation between individual message vectors, while retaining average problem content (and, for the second variant, average stance-specific content). They do not remove every trace of reasoning: common reasons can be present in the averages. They are inference-time distribution shifts, not independently trained controls. Small effects describe the fitted predictor's dependence on those distinctions; they cannot establish that the original LLM is insensitive to arguments. Protocol and unchanged-checkpoint audit: `runs/argument_centroid_probe_v1/`."]
    lines += ["", "## Scope and artifacts", "",
        "Neither an absent gain nor a benefit from the separate branch isolates logical argument quality. Full text contains topic, stance, numbers, wording and reasoning; the shuffled control retains much of that content while disrupting its pairing with a response. The current generic frozen encoder, three-dimensional question PCA, content redundancy and untested source-LLM sensitivity remain distinct possible limitations. The next focused representation check should retain the full problem and choices and compare same-stance arguments with changed numerical or logical claims against paraphrase controls. A causal claim requires controlled argument changes evaluated with the source LLM and new held-out questions.", "",
        f"Completed {len(cohort(cfg))} fits at `{plan['completed_utc']}`; the driver recorded {plan['elapsed_seconds']:.1f} seconds, including evaluation. All {test_count} focused checks passed. Protocol: `runs/argument_mechanisms_v1/protocol.json`. Run scores, per-question/round scores, paired contrasts, sensitivity, geometry and integrity audits are saved beside this report. The PNG and PDF are standalone figures. No new LLM calls or test scoring were performed.", ""]
    Path(f"{PREFIX}.md").write_text("\n".join(lines))
    json_write(f"{PREFIX}_integrity.json", {"protocol_sha256": digest(ROOT/"protocol.json"),
        "complete_runs": len(cohort(cfg)), "tests_passed": test_count,
        "source_implementation_unchanged": True, "donors_confined_to_roles": True,
        "identical_examples_and_targets_across_conditions": True,
        "evaluation_roles_excluded_from_fitting_and_selection": True,
        "saved_metrics_recomputed": True,
        "centroid_probe_integrity_sha256": digest(centroid_probe.OUT/"integrity.json"), "hashes": hashes})
    print(means.to_string(index=False))


if __name__ == "__main__":
    main()
