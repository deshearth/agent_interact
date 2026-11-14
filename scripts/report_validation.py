"""Audit and report the completed validation screen without accessing test outcomes."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mac_agents.common import code_digest, digest, json_write
from mac_agents.metrics import paired_bootstrap, score


ROOT = Path.cwd()
REPORTS = ROOT / "reports"
METRICS = ["nll", "brier", "accuracy", "fcba", "flip_accuracy", "stay_accuracy"]
LABELS = {"opinion": "Current opinions", "vote_counts": "+ Vote counts",
          "history": "+ Previous opinions", "existing_messages": "+ Existing messages"}


def main():
    plan = json.loads(Path("runs/validation_repeats_plan.json").read_text())
    if "completed_utc" not in plan or any(x["status"] != "complete" for x in plan["runs"]):
        raise ValueError("Complete validation runs and rollouts before reporting")
    rows, question_rows, round_rows, rollout_rows, rollout_question_rows = [], [], [], [], []
    episode_questions = {x["episode_id"]: x["qid"] for x in json.loads(Path("data/processed/author_pca/index.json").read_text())}
    reference = None
    fingerprints = set()
    expected_code = code_digest()
    for item in plan["runs"]:
        run = Path(item["run"])
        meta = json.loads((run / "manifest.json").read_text())
        if meta["smoke"] or meta["status"] != "complete" or meta["code_sha256"] != expected_code:
            raise ValueError(f"Ineligible/mismatched run: {run}")
        if (meta["train_snapshots"], meta["validation_snapshots"]) != (4096, 1024):
            raise ValueError("Unexpected comparison sample counts")
        fingerprints.add(tuple(meta["data_fingerprint"][k] for k in (
            "split_sha256", "episodes_sha256", "index_sha256", "train_pairs_sha256", "validation_pairs_sha256")))
        if meta["kind"].startswith("A") and digest(run / "best.pt") != meta["best_sha256"]:
            raise ValueError("Checkpoint changed")
        metrics = json.loads((run / "validation_metrics.json").read_text())
        identity = {"model": meta["kind"], "state": meta["state"], "seed": meta["seed"], "run": str(run)}
        rows.append(dict(identity, **{k: metrics[k] for k in METRICS}, parameters=meta["parameters"],
            seconds=meta["elapsed_seconds_this_session"], epochs=meta.get("epochs"),
            best_step=meta.get("best_step"), stopping_reason=meta["stopping_reason"],
            manifest_sha256=digest(run / "manifest.json")))
        q = pd.read_csv(run / "validation_by_question.csv")
        question_rows.append(q.assign(**identity))
        predictions = pd.read_parquet(run / "validation_predictions.parquet")
        observed = predictions.set_index(["episode_id", "round", "agent"])[["qid", "current_state", "target"]].sort_index()
        if reference is None:
            reference = observed
        else:
            pd.testing.assert_frame_equal(observed, reference)
        rescored, _ = score(predictions, meta["kind"])
        for metric in METRICS:
            if metrics[metric] is not None:
                np.testing.assert_allclose(metrics[metric], rescored[metric], atol=1e-12, rtol=0)
        for t, group in predictions.groupby("round"):
            _, qt = score(group, meta["kind"])
            round_rows.append(qt.assign(**identity, round=t))
        if meta["autonomous"]:
            p = run / "validation_closed_loop_metrics.json"
            ev = json.loads(p.read_text())
            if ev["preparation_run"] or ev["rollouts_per_initial_state"] != 32 or ev["episodes"] != 128:
                raise ValueError("Unexpected rollout protocol")
            true_hist = np.array(ev["observed_consensus_time_histogram"])
            sim_hist = np.array(ev["sampled_consensus_time_histogram"])
            rollout_rows.append(dict(identity,
                final_accuracy=ev["horizon_accuracy"][-1],
                mean_horizon_accuracy=np.mean(ev["horizon_accuracy"]),
                final_wasserstein=ev["question_mean_final_fraction_wasserstein"],
                observed_switches=ev["observed_mean_switches_per_agent"],
                sampled_switches=ev["sampled_mean_switches_per_agent"],
                observed_consensus_fraction=1-true_hist[-1]/true_hist.sum(),
                sampled_consensus_fraction=1-sim_hist[-1]/sim_hist.sum(),
                rollout_seconds=ev["elapsed_seconds"],
                **{f"accuracy_h{h+1}": x for h, x in enumerate(ev["horizon_accuracy"])}))
            conditions = pd.read_csv(run / "validation_closed_loop_conditions.csv")
            qw = conditions.groupby("qid").final_fraction_wasserstein.mean()
            with np.load(run / "validation_closed_loop_trajectories.npz", allow_pickle=False) as trajectories:
                qids = np.array([episode_questions[e] for e in trajectories["episode_ids"]])
                for qid in np.unique(qids):
                    mask = qids == qid
                    accuracy = (trajectories["deterministic"][mask, -1] == trajectories["target"][mask, -1]).mean()
                    rollout_question_rows.append(dict(identity, qid=qid, final_accuracy=float(accuracy),
                                                       final_wasserstein=float(qw.loc[qid])))
    if len(fingerprints) != 1:
        raise ValueError("Data or splits differ across comparisons")
    frame = pd.DataFrame(rows)
    questions = pd.concat(question_rows, ignore_index=True)
    rounds = pd.concat(round_rows, ignore_index=True)
    rollouts = pd.DataFrame(rollout_rows)
    rollout_questions = pd.DataFrame(rollout_question_rows)
    for name, data in [("runs", frame), ("by_question", questions), ("by_question_round", rounds), ("rollouts", rollouts), ("rollouts_by_question", rollout_questions)]:
        data.to_csv(REPORTS / f"validation_{name}.csv", index=False)
    aggregate = frame.groupby(["model", "state"], sort=False)[METRICS + ["seconds", "parameters"]].agg(["mean", "std", "count"])
    aggregate.to_csv(REPORTS / "validation_summary.csv")
    comparisons = [(("A2", "opinion"), ("I0", "opinion")),
                   (("A0", "opinion"), ("I0", "opinion")),
                   (("A2", "opinion"), ("A0", "opinion")),
                   (("A2", "opinion"), ("A1", "opinion"))]
    comparisons += [((model, state), (model, "opinion")) for model in ("A0", "A2")
                    for state in ("vote_counts", "history", "existing_messages")]
    comparisons += [(("A2", "history"), ("A0", "history"))]
    paired = []
    for first, second in comparisons:
        a = questions[(questions.model == first[0]) & (questions.state == first[1])]
        b = questions[(questions.model == second[0]) & (questions.state == second[1])]
        result = {"first": list(first), "reference": list(second), "metrics": {}}
        for metric in METRICS:
            result["metrics"][metric] = paired_bootstrap(a.groupby("qid")[metric].mean(), b.groupby("qid")[metric].mean())
        result["per_question_nll_difference"] = (a.groupby("qid").nll.mean()-b.groupby("qid").nll.mean()).to_dict()
        if second[0].startswith("A"):
            result["per_seed_nll_difference"] = (a.groupby("seed").nll.mean()-b.groupby("seed").nll.mean()).to_dict()
        ra = rollout_questions[(rollout_questions.model == first[0]) & (rollout_questions.state == first[1])]
        rb = rollout_questions[(rollout_questions.model == second[0]) & (rollout_questions.state == second[1])]
        if len(ra) and len(rb):
            result["rollout_metrics"] = {metric: paired_bootstrap(ra.groupby("qid")[metric].mean(), rb.groupby("qid")[metric].mean())
                                         for metric in ("final_accuracy", "final_wasserstein")}
        paired.append(result)
    json_write(REPORTS / "validation_paired_comparisons.json", {
        "scope": "Descriptive validation comparisons; four questions; also used for early stopping",
        "method": "Average each question's metric over seeds, then resample paired questions; 1000 replicates",
        "comparisons": paired})
    test_artifacts = [str(p) for item in plan["runs"] for p in Path(item["run"]).glob("test_*")]
    test_ids = {x["episode_id"] for x in json.loads(Path("data/processed/author_pca/index.json").read_text()) if x["question_split"] == "test"}
    encoded_test_ids = sorted(test_ids & {p.stem for p in Path("data/processed/message_features").glob("*.npz")})
    if test_artifacts or encoded_test_ids:
        raise ValueError("Unexpected controlled test evaluation/encoding")
    json_write(REPORTS / "validation_integrity.json", {
        "scope": "Author-context validation candidates; the training-only PCA test study is separate",
        "code_sha256": expected_code, "runs": len(frame), "neural_seeds": [42, 43, 44],
        "train_questions": 16, "validation_questions": 4, "validation_episodes": 128,
        "validation_graphs": 8, "validation_agent_transitions": len(reference),
        "all_prediction_keys_and_targets_identical": True, "all_saved_scores_recomputed": True,
        "common_data_fingerprint": True, "checkpoint_hashes_verified": True,
        "all_neural_runs_stopped_by": sorted(frame[frame.model.str.startswith("A")].stopping_reason.unique().tolist()),
        "autonomous_rollout_runs": len(rollouts), "controlled_test_artifacts": test_artifacts,
        "encoded_test_episodes": encoded_test_ids, "training_seconds_total": float(frame.seconds.sum()),
        "rollout_seconds_total": float(rollouts.rollout_seconds.sum()),
        "reference_observed_mean_switches": float(rollouts.observed_switches.iloc[0]),
        "reference_observed_consensus_fraction": float(rollouts.observed_consensus_fraction.iloc[0])})
    make_plots(frame, rollouts)
    write_report(frame, aggregate, paired, rollouts)
    write_findings(frame, questions, rounds, rollouts, reference)
    print(aggregate.loc[:, [("nll", "mean"), ("nll", "std"), ("accuracy", "mean")]].to_string())
    print("Rollout summary:\n", rollouts.groupby(["model", "state"])[["final_accuracy", "final_wasserstein", "sampled_switches", "sampled_consensus_fraction"]].mean().to_string())
    print("Audited 30 runs, all matched validation examples, and 18 autonomous rollout evaluations")


def make_plots(frame, rollouts):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    states = list(LABELS)
    for ax, metric, title in zip(axes, ("nll", "accuracy"), ("Probability error (NLL; lower is better)", "Next-opinion accuracy (%)")):
        scale = 100 if metric == "accuracy" else 1
        for model, color, offset in (("A0", "#537b9d", -.11), ("A2", "#b55c35", .11)):
            data = frame[frame.model == model]
            means = data.groupby("state")[metric].mean().reindex(states)*scale
            std = data.groupby("state")[metric].std().reindex(states)*scale
            x = np.arange(len(states)) + offset
            ax.errorbar(x, means, yerr=std, fmt="o", capsize=4, color=color, label=model, zorder=3)
            for k, state in enumerate(states):
                points = data[data.state == state][metric].to_numpy()*scale
                ax.scatter(x[k]+np.linspace(-.035,.035,len(points)), points, s=15, color=color, alpha=.5)
        baseline = frame[frame.model == "I0"][metric].iloc[0]*scale
        ax.axhline(baseline, color="#444444", ls="--", lw=1, label="I0 baseline")
        ax.set_xticks(range(len(states)), ["Opinions", "+ Votes", "+ History", "+ Messages"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Validation on 4 held-out questions · dots = seeds · bars = seed standard deviation", fontsize=11)
    fig.savefig(REPORTS / "validation_state_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    choices = [("I0", "opinion", "I0 baseline", "#444444"),
               ("A0", "opinion", "A0 opinions", "#537b9d"),
               ("A1", "opinion", "A1 opinions", "#8667a4"),
               ("A2", "opinion", "A2 opinions", "#b55c35"),
               ("A0", "history", "A0 + history", "#449186"),
               ("A2", "history", "A2 + history", "#a7841f")]
    for model, state, label, color in choices:
        values = rollouts[(rollouts.model == model) & (rollouts.state == state)][[f"accuracy_h{h}" for h in range(1,9)]].to_numpy()*100
        mean = values.mean(0)
        ax.plot(range(1,9), mean, marker="o", ms=3, label=label, color=color)
        if len(values)>1:
            std = values.std(0, ddof=1)
            ax.fill_between(range(1,9), mean-std, mean+std, color=color, alpha=.1)
    ax.set(xlabel="Predicted round from observed initialization", ylabel="Deterministic next-opinion accuracy (%)",
           title="Autonomous validation rollout · generated history only")
    ax.set_xticks(range(1,9))
    ax.grid(alpha=.2)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(REPORTS / "validation_rollout_horizons.png", dpi=180)
    plt.close(fig)


def write_findings(frame, questions, rounds, rollouts, reference):
    means = frame.groupby(["model", "state"])[METRICS].mean()
    a2, i0 = means.loc[("A2", "opinion")], means.loc[("I0", "opinion")]
    aq = questions[(questions.model=="A2") & (questions.state=="opinion")].groupby("qid")[METRICS].mean()
    iq = questions[(questions.model=="I0") & (questions.state=="opinion")].groupby("qid")[METRICS].mean()
    accuracy_interval = paired_bootstrap(aq.accuracy, iq.accuracy)["ci95"]
    nll_interval = paired_bootstrap(aq.nll, iq.nll)["ci95"]
    findings = ["## What this screen shows", "",
        f"This screen does not establish a reliable accuracy advantage for the transformer. The observed mean is {100*a2.accuracy:.2f}% for opinion-only A2 versus {100*i0.accuracy:.2f}% for I0, a difference of {100*(a2.accuracy-i0.accuracy):+.2f} percentage points. Its descriptive paired question-bootstrap 95% interval is [{100*accuracy_interval[0]:+.2f}, {100*accuracy_interval[1]:+.2f}] percentage points, including zero. With only four validation questions also used for checkpoint selection, statistical significance has not been established.", "",
        f"The simpler A0 fixed-aggregation control reaches {100*means.loc[('A0','opinion'),'accuracy']:.2f}% accuracy, only {100*(a2.accuracy-means.loc[('A0','opinion'),'accuracy']):+.2f} percentage points below A2. The observed NLL change versus I0 is {i0.nll:.4f} → {a2.nll:.4f}, but its paired interval [{nll_interval[0]:+.5f}, {nll_interval[1]:+.5f}] also includes zero. A2's NLL advantage over A1 is unresolved across questions. The current accuracy results do not justify preferring contextual attention to the simpler control.", "",
        f"Adding history lowers A2 NLL to {means.loc[('A2','history'),'nll']:.4f}, but changes accuracy by only {100*(means.loc[('A2','history'),'accuracy']-a2.accuracy):+.2f} percentage points. Vote counts give a small average change; this pooled message representation worsens NLL. These conclusions concern the implemented features and training protocol.", ""]
    a2_roll = rollouts[(rollouts.model=="A2") & (rollouts.state=="opinion")]
    i0_roll = rollouts[rollouts.model=="I0"]
    findings += [f"Autonomous simulation also provides no demonstrated transformer advantage: observed round-eight accuracy is {100*a2_roll.final_accuracy.mean():.2f}% for A2/opinion versus {100*i0_roll.final_accuracy.mean():.2f}% for I0. The sampled final-opinion distribution is also farther from observations for A2 (Wasserstein {a2_roll.final_wasserstein.mean():.4f} versus {i0_roll.final_wasserstein.mean():.4f}). History does not improve A2's average round-eight accuracy or distribution distance. Four validation questions leave substantial uncertainty; these are development findings.", ""]
    control_path = Path("runs/history_availability_control/plan.json")
    if control_path.exists() and "completed_utc" in json.loads(control_path.read_text()):
        plan = json.loads(control_path.read_text())
        if plan["adapter_sha256"] != digest("scripts/history_availability_control.py") or plan["base_code_sha256"] != code_digest():
            raise ValueError("Control code changed")
        cr, cq, ct = [], [], []
        for item in plan["runs"]:
            run = Path(item["run"])
            meta = json.loads((run / "control_manifest.json").read_text())
            if meta["status"] != "complete" or meta["smoke"] or meta["adapter_sha256"] != plan["adapter_sha256"]:
                raise ValueError("Incomplete/unmatched control")
            if digest(run / "best.pt") != meta["best_sha256"]:
                raise ValueError("Control checkpoint changed")
            expected = frame[(frame.model==item["model"]) & (frame.state=="history") & (frame.seed==item["seed"])].iloc[0]
            base = json.loads((Path(expected.run) / "manifest.json").read_text())
            if meta["data_fingerprint"] != base["data_fingerprint"] or meta["parameters"] != base["parameters"]:
                raise ValueError("Control data or architecture differs")
            f = pd.read_parquet(run / "validation_predictions.parquet")
            observed = f.set_index(["episode_id", "round", "agent"])[["qid", "current_state", "target"]].sort_index()
            pd.testing.assert_frame_equal(observed, reference)
            scores, q = score(f, meta["kind"])
            stored = json.loads((run / "validation_metrics.json").read_text())
            for metric in METRICS:
                np.testing.assert_allclose(scores[metric], stored[metric], atol=1e-12, rtol=0)
            identity = dict(model=item["model"], state="availability_only", seed=item["seed"], run=str(run))
            cr.append(dict(identity, **{k:scores[k] for k in METRICS}, parameters=meta["parameters"], seconds=meta["elapsed_seconds_this_session"]))
            cq.append(q.assign(**identity))
            for t, group in f.groupby("round"):
                _, qt = score(group, meta["kind"])
                ct.append(qt.assign(**identity, round=t))
        cf, qf, tf = pd.DataFrame(cr), pd.concat(cq), pd.concat(ct)
        cf.to_csv(REPORTS / "history_control_runs.csv", index=False)
        qf.to_csv(REPORTS / "history_control_by_question.csv", index=False)
        tf.to_csv(REPORTS / "history_control_by_question_round.csv", index=False)
        diagnostics = {}
        lines = ["# History availability control", "",
            "This follow-up was triggered by the validation screen, before any controlled test evaluation. It holds the six-feature history architecture, training examples, optimization protocol, and seeds fixed. The preceding opinion's two one-hot values are set to zero; its availability flag and all current-opinion features remain. Transform checks verify that changing past opinions cannot affect the control input and that the initialization input is unchanged. This isolates the incremental value of past opinion values within the tested architecture.", "",
            "| Model | State | NLL, mean ± seed SD | Accuracy % |",
            "| --- | --- | ---: | ---: |"]
        merged = pd.concat([frame[frame.state.isin(["opinion","history"]) & frame.model.isin(["A0","A2"])], cf])
        for model in ("A0", "A2"):
            for state in ("opinion", "availability_only", "history"):
                sub = merged[(merged.model==model)&(merged.state==state)]
                lines.append(f"| {model} | {state} | {sub.nll.mean():.5f} ± {sub.nll.std():.5f} | {100*sub.accuracy.mean():.2f} |")
            full = questions[(questions.model==model)&(questions.state=="history")]
            control = qf[qf.model==model]
            delta = paired_bootstrap(full.groupby("qid").nll.mean(), control.groupby("qid").nll.mean())
            per_round = rounds[rounds.model==model].groupby(["round","state"]).nll.mean().unstack()
            gap = per_round.opinion-per_round.history
            op_mean, hist_mean = means.loc[(model,"opinion"),"nll"], means.loc[(model,"history"),"nll"]
            control_mean = cf[cf.model==model].nll.mean()
            diagnostics[model] = {"full_history_minus_availability_nll": delta,
                "per_question_full_history_minus_availability_nll": (full.groupby("qid").nll.mean()-control.groupby("qid").nll.mean()).to_dict(),
                "per_seed_full_history_minus_availability_nll": (full.groupby("seed").nll.mean()-control.groupby("seed").nll.mean()).to_dict(),
                "fraction_of_original_history_nll_gain_at_initialization": float(gap.loc[0]/gap.sum()),
                "fraction_of_original_history_nll_gain_recovered_by_availability_control": float((op_mean-control_mean)/(op_mean-hist_mean)),
                "opinion_nll": float(op_mean), "availability_nll": float(control_mean), "history_nll": float(hist_mean)}
        lines += ["", "Full history minus availability-only NLL (negative favors past opinion values), with descriptive paired four-question bootstrap intervals after averaging seeds:", ""]
        for model, d in diagnostics.items():
            v=d["full_history_minus_availability_nll"]
            lines.append(f"- {model}: {v['difference']:+.5f}, interval [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}].")
        d=diagnostics["A2"]
        note = f"For A2, {100*d['fraction_of_original_history_nll_gain_at_initialization']:.1f}% of the original history-versus-opinion NLL gap occurs at initialization, where no past opinion exists. The availability-only control recovers {100*d['fraction_of_original_history_nll_gain_recovered_by_availability_control']:.1f}% of that aggregate gap. These are different descriptive comparisons, not an additive causal decomposition. The remaining full-history advantage over the matched control needs more independent questions; training-seed repeats alone cannot establish state sufficiency."
        lines += ["", note, "",
            "All six controls use the same validation keys and targets as the original matrix, have matching data/checkpoint hashes, and stop by the original training protocol. Run manifests are named `control_manifest.json` and stored under `runs/history_availability_control/`; standard CLI evaluation does not apply this adapter and is deliberately unsupported for these artifacts. The adapter and base-code checksums are retained in the plan and manifests. No test outcomes were used.", "",
            "Reproduce: `.venv/bin/python -u scripts/history_availability_control.py`, then `.venv/bin/python scripts/report_validation.py`.", ""]
        (REPORTS / "history_availability_control.md").write_text("\n".join(lines))
        json_write(REPORTS / "history_control_summary.json", {"diagnostics": diagnostics, "runs": len(cf),
            "data_and_prediction_keys_verified": True, "adapter_sha256": plan["adapter_sha256"],
            "fit_seconds": float(cf.seconds.sum()), "control_rollouts_performed": False})
        findings += [note, "", "The matched control and seed-level results are in [history_availability_control.md](history_availability_control.md).", ""]
    result_path = REPORTS / "paper_binary_results.md"
    text = result_path.read_text()
    text = text.replace("## Protocol and interpretation", "\n".join(findings)+"\n## Protocol and interpretation", 1)
    if "diagnostics" in locals():
        text = text.replace("a round/mask-only control is still needed to isolate memory from that cue.", "the matched availability-only follow-up is documented in the linked control report.")
        text = text.replace("run the training-only PCA and round/mask-only controls", "run the training-only PCA control and review the completed availability-only follow-up")
    result_path.write_text(text)


def write_report(frame, aggregate, paired, rollouts):
    def mean_sd(key, metric, percentage=False):
        values = aggregate.loc[key, metric]
        scale = 100 if percentage else 1
        precision = 2 if percentage else 4
        out = f"{values['mean']*scale:.{precision}f}"
        if values['count'] > 1:
            out += f" ± {values['std']*scale:.{precision}f}"
        return out
    lines = ["# Controlled validation results", "",
        "Historical author-context validation screen: nine neural model/state configurations at seeds 42, 43, and 44, plus three baselines. All neural fits stopped by the configured early-stopping rule. These candidates received validation evaluation only. The completed training-only PCA study and its frozen test results are reported separately in [inductive_test_results.md](inductive_test_results.md).", "",
        "## Protocol and interpretation", "",
        "Training uses 16 questions, 512 episodes, and 131,072 agent transitions. Validation uses four other questions, 128 episodes, eight familiar random graphs, and 32,768 agent transitions. Every comparison has identical episode/round/agent keys and targets. No snapshots were missing in this subset. Metrics weight questions equally; accuracy means matching the next agent opinion, not answering the task correctly.", "",
        "The forecast starts after the current opinions, before the next messages. Opinion models also receive the same static persona/question context and signed graph. History adds the preceding opinion snapshot and availability indicators. Vote counts describe the current vote samples. Message features summarize the existing inbox, from the previous exchange, with separate signed-neighbor means; the initialization inbox is empty. Future messages are excluded.", "",
        "I0 is the controlled author Ising-style baseline fitted on the same 16 questions. A0 uses fixed signed aggregation with nonlinear context; A1 learns routing from opinion/graph features; A2 permits context in routing. The paper's official 20-question reproduction is in [author_reproduction.md](author_reproduction.md) and is not the baseline score in these tables. The history comparison adds only 768 trainable parameters, but also provides an availability mask that distinguishes initialization; a round/mask-only control is still needed to isolate memory from that cue.", "",
        "## One-step validation", "",
        "Mean ± sample standard deviation across three neural training seeds. NLL is the primary selection metric; Brier is the sum over the two classes. Lower NLL/Brier is better; higher accuracy/FCBA is better. P0 persistence has no assigned probability score.", "",
        "| Model | State | Parameters | NLL | Brier | Accuracy % | FCBA % | Mean fit seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key in aggregate.index:
        nll = "—" if key[0] == "P0" else mean_sd(key, "nll")
        brier = "—" if key[0] == "P0" else mean_sd(key, "brier")
        lines.append(f"| {key[0]} | {LABELS[key[1]]} | {int(aggregate.loc[key, ('parameters','mean')]):,} | {nll} | {brier} | {mean_sd(key,'accuracy',True)} | {mean_sd(key,'fcba',True)} | {aggregate.loc[key,('seconds','mean')]:.2f} |")
    lines += ["", "![State comparisons](validation_state_comparison.png)", "", "## Paired differences", "",
        "Each question's score is averaged across seeds before paired question resampling (1,000 bootstrap draws). Intervals are descriptive: only four independent questions are available, and these same validation questions selected early-stopping checkpoints. They are not confirmatory significance tests. Seed standard deviations in the preceding table do not measure uncertainty across new questions.", "",
        "Accuracy differences are percentage points; positive favors the first model.", "",
        "| First model | Reference | Δ accuracy (pp) | Descriptive 95% interval (pp) |",
        "| --- | --- | ---: | --- |"]
    for item in paired:
        if item["first"] not in (["A2", "opinion"], ["A2", "history"]) or item["reference"]==["A0", "history"]:
            continue
        value = item["metrics"]["accuracy"]
        lo, hi = value["ci95"]
        lines.append(f"| {' / '.join(item['first'])} | {' / '.join(item['reference'])} | {100*value['difference']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ["", "NLL differences: negative favors the first model.", "",
        "| First model | Reference | Δ NLL | Descriptive 95% interval |",
        "| --- | --- | ---: | --- |"]
    for item in paired:
        value = item["metrics"]["nll"]
        lo, hi = value["ci95"]
        lines.append(f"| {' / '.join(item['first'])} | {' / '.join(item['reference'])} | {value['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines += ["", "## Autonomous validation simulation", "",
        "Each simulation starts from the observed initial opinions and uses only generated states afterward. History models retain their own generated past. Deterministic trajectories and 32 sampled trajectories per initial state cover eight rounds. Vote-count and message models have no autonomous update for their extra state, so they are excluded here.", "",
        "| Model | State | Round-8 accuracy % | Final fraction Wasserstein | Sampled switches / agent | Sampled unanimity by round 8 % | Mean evaluation seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, data in rollouts.groupby(["model", "state"], sort=False):
        v = data.mean(numeric_only=True)
        accuracy = f"{100*v.final_accuracy:.2f}" + (f" ± {100*data.final_accuracy.std():.2f}" if len(data)>1 else "")
        wasserstein = f"{v.final_wasserstein:.4f}" + (f" ± {data.final_wasserstein.std():.4f}" if len(data)>1 else "")
        lines.append(f"| {key[0]} | {LABELS[key[1]]} | {accuracy} | {wasserstein} | {v.sampled_switches:.3f} | {100*v.sampled_consensus_fraction:.2f} | {v.rollout_seconds:.2f} |")
    lines += ["", f"Observed trajectories average {rollouts.observed_switches.iloc[0]:.3f} switches per agent; {100*rollouts.observed_consensus_fraction.iloc[0]:.2f}% reach unanimity within eight rounds. Wasserstein distance compares final positive-opinion fractions within each question/graph condition, then averages conditions and questions. Each condition has only four observed replicas; extra simulated draws do not add reference observations. The trajectory metric compares individual outcomes of a stochastic process and should be read alongside distributional metrics.", "",
        "![Rollout horizons](validation_rollout_horizons.png)", "", "## Cost, limits, and next checks", "",
        f"Total recorded fit/validation time across the 30 runs: {frame.seconds.sum():.1f} seconds. Autonomous evaluation across 18 runs: {rollouts.rollout_seconds.sum():.1f} seconds. Neural computation used MPS float32; baselines used CPU float64. Fit times include checkpointing and validation, but exclude separate message encoding. GPU stages ran sequentially. These are elapsed times from this machine, not estimates of speedup over the source LLM.", "",
        "The frozen message encoder has 22,713,216 parameters, additional to the trainable predictor. Development message encoding took roughly six minutes plus loading/initial encoding; the exact pass timings and truncation counts are saved in the encoding reports. Opinion/history rollouts make no LLM or encoder calls. Peak process/system memory was not instrumented for these runs, so no measured memory comparison is claimed.", "",
        "All configurations use the authors' compact question PCA, whose fitting provenance is unverified. A separate training-only PCA configuration is prepared but has not yet been evaluated at full scale. Training includes one source model (GPT-4o-mini), objective tasks, 32 fixed agents, and eight familiar random graphs. Generalization to new graph families, the locked test questions, other source models, and interventions remains untested here. Training seeds measure optimization variability, not new independent question samples.", "",
        "Before final test evaluation, inspect the seed and question variation, run the training-only PCA and round/mask-only controls, and freeze the selected predictor and autonomous simulator with their controls. Short-history improvements alone do not establish a sufficient state or a universal interaction law. Message results concern this frozen encoder and pooling scheme; they cannot show that language content is irrelevant. Causal influence requires a later controlled intervention, such as replacing a sender's argument while holding receiver state, source identity, graph, and other messages fixed.", "",
        "## Reproducibility", "",
        "- [Per-run results](validation_runs.csv), [question scores](validation_by_question.csv), [round scores](validation_by_question_round.csv), and [rollout metrics](validation_rollouts.csv).",
        "- [Paired comparisons](validation_paired_comparisons.json) and [integrity audit](validation_integrity.json).",
        "- Manifests, configs, code/data/checkpoint hashes, validation predictions, training histories, and rollout trajectories are stored under the run paths in `validation_runs.csv`.",
        "- Regenerate this report: `.venv/bin/python scripts/report_validation.py`.",
        "- Resume unfinished repetitions, if needed: `.venv/bin/python -u scripts/validation_repeats.py --minutes 60`.",
        "- These author-context candidates were evaluated on validation only; the training-only PCA test study is separate. No test-message encoding was performed. The separately labeled author reproduction follows its original split.", ""]
    (REPORTS / "paper_binary_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
