"""Read-only post-fit evaluation and report for the capacity diagnostics."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from mac_agents.data.features import Snapshots
from mac_agents.data.prepare import load_prepared
from scripts.ising_transformer_capacity import (
    CONFIG, TensorData, make_model, model_sha, predictions, probability_metrics,
    snapshot_arrays, validate_prepared,
)


def complete_manifests(root):
    result = []
    for stage in ("imitation", "observed", "residual"):
        path = root/f"{stage}_complete.json"
        if stage == "residual" and not path.exists():
            continue
        done = json.loads(path.read_text())
        assert done["fits"] == 9
        for name, sha in done["manifests"].items():
            run = root/name
            assert digest(run/"manifest.json") == sha
            m = json.loads((run/"manifest.json").read_text())
            assert m["status"] == "complete"
            for file, expected in m["artifacts"].items():
                assert digest(run/file) == expected, file
            result.append((run, m))
    return result


def error_breakdown(root, logp, observed):
    groups = np.load(root/"input_groups.npz", allow_pickle=False)
    assignment, counts = groups["assignment"], groups["counts"]
    assert assignment.shape == observed.shape
    conflict = (counts.min(-1) > 0)[assignment]
    predicted = logp.argmax(-1)
    mistakes = predicted != observed
    floor = int(counts.min(-1).sum())
    errors_conflicting = int(mistakes[conflict].sum())
    errors_consistent = int(mistakes[~conflict].sum())
    low = np.full(len(counts), 2, dtype=np.int64)
    high = np.full(len(counts), -1, dtype=np.int64)
    np.minimum.at(low, assignment.ravel(), predicted.ravel())
    np.maximum.at(high, assignment.ravel(), predicted.ravel())
    # Floating-point permutation effects must not invalidate the exact-input bound.
    assert (low == high).all(), "Numerical label disagreement within identical-input groups"
    assert errors_conflicting >= floor
    assert errors_conflicting+errors_consistent == int(mistakes.sum())
    return {"transitions": int(observed.size), "conflicting_transitions": int(conflict.sum()),
            "unavoidable_errors": floor, "errors_on_conflicting_inputs": errors_conflicting,
            "excess_errors_on_conflicting_inputs": errors_conflicting-floor,
            "errors_on_consistent_inputs": errors_consistent,
            "total_errors": int(mistakes.sum()), "identical_input_predictions_agree": True}


def decide_residual(cfg):
    root = Path(cfg["output"])
    validate_prepared(cfg)
    plan = json.loads((root/"residual_trigger_plan.json").read_text())
    assert plan["protocol_sha256"] == digest(root/"protocol.json")
    assert not plan["decision_uses_validation"]
    primary = [(run, m) for run, m in complete_manifests(root) if m["stage"] != "residual"]
    assert len(primary) == plan["requires_primary_fits_complete"]
    identity = {"protocol_sha256": digest(root/"protocol.json"),
                "trigger_plan_sha256": digest(root/"residual_trigger_plan.json"),
                "primary_manifests": {str(run.relative_to(root)): digest(run/"manifest.json") for run, _ in primary}}
    destination = root/"residual_decision.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        assert all(existing[k] == v for k, v in identity.items())
        print(json.dumps(existing, indent=2))
        return
    close = sum(m["close_ising_imitation"] for _, m in primary if m["stage"] == "imitation")
    chosen = []
    for seed in cfg["training"]["seeds"]:
        run, m = min(((run, m) for run, m in primary if m["stage"] == "observed" and m["seed"] == seed),
                     key=lambda item: item[1]["selected_train"]["observed_nll"])
        chosen.append({"seed": seed, "run": str(run.relative_to(root)),
                       "training_error": m["selected_train"]["observed_error"],
                       "training_nll": m["selected_train"]["observed_nll"]})
    error = float(np.mean([r["training_error"] for r in chosen]))
    threshold = plan["trigger_if_mean_observed_training_error_above"]
    session = json.loads((root/"session.json").read_text())
    available = session["training_deadline_epoch_seconds"]-time.time()
    expected = 9*max(m["seconds"] for _, m in primary)+300
    enough_imitation = close >= plan["requires_close_imitation_fits_at_least"]
    unresolved_gap, enough_time = error > threshold, available > expected
    enabled = enough_imitation and unresolved_gap and enough_time
    if enabled:
        reason = f"Training-selected error {100*error:.2f}% remains above the predeclared {100*threshold:.2f}% trigger, {close}/9 imitation fits passed, and the remaining training budget covers nine additional fits plus a five-minute reserve."
    elif not unresolved_gap:
        reason = f"Training-selected error {100*error:.2f}% is at or below the predeclared {100*threshold:.2f}% trigger; the main fitting runs closed at least half the earlier gap above the identical-input bound."
    elif not enough_imitation:
        reason = f"Only {close}/9 imitation fits passed, below the seven-fit requirement."
    else:
        reason = "The remaining aggregate training budget cannot cover nine additional fits plus the five-minute reserve."
    decision = {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
                "enabled": enabled, "reason": reason, "decision_uses_validation": False,
                "chosen_observed_runs": chosen, "mean_observed_training_error": error,
                "trigger_threshold": threshold, "close_imitation_fits": close,
                "available_training_seconds": available, "estimated_seconds_with_reserve": expected}
    json_write(destination, decision)
    print(json.dumps(decision, indent=2))


def evaluate(cfg, device):
    root = Path(cfg["output"])
    protocol = validate_prepared(cfg)
    manifests = complete_manifests(root)
    decision = json.loads((root/"residual_decision.json").read_text())
    assert decision["protocol_sha256"] == digest(root/"protocol.json")
    assert not decision["decision_uses_validation"]
    assert len(manifests) == 18+9*decision["enabled"]
    bounds = json.loads((root/"bounds.json").read_text())
    reference = json.loads(Path(cfg["reference_comparison"]).read_text())
    baseline_training = json.loads((root/"ising_train_metrics.json").read_text())
    np.testing.assert_allclose(baseline_training["observed_error"], reference["summary"]["I0"]["train_error"], atol=1e-12)
    np.testing.assert_allclose(baseline_training["observed_nll"], reference["summary"]["I0"]["train_nll"], atol=1e-12)
    arrays, index, directory = load_prepared(protocol["original_model_config"])
    for key, file in (("episodes_sha256", "episodes.npz"), ("index_sha256", "index.json"), ("split_sha256", "splits.json")):
        assert digest(directory/file) == protocol["source_data"][key]
    ds = Snapshots(arrays, index, protocol["original_model_config"], "validation", "opinion")
    assert len(ds) == 1024
    baseline = json.loads(Path(protocol["baseline_path"]).read_text())
    val, val_rows = snapshot_arrays(arrays, index, ds, baseline)
    train = dict(np.load(root/"train.npz", allow_pickle=False))
    train_rows = json.loads((root/"train_rows.json").read_text())
    train_q = np.array([r["qid"] for r in train_rows])
    val_q = np.array([r["qid"] for r in val_rows])
    assert len(set(val_q)) == 4 and set(train_q).isdisjoint(val_q)
    val_data = TensorData(val, device)
    # The reproduction sample is fixed by input order, independent of outcomes.
    sample = np.linspace(0, len(train["s"])-1, 64, dtype=int)
    sample_data = TensorData({k: v[sample] for k, v in train.items()}, "cpu")
    records, question_records, reproducibility, breakdowns = [], [], [], []
    evaluated = root/"evaluation"
    evaluated.mkdir(exist_ok=True)
    for run, m in manifests:
        name = str(run.relative_to(root)).replace("/", "__")
        out = evaluated/name
        out.mkdir(exist_ok=True)
        identity = {"run_manifest_sha256": digest(run/"manifest.json"),
                    "protocol_sha256": digest(root/"protocol.json"), "report_code_sha256": digest(__file__),
                    "residual_decision_sha256": digest(root/"residual_decision.json")}
        if (out/"complete.json").exists():
            done = json.loads((out/"complete.json").read_text())
            assert all(done[k] == v for k, v in identity.items())
            for file, sha in done["artifacts"].items():
                assert digest(out/file) == sha
            records += done["records"]
            question_records += done["by_question"]
            reproducibility.append(done["reproduction"])
            breakdowns.append(done["error_breakdown"])
            continue
        residual = m["residual"]
        fields = {"run": name, "stage": m["stage"], "seed": m["seed"], "learning_rate": m["initial_learning_rate"]}
        model = make_model(protocol, residual).to(device)
        train_lp = np.load(run/"train_logp.npy", allow_pickle=False)
        assert train_lp.shape == (4096, 32, 2)
        mm = probability_metrics(train_lp, train["observed"], train["teacher"])
        breakdown = {**fields, **error_breakdown(root, train_lp, train["observed"])}
        assert mm["observed_error"]+1e-5 >= bounds["classification_error_lower_bound"]
        assert mm["observed_nll"]+1e-6 >= bounds["cross_entropy_lower_bound"]
        for key in mm:
            np.testing.assert_allclose(mm[key], m["selected_train"][key], atol=2e-7, rtol=2e-6)
        own_records, own_q = [], []
        for checkpoint in ("selected", "terminal"):
            file = "best.pt" if checkpoint == "selected" else "resume.pt"
            ck = torch.load(run/file, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            expected_sha = m["selected_weights_sha256"] if checkpoint == "selected" else m["terminal_weights_sha256"]
            assert model_sha(model) == expected_sha
            lp = predictions(model, val_data, 256, residual)
            np.save(out/f"{checkpoint}_validation_logp.npy", lp, allow_pickle=False)
            metrics = probability_metrics(lp, val["observed"], val["teacher"])
            own_records.append({**fields, "checkpoint": checkpoint, "split": "validation", "epoch": ck["epoch"], **metrics})
            if checkpoint == "selected":
                own_records.append({**fields, "checkpoint": checkpoint, "split": "train", "epoch": ck["epoch"], **mm})
                for partition, logp, source, questions in (("train", train_lp, train, train_q), ("validation", lp, val, val_q)):
                    for q in sorted(set(questions)):
                        ids = np.flatnonzero(questions == q)
                        metric = probability_metrics(logp[ids], source["observed"][ids], source["teacher"][ids])
                        own_q.append({**fields, "split": partition, "qid": q, **metric})
            else:
                terminal_metrics = {key: m["terminal_train"][key] for key in mm}
                own_records.append({**fields, "checkpoint": checkpoint, "split": "train", "epoch": ck["epoch"], **terminal_metrics})
        # All fitted selected checkpoints are checked on a different backend.
        cpu = make_model(protocol, residual)
        ck = torch.load(run/"best.pt", map_location="cpu", weights_only=False)
        cpu.load_state_dict(ck["model"])
        cpu_lp = predictions(cpu, sample_data, 64, residual)
        expected = np.exp(train_lp[sample, :, 1])
        actual = np.exp(cpu_lp[..., 1])
        np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)
        reproduction = {**fields, "snapshots": len(sample), "transitions": actual.size,
                        "maximum_probability_difference": float(np.max(np.abs(actual-expected)))}
        artifacts = {p.name: digest(p) for p in out.glob("*.npy")}
        json_write(out/"complete.json", {**identity, "completed_utc": datetime.now(timezone.utc).isoformat(),
            "records": own_records, "by_question": own_q, "reproduction": reproduction,
            "error_breakdown": breakdown, "artifacts": artifacts})
        records += own_records
        question_records += own_q
        reproducibility.append(reproduction)
        breakdowns.append(breakdown)
        print(json.dumps({"evaluated": name, "training_nll": mm["observed_nll"]}), flush=True)
    metrics = pd.DataFrame(records)
    byq = pd.DataFrame(question_records)
    repro = pd.DataFrame(reproducibility)
    metrics.to_csv(root/"metrics.csv", index=False)
    byq.to_csv(root/"metrics_by_question.csv", index=False)
    repro.to_csv(root/"checkpoint_reproduction.csv", index=False)
    pd.DataFrame(breakdowns).to_csv(root/"error_breakdown.csv", index=False)
    old = json.loads((root/"prior_artifacts.json").read_text())
    for file, sha in old.items():
        assert digest(file) == sha, f"Prior artifact changed: {file}"
    for seed in cfg["training"]["seeds"]:
        hashes = {m["initial_weights_sha256"] for _, m in manifests if m["seed"] == seed and not m["residual"]}
        assert len(hashes) == 1
    baseline_val = probability_metrics(val["teacher_logp"], val["observed"], val["teacher"])
    json_write(root/"ising_validation_metrics.json", baseline_val)
    json_write(root/"evaluation_complete.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "fits": len(manifests), "device": device, "no_test_evaluation": True,
        "selection_used_only_training": True, "prior_artifacts_verified_unchanged": len(old),
        "all_matched_initializations_verified": True,
        "maximum_cpu_mps_probability_difference": float(repro.maximum_probability_difference.max()),
        "implementation_sha256": digest(__file__),
        "artifacts": {name: digest(root/name) for name in ("metrics.csv", "metrics_by_question.csv", "checkpoint_reproduction.csv", "error_breakdown.csv", "ising_validation_metrics.json")}})


def summarize(cfg):
    root = Path(cfg["output"])
    manifests = complete_manifests(root)
    evaluation = json.loads((root/"evaluation_complete.json").read_text())
    assert evaluation["implementation_sha256"] == digest(__file__)
    for file, sha in evaluation["artifacts"].items():
        assert digest(root/file) == sha
    frame = pd.read_csv(root/"metrics.csv")
    selected = frame[frame.checkpoint == "selected"]
    metrics = ["observed_error", "observed_nll", "teacher_kl", "teacher_probability_mae", "teacher_label_disagreement"]
    summary = selected.groupby(["stage", "learning_rate", "split"])[metrics].agg(["mean", "std"])
    summary.to_csv(root/"summary_by_rate.csv")
    chosen_names = []
    for stage in selected.stage.unique():
        subset = selected[(selected.stage == stage) & (selected.split == "train")]
        key = "teacher_kl" if stage == "imitation" else "observed_nll"
        for _, group in subset.groupby("seed"):
            chosen_names.append(group.loc[group[key].idxmin(), "run"])
    chosen = selected[selected.run.isin(chosen_names)].copy()
    breakdown = pd.read_csv(root/"error_breakdown.csv")
    chosen_breakdown = breakdown[breakdown.run.isin(chosen_names)]
    error_fields = ["transitions", "conflicting_transitions", "unavoidable_errors",
                    "errors_on_conflicting_inputs", "excess_errors_on_conflicting_inputs",
                    "errors_on_consistent_inputs", "total_errors"]
    chosen.to_csv(root/"training_selected_rates.csv", index=False)
    aggregate = chosen.groupby(["stage", "split"])[metrics].mean().reset_index()
    aggregate.to_csv(root/"training_selected_summary.csv", index=False)
    rate_records = []
    for (stage, lr, split), part in selected.groupby(["stage", "learning_rate", "split"]):
        rate_records.append({"stage": stage, "learning_rate": float(lr), "split": split,
            **{key: float(part[key].mean()) for key in metrics},
            "seeds": len(part), "error_min_seed": float(part.observed_error.min()), "error_max_seed": float(part.observed_error.max())})
    histories = []
    for run, m in manifests:
        for h in json.loads((run/"history.json").read_text()):
            histories.append({"stage": m["stage"], "seed": m["seed"], "initial_learning_rate": m["initial_learning_rate"], **h})
    hist = pd.DataFrame(histories)
    hist.to_csv(root/"training_curves.csv", index=False)
    bounds = json.loads((root/"bounds.json").read_text())
    results = {"completed_utc": evaluation["completed_utc"], "fit_count": len(manifests),
        "parameters": 135458, "bounds": bounds,
        "ising_train": json.loads((root/"ising_train_metrics.json").read_text()),
        "ising_validation": json.loads((root/"ising_validation_metrics.json").read_text()),
        "all_rates": rate_records, "training_selected_rate_summary": aggregate.to_dict("records"),
        "chosen_rates": chosen[chosen.split == "train"][["stage", "seed", "learning_rate", "epoch"]].to_dict("records"),
        "training_selected_error_breakdown": chosen_breakdown.groupby("stage")[error_fields].mean().reset_index().to_dict("records"),
        "residual_decision": json.loads((root/"residual_decision.json").read_text()),
        "imitation_close_fits": sum(m["close_ising_imitation"] for _, m in manifests if m["stage"] == "imitation"),
        "plateau_counts": {stage: sum(m["descriptive_training_plateau"] for _, m in manifests if m["stage"] == stage) for stage in sorted(selected.stage.unique())},
        "training_seconds": float(sum(m["seconds"] for _, m in manifests)),
        "prior_artifacts_verified_unchanged": evaluation["prior_artifacts_verified_unchanged"],
        "maximum_cpu_mps_probability_difference": evaluation["maximum_cpu_mps_probability_difference"],
        "new_source_model_calls": 0, "new_test_evaluations": 0}
    json_write(root/"results.json", results)
    write_report(cfg, results, hist, selected)
    paths = [root/"results.json", root/"summary_by_rate.csv", root/"training_selected_rates.csv",
             root/"training_selected_summary.csv", root/"training_curves.csv",
             Path("reports/ising_transformer_capacity.md"), Path("reports/ising_transformer_capacity.png")]
    json_write(root/"report_integrity.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "report_script_sha256": digest(__file__), "artifacts": {str(p): digest(p) for p in paths}})
    print(json.dumps(results, indent=2))


def write_report(cfg, results, hist, selected):
    root = Path(cfg["output"])
    old = json.loads(Path(cfg["reference_comparison"]).read_text())
    terminal = json.loads(Path(cfg["reference_followup"]).read_text())
    agg = {(r["stage"], r["split"]): r for r in results["training_selected_rate_summary"]}
    teacher, actual = agg["imitation", "train"], agg["observed", "train"]
    teacher_val = agg["imitation", "validation"]
    old_val = []
    for item in old["runs"]:
        if item["model"] == "A2":
            old_val.append(json.loads((Path(item["run"])/"validation_metrics.json").read_text()))
    assert len(old_val) == 3
    old_val_error = 1-np.mean([x["accuracy"] for x in old_val])
    old_val_nll = np.mean([x["nll"] for x in old_val])
    lines = ["# Ising versus transformer: training-capacity diagnostics", "",
        f"Completed {results['fit_count']} full-data diagnostic fits on the original GPT-4o-mini training cohort. The architecture is the unchanged **135,458-parameter opinion-based A2 graph transformer**, with dropout disabled. This is separate from the newer 282,146-parameter Llama/argument study.", "",
        f"All nine probability-imitation fits met the predeclared closeness criterion." if results["imitation_close_fits"] == 9 else f"{results['imitation_close_fits']}/9 probability-imitation fits met the predeclared closeness criterion.",
        f"Selecting the learning rate separately for each seed using training objective only, mean absolute difference from the Ising probability is **{100*teacher['teacher_probability_mae']:.3f} percentage points**, with KL {teacher['teacher_kl']:.6g} nats. The observed-outcome transformer reaches **{100*actual['observed_error']:.2f}% training error**, versus Ising {100*results['ising_train']['observed_error']:.2f}%, the original validation-selected transformer {100*old['summary']['A2']['train_error']:.2f}%, and its old epoch-eight checkpoint {100*terminal['summary']['terminal_train_error_mean']:.2f}%.", "",
        "## What changed", "",
        "The same 16 training questions, 512 episodes, 4,096 snapshots and 131,072 agent transitions are used throughout. Inputs are current opinions, the original 16-dimensional persona/question context, signed graph and node degrees. The original PCA and data are frozen. No arguments, new text representations, generator calls or test outcomes enter these fits.", "",
        "There are two primary objectives. Ising imitation uses the frozen 19-coefficient Ising model's P(A) as a soft target at every existing input state. Observed-outcome fitting uses exactly the original next majority-of-five opinion. The soft-target loss has a nonzero entropy floor; KL to the teacher, not raw cross-entropy, measures imitation error.", "",
        "Both stages use seeds 42/43/44, initial learning rates 0.0001/0.0003/0.001, AdamW with zero weight decay, zero dropout, gradient clipping 1, and 128 epochs (16,384 optimizer updates) per fit. Learning rate decays by a predeclared cosine schedule to 1% of its initial value. There is no validation-based stopping. Checkpoints are chosen by full-training objective at initialization, epochs 1/2 and every 4 epochs; a separate checkpoint records the lowest observed training error. Initial weights match across the primary objectives and learning rates within each seed.", "",
        "The earlier selected models used dropout 0.1, weight decay 0.001 and validation selection at epoch 2, with optimization ending at epoch 8. The new comparison changes budget, regularization and learning-rate schedule together. It diagnoses whether stronger fitting is possible under this architecture; it does not isolate which individual training change causes the gain.", "",
        "## Ising probability-imitation control", "",
        "Means over three seeds. 'Training' uses observed training states; 'validation' uses the existing four development questions after all fitting is complete. This control evaluates reproduction of a known input-output function, not additional source-agent accuracy.", "",
        "| Initial learning rate | Training probability MAE | Training KL | Validation probability MAE | Validation KL |",
        "| --- | ---: | ---: | ---: | ---: |"]
    def metric(stage, lr, split):
        return next(r for r in results["all_rates"] if r["stage"] == stage and r["learning_rate"] == lr and r["split"] == split)
    for lr in cfg["training"]["learning_rates"]:
        tr, va = metric("imitation", lr, "train"), metric("imitation", lr, "validation")
        lines.append(f"| {lr:g} | {100*tr['teacher_probability_mae']:.3f} pp | {tr['teacher_kl']:.6g} | {100*va['teacher_probability_mae']:.3f} pp | {va['teacher_kl']:.6g} |")
    lines += ["", "Closeness was declared before fitting as training probability MAE ≤0.5 pp and KL ≤0.0005 nats. Reported continuous scores and validation transfer matter beyond the binary criterion. Successful imitation on sampled states does not prove exact representational inclusion for arbitrary inputs.", "",
        "## Fit to observed opinions", "",
        "| Model / initial learning rate | Training error | Training NLL | Existing validation error | Existing validation NLL |",
        "| --- | ---: | ---: | ---: | ---: |"]
    bt, bv = results["ising_train"], results["ising_validation"]
    lines.append(f"| Ising, frozen reference | {100*bt['observed_error']:.2f}% | {bt['observed_nll']:.5f} | {100*bv['observed_error']:.2f}% | {bv['observed_nll']:.5f} |")
    lines.append(f"| Original transformer, validation-selected epoch 2 | {100*old['summary']['A2']['train_error']:.2f}% | {old['summary']['A2']['train_nll']:.5f} | {100*old_val_error:.2f}% | {old_val_nll:.5f} |")
    for lr in cfg["training"]["learning_rates"]:
        tr, va = metric("observed", lr, "train"), metric("observed", lr, "validation")
        lines.append(f"| Unregularized transformer, {lr:g} | {100*tr['observed_error']:.2f}% | {tr['observed_nll']:.5f} | {100*va['observed_error']:.2f}% | {va['observed_nll']:.5f} |")
    for stage in ("observed", "residual"):
        if (stage, "train") not in agg:
            continue
        tr, va = agg[stage, "train"], agg[stage, "validation"]
        label = "Unregularized transformer" if stage == "observed" else "Ising + neural correction"
        lines.append(f"| {label}, rate selected on training NLL per seed | {100*tr['observed_error']:.2f}% | {tr['observed_nll']:.5f} | {100*va['observed_error']:.2f}% | {va['observed_nll']:.5f} |")
    lines += ["", "The rate-selection summary is a capacity diagnostic, not a new generalization-optimized model. All nine rates/seeds are reported above or in the saved tables. Validation is scored only after fitting and never chooses checkpoint, rate, budget or stopping. Its four questions have been repeatedly used in previous exploratory work; no fresh confirmatory claim follows from these scores.", "",
        "## Input conflicts and remaining optimization uncertainty", "",
        f"Grouping actual accessible receiver/sender features up to neighbor permutation gives {results['bounds']['groups']:,} distinct groups, with {results['bounds']['conflicting_groups']:,} groups containing opposite outcomes. At least {results['bounds']['unavoidable_errors']:,} of {results['bounds']['transitions']:,} labels must be misclassified by a deterministic predictor assigning one answer to each group: **{100*results['bounds']['classification_error_lower_bound']:.2f}%**. The corresponding empirical cross-entropy lower bound is **{results['bounds']['cross_entropy_lower_bound']:.5f}** nats. The equality key excludes arbitrary graph/persona/question IDs, next outcomes and unobserved history; it includes the degree features actually supplied to each sender and receiver.", "",
        "These are in-sample lower bounds. They are not a guaranteed attainable optimum for the architecture, an estimate of source-model Bayes error, or proof of sufficient state. Distinct feature groups may still be indistinguishable after the architecture's aggregation. No unique example identifiers were given to the learner.", "",
        f"The predeclared plateau description checks whether the best training objective improves by less than 0.0001 over the final 32 epochs. Plateau counts out of nine fits per stage are {results['plateau_counts']}. Every run used the fixed 128-epoch budget. Runs still improving at the end are budget-limited; none establishes a global optimization/capacity limit. The schedule itself also reduces late updates, so an observed plateau is not proof of convergence to a global optimum.", ""]
    decomp = next(x for x in results["training_selected_error_breakdown"] if x["stage"] == "observed")
    lines += [f"For the observed-outcome model with training-selected rates, its {100*actual['observed_error']:.2f}% total training error decomposes into {100*decomp['unavoidable_errors']/decomp['transitions']:.2f} pp forced by conflicting labels, {100*decomp['excess_errors_on_conflicting_inputs']/decomp['transitions']:.2f} pp additional errors on conflicting-input groups, and {100*decomp['errors_on_consistent_inputs']/decomp['transitions']:.2f} pp on groups with consistent observed labels. Predicted classes agree within every identical-input group. This bookkeeping separates an established information bound from the remaining empirical fitting gap; it does not say that every remaining error is removable by this architecture.", ""]
    if ("residual", "train") in agg:
        lines += ["## Conditional neural-correction control", "",
            "After the primary diagnostics, a separately recorded decision enabled the planned residual control. Its score is the frozen Ising logit plus the same-size neural graph output difference. The correction's final output layer starts at zero, reproducing Ising exactly before optimization. All other inputs, seeds, rates and training budgets match. The baseline is frozen; the neural correction has 135,458 trainable parameters. The decision and reason are preserved in `residual_decision.json`. This is adaptive exploratory follow-up, not a predeclared primary comparison.", ""]
    else:
        lines += ["The conditional Ising-plus-neural-correction stage was not run. " + results["residual_decision"]["reason"] + " Its training-only trigger and decision are preserved in `residual_trigger_plan.json` and `residual_decision.json`.", ""]
    val_actual = agg["observed", "validation"]
    improved_fit = actual["observed_nll"] < old["summary"]["A2"]["train_nll"]
    lines += ["## Interpretation", "",
        ("Successful probability imitation shows that this architecture and optimization setup can closely reproduce the simple Ising mapping on the observed training states. It rules out an inability to learn that reference function on those states as a sufficient explanation for the earlier small fitting gain; it does not certify every possible function or state." if results["imitation_close_fits"] else "The imitation control did not meet its predeclared criterion; inspect its continuous errors before interpreting observed-label fitting."), "",
        f"For the training-selected imitation rates, probability MAE on the separate development questions is {100*teacher_val['teacher_probability_mae']:.3f} pp and KL is {teacher_val['teacher_kl']:.6g}. Compare this with the training scores above when assessing whether the known rule transferred beyond the fitted question contexts. A small training imitation error alone does not establish such transfer.", "",
        ("The larger training-fit improvement shows that the earlier selected error was not the transformer's maximum demonstrated fitting ability. " if improved_fit else "The new budget did not substantially improve the measured training objective. ") +
        ("The stronger fit does not improve existing-validation NLL over the original selected model. This is consistent with overfitting the available training questions and outcomes; it does not identify whether missing state, limited independent questions, or other data properties dominate." if val_actual["observed_nll"] >= old_val_nll else "The stronger fit also improves existing-validation NLL, but these repeatedly inspected questions cannot serve as a fresh confirmation."), "",
        "This diagnosis concerns the opinion/context graph predictor on the original GPT cohort. It does not resolve why adding argument vectors helps little in the separate Llama pilot, and it does not measure causal influence or an autonomous multi-round law.", "",
        "## Checks and artifacts", "",
        f"Seven focused checks cover hard/soft objective equivalence, teacher entropy versus KL, count-weighted input bounds, no next-opinion leakage into teacher/features, neighborhood/permutation equality, exact Ising inclusion of a zero correction, and active learning/equivalence of the unregularized architecture. Every selected checkpoint was reloaded on CPU for 64 fixed training snapshots; maximum probability difference from saved MPS outputs is {results['maximum_cpu_mps_probability_difference']:.3g}. All {results['prior_artifacts_verified_unchanged']} recorded files from the original study remain unchanged. The total recorded fit time is {results['training_seconds']/60:.1f} minutes. No new LLM generation, paid API calls, consumed-test scoring or autonomous rollout occurred.", "",
        f"![Training diagnostics]({Path('reports/ising_transformer_capacity.png').resolve()})", "",
        "Curves show three-seed means for each fixed initial learning rate. The observed-error and NLL axes zoom into the fitted region, excluding the higher initialization values; full numerical histories include initialization.", "",
        "```sh", ".venv/bin/python -m pytest tests/test_ising_transformer_capacity.py",
        ".venv/bin/python scripts/ising_transformer_capacity.py prepare",
        ".venv/bin/python scripts/ising_transformer_capacity.py imitation --device mps",
        ".venv/bin/python scripts/ising_transformer_capacity.py observed --device mps",
        ".venv/bin/python scripts/report_ising_transformer_capacity.py decide-residual",
        ".venv/bin/python scripts/report_ising_transformer_capacity.py evaluate --device mps",
        ".venv/bin/python scripts/report_ising_transformer_capacity.py summarize", "```", "",
        "Completed fits and evaluations are reused only after provenance/hash checks. Models, curves, grouped inputs, all-rate scores, training-selected rates and validation outputs are under `runs/ising_transformer_capacity_v1/`. The conditional residual stage requires its decision file and is not launched by the primary commands.", ""]
    Path("reports/ising_transformer_capacity.md").write_text("\n".join(lines))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    colors = ["#31688e", "#35b779", "#d26b38"]
    for lr, color in zip(cfg["training"]["learning_rates"], colors):
        im = hist[(hist.stage == "imitation") & (hist.initial_learning_rate == lr)].groupby("epoch").teacher_probability_mae.mean()
        ob = hist[(hist.stage == "observed") & (hist.initial_learning_rate == lr)].groupby("epoch")[["observed_error", "observed_nll"]].mean()
        axes[0].plot(im.index, 100*im, label=f"lr {lr:g}", color=color)
        axes[1].plot(ob.index, 100*ob.observed_error, color=color)
        axes[2].plot(ob.index, ob.observed_nll, color=color)
    axes[0].set_yscale("log")
    axes[0].axhline(.5, color="0.5", ls="--", lw=1)
    axes[0].set_ylabel("Ising probability MAE (percentage points)")
    axes[0].set_title("Learn the known Ising mapping")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(100*results["ising_train"]["observed_error"], color="0.5", ls="--", lw=1, label="Ising")
    axes[1].axhline(100*results["bounds"]["classification_error_lower_bound"], color="0.5", ls=":", lw=1, label="Input-conflict lower bound")
    axes[1].set_ylim(0, 12)
    axes[1].set_ylabel("Training error (%)")
    axes[1].set_title("Fit observed agent opinions")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].axhline(results["ising_train"]["observed_nll"], color="0.5", ls="--", lw=1)
    axes[2].axhline(results["bounds"]["cross_entropy_lower_bound"], color="0.5", ls=":", lw=1)
    axes[2].set_ylim(0, .26)
    axes[2].set_ylabel("Training negative log likelihood")
    axes[2].set_title("Same labels, probability objective")
    for ax in axes:
        ax.set_xlabel("Training epoch")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Original 135,458-parameter graph transformer · no dropout/weight decay · three-seed means")
    fig.savefig("reports/ising_transformer_capacity.png", dpi=180)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("decide-residual", "evaluate", "summarize"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    cfg = config(CONFIG)
    torch.set_num_threads(4)
    if args.stage == "decide-residual":
        decide_residual(cfg)
    elif args.stage == "evaluate":
        evaluate(cfg, args.device)
    else:
        summarize(cfg)


if __name__ == "__main__":
    main()
