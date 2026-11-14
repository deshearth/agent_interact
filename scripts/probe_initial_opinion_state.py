"""Nested, exploratory probe of initial response tendencies as extra state.

Each episode supplies only its own time-zero votes. No later outcomes or other
episodes of its question are used to construct its new input features.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, digest, json_write
from scripts.diagnose_ising_gap import score_metrics

SOURCE = Path("runs/transformer_generalization_v1")
OUTPUT = Path("runs/ising_initial_state_probe_v1")
KINDS = ("calibration", "current_state", "initial_state")
PENALTIES = (0.0001, 0.001, 0.01)


def initial_features(votes):
    initial = np.asarray(votes)[:, 0]
    assert initial.ndim == 3 and initial.shape[1:] == (32, 5)
    assert np.isin(initial, [-1, 1]).all()
    count = (initial == 1).sum(-1)
    # Add-one-half smoothing supplies finite sample log-odds, not known truth.
    individual = logit((count+0.5)/6)
    network = logit((count.sum(-1)+0.5)/161)
    return np.stack([individual, np.broadcast_to(network[:, None], individual.shape)], -1)


def design(kind, prior, current, initial):
    columns = [np.ones_like(prior), prior]
    if kind in ("current_state", "initial_state"):
        columns.append(current)
    if kind == "initial_state":
        columns.extend([initial[..., 0], initial[..., 1]])
    assert kind in KINDS
    return np.stack(columns, -1).reshape(-1, len(columns))


def standardize(fitting, other):
    mean, scale = fitting.mean(0), fitting.std(0)
    mean[0], scale[0] = 0, 1
    scale[scale < 1e-10] = 1
    return (fitting-mean)/scale, (other-mean)/scale, mean, scale


def objective(weight, x, offset, target, penalty):
    z = offset+x@weight
    loss = np.mean(np.logaddexp(0, z)-target*z) + penalty*np.dot(weight, weight)/2
    gradient = x.T@(expit(z)-target)/len(target) + penalty*weight
    return float(loss), gradient


def fit(x, offset, target, penalty):
    result = minimize(objective, np.zeros(x.shape[1]), args=(x, offset, target, penalty),
        method="L-BFGS-B", jac=True, options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 1000})
    gradient = float(np.abs(objective(result.x, x, offset, target, penalty)[1]).max())
    assert result.success and gradient < 1e-6, (result.message, gradient)
    return result.x, {"iterations": int(result.nit), "maximum_gradient": gradient}


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(path):
        path = Path(path)
        sources[str(path)] = digest(path)
        return path

    source_protocol = json.loads(read(SOURCE/"protocol.json").read_text())
    audit = json.loads(read(SOURCE/"completion_audit.json").read_text())
    assert code_digest() == audit["core_sha256"]
    common = dict(np.load(read(SOURCE/"development.npz"), allow_pickle=False))
    index = json.loads(read(SOURCE/"episode_index.json").read_text())
    assert all(r["official_split"] == "train" and r["graph_role"] == "fit_graph" for r in index)
    original_path = Path("data/processed/train_only_pca/episodes.npz")
    index_path = Path("data/processed/train_only_pca/index.json")
    original_index = json.loads(read(index_path).read_text())
    lookup = {r["episode_id"]: i for i, r in enumerate(original_index)}
    ids = [lookup[r["episode_id"]] for r in index]
    original = np.load(read(original_path), allow_pickle=False)
    votes = original["votes"][ids]
    initial = initial_features(votes)
    np.testing.assert_array_equal(np.where((votes[:, 0] == 1).sum(-1) >= 3, 1, -1), common["spins"][:, 0])
    e, t = common["pairs"].T
    y = (common["spins"][e, t+1] == 1).astype(np.float64)
    qids = common["episode_qids"][e]
    current = common["spins"][e, t].astype(np.float64)
    initialization = initial[e]
    features = {f["id"]: dict(np.load(read(SOURCE/"folds"/f["id"]/"features.npz"), allow_pickle=False))
                for f in source_protocol["folds"]}
    for path, sha in sources.items():
        if path in audit["artifacts"]:
            assert sha == audit["artifacts"][path]
        if path in source_protocol["source"]:
            assert sha == source_protocol["source"][path]
    spec = {"script_sha256": digest(__file__), "source_artifacts": sources,
        "models": {"calibration": ["bias", "Ising logit"],
                   "current_state": ["bias", "Ising logit", "current own opinion"],
                   "initial_state": ["bias", "Ising logit", "current own opinion",
                                     "own initial response log-odds", "network initial response log-odds"]},
        "penalties": list(PENALTIES), "objective": "mean BCE + lambda/2 * sum(weights squared)",
        "input_scope": "Time-zero five votes per agent, from this episode only; never later votes",
        "new_global_context": "Initial network mean pools 32 agents; this explicitly enriches the local state",
        "folds": source_protocol["folds"], "selection_metric": "Mean inner-question NLL",
        "standardization": "Training-fold features only; prior fitted inside that same training fold",
        "fresh_confirmatory": False, "new_source_calls": 0, "official_test_scoring": False}
    if (OUTPUT/"protocol.json").exists():
        assert json.loads((OUTPUT/"protocol.json").read_text())["specification"] == spec
    else:
        json_write(OUTPUT/"protocol.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "specification": spec})
    tuning, fitted = [], []
    with threadpool_limits(limits=1):
        for fold in source_protocol["folds"]:
            if fold["phase"] != "tune":
                continue
            prior = features[fold["id"]]["prior_logit"].astype(np.float64)
            training = np.isin(qids, fold["train_questions"])
            validation = np.isin(qids, fold["validation_questions"])
            assert not (training & validation).any()
            for kind in KINDS:
                xfit, xval, mean, scale = standardize(
                    design(kind, prior[training], current[training], initialization[training]),
                    design(kind, prior[validation], current[validation], initialization[validation]))
                for penalty in PENALTIES:
                    w, details = fit(xfit, prior[training].ravel(), y[training].ravel(), penalty)
                    z = (prior[validation].ravel()+xval@w).reshape(y[validation].shape)
                    tuning.append({"outer": fold["outer"], "fold": fold["id"], "model": kind,
                        "penalty": penalty, **details, **score_metrics(z, y[validation])})
            print(json.dumps({"inner_fold_complete": fold["id"]}), flush=True)
        tune = pd.DataFrame(tuning)
        tune.to_csv(OUTPUT/"inner_scores.csv", index=False)
        selection = tune.groupby(["outer", "model", "penalty"])["nll"].mean().reset_index()
        selected = selection.sort_values(["nll", "penalty"], ascending=[True, False]).groupby(["outer", "model"]).head(1)
        selected = selected.sort_values(["outer", "model"])
        selection.to_csv(OUTPUT/"candidate_scores.csv", index=False)
        selected.to_csv(OUTPUT/"selected_settings.csv", index=False)
        json_write(OUTPUT/"selection_freeze.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": digest(OUTPUT/"protocol.json"), "selected_sha256": digest(OUTPUT/"selected_settings.csv"),
            "outer_outcomes_scored": False})
        for fold in source_protocol["folds"]:
            if fold["phase"] != "refit":
                continue
            prior = features[fold["id"]]["prior_logit"].astype(np.float64)
            training = np.isin(qids, fold["train_questions"])
            for kind in KINDS:
                penalty = float(selected[(selected.outer == fold["outer"]) & (selected.model == kind)].iloc[0].penalty)
                x = design(kind, prior[training], current[training], initialization[training])
                xfit, _, mean, scale = standardize(x, x[:1])
                w, details = fit(xfit, prior[training].ravel(), y[training].ravel(), penalty)
                fitted.append({"fold": fold["id"], "outer": fold["outer"], "model": kind, "penalty": penalty,
                    "weights": w.tolist(), "mean": mean.tolist(), "scale": scale.tolist(), **details})
        json_write(OUTPUT/"refits.json", fitted)
        json_write(OUTPUT/"refit_freeze.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
            "refits_sha256": digest(OUTPUT/"refits.json"), "outer_outcomes_scored": False})
        rows, saved = [], {kind: np.full_like(y, np.nan) for kind in KINDS}
        for record in fitted:
            fold = next(f for f in source_protocol["folds"] if f["id"] == record["fold"])
            assess = np.isin(qids, fold["assessment_questions"])
            prior = features[fold["id"]]["prior_logit"].astype(np.float64)
            x = design(record["model"], prior[assess], current[assess], initialization[assess])
            x = (x-np.array(record["mean"]))/np.array(record["scale"])
            z = (prior[assess].ravel()+x@np.array(record["weights"])).reshape(y[assess].shape)
            saved[record["model"]][assess] = z
            for q in fold["assessment_questions"]:
                mask = qids[assess] == q
                rows.append({"outer": fold["outer"], "model": record["model"], "qid": str(q),
                             **score_metrics(z[mask], y[assess][mask])})
    assert all(np.isfinite(z).all() for z in saved.values())
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT/"by_question.csv", index=False)
    summary = table.groupby("model")[["error", "nll", "brier_two_class"]].mean().reset_index()
    summary.to_csv(OUTPUT/"summary.csv", index=False)
    np.savez_compressed(OUTPUT/"predictions.npz", target=y, **saved)
    for path, sha in sources.items():
        assert digest(path) == sha, path
    json_write(OUTPUT/"audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": sources, "core_sha256": code_digest(), "inner_fits": len(tuning), "outer_fits": len(fitted),
        "initial_majorities_verified": True, "no_future_votes_in_features": True,
        "no_same_question_outcomes_in_fitting": True, "fresh_confirmatory": False,
        "source_calls": 0, "official_test_scoring": False})
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
