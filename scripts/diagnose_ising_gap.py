"""Post-hoc error audit and disjoint-replica question adaptation diagnostic.

The source models are frozen nested fits on the existing development pool.
Adaptation intentionally uses other conversations about each assessment question;
it is a familiar-question diagnostic, never an unseen-question performance claim.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, digest, json_write

SOURCE = Path("runs/transformer_generalization_v1")
OUTPUT = Path("runs/ising_gap_diagnosis_v1")
MODELS = ("I0", "full", "small", "ising_small")
SEEDS = (142, 143, 144)


def fit_offsets(score, target, group, n_groups, penalty=1.0):
    """Penalized Bernoulli intercepts, with the base logit held fixed.

    Objective is SUM BCE + penalty/2 * SUM offset**2, not mean BCE.
    """
    score = np.asarray(score, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    group = np.asarray(group, dtype=np.int64).ravel()
    assert score.shape == target.shape == group.shape and penalty > 0
    assert np.isfinite(score).all() and np.isin(target, [0, 1]).all()
    assert group.min() >= 0 and group.max() < n_groups
    offsets = np.zeros(n_groups)
    for iteration in range(100):
        p = expit(score + offsets[group])
        gradient = np.bincount(group, weights=p-target, minlength=n_groups) + penalty*offsets
        hessian = np.bincount(group, weights=p*(1-p), minlength=n_groups) + penalty
        if np.max(np.abs(gradient)) < 1e-8:
            return offsets, iteration, float(np.max(np.abs(gradient)))
        offsets -= np.clip(gradient/hessian, -2.0, 2.0)
    raise RuntimeError("Intercept optimization did not converge")


def crossfit_offsets(score, target, group, replica, penalty=1.0):
    """Fit on replicas 0/1 and assess 2/3, then reverse, for every question."""
    assert score.shape == target.shape == group.shape == replica.shape
    assert set(np.unique(replica)) == {0, 1, 2, 3}
    n_groups = int(group.max()) + 1
    result, visited, fitted = np.empty_like(score), np.zeros_like(target, dtype=int), []
    for direction, fitting_replicas in enumerate(((0, 1), (2, 3))):
        fit = np.isin(replica, fitting_replicas)
        assess = ~fit
        assert set(np.unique(group[fit])) == set(np.unique(group[assess]))
        offsets, iterations, gradient = fit_offsets(score[fit], target[fit], group[fit], n_groups, penalty)
        result[assess] = score[assess] + offsets[group[assess]]
        visited[assess] += 1
        fitted.append({"direction": direction, "fit_replicas": list(fitting_replicas),
                       "iterations": iterations, "maximum_gradient": gradient,
                       "offsets": offsets.tolist()})
    assert (visited == 1).all()
    return result, fitted


def score_metrics(score, target):
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target)
    return {"error": float(((score > 0) != target).mean()),
            "nll": float((np.logaddexp(0, score) - target*score).mean()),
            "brier_two_class": float((2*(expit(score)-target)**2).mean())}


def overlap(a_wrong, b_wrong):
    assert a_wrong.shape == b_wrong.shape
    return {"both_correct": float((~a_wrong & ~b_wrong).mean()),
            "fixed_ising_error": float((a_wrong & ~b_wrong).mean()),
            "introduced_error": float((~a_wrong & b_wrong).mean()),
            "shared_error": float((a_wrong & b_wrong).mean())}


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((SOURCE/"completion_audit.json").read_text())
    used = {}

    def verify(path):
        path = Path(path)
        actual = digest(path)
        if str(path) in prior["artifacts"]:
            assert actual == prior["artifacts"][str(path)], path
        used[str(path)] = actual
        return path

    verify(SOURCE/"completion_audit.json")
    protocol = json.loads(verify(SOURCE/"protocol.json").read_text())
    common = dict(np.load(verify(SOURCE/"development.npz"), allow_pickle=False))
    index = json.loads(verify(SOURCE/"episode_index.json").read_text())
    assert all(x["official_split"] == "train" and x["graph_role"] == "fit_graph" for x in index)
    assert code_digest() == prior["core_sha256"]
    e, t = common["pairs"].T
    target = common["spins"][e, t+1] == 1
    current = common["spins"][e, t] == 1
    qids = common["episode_qids"][e]
    questions, qindex = np.unique(qids, return_inverse=True)
    assert len(questions) == 20 and target.size == 163840
    assert (np.bincount(qindex) == 256).all()  # Equal transition counts per question.
    replica = np.broadcast_to(np.array([index[i]["replica"] for i in e])[:, None], target.shape)
    groups = {"question_offset": np.broadcast_to(qindex[:, None], target.shape),
              "question_persona_offset": qindex[:, None]*32 + np.arange(32)[None, :]}
    scores = {k: np.full((3, *target.shape), np.nan) for k in MODELS}
    for fold in protocol["folds"]:
        if fold["phase"] != "refit":
            continue
        ids = np.flatnonzero(np.isin(qids, fold["assessment_questions"]))
        feature = np.load(verify(SOURCE/"folds"/fold["id"]/"features.npz"), allow_pickle=False)
        scores["I0"][:, ids] = feature["prior_logit"][ids]
        for directory in sorted((SOURCE/"refit"/fold["id"]).iterdir()):
            manifest = json.loads(verify(directory/"manifest.json").read_text())
            name = str(directory.relative_to(SOURCE)).replace("/", "__")
            logp = np.load(verify(SOURCE/"assessment"/name/"assessment_logp.npy"), allow_pickle=False)
            scores[manifest["model"]][SEEDS.index(manifest["seed"]), ids] = (
                logp[..., 1].astype(np.float64) - logp[..., 0].astype(np.float64))
    assert all(np.isfinite(x).all() for x in scores.values())
    source_index_path = Path("data/processed/train_only_pca/index.json")
    source_arrays_path = Path("data/processed/train_only_pca/episodes.npz")
    for path in (source_index_path, source_arrays_path):
        assert digest(verify(path)) == protocol["source"][str(path)]
    original_index = json.loads(source_index_path.read_text())
    lookup = {x["episode_id"]: i for i, x in enumerate(original_index)}
    original_ids = [lookup[x["episode_id"]] for x in index]
    original = np.load(source_arrays_path, allow_pickle=False)
    np.testing.assert_array_equal(original["spins"][original_ids], common["spins"])
    votes = original["votes"][original_ids][e, t+1]
    assert np.isin(votes, [-1, 1]).all()
    yes = (votes == 1).sum(-1)
    np.testing.assert_array_equal(yes >= 3, target)
    majority = np.maximum(yes, 5-yes)
    specification = {
        "script_sha256": digest(__file__), "source_artifacts": used,
        "scope": "Post-hoc development diagnosis; no fresh confirmatory assessment",
        "observational_error_and_vote_audit": True,
        "adaptation_models": ["I0", "small"],
        "adaptation_groups": ["question_offset", "question_persona_offset"],
        "adaptation_penalty": 1.0, "adaptation_objective": "sum BCE + 0.5 * sum(offset squared)",
        "adaptation_replica_partitions": [[0, 1], [2, 3]],
        "adaptation_assesses": "Other complete conversations about the SAME questions",
        "adaptation_new_question_generalization": False,
        "source_llm_calls": 0, "official_test_evaluations": 0,
        "seed_aggregation": "Average scores; never ensemble probabilities",
    }
    specification_path = OUTPUT/"protocol.json"
    if specification_path.exists():
        assert json.loads(specification_path.read_text())["specification"] == specification
    else:
        json_write(specification_path, {"created_utc": datetime.now(timezone.utc).isoformat(),
                                        "specification": specification})
    records, qrecords, overlaps, subgroups = [], [], [], []
    error = {kind: (value > 0) != target for kind, value in scores.items()}
    for kind, values in scores.items():
        seed_ids = (0,) if kind == "I0" else range(3)
        for seed_i in seed_ids:
            row = {"model": kind, "seed": -1 if kind == "I0" else SEEDS[seed_i], "condition": "original"}
            records.append({**row, **score_metrics(values[seed_i], target)})
            for qi, question in enumerate(questions):
                mask = qindex == qi
                qrecords.append({**row, "qid": str(question), **score_metrics(values[seed_i, mask], target[mask])})
            if kind != "I0":
                overlaps.append({**row, **overlap(error["I0"][0], error[kind][seed_i])})
            masks = [("target_votes", str(n), majority == n) for n in (3, 4, 5)]
            masks += [("transition", "switch" if switch else "stay", (current != target) == switch)
                      for switch in (False, True)]
            masks += [("round", str(round_i+1), np.broadcast_to((t == round_i)[:, None], target.shape))
                      for round_i in range(8)]
            masks += [("ising_confidence", label, mask) for label, mask in (
                ("below_0.8", np.abs(scores["I0"][0]) < np.log(4)),
                ("0.8_to_0.95", (np.abs(scores["I0"][0]) >= np.log(4)) & (np.abs(scores["I0"][0]) < np.log(19))),
                ("at_least_0.95", np.abs(scores["I0"][0]) >= np.log(19)))]
            for variable, label, mask in masks:
                assert mask.any()
                subgroups.append({**row, "variable": variable, "group": label,
                    "transitions": int(mask.sum()), "transition_share": float(mask.mean()),
                    "fraction_of_errors": float(error[kind][seed_i][mask].sum()/error[kind][seed_i].sum()),
                    **score_metrics(values[seed_i][mask], target[mask])})
    # Reproduce the previously reported scores before doing any new fitting.
    previous = pd.read_csv(verify(SOURCE/"summary.csv"))
    initial = pd.DataFrame(records).groupby("model")[["error", "nll", "brier_two_class"]].mean()
    for kind in MODELS:
        old = previous[(previous.model == kind) & (previous.split == "assessment")].iloc[0]
        for metric in ("error", "nll", "brier_two_class"):
            np.testing.assert_allclose(initial.loc[kind, metric], old[metric], atol=1e-7, rtol=1e-6)
    adaptation_fits, saved_scores = [], {}
    for kind in ("I0", "small"):
        for seed_i in ((0,) if kind == "I0" else range(3)):
            for condition, group in groups.items():
                adjusted, fits = crossfit_offsets(scores[kind][seed_i], target, group, replica, 1.0)
                row = {"model": kind, "seed": -1 if kind == "I0" else SEEDS[seed_i], "condition": condition}
                records.append({**row, **score_metrics(adjusted, target)})
                for qi, question in enumerate(questions):
                    mask = qindex == qi
                    qrecords.append({**row, "qid": str(question), **score_metrics(adjusted[mask], target[mask])})
                adaptation_fits.extend([{**row, **fit} for fit in fits])
                saved_scores[f"{kind}_{seed_i}_{condition}"] = adjusted.astype(np.float32)
    tables = {"metrics.csv": pd.DataFrame(records), "by_question.csv": pd.DataFrame(qrecords),
              "error_overlap.csv": pd.DataFrame(overlaps), "subgroups.csv": pd.DataFrame(subgroups)}
    for name, frame in tables.items():
        frame.to_csv(OUTPUT/name, index=False)
    summary = tables["metrics.csv"].groupby(["model", "condition"])[["error", "nll", "brier_two_class"]].mean().reset_index()
    summary.to_csv(OUTPUT/"summary.csv", index=False)
    json_write(OUTPUT/"adaptation_fits.json", adaptation_fits)
    np.savez_compressed(OUTPUT/"adaptation_predictions.npz", **saved_scores)
    np.savez_compressed(OUTPUT/"assessment_inputs.npz", target=target, replica=replica, qindex=qindex,
                        majority=majority, **{f"score_{k}": v for k, v in scores.items()})
    for path, sha in used.items():
        assert digest(path) == sha, path
    audit = {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "questions": 20, "transitions": int(target.size), "source_files_verified_unchanged": len(used),
        "original_metrics_reproduced": True, "target_votes_reconstruct_labels": True,
        "no_official_test_scoring": True, "no_source_llm_calls": True,
        "no_new_transformer_training": True, "adaptation_partition": "Whole replicas; two directions",
        "adaptation_is_not_unseen_question_evaluation": True,
        "source_artifacts": used, "core_sha256": code_digest()}
    json_write(OUTPUT/"audit.json", audit)
    print(summary.to_string(index=False))
    print(tables["error_overlap.csv"].groupby("model")[["fixed_ising_error", "introduced_error", "shared_error"]].mean().to_string())


if __name__ == "__main__":
    run()
