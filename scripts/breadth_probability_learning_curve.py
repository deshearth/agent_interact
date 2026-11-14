"""Fixed-procedure Llama question-count curve and probability-target diagnostic.

This is a separate development experiment. No original predictor artifacts are
edited, no assessment outcomes are read, and no source LLM is called.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, device_for, digest, environment, json_write
from mac_agents.models.baselines import author_module, baseline_logp, design
from scripts import breadth_train as study
from scripts import transformer_generalization as base
from scripts.diagnose_breadth_probabilities import majority_probability, probability_metrics

CONFIG = Path("configs/breadth_probability_learning_curve.yaml")
ROOT = Path("runs/llama_probability_curve_v1")
SOURCE = study.ROOT
PROBABILITIES = SOURCE / "error_diagnosis_v1/development_source_probabilities_20261110.json.gz"
PROBABILITY_SHA = "b7c6fd634be77e6fd2a86eb80783066bce3f57f35970e1f9216493dc34a1c0f5"


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_originals():
    protocol = study.validate_prepared()
    frozen = json.loads((SOURCE / "models_frozen.json").read_text())
    for path, sha in frozen["artifacts"].items():
        assert digest(path) == sha, path
    audit = json.loads((SOURCE / "error_diagnosis_v1/completion_audit.json").read_text())
    for path, sha in {**audit["artifacts"], **audit["scripts"]}.items():
        assert digest(path) == sha, path
    assert not list((SOURCE / "episodes/assessment").glob("*.json"))
    assert not frozen["assessment_outcomes_observed"]
    assert digest(PROBABILITIES) == PROBABILITY_SHA
    return protocol


def balanced_subsets(questions, counts, seeds):
    """Seeded nested selections; constraints use metadata, never agent outcomes.

    Exactly half the mathematical answers are A at each size. Topic and graph
    counts lie at the floor or ceiling of their full-training-pool proportions.
    A random linear objective chooses among feasible subsets, conditional on
    the smaller subset. This is stratified sampling, not uniform sampling.
    """
    questions = [q for q in questions if q["split"] == "train"]
    qids = [q["qid"] for q in questions]
    assert len(qids) == len(set(qids)) == max(counts) == 80
    assert counts == sorted(set(counts)) and all(n % 2 == 0 for n in counts)
    assert Counter(q["answer"] for q in questions) == {"A": 40, "B": 40}
    rows = [np.ones(80), np.array([q["answer"] == "A" for q in questions])]
    for key in ("type", "graph_id"):
        rows += [np.array([q[key] == value for q in questions])
                 for value in sorted({q[key] for q in questions})]
    matrix = np.array(rows, dtype=float)
    result = []
    for path, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        selected = np.zeros(80)
        for n in counts[:-1]:
            expected = matrix.sum(1) * n / 80
            lower, upper = np.floor(expected), np.ceil(expected)
            lower[:2] = upper[:2] = [n, n / 2]
            solved = milp(rng.normal(size=80), integrality=np.ones(80),
                          bounds=Bounds(selected, np.ones(80)),
                          constraints=LinearConstraint(matrix, lower, upper),
                          options={"time_limit": 30., "mip_rel_gap": 0.})
            if not solved.success:
                raise RuntimeError(f"Infeasible subset path {path}, n={n}: {solved.message}")
            chosen = np.rint(solved.x)
            assert np.all(chosen >= selected) and chosen.sum() == n
            assert np.all(matrix @ chosen >= lower) and np.all(matrix @ chosen <= upper)
            selected = chosen
            selected_questions = [q for q, keep in zip(questions, chosen) if keep]
            result.append({"id": f"p{path}_n{n:02d}", "path": path, "subset_seed": seed,
                           "n_questions": n, "questions": [q["qid"] for q in selected_questions],
                           "strata": {key: dict(Counter(q[key] for q in selected_questions))
                                      for key in ("type", "answer", "graph_id", "level")}})
    result.insert(0, {"id": "all_n80", "path": None, "subset_seed": None,
                      "n_questions": 80, "questions": qids,
                      "strata": {key: dict(Counter(q[key] for q in questions))
                                 for key in ("type", "answer", "graph_id", "level")}})
    return result


def source_targets(common, records):
    """Map source probabilities to next-round labels; not to input features."""
    lookup = {str(q): i for i, q in enumerate(common["episode_qids"])}
    p = np.full(common["spins"].shape, np.nan)
    for row in records:
        if row["qid"] not in lookup or row["split"] not in ("train", "validation"):
            raise ValueError("Probability record is outside the development split")
        key = lookup[row["qid"]], row["round"], row["agent"]
        assert np.isnan(p[key]), "Duplicate probability record"
        np.testing.assert_array_equal(row["raw_votes"], common["raw_votes"][key])
        a, b = (np.asarray(row[k], dtype=float) for k in
                ("first_literal_A_probabilities", "first_literal_B_probabilities"))
        assert a.shape == b.shape == (5,) and np.ptp(a) == np.ptp(b) == 0
        assert np.isfinite(a).all() and np.isfinite(b).all() and a[0] + b[0] > 0
        p[key] = a[0] / (a[0] + b[0])
    assert np.isfinite(p).all(), "Missing probability records"
    e, t = common["pairs"].T
    return {"single_vote_probability": p,
            "soft": majority_probability(p[e, t + 1]),
            "hard": (common["spins"][e, t + 1] == 1).astype(np.float64)}


class TargetData(base.FoldData):
    def __init__(self, common, features, device, targets):
        super().__init__(common, features, device)
        targets = np.asarray(targets)
        assert targets.shape == self.y.shape and np.isfinite(targets).all()
        assert ((targets >= 0) & (targets <= 1)).all()
        # self.y remains the observed majority outcome used by the base scorer.
        self.values["target"] = torch.as_tensor(targets, dtype=torch.float32, device=device)


def fit_ising(common, features, questions, target, author):
    e, t = common["pairs"].T
    mask = np.isin(common["episode_qids"][e], questions)
    x = design(features["context"][e[mask]], common["J"][e[mask]],
               common["spins"][e[mask], t[mask]]).reshape(-1, 19).astype(np.float64)
    y = target[mask].ravel()
    # The pinned author's BCE/gradient accepts fractional Bernoulli targets.
    weights = author.two_stage_fit(x[:, :16], x[:, 16:18], x[:, 18],
                                   np.ones(len(y), dtype=bool), y)
    return {"kind": "I0", "parameters": 19, "weights": weights.tolist(),
            "adapter": "Unmodified pinned two_stage_fit, hard or fractional BCE targets"}


def score(logp, y, q, qids):
    aggregate, rows = base.metrics(logp, y, qids)
    for row in rows:
        mask = qids == row["qid"]
        row.update(probability_metrics(logp[mask], y[mask], q[mask]))
        row["accuracy"] = 1 - row["error"]
    for key in rows[0]:
        if key not in ("qid", "transitions"):
            aggregate[key] = float(np.mean([row[key] for row in rows]))
    return aggregate, rows


def validate():
    verify_originals()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert protocol["config_sha256"] == digest(CONFIG)
    assert protocol["implementation_sha256"] == digest(__file__)
    assert protocol["base_sha256"] == digest(base.__file__)
    for path, sha in protocol["inputs"].items():
        assert digest(path) == sha, path
    prepared = json.loads((ROOT / "prepared.json").read_text())
    assert prepared["protocol_sha256"] == digest(ROOT / "protocol.json")
    for path, sha in prepared["artifacts"].items():
        assert digest(ROOT / path) == sha, path
    return protocol


def prepare():
    if (ROOT / "prepared.json").exists():
        validate()
        print(json.dumps({"prepared": True, "reused": True}), flush=True)
        return
    cfg = config(CONFIG)
    original = verify_originals()
    assert cfg["output"] == str(ROOT) and cfg["source"] == str(SOURCE)
    main = original["fold"]
    common = dict(np.load(SOURCE / "predictors/development.npz", allow_pickle=False))
    vectors = dict(np.load(SOURCE / "predictors/context_vectors.npz", allow_pickle=False))
    questions = json.loads((SOURCE / "selection.json").read_text())["questions"]
    subsets = balanced_subsets(questions, cfg["curve"]["training_question_counts"], cfg["curve"]["subset_seeds"])
    assert subsets[0]["questions"] == main["train_questions"]
    for subset in subsets:
        assert set(subset["questions"]) <= set(main["train_questions"])
    with gzip.open(PROBABILITIES, "rt") as stream:
        raw = json.load(stream)
    assert raw["source_protocol_sha256"] == digest(SOURCE / "source_protocol.json")
    assert raw["selection_sha256"] == digest(SOURCE / "selection.json")
    targets = source_targets(common, raw["rows"])
    selected = json.loads((SOURCE / "predictors/selection.json").read_text())
    candidate = next(c for c in original["candidates"] if c["id"] == selected["candidate"])
    assert candidate["learning_rate"] == cfg["training"]["learning_rate"]
    assert candidate["penalty"] == cfg["training"]["output_penalty"]
    assert selected["refit_steps"] == cfg["training"]["fixed_steps"]
    assert sum(p.numel() for p in base.make_model(cfg, "nano").parameters()) == 818
    inputs = [SOURCE / "models_frozen.json", SOURCE / "selection.json", PROBABILITIES,
              SOURCE / "error_diagnosis_v1/completion_audit.json",
              SOURCE / "predictors/protocol.json", SOURCE / "predictors/selection.json",
              SOURCE / "predictors/development.npz", SOURCE / "predictors/context_vectors.npz"]
    folds = [{"id": f"{s['id']}_{condition}", "outer": s["path"], "phase": "refit",
              "subset_id": s["id"], "n_questions": s["n_questions"], "condition": condition,
              "train_questions": s["questions"], "validation_questions": main["validation_questions"],
              "assessment_questions": main["assessment_questions"]}
             for s in subsets for condition in cfg["curve"]["targets"]]
    protocol = {"created_utc": now(), "config": cfg, "config_sha256": digest(CONFIG),
        "implementation_sha256": digest(__file__), "base_sha256": digest(base.__file__),
        "inputs": {str(p): digest(p) for p in inputs}, "subsets": subsets, "folds": folds,
        "candidates": [candidate], "parameters": {"nano": 818, "I0": 19},
        "design": "Three metadata-balanced nested subsets of 10/20/40 questions, shared full-80 endpoint. All retained questions contribute eight snapshots and 32 agents per snapshot. Same 20 development validation questions throughout.",
        "optimization": "Fixed procedure from the original full-80 selection: lr=.001, logit penalty=.001, 1280 optimizer steps, seeds142/143/144. Identical settings, initialization policy and batch order across hard/soft targets. No new validation tuning. This does not estimate the best achievable performance at each sample size.",
        "preprocessing": "Question PCA3 refitted using each training subset only; frozen MiniLM embeddings and same persona PCA. No unused-training or validation embeddings enter PCA fitting. Same 16 input context features and signed graph.",
        "soft_targets": "Conditional majority-A probability from five independent votes: p^3(10-15p+6p^2), with p=P(first bare A)/(P(first bare A)+P(first bare B)). Labels use the NEXT round, never input features. No arguments or source probabilities are added to the input.",
        "probability_limit": "First-token probabilities approximate parsed answers (12/144000 responses not single bare tokens). Probabilities condition on realized future messages, so averaging vote sampling does not remove hidden-message uncertainty.",
        "baseline": "Pinned paper two-stage Ising fitter, 19 coefficients, identical subset-specific context; hard and soft targets. The original fractional BCE gradient is unchanged.",
        "evaluation": "Question-weighted observed majority accuracy, observed NLL and source-probability expected error/NLL/MSE. Average scores, not predictions. Report subset and optimizer variability separately. Paired validation-question bootstrap is exploratory and conditions on chosen subsets; reused validation is not an untouched assessment.",
        "planned_neural_fits": len(folds)*len(cfg["training"]["refit_seeds"]),
        "planned_ising_fits": len(folds), "source_llm_calls": 0,
        "assessment_generated_or_scored": False}
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / "protocol.json").exists(), "Partial preparation exists; inspect before retry"
    json_write(ROOT / "protocol.json", protocol)
    json_write(ROOT / "selection.json", {"protocol_sha256": digest(ROOT / "protocol.json"),
        "candidate": candidate, "fixed_steps": cfg["training"]["fixed_steps"],
        "inherited_from": str(SOURCE / "predictors/selection.json"), "no_new_validation_selection": True})
    np.savez_compressed(ROOT / "targets.npz", **targets)
    qvec = dict(zip(common["episode_qids"], vectors["question_vectors"]))
    artifacts = {"targets.npz": digest(ROOT / "targets.npz"), "selection.json": digest(ROOT / "selection.json")}
    author = author_module(config("configs/paper_binary_inductive.yaml"))
    e, t = common["pairs"].T
    for subset in subsets:
        features = base.fit_context(qvec, vectors["persona_vectors"], subset["questions"], common["episode_qids"])
        features["prior_logit"] = np.zeros_like(targets["hard"])
        if subset["n_questions"] == 80:
            old = np.load(SOURCE / "predictors/folds/main/features.npz", allow_pickle=False)
            for key in features:
                if key != "prior_logit":
                    np.testing.assert_array_equal(features[key], old[key])
        directory = ROOT / "subsets" / subset["id"]
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "features.npz", **features)
        feature_path = str((directory / "features.npz").relative_to(ROOT))
        artifacts[feature_path] = digest(directory / "features.npz")
        for fold in (f for f in folds if f["subset_id"] == subset["id"]):
            out = ROOT / "folds" / fold["id"]
            out.mkdir(parents=True, exist_ok=True)
            with threadpool_limits(limits=cfg["training"]["torch_threads"]):
                baseline = fit_ising(common, features, fold["train_questions"], targets[fold["condition"]], author)
            if fold["id"] == "all_n80_hard":
                old = json.loads((SOURCE / "predictors/folds/main/ising.json").read_text())
                np.testing.assert_allclose(baseline["weights"], old["weights"], rtol=0, atol=1e-11)
            lp = baseline_logp(baseline, common["spins"][e,t], features["context"][e], common["J"][e])
            np.save(out / "ising_logp.npy", lp, allow_pickle=False)
            json_write(out / "ising.json", baseline)
            manifest = {**fold, "protocol_sha256": digest(ROOT / "protocol.json"),
                        "features": feature_path, "features_sha256": artifacts[feature_path],
                        "targets_sha256": artifacts["targets.npz"],
                        "ising_sha256": digest(out / "ising.json"), "ising_logp_sha256": digest(out / "ising_logp.npy")}
            json_write(out / "manifest.json", manifest)
            for name in ("ising.json", "ising_logp.npy", "manifest.json"):
                artifacts[str((out / name).relative_to(ROOT))] = digest(out / name)
        print(json.dumps({"prepared_subset": subset["id"], "training_questions": subset["n_questions"]}), flush=True)
    json_write(ROOT / "prepared.json", {"completed_utc": now(),
        "protocol_sha256": digest(ROOT / "protocol.json"), "artifacts": artifacts})
    validate()


def run(device):
    protocol = validate()
    cfg = protocol["config"]
    torch.set_num_threads(cfg["training"]["torch_threads"])
    common = dict(np.load(SOURCE / "predictors/development.npz", allow_pickle=False))
    targets = dict(np.load(ROOT / "targets.npz", allow_pickle=False))
    timer = time.monotonic()
    results, completion = [], {}
    qids = common["episode_qids"][common["pairs"][:,0]]
    for fold in protocol["folds"]:
        fm = json.loads((ROOT / "folds" / fold["id"] / "manifest.json").read_text())
        features = dict(np.load(ROOT / fm["features"], allow_pickle=False))
        data = TargetData(common, features, device, targets[fold["condition"]])
        for seed in cfg["training"]["refit_seeds"]:
            item = {"phase": "refit", "fold": fold["id"], "outer": fold["outer"],
                    "model": "nano", "candidate": protocol["candidates"][0]["id"], "seed": seed}
            out, manifest = base.train_one(cfg, protocol, data, fold, item, device, float("inf"),
                                           fixed_steps=cfg["training"]["fixed_steps"])
            assert manifest["status"] == "complete" and manifest["parameters"] == 818
            model = base.make_model(cfg, "nano").to(device)
            ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            logp = base.predictions(model, data, np.arange(len(qids)), False, cfg["training"]["evaluation_batch_size"])
            if fold["id"] == "all_n80_hard":
                old = np.load(SOURCE / "error_diagnosis_v1/predictions.npz", allow_pickle=False)
                reference = old["nano_logp"][cfg["training"]["refit_seeds"].index(seed)]
                difference = float(np.abs(np.exp(logp) - np.exp(reference)).max())
                assert difference < 2e-5, f"Full-80 hard reproduction differs: {difference}"
            else:
                difference = None
            result = {"fold": fold["id"], "subset_id": fold["subset_id"], "n_questions": fold["n_questions"],
                      "condition": fold["condition"], "model": "nano", "seed": seed,
                      "checkpoint": str(out / "best.pt"), "checkpoint_sha256": digest(out / "best.pt"),
                      "original_endpoint_max_probability_difference": difference, "metrics": {}, "by_question": {}}
            for split in ("train", "validation"):
                ids = data.ids(fold[f"{split}_questions"])
                aggregate, rows = score(logp[ids], data.y[ids], targets["soft"][ids], qids[ids])
                result["metrics"][split], result["by_question"][split] = aggregate, rows
            # Separate scoring files do not alter the base trainer's manifest.
            scored = ROOT / "scored" / fold["id"] / f"s{seed}"
            scored.mkdir(parents=True, exist_ok=True)
            np.save(scored / "logp.npy", logp, allow_pickle=False)
            result["logp_path"] = str(scored / "logp.npy")
            result["logp_sha256"] = digest(scored / "logp.npy")
            json_write(scored / "scores.json", result)
            completion[str(out / "manifest.json")] = digest(out / "manifest.json")
            results.append(result)
            json_write(ROOT / "run_progress.json", {"updated_utc": now(), "completed_neural_fits": len(results),
                "planned_neural_fits": protocol["planned_neural_fits"], "last": {k:result[k] for k in
                ("fold", "condition", "seed")}, "validation": result["metrics"]["validation"]})
            print(json.dumps({"scored": fold["id"], "seed": seed,
                "validation_accuracy": result["metrics"]["validation"]["accuracy"],
                "expected_nll": result["metrics"]["validation"]["expected_nll"]}), flush=True)
        # Paired seeds must have exactly identical initial weights across target conditions.
        if fold["condition"] == "soft":
            hard_fold = fold["id"].removesuffix("_soft") + "_hard"
            for seed in cfg["training"]["refit_seeds"]:
                name = f"nano-{protocol['candidates'][0]['id']}-s{seed}"
                a, b = (json.loads((ROOT / "refit" / f / name / "manifest.json").read_text())
                        for f in (hard_fold, fold["id"]))
                assert a["initial_weights_sha256"] == b["initial_weights_sha256"]
    assert len(results) == protocol["planned_neural_fits"]
    verify_originals()
    json_write(ROOT / "run_complete.json", {"completed_utc": now(), "seconds_this_session": time.monotonic()-timer,
        "protocol_sha256": digest(ROOT / "protocol.json"), "manifests": completion,
        "neural_fits": len(results), "source_llm_calls": 0, "assessment_generated_or_scored": False,
        "environment": environment()})


def main():
    parser = ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run", "validate"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "validate":
        validate()
        print("Protocol, inputs, frozen originals and preparation verified.")
    else:
        run(str(device_for(args.device)))


if __name__ == "__main__":
    main()
