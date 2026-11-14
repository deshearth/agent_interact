"""Separate, resumable 320/80 training; never load the reserved assessment episodes.

Reuse the frozen architecture, optimizer loop, encoder and author Ising fitter.
Keep both original scientific code and original 80-question predictors immutable.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, config as load_config, digest, json_write
from mac_agents.models.baselines import baseline_logp, fit_baseline
from scripts import breadth_expand as collection
from scripts import breadth_train as old
from scripts import transformer_generalization as base

CONFIG = Path("configs/breadth_expand_train.yaml")
GRAPH_FILE = Path("data/raw/upstream/data/J_matrices.json")
METRICS = ("nll", "error", "brier_two_class")


def now():
    return datetime.now(timezone.utc).isoformat()


def config():
    return load_config(CONFIG)


def read(path):
    return json.loads(Path(path).read_text())


def verify_prior(cfg):
    parent = Path(cfg["parent_root"])
    frozen = read(parent / "models_frozen.json")
    for path, sha in frozen["artifacts"].items():
        assert digest(path) == sha, path
    return frozen


def verify_source(cfg):
    root = Path(cfg["data_root"])
    collection.verify(root)
    audit = read(root / "collection_complete.json")
    for key, name in (("index_sha256", "episode_index.json"),
                      ("selection_sha256", "selection.json"),
                      ("protocol_sha256", "source_protocol.json")):
        assert digest(root / name) == audit[key], name
    assert audit["counts"] == {k: cfg["counts"][k] for k in ("train", "validation")}
    assert not audit["assessment_generated_or_scored"]
    verify_prior(cfg)


def load_episodes(cfg, questions):
    """Check each episode against the frozen index and its own source generation.

    Retained episodes correctly retain the old source-protocol hash. Copying them
    into the new directory must not rewrite provenance or relabel their origin.
    """
    root, parent = Path(cfg["data_root"]), Path(cfg["parent_root"])
    index = read(root / "episode_index.json")
    assert len(index) == len(questions) == sum(cfg["counts"][s] for s in ("train", "validation"))
    rows = {r["qid"]: r for r in index}
    assert len(rows) == len(index) and set(rows) == {q["qid"] for q in questions}
    assert Counter(r["split"] for r in index) == {s: cfg["counts"][s] for s in ("train", "validation")}
    assert Counter(r["origin"] for r in index) == {s: cfg["counts"][s] for s in ("new", "retained")}
    hashes = {"new": digest(root / "source_protocol.json"),
              "retained": digest(parent / "source_protocol.json")}
    graphs = {g["id"]: np.asarray(g["J"], dtype=np.int8) for g in read(GRAPH_FILE)["graphs"]}
    spins, matrices, votes, qids, origins, sources = [], [], [], [], [], {}
    for q in questions:
        assert q["split"] in ("train", "validation"), "Assessment excluded"
        row = rows[q["qid"]]
        assert all(row[k] == q[k] for k in ("qid", "split", "graph_id", "origin"))
        expected = root / "episodes" / q["split"] / f"{q['qid']}__{q['graph_id']}__r00.json"
        assert Path(row["path"]) == expected
        assert digest(expected) == row["sha256"], str(expected)
        assert row["source_protocol_sha256"] == hashes[q["origin"]]
        episode = read(expected)
        assert all(episode["meta"][k] == q[k] for k in ("qid", "split", "graph_id"))
        assert episode["meta"]["protocol_sha256"] == hashes[q["origin"]]
        assert episode["statement"] == q["question"] and episode["choices"] == q["choices"]
        # Validate the uncast values first so a corrupt noninteger cannot be
        # silently truncated into a legitimate spin by int8 conversion.
        s = np.asarray(episode["spins_history"])
        raw = np.asarray(episode["spins_raw_history"])
        graph = np.asarray(episode["J"])
        assert s.shape == (9, 32) and raw.shape == (9, 32, 5) and graph.shape == (32, 32)
        assert np.isin(s, [-1, 1]).all() and np.isin(raw, [-1, 0, 1]).all()
        assert np.isin(graph, [-1, 0, 1]).all() and np.array_equal(graph, graph.T) and not np.diag(graph).any()
        np.testing.assert_array_equal(graph, graphs[q["graph_id"]])
        total = raw.sum(-1)
        derived = np.where(total > 0, 1, np.where(total < 0, -1, raw[..., 0]))
        np.testing.assert_array_equal(s, derived)
        assert row["invalid_votes"] == int((raw == 0).sum()) and row["valid_snapshots"] == 8
        spins.append(s.astype(np.int8)); votes.append(raw.astype(np.int8)); matrices.append(graph.astype(np.int8))
        qids.append(q["qid"]); origins.append(q["origin"]); sources[str(expected)] = row["sha256"]
    spins = np.array(spins)
    valid = ((spins[:, :-1] != 0) & (spins[:, 1:] != 0)).all(-1)
    assert valid.all()
    pairs = np.array([(e, t) for e in range(len(questions)) for t in range(8)], dtype=np.int64)
    return {"spins": spins, "J": np.array(matrices), "raw_votes": np.array(votes),
            "valid": valid, "pairs": pairs, "episode_qids": np.array(qids),
            "episode_origins": np.array(origins)}, sources


def vectors_for(cfg, questions, device):
    parent = Path(cfg["parent_root"]) / "predictors"
    common = dict(np.load(parent / "development.npz", allow_pickle=False))
    vectors = dict(np.load(parent / "context_vectors.npz", allow_pickle=False))
    cached = dict(zip(common["episode_qids"], vectors["question_vectors"]))
    retained = [q for q in questions if q["origin"] == "retained"]
    added = [q for q in questions if q["origin"] == "new"]
    assert set(cached) == {q["qid"] for q in retained}
    old_questions = {q["qid"]: q for q in read(Path(cfg["parent_root"]) / "selection.json")["questions"]}
    for q in retained:
        assert old.question_text(q) == old.question_text(old_questions[q["qid"]])
    new_vectors, encoding = old.encode_texts([old.question_text(q) for q in added], device)
    cached.update({q["qid"]: vec for q, vec in zip(added, new_vectors)})
    encoding.update(cached_question_vectors=len(retained), cached_persona_vectors=32,
                    total_question_vectors=len(questions), encoder_finetuned=False,
                    cache_policy="Byte-identical original vectors for retained questions and personas; same frozen encoder/windowing for added questions.")
    return np.stack([cached[q["qid"]] for q in questions]), vectors["persona_vectors"], encoding


def validate_prepared(cfg):
    root = Path(cfg["output"])
    protocol = read(root / "protocol.json")
    assert cfg == protocol["config"] and digest(CONFIG) == protocol["config_sha256"]
    assert code_digest() == protocol["core_sha256"] == base.CORE_SHA
    for key in ("code_files", "source_artifacts", "encoder_files", "prior_artifacts"):
        for path, sha in protocol[key].items():
            assert digest(path) == sha, path
    prepared = read(root / "prepared.json")
    assert digest(root / "protocol.json") == prepared["protocol_sha256"]
    for path, sha in prepared["artifacts"].items():
        assert digest(root / path) == sha, path
    for path in (Path(cfg["data_root"]), Path(cfg["parent_root"])):
        assert not list((path / "episodes/assessment").glob("*.json"))
    return protocol


def prepare(cfg, device):
    root = Path(cfg["output"])
    if (root / "prepared.json").exists():
        validate_prepared(cfg)
        return
    verify_source(cfg)
    source = Path(cfg["data_root"])
    selected = read(source / "selection.json")["questions"]
    questions = [q for q in selected if q["split"] in ("train", "validation")]
    splits = {s: [q["qid"] for q in selected if q["split"] == s] for s in ("train", "validation", "assessment")}
    assert all(len(v) == len(set(v)) == cfg["counts"][s] for s, v in splits.items())
    assert len(set().union(*[set(v) for v in splits.values()])) == sum(map(len, splits.values()))
    common, sources = load_episodes(cfg, questions)
    qvectors, pvectors, encoding = vectors_for(cfg, questions, device)
    fitted = base.fit_context(dict(zip(common["episode_qids"], qvectors)), pvectors, splits["train"], common["episode_qids"])
    np.testing.assert_allclose(fitted["context"], old.apply_context(qvectors, pvectors, fitted), atol=1e-7)
    dataset = base.fold_dataset(common, fitted["context"], splits["train"])
    with threadpool_limits(limits=cfg["training"]["torch_threads"]):
        baseline = fit_baseline("I0", dataset, load_config("configs/paper_binary_inductive.yaml"))
    assert baseline["parameters"] == 19
    e, t = common["pairs"].T
    lp = baseline_logp(baseline, common["spins"][e, t], fitted["context"][e], common["J"][e])
    fitted["prior_logit"] = lp[..., 1] - lp[..., 0]
    fold = {"id": "main", "outer": 0, "phase": "development", **{s + "_questions": qids for s, qids in splits.items()}}
    for name in ("selection.json", "episode_index.json", "source_protocol.json", "collection_complete.json"):
        sources[str(source / name)] = digest(source / name)
    parent = Path(cfg["parent_root"])
    frozen = verify_prior(cfg)
    prior = {**frozen["artifacts"], str(parent / "models_frozen.json"): digest(parent / "models_frozen.json")}
    protocol = {
        "created_utc": now(), "config": cfg, "config_sha256": digest(CONFIG),
        "core_sha256": code_digest(), "code_files": {str(p): digest(p) for p in [Path(__file__), Path(base.__file__), Path(old.__file__)]},
        "source_artifacts": sources, "prior_artifacts": prior,
        "encoder_files": {str(p): digest(p) for p in old.ENCODER.iterdir() if p.is_file()},
        "fold": fold, "candidates": base.candidates(cfg),
        "parameters": {"nano": sum(p.numel() for p in base.make_model(cfg, "nano").parameters()), "I0": 19},
        "selection_policy": "Nine LR/penalty candidates at seed42; choose raw equal-question validation NLL and checkpoint duration. Freeze before three fresh-seed refits on 320 training questions only.",
        "budget_policy": cfg["training"]["budget_policy"],
        "encoder_policy": "Frozen shared MiniLM. Reuse original 100 question and 32 persona vectors; encode 300 new question/choice texts with complete token windows. Refit question PCA3 on 320 training questions only; known-persona PCA3. No arguments, solutions, truth labels or screening statistics are predictor inputs.",
        "assessment_policy": "Do not generate, load or score any of the 40 reserved assessment episodes.",
        "reference_policy": cfg["evaluation"]["reference_policy"],
        "uncertainty_policy": cfg["evaluation"]["uncertainty"],
        "source_llm_calls": 0, "assessment_loaded": False,
    }
    assert protocol["parameters"] == {"nano": 818, "I0": 19} and code_digest() == base.CORE_SHA
    json_write(root / "protocol.json", protocol)
    np.savez_compressed(root / "development.npz", **common)
    np.savez_compressed(root / "context_vectors.npz", question_vectors=qvectors, persona_vectors=pvectors)
    json_write(root / "encoding.json", encoding)
    out = root / "folds/main"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "features.npz", **fitted)
    json_write(out / "ising.json", baseline)
    json_write(out / "manifest.json", {**fold, "features_sha256": digest(out / "features.npz"),
        "ising_sha256": digest(out / "ising.json"), "protocol_sha256": digest(root / "protocol.json")})
    names = ["development.npz", "context_vectors.npz", "encoding.json", "folds/main/features.npz", "folds/main/ising.json", "folds/main/manifest.json"]
    json_write(root / "prepared.json", {"created_utc": now(), "protocol_sha256": digest(root / "protocol.json"),
        "artifacts": {name: digest(root / name) for name in names}})
    print(json.dumps({"prepared": True, "counts": cfg["counts"], "parameters": protocol["parameters"],
                      "agent_transitions": len(common["pairs"]) * 32, "assessment_loaded": False}), flush=True)


def validate_frozen(cfg):
    validate_prepared(cfg)
    frozen = read(Path(cfg["output"]).parent / "models_frozen.json")
    for path, sha in frozen["artifacts"].items():
        assert digest(path) == sha, path
    return frozen


def fit(cfg, device):
    protocol = validate_prepared(cfg)
    root = Path(cfg["output"])
    if (root.parent / "models_frozen.json").exists():
        validate_frozen(cfg)
        return
    common = dict(np.load(root / "development.npz", allow_pickle=False))
    features = dict(np.load(root / "folds/main/features.npz", allow_pickle=False))
    data = base.FoldData(common, features, device)
    fold = protocol["fold"]
    deadline = time.time() + 60 * cfg["training"]["aggregate_minutes"]
    tuned = []
    for candidate in protocol["candidates"]:
        item = {"phase": "tune", "fold": "main", "outer": 0, "model": "nano", "candidate": candidate["id"], "seed": cfg["training"]["tuning_seed"]}
        directory, record = base.train_one(cfg, protocol, data, fold, item, device, deadline)
        if record["status"] != "complete":
            raise SystemExit(75)
        tuned.append((directory, record))
    chosen = min(tuned, key=lambda x: (x[1]["validation"]["nll"], x[1]["candidate"]))[1]
    selection = {"protocol_sha256": digest(root / "protocol.json"), "candidate": chosen["candidate"],
                 "refit_steps": chosen["selected_step"], "validation_nll": chosen["validation"]["nll"],
                 "tuning_manifests": {str(p): digest(p / "manifest.json") for p, _ in tuned}}
    if (root / "selection.json").exists():
        previous = read(root / "selection.json")
        assert all(previous[k] == v for k, v in selection.items())
    else:
        json_write(root / "selection.json", {**selection, "frozen_utc": now()})
    refs = []
    for seed in cfg["training"]["refit_seeds"]:
        item = {"phase": "refit", "fold": "main", "outer": 0, "model": "nano", "candidate": chosen["candidate"], "seed": seed}
        directory, record = base.train_one(cfg, protocol, data, fold, item, device, deadline, selection["refit_steps"])
        if record["status"] != "complete":
            raise SystemExit(75)
        assert record["validation"] is None
        refs.append({"directory": str(directory), "seed": seed, "manifest_sha256": digest(directory / "manifest.json")})
    artifacts = {str(p): digest(p) for p in root.rglob("*") if p.is_file()}
    artifacts.update({str(CONFIG): digest(CONFIG), str(Path(__file__)): digest(__file__)})
    json_write(root.parent / "models_frozen.json", {"frozen_utc": now(), "training_questions": cfg["counts"]["train"],
        "validation_questions": cfg["counts"]["validation"], "assessment_outcomes_observed": False,
        "refits": refs, "neural_parameters": 818, "ising_coefficients": 19, "artifacts": artifacts})
    print(json.dumps({"models_frozen": True, "refits": len(refs), "assessment_loaded": False}), flush=True)


def collapse(rows, model, qids):
    grouped = {}
    for row in rows:
        if row["model"] == model and row["qid"] in qids:
            grouped.setdefault(row["qid"], []).append(row)
    assert set(grouped) == set(qids)
    return {q: {metric: float(np.mean([r[metric] for r in rs])) for metric in METRICS} for q, rs in sorted(grouped.items())}


def paired(rows, left, right, qids, cfg):
    a, b = collapse(rows, left, qids), collapse(rows, right, qids)
    keys = sorted(a)
    rng = np.random.default_rng(cfg["evaluation"]["bootstrap_seed"])
    draws = rng.integers(0, len(keys), size=(cfg["evaluation"]["bootstrap_replicates"], len(keys)))
    result = {}
    for metric in METRICS:
        delta = np.array([a[q][metric] - b[q][metric] for q in keys])
        lo, hi = np.quantile(delta[draws].mean(1), [.025, .975])
        result[metric] = {"difference": float(delta.mean()), "lower": float(lo), "upper": float(hi),
                          "questions_better": int((delta < 0).sum()), "questions": len(keys)}
    return result


def evaluate(cfg):
    """Report only development aggregates; retain machine-readable case scores."""
    root = Path(cfg["output"])
    frozen = validate_frozen(cfg)
    protocol, selected = read(root / "protocol.json"), read(root / "selection.json")
    common = dict(np.load(root / "development.npz", allow_pickle=False))
    features = dict(np.load(root / "folds/main/features.npz", allow_pickle=False))
    vectors = dict(np.load(root / "context_vectors.npz", allow_pickle=False))
    data = base.FoldData(common, features, "cpu")
    fold = protocol["fold"]
    ids = {s: data.ids(fold[s + "_questions"]) for s in ("train", "validation")}
    assert not set(data.qids) & set(fold["assessment_questions"])
    rows, by_seed, reproduction, tuning = [], [], [], []
    baseline = read(root / "folds/main/ising.json")
    output = root.parent / "development_results"
    output.mkdir(parents=True, exist_ok=True)

    def score(lp, indices, model, seed, split):
        aggregate, details = base.metrics(lp, data.y[indices], data.qids[indices])
        rows.extend({"model": model, "seed": seed, "split": split, **r} for r in details)
        np.save(output / f"{model}_{seed}_{split}_logp.npy", lp, allow_pickle=False)
        return aggregate

    for split, indices in ids.items():
        e, t = common["pairs"][indices].T
        lp = baseline_logp(baseline, common["spins"][e, t], features["context"][e], common["J"][e])
        score(lp, indices, "I0", None, split)
    for candidate in protocol["candidates"]:
        path = root / "tune/main" / f"nano-{candidate['id']}-s{cfg['training']['tuning_seed']}"
        record = read(path / "manifest.json")
        assert record["status"] == "complete"
        tuning.append({"candidate": candidate, **{k: record[k] for k in ("steps", "selected_step", "train", "validation", "stopping_reason", "seconds")}})
    best = min(tuning, key=lambda r: (r["validation"]["nll"], r["candidate"]["id"]))
    assert best["candidate"]["id"] == selected["candidate"] and best["selected_step"] == selected["refit_steps"]
    assert [r["seed"] for r in frozen["refits"]] == cfg["training"]["refit_seeds"]
    runs = [{"directory": str(root / "tune/main" / f"nano-{selected['candidate']}-s{cfg['training']['tuning_seed']}"),
             "seed": cfg["training"]["tuning_seed"], "role": "tuning"}]
    runs += [{**r, "role": "refit"} for r in frozen["refits"]]
    for run in runs:
        directory = Path(run["directory"])
        record = read(directory / "manifest.json")
        model = base.make_model(cfg, "nano")
        model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=False)["model"])
        assert base.model_weights_sha(model) == record["selected_weights_sha256"]
        entry = {**run, "selected_step": record["selected_step"]}
        for split, indices in ids.items():
            lp = base.predictions(model, data, indices, False, cfg["training"]["evaluation_batch_size"])
            entry[split] = score(lp, indices, "nano" if run["role"] == "refit" else "tuned", run["seed"], split)
            saved = directory / f"{split}_logp.npy"
            if saved.exists():
                expected = np.load(saved, allow_pickle=False)
                np.testing.assert_allclose(np.exp(lp), np.exp(expected), atol=5e-5, rtol=5e-5)
                reproduction.append({"model": "nano", "seed": run["seed"], "split": split,
                    "max_probability_difference": float(np.abs(np.exp(lp) - np.exp(expected)).max()),
                    "class_disagreements": int((lp.argmax(-1) != expected.argmax(-1)).sum())})
        by_seed.append(entry)

    # Old models receive their ORIGINAL PCA; neither weights nor transforms are refitted.
    parent = Path(cfg["parent_root"])
    old_features = dict(np.load(parent / "predictors/folds/main/features.npz", allow_pickle=False))
    old_context = old.apply_context(vectors["question_vectors"], vectors["persona_vectors"], old_features)
    old_ising = read(parent / "predictors/folds/main/ising.json")
    e, t = common["pairs"].T
    old_lp = baseline_logp(old_ising, common["spins"][e, t], old_context[e], common["J"][e])
    old_data = base.FoldData(common, {"context": old_context, "prior_logit": old_lp[..., 1] - old_lp[..., 0]}, "cpu")
    score(old_lp[ids["validation"]], ids["validation"], "old_I0", None, "validation")
    old_cfg = read(parent / "predictors/protocol.json")["config"]
    old_common = dict(np.load(parent / "predictors/development.npz", allow_pickle=False))
    lookup = {q: i for i, q in enumerate(common["episode_qids"])}
    reorder = [lookup[q] for q in old_common["episode_qids"]]
    np.testing.assert_allclose(old_context[reorder], old_features["context"], atol=1e-7)
    for run in verify_prior(cfg)["refits"]:
        directory = Path(run["directory"])
        record = read(directory / "manifest.json")
        model = base.make_model(old_cfg, "nano")
        model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=False)["model"])
        assert base.model_weights_sha(model) == record["selected_weights_sha256"]
        lp = base.predictions(model, old_data, ids["validation"], False, cfg["training"]["evaluation_batch_size"])
        aggregate = score(lp, ids["validation"], "old_nano", run["seed"], "validation")
        by_seed.append({**run, "role": "old_reference", "validation": aggregate})

    origins = dict(zip(common["episode_qids"], common["episode_origins"]))
    groups = {"train": fold["train_questions"], "validation": fold["validation_questions"],
              "retained_validation": [q for q in fold["validation_questions"] if origins[q] == "retained"],
              "new_validation": [q for q in fold["validation_questions"] if origins[q] == "new"]}
    summary = {}
    comparisons = {}
    for group, qids in groups.items():
        models = ["I0", "nano"] if group == "train" else ["I0", "nano", "old_I0", "old_nano"]
        summary[group] = {}
        for model in models:
            collapsed = collapse(rows, model, qids)
            summary[group][model] = {k: float(np.mean([r[k] for r in collapsed.values()])) for k in METRICS}
            summary[group][model].update(questions=len(qids), transitions=256 * len(qids))
        if group != "train":
            comparisons[group] = {label: paired(rows, left, right, qids, cfg) for label, left, right in
                [("nano_minus_I0", "nano", "I0"), ("nano_320_minus_80", "nano", "old_nano"), ("I0_320_minus_80", "I0", "old_I0")]}
    result = {"created_utc": now(), "protocol_sha256": digest(root / "protocol.json"),
              "models_frozen_sha256": digest(root.parent / "models_frozen.json"),
              "scope": "Development validation, used for selection; not untouched assessment. Old-vs-new comparisons include refitting PCA and retuning, not only a change in sample count.",
              "target": "Next agent majority-of-five opinion, not correctness against the math answer.",
              "aggregation": "Average the three individual-seed scores within question, then weight questions equally. No ensemble.",
              "parameters": protocol["parameters"], "shared_frozen_encoder": read(root / "encoding.json"),
              "selected_candidate": best["candidate"], "refit_steps": selected["refit_steps"],
              "summary": summary, "paired_differences": comparisons, "by_seed": by_seed, "tuning": tuning,
              "cpu_reproduction": reproduction, "invalid_individual_votes": int((common["raw_votes"] == 0).sum()),
              "assessment_loaded": False, "prior_artifacts_unchanged": True}
    json_write(output / "by_question.json", rows)
    result["by_question_sha256"] = digest(output / "by_question.json")
    json_write(output / "summary.json", result)
    validate_frozen(cfg)
    print(json.dumps({"summary": summary, "paired_differences": comparisons["validation"],
                      "selected_candidate": best["candidate"], "refit_steps": selected["refit_steps"],
                      "output": str(output / "summary.json")}, indent=2), flush=True)
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "fit", "evaluate", "all"])
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = parser.parse_args()
    cfg = config()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Mac GPU unavailable; choose CPU explicitly or grant local GPU access")
    if args.stage in ("prepare", "all"):
        prepare(cfg, args.device)
    if args.stage in ("fit", "all"):
        fit(cfg, args.device)
    if args.stage in ("evaluate", "all"):
        evaluate(cfg)


if __name__ == "__main__":
    main()
