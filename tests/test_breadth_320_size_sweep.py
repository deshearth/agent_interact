"""Synthetic full-sweep checks; these scores are not research results."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mac_agents.common import code_digest, digest, json_write
from mac_agents.models.baselines import baseline_logp
from scripts import breadth_320_size_sweep as study
from scripts import report_breadth_320_size_sweep as report
from scripts import transformer_generalization as base


def test_family_counts_and_matching_protocol():
    cfg = study.config()
    counts = {k: sum(p.numel() for p in base.make_model(cfg, k).parameters()) for k in cfg["models"]}
    assert counts == {"pico": 314, "nano": 818, "tiny": 7370, "small": 34978, "full": 135458, "wide": 533026}
    assert all(m["heads"] == 4 for m in cfg["models"].values())
    ref = study.parent.read(Path(cfg["reference"]) / "protocol.json")
    study.check_compatibility(cfg, ref)
    changed = copy.deepcopy(cfg)
    changed["training"]["validation_every_steps"] //= 2
    with pytest.raises(AssertionError):
        study.check_compatibility(changed, ref)


def test_checkpoint_choice_is_independent_per_size():
    cfg = study.config()
    records = []
    for i, kind in enumerate(cfg["model_order"]):
        for j, candidate in enumerate(base.candidates(cfg)):
            records.append((Path(kind) / candidate["id"], {"model": kind, "candidate": candidate["id"],
                "initial_weights_sha256": kind, "selected_step": 100 + j,
                "validation": {"nll": 0.2 if j in (i, i + 1) else .3}}))
    choices = study.choose(cfg, records)
    assert [x["candidate"] for x in choices] == [f"c{i}" for i in range(6)]
    assert [x["refit_steps"] for x in choices] == [100 + i for i in range(6)]
    with pytest.raises(AssertionError):
        study.choose(cfg, records[:-1])


def test_question_metrics_do_not_mix_graph_regimes():
    cfg = study.config()
    cfg["model_order"] = ["nano", "small"]
    cfg["evaluation"]["bootstrap_replicates"] = 100
    qids = ["q0", "q1", "q2"]
    groups = {k: qids for k in ("validation", "graph_seen", "graph_unseen")}
    rows = []
    for split in groups:
        for model in ("I0", "nano", "small"):
            value = .4 if model == "I0" else .3 if model == "nano" else .6 if split == "graph_unseen" else .1
            for q in qids:
                for seed in ([-1] if model == "I0" else [142, 143, 144]):
                    rows.append({"split": split, "model": model, "qid": q, "seed": seed,
                                 **{m: value for m in report.METRICS}})
    actual = report.comparisons(cfg, rows, groups)
    assert actual["graph_unseen"]["small_minus_nano"]["error"]["difference"] == pytest.approx(.3)
    assert actual["validation"]["small_minus_nano"]["error"]["difference"] == pytest.approx(-.2)
    with pytest.raises(AssertionError):
        report.question_means(rows + [rows[-1]], "graph_unseen", "small", qids)


def synthetic_reference(tmp_path, cfg):
    import time
    root = Path(cfg["reference"])
    root.mkdir(parents=True)
    old_cfg = copy.deepcopy(cfg)
    old_cfg["output"] = str(root)
    old_cfg["models"] = {"nano": cfg["models"]["nano"]}
    qids = np.array([f"q{i:02d}" for i in range(12)])
    fold = {"id": "main", "outer": 0, "phase": "development", "train_questions": qids[:8].tolist(),
            "validation_questions": qids[8:].tolist(), "assessment_questions": ["a0", "a1"]}
    rng = np.random.default_rng(99)
    spins = np.where(rng.random((12, 9, 32)) < .85, 1, -1).astype(np.int8)
    graph = np.eye(32, k=1, dtype=np.int8) + np.eye(32, k=-1, dtype=np.int8)
    context = rng.normal(size=(12, 32, 16)).astype(np.float32)
    context[..., 0] = 1
    pairs = np.array([(e, t) for e in range(12) for t in range(8)])
    common = {"spins": spins, "J": np.tile(graph, (12, 1, 1)), "valid": np.ones((12, 8), bool),
              "episode_qids": qids, "pairs": pairs, "raw_votes": np.repeat(spins[..., None], 5, -1)}
    baseline = {"kind": "I0", "parameters": 19, "weights": [1.5] + [0.] * 18}
    e, t = pairs.T
    lp = baseline_logp(baseline, spins[e, t], context[e], common["J"][e])
    features = {"context": context, "prior_logit": lp[..., 1] - lp[..., 0]}
    protocol = {"core_sha256": code_digest(), "config": old_cfg, "fold": fold,
                "candidates": base.candidates(old_cfg), "source_artifacts": {}, "code_files": {}, "encoder_files": {}, "prior_artifacts": {}}
    json_write(root / "protocol.json", protocol)
    np.savez_compressed(root / "development.npz", **common)
    (root / "folds/main").mkdir(parents=True)
    np.savez_compressed(root / "folds/main/features.npz", **features)
    json_write(root / "folds/main/ising.json", baseline)
    json_write(root / "folds/main/manifest.json", fold)
    names = ("development.npz", "folds/main/features.npz", "folds/main/ising.json", "folds/main/manifest.json")
    json_write(root / "prepared.json", {"protocol_sha256": digest(root / "protocol.json"), "artifacts": {p: digest(root / p) for p in names}})
    data = base.FoldData(common, features, "cpu")
    tuned = []
    for c in protocol["candidates"]:
        item = {"phase": "tune", "fold": "main", "outer": 0, "model": "nano", "seed": 42, "candidate": c["id"]}
        directory, result = base.train_one(old_cfg, protocol, data, fold, item, "cpu", time.time() + 120)
        assert result["status"] == "complete"
        tuned.append(result)
    chosen = min(tuned, key=lambda r: (r["validation"]["nll"], r["candidate"]))
    json_write(root / "selection.json", {"candidate": chosen["candidate"], "refit_steps": chosen["selected_step"]})
    refs = []
    models = []
    for seed in old_cfg["training"]["refit_seeds"]:
        item = {"phase": "refit", "fold": "main", "outer": 0, "model": "nano", "seed": seed, "candidate": chosen["candidate"]}
        directory, result = base.train_one(old_cfg, protocol, data, fold, item, "cpu", time.time() + 120, chosen["selected_step"])
        refs.append({"seed": seed, "directory": str(directory)})
        model = base.make_model(old_cfg, "nano")
        model.load_state_dict(torch.load(directory / "best.pt", weights_only=False)["model"])
        models.append(model)
    json_write(root.parent / "models_frozen.json", {"refits": refs, "artifacts": {str(p): digest(p) for p in root.rglob("*") if p.is_file()}})
    json_write(root.parent / "completion_audit.json", {"synthetic": True})
    summary = {}
    for split in ("train", "validation"):
        ids = data.ids(fold[split + "_questions"])
        summary[split] = {"I0": base.metrics(lp[ids], data.y[ids], data.qids[ids])[0]}
        measures = [base.metrics(base.predictions(m, data, ids, False), data.y[ids], data.qids[ids])[0] for m in models]
        summary[split]["nano"] = {k: float(np.mean([r[k] for r in measures])) for k in report.METRICS}
    json_write(root.parent / "development_results/summary.json", {"summary": summary})
    json_write(root.parent / "development_results/by_question.json", [])
    return common, features, baseline, fold, data, models


def synthetic_graph_study(cfg, common, features, baseline, fold, data, models):
    root = Path(cfg["unseen_source"])
    n = 32
    first = common["J"][0].copy(); first[0, -1] = first[-1, 0] = 1
    second = np.zeros((n, n), np.int8); second[0, 1:] = second[1:, 0] = -1
    graphs = {"Jf0": first.tolist(), "Jf1": second.tolist()}
    qids = fold["validation_questions"]
    questions = [{"qid": q, "graph_id": f"Jf{i % 2}", "split": "validation", "seen_graph_id": "J0",
                  "question": f"Synthetic {q}", "choices": {"A": "1", "B": "2"}} for i, q in enumerate(qids)]
    protocol = {"questions": questions, "validation_question_ids": qids}
    json_write(root / "protocol.json", protocol)
    json_write(root / "source_protocol.json", protocol)
    json_write(root / "graphs.json", graphs)
    np.savez_compressed(root / "evaluation_inputs.npz", qids=np.array(qids), seen_spins=common["spins"][8:],
                        seen_graph=common["J"][8:], seen_votes=common["raw_votes"][8:])
    episode_hashes = {}
    for i, q in enumerate(questions):
        s = np.where(np.random.default_rng(700 + i).random((9, n)) < .85, 1, -1).astype(np.int8)
        s[0] = common["spins"][8 + i, 0]
        graph = graphs[q["graph_id"]]
        name = report.graph_study.episode_name(q)
        edges = {f"{a},{b}": "Synthetic" for a in range(n) for b in range(n) if graph[a][b]}
        path = root / "episodes/validation" / (name + ".json")
        json_write(path, {"meta": {"qid": q["qid"], "graph_id": q["graph_id"], "split": "validation", "name": name,
            "n": 32, "k": 5, "steps": 8, "replica": 0, "protocol_sha256": digest(root / "source_protocol.json")},
            "statement": q["question"], "choices": q["choices"], "J": graph, "spins_history": s.tolist(),
            "spins_raw_history": np.repeat(s[..., None], 5, -1).tolist(),
            "messages_history": [edges] * 8, "communicated_stances_history": [{}] * 8})
        episode_hashes[str(path)] = digest(path)
    np.savez_compressed(root / "predictions.npz", synthetic=np.ones(1))
    json_write(root / "by_question.json", []); json_write(root / "by_seed_question.json", [])
    json_write(root / "results.json", {"assessment_generated_or_scored": False, "scored_episode_sha256": episode_hashes,
        "output_sha256": {p: digest(root / p) for p in ("predictions.npz", "by_question.json", "by_seed_question.json")}})
    unseen, lp, _ = report.prepare_unseen(cfg, common, features, baseline)
    summary = []
    for regime, values, ids in (("seen", data, data.ids(qids)), ("unseen", unseen, np.arange(len(unseen.qids)))):
        if regime == "seen":
            e, t = common["pairs"][ids].T
            ilp = baseline_logp(baseline, common["spins"][e, t], features["context"][e], common["J"][e])
        else:
            ilp = lp
        im, _ = base.metrics(ilp, values.y[ids], values.qids[ids])
        summary.append({"model": "I0", "regime": regime, "training_questions": cfg["counts"]["train"], **im})
        ms = [base.metrics(base.predictions(m, values, ids, False), values.y[ids], values.qids[ids])[0] for m in models]
        summary.append({"model": "nano", "regime": regime, "training_questions": cfg["counts"]["train"],
                        **{k: float(np.mean([r[k] for r in ms])) for k in report.METRICS}})
    previous = Path(cfg["unseen_reference"])
    json_write(previous / "results.json", {"summary": summary})
    json_write(previous / "protocol.json", {"synthetic": True})
    json_write(previous / "completion_audit.json", {"synthetic": True})


def test_full_synthetic_sweep_reuses_freezes_and_evaluates_all_sizes(tmp_path, monkeypatch):
    cfg = copy.deepcopy(study.config())
    cfg.update(output=str(tmp_path / "sweep/predictors"), reference=str(tmp_path / "reference/predictors"),
               unseen_source=str(tmp_path / "graph_source"), unseen_reference=str(tmp_path / "graph_reference"),
               counts={"train": 8, "validation": 4, "assessment": 2})
    cfg["training"].update(max_steps=4, validation_every_steps=2, patience=99, min_delta=0.,
        batch_size=8, evaluation_batch_size=32, learning_rates=[.01, .03], output_penalties=[0.], aggregate_minutes=3, torch_threads=2)
    cfg["evaluation"].update(unseen_questions=4, unseen_graphs=2, bootstrap_replicates=50)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(study, "CONFIG", config_path)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        common, features, baseline, fold, data, models = synthetic_reference(tmp_path, cfg)
        synthetic_graph_study(cfg, common, features, baseline, fold, data, models)
        parent_before = digest(Path(cfg["reference"]).parent / "models_frozen.json")
        study.prepare(cfg)
        prepared = digest(Path(cfg["output"]) / "prepared.json")
        study.prepare(cfg)
        assert digest(Path(cfg["output"]) / "prepared.json") == prepared
        with pytest.raises(FileNotFoundError):
            report.evaluate(cfg)
        study.fit(cfg, "cpu")
        frozen = study.validate_frozen(cfg)
        assert len(frozen["refits"]) == 18 and sum(r["reused"] for r in frozen["refits"]) == 3
        assert parent_before == digest(Path(cfg["reference"]).parent / "models_frozen.json")
        result = report.evaluate(cfg)
        assert set(result["summary"]) == {"I0", *cfg["models"]}
        assert len(result["selected_tuning_terminal_diagnostic"]) == 6
        assert result["reference_scores_reproduced"] and not result["assessment_loaded"]
        path = Path(cfg["output"]).parent / "results/summary.json"
        before = digest(path)
        study.fit(cfg, "cpu")
        report.evaluate(cfg)
        assert digest(path) == before
        assert not list((Path(cfg["unseen_source"]) / "episodes/assessment").glob("*.json"))
    finally:
        torch.set_num_threads(old_threads)
