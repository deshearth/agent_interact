"""Leakage, nesting, selection, and full synthetic execution checks."""
from collections import Counter
import copy
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import yaml

from mac_agents.common import digest, json_write
from mac_agents.models.baselines import baseline_logp
from scripts import breadth_320_learning_curve as study
from scripts import report_breadth_320_learning_curve as report
from scripts import transformer_generalization as base


def questions():
    return [{"qid": f"q{i:03d}", "split": "train" if i < 16 else "validation",
             "answer": "A" if i % 2 else "B", "type": f"topic{i % 2}",
             "graph_id": f"J{(i // 2) % 2}", "origin": "new" if i % 4 else "retained",
             "level": f"level{(i // 4) % 2}"} for i in range(20)]


def test_parameters_and_training_budgets_preserve_data_passes():
    cfg = study.config()
    assert {k: sum(p.numel() for p in base.make_model(cfg, k).parameters()) for k in cfg["models"]} == {
        "nano": 818, "tiny": 7370, "wide": 533026}
    for n, steps, cadence in [(40, 4096, 64), (80, 8192, 128), (160, 16384, 256), (320, 32768, 512)]:
        scaled = study.scaled_config(cfg, n)
        assert scaled["training"]["max_steps"] == steps
        assert scaled["training"]["validation_every_steps"] == cadence
        assert steps / n == 32768 / 320 and cadence / n == 512 / 320
        for key in cfg["training"]:
            if key not in ("max_steps", "validation_every_steps"):
                assert scaled["training"][key] == cfg["training"][key]
    assert cfg["training"]["max_steps"] == 32768


def test_nested_metadata_subsets_ignore_holdouts_and_agent_outcomes():
    qs = questions()
    keys = ["type", "graph_id", "origin", "level"]
    actual = study.balanced_subsets(qs, [4, 8, 16], [101, 102], keys)
    assert len(actual) == 5 and sum(s["path"] is None for s in actual) == 1
    lookup = {q["qid"]: q for q in qs if q["split"] == "train"}
    for path in range(2):
        earlier = set()
        for n in (4, 8):
            part = next(s for s in actual if s["path"] == path and s["n_questions"] == n)
            chosen = set(part["questions"])
            assert earlier <= chosen <= set(lookup) and len(chosen) == n
            assert Counter(lookup[q]["answer"] for q in chosen) == {"A": n // 2, "B": n // 2}
            for key in keys:
                totals = Counter(q[key] for q in lookup.values())
                values = Counter(lookup[q][key] for q in chosen)
                for label, count in totals.items():
                    assert np.floor(count * n / 16) <= values[label] <= np.ceil(count * n / 16)
            earlier = chosen
    changed = copy.deepcopy(qs)
    for q in changed:
        q["future_agent_states"] = [999] * 32
        if q["split"] != "train":
            q.update(answer="X", type="unused", origin="unused", graph_id="unused", level="unused")
    assert study.balanced_subsets(changed, [4, 8, 16], [101, 102], keys) == actual


def test_preprocessing_does_not_fit_unused_training_or_validation_questions():
    rng = np.random.default_rng(121)
    vectors = {f"q{i:02d}": rng.normal(size=12) for i in range(16)}
    people = rng.normal(size=(32, 12))
    fitting = list(vectors)[:4]
    before = base.fit_context(vectors, people, fitting, list(vectors))
    changed = {k: v.copy() if k in fitting else 1000 * v + 500 for k, v in vectors.items()}
    after = base.fit_context(changed, people, fitting, list(vectors))
    for key in ("question_mean", "question_components", "persona_mean", "persona_components"):
        np.testing.assert_array_equal(before[key], after[key])
    np.testing.assert_array_equal(before["context"][:4], after["context"][:4])
    assert not np.allclose(before["context"][4:], after["context"][4:])


def test_question_pairing_interaction_and_single_shared_endpoint():
    cfg = copy.deepcopy(study.config())
    cfg["counts"]["train"] = 16
    cfg["curve"].update(training_question_counts=[4, 8, 16], subset_seeds=[101, 102])
    cfg["model_order"] = ["nano", "wide"]
    cfg["training"]["refit_seeds"] = [142, 143]
    cfg["evaluation"]["bootstrap_replicates"] = 100
    rows, qids = [], ["v0", "v1", "v2"]
    for n in (4, 8, 16):
        for path in ([None] if n == 16 else [0, 1]):
            for kind in ("I0", "nano", "wide"):
                value = .5 if kind == "I0" else .4 - .001 * n if kind == "nano" else .6 - .005 * n
                for seed in ([-1] if kind == "I0" else [142, 143]):
                    for q in qids:
                        rows.append({"fold": f"{path}_{n}", "path": path, "n_questions": n, "model": kind,
                                     "seed": seed, "qid": q, "split": "validation", **{m: value for m in report.METRICS}})
    vectors = report.validation_vectors(cfg, rows + [{**rows[0], "split": "train", "error": 99}], qids)
    contrasts = report.contrasts(cfg, vectors)
    value = contrasts["data_growth_interaction"]["4_to_16"]["wide_minus_nano"]["error"]
    assert value["difference"] == pytest.approx(-.048)
    assert value["lower"] == pytest.approx(-.048) and value["upper"] == pytest.approx(-.048)
    assert contrasts["data_growth"]["4_to_16"]["nano"]["error"]["difference"] == pytest.approx(-.012)
    with pytest.raises(AssertionError):
        report.validation_vectors(cfg, rows + [rows[-1]], qids)


def test_selection_is_per_subset_and_model_and_requires_complete_grid():
    cfg = study.config()
    folds = [{"id": f"p{i}", "path": i} for i in range(2)]
    protocol = {"folds": folds, "candidates": base.candidates(cfg), "reused_full_endpoint": "full"}
    records = []
    for i, fold in enumerate(folds):
        for j, kind in enumerate(cfg["model_order"]):
            for candidate in protocol["candidates"]:
                chosen = candidate["id"] == f"c{i + j}"
                records.append((Path(fold["id"]) / kind / candidate["id"], {
                    "fold": fold["id"], "model": kind, "candidate": candidate["id"],
                    "initial_weights_sha256": kind, "selected_step": 100,
                    "validation": {"nll": .2 if chosen else .3}}))
    choices = study.choose(cfg, protocol, records)
    assert [c["candidate"] for c in choices if not c["reused"]] == ["c0", "c1", "c2", "c1", "c2", "c3"]
    assert sum(c["reused"] for c in choices) == 3
    with pytest.raises(AssertionError):
        study.choose(cfg, protocol, records[:-1])


def synthetic_reference(cfg):
    """Build real fitted reference checkpoints from artificial conversations."""
    data_root = Path(cfg["data_reference"])
    size_root = Path(cfg["size_reference"]) / "predictors"
    data_root.mkdir(parents=True)
    qs = questions()
    qids = np.array([q["qid"] for q in qs])
    fold = {"id": "main", "outer": 0, "train_questions": qids[:16].tolist(),
            "validation_questions": qids[16:].tolist(), "assessment_questions": ["a0", "a1"]}
    rng = np.random.default_rng(331)
    spins = np.where(rng.random((20, 9, 32)) < .8, 1, -1).astype(np.int8)
    graph = np.eye(32, k=1, dtype=np.int8) + np.eye(32, k=-1, dtype=np.int8)
    common = {"spins": spins, "J": np.tile(graph, (20, 1, 1)), "valid": np.ones((20, 8), bool),
              "episode_qids": qids, "pairs": np.array([(e, t) for e in range(20) for t in range(8)]),
              "raw_votes": np.repeat(spins[..., None], 5, -1)}
    vectors = {"question_vectors": rng.normal(size=(20, 12)), "persona_vectors": rng.normal(size=(32, 12))}
    features = base.fit_context(dict(zip(qids, vectors["question_vectors"])), vectors["persona_vectors"], fold["train_questions"], qids)
    baseline = {"kind": "I0", "parameters": 19, "weights": [1.4] + [0.] * 18}
    e, t = common["pairs"].T
    lp = baseline_logp(baseline, spins[e, t], features["context"][e], common["J"][e])
    features["prior_logit"] = lp[..., 1] - lp[..., 0]
    np.savez_compressed(data_root / "development.npz", **common)
    np.savez_compressed(data_root / "context_vectors.npz", **vectors)
    (data_root / "folds/main").mkdir(parents=True)
    np.savez_compressed(data_root / "folds/main/features.npz", **features)
    json_write(data_root / "folds/main/ising.json", baseline)
    json_write(cfg["question_metadata"], {"questions": qs})
    old_cfg = copy.deepcopy(cfg)
    old_cfg["output"] = str(size_root)
    protocol = {"candidates": base.candidates(cfg)}
    json_write(size_root / "protocol.json", protocol)
    json_write(size_root / "folds/main/manifest.json", fold)
    data = base.FoldData(common, features, "cpu")
    choices = []
    for kind in cfg["model_order"]:
        tuned = []
        for candidate in protocol["candidates"]:
            item = {"phase": "tune", "fold": "main", "outer": 0, "model": kind,
                    "candidate": candidate["id"], "seed": cfg["training"]["tuning_seed"]}
            path, record = base.train_one(old_cfg, protocol, data, fold, item, "cpu", time.time() + 120)
            assert record["status"] == "complete"
            tuned.append((path, record))
        path, record = min(tuned, key=lambda x: (x[1]["validation"]["nll"], x[1]["candidate"]))
        choices.append({"model": kind, "candidate": record["candidate"], "refit_steps": record["selected_step"],
                        "validation_nll": record["validation"]["nll"], "tuning_directory": str(path)})
    json_write(size_root / "selection.json", {"choices": choices})
    refits, summary = [], {"I0": {}}
    for split in ("train", "validation"):
        ids = data.ids(fold[split + "_questions"])
        summary["I0"][split] = base.metrics(lp[ids], data.y[ids], data.qids[ids])[0]
    for kind in cfg["model_order"]:
        choice = next(c for c in choices if c["model"] == kind)
        scores = {"train": [], "validation": []}
        for seed in cfg["training"]["refit_seeds"]:
            item = {"phase": "refit", "fold": "main", "outer": 0, "model": kind,
                    "candidate": choice["candidate"], "seed": seed}
            path, record = base.train_one(old_cfg, protocol, data, fold, item, "cpu", time.time() + 120, choice["refit_steps"])
            assert record["status"] == "complete"
            refits.append({"model": kind, "seed": seed, "directory": str(path)})
            model = base.make_model(cfg, kind)
            model.load_state_dict(torch.load(path / "best.pt", weights_only=False)["model"])
            for split in scores:
                ids = data.ids(fold[split + "_questions"])
                values = base.predictions(model, data, ids, False)
                scores[split].append(base.metrics(values, data.y[ids], data.qids[ids])[0])
        summary[kind] = {split: {m: float(np.mean([r[m] for r in rs])) for m in report.METRICS}
                         for split, rs in scores.items()}
    json_write(size_root.parent / "models_frozen.json", {"refits": refits,
        "artifacts": {str(p): digest(p) for p in size_root.rglob("*") if p.is_file()}})
    json_write(size_root.parent / "results/summary.json", {"summary": summary})
    pinned = {str(p): digest(p) for directory in (data_root, size_root.parent) for p in directory.rglob("*") if p.is_file()}
    pinned[cfg["question_metadata"]] = digest(cfg["question_metadata"])
    return {"fold": fold}, {}, pinned


def test_full_synthetic_curve_freezes_reuses_and_scores_every_condition(tmp_path, monkeypatch):
    cfg = copy.deepcopy(study.config())
    cfg.update(output=str(tmp_path / "curve/predictors"), data_reference=str(tmp_path / "data/predictors"),
               size_reference=str(tmp_path / "sizes"), question_metadata=str(tmp_path / "questions.json"),
               counts={"train": 16, "validation": 4, "assessment": 2})
    cfg["curve"].update(training_question_counts=[4, 8, 16], subset_seeds=[101, 102])
    cfg["training"].update(max_steps=8, validation_every_steps=4, patience=99, min_delta=0.,
        learning_rates=[.01, .02], output_penalties=[0.], refit_seeds=[142, 143],
        batch_size=8, evaluation_batch_size=32, torch_threads=2, aggregate_minutes=5)
    cfg["evaluation"]["bootstrap_replicates"] = 50
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(study, "CONFIG", config_path)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        reference = synthetic_reference(cfg)
        # The source-validator is independently exercised on the real frozen
        # source at preparation. Substitute only artificial source metadata here.
        monkeypatch.setattr(study, "verify_reference", lambda _: reference)
        study.prepare(cfg)
        root = Path(cfg["output"])
        prepared = digest(root / "prepared.json")
        study.prepare(cfg)
        assert digest(root / "prepared.json") == prepared
        with pytest.raises(FileNotFoundError):
            report.evaluate(cfg)
        study.fit(cfg, "cpu")
        frozen = study.validate_frozen(cfg)
        assert len(frozen["refits"]) == 30 and sum(r["reused"] for r in frozen["refits"]) == 6
        result = report.evaluate(cfg)
        assert result["reference_scores_reproduced"] and not result["assessment_loaded"]
        assert len(result["choices"]) == 15 and len(result["cpu_reproduction"]) == 60
        assert set(result["summary"]) == {"4", "8", "16"}
        for n in (4, 8, 16):
            for kind in ("I0", "nano", "tiny", "wide"):
                assert result["summary"][str(n)][kind]["validation"]["subset_paths"] == (1 if n == 16 else 2)
        for path, sha in reference[2].items():
            assert digest(path) == sha
        summary = root.parent / "results/summary.json"
        before = digest(summary)
        study.fit(cfg, "cpu")
        report.evaluate(cfg)
        assert digest(summary) == before
        assert not (root.parent / "episodes/assessment").exists()
    finally:
        torch.set_num_threads(old_threads)
