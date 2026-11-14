"""Synthetic provenance, leakage, fitting and paired-statistic checks."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mac_agents.common import digest, json_write
from scripts import breadth_expand_train as study
from scripts import breadth_train as old
from scripts import transformer_generalization as base


def fixture(tmp_path):
    cfg = copy.deepcopy(study.config())
    root, parent = tmp_path / "source", tmp_path / "parent"
    cfg.update(data_root=str(root), parent_root=str(parent), output=str(tmp_path / "run/predictors"),
               counts={"train": 12, "validation": 4, "assessment": 2, "new": 8, "retained": 8})
    cfg["training"].update(max_steps=4, validation_every_steps=2, patience=99, min_delta=0.,
        learning_rates=[.01], output_penalties=[0.], torch_threads=2, aggregate_minutes=2)
    for directory in (root, parent):
        json_write(directory / "source_protocol.json", {"synthetic_source": str(directory)})
    questions, index = [], []
    graph = study.read(study.GRAPH_FILE)["graphs"][0]
    for i in range(18):
        split = "train" if i < 12 else "validation" if i < 16 else "assessment"
        origin = "retained" if i < 4 or 12 <= i < 16 else "new"
        q = {"qid": f"q{i:02d}", "split": split, "origin": origin, "graph_id": graph["id"],
             "question": f"Synthetic question {i}", "choices": {"A": "1", "B": "2"},
             "answer": "A", "solution": "Private", "screen": {"entropy": .2}}
        questions.append(q)
        if split == "assessment":
            continue
        rng = np.random.default_rng(i)
        spins = np.where(rng.random((9, 32)) < .85, 1, -1)
        raw = np.repeat(spins[..., None], 5, axis=-1)
        protocol = digest((parent if origin == "retained" else root) / "source_protocol.json")
        path = root / "episodes" / split / f"{q['qid']}__{q['graph_id']}__r00.json"
        json_write(path, {"meta": {k: q[k] for k in ("qid", "split", "graph_id")} | {"protocol_sha256": protocol},
            "statement": q["question"], "choices": q["choices"], "J": graph["J"],
            "spins_history": spins.tolist(), "spins_raw_history": raw.tolist()})
        index.append({**{k: q[k] for k in ("qid", "split", "graph_id", "origin")}, "path": str(path),
                      "sha256": digest(path), "source_protocol_sha256": protocol, "invalid_votes": 0, "valid_snapshots": 8})
    json_write(root / "episode_index.json", index)
    json_write(root / "selection.json", {"questions": questions})
    json_write(root / "collection_complete.json", {"synthetic": True})
    retained = [q for q in questions[:16] if q["origin"] == "retained"]
    json_write(parent / "selection.json", {"questions": retained})
    json_write(parent / "models_frozen.json", {"artifacts": {}, "refits": []})
    return cfg, questions, index


def test_mixed_generation_hashes_and_majorities(tmp_path):
    cfg, questions, index = fixture(tmp_path)
    common, sources = study.load_episodes(cfg, questions[:16])
    assert len(sources) == 16 and common["pairs"].shape == (128, 2)
    assert len(set(r["source_protocol_sha256"] for r in index)) == 2
    retained = next(r for r in index if r["origin"] == "retained")
    path = Path(retained["path"])
    episode = study.read(path)
    episode["meta"]["protocol_sha256"] = digest(Path(cfg["data_root"]) / "source_protocol.json")
    json_write(path, episode)
    retained["sha256"] = digest(path)
    json_write(Path(cfg["data_root"]) / "episode_index.json", index)
    with pytest.raises(AssertionError):
        study.load_episodes(cfg, questions[:16])


def test_corrupt_majority_rejected_even_with_matching_file_hash(tmp_path):
    cfg, questions, index = fixture(tmp_path)
    path = Path(index[0]["path"])
    episode = study.read(path)
    episode["spins_history"][1][0] *= -1
    json_write(path, episode)
    index[0]["sha256"] = digest(path)
    json_write(Path(cfg["data_root"]) / "episode_index.json", index)
    with pytest.raises(AssertionError):
        study.load_episodes(cfg, questions[:16])


def test_unindexed_and_assessment_episodes_cannot_enter(tmp_path):
    cfg, questions, index = fixture(tmp_path)
    changed = copy.deepcopy(questions[:16])
    changed[-1] = questions[-1]
    with pytest.raises(AssertionError):
        study.load_episodes(cfg, changed)
    index[0]["path"] = str(Path(cfg["data_root"]) / "episodes/assessment/incorrect.json")
    json_write(Path(cfg["data_root"]) / "episode_index.json", index)
    with pytest.raises(AssertionError):
        study.load_episodes(cfg, questions[:16])


def test_paired_statistics_average_seeds_within_question(tmp_path):
    cfg, _, _ = fixture(tmp_path)
    rows = []
    qids = [f"q{i}" for i in range(7)]
    for i, q in enumerate(qids):
        rows.append({"qid": q, "model": "I0", **{k: .2 + i / 100 for k in study.METRICS}})
        for offset in (-.03, -.02, -.01):
            rows.append({"qid": q, "model": "nano", **{k: .2 + i / 100 + offset for k in study.METRICS}})
    result = study.paired(rows, "nano", "I0", qids, cfg)
    permuted = study.paired(list(reversed(rows)), "nano", "I0", qids, cfg)
    for metric, record in result.items():
        np.testing.assert_allclose([record[k] for k in ("difference", "lower", "upper")],
                                   [permuted[metric][k] for k in ("difference", "lower", "upper")], atol=1e-15)
        np.testing.assert_allclose([record[k] for k in ("difference", "lower", "upper")], -.02, atol=1e-15)
        assert record["questions_better"] == 7


def test_prepare_fit_resume_freeze_and_no_validation_refit(tmp_path, monkeypatch):
    cfg, questions, index = fixture(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(study, "CONFIG", config_path)
    monkeypatch.setattr(study, "verify_source", lambda cfg: None)
    calls = []

    def encode(texts, device):
        calls.append(texts)
        values = np.array([np.random.default_rng(int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")).normal(size=384) for t in texts], dtype=np.float32)
        return values, {"synthetic_fixture": True, "text_count": len(texts), "tokens_dropped": 0}

    monkeypatch.setattr(old, "encode_texts", encode)
    parent = Path(cfg["parent_root"])
    retained = [q for q in questions[:16] if q["origin"] == "retained"]
    cached, _ = encode([old.question_text(q) for q in retained], "cpu")
    calls.clear()
    (parent / "predictors").mkdir()
    np.savez_compressed(parent / "predictors/development.npz", episode_qids=np.array([q["qid"] for q in retained]))
    np.savez_compressed(parent / "predictors/context_vectors.npz", question_vectors=cached,
                        persona_vectors=np.random.default_rng(90).normal(size=(32, 384)).astype(np.float32))
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        study.prepare(cfg, "cpu")
        assert len(calls) == 1 and len(calls[0]) == 8
        assert all("Private" not in text and "Synthetic question 16\n" not in text for text in calls[0])
        root = Path(cfg["output"])
        prepared_sha = digest(root / "prepared.json")
        study.prepare(cfg, "cpu")
        assert len(calls) == 1 and digest(root / "prepared.json") == prepared_sha
        protocol = study.validate_prepared(cfg)
        assert protocol["parameters"] == {"I0": 19, "nano": 818}
        assert len(protocol["fold"]["train_questions"]) == 12
        assert len(protocol["fold"]["validation_questions"]) == 4
        study.fit(cfg, "cpu")
        frozen = study.validate_frozen(cfg)
        selected = study.read(root / "selection.json")
        assert selected["refit_steps"] > 0 and len(frozen["refits"]) == 3
        for run in frozen["refits"]:
            record = study.read(Path(run["directory"]) / "manifest.json")
            assert record["validation"] is None and record["steps"] == selected["refit_steps"]
            assert record["initial_weights_sha256"] != record["selected_weights_sha256"]
        sha = digest(root.parent / "models_frozen.json")
        study.fit(cfg, "cpu")
        assert digest(root.parent / "models_frozen.json") == sha
        assert not list((Path(cfg["data_root"]) / "episodes/assessment").glob("*.json"))
    finally:
        torch.set_num_threads(threads)
