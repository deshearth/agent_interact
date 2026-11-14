"""Independently audit completed size-sweep artifacts and saved predictions."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import breadth_320_size_sweep as study
from scripts import breadth_expand_train as parent


def audit():
    cfg = study.config()
    root = Path(cfg["output"])
    frozen = study.validate_frozen(cfg)
    expanded = parent.validate_frozen(parent.config())
    original = parent.read("runs/question_breadth_v1/models_frozen.json")
    for path, sha in original["artifacts"].items():
        assert digest(path) == sha, path
    protocol = parent.read(root / "protocol.json")
    output = root.parent / "results"
    complete = parent.read(output / "complete.json")
    assert complete["models_frozen_sha256"] == digest(root.parent / "models_frozen.json")
    for name, sha in complete["artifacts"].items():
        assert digest(output / name) == sha, name
    result = parent.read(output / "summary.json")
    assert result["models_frozen_sha256"] == complete["models_frozen_sha256"]
    assert result["completed_utc"] >= frozen["frozen_utc"]
    assert result["reference_scores_reproduced"]
    assert not result["assessment_loaded"] and result["source_calls"] == 0
    common = dict(np.load(root / "development.npz", allow_pickle=False))
    episode, step = common["pairs"].T
    qids = common["episode_qids"][episode]
    target = (common["spins"][episode, step + 1] == 1).astype(int)
    assert target.shape == (400 * 8, 32)
    assert not set(qids) & set(protocol["fold"]["assessment_questions"])
    source = Path(cfg["unseen_source"])
    graph_protocol = parent.read(source / "protocol.json")
    graph_qids = graph_protocol["validation_question_ids"]
    episodes = {}
    for path in (source / "episodes/validation").glob("*.json"):
        record = parent.read(path)
        assert record["meta"]["qid"] not in episodes
        episodes[record["meta"]["qid"]] = record
    assert set(episodes) == set(graph_qids)
    graph_target = np.stack([(np.asarray(episodes[q]["spins_history"])[1:] == 1).astype(int)
                             for q in graph_qids]).reshape(20 * 8, 32)
    targets = {split: target[np.isin(qids, protocol["fold"][split + "_questions"])]
               for split in ("train", "validation")}
    targets["graph_seen"] = target[np.isin(qids, graph_qids)]
    targets["graph_unseen"] = graph_target
    assert {k: v.size for k, v in targets.items()} == {
        "train": 81920, "validation": 20480, "graph_seen": 5120, "graph_unseen": 5120}
    rows, max_difference, max_normalization_error = [], 0., 0.
    for model in ["I0"] + cfg["model_order"]:
        seeds = [-1] if model == "I0" else cfg["training"]["refit_seeds"]
        for split, y in targets.items():
            scores = []
            for seed in seeds:
                lp = np.load(output / f"{model}_s{seed}_{split}_logp.npy", allow_pickle=False).astype(float)
                assert lp.shape == (*y.shape, 2) and np.isfinite(lp).all()
                p = np.exp(lp)
                max_normalization_error = max(max_normalization_error, float(abs(p.sum(-1) - 1).max()))
                assert max_normalization_error < 1e-6
                values = {
                    "error": float((lp.argmax(-1) != y).mean()),
                    "nll": float(-np.where(y == 1, lp[..., 1], lp[..., 0]).mean()),
                    "brier_two_class": float(2 * ((p[..., 1] - y) ** 2).mean())}
                expected = next(r for r in result["by_seed"] if
                                r["model"] == model and r["seed"] == seed and r["split"] == split)
                for metric, value in values.items():
                    difference = abs(value - expected[metric])
                    max_difference = max(max_difference, difference)
                    assert difference < 1e-12
                scores.append(values)
                rows.append({"model": model, "seed": seed, "split": split, **values})
            for metric in ("error", "nll", "brier_two_class"):
                difference = abs(np.mean([v[metric] for v in scores]) - result["summary"][model][split][metric])
                max_difference = max(max_difference, float(difference))
                assert difference < 1e-12
    assert len(rows) == 76
    assert len(result["tuning"]) == 54 and len(frozen["refits"]) == 18
    assert sum(not r["reused"] for r in frozen["refits"]) == 15
    assert len(result["cpu_reproduction"]) == 30
    audit = {
        "completed_utc": parent.now(), "audit_code_sha256": digest(__file__),
        "summary_sha256": digest(output / "summary.json"),
        "completion_record_sha256": digest(output / "complete.json"),
        "models_frozen_sha256": digest(root.parent / "models_frozen.json"),
        "new_frozen_artifacts_verified": len(frozen["artifacts"]),
        "original_frozen_artifacts_verified": len(original["artifacts"]),
        "expanded_frozen_artifacts_verified": len(expanded["artifacts"]),
        "prior_artifacts_verified": len(protocol["prior_artifacts"]),
        "result_artifacts_verified": len(complete["artifacts"]),
        "prediction_arrays_independently_scored": len(rows),
        "max_aggregate_difference": max_difference,
        "max_probability_normalization_error": max_normalization_error,
        "max_gpu_cpu_probability_difference": max(r["max_probability_difference"] for r in result["cpu_reproduction"]),
        "gpu_cpu_class_disagreements": sum(r["class_disagreements"] for r in result["cpu_reproduction"]),
        "refits": 18, "new_refits": 15, "tuning_fits_including_reused": 54,
        "source_calls": 0, "assessment_loaded": False,
        "aggregate_scores": rows}
    json_write(root.parent / "completion_audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "aggregate_scores"}, indent=2))
    return audit


if __name__ == "__main__":
    audit()
