"""Independent saved-prediction and provenance audit for the learning curve."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import breadth_320_learning_curve as study

METRICS = ("nll", "error", "brier_two_class")


def audit(cfg):
    root = Path(cfg["output"])
    protocol = study.validate(cfg)
    frozen = study.validate_frozen(cfg)
    output = root.parent / "results"
    complete = study.parent.read(output / "complete.json")
    assert complete["models_frozen_sha256"] == digest(root.parent / "models_frozen.json")
    for name, sha in complete["artifacts"].items():
        assert digest(output / name) == sha, name
    result = study.parent.read(output / "summary.json")
    assert result["completed_utc"] >= frozen["frozen_utc"]
    assert result["reference_scores_reproduced"] and not result["assessment_loaded"]
    common = dict(np.load(root / "development.npz", allow_pickle=False))
    np.testing.assert_array_equal(common["pairs"], [(e, t) for e in range(len(common["episode_qids"])) for t in range(8)])
    e, t = common["pairs"].T
    targets = (common["spins"][e, t + 1] == 1).astype(int)
    qids = common["episode_qids"][e]
    folds = {f["id"]: f for f in protocol["folds"]}
    validation_ids = sorted(next(iter(folds.values()))["validation_questions"])
    assert not set(common["episode_qids"]) & set(next(iter(folds.values()))["assessment_questions"])
    checked, validation, seen = [], defaultdict(list), set()
    max_difference, normalization = 0., 0.
    for record in result["by_seed"]:
        fold = folds[record["fold"]]
        key = tuple(record[k] for k in ("fold", "model", "seed", "split"))
        assert key not in seen
        seen.add(key)
        split = record["split"]
        mask = np.isin(qids, fold[split + "_questions"])
        y = targets[mask]
        lp = np.load(output / f"{key[0]}_{key[1]}_s{key[2]}_{split}_logp.npy", allow_pickle=False).astype(float)
        assert lp.shape == (*y.shape, 2) and np.isfinite(lp).all()
        p = np.exp(lp)
        normalization = max(normalization, float(abs(p.sum(-1) - 1).max()))
        assert normalization < 1e-6
        arrays = {"nll": -np.where(y == 1, lp[..., 1], lp[..., 0]),
                  "error": lp.argmax(-1) != y, "brier_two_class": 2 * (p[..., 1] - y) ** 2}
        values = {m: float(v.mean()) for m, v in arrays.items()}
        for m, v in values.items():
            difference = abs(v - record[m])
            max_difference = max(max_difference, difference)
            assert difference < 1e-12
        checked.append({**{k: record[k] for k in ("fold", "model", "seed", "split", "n_questions")}, **values})
        if split == "validation":
            ordered_qids = common["episode_qids"][np.isin(common["episode_qids"], validation_ids)].tolist()
            per_question = np.stack([arrays[m].reshape(-1, 8, 32).mean((1, 2)) for m in METRICS], -1)
            assert len(per_question) == len(validation_ids)
            ordered = per_question[[ordered_qids.index(q) for q in validation_ids]]
            validation[(record["n_questions"], record["model"])].append(ordered)
    expected_arrays = len(folds) * 2 * (1 + len(cfg["model_order"]) * len(cfg["training"]["refit_seeds"]))
    assert len(checked) == expected_arrays
    for n in cfg["curve"]["training_question_counts"]:
        path_count = 1 if n == cfg["counts"]["train"] else len(cfg["curve"]["subset_seeds"])
        for kind in ["I0"] + cfg["model_order"]:
            expected = path_count * (1 if kind == "I0" else len(cfg["training"]["refit_seeds"]))
            assert len(validation[n, kind]) == expected
            for split in ("train", "validation"):
                records = [r for r in checked if r["n_questions"] == n and r["model"] == kind and r["split"] == split]
                assert len(records) == expected and len({r["fold"] for r in records}) == path_count
                for m in METRICS:
                    difference = abs(np.mean([r[m] for r in records]) - result["summary"][str(n)][kind][split][m])
                    max_difference = max(max_difference, float(difference))
                    assert difference < 1e-12
    vectors = {k: np.mean(v, axis=0) for k, v in validation.items()}
    draws = np.random.default_rng(cfg["evaluation"]["bootstrap_seed"]).integers(
        0, len(validation_ids), size=(cfg["evaluation"]["bootstrap_replicates"], len(validation_ids)))
    contrast_checks = 0

    def check(delta, expected):
        nonlocal max_difference, contrast_checks
        for j, metric in enumerate(METRICS):
            values = delta[:, j]
            lo, hi = np.quantile(values[draws].mean(-1), [.025, .975])
            for name, value in (("difference", values.mean()), ("lower", lo), ("upper", hi)):
                difference = abs(value - expected[metric][name])
                max_difference = max(max_difference, float(difference))
                assert difference < 1e-12
        contrast_checks += 1

    differences = result["paired_differences"]
    reference = cfg["evaluation"]["reference_model"]
    for growth, models in differences["data_growth"].items():
        low, high = map(int, growth.split("_to_"))
        for kind, expected in models.items():
            check(vectors[high, kind] - vectors[low, kind], expected)
        for name, expected in differences["data_growth_interaction"][growth].items():
            kind = name.split("_minus_")[0]
            check(vectors[high, kind] - vectors[low, kind] - vectors[high, reference] + vectors[low, reference], expected)
    for group in ("model_size", "versus_Ising"):
        for count, models in differences[group].items():
            for name, expected in models.items():
                left, right = name.split("_minus_")
                check(vectors[int(count), left] - vectors[int(count), right], expected)
    initials = defaultdict(set)
    for run in frozen["refits"]:
        manifest = study.parent.read(Path(run["directory"]) / "manifest.json")
        assert manifest["training_questions"] == folds[run["fold"]]["train_questions"]
        initials[run["model"], run["seed"]].add(manifest["initial_weights_sha256"])
    assert all(len(v) == 1 for v in initials.values())
    expected_refits = len(folds) * len(cfg["model_order"]) * len(cfg["training"]["refit_seeds"])
    expected_repro = len(folds) * len(cfg["model_order"]) * (len(cfg["training"]["refit_seeds"]) + 2)
    assert len(frozen["refits"]) == expected_refits and len(result["cpu_reproduction"]) == expected_repro
    audit = {"completed_utc": study.parent.now(), "audit_code_sha256": digest(__file__),
        "summary_sha256": digest(output / "summary.json"), "models_frozen_sha256": digest(root.parent / "models_frozen.json"),
        "new_frozen_artifacts_verified": len(frozen["artifacts"]), "prior_artifacts_verified": len(protocol["prior_artifacts"]),
        "result_artifacts_verified": len(complete["artifacts"]), "prediction_arrays_independently_scored": len(checked),
        "paired_contrasts_independently_reproduced": contrast_checks, "max_metric_or_interval_difference": max_difference,
        "max_probability_normalization_error": normalization,
        "gpu_cpu_comparisons": expected_repro,
        "max_gpu_cpu_probability_difference": max(r["max_probability_difference"] for r in result["cpu_reproduction"]),
        "gpu_cpu_class_disagreements": sum(r["class_disagreements"] for r in result["cpu_reproduction"]),
        "matched_initial_weights_across_subsets": True, "full_endpoint_counted_once": True,
        "source_calls": 0, "assessment_loaded": False}
    path = root.parent / "completion_audit.json"
    if path.exists():
        previous = study.parent.read(path)
        assert all(previous[k] == v for k, v in audit.items() if k != "completed_utc")
        audit = previous
    else:
        json_write(path, audit)
    print(audit)
    return audit


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=str(study.CONFIG))
    args = parser.parse_args()
    study.CONFIG = Path(args.config)
    audit(study.config())
