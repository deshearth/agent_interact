"""Question-count by model-size study using existing, frozen Llama data."""
from argparse import ArgumentParser
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, config as load_config, digest, json_write
from mac_agents.models.baselines import baseline_logp, fit_baseline
from scripts import breadth_expand_train as parent
from scripts import breadth_320_size_sweep as sizes
from scripts import transformer_generalization as base

CONFIG = Path("configs/breadth_320_learning_curve.yaml")
REPORT_CODE = Path("scripts/report_breadth_320_learning_curve.py")


def config():
    return load_config(CONFIG)


def scaled_config(cfg, count):
    result = copy.deepcopy(cfg)
    full = cfg["counts"]["train"]
    assert count in cfg["curve"]["training_question_counts"] and 0 < count <= full
    for key in ("max_steps", "validation_every_steps"):
        value, remainder = divmod(cfg["training"][key] * count, full)
        assert remainder == 0 and value > 0
        result["training"][key] = value
    assert result["training"]["max_steps"] % result["training"]["validation_every_steps"] == 0
    return result


def balanced_subsets(questions, counts, seeds, keys):
    """Metadata-constrained, nested samples; no agent outcomes enter selection."""
    questions = sorted((q for q in questions if q["split"] == "train"), key=lambda q: q["qid"])
    total = len(questions)
    assert total == max(counts) and counts == sorted(set(counts))
    assert len({q["qid"] for q in questions}) == total
    assert all(n % 2 == 0 and n >= 4 for n in counts)
    assert Counter(q["answer"] for q in questions) == {"A": total // 2, "B": total // 2}
    rows = [np.ones(total), np.array([q["answer"] == "A" for q in questions])]
    for key in keys:
        rows += [np.array([q[key] == value for q in questions]) for value in sorted({q[key] for q in questions})]
    matrix = np.asarray(rows, float)
    result = []
    for path, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        chosen = np.zeros(total)
        for n in counts[:-1]:
            expected = matrix.sum(1) * n / total
            lo, hi = np.floor(expected), np.ceil(expected)
            lo[:2] = hi[:2] = [n, n // 2]
            solution = milp(rng.normal(size=total), integrality=np.ones(total),
                            bounds=Bounds(chosen, np.ones(total)),
                            constraints=LinearConstraint(matrix, lo, hi),
                            options={"time_limit": 30., "mip_rel_gap": 0.})
            if not solution.success:
                raise RuntimeError(f"Subset path {path}, n={n}: {solution.message}")
            current = np.rint(solution.x)
            assert np.all(current >= chosen) and current.sum() == n
            assert np.all(matrix @ current >= lo) and np.all(matrix @ current <= hi)
            chosen = current
            members = [q for q, keep in zip(questions, chosen) if keep]
            result.append({"id": f"p{path}_n{n:03d}", "path": path, "subset_seed": seed,
                "n_questions": n, "questions": [q["qid"] for q in members],
                "strata": {key: dict(Counter(q[key] for q in members)) for key in ["answer"] + keys}})
    result.append({"id": f"all_n{total}", "path": None, "subset_seed": None,
        "n_questions": total, "questions": [q["qid"] for q in questions],
        "strata": {key: dict(Counter(q[key] for q in questions)) for key in ["answer"] + keys}})
    return result


def verify_reference(cfg):
    data = Path(cfg["data_reference"])
    size = Path(cfg["size_reference"])
    source = parent.read(data / "protocol.json")
    previous = parent.read(size / "predictors/protocol.json")
    assert source["core_sha256"] == previous["core_sha256"] == code_digest() == base.CORE_SHA
    assert set(cfg["models"]) == set(cfg["model_order"])
    for key in sizes.SCIENTIFIC_TRAINING_KEYS:
        assert cfg["training"][key] == previous["config"]["training"][key], key
    for kind in cfg["model_order"]:
        assert cfg["models"][kind] == previous["config"]["models"][kind]
    for split, count in cfg["counts"].items():
        assert len(source["fold"][split + "_questions"]) == count
    pinned = {}
    for root in (data.parent, size):
        freeze = parent.read(root / "models_frozen.json")
        pinned.update(freeze["artifacts"])
        pinned[str(root / "models_frozen.json")] = digest(root / "models_frozen.json")
        pinned[str(root / "completion_audit.json")] = digest(root / "completion_audit.json")
    for key in ("source_artifacts", "prior_artifacts", "encoder_files", "code_files"):
        pinned.update(source[key])
    for key in ("prior_artifacts", "code_files"):
        pinned.update(previous[key])
    for path in (data / "protocol.json", data / "context_vectors.npz", Path(cfg["question_metadata"]),
                 size / "results/summary.json", size / "results/complete.json", Path(cfg["baseline_config"])):
        pinned[str(path)] = digest(path)
    done = parent.read(size / "results/complete.json")
    pinned.update({str(size / "results" / name): sha for name, sha in done["artifacts"].items()})
    for path, sha in pinned.items():
        assert digest(path) == sha, path
    assert base.candidates(cfg) == previous["candidates"]
    return source, previous, pinned


def prepare(cfg):
    root = Path(cfg["output"])
    if (root / "prepared.json").exists():
        validate(cfg)
        return
    source, previous, pinned = verify_reference(cfg)
    reference = Path(cfg["data_reference"])
    common = dict(np.load(reference / "development.npz", allow_pickle=False))
    vectors = dict(np.load(reference / "context_vectors.npz", allow_pickle=False))
    source_fold = source["fold"]
    assert set(common["episode_qids"]) == set(source_fold["train_questions"]) | set(source_fold["validation_questions"])
    assert not set(common["episode_qids"]) & set(source_fold["assessment_questions"])
    assert common["valid"].all()
    questions = parent.read(cfg["question_metadata"])["questions"]
    subsets = balanced_subsets(questions, cfg["curve"]["training_question_counts"],
                               cfg["curve"]["subset_seeds"], cfg["curve"]["balance_keys"])
    full = next(s for s in subsets if s["path"] is None)
    assert set(full["questions"]) == set(source_fold["train_questions"])
    full["questions"] = source_fold["train_questions"]
    folds = [{"id": s["id"], "outer": s["path"], "phase": "development", "path": s["path"],
              "n_questions": s["n_questions"], "train_questions": s["questions"],
              "validation_questions": source_fold["validation_questions"],
              "assessment_questions": source_fold["assessment_questions"],
              "training": scaled_config(cfg, s["n_questions"])["training"]} for s in subsets]
    parameters = {k: sum(p.numel() for p in base.make_model(cfg, k).parameters()) for k in cfg["model_order"]}
    assert parameters == cfg["expected_parameters"]
    new_folds = [f for f in folds if f["path"] is not None]
    protocol = {"created_utc": parent.now(), "config": cfg, "config_sha256": digest(CONFIG),
        "core_sha256": code_digest(), "code_files": {str(p): digest(p) for p in (Path(__file__), REPORT_CODE, Path(base.__file__), Path(sizes.__file__))},
        "prior_artifacts": pinned, "parameters": parameters, "subsets": subsets, "folds": folds,
        "candidates": base.candidates(cfg),
        "new_tuning_fits": len(new_folds) * len(parameters) * len(base.candidates(cfg)),
        "new_refits": len(new_folds) * len(parameters) * len(cfg["training"]["refit_seeds"]),
        "new_Ising_fits": len(new_folds), "reused_full_endpoint": full["id"],
        "design": "Three metadata-balanced nested subset paths, shared across model sizes. One shared full-data endpoint, never counted as three independent subset replicates. Frozen source outcomes, opinions, signed graphs and text embeddings. No argument or vote-probability inputs.",
        "preprocessing": "Refit question PCA3 within each training subset only, using frozen question embeddings. Known-persona PCA3 remains shared. The 16-dimensional context and Ising fit are common to all widths in a subset. Copy the full-data context and Ising fit byte-for-byte.",
        "optimization": "Same nine candidates per model and subset at seed42; validation NLL selects candidate and duration. Maximum steps and validation intervals scale with question count to preserve data passes. Same patience, dropout, decay, batching and fresh seeds. All settings freeze globally before fresh-seed refits; no result-dependent pruning.",
        "interpretation": "Learning curve of the entire pipeline, including subset-fitted PCA and tuning, not an isolated labels-only or fixed-compute intervention. Existing full-data scores are known. Reused validation is exploratory. An 80-question subset from the expanded pool is not the original retained 80-question cohort.",
        "evaluation": cfg["evaluation"], "assessment_loaded": False, "source_calls": 0}
    json_write(root / "protocol.json", protocol)
    artifacts = {}
    for name in ("development.npz", "context_vectors.npz"):
        shutil.copyfile(reference / name, root / name)
        assert digest(root / name) == digest(reference / name)
        artifacts[name] = digest(root / name)
    qvec = dict(zip(common["episode_qids"], vectors["question_vectors"]))
    e, t = common["pairs"].T
    for fold in folds:
        directory = root / "folds" / fold["id"]
        directory.mkdir(parents=True, exist_ok=True)
        if fold["path"] is None:
            for name in ("features.npz", "ising.json"):
                shutil.copyfile(reference / "folds/main" / name, directory / name)
                assert digest(directory / name) == digest(reference / "folds/main" / name)
        else:
            with threadpool_limits(limits=cfg["training"]["torch_threads"]):
                features = base.fit_context(qvec, vectors["persona_vectors"], fold["train_questions"], common["episode_qids"])
                baseline = fit_baseline("I0", base.fold_dataset(common, features["context"], fold["train_questions"]), load_config(cfg["baseline_config"]))
            assert baseline["parameters"] == 19
            lp = baseline_logp(baseline, common["spins"][e, t], features["context"][e], common["J"][e])
            features["prior_logit"] = lp[..., 1] - lp[..., 0]
            np.savez_compressed(directory / "features.npz", **features)
            json_write(directory / "ising.json", baseline)
        json_write(directory / "manifest.json", {**fold, "protocol_sha256": digest(root / "protocol.json"),
            "features_sha256": digest(directory / "features.npz"), "ising_sha256": digest(directory / "ising.json")})
        for name in ("features.npz", "ising.json", "manifest.json"):
            artifacts[str((directory / name).relative_to(root))] = digest(directory / name)
        print(json.dumps({"prepared": fold["id"], "training_questions": fold["n_questions"]}), flush=True)
    json_write(root / "prepared.json", {"completed_utc": parent.now(), "protocol_sha256": digest(root / "protocol.json"), "artifacts": artifacts})
    validate(cfg)


def validate(cfg):
    root = Path(cfg["output"])
    protocol = parent.read(root / "protocol.json")
    assert cfg == protocol["config"] and digest(CONFIG) == protocol["config_sha256"]
    assert code_digest() == protocol["core_sha256"] == base.CORE_SHA
    for key in ("code_files", "prior_artifacts"):
        for path, sha in protocol[key].items():
            assert digest(path) == sha, path
    prepared = parent.read(root / "prepared.json")
    assert prepared["protocol_sha256"] == digest(root / "protocol.json")
    for path, sha in prepared["artifacts"].items():
        assert digest(root / path) == sha, path
    for data_root in ("runs/question_breadth_v1", "runs/question_breadth_320_v1"):
        assert not list((Path(data_root) / "episodes/assessment").glob("*.json"))
    return protocol


def choose(cfg, protocol, records):
    choices = []
    for fold in protocol["folds"]:
        if fold["path"] is None:
            continue
        for kind in cfg["model_order"]:
            members = [(p, m) for p, m in records if m["fold"] == fold["id"] and m["model"] == kind]
            assert len(members) == len(protocol["candidates"])
            assert {m["candidate"] for _, m in members} == {c["id"] for c in protocol["candidates"]}
            assert len({m["initial_weights_sha256"] for _, m in members}) == 1
            directory, selected = min(members, key=lambda x: (x[1]["validation"]["nll"], x[1]["candidate"]))
            choices.append({"fold": fold["id"], "model": kind, "candidate": selected["candidate"],
                "refit_steps": selected["selected_step"], "validation_nll": selected["validation"]["nll"],
                "tuning_directory": str(directory), "reused": False})
    original = parent.read(Path(cfg["size_reference"]) / "predictors/selection.json")
    for kind in cfg["model_order"]:
        selected = next(r for r in original["choices"] if r["model"] == kind)
        choices.append({**selected, "fold": protocol["reused_full_endpoint"], "reused": True})
    return choices


def validate_frozen(cfg):
    validate(cfg)
    frozen = parent.read(Path(cfg["output"]).parent / "models_frozen.json")
    for path, sha in frozen["artifacts"].items():
        assert digest(path) == sha, path
    return frozen


def fit(cfg, device):
    protocol = validate(cfg)
    root = Path(cfg["output"])
    if (root.parent / "models_frozen.json").exists():
        validate_frozen(cfg)
        return
    deadline = time.time() + cfg["training"]["aggregate_minutes"] * 60
    common = dict(np.load(root / "development.npz", allow_pickle=False))
    folds = [f for f in protocol["folds"] if f["path"] is not None]
    datasets = {f["id"]: base.FoldData(common, dict(np.load(root / "folds" / f["id"] / "features.npz", allow_pickle=False)), device) for f in folds}
    records = []
    # Keep all paths and widths in the declared grid; no result-dependent pruning.
    for fold in folds:
        local = scaled_config(cfg, fold["n_questions"])
        for candidate in protocol["candidates"]:
            for kind in cfg["model_order"]:
                item = {"phase": "tune", "fold": fold["id"], "outer": fold["path"],
                        "model": kind, "candidate": candidate["id"], "seed": cfg["training"]["tuning_seed"]}
                directory, record = base.train_one(local, protocol, datasets[fold["id"]], fold, item, device, deadline)
                if record["status"] != "complete":
                    raise SystemExit(75)
                records.append((directory, record))
    choices = choose(cfg, protocol, records)
    selection = {"protocol_sha256": digest(root / "protocol.json"), "choices": choices,
                 "tuning_manifests": {str(p): digest(p / "manifest.json") for p, _ in records}}
    if (root / "selection.json").exists():
        old = parent.read(root / "selection.json")
        assert all(old[k] == value for k, value in selection.items())
    else:
        json_write(root / "selection.json", {**selection, "frozen_utc": parent.now()})
    refits = []
    for fold in folds:
        local = scaled_config(cfg, fold["n_questions"])
        for seed in cfg["training"]["refit_seeds"]:
            for kind in cfg["model_order"]:
                choice = next(c for c in choices if c["fold"] == fold["id"] and c["model"] == kind)
                item = {"phase": "refit", "fold": fold["id"], "outer": fold["path"],
                        "model": kind, "candidate": choice["candidate"], "seed": seed}
                directory, record = base.train_one(local, protocol, datasets[fold["id"]], fold, item, device, deadline, choice["refit_steps"])
                if record["status"] != "complete":
                    raise SystemExit(75)
                assert record["validation"] is None
                refits.append({"fold": fold["id"], "model": kind, "seed": seed, "directory": str(directory), "reused": False})
    previous = parent.read(Path(cfg["size_reference"]) / "models_frozen.json")
    refits += [{**r, "fold": protocol["reused_full_endpoint"], "reused": True}
               for r in previous["refits"] if r["model"] in cfg["model_order"]]
    artifacts = {str(p): digest(p) for p in root.rglob("*") if p.is_file()}
    artifacts.update({str(p): digest(p) for p in (CONFIG, Path(__file__), REPORT_CODE)})
    json_write(root.parent / "models_frozen.json", {"frozen_utc": parent.now(),
        "protocol_sha256": digest(root / "protocol.json"), "selection_sha256": digest(root / "selection.json"),
        "refits": refits, "artifacts": artifacts, "parameters": protocol["parameters"], "assessment_loaded": False})
    print(json.dumps({"all_models_frozen": True, "new_refits": protocol["new_refits"], "total_refits": len(refits)}), flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "fit", "evaluate", "all"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    cfg = config()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    if args.stage != "evaluate" and args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Mac GPU unavailable")
    if args.stage in ("prepare", "all"):
        prepare(cfg)
    if args.stage in ("fit", "all"):
        fit(cfg, args.device)
    if args.stage in ("evaluate", "all"):
        from scripts.report_breadth_320_learning_curve import evaluate
        evaluate(cfg)


if __name__ == "__main__":
    main()
