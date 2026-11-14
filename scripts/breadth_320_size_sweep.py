"""Wide predictor-size comparison on the frozen 320/80 Llama corpus.

Preserve the existing 818 reference, Ising fit, PCA, data, and training loop.
All new fits finish and freeze before any new unseen-graph scoring.
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, config as load_config, digest, json_write
from scripts import breadth_expand_train as parent
from scripts import transformer_generalization as base

CONFIG = Path("configs/breadth_320_size_sweep.yaml")
REPORT_CODE = Path("scripts/report_breadth_320_size_sweep.py")
SCIENTIFIC_TRAINING_KEYS = ("tuning_seed", "refit_seeds", "learning_rates", "output_penalties",
    "dropout", "weight_decay", "batch_size", "evaluation_batch_size", "validation_every_steps",
    "max_steps", "patience", "min_delta", "gradient_clip", "torch_threads")


def config():
    return load_config(CONFIG)


def check_compatibility(cfg, reference):
    old = reference["config"]
    assert cfg["models"][cfg["reused_model"]] == old["models"][cfg["reused_model"]]
    for key in SCIENTIFIC_TRAINING_KEYS:
        assert cfg["training"][key] == old["training"][key], key
    assert base.candidates(cfg) == reference["candidates"]
    assert set(cfg["new_models"]) | {cfg["reused_model"]} == set(cfg["models"])
    assert cfg["reused_model"] not in cfg["new_models"]
    assert len(cfg["model_order"]) == len(cfg["models"]) and set(cfg["model_order"]) == set(cfg["models"])
    for split, count in cfg["counts"].items():
        assert len(reference["fold"][split + "_questions"]) == count


def reference_record(cfg):
    root = Path(cfg["reference"])
    frozen = parent.read(root.parent / "models_frozen.json")
    for path, sha in frozen["artifacts"].items():
        assert digest(path) == sha, path
    protocol = parent.read(root / "protocol.json")
    assert protocol["core_sha256"] == code_digest() == base.CORE_SHA
    for key in ("source_artifacts", "code_files", "encoder_files", "prior_artifacts"):
        for path, sha in protocol[key].items():
            assert digest(path) == sha, path
    prepared = parent.read(root / "prepared.json")
    assert prepared["protocol_sha256"] == digest(root / "protocol.json")
    for path, sha in prepared["artifacts"].items():
        assert digest(root / path) == sha, path
    check_compatibility(cfg, protocol)
    assert [r["seed"] for r in frozen["refits"]] == cfg["training"]["refit_seeds"]
    return protocol, frozen


def reference_tuning(cfg):
    root = Path(cfg["reference"])
    result = []
    for c in base.candidates(cfg):
        path = root / "tune/main" / f"{cfg['reused_model']}-{c['id']}-s{cfg['training']['tuning_seed']}"
        record = parent.read(path / "manifest.json")
        assert record["status"] == "complete"
        for name, sha in record["artifacts"].items():
            assert digest(path / name) == sha
        result.append((path, record))
    return result


def prepare(cfg):
    root = Path(cfg["output"])
    if (root / "prepared.json").exists():
        validate(cfg)
        return
    previous, frozen = reference_record(cfg)
    reference_tuning(cfg)
    reference = Path(cfg["reference"])
    parameters = {kind: sum(p.numel() for p in base.make_model(cfg, kind).parameters()) for kind in cfg["models"]}
    assert parameters == cfg["expected_parameters"]
    common = dict(np.load(reference / "development.npz", allow_pickle=False))
    fold = previous["fold"]
    assert set(common["episode_qids"]) == set(fold["train_questions"]) | set(fold["validation_questions"])
    assert not set(common["episode_qids"]) & set(fold["assessment_questions"])
    assert common["valid"].all() and len(common["pairs"]) == 8 * len(common["episode_qids"])
    prior = dict(frozen["artifacts"])
    for name in ("models_frozen.json", "completion_audit.json", "development_results/summary.json", "development_results/by_question.json"):
        path = reference.parent / name
        prior[str(path)] = digest(path)
    # Hash the existing graph artifacts without loading their targets in fitting.
    unseen = Path(cfg["unseen_source"])
    result = parent.read(unseen / "results.json")
    assert not result["assessment_generated_or_scored"]
    graph_protocol = parent.read(unseen / "protocol.json")
    assert set(graph_protocol["validation_question_ids"]) <= set(fold["validation_questions"])
    for name in ("protocol.json", "source_protocol.json", "results.json", "evaluation_inputs.npz", "graphs.json"):
        prior[str(unseen / name)] = digest(unseen / name)
    prior.update(result["scored_episode_sha256"])
    prior.update({str(unseen / name): sha for name, sha in result["output_sha256"].items()})
    for name in ("results.json", "protocol.json", "completion_audit.json"):
        path = Path(cfg["unseen_reference"]) / name
        prior[str(path)] = digest(path)
    for path, sha in prior.items():
        assert digest(path) == sha, path
    protocol = {"created_utc": parent.now(), "config": cfg, "config_sha256": digest(CONFIG),
        "core_sha256": code_digest(), "code_files": {str(p): digest(p) for p in (Path(__file__), REPORT_CODE, Path(base.__file__))},
        "reference_protocol_sha256": digest(reference / "protocol.json"), "prior_artifacts": prior,
        "parameters": parameters, "fold": fold, "candidates": base.candidates(cfg),
        "tuning_cohort": [{"phase": "tune", "fold": "main", "outer": 0, "model": kind,
            "candidate": c["id"], "seed": cfg["training"]["tuning_seed"]}
            for c in base.candidates(cfg) for kind in cfg["new_models"]],
        "input_policy": cfg["inputs"] + "; byte-identical prepared data and PCA to the 818 reference. No arguments, vote probabilities, new source calls or additional encoder training.",
        "training_policy": "Same frozen optimizer/checkpoint code, nine candidates, maximum updates, validation interval, patience, dropout, decay, loss and seed numbers for every width. One attention block and four heads. Vary hidden/query/key/value/FFN widths; no new state information. Execute candidates across all widths in turn, with no result-dependent pruning of sizes.",
        "selection_policy": "Complete all tuning, independently minimize validation-question NLL per size with candidate-ID tie-break. Freeze settings and each selected duration before three fresh-seed refits. Do not fit on validation questions or pick a best seed. Reuse all existing 818 fits exactly.",
        "evaluation_policy": cfg["evaluation"], "new_neural_fits": len(cfg["new_models"]) * (len(base.candidates(cfg)) + len(cfg["training"]["refit_seeds"])),
        "reused_neural_fits": len(base.candidates(cfg)) + len(cfg["training"]["refit_seeds"]),
        "reused_Ising_fits": 1, "known_results": "818 and Ising development and graph results were already observed. This is an exploratory model-size follow-up, not a fresh confirmatory assessment.",
        "source_calls": 0, "assessment_loaded": False}
    json_write(root / "protocol.json", protocol)
    copied = []
    for name in ("development.npz", "folds/main/features.npz", "folds/main/ising.json"):
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / name, dest)
        assert digest(dest) == digest(reference / name)
        copied.append(name)
    json_write(root / "folds/main/manifest.json", {**fold, "protocol_sha256": digest(root / "protocol.json"),
        "features_sha256": digest(root / "folds/main/features.npz"), "ising_sha256": digest(root / "folds/main/ising.json")})
    copied.append("folds/main/manifest.json")
    json_write(root / "prepared.json", {"protocol_sha256": digest(root / "protocol.json"),
        "artifacts": {name: digest(root / name) for name in copied}})
    print(json.dumps({"prepared": True, "parameters": parameters, "new_fits": protocol["new_neural_fits"], "reused_fits": protocol["reused_neural_fits"]}), flush=True)


def validate(cfg):
    root = Path(cfg["output"])
    protocol = parent.read(root / "protocol.json")
    assert cfg == protocol["config"] and digest(CONFIG) == protocol["config_sha256"]
    assert code_digest() == protocol["core_sha256"] == base.CORE_SHA
    for key in ("code_files", "prior_artifacts"):
        for path, sha in protocol[key].items():
            assert digest(path) == sha, path
    prepared = parent.read(root / "prepared.json")
    assert digest(root / "protocol.json") == prepared["protocol_sha256"]
    for path, sha in prepared["artifacts"].items():
        assert digest(root / path) == sha, path
    previous = parent.read(Path(cfg["reference"]) / "protocol.json")
    check_compatibility(cfg, previous)
    for data_root in ("runs/question_breadth_v1", "runs/question_breadth_320_v1", cfg["unseen_source"]):
        assert not list((Path(data_root) / "episodes/assessment").glob("*.json"))
    return protocol


def choose(cfg, records):
    choices = []
    for kind in cfg["model_order"]:
        matches = [(p, r) for p, r in records if r["model"] == kind]
        assert len(matches) == len(base.candidates(cfg))
        assert {r["candidate"] for p, r in matches} == {c["id"] for c in base.candidates(cfg)}
        assert len({r["initial_weights_sha256"] for p, r in matches}) == 1
        directory, chosen = min(matches, key=lambda item: (item[1]["validation"]["nll"], item[1]["candidate"]))
        choices.append({"model": kind, "candidate": chosen["candidate"], "refit_steps": chosen["selected_step"],
                        "validation_nll": chosen["validation"]["nll"], "tuning_directory": str(directory)})
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
    features = dict(np.load(root / "folds/main/features.npz", allow_pickle=False))
    data = base.FoldData(common, features, device)
    fold = protocol["fold"]
    records = []
    for item in protocol["tuning_cohort"]:
        directory, record = base.train_one(cfg, protocol, data, fold, item, device, deadline)
        if record["status"] != "complete":
            raise SystemExit(75)
        records.append((directory, record))
    reused = reference_tuning(cfg)
    choices = choose(cfg, records + reused)
    original = parent.read(Path(cfg["reference"]) / "selection.json")
    selected_reference = next(c for c in choices if c["model"] == cfg["reused_model"])
    assert all(selected_reference[k] == original[k] for k in ("candidate", "refit_steps"))
    selection = {"protocol_sha256": digest(root / "protocol.json"), "choices": choices,
                 "tuning_manifests": {str(p): digest(p / "manifest.json") for p, _ in records + reused},
                 "unseen_graph_scores_used": False}
    if (root / "selection.json").exists():
        previous = parent.read(root / "selection.json")
        assert all(previous[k] == v for k, v in selection.items())
    else:
        json_write(root / "selection.json", {**selection, "frozen_utc": parent.now()})
    refits = []
    for seed in cfg["training"]["refit_seeds"]:
        for kind in cfg["new_models"]:
            selected = next(c for c in choices if c["model"] == kind)
            item = {"phase": "refit", "fold": "main", "outer": 0, "model": kind,
                    "candidate": selected["candidate"], "seed": seed}
            directory, record = base.train_one(cfg, protocol, data, fold, item, device, deadline, selected["refit_steps"])
            if record["status"] != "complete":
                raise SystemExit(75)
            assert record["validation"] is None
            refits.append({"model": kind, "seed": seed, "directory": str(directory), "reused": False})
    reference_freeze = parent.read(Path(cfg["reference"]).parent / "models_frozen.json")
    refits += [{**r, "model": cfg["reused_model"], "reused": True} for r in reference_freeze["refits"]]
    artifacts = {str(p): digest(p) for p in root.rglob("*") if p.is_file()}
    artifacts.update({str(CONFIG): digest(CONFIG), str(Path(__file__)): digest(__file__), str(REPORT_CODE): digest(REPORT_CODE)})
    json_write(root.parent / "models_frozen.json", {"frozen_utc": parent.now(), "protocol_sha256": digest(root / "protocol.json"),
        "selection_sha256": digest(root / "selection.json"), "refits": refits, "parameters": protocol["parameters"],
        "artifacts": artifacts, "assessment_loaded": False, "unseen_graphs_scored_in_this_sweep": False})
    print(json.dumps({"all_models_frozen": True, "new_refits": len(cfg["new_models"]) * len(cfg["training"]["refit_seeds"]), "reused_refits": len(cfg["training"]["refit_seeds"])}), flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "fit", "evaluate", "all"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    cfg = config()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    if args.stage != "evaluate" and args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Mac GPU unavailable in this process")
    if args.stage in ("prepare", "all"):
        prepare(cfg)
    if args.stage in ("fit", "all"):
        fit(cfg, args.device)
    if args.stage in ("evaluate", "all"):
        from scripts.report_breadth_320_size_sweep import evaluate
        evaluate(cfg)


if __name__ == "__main__":
    main()
