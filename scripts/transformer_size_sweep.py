"""Nested whole-question transformer width comparison using frozen inputs.

Reuse the verified 34,978-parameter reference and train only two new sizes.
The unchanged generalization driver supplies the exact optimizer/stopping loop.
"""
from argparse import ArgumentParser
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, config, digest, environment, json_write
from scripts import transformer_generalization as base

CONFIG = Path("configs/transformer_size_sweep.yaml")
MUTABLE_DOCS = {"README.md", "STATUS.md", "mac_experiment_work_mode.md"}


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_reference_compatibility(cfg, original):
    assert cfg["models"][cfg["reused_model"]] == original["models"][cfg["reused_model"]]
    assert cfg["source_config"] == original["source_config"]
    assert cfg["training"] == original["training"]
    assert base.candidates(cfg) == base.candidates(original)
    assert set(cfg["new_models"]) | {cfg["reused_model"]} == set(cfg["models"])
    assert not set(cfg["new_models"]) & {cfg["reused_model"]}


def reference_runs(root, phase, kind="small"):
    done = json.loads((root/f"{phase}_complete.json").read_text())
    assert done["protocol_sha256"] == digest(root/"protocol.json")
    result = []
    for name, sha in done["manifests"].items():
        path = root/name
        if not path.name.startswith(kind+"-"):
            continue
        assert digest(path/"manifest.json") == sha
        m = json.loads((path/"manifest.json").read_text())
        assert m["model"] == kind and m["status"] == "complete" and not m["outer_outcomes_scored"]
        for file, checksum in m["artifacts"].items():
            assert digest(path/file) == checksum
        result.append((path, m))
    assert len(result) == (72 if phase == "tune" else 12)
    return result


def prior_artifacts():
    result = {}
    for name in ("runs/transformer_generalization_v1/completion_audit.json",
                 "runs/ising_gap_diagnosis_v1/completion_audit.json"):
        record = json.loads(Path(name).read_text())
        for path, sha in record["artifacts"].items():
            if path in MUTABLE_DOCS:
                continue
            if path in result:
                assert result[path] == sha
            result[path] = sha
        result[name] = digest(name)
    for path, sha in result.items():
        assert digest(path) == sha, path
    return result


def prepare(cfg):
    root, reference = Path(cfg["output"]), Path(cfg["reference"])
    if (root/"prepared.json").exists():
        validate(cfg)
        print(json.dumps({"prepared": True, "reused": True})); return
    original = config(base.CONFIG)
    old_protocol = base.validate(original)
    verify_reference_compatibility(cfg, original)
    old_tune = reference_runs(reference, "tune")
    old_refit = reference_runs(reference, "refit")
    prior = prior_artifacts()
    assert code_digest() == base.CORE_SHA
    root.mkdir(parents=True, exist_ok=True)
    reused = {phase: [{"directory": str(path), "manifest_sha256": digest(path/"manifest.json")}
                      for path, m in records]
              for phase, records in (("tune", old_tune), ("refit", old_refit))}
    cohort = [{"phase": "tune", "fold": f["id"], "outer": f["outer"], "model": model,
               "candidate": candidate["id"], "seed": cfg["training"]["tuning_seed"]}
              for f in old_protocol["folds"] if f["phase"] == "tune"
              for model in cfg["new_models"] for candidate in base.candidates(cfg)]
    assert len(cohort) == 144
    protocol = {"config": cfg, "config_sha256": digest(CONFIG),
        "implementation_sha256": digest(__file__), "training_driver_sha256": digest(base.__file__),
        "core_sha256": code_digest(), "reference_protocol_sha256": digest(reference/"protocol.json"),
        "reference_config_sha256": digest(base.CONFIG), "questions": old_protocol["questions"],
        "folds": old_protocol["folds"], "candidates": base.candidates(cfg), "tuning_cohort": cohort,
        "parameters": {kind: sum(p.numel() for p in base.make_model(cfg, kind).parameters()) for kind in cfg["models"]},
        "reused_runs": reused, "new_tuning_fits": 144, "new_refits": 24, "reused_tuning_fits": 72,
        "reused_refits": 12, "reused_pca_and_ising_folds": 16,
        "input_policy": "Identical current opinions, signed graph/degrees and 16-dimensional persona/question context. No arguments, extra history, IDs or new source responses. Four heads and one attention block; hidden/key/value/feedforward widths change.",
        "training_policy": "Exact frozen optimizer, dropout, output penalty, initialization, batch order, seed and inner stopping implementation. Six candidates per size. Max3072 inner updates; checks every128; patience6; NLL selection. Refit for median selected inner data passes, scaled10-to15 questions, at seeds142/143/144.",
        "selection_policy": "Independently select one candidate per outer fold and size by mean inner-question NLL, tie-break candidate ID. Freeze all eight new choices and complete all24 refits before new outer outcomes are scored. Report all sizes; no best-seed selection.",
        "assessment_policy": "Same four outer folds of five of twenty previously studied development questions. NLL primary; error/Brier secondary. Average seed scores within question, never ensemble predictions. Paired question-bootstrap intervals descriptive, not confirmatory or equivalence tests.",
        "baseline_policy": "Reuse original fold-fitted Ising and the independently nested Ising plus current-opinion correction (19 baseline coefficients plus3 correction coefficients). Baseline has3 penalty choices; neural models each have6. It is a benchmark, not an equal-search-budget architectural ablation.",
        "reuse_policy": "Reference weights, preprocessing, raw predictions and training settings checksum-verified; old results were already observed. Reference fits are reused rather than counted as new training.",
        "fresh_confirmatory_dataset": False, "new_official_test_evaluations": 0,
        "new_source_model_calls": 0, "autonomous_rollouts": False}
    if (root/"protocol.json").exists():
        old = json.loads((root/"protocol.json").read_text())
        assert all(old[k] == v for k, v in protocol.items())
    else:
        json_write(root/"protocol.json", {**protocol, "created_utc": now()})
    copied = []
    for name in ("development.npz", "episode_index.json"):
        shutil.copyfile(reference/name, root/name); copied.append(name)
    for fold in old_protocol["folds"]:
        (root/"folds"/fold["id"]).mkdir(parents=True, exist_ok=True)
        for name in ("features.npz", "ising.json", "manifest.json"):
            rel = Path("folds")/fold["id"]/name
            shutil.copyfile(reference/rel, root/rel)
            assert digest(root/rel) == digest(reference/rel)
            copied.append(str(rel))
    json_write(root/"prior_artifacts.json", prior)
    json_write(root/"prepared.json", {"created_utc": now(), "protocol_sha256": digest(root/"protocol.json"),
        "artifacts": {name: digest(root/name) for name in copied},
        "prior_artifacts_sha256": digest(root/"prior_artifacts.json"),
        "reference_input_audit_sha256": digest(reference/"input_audit.json"),
        "reference_input_audit": json.loads((reference/"input_audit.json").read_text()),
        "prior_artifacts_verified": len(prior)})
    print(json.dumps({"parameters": protocol["parameters"], "new_tuning": 144, "new_refits": 24,
                      "reused_tuning": 72, "reused_refits": 12, "prior_artifacts_verified": len(prior)}), flush=True)


def validate(cfg):
    root = Path(cfg["output"])
    protocol = json.loads((root/"protocol.json").read_text())
    assert protocol["config"] == cfg
    assert protocol["config_sha256"] == digest(CONFIG)
    assert protocol["implementation_sha256"] == digest(__file__)
    assert protocol["training_driver_sha256"] == digest(base.__file__)
    assert protocol["core_sha256"] == code_digest() == base.CORE_SHA
    prepared = json.loads((root/"prepared.json").read_text())
    assert prepared["protocol_sha256"] == digest(root/"protocol.json")
    for path, sha in prepared["artifacts"].items():
        assert digest(root/path) == sha, path
    assert digest(root/"prior_artifacts.json") == prepared["prior_artifacts_sha256"]
    assert digest(Path(cfg["reference"])/"protocol.json") == protocol["reference_protocol_sha256"]
    verify_reference_compatibility(cfg, config(base.CONFIG))
    return protocol


def choose(records, candidates, kinds, outer_count=4):
    choices, details = [], []
    for outer in range(outer_count):
        for kind in kinds:
            options = []
            for candidate in candidates:
                matched = [(p, m) for p, m in records if m["outer"] == outer and m["model"] == kind
                           and m["candidate"] == candidate["id"]]
                assert len(matched) == 3 and len({m["fold"] for p, m in matched}) == 3
                row = {"outer": outer, "model": kind, "candidate": candidate["id"],
                    "inner_mean_nll": float(np.mean([m["validation"]["nll"] for p, m in matched])),
                    "inner_selected_steps": [m["selected_step"] for p, m in matched],
                    "runs": [str(p) for p, m in matched]}
                options.append(row); details.append(row)
            best = min(options, key=lambda r: (r["inner_mean_nll"], r["candidate"]))
            duration = int(math.ceil(float(np.median(best["inner_selected_steps"]))*1.5))
            choices.append({"phase": "refit", "fold": f"o{outer}_refit", "outer": outer, "model": kind,
                            "candidate": best["candidate"], "refit_steps": duration})
    return choices, details


def select(cfg):
    root = Path(cfg["output"])
    protocol = validate(cfg)
    records = base.completed_runs(root, "tune")
    assert len(records) == 144
    reused = reference_runs(Path(cfg["reference"]), "tune")
    choices, details = choose(records+reused, protocol["candidates"], cfg["models"])
    original_selection = json.loads((Path(cfg["reference"])/"selection.json").read_text())
    for choice in choices:
        if choice["model"] == cfg["reused_model"]:
            original = next(r for r in original_selection["choices"] if r["outer"] == choice["outer"] and r["model"] == choice["model"])
            assert choice["candidate"] == original["candidate"] and choice["refit_steps"] == original["refit_steps"]
    payload = {"protocol_sha256": digest(root/"protocol.json"),
        "tuning_complete_sha256": digest(root/"tune_complete.json"),
        "choices": [r for r in choices if r["model"] in cfg["new_models"]],
        "reused_choices": [r for r in choices if r["model"] == cfg["reused_model"]],
        "all_candidates": details, "selection_uses_outer_outcomes": False}
    if (root/"selection.json").exists():
        old = json.loads((root/"selection.json").read_text())
        assert all(old[k] == v for k, v in payload.items())
    else:
        json_write(root/"selection.json", {**payload, "frozen_utc": now()})
    print(json.dumps({"new_choices": payload["choices"], "reused_choices": payload["reused_choices"]}, indent=2))


def stage(cfg, phase, device):
    root = Path(cfg["output"])
    protocol = validate(cfg)
    common = dict(np.load(root/"development.npz", allow_pickle=False))
    if phase == "tune":
        cohort, expected = protocol["tuning_cohort"], 144
    else:
        selected = json.loads((root/"selection.json").read_text())
        assert selected["protocol_sha256"] == digest(root/"protocol.json")
        assert selected["tuning_complete_sha256"] == digest(root/"tune_complete.json")
        cohort = [{**entry, "seed": seed} for entry in selected["choices"] for seed in cfg["training"]["refit_seeds"]]
        expected = 24
    timer = base.session(root, cfg)
    data, cache_key, manifests = None, None, {}
    for item in cohort:
        fold = next(f for f in protocol["folds"] if f["id"] == item["fold"])
        if cache_key != fold["id"]:
            if data is not None:
                del data
            features = dict(np.load(root/"folds"/fold["id"]/"features.npz", allow_pickle=False))
            data = base.FoldData(common, features, device)
            cache_key = fold["id"]
        out, result = base.train_one(cfg, protocol, data, fold, item, device, timer["deadline"], item.get("refit_steps"))
        if result["status"] != "complete":
            raise RuntimeError("Training time guard reached; unfinished runs preserved for diagnosis")
        manifests[str(out.relative_to(root))] = digest(out/"manifest.json")
    assert len(manifests) == expected
    json_write(root/f"{phase}_complete.json", {"completed_utc": now(), "fits": expected,
        "protocol_sha256": digest(root/"protocol.json"), "manifests": manifests, "environment": environment()})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "tune", "select", "refit", "fit"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    cfg = config(CONFIG)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    if args.stage == "prepare":
        prepare(cfg)
    elif args.stage == "select":
        select(cfg)
    elif args.stage == "fit":
        stage(cfg, "tune", args.device)
        select(cfg)
        stage(cfg, "refit", args.device)
    else:
        stage(cfg, args.stage, args.device)


if __name__ == "__main__":
    main()
