"""Extend the frozen transformer size comparison to 2,626, 818 and 314 weights."""
from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, config, digest, environment, json_write
from scripts import transformer_generalization as base
from scripts import transformer_size_sweep as previous

CONFIG = Path("configs/transformer_micro_sweep.yaml")
now = previous.now


def reference_runs(cfg, phase):
    return previous.reference_runs(Path(cfg["reference"]), phase, cfg["reused_model"])


def expected(cfg, phase):
    return len(cfg["new_models"])*(72 if phase == "tune" else 12)


def prior_artifacts(cfg):
    result = previous.prior_artifacts()
    path = Path(cfg["reference"])/"completion_audit.json"
    audit = json.loads(path.read_text())
    for file, sha in audit["artifacts"].items():
        if file in previous.MUTABLE_DOCS:
            continue
        if file in result:
            assert result[file] == sha, file
        result[file] = sha
    result[str(path)] = digest(path)
    for file, sha in result.items():
        assert digest(file) == sha, file
    return result


def prepare(cfg):
    root, reference = Path(cfg["output"]), Path(cfg["reference"])
    if (root/"prepared.json").exists():
        validate(cfg); print(json.dumps({"prepared": True, "reused": True})); return
    original = config(cfg["reference_config"])
    source = previous.validate(original)
    previous.verify_reference_compatibility(cfg, original)
    old_tune, old_refit = reference_runs(cfg, "tune"), reference_runs(cfg, "refit")
    prior = prior_artifacts(cfg)
    root.mkdir(parents=True, exist_ok=True)
    cohort = [{"phase": "tune", "fold": fold["id"], "outer": fold["outer"], "model": kind,
               "candidate": candidate["id"], "seed": cfg["training"]["tuning_seed"]}
        for fold in source["folds"] if fold["phase"] == "tune"
        for kind in cfg["new_models"] for candidate in base.candidates(cfg)]
    assert len(cohort) == expected(cfg, "tune") == 216
    protocol = {"config": cfg, "config_sha256": digest(CONFIG), "implementation_sha256": digest(__file__),
        "training_driver_sha256": digest(base.__file__), "parent_driver_sha256": digest(previous.__file__),
        "core_sha256": code_digest(), "reference_protocol_sha256": digest(reference/"protocol.json"),
        "reference_config_sha256": digest(cfg["reference_config"]), "questions": source["questions"],
        "folds": source["folds"], "candidates": base.candidates(cfg), "tuning_cohort": cohort,
        "parameters": {k: sum(p.numel() for p in base.make_model(cfg, k).parameters()) for k in cfg["models"]},
        "reused_runs": {phase: [{"directory": str(path), "manifest_sha256": digest(path/"manifest.json")}
                               for path, _ in runs] for phase, runs in (("tune", old_tune), ("refit", old_refit))},
        "new_tuning_fits": 216, "new_refits": 36, "reused_tuning_fits": 72, "reused_refits": 12,
        "input_policy": "Exact same current opinions, signed graph/degrees and 16-dimensional persona/question context. Four heads and one attention block at every size; only widths shrink. No text, extra history, IDs, source calls or test questions.",
        "training_policy": "Unchanged frozen training loop, zero output initialization, dropout .1, AdamW decay .001, same six learning-rate/logit-penalty choices, batch32, clip1, checks every128 updates, patience6 and3072-update cap. Seeds42 for tuning and142/143/144 for final fits.",
        "selection_policy": "For each size/outer fold choose mean inner-question NLL, tie-break candidate ID. Final duration equals median selected inner data passes scaled10-to15 fitting questions. All12 new choices and36 final fits freeze before new outer scoring; report every size and seed.",
        "assessment_policy": "Reuse four outer whole-question folds from the twenty previously studied GPT development questions; three inner folds per outer fit ten/select five. PCA and Ising are exactly reused from their verified training-only fits. Equal-question mean NLL primary, error/Brier secondary; average seed scores, not probabilities. Paired bootstrap intervals are descriptive, not confirmatory or equivalence tests.",
        "optimization_policy": "Report cap-reaching candidates and selected checkpoints separately. A poor capped fit is not evidence of an architecture capacity limit. This finite common search does not estimate each model's fully optimized potential.",
        "baseline_policy": "Reuse original Ising and its independently nested current-opinion correction. The latter has19 baseline coefficients plus3 corrections, with3 tuning choices versus6 per neural model; it is a benchmark rather than an equal-budget architecture ablation.",
        "fresh_confirmatory_dataset": False, "new_source_model_calls": 0,
        "new_official_test_evaluations": 0, "autonomous_rollouts": False}
    assert protocol["core_sha256"] == base.CORE_SHA
    if (root/"protocol.json").exists():
        old = json.loads((root/"protocol.json").read_text())
        assert all(old[k] == v for k, v in protocol.items())
    else:
        json_write(root/"protocol.json", {**protocol, "created_utc": now()})
    copied = []
    for name in ("development.npz", "episode_index.json"):
        shutil.copyfile(reference/name, root/name); copied.append(name)
    for fold in source["folds"]:
        (root/"folds"/fold["id"]).mkdir(parents=True, exist_ok=True)
        for name in ("features.npz", "ising.json", "manifest.json"):
            rel = Path("folds")/fold["id"]/name
            shutil.copyfile(reference/rel, root/rel); copied.append(str(rel))
    for name in copied:
        assert digest(reference/name) == digest(root/name)
    json_write(root/"prior_artifacts.json", prior)
    json_write(root/"prepared.json", {"created_utc": now(), "protocol_sha256": digest(root/"protocol.json"),
        "artifacts": {name: digest(root/name) for name in copied},
        "prior_artifacts_sha256": digest(root/"prior_artifacts.json"),
        "reference_preparation_sha256": digest(reference/"prepared.json"),
        "prior_artifacts_verified": len(prior)})
    print(json.dumps({"parameters": protocol["parameters"], "new_tuning": 216, "new_refits": 36,
                      "reused_tuning": 72, "reused_refits": 12, "prior_artifacts_verified": len(prior)}), flush=True)


def validate(cfg):
    root = Path(cfg["output"])
    protocol = json.loads((root/"protocol.json").read_text())
    assert protocol["config"] == cfg and protocol["config_sha256"] == digest(CONFIG)
    assert protocol["implementation_sha256"] == digest(__file__)
    assert protocol["training_driver_sha256"] == digest(base.__file__)
    assert protocol["parent_driver_sha256"] == digest(previous.__file__)
    assert protocol["core_sha256"] == code_digest() == base.CORE_SHA
    assert protocol["reference_protocol_sha256"] == digest(Path(cfg["reference"])/"protocol.json")
    assert protocol["reference_config_sha256"] == digest(cfg["reference_config"])
    previous.verify_reference_compatibility(cfg, config(cfg["reference_config"]))
    prepared = json.loads((root/"prepared.json").read_text())
    assert prepared["protocol_sha256"] == digest(root/"protocol.json")
    for file, sha in prepared["artifacts"].items():
        assert digest(root/file) == sha, file
    assert digest(root/"prior_artifacts.json") == prepared["prior_artifacts_sha256"]
    return protocol


def select(cfg):
    root = Path(cfg["output"])
    protocol = validate(cfg)
    new = base.completed_runs(root, "tune")
    assert len(new) == expected(cfg, "tune")
    choices, details = previous.choose(new+reference_runs(cfg, "tune"), protocol["candidates"], cfg["models"])
    reference_selection = json.loads((Path(cfg["reference"])/"selection.json").read_text())
    for choice in choices:
        if choice["model"] == cfg["reused_model"]:
            old = next(r for r in reference_selection["choices"] if r["outer"] == choice["outer"] and r["model"] == choice["model"])
            assert choice["candidate"] == old["candidate"] and choice["refit_steps"] == old["refit_steps"]
    payload = {"protocol_sha256": digest(root/"protocol.json"), "tuning_complete_sha256": digest(root/"tune_complete.json"),
        "choices": [r for r in choices if r["model"] in cfg["new_models"]],
        "reused_choices": [r for r in choices if r["model"] == cfg["reused_model"]],
        "all_candidates": details, "selection_uses_outer_outcomes": False}
    if (root/"selection.json").exists():
        old = json.loads((root/"selection.json").read_text())
        assert all(old[k] == v for k, v in payload.items())
    else:
        json_write(root/"selection.json", {**payload, "frozen_utc": now()})
    print(json.dumps({"choices": payload["choices"], "reused_choices": payload["reused_choices"]}, indent=2))


def stage(cfg, phase, device):
    root = Path(cfg["output"])
    protocol = validate(cfg)
    common = dict(np.load(root/"development.npz", allow_pickle=False))
    if phase == "tune":
        cohort = protocol["tuning_cohort"]
    else:
        selected = json.loads((root/"selection.json").read_text())
        assert selected["protocol_sha256"] == digest(root/"protocol.json")
        assert selected["tuning_complete_sha256"] == digest(root/"tune_complete.json")
        cohort = [{**item, "seed": seed} for item in selected["choices"] for seed in cfg["training"]["refit_seeds"]]
    timer = base.session(root, cfg)
    data, key, manifests = None, None, {}
    for item in cohort:
        fold = next(f for f in protocol["folds"] if f["id"] == item["fold"])
        if key != fold["id"]:
            if data is not None:
                del data
            data = base.FoldData(common, dict(np.load(root/"folds"/fold["id"]/"features.npz", allow_pickle=False)), device)
            key = fold["id"]
        out, result = base.train_one(cfg, protocol, data, fold, item, device, timer["deadline"], item.get("refit_steps"))
        if result["status"] != "complete":
            raise RuntimeError("Training time guard reached; checkpoint preserved")
        manifests[str(out.relative_to(root))] = digest(out/"manifest.json")
    assert len(manifests) == expected(cfg, phase)
    json_write(root/f"{phase}_complete.json", {"completed_utc": now(), "fits": len(manifests),
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
        stage(cfg, "tune", args.device); select(cfg); stage(cfg, "refit", args.device)
    else:
        stage(cfg, args.stage, args.device)


if __name__ == "__main__":
    main()
