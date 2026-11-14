"""Matched comparison of frozen message-only and question-conditioned encodings."""
from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, device_for, digest, json_write
from scripts import argument_response as original
from scripts import argument_mechanisms as previous
from scripts import question_context as question
from scripts import joint_argument_encoding as session
from scripts import joint_argument_encoding_windowed as encoding
from scripts.argument_features import load_cache

CONFIG = "configs/joint_argument_context.yaml"
ROOT = session.SESSION/"joint_predictors"
VARIANTS = encoding.MODES
CONDITIONS = ("full_arguments", "shuffled_arguments")


def implementation():
    return {**question.implementation(), "joint_context_sha256": digest(__file__),
            "windowed_encoding_sha256": digest(encoding.__file__)}


def make_model(cfg, kind):
    if kind not in VARIANTS:
        raise ValueError(kind)
    return question.make_model(cfg, "full_embedding")


def cohort(cfg):
    return [{"kind": kind, "condition": condition, "seed": seed}
            for seed in cfg["training"]["seeds"] for kind in VARIANTS for condition in CONDITIONS]


def prepare(cfg):
    arrays, index, old_vectors, metadata = load_cache(cfg, verify=True)
    parent = json.loads((question.ROOT/"protocol.json").read_text())
    assert parent["implementation"] == question.implementation()
    for field in ("data", "model", "training", "evaluation", "diagnostic", "question"):
        assert cfg[field] == parent["config"][field]
    for file, sha in parent["prepared_sha256"].items():
        assert digest(question.ROOT/file) == sha
    assert tuple(cfg["encoding"]["variants"]) == VARIANTS
    complete = json.loads((encoding.ROOT/"complete.json").read_text())
    assert complete["protocol_sha256"] == digest(encoding.ROOT/"protocol.json")
    assert complete["tokens_truncated"] == 0 and complete["all_question_and_message_tokens_covered"]
    for file, sha in complete["artifacts"].items():
        assert digest(encoding.ROOT/file) == sha
    with np.load(cfg["encoding"]["mapping"], allow_pickle=False) as mapping:
        ids = mapping["text_ids"]
        np.testing.assert_array_equal(mapping["source_text_ids"][ids], arrays["text_ids"])
    index = previous.split_index(index, cfg)
    assert index == json.loads((previous.ROOT/"index.json").read_text())
    arrays = question.augmented_arrays(arrays, index, cfg["encoding"]["question_features"], "full_embedding")
    arrays = {**arrays, "text_ids": ids}
    baselines = {}
    for seed in cfg["training"]["seeds"]:
        for condition in original.CONDITIONS:
            name = f"full_embedding-{condition}-s{seed}"
            run, ev = question.ROOT/name, question.ROOT/"evaluation"/name
            manifest = json.loads((run/"manifest.json").read_text())
            assert manifest["status"] == "complete" and manifest["parameters"] == 382498
            for file, sha in manifest["artifact_sha256"].items():
                assert digest(run/file) == sha
            evaluation = json.loads((ev/"identity.json").read_text())
            assert evaluation["run_manifest_sha256"] == digest(run/"manifest.json")
            for file, sha in evaluation["artifacts"].items():
                assert digest(ev/file) == sha
            baselines[name] = {"manifest_sha256": digest(run/"manifest.json"), "evaluation_sha256": digest(ev/"identity.json")}
    identity = {"implementation": implementation(), "config": cfg, "config_sha256": digest(CONFIG),
        "parent_protocol_sha256": digest(question.ROOT/"protocol.json"),
        "feature_metadata_sha256": digest(Path(cfg["data"]["cache"])/"metadata.json"),
        "encoding_complete_sha256": digest(encoding.ROOT/"complete.json"),
        "mapping_sha256": digest(cfg["encoding"]["mapping"]), "cohort": cohort(cfg), "baselines": baselines}
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/"protocol.json"
    if path.exists():
        saved = json.loads(path.read_text())
        assert all(saved[k] == v for k, v in identity.items())
    else:
        json_write(path, {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Does allowing frozen message tokens to attend to the full question and choices improve predictive argument use?",
            "matched_control": "Same frozen local MiniLM weights, BERT paired segments, message-content-token pooling and L2 normalization. Only question tokens differ: absent versus observed complete question and choices. Long questions use weighted nonoverlapping windows with the whole message in each; no tokens are truncated.",
            "predictor": "Identical 382498-parameter graph transformer with full cached question context in both conditions. All training/splits/stopping match question_context_v1. Message inputs remain individual 384-dimensional vectors.",
            "reference": "Reuse nine frozen full-question-context models. Three stance-only references have identically zero message inputs; six old argument references show the effect relative to the prior standalone sentence pooling.",
            "shuffle": "Reuse the exact role-specific donor position bijections, which match on question and other prescribed features. New IDs preserve every original message/edge placement and donors remain within question.",
            "primary_comparison": "Question-conditioned minus matched message-only encoding with full arguments; NLL primary, accuracy alongside, train/familiar/unseen reported separately.",
            "secondary_comparisons": "Each full-argument condition against the shared no-text reference, matched substitutions and original standalone embeddings.",
            "limitations": "Frozen encoder, not supervised token fine-tuning or a mathematical solver. Long-question windows cannot interact with each other. Sixteen fitting and four repeatedly used unseen development questions; descriptive uncertainty only. No new source-LLM calls, test scores or graph changes."})
    adapted = deepcopy(metadata)
    fitting = sorted({r["qid"] for r in index if r["diagnostic_role"] == "train"})
    adapted["questions"] = {"train": fitting, "validation": fitting}
    return {kind: (arrays, index, np.load(encoding.ROOT/f"{kind}.npy", mmap_mode="r", allow_pickle=False), adapted)
            for kind in VARIANTS}


@contextmanager
def runtime(seed):
    donors = np.load(previous.ROOT/f"donors_s{seed}.npy", allow_pickle=False)
    class TrainingSnapshots(previous.SplitSnapshots):
        def __init__(self, arrays, index, vectors, partition, condition, ignored):
            super().__init__(arrays, index, vectors, partition, condition, donors)
    names = ("ROOT", "CONFIG", "implementation", "make_model", "ResponseSnapshots")
    before = {name: getattr(original, name) for name in names}
    original.ROOT, original.CONFIG = ROOT, CONFIG
    original.implementation, original.make_model = implementation, make_model
    original.ResponseSnapshots = TrainingSnapshots
    try:
        yield
    finally:
        for name, value in before.items():
            setattr(original, name, value)


def evaluate_run(item, cfg, cache, device, smoke=False):
    name = f"{item['kind']}-{item['condition']}-s{item['seed']}"
    run = (ROOT/"smoke" if smoke else ROOT)/name
    out = (ROOT/"smoke_evaluation" if smoke else ROOT/"evaluation")/name
    out.mkdir(parents=True, exist_ok=True)
    identity = {"run_manifest_sha256": digest(run/"manifest.json"), "best_sha256": digest(run/"best.pt"),
                "protocol_sha256": digest(ROOT/"protocol.json")}
    record = out/"identity.json"
    if record.exists():
        saved = json.loads(record.read_text())
        assert all(saved[k] == v for k, v in identity.items())
        for file, sha in saved["artifacts"].items():
            assert digest(out/file) == sha
        return
    arrays, index, vectors, _ = cache
    donors = np.load(previous.ROOT/f"donors_s{item['seed']}.npy", allow_pickle=False)
    model = make_model(cfg, item["kind"]).to(device)
    model.load_state_dict(torch.load(run/"best.pt", map_location=device, weights_only=False)["model"])
    names = []
    for role in ("train", "familiar", "unseen"):
        ds = previous.SplitSnapshots(arrays, index, vectors, role, item["condition"], donors)
        original.save_predictions(out, role, model, ds, cfg, device)
        names.extend([f"{role}_{s}" for s in ("predictions.parquet", "metrics.json", "by_question.csv")])
        if role != "train" and item["condition"] == "full_arguments":
            ds = previous.SplitSnapshots(arrays, index, vectors, role, "shuffled_arguments", donors)
            original.save_predictions(out, f"{role}_substitution", model, ds, cfg, device)
            names.extend([f"{role}_substitution_{s}" for s in ("predictions.parquet", "metrics.json", "by_question.csv")])
    json_write(record, {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
                        "artifacts": {file: digest(out/file) for file in names}})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "run"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    cfg = config(CONFIG)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    device = device_for(args.device)
    caches = prepare(cfg)
    if args.stage == "prepare":
        print(json.dumps({"prepared": str(ROOT), "cohort_size": len(cohort(cfg))}), flush=True)
        return
    if args.stage == "smoke":
        for kind in VARIANTS:
            item = {"kind": kind, "condition": "full_arguments", "seed": 42}
            with runtime(42):
                original.train_one(item, cfg, caches[kind], device, time.monotonic()+120, smoke=True)
            evaluate_run(item, cfg, caches[kind], device, smoke=True)
        return
    path = ROOT/"plan.json"
    plan = json.loads(path.read_text()) if path.exists() else {"protocol_sha256": digest(ROOT/"protocol.json"), "elapsed_seconds": 0., "runs": []}
    assert plan["protocol_sha256"] == digest(ROOT/"protocol.json")
    started = time.monotonic()
    stop_at = min(started+cfg["training"]["aggregate_minutes"]*60-plan["elapsed_seconds"],
                  started+session.deadline(reserve=900)-time.time())
    try:
        for item in cohort(cfg):
            if time.monotonic() >= stop_at:
                break
            current = caches[item["kind"]]
            with runtime(item["seed"]):
                manifest = original.train_one(item, cfg, current, device, stop_at)
            row = {**item, "status": manifest["status"]}
            plan["runs"] = [r for r in plan["runs"] if (r["kind"], r["condition"], r["seed"]) !=
                            (item["kind"], item["condition"], item["seed"])]+[row]
            if manifest["status"] == "complete":
                evaluate_run(item, cfg, current, device)
            json_write(path, plan)
            if manifest["status"] != "complete":
                break
        plan["complete"] = len(plan["runs"]) == len(cohort(cfg)) and all(r["status"] == "complete" for r in plan["runs"])
        if plan["complete"]:
            plan["completed_utc"] = datetime.now(timezone.utc).isoformat()
    finally:
        plan["elapsed_seconds"] += time.monotonic()-started
        json_write(path, plan)


if __name__ == "__main__":
    main()
