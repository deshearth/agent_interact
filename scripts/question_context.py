"""Restore full cached question/choice embeddings with a matched-capacity control.

Only development questions are fitted/transformed. All old study implementations,
checkpoints and caches remain immutable. No new embedding or source-LLM calls.
"""
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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, device_for, digest, json_write
from scripts import argument_response as original
from scripts import argument_mechanisms as previous
from scripts.argument_features import load_cache
from scripts.argument_model import ResponseUpdate

CONFIG = "configs/question_context.yaml"
ROOT = Path("runs/question_context_v1")
VARIANTS = ("pca3_matched", "full_embedding")
BASE_ROOT = previous.ROOT
BASE_IMPLEMENTATION = previous.implementation
ROLES = previous.ROLES


class QuestionEncoder(nn.Module):
    """Keep the original local encoder; add question input before its GELU."""
    def __init__(self, local, question_dim):
        super().__init__()
        self.local, self.question_dim = local, question_dim
        with torch.random.fork_rng(devices=[]):
            self.question_projection = nn.Linear(question_dim, local[0].out_features, bias=False)
        nn.init.zeros_(self.question_projection.weight)

    def forward(self, x):
        # ResponseUpdate concatenates state(2), context(16 + question),
        # history(6), and degrees(3). The question is identical at all nodes.
        if x.shape[-1] != 27+self.question_dim:
            raise ValueError("Unexpected augmented context width")
        local = torch.cat([x[..., :18], x[..., 18+self.question_dim:]], -1)
        question = x[:, 0, 18:18+self.question_dim]
        h = self.local[0](local)+self.question_projection(question)[:, None]
        return self.local[2](self.local[1](h))


class QuestionResponseUpdate(ResponseUpdate):
    def __init__(self, question_dim=1536, **kwargs):
        super().__init__(kind="A2", **kwargs)
        self.encoder = QuestionEncoder(self.encoder, question_dim)


def make_model(cfg, kind):
    if kind not in VARIANTS:
        raise ValueError(kind)
    model = QuestionResponseUpdate(question_dim=cfg["question"]["dimension"],
        text_dim=cfg["data"]["encoder_dimension"],
        **{k: v for k, v in cfg["model"].items() if k != "max_parameters"})
    assert sum(p.numel() for p in model.parameters()) < cfg["model"]["max_parameters"]
    return model


def implementation():
    return {**BASE_IMPLEMENTATION(), "question_context_sha256": digest(__file__)}


def fit_features(values, fit_indices, width=3):
    """Center/fit PCA/fit scales on fitting questions only, with no targets."""
    x = np.asarray(values, dtype=np.float64)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    if len(fit_indices) <= width or len(set(fit_indices.tolist())) != len(fit_indices):
        raise ValueError("Need distinct fitting questions exceeding PCA width")
    center = x[fit_indices].mean(0)
    _, singular, vh = np.linalg.svd(x[fit_indices]-center, full_matrices=False)
    components = vh[:width]
    full = x-center
    reduced = (full@components.T)@components
    full_scale = float(np.sqrt((full[fit_indices]**2).sum(-1).mean()))
    reduced_scale = float(np.sqrt((reduced[fit_indices]**2).sum(-1).mean()))
    if min(full_scale, reduced_scale) <= 0:
        raise ValueError("Degenerate question representation")
    return {"mean": center, "components": components,
            "full_embedding": (full/full_scale).astype(np.float32),
            "pca3_matched": (reduced/reduced_scale).astype(np.float32),
            "full_scale": np.array(full_scale), "pca3_scale": np.array(reduced_scale),
            "training_singular_values": singular,
            "pca3_variance_fraction": np.array((singular[:width]**2).sum()/(singular**2).sum())}


def augmented_arrays(arrays, index, feature_path, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    with np.load(feature_path, allow_pickle=False) as saved:
        lookup = dict(zip(saved["qids"].tolist(), saved[variant]))
    q = np.stack([lookup[row["qid"]] for row in index]).astype(np.float32)
    q = np.broadcast_to(q[:, None], (*arrays["context"].shape[:2], q.shape[-1]))
    result = {**arrays, "context": np.concatenate([arrays["context"], q], -1)}
    np.testing.assert_array_equal(result["context"][..., :16], arrays["context"])
    return result


def cohort(cfg):
    return [{"kind": kind, "condition": condition, "seed": seed}
            for seed in cfg["training"]["seeds"] for kind in cfg["question"]["variants"]
            for condition in original.CONDITIONS]


def prepare(cfg, cache):
    arrays, index, vectors, metadata = cache
    previous_protocol = json.loads((BASE_ROOT/"protocol.json").read_text())
    assert previous_protocol["implementation"] == BASE_IMPLEMENTATION()
    assert previous_protocol["config"]["diagnostic"] == cfg["diagnostic"]
    for section in ("data", "model", "training", "evaluation"):
        assert previous_protocol["config"][section] == cfg[section]
    assert tuple(cfg["question"]["variants"]) == VARIANTS
    index = previous.split_index(index, cfg)
    assert index == json.loads((BASE_ROOT/"index.json").read_text())
    for name, sha in previous_protocol["prepared_sha256"].items():
        assert digest(BASE_ROOT/name) == sha
    source = Path(cfg["question"]["source"])
    pinned = json.loads(Path("data/data_manifest.json").read_text())["files"]
    assert digest(source) == pinned[str(source.relative_to("data/raw/upstream"))]["sha256"]
    raw = json.loads(source.read_text())
    assert raw["model"] == cfg["question"]["encoder"] and raw["dim"] == cfg["question"]["dimension"]
    qids = sorted({r["qid"] for r in index})
    records = {r["qid"]: r for r in raw["items"] if r["qid"] in qids}
    assert len(records) == len(qids) == 20
    for qid in qids:
        row = next(r for r in index if r["qid"] == qid)
        forms = [separator.join([row["statement"], f"A: {row['choices']['A']}", f"B: {row['choices']['B']}"])
                 for separator in ("\n", "\\n")]
        assert records[qid]["question"] in forms, f"Question/choices mismatch: {qid}"
    fitting = sorted({r["qid"] for r in index if r["diagnostic_role"] == "train"})
    features = fit_features([records[q]["embedding"] for q in qids], [qids.index(q) for q in fitting],
                            cfg["question"]["reduced_components"])
    with np.load(Path(cfg["data"]["prepared"])/"question_pca.npz", allow_pickle=False) as old:
        np.testing.assert_allclose(features["mean"], old["mean"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(features["components"], old["components"], rtol=0, atol=1e-12)
    baselines = {}
    for seed in cfg["training"]["seeds"]:
        for condition in original.CONDITIONS:
            name = f"A2-{condition}-s{seed}"
            run = BASE_ROOT/name
            manifest = json.loads((run/"manifest.json").read_text())
            assert manifest["status"] == "complete"
            for file, sha in manifest["artifact_sha256"].items():
                assert digest(run/file) == sha
            evaluation = BASE_ROOT/"evaluation"/name/"identity.json"
            saved = json.loads(evaluation.read_text())
            assert saved["run_manifest_sha256"] == digest(run/"manifest.json")
            for file, sha in saved["artifacts"].items():
                assert digest(evaluation.parent/file) == sha
            baselines[name] = {"manifest_sha256": digest(run/"manifest.json"),
                               "evaluation_sha256": digest(evaluation)}
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/"protocol.json"
    identity = {"implementation": implementation(), "config_sha256": digest(CONFIG), "config": cfg,
                "source_embedding_sha256": digest(source), "baseline_protocol_sha256": digest(BASE_ROOT/"protocol.json"),
                "feature_metadata_sha256": digest(Path(cfg["data"]["cache"])/"metadata.json"),
                "cohort": cohort(cfg), "baselines": baselines}
    if path.exists():
        protocol = json.loads(path.read_text())
        if any(protocol[k] != v for k, v in identity.items()):
            raise ValueError("Question-context protocol changed; use a new namespace")
        for file, sha in protocol["prepared_sha256"].items():
            assert digest(ROOT/file) == sha
        with np.load(ROOT/"question_features.npz", allow_pickle=False) as saved:
            for key, value in features.items():
                np.testing.assert_array_equal(saved[key], value)
    else:
        np.savez(ROOT/"question_features.npz", qids=np.array(qids), fit_qids=np.array(fitting), **features)
        json_write(ROOT/"questions.json", [{"qid": q, "text": records[q]["question"],
                   "fit": q in fitting} for q in qids])
        protocol = {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "prepared_sha256": {file: digest(ROOT/file) for file in ("question_features.npz", "questions.json")},
            "roles": previous_protocol["roles"], "shuffle_audit": previous_protocol["shuffle_audit"],
            "purpose": "Does restoring information beyond the original three question/choice PCA components improve prediction or unlock extra value from argument text?",
            "inputs": "Exact cached text-embedding-3-small vectors of complete problem plus choices, 1536 dimensions. Only the 16 fitting and four development-evaluation questions are transformed. All original persona/context/history/graph/message inputs remain unchanged.",
            "preprocessing": "Fit mean and PCA on the 16 fitting questions only. Full branch uses the centered embedding; matched control reconstructs only its original first three PCs in the same 1536-dimensional coordinates. Each branch is divided by its own fitting-question RMS L2 norm. No per-question or outcome normalization; no fitting on held-out questions.",
            "architecture": "Both new conditions use the identical 1536-to-128 linear question projection, added before the original local encoder GELU. It is initialized to zero, preserving the original initial function; shared weights and RNG also agree. Extra 196608 parameters, total 382498. Question projection is computed once per community and broadcast to nodes. Original 16-dimensional context remains available.",
            "capacity_control": "The PCA3 reconstruction adds no information beyond existing question features. Nominal parameter counts, initialization and training settings match the full-vector condition. Effective input rank differs by design; this does not equalize effective statistical capacity or optimization geometry.",
            "baseline": "Reuse nine frozen A2 models from the identical conversation split, with no added question branch. No refitting or selection of those baselines.",
            "training": "Same frozen trainer and configuration as argument_mechanisms_v1: fit replicas 0/1, select checkpoints on replica 2 of the same 16 questions, evaluate familiar replica 3 and the four unseen development questions. 64 updates per epoch, 50 epochs, 5000-update outer cap, six-epoch patience, three seeds. Neither evaluation cohort selects checkpoints.",
            "primary_comparisons": ["full_embedding versus pca3_matched, separately for stance-only and full arguments, question-mean NLL primary and accuracy alongside", "Argument contribution (full arguments minus stance-only) under each question representation; paired question-level difference of those contributions"],
            "secondary_comparisons": ["Both new representations versus the frozen original PCA3 model", "Full versus matched-shuffled arguments; original matching coverage/limitations retained", "Saved full-model prediction sensitivity to matched message substitutions"],
            "uncertainty": "Average seeds within question, then 1000 paired question bootstraps for each within-cohort contrast. Four unseen versus 16 familiar questions; cohorts differ and are not paired. Descriptive pointwise intervals only, after repeated development use.",
            "limitations": "This tests a PCA compression bottleneck in a frozen question embedding, not raw-token reasoning, a jointly encoded question/argument pair, or a fine-tuned encoder. Cached embeddings can omit crucial mathematical distinctions. Only 16 fitting questions; additional dimensions may overfit. No new APIs/LLM calls, test scoring, graph changes, or autonomous rollouts."}
        json_write(path, protocol)
    adapted_meta = deepcopy(metadata)
    adapted_meta["questions"] = {"train": fitting, "validation": fitting}
    return (arrays, index, vectors, adapted_meta)


@contextmanager
def runtime(seed):
    donors = np.load(BASE_ROOT/f"donors_s{seed}.npy", allow_pickle=False)

    class TrainingSnapshots(previous.SplitSnapshots):
        def __init__(self, arrays, index, vectors, partition, condition, ignored_original_donors):
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


def cache_for(cache, kind):
    arrays, index, vectors, metadata = cache
    return (augmented_arrays(arrays, index, ROOT/"question_features.npz", kind), index, vectors, metadata)


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
    donors = np.load(BASE_ROOT/f"donors_s{item['seed']}.npy", allow_pickle=False)
    model = make_model(cfg, item["kind"]).to(device)
    model.load_state_dict(torch.load(run/"best.pt", map_location=device, weights_only=False)["model"])
    names = []
    for role in ("train", "familiar", "unseen"):
        ds = previous.SplitSnapshots(arrays, index, vectors, role, item["condition"], donors)
        original.save_predictions(out, role, model, ds, cfg, device)
        names.extend([f"{role}_{suffix}" for suffix in ("predictions.parquet", "metrics.json", "by_question.csv")])
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
    cache = prepare(cfg, load_cache(cfg, verify=True))
    if args.stage == "prepare":
        print(json.dumps({"prepared": str(ROOT), "cohort_size": len(cohort(cfg))}))
        return
    variants = {kind: cache_for(cache, kind) for kind in VARIANTS}
    if args.stage == "smoke":
        for kind in VARIANTS:
            item = {"kind": kind, "condition": "full_arguments", "seed": 42}
            with runtime(42):
                original.train_one(item, cfg, variants[kind], device, time.monotonic()+120, smoke=True)
            evaluate_run(item, cfg, variants[kind], device, smoke=True)
        return
    path = ROOT/"plan.json"
    plan = json.loads(path.read_text()) if path.exists() else {
        "protocol_sha256": digest(ROOT/"protocol.json"), "elapsed_seconds": 0., "runs": []}
    assert plan["protocol_sha256"] == digest(ROOT/"protocol.json")
    started = time.monotonic()
    stop_at = started+cfg["training"]["aggregate_minutes"]*60-plan["elapsed_seconds"]
    try:
        for item in cohort(cfg):
            if time.monotonic() >= stop_at:
                break
            current = variants[item["kind"]]
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
