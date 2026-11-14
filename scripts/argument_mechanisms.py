"""Isolate within-question transfer and separate text fusion, development only.

All source implementations and caches remain frozen. Selection uses independent
conversations on the 16 fitting questions; neither evaluation cohort selects
checkpoints. They are research development data, not fresh confirmation.
"""
from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, device_for, digest, json_write
from scripts import argument_response as original
from scripts.argument_features import (ResponseSnapshots, load_cache, matched_donors,
                                       shuffle_groups)
from scripts.argument_model import ResponseUpdate

CONFIG = "configs/argument_mechanisms.yaml"
ROOT = Path("runs/argument_mechanisms_v1")
BASE_IMPLEMENTATION = original.implementation
BASE_MAKE_MODEL = original.make_model
ROLES = ("train", "validation", "familiar", "unseen")


class SeparateTextUpdate(ResponseUpdate):
    """Separate normalized state/text pathways into attention keys and values.

The original stance-only function and initialization are preserved exactly.
Text has its own normalization and K/V maps; it is never added to sender state
before normalization. This tests one alternative fusion architecture, not all
possible architectures, and does not change the frozen text encoder.
"""
    def __init__(self, **kwargs):
        super().__init__(kind="A2", **kwargs)
        width = self.text_projection.out_features
        self.text_norm = nn.LayerNorm(width, elementwise_affine=False)
        # Extra initialization must not change shared weights or dropout RNG.
        with torch.random.fork_rng(devices=[]):
            self.text_k = nn.Linear(width, self.heads*self.d_k, bias=False)
            self.text_v = nn.Linear(width, self.heads*self.d_v, bias=False)

    def forward(self, batch):
        s, context, signs = batch["s"], batch["context"], batch["signs"]
        b, n = s.shape
        width = signs.shape[-1]
        state = torch.stack([s == -1, s == 1], -1).to(context.dtype)
        structure = torch.stack([(signs > 0).sum(-1), (signs < 0).sum(-1),
                                 signs.abs().sum(-1)], -1).to(context.dtype).log1p()
        h = self.encoder(torch.cat([state, context, batch["history"], structure], -1))
        hn = self.norm(h)
        sender = hn[torch.arange(b, device=s.device)[:, None, None], batch["neighbors"]]
        stance = torch.stack([batch["stance"] == -1, batch["stance"] == 1], -1).to(context.dtype)
        edge = self.edge_norm(sender + self.stance_projection(stance))
        text = self.text_norm(self.text_projection(batch["text"]))
        values = (self.v(edge)+self.text_v(text)).view(
            b, n, width, self.heads, self.d_v).permute(0, 3, 1, 2, 4)
        keys = (self.k(edge)+self.text_k(text)).view(
            b, n, width, self.heads, self.d_k).permute(0, 3, 1, 2, 4)
        queries = self.q(hn).view(b, n, self.heads, self.d_k).permute(0, 2, 1, 3)
        mask = signs.ne(0)[:, None]
        isolated = ~mask.any(-1, keepdim=True)
        scores = (queries[..., None, :]*keys).sum(-1)/math.sqrt(self.d_k)
        scores = scores + (signs > 0)[:, None]*self.edge_bias[None, :, None, None, 0] + (
            signs < 0)[:, None]*self.edge_bias[None, :, None, None, 1]
        scores = torch.where(isolated, torch.zeros_like(scores), scores.masked_fill(~mask, -torch.inf))
        alpha = scores.softmax(-1)*mask
        signed = (signs > 0)[:, None]*self.gamma[None, :, None, None, 0] + (
            signs < 0)[:, None]*self.gamma[None, :, None, None, 1]
        rho = F.softplus(self.strength(structure)).permute(0, 2, 1)[..., None]
        aggregate = rho*((alpha*signed)[..., None]*values).sum(-2)
        social = self.out(aggregate.permute(0, 2, 1, 3).reshape(b, n, -1))
        h = h+self.dropout(social)
        h = h+self.dropout(self.ffn(h))
        return self.classifier(h)


def split_index(index, cfg):
    result = []
    spec = cfg["diagnostic"]
    mapping = {rep: role for key, role in (("fit_replicas", "train"),
               ("selection_replicas", "validation"), ("familiar_evaluation_replicas", "familiar"))
               for rep in spec[key]}
    if len(mapping) != sum(len(spec[k]) for k in (
            "fit_replicas", "selection_replicas", "familiar_evaluation_replicas")):
        raise ValueError("Replica splits overlap")
    for row in index:
        if row["question_split"] not in ("train", "validation"):
            raise ValueError("Only development questions are allowed")
        role = mapping[row["replica"]] if row["question_split"] == "train" else "unseen"
        result.append({**row, "diagnostic_role": role})
    return result


class SplitSnapshots(ResponseSnapshots):
    def __init__(self, arrays, index, vectors, partition, condition, donors):
        if partition not in ROLES:
            raise ValueError("Only development roles are allowed")
        super().__init__(arrays, index, vectors, "train", condition, donors)
        self.pairs = [(e, t) for e, row in enumerate(index) if row["diagnostic_role"] == partition
                      for t in range(arrays["text_ids"].shape[1]) if arrays["valid"][e, t]]


def make_model(cfg, kind):
    if kind == "A2":
        return BASE_MAKE_MODEL(cfg, kind)
    if kind != "separate":
        raise ValueError(kind)
    model = SeparateTextUpdate(text_dim=cfg["data"]["encoder_dimension"],
        **{k: v for k, v in cfg["model"].items() if k != "max_parameters"})
    assert sum(p.numel() for p in model.parameters()) < cfg["model"]["max_parameters"]
    return model


def implementation():
    return {**BASE_IMPLEMENTATION(), "mechanisms_sha256": digest(__file__)}


def cohort(cfg):
    return [{"kind": kind, "condition": condition, "seed": seed}
            for seed in cfg["training"]["seeds"]
            for kind, conditions in (("A2", original.CONDITIONS),
                                     ("separate", ("full_arguments", "shuffled_arguments")))
            for condition in conditions]


def split_groups(arrays, index):
    # Original matching uses question_split. Substitute the four disjoint roles
    # only for donor construction; preserve original question labels elsewhere.
    matching_index = [{**row, "question_split": row["diagnostic_role"]} for row in index]
    return shuffle_groups(arrays, matching_index, arrays["text_ids"], arrays["stance"],
                          arrays["neighbors"], arrays["signs"])


def audit_donors(arrays, index, donors):
    ids = arrays["text_ids"]
    real = np.flatnonzero(ids)
    dest = donors.ravel()[real]
    np.testing.assert_array_equal(np.sort(dest), real)
    e, t, r, k = np.unravel_index(real, ids.shape)
    de, dt, dr, dk = np.unravel_index(dest, ids.shape)
    qids = np.array([row["qid"] for row in index])
    roles = np.array([row["diagnostic_role"] for row in index])
    sender = arrays["neighbors"][e, r, k]
    dsender = arrays["neighbors"][de, dr, dk]
    for a, b in ((roles[e], roles[de]), (qids[e], qids[de]), (t, dt), (r, dr), (sender, dsender),
                 (arrays["signs"][e, r, k], arrays["signs"][de, dr, dk]),
                 (arrays["stance"][e, t, r, k], arrays["stance"][de, dt, dr, dk]),
                 (arrays["spins"][e, t, r], arrays["spins"][de, dt, dr]),
                 (arrays["spins"][e, t, sender], arrays["spins"][de, dt, dsender])):
        np.testing.assert_array_equal(a, b)
    assert np.all((real == dest) | (e != de))
    changed = ids.ravel()[dest] != ids.ravel()[real]
    return {role: {"messages": int((roles[e] == role).sum()),
                   "different_text_fraction": float(changed[roles[e] == role].mean())}
            for role in ROLES}


def geometry(arrays, index, vectors):
    """Input-only centroids fitted exclusively to the two training replicas."""
    qids = sorted({r["qid"] for r in index if r["diagnostic_role"] == "train"})
    qmeans, qsmeans, total, count = {}, {}, np.zeros(vectors.shape[1], dtype=np.float64), 0
    train_ids = set()
    for qid in qids:
        episodes = [e for e, r in enumerate(index) if r["qid"] == qid and r["diagnostic_role"] == "train"]
        ids = arrays["text_ids"][episodes].ravel()
        stances = arrays["stance"][episodes].ravel()[ids != 0]
        ids = ids[ids != 0]
        train_ids.update(ids.tolist())
        x = np.array(vectors[ids], dtype=np.float64)
        total += x.sum(0)
        count += len(x)
        qmeans[qid] = x.mean(0)
        for stance in (-1, 1):
            qsmeans[qid, stance] = x[stances == stance].mean(0)
    center = total/count
    result = []
    for role in ROLES:
        rows = [e for e, r in enumerate(index) if r["diagnostic_role"] == role]
        ids = arrays["text_ids"][rows].ravel()
        ids = ids[ids != 0]
        out = {"role": role, "messages": len(ids), "unique_texts": len(set(ids.tolist())),
               "exact_text_seen_in_fit_fraction": float(np.isin(ids, list(train_ids)).mean())}
        if role != "unseen":
            sums = np.zeros(3, dtype=np.float64)
            nearest_q_correct = 0
            centers = np.stack([qmeans[q] for q in qids])
            for qid in qids:
                es = [e for e in rows if index[e]["qid"] == qid]
                ids = arrays["text_ids"][es].ravel()
                stance = arrays["stance"][es].ravel()[ids != 0]
                ids = ids[ids != 0]
                for start in range(0, len(ids), 4096):
                    x = np.array(vectors[ids[start:start+4096]], dtype=np.float64)
                    st = stance[start:start+4096]
                    qs = np.stack([qsmeans[qid, int(s)] for s in st])
                    sums += [((x-center)**2).sum(), ((x-qmeans[qid])**2).sum(), ((x-qs)**2).sum()]
                    distances = (centers**2).sum(-1)[None]-2*x@centers.T
                    nearest_q_correct += int((distances.argmin(-1) == qids.index(qid)).sum())
            out.update(question_centroid_explained_fraction=float(1-sums[1]/sums[0]),
                       question_stance_centroid_explained_fraction=float(1-sums[2]/sums[0]),
                       nearest_question_centroid_accuracy=nearest_q_correct/out["messages"])
        result.append(out)
    return {"fit_only_centroids": True, "target_labels_used": False,
            "weighting": "message occurrence weighted, including repeated text",
            "interpretation": "Variance explained is not mutual information or evidence that remaining dimensions lack reasoning.",
            "rows": result}


def prepare(cfg, cache):
    arrays, index, vectors, metadata = cache
    index = split_index(index, cfg)
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/"protocol.json"
    identity = {"implementation": implementation(), "config": cfg, "config_sha256": digest(CONFIG),
                "feature_metadata_sha256": digest(Path(cfg["data"]["cache"])/"metadata.json"),
                "cohort": cohort(cfg)}
    if path.exists():
        protocol = json.loads(path.read_text())
        if any(protocol[k] != v for k, v in identity.items()):
            raise ValueError("Frozen protocol changed; use a new namespace")
        for name, sha in protocol["prepared_sha256"].items():
            if digest(ROOT/name) != sha:
                raise ValueError(f"Prepared artifact changed: {name}")
    else:
        json_write(ROOT/"index.json", index)
        groups = split_groups(arrays, index)
        audits = {}
        names = ["index.json", "geometry.json"]
        for seed in cfg["training"]["seeds"]:
            donors = matched_donors(groups, arrays["text_ids"].shape, seed)
            audits[str(seed)] = audit_donors(arrays, index, donors)
            name = f"donors_s{seed}.npy"
            np.save(ROOT/name, donors, allow_pickle=False)
            names.append(name)
        json_write(ROOT/"geometry.json", geometry(arrays, index, vectors))
        roles = {role: {"episodes": sum(r["diagnostic_role"] == role for r in index),
                        "questions": sorted({r["qid"] for r in index if r["diagnostic_role"] == role}),
                        "episode_ids": sorted(r["episode_id"] for r in index if r["diagnostic_role"] == role)}
                 for role in ROLES}
        protocol = {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "prepared_sha256": {name: digest(ROOT/name) for name in names}, "roles": roles, "shuffle_audit": audits,
            "purpose": "Separate generalization across conversations from transfer across questions, and test one independently normalized text K/V branch.",
            "split": "Fit replicas 0/1 of all 16 training questions (256 episodes); select stopping on replica 2 (128). Evaluate familiar questions on replica 3 (128) and unseen questions on all 128 existing development-validation episodes. No evaluation episode enters fitting or selection in this study.",
            "preprocessing": "Reuse original frozen embeddings and context. Question PCA fits the same 16 questions present in fitting replicas; persona context uses the same 32 personas. No target labels enter preprocessing. Identical strings across episodes are allowed and overlap is reported.",
            "architecture": "Original A2 versus separate normalized text and sender/stance branches into independent attention K/V maps. Adds 24,576 parameters. Stance-only outputs, shared initialization, training RNG and shared gradients are identical when text is zero, verified by tests; share the A2 stance control.",
            "comparisons": ["Within each evaluation cohort/fusion: full minus stance and full minus matched-shuffled",
                            "Difference in full-minus-stance accuracy gains between familiar and unseen cohorts (descriptive, not paired across different questions)",
                            "Separate versus additive fusion, with both full and shuffled text reported"],
            "training": "Original trainer, 64 batches per epoch; same optimizer and epoch limits for all runs, choose checkpoints by question-mean selection-replica NLL. Three initializations. Neither evaluation cohort determines stopping.",
            "shuffle": "Rebuild all matched donor bijections separately within fit, selection, familiar evaluation and unseen evaluation. Preserve question, round, sender, receiver, sign, communicated stance and current opinions. Graph identity and complete history are not matched; paraphrases may replace each other.",
            "uncertainty": "Average seeds within question, then 1000 paired question bootstraps for contrasts within cohort. 16 familiar versus only 4 unseen questions; the cohorts differ in content and are not paired. No confirmatory transfer-interaction claim.",
            "limitations": "Exploratory after many development analyses. Fixed replica split, not cross-validation. Halves original fitting episodes and changes selection cohort, so do not directly attribute changes versus previous full-data metrics to architecture. No token-level reasoning encoder training or source-LLM intervention; no test data or new LLM calls."}
        json_write(path, protocol)
    adapted_meta = deepcopy(metadata)
    adapted_meta["questions"] = {"train": protocol["roles"]["train"]["questions"],
                                 "validation": protocol["roles"]["validation"]["questions"]}
    return (arrays, index, vectors, adapted_meta)


@contextmanager
def runtime(seed):
    donors = np.load(ROOT/f"donors_s{seed}.npy", allow_pickle=False)

    class TrainingSnapshots(SplitSnapshots):
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


def evaluate_run(item, cfg, cache, device, smoke=False):
    name = f"{item['kind']}-{item['condition']}-s{item['seed']}"
    run = (ROOT/"smoke" if smoke else ROOT)/name
    out = (ROOT/"smoke_evaluation" if smoke else ROOT/"evaluation")/name
    out.mkdir(parents=True, exist_ok=True)
    identity = {"run_manifest_sha256": digest(run/"manifest.json"),
                "best_sha256": digest(run/"best.pt"), "protocol_sha256": digest(ROOT/"protocol.json")}
    record = out/"identity.json"
    if record.exists():
        saved = json.loads(record.read_text())
        if any(saved[k] != v for k, v in identity.items()):
            raise ValueError("Evaluation source changed")
        for name, sha in saved["artifacts"].items():
            assert digest(out/name) == sha
        return
    arrays, index, vectors, _ = cache
    donors = np.load(ROOT/f"donors_s{item['seed']}.npy", allow_pickle=False)
    model = make_model(cfg, item["kind"]).to(device)
    model.load_state_dict(torch.load(run/"best.pt", map_location=device, weights_only=False)["model"])
    names = []
    for role in ("train", "familiar", "unseen"):
        ds = SplitSnapshots(arrays, index, vectors, role, item["condition"], donors)
        original.save_predictions(out, role, model, ds, cfg, device)
        names.extend([f"{role}_{suffix}" for suffix in ("predictions.parquet", "metrics.json", "by_question.csv")])
        if role != "train" and item["condition"] == "full_arguments":
            ds = SplitSnapshots(arrays, index, vectors, role, "shuffled_arguments", donors)
            original.save_predictions(out, f"{role}_substitution", model, ds, cfg, device)
            names.extend([f"{role}_substitution_{s}" for s in ("predictions.parquet", "metrics.json", "by_question.csv")])
    json_write(record, {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
                        "artifacts": {name: digest(out/name) for name in names}})


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
        print(json.dumps({"prepared": str(ROOT), "roles": {r: sum(x["diagnostic_role"] == r for x in cache[1]) for r in ROLES}}))
        return
    if args.stage == "smoke":
        item = {"kind": "separate", "condition": "full_arguments", "seed": 42}
        with runtime(42):
            original.train_one(item, cfg, cache, device, time.monotonic()+120, smoke=True)
        evaluate_run(item, cfg, cache, device, smoke=True)
        return
    plan_path = ROOT/"plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {
        "protocol_sha256": digest(ROOT/"protocol.json"), "elapsed_seconds": 0., "runs": []}
    if plan["protocol_sha256"] != digest(ROOT/"protocol.json"):
        raise ValueError("Plan changed")
    start = time.monotonic()
    stop_at = start+cfg["training"]["aggregate_minutes"]*60-plan["elapsed_seconds"]
    try:
        for item in cohort(cfg):
            if time.monotonic() >= stop_at:
                break
            with runtime(item["seed"]):
                manifest = original.train_one(item, cfg, cache, device, stop_at)
            row = {**item, "status": manifest["status"]}
            plan["runs"] = [r for r in plan["runs"] if (r["kind"], r["condition"], r["seed"]) !=
                            (item["kind"], item["condition"], item["seed"])]+[row]
            if manifest["status"] == "complete":
                evaluate_run(item, cfg, cache, device)
            json_write(plan_path, plan)
            if manifest["status"] != "complete":
                break
        plan["complete"] = len(plan["runs"]) == len(cohort(cfg)) and all(r["status"] == "complete" for r in plan["runs"])
        if plan["complete"]:
            plan["completed_utc"] = datetime.now(timezone.utc).isoformat()
    finally:
        plan["elapsed_seconds"] += time.monotonic()-start
        json_write(plan_path, plan)


if __name__ == "__main__":
    main()
