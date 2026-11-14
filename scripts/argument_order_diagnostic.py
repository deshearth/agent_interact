"""Exploratory inbox-order diagnostic under the released generation algorithm.

Actual historical prompt order is not logged in the legacy episodes. Therefore
the released-order input is a reconstruction assumption, not observed truth.
"""
from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, device_for, digest, json_write
from scripts import argument_response as original
from scripts import argument_mechanisms as previous
from scripts import question_context as question
from scripts import joint_argument_encoding as session
from scripts.argument_features import load_cache

CONFIG = "configs/argument_order_diagnostic.yaml"
ROOT = session.SESSION/"order_diagnostic"
VARIANTS = ("released_order", "random_order")


def positions(name, round_index, neighbors, signs, mode):
    result = np.zeros((*signs.shape, 3), dtype=np.float32)
    for receiver in range(len(signs)):
        active = np.flatnonzero(signs[receiver])
        items = sorted(active.tolist(), key=lambda k: int(neighbors[receiver, k]))
        seed = f"{name}|{round_index+1}|{receiver}"
        if mode == "random_order":
            seed = "independent-order-control|"+seed
        elif mode != "released_order":
            raise ValueError(mode)
        random.Random(seed).shuffle(items)
        for sign in (1, -1):
            group = [k for k in items if signs[receiver, k] == sign]
            for rank, k in enumerate(group):
                result[receiver, k] = [2*rank/(len(group)-1)-1 if len(group)>1 else 0,
                                      float(rank == 0), float(rank == len(group)-1)]
    return result


class OrderUpdate(question.QuestionResponseUpdate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.order_projection = nn.Linear(3, self.text_projection.out_features, bias=False)
        nn.init.zeros_(self.order_projection.weight)

    def forward(self, batch):
        s, context, signs = batch["s"], batch["context"], batch["signs"]
        b, n = s.shape
        width = signs.shape[-1]
        state = torch.stack([s == -1, s == 1], -1).to(context.dtype)
        structure = torch.stack([(signs > 0).sum(-1), (signs < 0).sum(-1), signs.abs().sum(-1)], -1).to(context.dtype).log1p()
        h = self.encoder(torch.cat([state, context, batch["history"], structure], -1))
        hn = self.norm(h)
        sender = hn[torch.arange(b, device=s.device)[:, None, None], batch["neighbors"]]
        stance = torch.stack([batch["stance"] == -1, batch["stance"] == 1], -1).to(context.dtype)
        edge = self.edge_norm(sender+self.text_projection(batch["text"])+self.stance_projection(stance)+self.order_projection(batch["order"]))
        values = self.v(edge).view(b, n, width, self.heads, self.d_v).permute(0, 3, 1, 2, 4)
        mask = signs.ne(0)[:, None]
        isolated = ~mask.any(-1, keepdim=True)
        queries = self.q(hn).view(b, n, self.heads, self.d_k).permute(0, 2, 1, 3)
        keys = self.k(edge).view(b, n, width, self.heads, self.d_k).permute(0, 3, 1, 2, 4)
        scores = (queries[..., None, :]*keys).sum(-1)/math.sqrt(self.d_k)
        scores = scores+(signs > 0)[:, None]*self.edge_bias[None, :, None, None, 0]+(signs < 0)[:, None]*self.edge_bias[None, :, None, None, 1]
        scores = torch.where(isolated, torch.zeros_like(scores), scores.masked_fill(~mask, -torch.inf))
        alpha = scores.softmax(-1)*mask
        signed = (signs > 0)[:, None]*self.gamma[None, :, None, None, 0]+(signs < 0)[:, None]*self.gamma[None, :, None, None, 1]
        rho = F.softplus(self.strength(structure)).permute(0, 2, 1)[..., None]
        aggregate = rho*((alpha*signed)[..., None]*values).sum(-2)
        social = self.out(aggregate.permute(0, 2, 1, 3).reshape(b, n, -1))
        h = h+self.dropout(social)
        h = h+self.dropout(self.ffn(h))
        return self.classifier(h)


class OrderSnapshots(previous.SplitSnapshots):
    def __getitem__(self, item):
        result = super().__getitem__(item)
        e, t = self.pairs[item]
        result["order"] = torch.from_numpy(self.arrays["order"][e, t])
        return result


def make_model(cfg, kind):
    if kind not in VARIANTS:
        raise ValueError(kind)
    return OrderUpdate(question_dim=cfg["question"]["dimension"], text_dim=cfg["data"]["encoder_dimension"],
        **{k: v for k, v in cfg["model"].items() if k != "max_parameters"})


def implementation():
    return {**question.implementation(), "order_diagnostic_sha256": digest(__file__)}


def cohort(cfg):
    return [{"kind": kind, "condition": condition, "seed": seed} for seed in cfg["training"]["seeds"]
            for kind in VARIANTS for condition in original.CONDITIONS]


def prepare(cfg):
    arrays, index, vectors, metadata = load_cache(cfg, verify=True)
    parent = json.loads((question.ROOT/"protocol.json").read_text())
    assert parent["implementation"] == question.implementation()
    for field in ("data", "model", "training", "evaluation", "diagnostic", "question"):
        assert cfg[field] == parent["config"][field]
    source = Path(cfg["order"]["source"])
    manifest = json.loads(Path("data/data_manifest.json").read_text())["files"]
    assert digest(source) == manifest[str(source.relative_to("data/raw/upstream"))]["sha256"]
    for file, sha in parent["prepared_sha256"].items():
        assert digest(question.ROOT/file) == sha
    index = previous.split_index(index, cfg)
    assert index == json.loads((previous.ROOT/"index.json").read_text())
    arrays = question.augmented_arrays(arrays, index, cfg["order"]["question_features"], "full_embedding")
    identity = {"implementation": implementation(), "config": cfg, "config_sha256": digest(CONFIG),
        "parent_protocol_sha256": digest(question.ROOT/"protocol.json"), "source_sha256": digest(source),
        "features_sha256": digest(Path(cfg["data"]["cache"])/"metadata.json"), "cohort": cohort(cfg)}
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/"protocol.json"
    if path.exists():
        saved = json.loads(path.read_text())
        assert all(saved[k] == v for k, v in identity.items())
        for file, sha in saved["prepared_sha256"].items():
            assert digest(ROOT/file) == sha
    else:
        for mode in VARIANTS:
            values = np.stack([np.stack([positions(row["name"], t, arrays["neighbors"][e], arrays["signs"][e], mode)
                for t in range(8)]) for e, row in enumerate(index)])
            np.save(ROOT/f"{mode}.npy", values)
        json_write(path, {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "prepared_sha256": {f"{m}.npy": digest(ROOT/f"{m}.npy") for m in VARIANTS},
            "purpose": "Could omitted inbox presentation order explain weak argument benefits?",
            "historical_limitation": "The legacy raw episodes do not record full prompts or inbox order. Released source contains deterministic sorting/shuffling, but historical algorithm identity is unverified. This experiment is conditional on that reconstruction; it cannot establish or exclude the actual order effect.",
            "features": "Normalized rank, first, last inside friendly and unfriendly inbox groups, separately. Sender IDs stay aligned to the original edge slots. Nonedges remain zero. No targets enter reconstruction.",
            "control": "Independent deterministic random permutation for every inbox, with the same position features and architecture. This controls extra capacity and random instance features. It is not a reversed order that the model could simply invert.",
            "architecture": "Original full-question-context graph model plus zero-initialized 3-to-128 edge-position map before edge LayerNorm. Shared weights and RNG initially agree. Total 382882 parameters.",
            "training": "Same split, stopping, 3 seeds, and stance/full/shuffled argument conditions as the prior full-question study. Donor message substitutions remain unchanged; order belongs to the target slot.",
            "comparison": "Reconstructed order versus random order; each versus frozen no-order full-question references. NLL primary, error alongside. Both familiar and unseen evaluation excluded from stopping; repeated development use remains exploratory."})
    adapted = deepcopy(metadata)
    fitting = sorted({r["qid"] for r in index if r["diagnostic_role"] == "train"})
    adapted["questions"] = {"train": fitting, "validation": fitting}
    return {mode: ({**arrays, "order": np.load(ROOT/f"{mode}.npy", allow_pickle=False)}, index, vectors, adapted) for mode in VARIANTS}


@contextmanager
def runtime(seed):
    donors = np.load(previous.ROOT/f"donors_s{seed}.npy", allow_pickle=False)
    class TrainingSnapshots(OrderSnapshots):
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
        ds = OrderSnapshots(arrays, index, vectors, role, item["condition"], donors)
        original.save_predictions(out, role, model, ds, cfg, device)
        names.extend([f"{role}_{s}" for s in ("predictions.parquet", "metrics.json", "by_question.csv")])
        if role != "train" and item["condition"] == "full_arguments":
            ds = OrderSnapshots(arrays, index, vectors, role, "shuffled_arguments", donors)
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
            with runtime(item["seed"]):
                manifest = original.train_one(item, cfg, caches[item["kind"]], device, stop_at)
            plan["runs"] = [r for r in plan["runs"] if (r["kind"], r["condition"], r["seed"]) !=
                            (item["kind"], item["condition"], item["seed"])]+[{**item, "status": manifest["status"]}]
            if manifest["status"] == "complete":
                evaluate_run(item, cfg, caches[item["kind"]], device)
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
