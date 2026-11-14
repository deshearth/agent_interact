from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from ..common import digest, json_write, new_run


def author_module(cfg):
    root = Path(cfg["data"]["root"])
    path = root / "raw/upstream/res/utils.py"
    manifest = json.loads((root / "data_manifest.json").read_text())
    if digest(path) != manifest["files"]["res/utils.py"]["sha256"]:
        raise ValueError("Author code differs from pinned checksum")
    spec = importlib.util.spec_from_file_location("physics_of_agents_author_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def design(phi, J, s):
    pos = np.einsum("bij,bj->bi", np.maximum(J, 0), s)
    neg = np.einsum("bij,bj->bi", np.minimum(J, 0), s)
    return np.concatenate([phi, np.stack([pos, neg, pos-neg], -1)], -1)


def fit_baseline(kind, dataset, cfg):
    a = dataset.arrays
    e, t = np.array(dataset.pairs).T
    s, y = a["spins"][e, t], a["spins"][e, t+1]
    if kind == "P0":
        return {"kind": kind, "parameters": 0}
    if kind == "P1":
        counts = np.ones((2, 2), dtype=np.float64)
        for current in (-1, 1):
            for target in (-1, 1):
                counts[int(current == 1), int(target == 1)] += ((s == current) & (y == target)).sum()
        return {"kind": kind, "smoothing": "add-one per cell", "probabilities": (counts/counts.sum(1, keepdims=True)).tolist(), "parameters": 2}
    module = author_module(cfg)
    # Author standardization expects float64. Float32 means can give constant
    # question columns a spurious tiny variance on small subsets.
    x = design(a["context"][e], a["J"][e], s).reshape(-1, 19).astype(np.float64)
    parsed = y.ravel() != 0
    w = module.two_stage_fit(x[:, :16], x[:, 16:18], x[:, 18], parsed, (y.ravel()[parsed] == 1).astype(float))
    return {"kind": kind, "weights": w.tolist(), "parameters": len(w),
            "adapter": "Pinned author two_stage_fit; signed negative channel preserved"}


def baseline_logp(record, s, phi, J):
    if record["kind"] == "P0":
        # Deterministic reference: likelihood/Brier explicitly omitted in reporting.
        p = np.stack([s == -1, s == 1], -1).astype(np.float64)
        with np.errstate(divide="ignore"):
            return np.log(p)
    if record["kind"] == "P1":
        return np.log(np.asarray(record["probabilities"])[(s == 1).astype(int)])
    logits = design(phi, J, s) @ np.asarray(record["weights"])
    return np.stack([-np.logaddexp(0, logits), -np.logaddexp(0, -logits)], -1)


def reproduce(cfg):
    from ..data.prepare import load_prepared
    from threadpoolctl import threadpool_limits
    if cfg["data"].get("context") != "author_pca":
        raise ValueError("Author reproduction requires the author's original context artifacts")
    arrays, index, directory = load_prepared(cfg)
    module = author_module(cfg)
    # Use the author's float64 preprocessing directly for this reproduction.
    # The controlled neural dataset deliberately uses float32 and is separate.
    upstream = Path(cfg["data"]["root"]) / "raw/upstream/data"
    rows = [r for r in json.loads((upstream / "clean_runs.json").read_text())
            if r["model"] == "gpt-4o-mini" and r["mode"] == "objective"]
    bank = {}
    for split in ("train", "test"):
        for line in (upstream / "obj" / f"{split}.jsonl").read_text().splitlines():
            q = json.loads(line)
            bank[q["question"]] = {"split": split, "qid": q["qid"],
                                   "q": np.array(q["embedding_pca10"][:3]),
                                   "truth": {"A": 1, "B": -1}[q["answer"]]}
    items = sorted(json.loads((upstream / "obj/persona_embeddings.json").read_text())["items"], key=lambda x: x["idx"])
    persona = module.pca3(np.asarray([x["embedding"] for x in items], dtype=float))
    banks = {"objective": bank}
    gclass = module.graph_classes(rows, banks)
    original_x, original_y, ep = module.build_xy("gpt-4o-mini", "objective", rows, banks, {"objective": persona}, gclass)
    train = (ep["split"] == "train") & np.isin(ep["gcls"], ["seen", "train_only"])
    test = (ep["split"] == "test") & np.isin(ep["gcls"], ["seen", "fresh"])
    flat = original_x[train].reshape(-1, 19)
    x = np.concatenate([flat[:, module.FIELD], flat[:, 17:19], (flat[:, 17]-flat[:, 18])[:, None]], -1)
    y = original_y[train].ravel()
    parsed = y != 0
    with threadpool_limits(limits=cfg["training"]["torch_threads"]):
        w = module.two_stage_fit(x[:, :16], x[:, 16:18], x[:, 18], parsed, (y[parsed] == 1).astype(float))
    test_x = original_x[test]
    signed = test_x[..., 17:19]
    full_design = np.concatenate([test_x[..., module.FIELD], signed,
                                  (signed[..., 0]-signed[..., 1])[..., None]], -1)
    logits = full_design @ w
    one = np.where(logits > 0, 1, -1)
    rollout = module.discrete_rollout(ep["phi"][test], ep["J"][test], ep["spins"][test, 0], w, 3)
    truth, initial = original_y[test], ep["spins"][test, 0]
    result = {"scope": "Author reproduction only; official 20 training questions; not controlled model selection",
              "training_episodes": int(train.sum()), "test_episodes": int(test.sum()), "weights": w.tolist(),
              "source_revision": cfg["data"]["revision"], "preprocessing": "Author compact question PCA and persona PCA", "metrics": {}}
    for label, role in (("seen_random", "seen"), ("fresh_random", "fresh")):
        mask = ep["gcls"][test] == role
        result["metrics"][label] = {"one_step_fcba_percent": module.fcba(one, truth, initial, mask),
                                     "rollout_fcba_percent": module.fcba(rollout, truth, initial, mask),
                                     "one_step_accuracy_percent": module.raw_acc(one, truth, mask),
                                     "rollout_accuracy_percent": module.raw_acc(rollout, truth, mask)}
    output = new_run("author-reproduction")
    from ..data.prepare import graph_hash
    lookup = {(row["qid"], row["graph_hash"], row["replica"]): row["episode_id"] for row in index}
    ids = [lookup[(qid, graph_hash(graph), int(rep))] for qid, graph, rep in zip(ep["qid"][test], ep["J"][test], ep["rep"][test])]
    np.savez_compressed(output / "predictions.npz", episode_ids=np.array(ids),
                        one_step=one, rollout=rollout, target=truth)
    result["run"] = str(output)
    json_write(output / "reproduction.json", result)
    json_write("reports/author_reproduction.json", result)
    Path("reports/author_reproduction.md").write_text("# Author baseline reproduction\n\n" + result["scope"] +
        ".\n\n```json\n" + json.dumps(result["metrics"], indent=2) + "\n```\n\n" +
        "This checks the released fitting procedure. It is separate from the 16-question controlled comparison.\n")
    print(json.dumps(result, indent=2))
    return output
