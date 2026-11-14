"""Training-only capacity diagnostic on conflicting state-matched message pairs.

This deliberately selects labels to construct a memorization task. It is not an
estimate of generalization, a causal argument effect, or mathematical reasoning.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write, seed_all
from scripts import argument_mechanisms as previous
from scripts.argument_features import load_cache
from scripts.argument_model import ResponseUpdate
from scripts import joint_argument_encoding as session

ROOT = session.SESSION/"pair_capacity"
SETTINGS = {"pairs_per_question": 4, "seeds": [42, 43, 44], "text_scales": [1., 32.],
            "steps": 3000, "learning_rate": .0003, "weight_decay": 0., "dropout": 0.,
            "stop_error": 0., "stop_nll": .05, "aggregate_minutes": 15}


def receiver_logits(model, batch, receivers):
    """Exactly the original one-hop forward, evaluating only selected receivers."""
    s, context, signs = batch["s"], batch["context"], batch["signs"]
    b = len(s)
    rows = torch.arange(b, device=s.device)
    state = torch.stack([s == -1, s == 1], -1).to(context.dtype)
    structure = torch.stack([(signs > 0).sum(-1), (signs < 0).sum(-1), signs.abs().sum(-1)], -1).to(context.dtype).log1p()
    h = model.encoder(torch.cat([state, context, batch["history"], structure], -1))
    hn = model.norm(h)
    neighbors = batch["neighbors"][rows, receivers]
    sender = hn[rows[:, None], neighbors]
    stance = batch["stance"][rows, receivers]
    stance = torch.stack([stance == -1, stance == 1], -1).to(context.dtype)
    edge = model.edge_norm(sender+model.text_projection(batch["text"][rows, receivers])+model.stance_projection(stance))
    width = neighbors.shape[-1]
    values = model.v(edge).view(b, width, model.heads, model.d_v).permute(0, 2, 1, 3)
    signs = signs[rows, receivers]
    mask = signs.ne(0)[:, None]
    isolated = ~mask.any(-1, keepdim=True)
    q = model.q(hn[rows, receivers]).view(b, model.heads, model.d_k)
    k = model.k(edge).view(b, width, model.heads, model.d_k).permute(0, 2, 1, 3)
    scores = (q[:, :, None]*k).sum(-1)/math.sqrt(model.d_k)
    scores = scores+(signs > 0)[:, None]*model.edge_bias[None, :, None, 0]+(signs < 0)[:, None]*model.edge_bias[None, :, None, 1]
    scores = torch.where(isolated, torch.zeros_like(scores), scores.masked_fill(~mask, -torch.inf))
    alpha = scores.softmax(-1)*mask
    signed = (signs > 0)[:, None]*model.gamma[None, :, None, 0]+(signs < 0)[:, None]*model.gamma[None, :, None, 1]
    rho = F.softplus(model.strength(structure[rows, receivers]))[..., None]
    aggregate = rho*((alpha*signed)[..., None]*values).sum(-2)
    h = h[rows, receivers]+model.dropout(model.out(aggregate.reshape(b, -1)))
    h = h+model.dropout(model.ffn(h))
    return model.classifier(h)


def equivalence(model, batch, receivers):
    small = {k: v[:4] for k, v in batch.items() if isinstance(v, torch.Tensor)}
    selected = receivers[:4]
    full = model(small)[torch.arange(4), selected]
    fast = receiver_logits(model, small, selected)
    torch.testing.assert_close(full, fast, rtol=1e-5, atol=2e-6)
    target = small["target"][torch.arange(4), selected]
    F.cross_entropy(full, target).backward()
    grads = {k: p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    F.cross_entropy(receiver_logits(model, small, selected), target).backward()
    for k, p in model.named_parameters():
        torch.testing.assert_close(p.grad, grads[k], rtol=2e-4, atol=3e-6)
    model.zero_grad(set_to_none=True)
    return float((full-fast).abs().max().detach())


def prepare():
    cfg = config("configs/argument_mechanisms.yaml")
    arrays, index, vectors, metadata = load_cache(cfg, verify=True)
    index = previous.split_index(index, cfg)
    donors = np.load(previous.ROOT/"donors_s42.npy", allow_pickle=False)
    ds = previous.SplitSnapshots(arrays, index, vectors, "train", "full_arguments", donors)
    positions = {p: j for j, p in enumerate(ds.pairs)}
    pairs = np.load(session.AUDIT/"ambiguous_training_pairs.npy", allow_pickle=False)
    chosen, counts = [], {}
    for pair in pairs:
        qid = index[int(pair[0, 0])]["qid"]
        if counts.get(qid, 0) >= SETTINGS["pairs_per_question"]:
            continue
        assert pair[0, 3] == -pair[1, 3]
        assert all(index[int(p[0])]["diagnostic_role"] == "train" for p in pair)
        chosen.append(pair)
        counts[qid] = counts.get(qid, 0)+1
    selected = np.array(chosen)
    records = selected.reshape(-1, 4)
    batch = default_collate([ds[positions[(int(e), int(t))]] for e, t, r, y in records])
    receivers = torch.tensor(records[:, 2], dtype=torch.long)
    targets = batch["target"][torch.arange(len(records)), receivers]
    np.testing.assert_array_equal(targets.numpy(), (records[:, 3] == 1).astype(np.int64))
    edge_text = batch["text"][torch.arange(len(records)), receivers]
    differences = (edge_text[::2]-edge_text[1::2]).flatten(1).norm(dim=-1)
    assert (differences > 1e-6).all()
    ROOT.mkdir(parents=True, exist_ok=True)
    identity = {"code_sha256": digest(__file__), "model_sha256": digest("scripts/argument_model.py"),
        "source_pairs_sha256": digest(session.AUDIT/"ambiguous_training_pairs.npy"),
        "features_sha256": digest(Path(cfg["data"]["cache"])/"metadata.json"), "settings": SETTINGS}
    if (ROOT/"protocol.json").exists():
        saved = json.loads((ROOT/"protocol.json").read_text())
        assert all(saved[k] == v for k, v in identity.items())
        np.testing.assert_array_equal(np.load(ROOT/"selected_pairs.npy"), selected)
    else:
        np.save(ROOT/"selected_pairs.npy", selected)
        json_write(ROOT/"protocol.json", {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "pair_counts": counts, "pairs": len(selected), "examples": len(records),
            "selected_sha256": digest(ROOT/"selected_pairs.npy"),
            "minimum_pair_embedding_l2": float(differences.min()),
            "purpose": "Can the existing graph transformer use message-vector differences to memorize opposite outcomes with identical nontext state?",
            "selection": "First at most four audited conflicting pairs per fitting question; only replicas 0/1. Selection uses outcomes and is intentionally unsuitable for generalization evaluation.",
            "baseline": "Every pair has identical receiver and neighbor model inputs except argument vectors and opposite labels. Deterministic stance/history predictors necessarily have 50% error on this balanced set.",
            "training": "Full-batch cross entropy, no dropout/weight decay, no held-out stopping. Same original 185890-parameter model; scales 1 and 32. Training-fit success is not evidence of reasoning, causality, or transfer."})
    return batch, receivers, targets


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "run"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    batch, receivers, targets = prepare()
    seed_all(42)
    model = ResponseUpdate(dropout=0.)
    maximum_error = equivalence(model, batch, receivers)
    no_text = {**batch, "text": torch.zeros_like(batch["text"])}
    with torch.no_grad():
        z = receiver_logits(model, no_text, receivers)
    torch.testing.assert_close(z[::2], z[1::2], rtol=0, atol=2e-6)
    json_write(ROOT/"checks.json", {"forward_max_abs_error": maximum_error, "gradient_equivalence": True,
        "opposite_targets_identical_nontext_logits": True, "nontext_minimum_error": .5})
    if args.stage == "check":
        print(json.dumps({"checks": "passed", "examples": len(targets)}), flush=True)
        return
    stop = min(time.time()+SETTINGS["aggregate_minutes"]*60, session.deadline(reserve=900))
    result = []
    for scale in SETTINGS["text_scales"]:
        for seed in SETTINGS["seeds"]:
            out = ROOT/f"scale{int(scale)}-s{seed}"
            out.mkdir(exist_ok=True)
            if (out/"metrics.json").exists():
                result.append(json.loads((out/"metrics.json").read_text()))
                continue
            seed_all(seed)
            model = ResponseUpdate(dropout=0.)
            optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=0.)
            scaled = {**batch, "text": batch["text"]*scale}
            start, curves = time.monotonic(), []
            for step in range(SETTINGS["steps"]+1):
                logits = receiver_logits(model, scaled, receivers)
                loss = F.cross_entropy(logits, targets)
                assert torch.isfinite(loss)
                error = float((logits.argmax(-1) != targets).float().mean())
                if step % 100 == 0 or (error == 0. and loss.item() < SETTINGS["stop_nll"]):
                    row = {"step": step, "nll": float(loss.detach()), "error": error}
                    curves.append(row)
                    json_write(out/"curve.json", curves)
                    print(json.dumps({"scale": scale, "seed": seed, **row}), flush=True)
                if error == 0. and loss.item() < SETTINGS["stop_nll"]:
                    reason = "memorized"
                    break
                if time.time() >= stop or step == SETTINGS["steps"]:
                    reason = "time_limit" if time.time() >= stop else "step_limit"
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            torch.save({"model": model.state_dict(), "step": step, "scale": scale, "seed": seed}, out/"model.pt")
            np.save(out/"logits.npy", logits.detach().numpy())
            row = {"scale": scale, "seed": seed, "step": step, "error": error, "nll": float(loss.detach()),
                "seconds": time.monotonic()-start, "reason": reason, "protocol_sha256": digest(ROOT/"protocol.json"),
                "artifacts": {f: digest(out/f) for f in ("model.pt", "logits.npy", "curve.json")}}
            json_write(out/"metrics.json", row)
            result.append(row)
            json_write(ROOT/"summary.json", result)
            if reason == "time_limit":
                return


if __name__ == "__main__":
    main()
