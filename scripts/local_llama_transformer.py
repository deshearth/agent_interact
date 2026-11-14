"""Train new matched transformers on the completed local Llama conversations.

The source pilot and all earlier predictor implementations remain immutable.
Question-group cross-validation is exploratory, not a fresh confirmatory test.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, config, digest, environment, json_write, seed_all
from scripts.argument_model import ResponseUpdate

CONFIG = Path("configs/local_llama_transformer.yaml")
CORE_SHA = "92c500581c6d74e0d5effa4b89f9c402bb76f7800151ac6dbb3b3afcae939dda"
OLD_FILES = {
    "scripts/argument_features.py": "d7aba211819a378cafd94a5b011e0ab69005cdfc958bef781195d8650c021b2f",
    "scripts/argument_model.py": "cfdc2ab81c7830d5a56b3a23b1e95df10d361d85e2b8021c311290ea3990fa84",
    "scripts/argument_response.py": "471a4f91f6d18f04e815bbc96f7c15de0efac015c3269cfa19333583b08b6e8e",
}
CONDITIONS = ("opinion_history", "arguments")
VARIANTS = ("original", "conclusion_only")


def now():
    return datetime.now(timezone.utc).isoformat()


def state_sha(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def verify_source(cfg):
    assert code_digest() == CORE_SHA, "Earlier core changed"
    for path, sha in OLD_FILES.items():
        assert digest(path) == sha, path
    root = Path(cfg["source"])
    manifest = json.loads((root/"manifest.json").read_text())
    assert manifest["status"] == "complete"
    for name, sha in manifest["artifacts"].items():
        assert digest(root/name) == sha, name
    rows = [json.loads(line) for line in (root/"examples.jsonl").read_text().splitlines()]
    static = json.loads((root/"static_context.json").read_text())
    assert len(rows) == 512 and len({r["question_group"] for r in rows}) == 8
    assert len({r["context_id"] for r in rows}) == len(rows)
    for r in rows:
        for v in VARIANTS:
            votes = r["outcomes"][v]["raw_votes"]
            assert len(votes) == 5 and set(votes) <= {-1, 1}
            assert r["outcomes"][v]["majority_sign"] == (1 if sum(votes) > 0 else -1)
    return rows, static


def question_folds(qids, seed):
    """Four disjoint two-question assessment folds; cyclic separate selection."""
    assert len(qids) == 8
    shuffled = np.random.default_rng(seed).permutation(sorted(qids)).tolist()
    chunks = [shuffled[i:i+2] for i in range(0, 8, 2)]
    result = []
    for i in range(4):
        test, val = chunks[i], chunks[(i+1) % 4]
        train = sorted(set(shuffled)-set(test)-set(val))
        assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
        result.append({"fold": i, "train": train, "selection": val, "heldout": test})
    return result


def build_inputs(rows, static, lookup):
    """Only pre-update inputs are consulted; outcome mutation cannot alter features."""
    width = max(sum(len(x) for x in r["inputs"]["delivered_messages"].values()) for r in rows)
    n, nodes = len(rows), width+1
    a = {"s": np.zeros((n, nodes), np.float32),
         "history": np.zeros((n, nodes, 6), np.float32),
         "structure": np.zeros((n, nodes, 3), np.float32),
         "persona_ids": np.zeros((n, nodes), np.int64),
         "question_ids": np.zeros(n, np.int64),
         "signs": np.zeros((n, width), np.float32),
         "stance": np.zeros((n, width), np.float32),
         "original_ids": np.zeros((n, width), np.int64),
         "conclusion_only_ids": np.zeros((n, width), np.int64)}
    questions = {q["qid"]: q for q in static["questions"]}
    for i, row in enumerate(rows):
        x = row["inputs"]
        assert row["question_group"] == x["qid"]
        J = np.array(static["graphs"][x["graph_id"]], dtype=np.float32)
        incoming = x["delivered_messages"]["agree"]+x["delivered_messages"]["disagree"]
        receiver = x["receiver"]
        senders = [m["sender"] for m in incoming]
        assert len(senders) == len(set(senders))
        assert set(senders) == set(np.flatnonzero(J[:, receiver]))
        for k, m in enumerate(incoming):
            assert m["relationship_sign"] == J[m["sender"], receiver]
            assert m["prompted_stance"] == x["current_spins"][m["sender"]]
            a["signs"][i, k] = m["relationship_sign"]
            a["stance"][i, k] = m["prompted_stance"]
            for v in VARIANTS:
                a[v+"_ids"][i, k] = lookup(m[v+"_message"])
        real_nodes = [receiver]+senders
        for j, node in enumerate(real_nodes):
            s = x["current_spins"][node]
            a["s"][i, j] = s
            a["persona_ids"][i, j] = lookup(static["personas"][node])
            a["history"][i, j, 3:] = [s == -1, s == 1, 1]
            if x["previous_spins"] is not None:
                prev = x["previous_spins"][node]
                a["history"][i, j, :3] = [prev == -1, prev == 1, 1]
            a["structure"][i, j] = np.log1p([(J[:, node] > 0).sum(), (J[:, node] < 0).sum(), (J[:, node] != 0).sum()])
        q = questions[x["qid"]]
        a["question_ids"][i] = lookup(f"Problem: {q['question']}\nA) {q['choices']['A']}\nB) {q['choices']['B']}")
    return a


class LocalResponseUpdate(ResponseUpdate):
    """The existing one-hop signed attention rule, packed around one receiver.

    Uses full frozen persona/question vectors in place of the old PCA context.
    Padding cannot affect the receiver and no learned agent/question IDs exist.
    """
    def __init__(self, **kwargs):
        super().__init__(kind="A2", text_dim=384, **kwargs)
        self.encoder[0] = nn.Linear(2+768+6+3, kwargs.get("d_model", 128))

    def forward(self, batch):
        s, context, signs = batch["s"], batch["context"], batch["signs"]
        b, width = signs.shape
        state = torch.stack([s == -1, s == 1], -1).to(context.dtype)
        h = self.encoder(torch.cat([state, context, batch["history"], batch["structure"]], -1))
        hn = self.norm(h)
        stance = torch.stack([batch["stance"] == -1, batch["stance"] == 1], -1).to(context.dtype)
        edge = self.edge_norm(hn[:, 1:] + self.text_projection(batch["text"]) + self.stance_projection(stance))
        queries = self.q(hn[:, 0]).view(b, self.heads, self.d_k)
        keys = self.k(edge).view(b, width, self.heads, self.d_k).transpose(1, 2)
        values = self.v(edge).view(b, width, self.heads, self.d_v).transpose(1, 2)
        mask = signs.ne(0)[:, None]
        isolated = ~mask.any(-1, keepdim=True)
        scores = (queries[:, :, None]*keys).sum(-1)/math.sqrt(self.d_k)
        scores = scores + (signs > 0)[:, None]*self.edge_bias[None, :, None, 0] + (signs < 0)[:, None]*self.edge_bias[None, :, None, 1]
        scores = torch.where(isolated, torch.zeros_like(scores), scores.masked_fill(~mask, -torch.inf))
        alpha = scores.softmax(-1)*mask
        signed = (signs > 0)[:, None]*self.gamma[None, :, None, 0] + (signs < 0)[:, None]*self.gamma[None, :, None, 1]
        rho = F.softplus(self.strength(batch["structure"][:, 0]))[..., None]
        aggregate = rho*((alpha*signed)[..., None]*values).sum(-2)
        h = h[:, 0] + self.dropout(self.out(aggregate.reshape(b, -1)))
        h = h + self.dropout(self.ffn(h))
        return self.classifier(h)


def prepare(cfg, device):
    rows, static = verify_source(cfg)
    root = Path(cfg["output"])
    root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(cfg["encoder_snapshot"])
    identity = {"config": cfg, "code_sha256": digest(__file__), "config_sha256": digest(CONFIG),
        "core_sha256": code_digest(), "old_predictor_sha256": OLD_FILES,
        "source_manifest_sha256": digest(Path(cfg["source"])/"manifest.json"),
        "encoder_files": {p.name: digest(p) for p in snapshot.iterdir() if p.is_file()},
        "folds": question_folds({r["question_group"] for r in rows}, cfg["split"]["seed"])}
    if (root/"protocol.json").exists():
        protocol = json.loads((root/"protocol.json").read_text())
        assert all(protocol[k] == v for k, v in identity.items()), "Frozen protocol changed"
    else:
        json_write(root/"protocol.json", {**identity, "created_utc": now(),
            "purpose": "Train new matched compact predictors of local Llama opinion updates.",
            "primary_target": "Natural original-message next opinion, majority of five raw Llama responses; A=1, B=0. Not mathematical answer correctness.",
            "selection": "Selection-question NLL, never heldout outcomes. 4 training / 2 selection / 2 heldout questions per fold; all agents, rounds and paired variants stay together.",
            "primary_contrast": "Arguments minus opinion/history, question-weighted heldout NLL; error, accuracy and two-class Brier alongside. Average seeds, not ensemble predictions.",
            "training_budget": "Both use the prior learning rate, 50-epoch maximum and patience 6. 256 receiver transitions per fold, minibatch32. Same nominal parameters, seed/initial weights and shuffled minibatch schedule; no hyperparameter search.",
            "input_control": "Both see current opinions, two-snapshot history, full question/choice and persona embeddings, degrees, sender stance and prescribed signed graph. Only the full model sees incoming argument vectors. Stance is verified equal to sender's current opinion. Text projection inactive in control.",
            "encoder": "Pinned frozen MiniLM. Each complete message/persona/question encoded separately; 510-token nonoverlapping windows, token-count-weighted mean of nonspecial hidden-token means, then one L2 normalization. No truncation, no PCA, no encoder fitting or question-ID table.",
            "paired_diagnostic": "After selection, evaluate both original and conclusions-only inputs on heldout questions without training on conclusion-only outcomes. Evaluate majority targets and model probability changes. Do not compare majority-probability predictions to first-token teacher probabilities as if they were the same target.",
            "baseline": "Persistence and training-fold global majority; majority probability uses add-one smoothing. No heldout outcome fitting.",
            "uncertainty": "Paired descriptive 10000-draw bootstrap of eight questions after averaging seeds. Overlapping training folds, one episode per question and prior inspection mean intervals are not confirmatory.",
            "limitations": "Eight reused objective mathematics questions; one graph and one replica; two synchronous updates. Frozen sentence vectors can lose logical details and pooling ignores inbox order. Observed histories/messages supplied; no autonomous rollout or universal-law claim.",
            "fresh_confirmation": False, "source_llm_calls": 0})
    protocol_sha = digest(root/"protocol.json")
    if (root/"features.json").exists():
        info = json.loads((root/"features.json").read_text())
        assert info["protocol_sha256"] == protocol_sha
        for name, sha in info["artifacts"].items():
            assert digest(root/name) == sha
        return
    texts, mapping = [""], {"": 0}
    def lookup(s):
        if s not in mapping:
            mapping[s] = len(texts)
            texts.append(s)
        return mapping[s]
    inputs = build_inputs(rows, static, lookup)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot.resolve()), local_files_only=True)
    encoder = AutoModel.from_pretrained(str(snapshot.resolve()), local_files_only=True).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    tokenized = tokenizer(texts[1:], add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    chunks = [(i+1, tokens[j:j+510]) for i, tokens in enumerate(tokenized) for j in range(0, len(tokens), 510)]
    chunks.sort(key=lambda r: len(r[1]))
    accum = np.zeros((len(texts), 384), np.float64)
    counts = np.zeros(len(texts), np.int64)
    started = time.monotonic()
    for offset in range(0, len(chunks), 32):
        pack = chunks[offset:offset+32]
        width = max(len(t) for _, t in pack)+2
        ids = torch.full((len(pack), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        pool = torch.zeros_like(ids)
        for k, (_, tokens) in enumerate(pack):
            seq = [tokenizer.cls_token_id]+tokens+[tokenizer.sep_token_id]
            ids[k, :len(seq)] = torch.tensor(seq, device=device)
            attention[k, :len(seq)] = 1
            pool[k, 1:len(seq)-1] = 1
        with torch.inference_mode():
            hidden = encoder(input_ids=ids, attention_mask=attention).last_hidden_state
            sums = (hidden*pool[..., None]).sum(1).cpu().numpy()
        for k, (i, tokens) in enumerate(pack):
            accum[i] += sums[k]
            counts[i] += len(tokens)
        if offset % 160 == 0:
            print(json.dumps({"stage": "encoding", "windows": min(offset+32, len(chunks)), "total": len(chunks)}), flush=True)
    assert np.array_equal(counts[1:], np.array([len(t) for t in tokenized]))
    vectors = (accum/np.linalg.norm(accum, axis=-1, keepdims=True).clip(1e-30)).astype(np.float32)
    assert np.isfinite(vectors).all()
    np.save(root/"vectors.npy", vectors, allow_pickle=False)
    np.savez_compressed(root/"inputs.npz", **inputs)
    json_write(root/"texts.json", texts)
    json_write(root/"features.json", {"protocol_sha256": protocol_sha, "created_utc": now(),
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()), "encoder_device": str(device),
        "unique_texts": len(texts)-1, "maximum_tokens": int(counts.max()), "windows": len(chunks),
        "multiwindow_texts": int((counts > 510).sum()), "tokens_dropped": 0,
        "seconds": time.monotonic()-started,
        "artifacts": {name: digest(root/name) for name in ("vectors.npy", "inputs.npz", "texts.json")}})


class Data:
    def __init__(self, cfg, device):
        self.rows, _ = verify_source(cfg)
        root = Path(cfg["output"])
        info = json.loads((root/"features.json").read_text())
        assert info["protocol_sha256"] == digest(root/"protocol.json")
        for name, sha in info["artifacts"].items():
            assert digest(root/name) == sha
        arrays = dict(np.load(root/"inputs.npz", allow_pickle=False))
        v = np.load(root/"vectors.npy", allow_pickle=False)
        context = np.concatenate([v[arrays["persona_ids"]], np.broadcast_to(v[arrays["question_ids"]][:, None], (*arrays["s"].shape, 384))], -1)
        self.a = {k: torch.as_tensor(arrays[k], device=device) for k in ("s", "history", "structure", "signs", "stance")}
        self.a["context"] = torch.as_tensor(context, device=device)
        self.text = {name: torch.as_tensor(v[arrays[name+"_ids"]], device=device) for name in VARIANTS}
        self.targets = {name: torch.tensor([r["outcomes"][name]["majority_sign"] == 1 for r in self.rows], dtype=torch.long, device=device) for name in VARIANTS}
        self.qids = np.array([r["question_group"] for r in self.rows])

    def ids(self, qids):
        return np.flatnonzero(np.isin(self.qids, qids))

    def batch(self, ids, condition, variant="original"):
        b = {k: v[ids] for k, v in self.a.items()}
        text = self.text[variant][ids]
        b["text"] = text if condition == "arguments" else torch.zeros_like(text)
        return b


def predict(model, data, ids, condition, variant="original"):
    model.eval()
    output = []
    with torch.inference_mode():
        for offset in range(0, len(ids), 64):
            logits = model(data.batch(ids[offset:offset+64], condition, variant))
            output.append(logits.log_softmax(-1).cpu().numpy())
    return np.concatenate(output)


def scores(logp, y):
    p = np.exp(logp[:, 1])
    err = float(np.mean(logp.argmax(-1) != y))
    return {"n": len(y), "error": err, "accuracy": 1-err,
            "nll": float(-logp[np.arange(len(y)), y].mean()),
            "brier_two_class": float(2*np.mean((p-y)**2))}


def train_one(cfg, data, fold, seed, condition, device):
    root = Path(cfg["output"])
    run = root/f"fold{fold['fold']}-{condition}-s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    identity = {"fold": fold, "seed": seed, "condition": condition,
        "protocol_sha256": digest(root/"protocol.json"), "features_sha256": digest(root/"features.json")}
    if (run/"manifest.json").exists():
        old = json.loads((run/"manifest.json").read_text())
        assert all(old[k] == v for k, v in identity.items())
        if old["status"] == "complete":
            for name, sha in old["artifacts"].items():
                assert digest(run/name) == sha
            return old
        assert (run/"resume.pt").exists(), "Interrupted before checkpoint; inspect before restarting"
    tr, val, held = (data.ids(fold[k]) for k in ("train", "selection", "heldout"))
    assert (len(tr), len(val), len(held)) == (256, 128, 128)
    seed_all(seed)
    model = LocalResponseUpdate(**cfg["model"]).to(device)
    initial_sha = state_sha(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    history, start_epoch, best, bad, elapsed_before = [], 0, float("inf"), 0, 0.
    if (run/"resume.pt").exists():
        ck = torch.load(run/"resume.pt", map_location=device, weights_only=False)
        assert ck["identity"] == identity
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        history, start_epoch, best, bad = ck["history"], ck["epoch"], ck["best"], ck["bad"]
        elapsed_before = ck["elapsed"]
        torch.set_rng_state(ck["cpu_rng"].cpu())
        if device == "mps":
            torch.mps.set_rng_state(ck["mps_rng"].cpu())
    json_write(run/"manifest.json", {**identity, "status": "running", "started_utc": now(), "initial_weights_sha256": initial_sha})
    started = time.monotonic()
    cfgtr = cfg["training"]
    reason = "epoch_limit"
    # A completed stopping epoch can be resumed without further training.
    for epoch in range(start_epoch, cfgtr["epochs"] if bad < cfgtr["patience"] else start_epoch):
        order = np.random.default_rng(seed+epoch).permutation(tr)
        model.train()
        losses = []
        for start in range(0, len(order), cfgtr["batch_size"]):
            ids = order[start:start+cfgtr["batch_size"]]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(data.batch(ids, condition)), data.targets["original"][ids])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfgtr["gradient_clip"])
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_score = scores(predict(model, data, val, condition), data.targets["original"][val].cpu().numpy())
        history.append({"epoch": epoch+1, "optimization_loss": float(np.mean(losses)), "selection_nll": val_score["nll"], "selection_error": val_score["error"]})
        if val_score["nll"] < best-cfgtr["min_delta"]:
            best, bad = val_score["nll"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch+1, "selection_nll": best}, run/"best.pt")
        else:
            bad += 1
        torch.save({"identity": identity, "model": model.state_dict(), "optimizer": opt.state_dict(),
            "epoch": epoch+1, "history": history, "best": best, "bad": bad,
            "cpu_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
            "elapsed": elapsed_before+time.monotonic()-started}, run/"resume.pt")
        if (epoch+1) % 10 == 0:
            print(json.dumps({"run": run.name, **history[-1]}), flush=True)
        if bad >= cfgtr["patience"]:
            reason = "early_stopping"
            break
    terminal = scores(predict(model, data, tr, condition), data.targets["original"][tr].cpu().numpy())
    terminal_sha = state_sha(model)
    selected = torch.load(run/"best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    records, summary = [], {}
    for split, ids in (("train", tr), ("selection", val), ("heldout", held)):
        for variant in (VARIANTS if split == "heldout" else ("original",)):
            logp = predict(model, data, ids, condition, variant)
            y = data.targets[variant][ids].cpu().numpy()
            summary[f"{split}/{variant}"] = scores(logp, y)
            for j, i in enumerate(ids):
                r = data.rows[i]
                records.append({"context_id": r["context_id"], "qid": r["question_group"],
                    "fold": fold["fold"], "seed": seed, "condition": condition,
                    "split": split, "variant": variant, "receiver": r["inputs"]["receiver"],
                    "round": r["inputs"]["update_round"], "y": int(y[j]), "prediction": int(logp[j].argmax()),
                    "p_A": float(np.exp(logp[j, 1])), "nll": float(-logp[j, y[j]]),
                    "brier_two_class": float(2*(np.exp(logp[j, 1])-y[j])**2),
                    "current_y": int(r["inputs"]["current_spins"][r["inputs"]["receiver"]] == 1),
                    "empty_inbox": r["audit"]["empty_inbox"],
                    "consistent_stances": r["audit"]["all_incoming_stances_explicit_and_consistent"]})
    pd.DataFrame(records).to_parquet(run/"predictions.parquet", index=False)
    json_write(run/"history.json", history)
    json_write(run/"metrics.json", summary)
    final_sha = state_sha(model)
    assert final_sha != initial_sha and terminal_sha != initial_sha, "Model was not trained"
    manifest = {**identity, "status": "complete", "completed_utc": now(), "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "inactive_text_parameters": sum(p.numel() for p in model.text_projection.parameters()) if condition == "opinion_history" else 0,
        "initial_weights_sha256": initial_sha, "selected_weights_sha256": final_sha,
        "selected_epoch": selected["epoch"], "terminal_epoch": history[-1]["epoch"],
        "steps": len(history)*8, "stopping_reason": reason, "seconds": elapsed_before+time.monotonic()-started,
        "terminal_train_metrics": terminal, "metrics": summary,
        "artifacts": {name: digest(run/name) for name in ("best.pt", "resume.pt", "predictions.parquet", "history.json", "metrics.json")}}
    json_write(run/"manifest.json", manifest)
    print(json.dumps({"run": run.name, "status": "complete", "selected_epoch": selected["epoch"], "seconds": manifest["seconds"]}), flush=True)
    return manifest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    cfg = config(CONFIG)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if args.stage == "prepare":
        prepare(cfg, args.device)
        return
    root = Path(cfg["output"])
    protocol = json.loads((root/"protocol.json").read_text())
    assert protocol["code_sha256"] == digest(__file__) and protocol["config_sha256"] == digest(CONFIG)
    data = Data(cfg, args.device)
    started = time.monotonic()
    result = []
    for fold in protocol["folds"]:
        for seed in cfg["training"]["seeds"]:
            pair = []
            for condition in CONDITIONS:
                done = train_one(cfg, data, fold, seed, condition, args.device)
                result.append(done)
                pair.append(done)
            assert pair[0]["initial_weights_sha256"] == pair[1]["initial_weights_sha256"]
            assert pair[0]["parameters"] == pair[1]["parameters"]
    verify_source(cfg)
    json_write(root/"training_complete.json", {"completed_utc": now(), "models": len(result),
        "all_complete": all(r["status"] == "complete" for r in result),
        "seconds_this_invocation": time.monotonic()-started, "environment": environment(),
        "manifests": {str(p.parent.relative_to(root)): digest(p) for p in sorted(root.glob("fold*/manifest.json"))}})


if __name__ == "__main__":
    main()
