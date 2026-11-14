"""Frozen MiniLM encoding of message tokens with/without the full question.

Both controls use identical paired-input tokenization and pool only the message
tokens. Contextual hidden states can attend to question/choice tokens in one
condition; the other receives an empty question. No supervised encoder update.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from scripts.argument_features import load_cache

SESSION = Path("runs/argument_investigation_20261108")
ROOT = SESSION/"joint_encoding"
AUDIT = SESSION/"input_audit"
SNAPSHOT = Path(".cache/huggingface/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
MODES = ("message_only", "question_conditioned")


def deadline(reserve=600):
    saved = json.loads((SESSION/"session.json").read_text())
    return datetime.fromisoformat(saved["deadline_utc"]).timestamp()-reserve


def load_encoder(device):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT.resolve()), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(str(SNAPSHOT.resolve()), local_files_only=True, trust_remote_code=False).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert model.config.max_position_embeddings == 512
    return tokenizer, model


def encode_batch(tokenizer, model, questions, messages, device):
    tokens = tokenizer(questions, messages, padding=True, truncation="longest_first", max_length=512,
                       return_special_tokens_mask=True, return_tensors="pt")
    mask = ((tokens["token_type_ids"] == 1)&tokens["attention_mask"].bool()&~tokens["special_tokens_mask"].bool())
    assert mask.any(-1).all()
    inputs = {key: value.to(device) for key, value in tokens.items() if key != "special_tokens_mask"}
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
        weight = mask.to(device=device, dtype=hidden.dtype)[..., None]
        pooled = (hidden*weight).sum(1)/weight.sum(1)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return pooled.cpu().numpy().astype(np.float32), tokens["input_ids"].shape[1]


def prepare():
    cfg = config("configs/argument_mechanisms.yaml")
    a, index, vectors, metadata = load_cache(cfg, verify=True)
    audit = json.loads((AUDIT/"integrity.json").read_text())
    assert digest(AUDIT/"raw_texts.jsonl") == audit["artifacts"]["raw_texts.jsonl"]
    texts = [json.loads(line)["text"] for line in (AUDIT/"raw_texts.jsonl").open()]
    assert len(texts) == len(vectors)
    qids = sorted({r["qid"] for r in index})
    questions = []
    ids = np.zeros_like(a["text_ids"])
    source_ids, question_indices = [0], [-1]
    for qi, qid in enumerate(qids):
        es = [e for e, row in enumerate(index) if row["qid"] == qid]
        row = index[es[0]]
        questions.append(f"Problem: {row['statement']}\nA) {row['choices']['A']}\nB) {row['choices']['B']}")
        unique = np.unique(a["text_ids"][es])
        unique = unique[unique != 0]
        mapping = np.zeros(len(texts), dtype=np.int32)
        mapping[unique] = np.arange(len(source_ids), len(source_ids)+len(unique), dtype=np.int32)
        ids[es] = mapping[a["text_ids"][es]]
        source_ids.extend(unique.tolist())
        question_indices.extend([qi]*len(unique))
    source_ids = np.array(source_ids, dtype=np.int32)
    question_indices = np.array(question_indices, dtype=np.int16)
    np.testing.assert_array_equal(source_ids[ids], a["text_ids"])
    old_tokens = np.load(Path(cfg["data"]["cache"])/"text_tokens.npy", allow_pickle=False)
    ROOT.mkdir(parents=True, exist_ok=True)
    identity = {"code_sha256": digest(__file__), "source_metadata_sha256": digest(Path(cfg["data"]["cache"])/"metadata.json"),
        "input_audit_integrity_sha256": digest(AUDIT/"integrity.json"),
        "model_sha256": digest(SNAPSHOT/"model.safetensors"), "config_sha256": digest(SNAPSHOT/"config.json"),
        "modes": list(MODES), "rows": len(source_ids), "dimensions": 384, "max_tokens": 512,
        "pooling": "mean over non-special message tokens (segment 1), then unit normalize"}
    if (ROOT/"protocol.json").exists():
        protocol = json.loads((ROOT/"protocol.json").read_text())
        assert all(protocol[k] == v for k, v in identity.items())
        assert digest(ROOT/"ids.npz") == protocol["ids_sha256"]
        assert digest(ROOT/"questions.json") == protocol["questions_sha256"]
    else:
        np.savez_compressed(ROOT/"ids.npz", text_ids=ids, source_text_ids=source_ids, question_indices=question_indices)
        json_write(ROOT/"questions.json", dict(zip(qids, questions)))
        json_write(ROOT/"protocol.json", {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "ids_sha256": digest(ROOT/"ids.npz"), "questions_sha256": digest(ROOT/"questions.json"),
            "scope": "Frozen local encoder; development questions only. No response target, answer key, sender/receiver outcome, or test question enters encoding. Question and message are observed before response.",
            "control": "Identical frozen model, segment IDs, message-token pooling and 512-token cap. message_only supplies an empty first sequence; question_conditioned supplies the full problem and A/B choices. This differs from old standalone sentence pooling, so both controls are encoded.",
            "limitations": "A contextual frozen representation, not trained mathematical reasoning. Truncation, if any, is counted. Source LLM responses are not regenerated."})
    return cfg, a, index, texts, qids, questions, source_ids, question_indices, old_tokens


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("benchmark", "encode"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    cfg, a, index, texts, qids, questions, source_ids, qis, tokens = prepare()
    tokenizer, model = load_encoder(args.device)
    qlengths = np.array([len(tokenizer(q, add_special_tokens=False)["input_ids"]) for q in questions])
    sizes = {"message_only": tokens[source_ids]+1,
             "question_conditioned": tokens[source_ids]+1+qlengths[np.maximum(qis, 0)]}
    if args.stage == "benchmark":
        sample = np.random.default_rng(72019).choice(np.arange(1, len(source_ids)), size=1024, replace=False)
        rows = []
        for mode in MODES:
            start = time.monotonic()
            for offset in range(0, len(sample), args.batch_size):
                batch = sample[offset:offset+args.batch_size]
                msgs = [texts[source_ids[i]] for i in batch]
                qs = [questions[qis[i]] if mode == "question_conditioned" else "" for i in batch]
                result, length = encode_batch(tokenizer, model, qs, msgs, args.device)
                assert result.shape == (len(batch), 384) and np.isfinite(result).all()
            seconds = time.monotonic()-start
            rows.append({"mode": mode, "sample": len(sample), "seconds": seconds,
                "rough_full_minutes": seconds/len(sample)*(len(source_ids)-1)/60,
                "unique_truncated": int((sizes[mode][1:] > 512).sum()), "max_raw_tokens": int(sizes[mode][1:].max())})
        json_write(ROOT/"benchmark.json", rows)
        print(json.dumps(rows), flush=True)
        return
    protocol_sha = digest(ROOT/"protocol.json")
    if (ROOT/"complete.json").exists():
        done = json.loads((ROOT/"complete.json").read_text())
        assert done["protocol_sha256"] == protocol_sha
        for file, sha in done["artifacts"].items():
            assert digest(ROOT/file) == sha
        print(json.dumps({"complete": True, "reused": True}))
        return
    progress_path = ROOT/"progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "protocol_sha256": protocol_sha, "completed_rows": {mode: 0 for mode in MODES}, "elapsed_seconds": 0.}
    assert progress["protocol_sha256"] == protocol_sha
    started = time.monotonic()
    try:
        for mode in MODES:
            order = np.argsort(sizes[mode][1:], kind="stable")+1
            path = ROOT/f"{mode}.npy"
            cursor = progress["completed_rows"][mode]
            if path.exists():
                out = np.load(path, mmap_mode="r+", allow_pickle=False)
                assert out.shape == (len(source_ids), 384)
            else:
                assert cursor == 0
                out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(source_ids), 384))
                out[0] = 0
            for offset in range(cursor, len(order), args.batch_size):
                if time.time() >= deadline():
                    print(json.dumps({"bounded": True, "mode": mode, "rows": offset}), flush=True)
                    return
                batch = order[offset:offset+args.batch_size]
                msgs = [texts[source_ids[i]] for i in batch]
                qs = [questions[qis[i]] if mode == "question_conditioned" else "" for i in batch]
                result, length = encode_batch(tokenizer, model, qs, msgs, args.device)
                out[batch] = result
                progress["completed_rows"][mode] = offset+len(batch)
                if offset % 4096 == 0 or offset+len(batch) == len(order):
                    out.flush()
                    json_write(progress_path, progress)
                    print(json.dumps({"mode": mode, "encoded": offset+len(batch), "total": len(order),
                                      "padded_tokens": length, "seconds": time.monotonic()-started}), flush=True)
            out.flush()
            del out
        audits = {}
        for mode in MODES:
            vec = np.load(ROOT/f"{mode}.npy", mmap_mode="r", allow_pickle=False)
            sample = np.random.default_rng(19).choice(np.arange(1, len(vec)), size=10000, replace=False)
            assert np.isfinite(vec).all() and not vec[0].any()
            np.testing.assert_allclose(np.linalg.norm(vec[sample], axis=-1), 1., rtol=0, atol=3e-6)
            real = a["text_ids"] != 0
            with np.load(ROOT/"ids.npz") as stored:
                used = stored["text_ids"][real]
            audits[mode] = {"unique_truncated": int((sizes[mode][1:] > 512).sum()),
                "occurrences_truncated": int((sizes[mode][used] > 512).sum()),
                "max_raw_tokens": int(sizes[mode][1:].max()), "finite_unit_vectors": True}
        json_write(ROOT/"complete.json", {"protocol_sha256": protocol_sha,
            "completed_utc": datetime.now(timezone.utc).isoformat(), "encoder_parameters": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": progress["elapsed_seconds"]+time.monotonic()-started, "audits": audits,
            "artifacts": {f"{mode}.npy": digest(ROOT/f"{mode}.npy") for mode in MODES}})
        print(json.dumps({"complete": True, "audits": audits}), flush=True)
    finally:
        progress["elapsed_seconds"] += time.monotonic()-started
        json_write(progress_path, progress)


if __name__ == "__main__":
    main()
