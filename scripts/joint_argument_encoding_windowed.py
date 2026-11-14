"""No-truncation version of frozen question-conditioned message encoding.

Long questions are partitioned into nonoverlapping token windows, each paired
with the entire message. Message hidden-state means are averaged with question
token-count weights, then normalized. Every input token is retained.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import joint_argument_encoding as source

ROOT = source.SESSION/"joint_encoding_windowed"
MODES = source.MODES


def windows(question, message, maximum=512):
    budget = maximum-len(message)-3
    if budget < 1:
        raise ValueError("Message alone exceeds model context; do not silently truncate")
    if not question:
        return [([], message, 1.)]
    return [(question[i:i+budget], message, len(question[i:i+budget])/len(question))
            for i in range(0, len(question), budget)]


def encode_records(tokenizer, model, question_tokens, messages, device, microbatch=64):
    tokenized = tokenizer(messages, add_special_tokens=False, padding=False, truncation=False,
                          return_attention_mask=False, return_token_type_ids=False, verbose=False)["input_ids"]
    parts = []
    for row, (q, m) in enumerate(zip(question_tokens, tokenized)):
        for chunk, msg, weight in windows(q, m):
            parts.append((row, chunk, msg, weight))
    parts.sort(key=lambda p: len(p[1])+len(p[2]))
    accum = np.zeros((len(messages), model.config.hidden_size), dtype=np.float64)
    weights = np.zeros(len(messages), dtype=np.float64)
    for start in range(0, len(parts), microbatch):
        batch = parts[start:start+microbatch]
        length = max(len(p[1])+len(p[2])+3 for p in batch)
        assert length <= model.config.max_position_embeddings
        ids = torch.full((len(batch), length), tokenizer.pad_token_id, dtype=torch.long)
        types, attention, mask = torch.zeros_like(ids), torch.zeros_like(ids), torch.zeros_like(ids)
        for j, (row, q, m, weight) in enumerate(batch):
            seq = [tokenizer.cls_token_id]+q+[tokenizer.sep_token_id]+m+[tokenizer.sep_token_id]
            ids[j, :len(seq)] = torch.tensor(seq)
            types[j, len(q)+2:len(seq)] = 1
            attention[j, :len(seq)] = 1
            mask[j, len(q)+2:len(q)+2+len(m)] = 1
        assert mask.any(-1).all()
        with torch.inference_mode():
            hidden = model(input_ids=ids.to(device), token_type_ids=types.to(device),
                           attention_mask=attention.to(device)).last_hidden_state
            m = mask.to(device=device, dtype=hidden.dtype)[..., None]
            pooled = ((hidden*m).sum(1)/m.sum(1)).cpu().numpy()
        for j, (row, _, _, weight) in enumerate(batch):
            accum[row] += weight*pooled[j]
            weights[row] += weight
    np.testing.assert_allclose(weights, 1., rtol=0, atol=1e-12)
    norms = np.linalg.norm(accum, axis=-1, keepdims=True)
    assert np.isfinite(accum).all() and (norms > 0).all()
    return (accum/norms).astype(np.float32), len(parts)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("benchmark", "encode"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    cfg, arrays, index, texts, qids, questions, source_ids, qis, raw_tokens = source.prepare()
    tokenizer, model = source.load_encoder(args.device)
    qtokens = [tokenizer(q, add_special_tokens=False, truncation=False, verbose=False)["input_ids"] for q in questions]
    msg_lengths = raw_tokens[source_ids]-2
    assert msg_lengths[1:].max()+3 <= 512
    qlengths = np.array([len(q) for q in qtokens])
    sizes = {"message_only": msg_lengths+3,
             "question_conditioned": np.minimum(512, msg_lengths+3+qlengths[np.maximum(qis, 0)])}
    n_windows = np.maximum(1, np.ceil(qlengths[np.maximum(qis, 0)]/(509-msg_lengths)).astype(np.int32))
    ROOT.mkdir(parents=True, exist_ok=True)
    identity = {"code_sha256": digest(__file__), "base_encoder_code_sha256": digest(source.__file__),
        "mapping_protocol_sha256": digest(source.ROOT/"protocol.json"), "ids_sha256": digest(source.ROOT/"ids.npz"),
        "modes": list(MODES), "rows": len(source_ids), "dimensions": 384,
        "pooling": "message-token hidden means; question-token-count-weighted window mean; L2 normalize once",
        "question_windows": int(n_windows[1:].sum()), "multiwindow_records": int((n_windows[1:] > 1).sum()),
        "maximum_question_tokens": int(qlengths.max()), "maximum_message_tokens": int(msg_lengths[1:].max()),
        "tokens_truncated": 0}
    if (ROOT/"protocol.json").exists():
        old = json.loads((ROOT/"protocol.json").read_text())
        assert all(old[k] == v for k, v in identity.items())
    else:
        json_write(ROOT/"protocol.json", {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "reason_for_windowing": "A benchmark of naive 512-token concatenation would truncate 43584 unique pairs. This variant retains every question and message token, without changing cohort membership.",
            "scope": "Same cached frozen MiniLM weights; no supervised encoder training, targets, answer keys, test questions, or new LLM calls. Question strings contain only the observed statement and A/B choices.",
            "limitation": "Each long-question window is contextualized separately; averaging does not preserve all interactions between distant question fragments. This is a frozen contextual encoder, not a learned arithmetic reasoner."})
    if args.stage == "benchmark":
        sample = np.random.default_rng(1281).choice(np.arange(1, len(source_ids)), size=1024, replace=False)
        rows = []
        for mode in MODES:
            sample_order = sample[np.argsort(sizes[mode][sample], kind="stable")]
            start, count = time.monotonic(), 0
            for offset in range(0, len(sample_order), args.batch_size):
                batch = sample_order[offset:offset+args.batch_size]
                messages = [texts[source_ids[i]] for i in batch]
                q = [qtokens[qis[i]] if mode == "question_conditioned" else [] for i in batch]
                _, n = encode_records(tokenizer, model, q, messages, args.device, args.batch_size)
                count += n
            seconds = time.monotonic()-start
            rows.append({"mode": mode, "records": len(sample), "windows": count, "seconds": seconds,
                "rough_full_minutes": seconds/len(sample)*(len(source_ids)-1)/60})
        # Controlled sanity check: exact same message, changed question.
        example = texts[source_ids[sample[0]]]
        check, _ = encode_records(tokenizer, model, [qtokens[0], qtokens[-1]], [example, example], args.device)
        assert np.linalg.norm(check[0]-check[1]) > .001
        json_write(ROOT/"benchmark.json", {"rows": rows, "same_message_different_question_distance": float(np.linalg.norm(check[0]-check[1]))})
        print(json.dumps(rows), flush=True)
        return
    protocol_sha = digest(ROOT/"protocol.json")
    if (ROOT/"complete.json").exists():
        saved = json.loads((ROOT/"complete.json").read_text())
        assert saved["protocol_sha256"] == protocol_sha
        for file, sha in saved["artifacts"].items():
            assert digest(ROOT/file) == sha
        print(json.dumps({"complete": True, "reused": True}))
        return
    path = ROOT/"progress.json"
    progress = json.loads(path.read_text()) if path.exists() else {"protocol_sha256": protocol_sha,
        "completed_rows": {m: 0 for m in MODES}, "elapsed_seconds": 0.}
    assert progress["protocol_sha256"] == protocol_sha
    started = time.monotonic()
    try:
        for mode in MODES:
            order = np.argsort(sizes[mode][1:], kind="stable")+1
            output = ROOT/f"{mode}.npy"
            cursor = progress["completed_rows"][mode]
            if output.exists():
                vec = np.load(output, mmap_mode="r+", allow_pickle=False)
                assert vec.shape == (len(source_ids), 384)
            else:
                assert cursor == 0
                vec = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(len(source_ids), 384))
                vec[0] = 0
            for offset in range(cursor, len(order), args.batch_size):
                if time.time() >= source.deadline(reserve=900):
                    print(json.dumps({"bounded": True, "mode": mode, "rows": offset}), flush=True)
                    return
                batch = order[offset:offset+args.batch_size]
                messages = [texts[source_ids[i]] for i in batch]
                q = [qtokens[qis[i]] if mode == "question_conditioned" else [] for i in batch]
                encoded, n = encode_records(tokenizer, model, q, messages, args.device, args.batch_size)
                vec[batch] = encoded
                progress["completed_rows"][mode] = offset+len(batch)
                if offset % 8192 == 0 or offset+len(batch) == len(order):
                    vec.flush()
                    json_write(path, progress)
                    print(json.dumps({"mode": mode, "encoded": offset+len(batch), "total": len(order),
                        "windows_in_batch": n, "seconds": time.monotonic()-started}), flush=True)
            vec.flush()
            del vec
        for mode in MODES:
            vec = np.load(ROOT/f"{mode}.npy", mmap_mode="r", allow_pickle=False)
            assert np.isfinite(vec).all() and not vec[0].any()
            sample = np.random.default_rng(21).choice(np.arange(1, len(vec)), size=10000, replace=False)
            np.testing.assert_allclose(np.linalg.norm(vec[sample], axis=-1), 1., rtol=0, atol=3e-6)
        json_write(ROOT/"complete.json", {"protocol_sha256": protocol_sha,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": progress["elapsed_seconds"]+time.monotonic()-started,
            "tokens_truncated": 0, "all_question_and_message_tokens_covered": True,
            "artifacts": {f"{mode}.npy": digest(ROOT/f"{mode}.npy") for mode in MODES}})
        print(json.dumps({"complete": True, "tokens_truncated": 0}), flush=True)
    finally:
        progress["elapsed_seconds"] += time.monotonic()-started
        json_write(path, progress)


if __name__ == "__main__":
    main()
