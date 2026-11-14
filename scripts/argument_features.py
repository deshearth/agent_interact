"""Development-only, individual-edge message features for response prediction.

No core files or prior feature caches are mutated. The frozen encoder runs only
from its local snapshot. Features include messages[t], which precede s[t+1].
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from mac_agents.common import digest, json_write
from mac_agents.data.features import history_features
from mac_agents.data.prepare import message_round_valid


STANCE = re.compile(r"\b(?:favor|favour|support|choose|prefer|select|endorse)\s+"
                    r"(?:option\s+|answer\s+)?[\s*\"'`]*([AB])\b", re.I)


def parse_stance(text):
    match = STANCE.search(text[:160])
    if match is None:
        raise ValueError("Message has no explicit stance; revise protocol before proceeding")
    return 1 if match[1].upper() == "A" else -1


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def neighbor_slots(graphs):
    """graphs[e, receiver, sender]; padded sender slots do not form edges."""
    width = max(1, int((graphs != 0).sum(-1).max()))
    neighbors = np.zeros((*graphs.shape[:2], width), dtype=np.int64)
    signs = np.zeros_like(neighbors, dtype=np.float32)
    for e, graph in enumerate(graphs):
        for receiver in range(len(graph)):
            senders = np.flatnonzero(graph[receiver])
            neighbors[e, receiver, :len(senders)] = senders
            signs[e, receiver, :len(senders)] = graph[receiver, senders]
    return neighbors, signs


def response_ids(raw, neighbors, signs, lookup):
    """Read no target opinions; retain each recipient-specific message at t."""
    steps = len(raw["messages_history"])
    ids = np.zeros((steps, *neighbors.shape), dtype=np.int32)
    stance = np.zeros_like(ids, dtype=np.int8)
    for t, messages in enumerate(raw["messages_history"]):
        if not message_round_valid(messages, np.asarray(raw["J"])):
            raise ValueError("Incomplete response inbox")
        for receiver, slot in zip(*np.nonzero(signs)):
            sender = int(neighbors[receiver, slot])
            text = messages[f"{sender},{receiver}"]
            ids[t, receiver, slot] = lookup(text)
            stance[t, receiver, slot] = parse_stance(text)
    return ids, stance


def shuffle_groups(arrays, index, ids, stance, neighbors, signs):
    """Targets never enter the matching keys; partitions remain separate."""
    groups = defaultdict(list)
    shape = ids.shape
    for e, row in enumerate(index):
        for t in range(shape[1]):
            for receiver, slot in zip(*np.nonzero(signs[e])):
                sender = int(neighbors[e, receiver, slot])
                key = (row["question_split"], row["qid"], t, sender, int(receiver),
                       int(signs[e, receiver, slot]), int(stance[e, t, receiver, slot]),
                       int(arrays["spins"][e, t, sender]), int(arrays["spins"][e, t, receiver]))
                groups[key].append(int(np.ravel_multi_index((e, t, receiver, slot), shape)))
    return [np.array(values, dtype=np.int32) for _, values in sorted(groups.items())]


def matched_donors(groups, shape, seed):
    """Bijective substitutions within strata; singleton strata retain their text."""
    donor = np.arange(np.prod(shape), dtype=np.int32)
    rng = np.random.default_rng(seed + 92173)
    for group in groups:
        if len(group) > 1:
            order = rng.permutation(group)
            donor[order] = np.roll(order, int(rng.integers(1, len(order))))
    return donor.reshape(shape)


def prepare(cfg, device):
    started = time.monotonic()
    directory = Path(cfg["data"]["cache"])
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "metadata.json"
    if meta_path.exists():
        return load_cache(cfg, verify=True)[-1]
    source = Path(cfg["data"]["prepared"])
    all_index = json.loads((source / "index.json").read_text())
    selected = [e for e, row in enumerate(all_index)
                if row["question_split"] in ("train", "validation") and row["graph_role"] == "fit_graph"]
    index = [all_index[e] for e in selected]
    split = json.loads((source / "splits.json").read_text())
    if split["source_revision"] != cfg["data"]["source_revision"]:
        raise ValueError("Source revision mismatch")
    with np.load(source / "episodes.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key][selected] for key in ("spins", "J", "context", "valid")}
    if len(index) != 640 or not arrays["valid"].all():
        raise ValueError("Declared common 640-episode development cohort changed")
    neighbors, signs = neighbor_slots(arrays["J"])
    texts, lookup_ids = [""], {}

    def lookup(text):
        if text not in lookup_ids:
            lookup_ids[text] = len(texts)
            texts.append(text)
        return lookup_ids[text]

    ids, stance, raw_hashes = [], [], {}
    for e, row in enumerate(index):
        raw_path = Path(row["raw_path"])
        raw_hashes[row["episode_id"]] = digest(raw_path)
        raw = json.loads(raw_path.read_text())
        np.testing.assert_array_equal(raw["spins_history"], arrays["spins"][e])
        np.testing.assert_array_equal(np.asarray(raw["J"]).T, arrays["J"][e])
        if raw["statement"] != row["statement"] or raw["choices"] != row["choices"]:
            raise ValueError("Question or options mismatch")
        one_ids, one_stance = response_ids(raw, neighbors[e], signs[e], lookup)
        if one_ids.shape[0] != 8:
            raise ValueError("Expected eight response rounds")
        expected = arrays["spins"][e, :-1, neighbors[e]]
        # Advanced indexing above yields receiver, slot, round.
        expected = expected.transpose(2, 0, 1)
        if not np.array_equal(one_stance[:, signs[e] != 0], expected[:, signs[e] != 0]):
            raise ValueError("Communicated stance differs from sender opinion; audit before running")
        ids.append(one_ids)
        stance.append(one_stance)
    ids, stance = np.stack(ids), np.stack(stance)
    del lookup_ids
    arrays.update(neighbors=neighbors, signs=signs, text_ids=ids, stance=stance)
    np.savez_compressed(directory / "episodes.npz", **arrays)
    json_write(directory / "index.json", index)
    groups = shuffle_groups(arrays, index, ids, stance, neighbors, signs)
    shuffle_audit = {}
    for seed in cfg["training"]["seeds"]:
        donor = matched_donors(groups, ids.shape, seed)
        np.save(directory / f"donors_s{seed}.npy", donor)
        shuffled = ids.reshape(-1)[donor]
        support = (donor != np.arange(ids.size).reshape(ids.shape)) & (ids != 0)
        changed = (shuffled != ids) & (ids != 0)
        shuffle_audit[str(seed)] = {}
        for partition in ("train", "validation"):
            which = [e for e, row in enumerate(index) if row["question_split"] == partition]
            shuffle_audit[str(seed)][partition] = {
                "messages": int((ids[which] != 0).sum()),
                "matched_messages": int(support[which].sum()),
                "different_text_messages": int(changed[which].sum()),
                "receiver_updates_with_match": int(support[which].any(-1).sum()),
                "receiver_updates_with_different_text": int(changed[which].any(-1).sum())}
    del groups
    # Read the prior database without extending or altering the frozen old cache.
    prior = Path(cfg["data"]["prior_encoder_cache"])
    prior_meta = json.loads((prior / "metadata.json").read_text())
    if (prior_meta["encoder"], prior_meta["revision"], prior_meta["dim"],
        prior_meta["max_sequence_length"], prior_meta["normalize_embeddings"]) != (
        cfg["data"]["encoder"], cfg["data"]["encoder_revision"], cfg["data"]["encoder_dimension"],
        cfg["data"]["max_sequence_length"], True):
        raise ValueError("Frozen encoder cache contract mismatch")
    vectors = np.lib.format.open_memmap(directory / "vectors.npy", mode="w+", dtype=np.float32,
                                        shape=(len(texts), cfg["data"]["encoder_dimension"]))
    vectors[0] = 0
    tokens = np.zeros(len(texts), dtype=np.int32)
    hashes = np.array([text_hash(text) for text in texts], dtype="S64")
    missing = []
    with sqlite3.connect(f"file:{(prior / 'text_vectors.sqlite').resolve()}?mode=ro", uri=True) as db:
        for i in range(1, len(texts)):
            record = db.execute("SELECT vector,tokens FROM vectors WHERE key=?", (hashes[i].decode(),)).fetchone()
            if record is None:
                missing.append(i)
            else:
                vectors[i] = np.frombuffer(record[0], dtype=np.float32)
                tokens[i] = record[1]
    print(json.dumps({"stage": "encoding", "unique_texts": len(texts)-1,
                      "reused_vectors": len(texts)-1-len(missing), "new_vectors": len(missing)}), flush=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer
    snapshot = Path(".cache/huggingface") / ("models--" + cfg["data"]["encoder"].replace("/", "--")) / "snapshots" / cfg["data"]["encoder_revision"]
    encoder = SentenceTransformer(str(snapshot.resolve()), device=str(device), trust_remote_code=False,
                                  local_files_only=True)
    encoder.max_seq_length = cfg["data"]["max_sequence_length"]
    encoder.eval()
    encode_start = time.monotonic()
    for start in range(0, len(missing), 64):
        positions = missing[start:start+64]
        batch = [texts[i] for i in positions]
        lengths = encoder.tokenizer(batch, truncation=False, padding=False, return_length=True)["length"]
        with torch.inference_mode():
            embedded = encoder.encode(batch, batch_size=64, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        vectors[positions] = embedded.astype(np.float32)
        tokens[positions] = lengths
        if start % 4096 == 0:
            vectors.flush()
            print(f"Encoded new message vectors: {min(start+64,len(missing))}/{len(missing)}", flush=True)
    vectors.flush()
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite message embeddings")
    np.testing.assert_allclose(np.linalg.norm(vectors[1:], axis=1), 1, atol=2e-5, rtol=0)
    np.save(directory / "text_tokens.npy", tokens)
    np.save(directory / "text_hashes.npy", hashes)
    files = ["episodes.npz", "index.json", "vectors.npy", "text_tokens.npy", "text_hashes.npy"] + [
        f"donors_s{seed}.npy" for seed in cfg["training"]["seeds"]]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "schema": "argument_response_v1",
        "feature_code_sha256": digest(__file__), "data_config": cfg["data"],
        "source_files_sha256": {name: digest(source / name) for name in
                                ("episodes.npz", "index.json", "splits.json", "question_pca.npz", "persona_pca.npz")},
        "raw_files_sha256": raw_hashes, "files_sha256": {name: digest(directory / name) for name in files},
        "episodes": len(index), "snapshots": int(arrays["valid"].sum()),
        "agent_transitions": int(arrays["valid"].sum()*32),
        "questions": {key: sorted({row["qid"] for row in index if row["question_split"] == key})
                      for key in ("train", "validation")},
        "messages": int((ids != 0).sum()), "unique_texts": len(texts)-1,
        "sender_stance_matches_explicit_message_stance": True,
        "forecast_boundary": "after_messages_before_next_opinion", "message_offset": 0,
        "max_neighbors": signs.shape[-1], "shuffle": shuffle_audit,
        "shuffle_matching_fields": ["partition", "question", "round", "sender", "receiver", "relationship",
                                    "explicit_message_stance", "sender_current_opinion", "receiver_current_opinion"],
        "shuffle_note": "Different-episode bijection within matched strata; singleton messages unchanged. Graph identity and full history not matched. No target opinions used.",
        "encoder": {**prior_meta, "forecast_boundary": "after_messages_before_next_opinion", "message_offset": 0,
                    "pooling": "individual_edge_vectors_no_inbox_pooling", "frozen_parameters": sum(p.numel() for p in encoder.parameters()),
                    "reused_vectors": len(texts)-1-len(missing), "new_vectors": len(missing),
                    "truncated_unique_texts": int((tokens[1:] > encoder.max_seq_length).sum()),
                    "truncated_message_occurrences": int((tokens[ids] > encoder.max_seq_length).sum()),
                    "new_encoding_seconds": time.monotonic()-encode_start, "device": str(device)},
        "elapsed_seconds": time.monotonic()-started, "test_messages_read_or_encoded": False,
        "no_ground_truth_answer_keys_used": True}
    json_write(meta_path, metadata)
    json_write("reports/argument_feature_audit.json", metadata)
    print(json.dumps({"stage": "features_complete", "episodes": len(index),
                      "elapsed_seconds": metadata["elapsed_seconds"], "shuffle": shuffle_audit}), flush=True)
    return metadata


def load_cache(cfg, verify=True):
    directory = Path(cfg["data"]["cache"])
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["data_config"] != cfg["data"] or metadata["feature_code_sha256"] != digest(__file__):
        raise ValueError("Argument feature implementation or config changed; use a new namespace")
    if verify:
        for name, expected in metadata["files_sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"Argument feature checksum mismatch: {name}")
        source = Path(cfg["data"]["prepared"])
        for name, expected in metadata["source_files_sha256"].items():
            if digest(source / name) != expected:
                raise ValueError(f"Prepared input changed: {name}")
    with np.load(directory / "episodes.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    index = json.loads((directory / "index.json").read_text())
    if any(row["question_split"] not in ("train", "validation") or row["graph_role"] != "fit_graph" for row in index):
        raise ValueError("Argument diagnostic may not read a test partition")
    vectors = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
    return arrays, index, vectors, metadata


class ResponseSnapshots(Dataset):
    def __init__(self, arrays, index, vectors, partition, condition, donors):
        if partition not in ("train", "validation"):
            raise ValueError("Response study is development-only; no test or rollout endpoint")
        if condition not in ("stance_only", "full_arguments", "shuffled_arguments"):
            raise ValueError(condition)
        self.arrays, self.index, self.vectors = arrays, index, vectors
        self.condition = condition
        self.pairs = [(e, t) for e, row in enumerate(index) if row["question_split"] == partition
                      for t in range(arrays["text_ids"].shape[1]) if arrays["valid"][e, t]]
        self.ids = arrays["text_ids"]
        self.shuffled_ids = self.ids.reshape(-1)[donors]
        self.changed = (self.shuffled_ids != self.ids).any(-1)
        self.support = (donors != np.arange(donors.size).reshape(donors.shape)).any(-1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        e, t = self.pairs[item]
        a = self.arrays
        ids = self.shuffled_ids[e, t] if self.condition == "shuffled_arguments" else self.ids[e, t]
        text = (np.zeros((*ids.shape, self.vectors.shape[1]), dtype=np.float32) if self.condition == "stance_only"
                else np.array(self.vectors[ids], dtype=np.float32))
        return {"s": torch.from_numpy(a["spins"][e, t].astype(np.float32)),
                "context": torch.from_numpy(a["context"][e]),
                "history": torch.from_numpy(history_features(a["spins"][e], t, 2)),
                "neighbors": torch.from_numpy(a["neighbors"][e]),
                "signs": torch.from_numpy(a["signs"][e]),
                "stance": torch.from_numpy(a["stance"][e, t].astype(np.float32)),
                "text": torch.from_numpy(text),
                "target": torch.from_numpy((a["spins"][e, t+1] == 1).astype(np.int64)),
                "episode": e, "round": t}
