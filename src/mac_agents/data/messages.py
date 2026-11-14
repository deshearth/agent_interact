"""Frozen message embeddings, retained by edge and pooled into each prior inbox.

Cache one complete episode at a time. Only rounds before the forecast boundary
are encoded; t=0 has an explicitly empty inbox.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch

from ..common import digest, json_write
from .features import message_directory, partition_indices
from .prepare import load_prepared


def pool_existing_messages(raw, vectors, dimension):
    n, steps = len(raw["personas"]), len(raw["spins_history"])-1
    out = np.zeros((steps, n, 2*dimension+2), dtype=np.float32)
    graph = np.asarray(raw["J"])
    for t in range(1, steps):
        counts = np.zeros((n, 2), dtype=np.float32)
        for key, text in raw["messages_history"][t-1].items():
            sender, receiver = map(int, key.split(","))
            if graph[sender, receiver] == 0:
                raise ValueError("Message on nonexistent edge")
            channel = int(graph[sender, receiver] < 0)
            out[t, receiver, channel*dimension:(channel+1)*dimension] += vectors[text]
            counts[receiver, channel] += 1
        for channel in range(2):
            out[t, :, channel*dimension:(channel+1)*dimension] /= np.maximum(counts[:, channel:channel+1], 1)
        out[t, :, -2:] = np.log1p(counts)
    return out


def encode_messages(cfg, partition="train", limit_episodes=None, device="cpu"):
    cache = Path(".cache/huggingface").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer
    arrays, index, _ = load_prepared(cfg)
    directory = message_directory(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["encoder"] != cfg["state"]["encoder"]:
            raise ValueError("Existing message cache uses a different encoder")
    else:
        info = model_info(cfg["state"]["encoder"])
        revision = info.sha
        metadata = {"encoder": cfg["state"]["encoder"], "revision": revision,
                    "dim": cfg["state"]["message_dim"], "max_sequence_length": 256,
                    "normalize_embeddings": True, "pooling": "mean_by_signed_inbox_channel_v1",
                    "forecast_boundary": "after_opinion_before_messages", "message_offset": -1,
                    "source_revision": cfg["data"]["revision"],
                    "license": getattr(info.card_data, "license", None)}
        json_write(metadata_path, metadata)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    encoder = SentenceTransformer(metadata["encoder"], revision=metadata["revision"],
                                  cache_folder=str(cache), device=device, trust_remote_code=False)
    encoder.max_seq_length = metadata["max_sequence_length"]
    encoder.eval()
    dimension_method = getattr(encoder, "get_embedding_dimension", encoder.get_sentence_embedding_dimension)
    if dimension_method() != metadata["dim"]:
        raise ValueError("Encoder dimension differs from config")
    episodes = (list(range(len(index))) if partition == "all" else
                partition_indices(index, "train") + partition_indices(index, "validation") if partition == "development" else
                partition_indices(index, partition))
    if limit_episodes is not None:
        episodes = episodes[:limit_episodes]
    db = sqlite3.connect(directory / "text_vectors.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB, tokens INTEGER)")
    total, truncated, finished = 0, 0, 0
    started = time.monotonic()
    for e in episodes:
        row = index[e]
        if not arrays["message_ok"][e].all():
            continue
        destination = directory / f"{row['episode_id']}.npz"
        raw_path = Path(row["raw_path"])
        raw_sha = digest(raw_path)
        if destination.exists():
            with np.load(destination, allow_pickle=False) as stored:
                if str(stored["raw_sha256"]) != raw_sha or str(stored["encoder_revision"]) != metadata["revision"]:
                    raise ValueError("Existing feature cache source mismatch; use a new cache directory")
            finished += 1
            continue
        raw = json.loads(raw_path.read_text())
        texts = sorted({text for messages in raw["messages_history"][:7] for text in messages.values()})
        vectors, missing = {}, []
        for text in texts:
            key = hashlib.sha256(text.encode()).hexdigest()
            stored = db.execute("SELECT vector, tokens FROM vectors WHERE key=?", (key,)).fetchone()
            if stored:
                vectors[text] = np.frombuffer(stored[0], dtype=np.float32).copy()
                total += 1
                truncated += int(stored[1] > encoder.max_seq_length)
            else:
                missing.append(text)
        for start in range(0, len(missing), 32):
            batch = missing[start:start+32]
            lengths = encoder.tokenizer(batch, truncation=False, padding=False, return_length=True)["length"]
            with torch.inference_mode():
                embedded = encoder.encode(batch, batch_size=32, normalize_embeddings=True,
                                          convert_to_numpy=True, show_progress_bar=False)
            for text, vector, tokens in zip(batch, embedded, lengths):
                vector = vector.astype(np.float32)
                vectors[text] = vector
                db.execute("INSERT INTO vectors VALUES (?, ?, ?)",
                           (hashlib.sha256(text.encode()).hexdigest(), vector.tobytes(), int(tokens)))
                total += 1
                truncated += int(tokens > encoder.max_seq_length)
            db.commit()
        features = pool_existing_messages(raw, vectors, metadata["dim"])
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, features=features, episode_id=row["episode_id"], raw_sha256=raw_sha,
                                encoder_revision=metadata["revision"])
        temporary.replace(destination)
        finished += 1
        print(f"Message features: {finished}/{len(episodes)} episodes", flush=True)
    db.close()
    result = {"partition": partition, "requested_episodes": len(episodes), "complete_episodes": finished,
              "message_texts_checked": total, "truncated_texts": truncated,
              "elapsed_seconds": time.monotonic()-started, "device": device,
              "encoder_revision": metadata["revision"], "frozen_parameters": sum(p.numel() for p in encoder.parameters())}
    with (directory / "encoding_log.jsonl").open("a") as handle:
        handle.write(json.dumps(result) + "\n")
    json_write("reports/message_encoding.json", result)
    print(json.dumps(result, indent=2))
    return result
