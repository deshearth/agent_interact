from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


STATE_CONDITIONS = ("opinion", "vote_counts", "history", "existing_messages")


def history_features(spins, t, window):
    """Local history only, with explicit missing-history indicators."""
    n = spins.shape[-1]
    out = np.zeros((n, window, 3), dtype=np.float32)
    for slot, step in enumerate(range(t-window+1, t+1)):
        if step >= 0:
            s = spins[step]
            out[:, slot, 0] = s == -1
            out[:, slot, 1] = s == 1
            out[:, slot, 2] = np.abs(s) == 1
    return out.reshape(n, -1)


def extra_dim(cfg, state):
    return {"opinion": 0, "vote_counts": 3,
            "history": 3 * cfg["state"]["history_window"],
            "existing_messages": 2 * cfg["state"]["message_dim"] + 2}[state]


def message_directory(cfg):
    return Path(cfg["data"]["root"]) / "processed" / "message_features"


def partition_indices(index, partition):
    if partition in ("train", "validation"):
        return [i for i, x in enumerate(index) if x["question_split"] == partition and x["graph_role"] == "fit_graph"]
    if partition == "test":
        return [i for i, x in enumerate(index) if x["question_split"] == "test" or
                (x["question_split"] == "train" and x["graph_role"] != "fit_graph")]
    raise ValueError(partition)


def evaluation_role(row):
    if row["question_split"] == "validation":
        return "validation"
    return ("new_question" if row["question_split"] == "test" else "known_question") + "/" + row["graph_role"] + "/" + row["graph_family"]


class Snapshots(Dataset):
    def __init__(self, arrays, index, cfg, partition, state, limit_episodes=None, match_state=None):
        self.arrays, self.index, self.cfg, self.state = arrays, index, cfg, state
        if state not in STATE_CONDITIONS:
            raise ValueError(state)
        episode_ids = partition_indices(index, partition)
        self.expected_episodes = len(episode_ids)
        needs_messages = state == "existing_messages" or match_state == "existing_messages"
        if needs_messages:
            directory = message_directory(cfg)
            metadata = json.loads((directory / "metadata.json").read_text())
            if metadata["encoder"] != cfg["state"]["encoder"] or metadata["dim"] != cfg["state"]["message_dim"]:
                raise ValueError("Message cache/config mismatch")
            available = {p.stem for p in directory.glob("*.npz")}
            episode_ids = [e for e in episode_ids if index[e]["episode_id"] in available]
        if limit_episodes is not None:
            episode_ids = episode_ids[:limit_episodes]
        self.episode_ids = episode_ids
        self.incomplete_messages = needs_messages and len(episode_ids) < self.expected_episodes
        self.pairs = [(e, t) for e in episode_ids for t in range(arrays["spins"].shape[1]-1)
                      if arrays["valid"][e, t] and (not needs_messages or arrays["message_ok"][e, t])]
        if not self.pairs:
            raise ValueError(f"No eligible {partition}/{state} snapshots; prepare data/message features first")

    @lru_cache(maxsize=32)
    def messages(self, e):
        path = message_directory(self.cfg) / f"{self.index[e]['episode_id']}.npz"
        with np.load(path, allow_pickle=False) as source:
            if str(source["episode_id"]) != self.index[e]["episode_id"]:
                raise ValueError("Message feature alignment mismatch")
            return source["features"].astype(np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        e, t = self.pairs[item]
        a = self.arrays
        n = a["spins"].shape[-1]
        if self.state == "vote_counts":
            v = a["votes"][e, t]
            extra = np.stack([(v == 1).sum(-1), (v == -1).sum(-1), (v != 0).sum(-1)], -1).astype(np.float32) / v.shape[-1]
        elif self.state == "history":
            extra = history_features(a["spins"][e], t, self.cfg["state"]["history_window"])
        elif self.state == "existing_messages":
            extra = self.messages(e)[t]
        else:
            extra = np.zeros((n, 0), dtype=np.float32)
        return {"s": torch.from_numpy(a["spins"][e, t].copy()).float(),
                "context": torch.from_numpy(a["context"][e]),
                "J": torch.from_numpy(a["J"][e]), "extra": torch.from_numpy(extra),
                "target": torch.from_numpy((a["spins"][e, t+1] == 1).astype(np.int64)),
                "episode": e, "round": t}


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
