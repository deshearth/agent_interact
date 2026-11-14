"""Input ambiguity and released-prompt audit; no fits or test-set scoring."""
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from scripts.argument_features import load_cache, text_hash
from scripts.argument_mechanisms import split_index

ROOT = Path("runs/argument_investigation_20261108/input_audit")
CONFIG = "configs/argument_mechanisms.yaml"


def node_keys(a, row, e, t, receiver):
    slots = np.flatnonzero(a["signs"][e, receiver])
    senders = a["neighbors"][e, receiver, slots]
    signs = a["signs"][e, receiver, slots].astype(np.int8)
    degree = np.stack([(a["J"][e] > 0).sum(-1), (a["J"][e] < 0).sum(-1)], -1)
    # Sender IDs stand for the fixed, distinct persona context vectors. Include
    # exactly the one-hop state/history and sender/receiver degree information.
    topology = np.column_stack([senders, signs, degree[senders]]).astype(np.int16).tobytes()
    header = row["qid"].encode()+b"|"+int(receiver).to_bytes(2, "little")+degree[receiver].astype(np.int16).tobytes()+topology
    nodes = np.r_[receiver, senders]
    current = a["spins"][e, t, nodes].astype(np.int8).tobytes()
    past = a["spins"][e, t-1, nodes].astype(np.int8).tobytes() if t else bytes(len(nodes))
    opinion = header+current
    history = opinion+bytes([int(t > 0)])+past
    ids = a["text_ids"][e, t, receiver, slots]
    full = history+ids.astype(np.int32).tobytes()
    prefix = row["qid"].encode()+b"|"+int(receiver).to_bytes(2, "little")
    pos, neg = sorted(ids[signs > 0].tolist()), sorted(ids[signs < 0].tolist())
    bag = prefix+json.dumps([pos, neg], separators=(",", ":")).encode()
    order = list(range(len(slots)))
    # This reproduces the released sampler, not a verified historical prompt.
    random.Random(f"{row['name']}|{t+1}|{receiver}").shuffle(order)
    ordered = [[int(ids[i]) for i in order if signs[i] == sign] for sign in (1, -1)]
    released_order = prefix+json.dumps(ordered, separators=(",", ":")).encode()
    return {name: hashlib.sha256(value).digest() for name, value in (
        ("opinion", opinion), ("history", history), ("history_text", full),
        ("prompt_message_bag", bag), ("released_prompt_order", released_order))}


def collision_summary(groups, role, kind):
    total = sum(len(v) for v in groups.values())
    repeated = [v for v in groups.values() if len(v) > 1]
    conflicts = [v for v in groups.values() if len({r[3] for r in v}) > 1]
    errors = sum(min(sum(r[3] == -1 for r in v), sum(r[3] == 1 for r in v)) for v in groups.values())
    return {"role": role, "input": kind, "targets": total, "unique_inputs": len(groups),
            "repeated_input_targets": sum(map(len, repeated)), "conflicting_input_groups": len(conflicts),
            "conflicting_input_targets": sum(map(len, conflicts)), "minimum_empirical_errors": errors,
            "minimum_empirical_error": errors/total,
            "repeat_coverage": sum(map(len, repeated))/total}


def main():
    cfg = config(CONFIG)
    a, original_index, vectors, metadata = load_cache(cfg, verify=True)
    index = split_index(original_index, cfg)
    ROOT.mkdir(parents=True, exist_ok=True)
    identity = {"code_sha256": digest(__file__),
                "source_feature_metadata_sha256": digest(Path(cfg["data"]["cache"])/"metadata.json"),
                "split_protocol_sha256": digest("runs/argument_mechanisms_v1/protocol.json")}
    if (ROOT/"integrity.json").exists():
        done = json.loads((ROOT/"integrity.json").read_text())
        assert all(done[k] == v for k, v in identity.items())
        for file, sha in done["artifacts"].items():
            assert digest(ROOT/file) == sha
        print(json.dumps({"complete": True, "reused": True}))
        return
    with (ROOT/"protocol.json").open("w") as stream:
        json.dump({**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Measure identical-input outcome ambiguity, source question selection and prompt limitations; identify balanced training-only conflicting state pairs for a capacity check.",
            "scope": "Development only. Empirical identical-input error is an in-sample lower bound for a deterministic function of that input, not an out-of-sample Bayes ceiling. Text IDs give a conservative bound if different text has identical encoder vectors.",
            "prompt_provenance": "Pinned released sampler, which explicitly documents legacy message-generation differences. Raw GPT-4o-mini episodes do not save complete prompts or presentation order. Reconstructed order is therefore an assumption, not proof of historical ordering.",
            "ground_truth": "Question answer keys used only for retrospective source-behavior descriptions, never supplied to a predictive model."}, stream, indent=2)
    qbank = {r["qid"]: r for r in (json.loads(line) for line in
        Path("data/raw/upstream/data/obj/train.jsonl").read_text().splitlines())}
    texts = [None]*len(vectors)
    texts[0] = ""
    expected_hashes = np.load(Path(cfg["data"]["cache"])/"text_hashes.npy", allow_pickle=False)
    math_rows, question_rows = [], []
    checked = set()
    for e, row in enumerate(index):
        raw = json.loads(Path(row["raw_path"]).read_text())
        assert digest(row["raw_path"]) == metadata["raw_files_sha256"][row["episode_id"]]
        assert raw["mode"] == "objective"
        correct = 1 if qbank[row["qid"]]["answer"] == "A" else -1
        votes = np.asarray(raw["spins_raw_history"])
        assert np.array_equal(np.sign(votes.sum(-1)), a["spins"][e])
        for t in range(8):
            math_rows.append({"qid": row["qid"], "role": row["diagnostic_role"], "round": t,
                "correct_option": correct, "current_math_accuracy": float((a["spins"][e, t] == correct).mean()),
                "next_math_accuracy": float((a["spins"][e, t+1] == correct).mean()),
                "unanimous_vote_fraction": float((np.abs(votes[t+1].sum(-1)) == 5).mean())})
            for receiver, slot in zip(*np.nonzero(a["signs"][e])):
                sender = int(a["neighbors"][e, receiver, slot])
                text = raw["messages_history"][t][f"{sender},{receiver}"]
                tid = int(a["text_ids"][e, t, receiver, slot])
                if texts[tid] is None:
                    assert text_hash(text) == expected_hashes[tid].decode()
                    texts[tid] = text
                else:
                    assert texts[tid] == text
        if row["qid"] not in checked:
            q = qbank[row["qid"]]
            question_rows.append({"qid": row["qid"], "partition": row["question_split"], "question": row["statement"],
                "choices": row["choices"], "correct_option": q["answer"], "type": q.get("type"), "level": q.get("level"),
                "source_probe_pA": q.get("probe", {}).get("pA"), "initial_correctness_in_this_episode": float((a["spins"][e, 0] == correct).mean())})
            checked.add(row["qid"])
    assert all(t is not None for t in texts)
    with (ROOT/"raw_texts.jsonl").open("w") as f:
        for i, text in enumerate(texts):
            f.write(json.dumps({"id": i, "text": text}, ensure_ascii=False)+"\n")
    json_write(ROOT/"questions.json", question_rows)
    pd.DataFrame(math_rows).to_csv(ROOT/"source_math_behavior.csv", index=False)
    # Hash keys once, then summarize the original and current episode splits.
    key_arrays = {name: np.empty((len(index), 8, 32), dtype="S32") for name in
                  ("opinion", "history", "history_text", "prompt_message_bag", "released_prompt_order")}
    for e, row in enumerate(index):
        for t in range(8):
            for r in range(32):
                for name, key in node_keys(a, row, e, t, r).items():
                    key_arrays[name][e, t, r] = key
    np.savez_compressed(ROOT/"input_keys.npz", **key_arrays)
    summaries, chosen = [], []
    roles = {role: [e for e, row in enumerate(index) if row["diagnostic_role"] == role]
             for role in ("train", "validation", "familiar", "unseen")}
    roles["original_full_train"] = [e for e, row in enumerate(index) if row["question_split"] == "train"]
    rng = np.random.default_rng(72191)
    for role, episodes in roles.items():
        for kind, keys in key_arrays.items():
            groups = defaultdict(list)
            for e in episodes:
                for t in range(8):
                    for r in range(32):
                        groups[bytes(keys[e, t, r])].append((e, t, r, int(a["spins"][e, t+1, r])))
            summaries.append(collision_summary(groups, role, kind))
            if role == "train" and kind == "history":
                candidates = defaultdict(list)
                for key, members in groups.items():
                    pos = [x for x in members if x[3] == 1]
                    neg = [x for x in members if x[3] == -1]
                    if not pos or not neg:
                        continue
                    p, n = pos[0], neg[0]
                    if key_arrays["history_text"][p[:3]] == key_arrays["history_text"][n[:3]]:
                        continue
                    candidates[index[p[0]]["qid"]].append([p, n])
                # Up to 16 pairs per fitting question; outcome-selected ONLY
                # for a memorization diagnostic, not a generalization score.
                for qid in sorted(candidates):
                    pairs = candidates[qid]
                    take = rng.permutation(len(pairs))[:16]
                    chosen.extend(pairs[i] for i in take)
    np.save(ROOT/"ambiguous_training_pairs.npy", np.array(chosen, dtype=np.int32), allow_pickle=False)
    pd.DataFrame(summaries).to_csv(ROOT/"collisions.csv", index=False)
    statements = {"released_objective_choice_instruction": "Give a direct snap judgement; do not work through, compute, show steps or explain.",
        "released_message_instruction": "Defend the sender's current A/B answer with its strongest reason in two sentences.",
        "observed_stance_alignment": "The frozen feature audit verifies that every one of 573440 delivered development messages explicitly states the sender's current answer.",
        "released_inbox_order": "Messages are shuffled with Random(replica_name|step|receiver), then split into agree/disagree groups. Existing predictors are order-invariant.",
        "historical_caveat": "Raw episodes store messages but not full prompts or actual inbox order; the release documents legacy generation differences. Prompt/order explanations require cautious historical attribution.",
        "selected_memorization_pairs": len(chosen), "memorization_target_floor_without_text": .5,
        "interpretation": "Chosen opposite-label pairs have identical observable history-model inputs but different delivered text. Perfect deterministic fitting without text is impossible on these selected pairs; fitting with text is a capacity test, not evidence of causal influence or transferable reasoning."}
    json_write(ROOT/"findings.json", statements)
    files = ["protocol.json", "raw_texts.jsonl", "questions.json", "source_math_behavior.csv",
             "input_keys.npz", "ambiguous_training_pairs.npy", "collisions.csv", "findings.json"]
    json_write(ROOT/"integrity.json", {**identity, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {file: digest(ROOT/file) for file in files}})
    print(pd.DataFrame(summaries).to_string(index=False))
    print(json.dumps(statements))


if __name__ == "__main__":
    main()
