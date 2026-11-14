"""Descriptive errors of frozen breadth predictors; never read assessment episodes."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from mac_agents.models.baselines import baseline_logp
from scripts import breadth_train as study
from scripts import transformer_generalization as base
from scripts.local_conversation_pilot import deliver, formatter, opinion_prompt

OUTPUT = study.ROOT / "error_diagnosis_v1"
STANCE = re.compile(r"\bI\s+(?:strongly\s+)?(?:favor|favour|prefer|support|choose|select|recommend)\s+option\s+([AB])\b", re.I)


def stated_option(text):
    letters = {match.upper() for match in STANCE.findall(text)}
    return (1 if letters == {"A"} else -1 if letters == {"B"} else 0)


def statistics(frame):
    result = {"updates": len(frame), "questions": int(frame.qid.nunique())}
    for name in ["ising_error", "nano_error", "ising_nll", "nano_nll", "fixed", "introduced", "shared", "both_correct"]:
        result[name] = float(frame[name].mean())
    result["net_error_gain"] = result["ising_error"] - result["nano_error"]
    result["ising_errors"] = float(frame.ising_error.sum())
    result["nano_errors_seed_average"] = float(frame.nano_error.sum())
    return result


def summarize_groups(frame, column):
    rows = []
    for (split, group), part in frame.groupby(["split", column], observed=True, sort=True):
        total = frame[frame.split == split]
        result = {"split": split, "group": str(group), **statistics(part)}
        result["update_share"] = len(part) / len(total)
        result["ising_error_share"] = float(part.ising_error.sum() / total.ising_error.sum())
        result["nano_error_share"] = float(part.nano_error.sum() / total.nano_error.sum())
        rows.append(result)
    return rows


def repeated_group_summary(groups, split, isolated_only=False):
    selected = [rows for rows in groups.values() if rows[0]["split"] == split
                and (not isolated_only or rows[0]["isolated"]) and len(rows) > 1]
    pairs = sum(len(rows) * (len(rows) - 1) // 2 for rows in selected)
    different = sum(sum(row["target"] for row in rows) * (len(rows) - sum(row["target"] for row in rows)) for rows in selected)
    conflicted = [rows for rows in selected if 0 < sum(row["target"] for row in rows) < len(rows)]
    return {"repeated_groups": len(selected), "observations": sum(map(len, selected)),
            "conflicted_groups": len(conflicted), "different_target_pairs": int(different),
            "pairs": pairs, "pair_disagreement_rate": different / pairs if pairs else None,
            "empirical_minimum_mistakes": sum(min(sum(r["target"] for r in rows), len(rows) - sum(r["target"] for r in rows)) for rows in selected)}


def main():
    protocol = study.validate_prepared()
    study.authorize_split(study.ROOT, "assessment")  # Hash verification only.
    frozen_path = study.ROOT / "models_frozen.json"
    freeze_sha = digest(frozen_path)
    frozen = json.loads(frozen_path.read_text())
    cfg = study.predictor_config()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    root = Path(cfg["output"])
    common = dict(np.load(root / "development.npz", allow_pickle=False))
    features = dict(np.load(root / "folds/main/features.npz", allow_pickle=False))
    data = base.FoldData(common, features, "cpu")
    ids = np.arange(len(common["pairs"]))
    e, t = common["pairs"].T
    qlookup = {q["qid"]: q for q in json.loads((study.ROOT / "selection.json").read_text())["questions"]
               if q["split"] in ["train", "validation"]}
    assert set(common["episode_qids"]) == set(qlookup)
    assert not set(common["episode_qids"]) & set(protocol["fold"]["assessment_questions"])
    baseline = json.loads((root / "folds/main/ising.json").read_text())
    ilp = baseline_logp(baseline, common["spins"][e, t], features["context"][e], common["J"][e])
    neural = []
    for run in frozen["refits"]:
        model = base.make_model(cfg, "nano")
        model.load_state_dict(torch.load(Path(run["directory"]) / "best.pt", map_location="cpu", weights_only=False)["model"])
        neural.append(base.predictions(model, data, ids, False, 256))
    nlp = np.stack(neural)
    target = data.y
    iw = ilp.argmax(-1) != target
    nw = nlp.argmax(-1) != target[None]
    ip = np.exp(ilp[..., 1])
    np_ = np.exp(nlp[..., 1])
    current = common["spins"][e, t]
    next_spin = common["spins"][e, t + 1]
    degree = np.abs(common["J"][e]).sum(-1)
    current_votes = (common["raw_votes"][e, t] == 1).sum(-1)
    next_votes = (common["raw_votes"][e, t + 1] == 1).sum(-1)
    qids = np.repeat(common["episode_qids"][e], 32)
    flat = lambda x: np.asarray(x).reshape(-1)
    frame = pd.DataFrame({
        "qid": qids, "split": [qlookup[q]["split"] for q in qids],
        "topic": [qlookup[q]["type"] for q in qids], "level": [qlookup[q]["level"] for q in qids],
        "graph": [qlookup[q]["graph_id"] for q in qids], "agent": np.tile(np.arange(32), len(e)),
        "round": np.repeat(t + 1, 32), "current_A": flat(current == 1), "target_A": flat(target),
        "target_correct": flat(next_spin) == np.array([1 if qlookup[q]["answer"] == "A" else -1 for q in qids]),
        "degree": flat(degree), "positive_degree": flat((common["J"][e] > 0).sum(-1)),
        "negative_degree": flat((common["J"][e] < 0).sum(-1)),
        "switch": flat(current != next_spin),
        "current_agreement": flat(np.maximum(current_votes, 5 - current_votes)),
        "target_agreement": flat(np.maximum(next_votes, 5 - next_votes)),
        "ising_probability_A": flat(ip), "nano_probability_A_mean": flat(np_.mean(0)),
        "ising_error": flat(iw.astype(float)), "nano_error": flat(nw.mean(0)),
        "ising_nll": flat(-np.take_along_axis(ilp, target[..., None], -1)[..., 0]),
        "nano_nll": flat(-np.take_along_axis(nlp, np.broadcast_to(target[None, ..., None], (*nlp.shape[:-1], 1)), -1)[..., 0].mean(0)),
        "fixed": flat((iw[None] & ~nw).mean(0)), "introduced": flat((~iw[None] & nw).mean(0)),
        "shared": flat((iw[None] & nw).mean(0)), "both_correct": flat((~iw[None] & ~nw).mean(0)),
    })
    frame["ising_confidence"] = np.maximum(frame.ising_probability_A, 1 - frame.ising_probability_A)
    frame["confidence_bin"] = pd.cut(frame.ising_confidence, [0, .8, .95, 1.00000001], right=False,
                                      labels=["<0.80", "0.80-0.95", ">=0.95"])
    frame["degree_bin"] = pd.cut(frame.degree, [-1, 0, 4, 8, 32], labels=["isolated", "1-4", "5-8", "9+"])
    np.testing.assert_allclose(frame.fixed - frame.introduced, frame.ising_error - frame.nano_error)
    np.testing.assert_allclose(frame.shared + frame.introduced, frame.nano_error)
    np.testing.assert_allclose(frame[["fixed", "introduced", "shared", "both_correct"]].sum(axis=1), 1)
    frame["contradictory_incoming_messages"] = 0
    frame["parsed_incoming_messages"] = 0
    prompt_groups, state_groups = defaultdict(list), defaultdict(list)
    message_rows, state_examples = [], []
    fmt = formatter()
    for qi, qid in enumerate(common["episode_qids"]):
        q = qlookup[qid]
        ep_path = study.ROOT / "episodes" / q["split"] / f"{qid}__{q['graph_id']}__r00.json"
        episode = json.loads(ep_path.read_text())
        graph = episode["J"]
        agree, disagree = [[] for _ in range(32)], [[] for _ in range(32)]
        for step in range(9):
            parsed, mismatch = {}, {}
            if step:
                edges = episode["messages_history"][step - 1]
                senders = {}
                for key, text in edges.items():
                    sender, receiver = map(int, key.split(","))
                    if sender in senders: assert senders[sender] == text
                    senders[sender] = text
                agree, disagree, reproduced = deliver(graph, senders, episode["meta"]["name"], step)
                assert reproduced == edges
                for sender, text in senders.items():
                    assigned = int(common["spins"][qi, step - 1, sender])
                    assert episode["communicated_stances_history"][step - 1][str(sender)] == assigned
                    parsed[sender] = stated_option(text)
                    mismatch[sender] = bool(parsed[sender] and parsed[sender] != assigned)
                    message_rows.append({"qid": qid, "split": q["split"], "round": step,
                                         "sender": sender, "assigned": assigned, "parsed": parsed[sender],
                                         "mismatch": mismatch[sender], "text": text})
            for agent in range(32):
                isolated = not any(graph[agent])
                prompt = opinion_prompt(fmt, episode["personas"][agent], q, agree[agent], disagree[agent])
                key = hashlib.sha256(prompt.encode()).hexdigest()
                observation = {"qid": qid, "split": q["split"], "agent": agent, "round": step,
                               "target": int(common["spins"][qi, step, agent] == 1), "isolated": isolated}
                prompt_groups[key].append(observation)
                if step:
                    mask = (frame.qid == qid) & (frame["round"] == step) & (frame.agent == agent)
                    incoming = [i for i in range(32) if graph[i][agent]]
                    frame.loc[mask, "parsed_incoming_messages"] = sum(bool(parsed[i]) for i in incoming)
                    frame.loc[mask, "contradictory_incoming_messages"] = sum(mismatch[i] for i in incoming)
                    # This one-layer model depends on the receiver and its neighbors;
                    # context and graph-derived degree features are fixed within qid.
                    neighbors = np.flatnonzero(common["J"][qi, agent])
                    relevant = np.unique(np.r_[agent, neighbors])
                    local = tuple(int(s) for s in common["spins"][qi, step - 1, relevant])
                    state_key = (qid, agent, local)
                    state_groups[state_key].append({**observation, "frame_index": int(frame.index[mask][0])})
    frame["has_contradictory_incoming"] = frame.contradictory_incoming_messages > 0
    # Equal supplied local states must produce equal frozen-model probabilities.
    repeated_max = 0.
    for (qid, agent, local), observations in state_groups.items():
        if len(observations) < 2: continue
        indices = [r["frame_index"] for r in observations]
        spread = float(np.ptp(frame.loc[indices, "nano_probability_A_mean"]))
        repeated_max = max(repeated_max, spread)
        assert spread < 2e-6
        if 0 < sum(r["target"] for r in observations) < len(observations):
            state_examples.append({"qid": qid, "agent": agent, "local_state": local, "observations": observations})
    summary = {split: statistics(frame[frame.split == split]) for split in ["train", "validation"]}
    groups = {name: summarize_groups(frame, name) for name in ["target_agreement", "current_agreement", "switch",
              "confidence_bin", "degree_bin", "round", "graph", "topic", "level", "agent", "target_correct", "has_contradictory_incoming"]}
    questions = []
    for (split, qid), part in frame.groupby(["split", "qid"]):
        q = qlookup[qid]
        questions.append({"qid": qid, "split": split, "topic": q["type"], "level": q["level"],
            "question": q["question"], "choices": q["choices"], "answer": q["answer"],
            **statistics(part), "target_A_fraction": float(part.target_A.mean()),
            "ising_probability_A_mean": float(part.ising_probability_A.mean()),
            "nano_probability_A_mean": float(part.nano_probability_A_mean.mean()),
            "unanimous_target_fraction": float((part.target_agreement == 5).mean()),
            "switch_fraction": float(part["switch"].mean()), "screen": q["screen"]})
    messages = pd.DataFrame(message_rows)
    message_summary = {}
    for split, part in messages.groupby("split"):
        parsed = part[part.parsed != 0]
        message_summary[split] = {"messages": len(part), "explicit_stance_parsed": len(parsed),
            "coverage": len(parsed) / len(part), "mismatches": int(parsed.mismatch.sum()),
            "mismatch_fraction_of_parsed": float(parsed.mismatch.mean())}
    repeated = {split: {"source_prompts": repeated_group_summary(prompt_groups, split),
                       "isolated_source_prompts": repeated_group_summary(prompt_groups, split, True),
                       "supplied_local_states": repeated_group_summary(state_groups, split)}
                for split in ["train", "validation"]}
    screen = {split: {k: float(np.mean([q["screen"][k] for q in qlookup.values() if q["split"] == split]))
                     for k in ["pooled_entropy_bits", "within_agent_entropy_bits", "between_agent_entropy_bits"]}
              for split in ["train", "validation"]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "transitions.csv", index=False)
    np.savez_compressed(OUTPUT / "predictions.npz", ising_logp=ilp, nano_logp=nlp, target=target, pairs=common["pairs"], qids=common["episode_qids"])
    json_write(OUTPUT / "by_question.json", questions)
    json_write(OUTPUT / "groups.json", groups)
    json_write(OUTPUT / "message_mismatch_examples.json", [row for row in message_rows if row["mismatch"]])
    json_write(OUTPUT / "state_conflict_examples.json", state_examples)
    json_write(OUTPUT / "repeated_prompt_groups.json", [rows for rows in prompt_groups.values() if len(rows) > 1])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Post-hoc descriptive development diagnosis; no source calls, refitting, or assessment outcomes.",
              "group_weighting": "Pooled agent updates within subgroup; full cohorts have 256 updates per question and equal question weight. Neural scores average three seeds, not ensemble predictions.",
              "models_frozen_sha256": freeze_sha, "script_sha256": digest(__file__), "summary": summary,
              "messages": message_summary, "repeated_inputs": repeated, "screening_entropy": screen,
              "repeated_state_probability_spread_maximum": repeated_max,
              "limitations": ["Target vote agreement and newly emitted messages are retrospective diagnostics, not original predictor inputs.",
                              "Repeated-label conflicts give finite-sample constraints, not a population Bayes error estimate.",
                              "Question/graph effects are confounded because there is one graph/conversation per question.",
                              "Subgroups are exploratory and validation has already selected the model settings."],
              "artifacts": {str(p): digest(p) for p in OUTPUT.iterdir() if p.is_file() and p.name != "summary.json"}}
    json_write(OUTPUT / "summary.json", report)
    study.validate_prepared()
    study.authorize_split(study.ROOT, "assessment")
    assert digest(frozen_path) == freeze_sha
    print(json.dumps({"summary": summary, "target_agreement": groups["target_agreement"], "switch": groups["switch"],
                      "messages": message_summary, "repeated_inputs": repeated, "screening_entropy": screen}, indent=2))


if __name__ == "__main__":
    main()
