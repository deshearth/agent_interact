"""Validate source records, freeze question splits, and build compact arrays."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from ..common import digest, json_write


def graph_hash(graph):
    return hashlib.sha256(np.asarray(graph, dtype=np.int8).tobytes()).hexdigest()


def question_splits(bank, seed=1729, validation_count=4):
    training = sorted(qid for qid, q in bank.items() if q["official_split"] == "train")
    testing = sorted(qid for qid, q in bank.items() if q["official_split"] == "test")
    validation = sorted(np.random.default_rng(seed).permutation(training)[:validation_count].tolist())
    fitting = sorted(set(training) - set(validation))
    return {"train": fitting, "validation": validation, "test": testing}


def pca_fit_transform(values, fit_indices, width=3):
    values = np.asarray(values, dtype=np.float64)
    mean = values[fit_indices].mean(axis=0)
    _, _, vectors = np.linalg.svd(values[fit_indices] - mean, full_matrices=False)
    components = vectors[:width]
    return (values - mean) @ components.T, mean, components


def message_round_valid(messages, source_graph):
    expected = {f"{i},{j}" for i, j in zip(*np.nonzero(source_graph))}
    return set(messages) == expected and all(
        isinstance(text, str) and bool(text.strip()) and not text.startswith("[error:")
        for text in messages.values())


def prepare(cfg):
    root = Path(cfg["data"]["root"])
    if (cfg["data"]["source_model"], cfg["data"]["mode"]) != ("gpt-4o-mini", "objective"):
        raise ValueError("This initial protocol is implemented for gpt-4o-mini/objective only")
    upstream = root / "raw" / "upstream"
    source_manifest = json.loads((root / "data_manifest.json").read_text())
    if source_manifest["revision"] != cfg["data"]["revision"]:
        raise ValueError("Config and acquired source revisions differ")
    for name, record in source_manifest["files"].items():
        if digest(upstream / name) != record["sha256"]:
            raise ValueError(f"Source checksum changed: {name}")
    processed = root / "processed" / cfg["data"].get("context", "author_pca")
    processed.mkdir(parents=True, exist_ok=True)
    bank, by_text = {}, {}
    normalized = {}
    for split in ("train", "test"):
        for line in (upstream / f"data/obj/{split}.jsonl").read_text().splitlines():
            q = json.loads(line)
            q["official_split"] = split
            norm = re.sub(r"\s+", " ", q["question"]).strip().casefold()
            if norm in normalized or q["qid"] in bank:
                raise ValueError("Duplicate question/template requires an explicit grouped split")
            normalized[norm] = q["qid"]
            bank[q["qid"]] = q
            by_text[q["question"]] = q["qid"]
    splits = question_splits(bank, cfg["data"]["split_seed"], cfg["data"]["validation_questions"])
    if [len(splits[k]) for k in ("train", "validation", "test")] != [16, 4, 20]:
        raise ValueError("Question counts differ from the declared 16/4/20 protocol")
    assigned = {qid: split for split, qids in splits.items() for qid in qids}
    personas = json.loads((upstream / "data/obj/personas.json").read_text())
    persona_texts = [p["persona"] if isinstance(p, dict) else p for p in personas]
    items = sorted(json.loads((upstream / "data/obj/persona_embeddings.json").read_text())["items"],
                   key=lambda x: x["idx"])
    if [p["idx"] for p in items] != list(range(32)) or [p["text"] for p in items] != persona_texts:
        raise ValueError("Persona text, feature order, or indices do not align")
    persona_features, pmean, pcomponents = pca_fit_transform(
        [x["embedding"] for x in items], np.arange(32))
    context_mode = cfg["data"].get("context", "author_pca")
    if context_mode == "author_pca":
        qfeatures = {qid: np.array(q["embedding_pca10"][:3]) for qid, q in bank.items()}
        provenance = "Author question PCA artifact; fitting provenance not independently verified. Not claimed inductive."
    elif context_mode == "train_only_pca":
        qi = {x["qid"]: x["embedding"] for x in json.loads(
            (upstream / "data/obj/question_embeddings_choices.json").read_text())["items"]}
        qids = sorted(bank)
        transformed, mean, components = pca_fit_transform(
            [qi[q] for q in qids], [qids.index(q) for q in splits["train"]])
        qfeatures = dict(zip(qids, transformed))
        np.savez(processed / "question_pca.npz", mean=mean, components=components)
        provenance = "Raw frozen question-plus-options embeddings; PCA fitted on the 16 training questions only."
    else:
        raise ValueError(f"Unknown context mode: {context_mode}")
    np.savez(processed / "persona_pca.npz", mean=pmean, components=pcomponents)
    graphs = json.loads((upstream / "data/J_matrices.json").read_text())["graphs"]
    catalog = {graph_hash(g["J"]): g for g in graphs}
    raw_dir = upstream / "data/models/gpt-4o-mini/objective_energy"
    rows = [r for r in json.loads((upstream / "data/clean_runs.json").read_text())
            if r["model"] == cfg["data"]["source_model"] and r["mode"] == cfg["data"]["mode"]]
    if len(rows) != 1600:
        raise ValueError(f"Expected 1600 main episodes; found {len(rows)}")
    candidates = []
    for r in rows:
        qid = by_text.get(r["statement"])
        if qid is None:
            raise ValueError("Unmatched question text")
        g = catalog.get(graph_hash(r["J"]))
        if g is None:
            raise ValueError("Unknown graph in consolidated source")
        name = f"{qid}__{g['id']}__rep{int(r['replica']):02d}"
        candidates.append((name, qid, g["id"], r, raw_dir / f"{name}.json", False))
    lowrank = upstream / "data/lowrank/gpt-4o-mini/ood_lowrank/objective_rank_frustration"
    for path in sorted(lowrank.glob("*.json")):
        if path.name == "manifest.json":
            continue
        r = json.loads(path.read_text())
        qid = by_text[r["statement"]]
        _, gid, rep = path.stem.rsplit("__", 2)
        r["replica"] = int(rep.removeprefix("rep"))
        candidates.append((path.stem, qid, gid, r, path, True))
    seen = set()
    index, states, votes, adjacency, context, message_ok = [], [], [], [], [], []
    missing_raw, invalid_messages, raw_failures = [], 0, 0
    for name, qid, gid, row, raw_path, supplemental in sorted(candidates):
        source_graph = np.asarray(row["J"], dtype=np.int8)
        spins = np.asarray(row["spins_history"], dtype=np.int8)
        samples = np.asarray(row["spins_raw_history"], dtype=np.int8)
        if source_graph.shape != (32, 32) or spins.shape != (9, 32) or samples.shape != (9, 32, 5):
            raise ValueError(f"Shape mismatch: {name}")
        if not np.isin(source_graph, [-1, 0, 1]).all() or not np.isin(spins, [-1, 0, 1]).all() or not np.isin(samples, [-1, 0, 1]).all():
            raise ValueError(f"Unexpected labels: {name}")
        if np.diag(source_graph).any() or not np.array_equal(source_graph, source_graph.T):
            raise ValueError(f"Unexpected directed/self-edge graph; audit before proceeding: {name}")
        gh = graph_hash(source_graph)
        eid = hashlib.sha256(f"{cfg['data']['source_model']}|objective|{qid}|{gh}|{row['replica']}".encode()).hexdigest()
        if eid in seen:
            raise ValueError(f"Duplicate episode: {name}")
        seen.add(eid)
        available = [True] + [False] * 7
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
            if raw["personas"] != persona_texts or raw["statement"] != row["statement"]:
                raise ValueError(f"Raw context mismatch: {name}")
            for field in ("J", "spins_history", "spins_raw_history"):
                if not np.array_equal(raw[field], row[field]):
                    raise ValueError(f"Raw/consolidated mismatch: {name}/{field}")
            if len(raw["messages_history"]) != 8:
                raise ValueError(f"Message round count mismatch: {name}")
            checks = [message_round_valid(m, source_graph) for m in raw["messages_history"]]
            invalid_messages += checks.count(False)
            available = [True] + checks[:7]
        else:
            missing_raw.append(name)
        family = "lowrank" if supplemental else (gid if gid in ("square", "triangular") else "random")
        role = "fit_graph" if gid in {f"J{i}" for i in range(8)} else (
            "heldout_random" if gid.startswith("Jf") else "heldout_family")
        q = qfeatures[qid]
        phi = np.concatenate([np.ones((32, 1)), persona_features, np.tile(q, (32, 1)),
                              (persona_features[:, :, None] * q[None, None, :]).reshape(32, 9)], axis=-1)
        index.append({"episode_id": eid, "name": name, "qid": qid,
                      "question_split": assigned[qid], "official_split": bank[qid]["official_split"],
                      "graph_id": gid, "graph_hash": gh, "graph_family": family, "graph_role": role,
                      "replica": int(row["replica"]), "supplemental": supplemental,
                      "raw_path": str(raw_path), "statement": row["statement"], "choices": bank[qid]["choices"]})
        states.append(spins)
        votes.append(samples)
        adjacency.append(source_graph.T.astype(np.float32))  # source sender-first -> receiver-first
        context.append(phi.astype(np.float32))
        message_ok.append(available)
        raw_failures += int((samples == 0).sum())
    arrays = {"spins": np.stack(states), "votes": np.stack(votes), "J": np.stack(adjacency),
              "context": np.stack(context), "message_ok": np.array(message_ok, dtype=bool)}
    window = cfg["state"]["history_window"]
    valid = np.ones((len(index), 8), dtype=bool)
    for t in range(8):
        valid[:, t] = (arrays["spins"][:, max(0, t-window+1):t+2] != 0).all(axis=(1, 2))
    arrays["valid"] = valid
    np.savez_compressed(processed / "episodes.npz", **arrays)
    json_write(processed / "index.json", index)
    split_manifest = {"questions": splits, "seed": cfg["data"]["split_seed"],
                      "source_revision": cfg["data"]["revision"], "context": context_mode,
                      "context_provenance": provenance, "history_window_for_common_mask": window,
                      "normalization": "Exact whitespace/case normalized duplicates rejected; semantic paraphrase grouping not independently established.",
                      "graph_roles": {x["graph_hash"]: x["graph_role"] for x in index}}
    json_write(processed / "splits.json", split_manifest)
    counts = Counter((x["question_split"], x["graph_role"], x["graph_family"]) for x in index)
    role_counts = []
    for (qsplit, role, family), count in sorted(counts.items()):
        ids = [i for i, x in enumerate(index) if (x["question_split"], x["graph_role"], x["graph_family"]) == (qsplit, role, family)]
        role_counts.append({"question_split": qsplit, "graph_role": role, "family": family,
                            "episodes": count, "valid_snapshots": int(valid[ids].sum()),
                            "valid_agent_transitions": int(valid[ids].sum() * 32),
                            "message_eligible_snapshots": int((valid[ids] & arrays["message_ok"][ids]).sum())})
    audit = {"episodes": len(index), "main_episodes": len(rows), "supplemental_episodes": len(index)-len(rows),
             "questions": {k: len(v) for k, v in splits.items()}, "graphs": len(set(x['graph_hash'] for x in index)),
             "agents": 32, "updates": 8, "raw_vote_parse_failures": raw_failures,
             "invalid_majority_states": int((arrays["spins"] == 0).sum()), "duplicates": 0,
             "metadata_join_coverage": 1.0, "missing_raw_episodes": missing_raw,
             "invalid_message_rounds": invalid_messages, "roles": role_counts,
             "orientation": "Source J[sender,receiver] transposed to J[receiver,sender]; all observed source matrices symmetric.",
             "invalid_policy": f"All models exclude a snapshot if any agent has an invalid opinion in the current/target or available {window}-snapshot history. Invalid votes excluded from counts; valid count retained.",
             "message_timing": "At cutoff t use messages_history[t-1], empty at t=0; messages_history[t] excluded.",
             "historical_protocol": "GPT-4o-mini has recipient-specific sampled messages; edge records retained.",
             "context_provenance": provenance, "source_revision": cfg["data"]["revision"],
             "split_manifest_sha256": digest(processed / "splits.json"),
             "episodes_sha256": digest(processed / "episodes.npz"), "index_sha256": digest(processed / "index.json")}
    json_write(processed / "audit.json", audit)
    json_write(f"reports/data_audit_{context_mode}.json", audit)
    if context_mode == "author_pca":
        json_write("reports/data_audit.json", audit)
    text = ["# Released-data audit", "", f"Source revision: `{cfg['data']['revision']}`.", "",
            f"{len(index)} episodes: {len(rows)} main and {len(index)-len(rows)} supplemental low-rank episodes. 32 agents, 8 updates.", "",
            f"Question split: 16 train / 4 validation / 20 locked test; seed {cfg['data']['split_seed']}.", "",
            "| Question split | Graph role | Family | Episodes | Valid agent transitions |",
            "| --- | --- | --- | ---: | ---: |"]
    text.extend(f"| {r['question_split']} | {r['graph_role']} | {r['family']} | {r['episodes']} | {r['valid_agent_transitions']} |" for r in role_counts)
    text += ["", f"Raw vote parse failures: {raw_failures}; invalid majority states: {audit['invalid_majority_states']}.",
             f"Missing raw episodes: {len(missing_raw)}; invalid message rounds: {invalid_messages}.", "",
             audit["orientation"], "", audit["invalid_policy"], "", audit["message_timing"], "",
             audit["historical_protocol"], "", provenance, "",
             "Exact normalized duplicate questions: none. Semantic paraphrase groups are not established by this audit.", "",
             "The low-rank subset covers ten official test questions, not all twenty. Test outcomes remain reserved for frozen model selection."]
    Path(f"reports/data_audit_{context_mode}.md").write_text("\n".join(text) + "\n")
    if context_mode == "author_pca":
        Path("reports/data_audit.md").write_text("\n".join(text) + "\n")
    print(json.dumps({k: audit[k] for k in ("episodes", "questions", "graphs", "raw_vote_parse_failures", "invalid_majority_states", "missing_raw_episodes")}, indent=2))
    return audit


def load_prepared(cfg):
    directory = Path(cfg["data"]["root"]) / "processed" / cfg["data"].get("context", "author_pca")
    splits = json.loads((directory / "splits.json").read_text())
    if not splits.get("synthetic") and splits["history_window_for_common_mask"] != cfg["state"]["history_window"]:
        raise ValueError("History window differs from the common audit mask; prepare data again in a separate experiment")
    with np.load(directory / "episodes.npz", allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files}
    index = json.loads((directory / "index.json").read_text())
    return arrays, index, directory
