"""A separate, frozen 3,200/800 MATH expansion with availability-based sampling.

The old collection, source functions and all predictors are immutable. Selection
uses only pre-interaction screening. Reserved assessment episodes are excluded.
"""
from argparse import ArgumentParser
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hpc3_prepare import read, sha, write
from scripts import question_breadth as original
from scripts.breadth_expand import now
from scripts.hpc3_parallel_collect import publish_immutable

ROOT = Path("runs/question_breadth_3200_v1")
PARENT = Path("runs/question_breadth_320_v1")
FIRST = Path("runs/question_breadth_v1")
CONFIG = Path("configs/question_breadth_3200.yaml")
CODE = ["scripts/breadth_scale.py", "scripts/hpc3_scale_worker.py",
        "scripts/hpc3_scale_controller.py", "hpc3/scale_worker.sbatch",
        "hpc3/scale_controller.sbatch", "hpc3/scale_tick.sh"]
METRICS = ("pooled_entropy_bits", "within_agent_entropy_bits")


def config():
    return yaml.safe_load(CONFIG.read_text())


def apportion(total, weights, capacities=None):
    """Deterministic largest-remainder allocation, with explicit capacity checks."""
    keys = sorted(weights)
    capacities = capacities or {k: total for k in keys}
    if total < 0 or total > sum(capacities.values()) or sum(weights.values()) <= 0:
        raise ValueError("Insufficient question capacity")
    ideals = {k: total * weights[k] / sum(weights.values()) for k in keys}
    out = {k: min(int(ideals[k]), capacities[k]) for k in keys}
    while sum(out.values()) < total:
        k = min((k for k in keys if out[k] < capacities[k]), key=lambda k: (out[k] - ideals[k], k))
        out[k] += 1
    return out


def quotas(bank, settings):
    counts = Counter((q["type"], q["answer"]) for q in bank)
    topics = sorted({q["type"] for q in bank})
    total = sum(settings["new_questions"].values())
    nval = settings["new_questions"]["validation"]
    if total % 2 or nval % 2:
        raise ValueError("Exact A/B balance needs even split sizes")
    capacity = {t: min(counts[t, "A"], counts[t, "B"]) for t in topics}
    weights = {t: counts[t, "A"] + counts[t, "B"] for t in topics}
    selected = apportion(total // 2, weights, capacity)
    val = apportion(nval // 2, selected, selected)
    return {"new_selected": {t: {a: selected[t] for a in "AB"} for t in topics},
            "new_validation": {t: {a: val[t] for a in "AB"} for t in topics},
            "available": {t: {a: counts[t, a] for a in "AB"} for t in topics}}


def build_candidates():
    """Read only source problem text/solutions and earlier candidate metadata."""
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("The candidate bank is already frozen")
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    settings = config()
    source = FIRST / "source/math/competition_math.parquet"
    provenance = read(FIRST / "source/math/provenance.json")
    assert sha(source) == provenance["sha256"]
    prior = read(FIRST / "candidate_bank.json") + read(PARENT / "candidate_bank.json")
    old = original.read_jsonl(original.UPSTREAM / "obj/train.jsonl") + original.read_jsonl(original.UPSTREAM / "obj/test.jsonl")
    personas = read(original.UPSTREAM / "obj/personas.json")
    excluded = [q["question"] for q in prior + old] + [p["problem"] for p in personas]
    excluded_ids = {q["qid"] for q in prior}
    excluded_families = {original.normalize(t, True) for t in excluded}
    eligible, reasons = [], Counter()
    frame = pd.read_parquet(source)
    for idx, row in frame.iterrows():
        qid = f"breadth_math_{idx:05d}"
        if qid in excluded_ids:
            reasons["previous_candidate"] += 1
            continue
        question, solution = str(row["problem"]), str(row["solution"])
        boxed = original.last_boxed(solution)
        value = original.rational_answer(boxed)
        if value is None or abs(value) > 10**12:
            reasons["unsupported_rational_answer"] += 1
            continue
        if len(question) > 6000 or "[asy]" in question.lower():
            reasons["diagram_or_length"] += 1
            continue
        family = original.normalize(question, True)
        if family in excluded_families:
            reasons["previous_family"] += 1
            continue
        wrong = original.distractors(value, question)
        if not wrong:
            reasons["insufficient_distractors"] += 1
            continue
        eligible.append({"qid": qid, "source_row": int(idx), "question": question, "solution": solution,
                         "correct_answer": original.display_rational(value), "original_boxed_answer": boxed,
                         "distractor_candidates": wrong, "type": str(row["type"]), "level": str(row["level"]),
                         "family_key": hashlib.sha256(family.encode()).hexdigest()})
    corpus = [original.normalize(t, True) for t in excluded] + [original.normalize(q["question"], True) for q in eligible]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=60000).fit_transform(corpus)
    offset = len(excluded)
    conflicts = {}
    for start in range(0, len(eligible), 128):
        similarity = (matrix[offset + start:offset + start + 128] @ matrix.T).tocsr()
        similarity.data[similarity.data < settings["selection"]["near_duplicate_cosine"]] = 0
        similarity.eliminate_zeros()
        for j in range(similarity.shape[0]):
            conflicts[start + j] = set(similarity.indices[similarity.indptr[j]:similarity.indptr[j + 1]])
        print(json.dumps({"candidate_similarity_rows": min(start + 128, len(eligible)), "total": len(eligible)}), flush=True)
    order = list(range(len(eligible)))
    random.Random(settings["selection"]["bank_seed"]).shuffle(order)
    selected_indices, families, bank = set(range(offset)), set(), []
    for i in order:
        q = eligible[i]
        if q["family_key"] in families:
            reasons["duplicate_new_family"] += 1
        elif conflicts[i] & selected_indices:
            reasons["near_duplicate"] += 1
        else:
            selected_indices.add(offset + i)
            families.add(q["family_key"])
            bank.append(q)
    # Label placement is fixed before any source screening, balanced per topic.
    rng = random.Random(settings["selection"]["quota_seed"])
    extra_a = True
    for topic in sorted({q["type"] for q in bank}):
        rows = [q for q in bank if q["type"] == topic]
        rng.shuffle(rows)
        n_a = len(rows) // 2 + int(len(rows) % 2 and extra_a)
        if len(rows) % 2:
            extra_a = not extra_a
        for i, q in enumerate(rows):
            q["answer"] = "A" if i < n_a else "B"
    bank.sort(key=lambda q: q["qid"])
    quota = quotas(bank, settings)
    ROOT.mkdir(parents=True, exist_ok=True)
    write(ROOT / "candidate_bank.json", bank)
    write(ROOT / "quotas.json", quota)
    write(ROOT / "candidate_preparation.json", {"created_utc": now(), "source_sha256": sha(source),
        "source_revision": provenance["revision"], "source_rows": len(frame), "eligible_fresh_before_deduplication": len(eligible),
        "excluded_previous_candidates": len(prior), "candidate_count": len(bank), "filter_counts": dict(reasons),
        "topic_counts": dict(Counter(q["type"] for q in bank)), "level_counts": dict(Counter(q["level"] for q in bank)),
        "new_selected": quota["new_selected"], "maximum_allowed_character_ngram_cosine_exclusive": settings["selection"]["near_duplicate_cosine"],
        "config_sha256": sha(CONFIG), "candidate_bank_sha256": sha(ROOT / "candidate_bank.json")})
    print(json.dumps(read(ROOT / "candidate_preparation.json"), indent=2))


def prepare():
    if (ROOT / "protocol.json").exists():
        return verify()
    settings, prep = config(), read(ROOT / "candidate_preparation.json")
    assert prep["config_sha256"] == sha(CONFIG) and prep["candidate_bank_sha256"] == sha(ROOT / "candidate_bank.json")
    parents = read(PARENT / "selection.json")["questions"]
    previous = read(PARENT / "episode_index.json")
    assert Counter(q["split"] for q in parents) == {"train": 320, "validation": 80, "assessment": 40}
    parent_complete = read(PARENT / "collection_complete.json")
    assert parent_complete["index_sha256"] == sha(PARENT / "episode_index.json")
    retained = [{"qid": r["qid"], "split": r["split"], "graph_id": r["graph_id"], "source_path": r["path"],
                 "sha256": r["sha256"], "source_protocol_sha256": r["source_protocol_sha256"]} for r in previous]
    for r in retained:
        assert sha(r["source_path"]) == r["sha256"]
    write(ROOT / "retained_episodes.json", retained)
    write(ROOT / "parent_selection.json", read(PARENT / "selection.json"))
    bank = read(ROOT / "candidate_bank.json")
    assert len({q["qid"] for q in bank}) == len({q["family_key"] for q in bank}) == len(bank)
    assert not {q["qid"] for q in bank} & {q["qid"] for q in parents}
    assets = ["candidate_bank.json", "candidate_preparation.json", "quotas.json", "retained_episodes.json", "parent_selection.json"]
    sources = ["scripts/breadth_collect.py", "scripts/breadth_source.py", "scripts/question_breadth.py",
               "scripts/local_conversation_pilot.py", "scripts/hpc3_prepare.py", "scripts/hpc3_breadth.py",
               "scripts/hpc3_parallel_collect.py", "scripts/hpc3_recovery.py", "scripts/breadth_expand.py",
               "hpc3/container.sh", "data/raw/upstream/data/J_matrices.json", "data/raw/upstream/data/obj/personas.json"]
    prior = [PARENT / "selection.json", PARENT / "episode_index.json", PARENT / "collection_complete.json",
             PARENT / "source_protocol.json", FIRST / "source_protocol.json", FIRST / "candidate_bank.json", PARENT / "candidate_bank.json"]
    protocol = {"created_utc": now(), "config": settings, "config_sha256": sha(CONFIG),
        "code_sha256": {p: sha(p) for p in CODE}, "source_files_sha256": {p: sha(p) for p in sources},
        "prior_artifacts": {str(p): sha(p) for p in prior}, "assets": {p: sha(ROOT / p) for p in assets},
        "new_questions": settings["new_questions"], "target_questions": settings["target_splits"],
        "candidate_count": len(bank), "reserved_assessment_qids": [q["qid"] for q in parents if q["split"] == "assessment"],
        "selection_policy": "User-approved revised sampling: screen all fresh eligible deduplicated MATH rows. Added topic quotas are proportional to availability, with exact A/B balance. Rank pooled 32-persona by 8-vote entropy within topic/label cells. Retain all 400 prior conversations and their split assignments. Never use predictor errors or interaction outcomes for selection.",
        "population_change": "The old equal-topic top-quarter rule cannot supply 3,600 fresh questions. Selection fractions and topic mix change. Record entropy distributions before collection; entropy drift is descriptive, not a post-hoc veto. Later learning curves must use matched subsets/validation, with the prior 80 validation questions reported separately.",
        "split_policy": "Fixed topic/label quotas, exact 2,880/720 added split, seeded level/entropy balance, fixed graph totals 400 training and 100 validation per J0-J7. No reassignment of prior questions.",
        "source_policy": "Unchanged pinned 4-bit Llama 3.1 8B/MLX float32 generator, temperature .7, original prompts and RNG IDs, 32 personas, 8 updates, 5 fresh votes per state. New GPU environments require the complete original numerical replay and each worker repeats it. Partial units stay pinned to their initial runtime and batch configuration.",
        "assessment_generated_or_scored": False, "automatic_retraining": False, "paid_api_calls": 0}
    write(ROOT / "protocol.json", protocol)
    publish_immutable(ROOT / "protocol.json", ROOT / "source_protocol.json")
    return verify()


def verify():
    p = read(ROOT / "protocol.json")
    assert sha(ROOT / "source_protocol.json") == sha(ROOT / "protocol.json")
    assert sha(CONFIG) == p["config_sha256"]
    for group in ("code_sha256", "source_files_sha256", "prior_artifacts"):
        for name, digest in p[group].items():
            assert sha(name) == digest, name
    for name, digest in p["assets"].items():
        assert sha(ROOT / name) == digest, name
    for root in (FIRST, PARENT, ROOT):
        assert not list((root / "episodes/assessment").glob("*.json"))
    return p


def choose_split(selected, parents, quota, settings):
    retained = [q for q in parents if q["split"] in ("train", "validation")]
    combined = retained + selected
    total = len(combined)
    levels = Counter(q["level"] for q in combined)
    means = {f: float(np.mean([q["screen"][f] for q in combined])) for f in METRICS}
    scales = {f: max(float(np.std([q["screen"][f] for q in combined])), .01) for f in METRICS}
    cells = {(t, a): [q for q in selected if q["type"] == t and q["answer"] == a]
             for t in sorted(quota["new_selected"]) for a in "AB"}
    best = None
    for attempt in range(settings["selection"]["split_attempts"]):
        rng = random.Random(settings["selection"]["split_seed"] + attempt)
        parts = {"train": [], "validation": []}
        for (topic, label), values in cells.items():
            values = values.copy()
            rng.shuffle(values)
            n = quota["new_validation"][topic][label]
            parts["validation"] += values[:n]
            parts["train"] += values[n:]
        loss = 0.
        for split, rows in parts.items():
            full = rows + [q for q in retained if q["split"] == split]
            counts = Counter(q["level"] for q in full)
            loss += sum((counts[level] - n * len(full) / total)**2 / max(n * len(full) / total, 1.) for level, n in levels.items())
            loss += sum(((np.mean([q["screen"][f] for q in full]) - means[f]) / scales[f])**2 for f in METRICS)
        if best is None or (loss, attempt) < best[:2]:
            best = (float(loss), attempt, parts)
    added = []
    for split, rows in best[2].items():
        graphs = settings["source"]["graph_ids"]
        existing = Counter(q["graph_id"] for q in retained if q["split"] == split)
        target = settings["target_splits"][split]
        assert target % len(graphs) == 0
        schedule = [g for g in graphs for _ in range(target // len(graphs) - existing[g])]
        assert len(schedule) == len(rows)
        random.Random(settings["selection"]["graph_seed"] + sum(map(ord, split))).shuffle(schedule)
        added += [{**q, "split": split, "graph_id": g, "origin": "new"} for q, g in zip(sorted(rows, key=lambda q: q["qid"]), schedule)]
    return added, {"attempt": best[1], "imbalance_score": best[0]}


def select():
    protocol = verify()
    if (ROOT / "selection_ready.json").exists():
        return verify_selection()
    settings, bank, quota = protocol["config"], read(ROOT / "candidate_bank.json"), read(ROOT / "quotas.json")
    records, invalid = [], []
    for row in bank:
        qpath, spath = ROOT / "constructed" / (row["qid"] + ".json"), ROOT / "screen" / (row["qid"] + ".json")
        q, screen = read(qpath), read(spath)
        assert q["qid"] == screen["qid"] == row["qid"] and screen["question_sha256"] == sha(qpath)
        assert screen["protocol_sha256"] == sha(ROOT / "source_protocol.json")
        if screen["status"] != "complete":
            invalid.append(q["qid"])
            continue
        stats = original.screening_statistics(screen["votes"])
        for name, value in stats.items():
            np.testing.assert_allclose(value, screen["statistics"][name], atol=1e-12, rtol=0)
        records.append({**q, "screen": stats})
    selected = []
    for topic, labels in quota["new_selected"].items():
        for label, n in labels.items():
            rows = sorted((q for q in records if q["type"] == topic and q["answer"] == label),
                          key=lambda q: (-q["screen"]["pooled_entropy_bits"], q["qid"]))
            if len(rows) < n:
                raise RuntimeError(f"Insufficient valid screens for {topic}/{label}: {len(rows)} < {n}")
            selected += rows[:n]
    parents = read(ROOT / "parent_selection.json")["questions"]
    added, split = choose_split(selected, parents, quota, settings)
    retained = [q for q in parents if q["split"] in ("train", "validation")]
    distributions = {label: {f: {"mean": float(np.mean([q["screen"][f] for q in rows])),
                       "quantiles": np.quantile([q["screen"][f] for q in rows], [0, .1, .25, .5, .75, .9, 1]).tolist()}
                    for f in METRICS} for label, rows in [("candidates", records), ("retained", retained), ("new", added), ("combined", retained + added)]}
    questions = sorted([{**q, "parent_origin": q.get("origin"), "origin": "retained"} for q in parents] + added, key=lambda q: q["qid"])
    assert Counter(q["split"] for q in questions) == settings["target_splits"]
    assert Counter(q["split"] for q in added) == settings["new_questions"]
    assert len({q["qid"] for q in questions}) == len({q["family_key"] for q in questions}) == len(questions)
    for part in ("train", "validation"):
        rows = [q for q in questions if q["split"] == part]
        assert Counter(q["answer"] for q in rows) == {"A": len(rows) // 2, "B": len(rows) // 2}
        assert set(Counter(q["graph_id"] for q in rows).values()) == {len(rows) // 8}
    write(ROOT / "selection_diagnostic.json", {"created_utc": now(), "entropy": distributions, "split": split,
        "invalid_screens": invalid, "candidate_screens": len(bank), "topic_quotas": quota,
        "population_changed_from_prior": True, "entropy_drift_is_descriptive": True})
    write(ROOT / "selection.json", {"questions": questions, "new_questions": added, "created_utc": now(),
        "protocol_sha256": sha(ROOT / "protocol.json"), "split_details": split})
    write(ROOT / "selection_ready.json", {"completed_utc": now(), "protocol_sha256": sha(ROOT / "protocol.json"),
        "selection_sha256": sha(ROOT / "selection.json"), "diagnostic_sha256": sha(ROOT / "selection_diagnostic.json"),
        "counts_and_integrity_passed": True, "assessment_included_in_generation": False})
    return verify_selection()


def verify_selection():
    verify()
    ready = read(ROOT / "selection_ready.json")
    assert ready["protocol_sha256"] == sha(ROOT / "protocol.json") and ready["counts_and_integrity_passed"]
    assert ready["selection_sha256"] == sha(ROOT / "selection.json")
    assert ready["diagnostic_sha256"] == sha(ROOT / "selection_diagnostic.json")
    return read(ROOT / "selection.json")


def episode_summary(q, path, graphs, expected_sha):
    assert q["split"] in ("train", "validation")
    e = read(path)
    expected = {**{k: q[k] for k in ("qid", "graph_id", "split")}, "protocol_sha256": expected_sha,
                "name": f"{q['qid']}__{q['graph_id']}__r00", "n": 32, "k": 5, "steps": 8, "replica": 0}
    assert all(e["meta"].get(k) == v for k, v in expected.items())
    spins, votes = np.asarray(e["spins_history"]), np.asarray(e["spins_raw_history"])
    assert spins.shape == (9, 32) and votes.shape == (9, 32, 5)
    assert np.isin(spins, [-1, 0, 1]).all() and np.isin(votes, [-1, 0, 1]).all()
    total = votes.sum(-1)
    np.testing.assert_array_equal(spins, np.where(total > 0, 1, np.where(total < 0, -1, votes[..., 0])))
    np.testing.assert_array_equal(e["J"], graphs[q["graph_id"]])
    assert len(e["messages_history"]) == len(e["communicated_stances_history"]) == 8
    return {"qid": q["qid"], "split": q["split"], "graph_id": q["graph_id"], "origin": q["origin"],
            "path": str(path), "sha256": sha(path), "source_protocol_sha256": expected_sha,
            "valid_snapshots": int(((spins[:-1] != 0) & (spins[1:] != 0)).all(-1).sum()), "invalid_votes": int((votes == 0).sum())}


def finalize():
    selection = verify_selection()
    retained = {r["qid"]: r for r in read(ROOT / "retained_episodes.json")}
    graphs = {g["id"]: g["J"] for g in read(original.UPSTREAM / "J_matrices.json")["graphs"]}
    rows = []
    for q in selection["questions"]:
        if q["split"] == "assessment":
            continue
        path = ROOT / "episodes" / q["split"] / f"{q['qid']}__{q['graph_id']}__r00.json"
        expected_sha = retained[q["qid"]]["source_protocol_sha256"] if q["qid"] in retained else sha(ROOT / "protocol.json")
        if q["qid"] in retained:
            assert sha(path) == retained[q["qid"]]["sha256"]
        rows.append(episode_summary(q, path, graphs, expected_sha))
    expected = {k: v for k, v in config()["target_splits"].items() if k != "assessment"}
    assert Counter(r["split"] for r in rows) == expected
    write(ROOT / "episode_index.json", rows)
    write(ROOT / "collection_complete.json", {"completed_utc": now(), "protocol_sha256": sha(ROOT / "protocol.json"),
        "selection_sha256": sha(ROOT / "selection.json"), "index_sha256": sha(ROOT / "episode_index.json"),
        "counts": expected, "new_conversations": sum(config()["new_questions"].values()), "retained_conversations": len(retained),
        "invalid_votes": sum(r["invalid_votes"] for r in rows), "assessment_generated_or_scored": False, "predictors_trained": False})
    return read(ROOT / "collection_complete.json")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["candidates", "prepare", "verify", "select", "finalize"])
    args = parser.parse_args()
    result = {"candidates": build_candidates, "prepare": prepare, "verify": verify, "select": select, "finalize": finalize}[args.stage]()
    if result is not None:
        print(json.dumps({"stage": args.stage, "completed": True, "root": str(ROOT)}), flush=True)
