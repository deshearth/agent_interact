import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pytest

from scripts import breadth_scale as study
from scripts import hpc3_scale_controller as control
from scripts import hpc3_scale_worker as worker
from scripts.hpc3_prepare import read, sha, write
from scripts.breadth_collect import construct, screen, run_episode, TimeLimit
from scripts.local_conversation_pilot import formatter


def test_available_topic_quotas_and_split_preserve_balance_and_old_questions():
    settings = study.config()
    settings["new_questions"] = {"train": 32, "validation": 16}
    settings["target_splits"] = {"train": 48, "validation": 32, "assessment": 2}
    settings["selection"]["split_attempts"] = 5
    bank = []
    for t, count in [("a", 100), ("b", 30), ("c", 14)]:
        for i in range(count):
            bank.append({"qid": f"{t}{i}", "type": t, "answer": "AB"[i % 2], "level": str(i % 3),
                         "screen": {f: (i % 9) / 10 for f in study.METRICS}})
    quota = study.quotas(bank, settings)
    selected = []
    for t, labels in quota["new_selected"].items():
        for a, n in labels.items():
            selected += [q for q in bank if q["type"] == t and q["answer"] == a][:n]
    parents = [{"qid": f"old{s}{i}", "type": "a", "answer": "AB"[i % 2], "level": str(i % 3),
                "split": s, "graph_id": f"J{i % 8}", "screen": {f: .5 for f in study.METRICS}}
               for s in ["train", "validation"] for i in range(16)]
    saved = copy.deepcopy(parents)
    added, _ = study.choose_split(selected, parents, quota, settings)
    assert parents == saved
    assert Counter(q["split"] for q in added) == settings["new_questions"]
    for s in ["train", "validation"]:
        rows = [q for q in parents + added if q["split"] == s]
        assert Counter(q["answer"] for q in rows) == {a: len(rows) // 2 for a in "AB"}
        assert set(Counter(q["graph_id"] for q in rows).values()) == {len(rows) // 8}
    assert quota["new_selected"]["a"]["A"] > quota["new_selected"]["c"]["A"]
    with pytest.raises(ValueError, match="capacity"):
        study.apportion(9, {"a": 1}, {"a": 8})


class Source:
    def __init__(self, execution, stop_after=None):
        self.execution, self.stop_after, self.calls = execution, stop_after, []

    def generate_many(self, prompts, ids, **kwargs):
        if self.stop_after is not None and len(self.calls) == self.stop_after:
            raise TimeLimit()
        self.calls.append((prompts, ids, kwargs))
        rows = []
        for sid in ids:
            sign = 1 if hashlib.sha256(sid.encode()).digest()[0] % 2 else -1
            rows.append({"sample_id": sid, "raw_response": "1" if sid.startswith("construct|") else "A" if sign == 1 else "B",
                         "finish_reason": "stop", "parsed_sign": sign})
        return {"responses": rows, "hpc3_execution": self.execution}


@pytest.mark.parametrize("phase", ["screen", "collect"])
def test_interruption_preserves_original_requests_responses_and_resume_execution(tmp_path, monkeypatch, phase):
    fmt = formatter()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(control, "ROOT", tmp_path / "controller")
    monkeypatch.setattr(study, "ROOT", tmp_path / "data")
    q = {"qid": "q0", "question": "What is 1+1?", "correct_answer": "2", "distractor_candidates": ["3", "4", "5", "6"],
         "answer": "A", "choices": {"A": "2", "B": "3"}, "split": "train", "graph_id": "J0"}
    graph = [[0] * 32 for _ in range(32)]
    for i in range(32):
        graph[i][(i + 1) % 32] = 1 if i % 2 else -1
    personas, execution = ["persona"] * 32, {"runtime": {"gpu": "same"}}
    serial, resumed_root = tmp_path / "serial", tmp_path / "resumed"
    for root in [serial, resumed_root]:
        write(root / "source_protocol.json", {"fixed": True})

    def generate(source, root):
        recorder = worker.make_recorder(source, root, time.time() + 60, [q], phase, execution)
        if phase == "screen":
            construct(recorder, [q])
            screen(recorder, [q], personas, fmt)
            return read(root / "screen/q0.json")
        return run_episode(recorder, q, personas, graph, fmt, 2)

    baseline_source = Source(execution)
    expected = generate(baseline_source, serial)
    monkeypatch.setattr(study, "ROOT", tmp_path / "other_data")
    partial = Source(execution, stop_after=5)
    with pytest.raises(TimeLimit):
        generate(partial, resumed_root)
    saved = {p: sha(p) for p in (resumed_root / "chunks").glob("*.json")}
    resumed_source = Source(execution)
    actual = generate(resumed_source, resumed_root)
    assert partial.calls + resumed_source.calls == baseline_source.calls
    assert {k: v for k, v in actual.items() if k != "completed_utc"} == {k: v for k, v in expected.items() if k != "completed_utc"}
    assert all(sha(p) == d for p, d in saved.items())
    record = read(next((resumed_root / "chunks").glob("*.json")))
    changed = worker.make_recorder(Source({"changed": True}), resumed_root, time.time() + 60, [q], phase, {"changed": True})
    group, calls = next((record["group_id"], c) for c in baseline_source.calls if c[1][0] in [r["sample_id"] for r in record["responses"]])
    metadata = {"qid": "q0", "kind": "opinion", "phase": "screen" if phase == "screen" else "interaction"}
    if group.startswith("scale_construct|"):
        group = "construct|group0"
    with pytest.raises(ValueError, match="Execution settings"):
        changed.request(group, [{"sample_id": calls[1][0], "prompt": calls[0][0], "metadata": metadata}])


def test_unit_locks_and_runtime_binding_protect_partial_work(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "ROOT", tmp_path / "control")
    monkeypatch.setattr(control, "STUDY", tmp_path / "data")
    monkeypatch.setattr(study, "ROOT", tmp_path / "data")
    write(study.ROOT / "protocol.json", {"fixed": True})
    write(study.ROOT / "source_protocol.json", {"fixed": True})
    units = [{"id": "screen-0", "phase": "screen", "qids": ["q0"]}]
    one, two = {"runtime": "A"}, {"runtime": "B"}
    with worker.claim_next(units, one) as claimed:
        assert claimed == units[0]
        with worker.claim_next(units, one) as other:
            assert other is None
    with worker.claim_next(units, two) as other:
        assert other is None
    with worker.claim_next(units, one) as claimed:
        assert claimed == units[0]


class Scheduler:
    def __init__(self):
        self.current, self.ended, self.submissions = {}, {}, []
        self.counter = 100

    def command(self, argv, **kwargs):
        assert argv[0] == "sbatch"
        job = str(self.counter)
        self.counter += 1
        name = next(a.split("=", 1)[1] for a in argv if a.startswith("--job-name="))
        partition = next((a.split("=", 1)[1] for a in argv if a.startswith("--partition=")), "free")
        self.current[job] = {"job_id": job, "name": name, "partition": partition, "state": "PENDING", "gpu_count": int(partition.startswith("free-gpu"))}
        self.submissions.append(argv)
        return job + "\n"

    def finish(self, job, state="COMPLETED", code="0:0"):
        self.current.pop(job)
        self.ended[job] = {"state": state, "exit_code": code}


@pytest.fixture
def scheduler_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    plan = {"account": "fixture", "partition_limits": {"free-gpu": 24, "free-gpu32": 8},
            "new_questions": {"train": 4, "validation": 4}, "target_questions": {"train": 4, "validation": 4, "assessment": 2},
            "candidate_count": 40, "unit_seed": 123, "reserved_assessment_qids": ["assessment"]}
    monkeypatch.setattr(control, "verify_plan", lambda: plan)
    monkeypatch.setattr(control, "refresh_hardware", lambda: {"entries": [{"gpu": g, "runtime_key": g} for g in ["A30", "L40S"]]})
    scheduler = Scheduler()
    monkeypatch.setattr(control, "queue", lambda account: copy.deepcopy(scheduler.current))
    monkeypatch.setattr(control, "accounting", lambda ids: {i: scheduler.ended[i] for i in ids if i in scheduler.ended})
    monkeypatch.setattr(control, "command", scheduler.command)
    write(control.ROOT / "jobs.json", {"latest": {}, "history": []})
    write(control.ROOT / "plan.json", plan)
    write(control.ROOT / "state.json", {"status": "prepared"})
    units = [{"id": f"screen-{i}", "phase": "screen", "qids": [f"q{i}"]} for i in range(40)]
    write(control.ROOT / "screen_units.json", units)
    return scheduler, units, plan


def test_maximum_two_partition_pool_counts_other_gpu_allocations_and_recovers(scheduler_fixture):
    scheduler, _, _ = scheduler_fixture
    scheduler.current["other"] = {"job_id": "other", "name": "other-work", "partition": "free-gpu", "state": "RUNNING", "gpu_count": 3}
    control.controller()
    assert len(scheduler.submissions) == 29  # 21 main plus 8 newer GPUs
    control.controller()
    assert len(scheduler.submissions) == 29
    scheduler.finish("100", "PREEMPTED")
    control.controller()
    assert len(scheduler.submissions) == 30
    assert read(control.ROOT / "jobs.json")["history"][0]["terminal"]["state"] == "PREEMPTED"
    scheduler.finish("129", "FAILED", "1:0")
    control.controller()
    assert control.stopped() and len(scheduler.submissions) == 30


def test_uncertified_partition_is_not_used_and_small_tail_is_not_oversubscribed(scheduler_fixture, monkeypatch):
    scheduler, units, _ = scheduler_fixture
    monkeypatch.setattr(control, "refresh_hardware", lambda: {"entries": [{"gpu": "A30", "runtime_key": "A30"}]})
    write(control.ROOT / "screen_units.json", units[:3])
    control.controller()
    assert len(scheduler.submissions) == 3
    assert all(r["partition"] == "free-gpu" for r in scheduler.current.values())
    monkeypatch.setattr(control, "refresh_hardware", lambda: {"entries": [{"gpu": g, "runtime_key": g} for g in ["A30", "L40S"]]})
    control.controller()
    assert len(scheduler.submissions) == 3


def test_stage_barriers_recovery_and_assessment_exclusion(scheduler_fixture):
    scheduler, units, plan = scheduler_fixture
    for u in units:
        write(control.STUDY / "screen" / (u["qids"][0] + ".json"), {})
    control.controller()
    assert scheduler.submissions[-1][-1] == "select"
    scheduler.finish("100", "TIMEOUT")
    control.controller()
    assert scheduler.submissions[-1][-1] == "select" and len(scheduler.submissions) == 2
    scheduler.finish("101")
    questions = [{"qid": f"new{i}", "split": "train" if i < 4 else "validation", "graph_id": f"J{i % 8}"} for i in range(8)]
    write(control.STUDY / "selection.json", {"new_questions": questions})
    write(control.STUDY / "selection_ready.json", {"selection_sha256": sha(control.STUDY / "selection.json")})
    control.controller()
    assert len(scheduler.current) == 8
    for q in questions:
        write(control.STUDY / "episodes" / q["split"] / f"{q['qid']}__{q['graph_id']}__r00.json", {})
    for job in list(scheduler.current):
        scheduler.finish(job)
    control.controller()
    assert scheduler.submissions[-1][-1] == "finalize"
    scheduler.finish(next(iter(scheduler.current)))
    write(control.STUDY / "collection_complete.json", {"complete": True})
    write(control.STUDY / "episode_index.json", [])
    control.controller()
    assert read(control.ROOT / "state.json")["status"] == "complete"
    bad = copy.deepcopy(questions)
    bad[0]["qid"] = "assessment"
    with pytest.raises(AssertionError):
        control.collect_units({"new_questions": bad}, plan)


def test_cron_is_independent_and_preserves_existing_jobs(scheduler_fixture, monkeypatch):
    scheduler, _, _ = scheduler_fixture
    before = "MAILTO=test@example.test\n0 1 * * * other\n"
    entry = "*/5 * * * * /task " + control.MARKER
    assert control.cron_text(control.cron_text(before, entry), entry) == control.cron_text(before, entry)
    assert control.cron_text(control.cron_text(before, entry), None) == before
    control.tick()
    control.tick()
    assert len(scheduler.submissions) == 1
    (control.ROOT / "STOP").touch()
    monkeypatch.setattr(control, "existing_cron", lambda: control.cron_text(before, entry))
    calls = []
    monkeypatch.setattr(control, "command", lambda cmd, **kw: calls.append((cmd, kw)))
    control.tick(True)
    assert calls == [(["crontab", "-"], {"input": before})]


def test_full_synthetic_selection_and_collection_keep_reserved_questions_separate(tmp_path, monkeypatch):
    fmt = formatter()
    settings = study.config()
    settings["new_questions"] = {"train": 16, "validation": 8}
    settings["target_splits"] = {"train": 24, "validation": 16, "assessment": 2}
    settings["selection"]["split_attempts"] = 3
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(study, "ROOT", tmp_path / "data")
    monkeypatch.setattr(control, "ROOT", tmp_path / "control")
    monkeypatch.setattr(study.original, "UPSTREAM", tmp_path / "upstream")
    monkeypatch.setattr(study, "config", lambda: settings)
    protocol = {"config": settings}
    monkeypatch.setattr(study, "verify", lambda: protocol)
    write(study.ROOT / "protocol.json", protocol)
    write(study.ROOT / "source_protocol.json", protocol)
    rng = np.random.default_rng(12)
    bank = []
    for i in range(48):
        row = {"qid": f"q{i:03d}", "question": f"Example {i}", "choices": {"A": "2", "B": "3"},
               "type": str(i % 2), "answer": "AB"[(i // 2) % 2], "level": str(i % 3), "family_key": f"family{i}"}
        bank.append(row)
        qpath = study.ROOT / "constructed" / (row["qid"] + ".json")
        write(qpath, row)
        votes = rng.choice([-1, 1], (32, 8)).tolist()
        write(study.ROOT / "screen" / (row["qid"] + ".json"), {
            "qid": row["qid"], "question_sha256": sha(qpath), "protocol_sha256": sha(study.ROOT / "protocol.json"),
            "status": "complete", "votes": votes, "statistics": study.original.screening_statistics(votes)})
    parents = []
    for split in ["train", "validation", "assessment"]:
        for i in range(2 if split == "assessment" else 8):
            parents.append({"qid": f"old-{split}-{i}", "question": "Old example", "choices": {"A": "2", "B": "3"},
                "type": str(i % 2), "answer": "AB"[i % 2], "level": str(i % 3), "family_key": f"old-family-{split}-{i}",
                "split": split, "graph_id": f"J{i % 8}", "origin": "old", "screen": {f: .9 for f in study.METRICS}})
    write(study.ROOT / "candidate_bank.json", bank)
    write(study.ROOT / "quotas.json", study.quotas(bank, settings))
    write(study.ROOT / "parent_selection.json", {"questions": parents})
    selected = study.select()
    assert Counter(q["split"] for q in selected["questions"]) == settings["target_splits"]
    assert len(selected["new_questions"]) == 24
    before = sha(study.ROOT / "selection.json")
    assert study.select() == selected and sha(study.ROOT / "selection.json") == before
    graphs = {f"J{i}": [[0] * 32 for _ in range(32)] for i in range(8)}
    write(study.original.UPSTREAM / "J_matrices.json", {"graphs": [{"id": k, "J": v} for k, v in graphs.items()]})
    old_root = tmp_path / "old"
    write(old_root / "source_protocol.json", {"old": True})
    retained = []
    for q in selected["questions"]:
        if q["split"] == "assessment":
            continue
        root = old_root if q["origin"] == "retained" else study.ROOT
        execution = {"runtime": {"fake": True}}
        recorder = worker.make_recorder(Source(execution), root, time.time() + 60, [q], "collect", execution)
        run_episode(recorder, q, ["persona"] * 32, graphs[q["graph_id"]], fmt, 8)
        path = root / "episodes" / q["split"] / f"{q['qid']}__{q['graph_id']}__r00.json"
        if q["origin"] == "retained":
            destination = study.ROOT / "episodes" / q["split"] / path.name
            worker.parallel.publish_immutable(path, destination)
            retained.append({"qid": q["qid"], "sha256": sha(path), "source_protocol_sha256": sha(old_root / "source_protocol.json")})
    write(study.ROOT / "retained_episodes.json", retained)
    result = study.finalize()
    assert result["counts"] == {"train": 24, "validation": 16}
    assert result["new_conversations"] == 24 and result["retained_conversations"] == 16
    assert not result["assessment_generated_or_scored"]
    assert not (study.ROOT / "episodes/assessment").exists()
    assert len(read(study.ROOT / "episode_index.json")) == 40
