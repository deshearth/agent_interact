"""Recovery, disjoint ownership and unchanged generated requests under resume."""
import copy
import hashlib
import os
from pathlib import Path
import time

import pytest

from scripts import hpc3_recovery as control
from scripts import hpc3_expansion_pool as pool
from scripts import hpc3_breadth_expand as original
from scripts import hpc3_parallel_collect as parallel
from scripts.hpc3_prepare import read, sha, write
from scripts.breadth_collect import TimeLimit, construct, screen, run_episode
from scripts.local_conversation_pilot import formatter


def test_screen_units_preserve_original_four_request_batches():
    bank = [{"qid": f"q{i:04d}"} for i in range(1200)]
    shards = original.assignments(bank, 8, "screen")
    units = control.screen_units(bank, [[q["qid"] for q in shard] for shard in shards])
    assert len(units) == 304
    assert len({qid for unit in units for qid in unit["qids"]}) == 1200
    for owner, shard in enumerate(shards):
        actual = [u["qids"] for u in units if u["legacy_root"].endswith(f"screen_{owner:02d}")]
        expected = [[q["qid"] for q in shard[i:i + 4]] for i in range(0, 150, 4)]
        assert actual == expected


def selected():
    return {"new_questions": [{"qid": f"q{i:04d}", "split": "train" if i < 240 else "validation",
                               "graph_id": f"J{i % 8}", "origin": "new"} for i in range(300)]}


def test_collection_units_are_disjoint_preserve_graphs_and_reject_assessment():
    selection = selected()
    units = control.collection_units(selection)
    assert len(units) == len({u["qids"][0] for u in units}) == 300
    shards = original.assignments(selection["new_questions"], 8, "collect")
    owners = {q["qid"]: owner for owner, shard in enumerate(shards) for q in shard}
    for unit in units:
        assert unit["legacy_root"].endswith(f"collect_{owners[unit['qids'][0]]:02d}")
    selection["new_questions"][-1]["split"] = "assessment"
    with pytest.raises(ValueError, match="Assessment"):
        control.collection_units(selection)


class Source:
    def __init__(self, execution, stop_after=None):
        self.execution, self.stop_after, self.calls = execution, stop_after, []

    def generate_many(self, prompts, sample_ids, **kwargs):
        if self.stop_after is not None and len(self.calls) == self.stop_after:
            raise TimeLimit()
        self.calls.append((prompts, sample_ids, kwargs))
        rows = []
        for sid in sample_ids:
            sign = 1 if hashlib.sha256(sid.encode()).digest()[0] % 2 else -1
            rows.append({"sample_id": sid, "raw_response": "1" if sid.startswith("construct|") else "A" if sign == 1 else "B",
                         "finish_reason": "stop", "parsed_sign": sign})
        return {"responses": rows, "hpc3_execution": self.execution}


def synthetic_bank():
    return [{"qid": f"q{i}", "question": f"What is {i} + 2?", "correct_answer": str(i + 2),
             "distractor_candidates": [str(i + 3), str(i + 4), str(i + 5), str(i + 6)], "answer": "A"}
            for i in range(4)]


def test_screen_handoff_reuses_partial_cache_with_identical_prompts_ids_and_votes(tmp_path, monkeypatch):
    fmt = formatter()
    monkeypatch.chdir(tmp_path)
    legacy, destination, canonical, serial = [tmp_path / p for p in ["legacy", "unit", "canonical", "serial"]]
    monkeypatch.setattr(original, "ROOT", canonical)
    for root in [legacy, destination, serial]:
        write(root / "source_protocol.json", {"frozen": True})
    bank, personas, execution = synthetic_bank(), ["persona"] * 32, {"gpu": "fixed"}
    expected_source = Source(execution)
    baseline = original.make_recorder(expected_source, serial, time.time() + 60, bank, "screen", execution)
    # Keep the baseline's canonical publication separate from the handoff.
    monkeypatch.setattr(original, "ROOT", tmp_path / "baseline_canonical")
    construct(baseline, bank)
    screen(baseline, bank, personas, fmt)
    monkeypatch.setattr(original, "ROOT", canonical)
    partial = Source(execution, stop_after=5)
    old = original.make_recorder(partial, legacy, time.time() + 60, bank, "screen", execution)
    construct(old, bank)
    with pytest.raises(TimeLimit):
        screen(old, bank, personas, fmt)
    preserved = {p: sha(p) for p in legacy.rglob("*.json")}
    resumed = Source(execution)
    recorder = pool.make_recorder(resumed, destination, time.time() + 60, bank, "screen", execution, legacy)
    construct(recorder, bank)
    screen(recorder, bank, personas, fmt)
    assert partial.calls + resumed.calls == expected_source.calls
    for q in bank:
        expected = read(serial / "screen" / (q["qid"] + ".json"))
        actual = read(destination / "screen" / (q["qid"] + ".json"))
        assert {k: v for k, v in actual.items() if k != "completed_utc"} == {
            k: v for k, v in expected.items() if k != "completed_utc"}
    assert all(sha(p) == digest for p, digest in preserved.items())
    changed = pool.make_recorder(Source({"gpu": "changed"}), destination, time.time() + 60, bank,
                                 "screen", {"gpu": "changed"}, legacy)
    requests = [{"sample_id": f"construct|{q['qid']}", "prompt": "changed", "metadata": {"qid": q["qid"]}} for q in bank]
    with pytest.raises(AssertionError, match="Changed execution"):
        changed.request("construct|group0", requests, 64)


def test_conversation_handoff_is_exact_after_interruption(tmp_path, monkeypatch):
    fmt = formatter()
    monkeypatch.chdir(tmp_path)
    canonical = tmp_path / "canonical"
    monkeypatch.setattr(original, "ROOT", canonical)
    q = {"qid": "q0", "split": "train", "graph_id": "J0", "question": "What is 1 + 1?",
         "choices": {"A": "2", "B": "3"}}
    graph = [[0] * 32 for _ in range(32)]
    for i in range(32):
        graph[i][(i + 1) % 32] = 1 if i % 2 else -1
    personas, execution = ["persona"] * 32, {"fixed": True}
    serial, legacy, destination = [tmp_path / x for x in ["serial", "legacy", "unit"]]
    for root in [serial, legacy, destination]:
        write(root / "source_protocol.json", {"frozen": True})
    expected_source = Source(execution)
    monkeypatch.setattr(original, "ROOT", tmp_path / "baseline_canonical")
    expected = run_episode(original.make_recorder(expected_source, serial, time.time() + 60, [q], "collect", execution),
                           q, personas, graph, fmt, 2)
    monkeypatch.setattr(original, "ROOT", canonical)
    partial = Source(execution, stop_after=13)
    with pytest.raises(TimeLimit):
        run_episode(original.make_recorder(partial, legacy, time.time() + 60, [q], "collect", execution),
                    q, personas, graph, fmt, 2)
    resumed = Source(execution)
    actual = run_episode(pool.make_recorder(resumed, destination, time.time() + 60, [q], "collect", execution, legacy),
                         q, personas, graph, fmt, 2)
    assert partial.calls + resumed.calls == expected_source.calls
    assert {k: v for k, v in actual.items() if k != "completed_utc"} == {
        k: v for k, v in expected.items() if k != "completed_utc"}


def test_unit_locks_prevent_duplicate_claims_and_exclude_legacy_writer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    units = [{"id": f"u{i}", "phase": "screen", "qids": [f"q{i}"], "legacy_root": "legacy"} for i in range(2)]
    with control.lock(Path("legacy/.worker.lock")):
        with pool.claim_next(units) as unit:
            assert unit is None
    with pool.claim_next(units) as first:
        with pool.claim_next(units) as second:
            assert first["id"] != second["id"]
            with pool.claim_next(units) as third:
                assert third is None
    with pool.claim_next(units) as resumed:
        assert resumed["id"] == "u0"


@pytest.mark.parametrize("state,code,expected", [
    ("PREEMPTED", "0:0", "retry"), ("NODE_FAIL", "0:0", "retry"),
    ("TIMEOUT", "0:0", "retry"), ("FAILED", "75:0", "retry"),
    ("FAILED", "1:0", "halt"), ("OUT_OF_MEMORY", "0:0", "halt"),
    ("CANCELLED", "0:0", "halt"), ("COMPLETED", "0:0", "finished")])
def test_retry_only_scheduler_interruptions_or_declared_checkpoint_exit(state, code, expected):
    assert control.terminal_action({"state": state, "exit_code": code}) == expected
    assert control.terminal_action(None) == "wait"


class Scheduler:
    def __init__(self):
        self.current, self.ended, self.submissions = {}, {}, []
        self.next_id = 100

    def command(self, argv, **kwargs):
        assert argv[0] == "sbatch"
        job_id = str(self.next_id)
        self.next_id += 1
        name = next(a.split("=", 1)[1] for a in argv if a.startswith("--job-name="))
        partition = "free-gpu" if "pool" in name or "unseen" in name else "free"
        self.current[job_id] = {"job_id": job_id, "name": name, "partition": partition, "state": "PENDING"}
        self.submissions.append(argv)
        return job_id + "\n"

    def finish(self, job_id, state="COMPLETED", code="0:0"):
        self.current.pop(job_id)
        self.ended[job_id] = {"state": state, "exit_code": code}


@pytest.fixture
def controller_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "900")
    plan = {"account": "fixture", "maximum_gpu_jobs": 24, "legacy_expansion_jobs": []}
    monkeypatch.setattr(control, "verify_plan", lambda: plan)
    scheduler = Scheduler()
    monkeypatch.setattr(control, "queue", lambda account: copy.deepcopy(scheduler.current))
    monkeypatch.setattr(control, "accounting", lambda ids: {i: scheduler.ended[i] for i in ids if i in scheduler.ended})
    monkeypatch.setattr(control, "command", scheduler.command)
    write(control.ROOT / "jobs.json", {"latest": {}, "history": []})
    write(control.ROOT / "plan.json", plan)
    write(control.ROOT / "state.json", {"status": "prepared"})
    write(control.UNSEEN / "protocol.json", {"questions": []})
    write(control.UNSEEN / "deployment.json", {"assignments": [[], [], [], []]})
    write(control.UNSEEN / "results.json", {})
    units = [{"id": f"s{i}", "phase": "screen", "qids": [f"q{i}"], "legacy_root": f"legacy{i}"} for i in range(30)]
    write(control.ROOT / "screen_units.json", units)
    return scheduler, units


def test_controller_resumes_preemption_without_duplicates_and_obeys_total_gpu_cap(controller_fixture):
    scheduler, units = controller_fixture
    for i in range(23):
        scheduler.current[f"other{i}"] = {"job_id": f"other{i}", "name": "other-work",
                                          "partition": "free-gpu", "state": "RUNNING"}
    control.controller()
    assert len(scheduler.submissions) == 1
    control.controller()
    assert len(scheduler.submissions) == 1
    scheduler.finish("100", "PREEMPTED")
    control.controller()
    assert len(scheduler.submissions) == 2 and len(scheduler.current) == 24
    assert read(control.ROOT / "jobs.json")["history"][0]["terminal"]["state"] == "PREEMPTED"
    scheduler.finish("101", "FAILED", "1:0")
    control.controller()
    assert control.is_stopped() and len(scheduler.submissions) == 2


def test_all_stages_advance_and_selection_preemption_is_retried(controller_fixture, monkeypatch):
    scheduler, units = controller_fixture
    for unit in units:
        write(control.EXPANSION / "screen" / (unit["qids"][0] + ".json"), {})
    control.controller()
    assert scheduler.submissions[-1][-2:] == ["hpc3/breadth_expand_finish.sbatch", "select"]
    scheduler.finish("100", "PREEMPTED")
    control.controller()
    assert len(scheduler.submissions) == 2
    scheduler.finish("101")
    selection = selected()
    write(control.EXPANSION / "selection.json", selection)
    write(control.EXPANSION / "selection_ready.json", {})
    from scripts import breadth_expand
    monkeypatch.setattr(breadth_expand, "verify_selection", lambda: selection)
    control.controller()
    assert len(scheduler.current) == 24
    for unit in control.collection_units(selection):
        write(control.EXPANSION / "episodes" / unit["split"] /
              f"{unit['qids'][0]}__{unit['graph_id']}__r00.json", {})
    for job_id in list(scheduler.current):
        scheduler.finish(job_id)
    control.controller()
    assert scheduler.submissions[-1][-2:] == ["hpc3/breadth_expand_finish.sbatch", "finalize"]
    for job_id in list(scheduler.current):
        scheduler.finish(job_id)
    write(control.EXPANSION / "collection_complete.json", {})
    write(control.EXPANSION / "episode_index.json", [])
    control.controller()
    assert read(control.ROOT / "state.json")["status"] == "complete"
    assert (control.ROOT / "complete.json").exists()


def test_orphan_submission_is_adopted_before_retry(controller_fixture):
    scheduler, units = controller_fixture
    scheduler.current["orphan"] = {"job_id": "orphan", "name": "ag-auto-pool-00",
                                   "partition": "free-gpu", "state": "RUNNING"}
    control.controller()
    assert len(scheduler.submissions) == 23
    assert read(control.ROOT / "jobs.json")["latest"]["pool-00"]["job_id"] == "orphan"


def test_cron_edit_preserves_unrelated_tasks_and_is_idempotent():
    previous = "MAILTO=user@example.test\n0 1 * * * unrelated-task\n"
    entry = "*/5 * * * * /bin/bash /project/hpc3/recovery_tick.sh " + control.CRON_MARKER
    first = control.cron_text(previous, entry)
    assert control.cron_text(first, entry) == first
    assert control.cron_text(first, None) == previous


def test_tick_does_not_duplicate_a_controller(controller_fixture):
    scheduler, units = controller_fixture
    control.tick()
    control.tick()
    assert len(scheduler.submissions) == 1
    assert scheduler.submissions[0][-2:] == ["hpc3/recovery_controller.sbatch", "controller"]


def test_stop_disables_cron_without_submitting_more_jobs(controller_fixture, monkeypatch):
    scheduler, units = controller_fixture
    (control.ROOT / "STOP").touch()
    monkeypatch.setattr(control, "existing_cron", lambda: "0 1 * * * unrelated-task\n")
    edits = []
    monkeypatch.setattr(control, "command", lambda argv, **kwargs: edits.append((argv, kwargs)) or "")
    control.tick(True)
    assert edits == [(["crontab", "-"], {"input": "0 1 * * * unrelated-task\n"})]
    assert not scheduler.submissions


def test_only_startup_cuda_oom_is_retryable_and_success_resets_streak(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    error = RuntimeError("cudaMallocAsync(&data, size, stream) failed: out of memory")
    assert pool.startup_oom("loading_model", error)
    assert not pool.startup_oom("generating", error)
    assert not pool.startup_oom("loading_model", RuntimeError("Changed execution"))
    for attempt in range(1, 5):
        assert pool.record_load_attempt(0, str(attempt), str(error)) == attempt
    assert pool.record_load_attempt(0, "5") == 0
    assert pool.record_load_attempt(0, "6", str(error)) == 1
