"""Parallel scheduling preserves frozen prompts, IDs, split boundaries and caches."""
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import time

import pytest

from scripts import hpc3_parallel_collect as parallel
from scripts.breadth_collect import Recorder, TimeLimit, run_episode
from scripts.hpc3_prepare import STUDY, read, sha, write
from scripts import local_conversation_pilot as prompts


class Source:
    def __init__(self, execution=None, stop_after=None):
        self.calls, self.execution, self.stop_after = [], execution, stop_after

    def generate_many(self, texts, ids, **kwargs):
        if self.stop_after is not None and len(self.calls) == self.stop_after:
            raise TimeLimit()
        self.calls.append((texts, ids, kwargs))
        responses = []
        for sid in ids:
            bit = hashlib.sha256(sid.encode()).digest()[0]
            # Exercise saved retry groups as well as ordinary opinions/messages.
            sign = 0 if sid.endswith("attempt0") and bit % 17 == 0 else (1 if bit % 2 else -1)
            responses.append({"sample_id": sid, "raw_response": "A" if sign == 1 else "B",
                              "parsed_sign": sign})
        result = {"responses": responses}
        if self.execution is not None:
            result["hpc3_execution"] = self.execution
        return result


def inputs():
    persona = read("data/raw/upstream/data/obj/personas.json")[0]["persona"]
    graph = [[0] * 32 for _ in range(32)]
    for i in range(32):
        graph[i][(i + 1) % 32] = 1 if i % 2 else -1
    q = {"qid": "q0", "graph_id": "J0", "split": "train", "question": "What is 1 + 1?",
         "choices": {"A": "2", "B": "3"}}
    return q, [persona] * 32, graph, prompts.formatter()


def test_real_selection_partitions_100_questions_without_assessment():
    selection = read(STUDY / "selection.json")
    splits = read(STUDY / "source_protocol.json")["config"]["questions"]["splits"]
    shards = parallel.assignments(selection, splits, 4)
    assert [len(shard) for shard in shards] == [25] * 4
    assert [sum(q["split"] == "train" for q in shard) for shard in shards] == [20] * 4
    ids = [q["qid"] for shard in shards for q in shard]
    assert len(ids) == len(set(ids)) == 100
    assert set(ids).isdisjoint(q["qid"] for q in selection["questions"] if q["split"] == "assessment")
    bad = copy.deepcopy(selection)
    bad["questions"][0]["split"] = "train"
    with pytest.raises(ValueError, match="split counts"):
        parallel.assignments(bad, splits, 4)


def test_serial_and_interrupted_worker_have_identical_prompts_ids_and_outcomes(tmp_path, monkeypatch):
    q, personas, graph, fmt = inputs()
    monkeypatch.chdir(tmp_path)
    serial, worker = tmp_path / "serial", tmp_path / "worker"
    for root in [serial, worker, STUDY]:
        write(root / "source_protocol.json", {"frozen": True})
    expected_source = Source()
    expected = run_episode(Recorder(expected_source, serial, time.time() + 60), q, personas, graph, fmt, 2)
    name = parallel.episode_name(q)
    first = Source(stop_after=7)
    with pytest.raises(TimeLimit):
        run_episode(parallel.make_recorder(first, worker, time.time() + 60, {name}, {}), q, personas, graph, fmt, 2)
    saved = {p.name: sha(p) for p in (worker / "chunks").glob("*.json")}
    resumed = Source()
    result = run_episode(parallel.make_recorder(resumed, worker, time.time() + 60, {name}, {}), q, personas, graph, fmt, 2)
    assert first.calls + resumed.calls == expected_source.calls
    assert {k: v for k, v in result.items() if k != "completed_utc"} == {
           k: v for k, v in expected.items() if k != "completed_utc"}
    assert all(sha(worker / "chunks" / p) == d == sha(STUDY / "chunks" / p) for p, d in saved.items())
    generated_groups = {read(p)["group_id"] for p in (worker / "chunks").glob("*.json")}
    cfg = {"source": {"n_agents": 32, "update_rounds": 2}}
    assert generated_groups <= set(parallel.group_ids(q, graph, cfg))
    recorder = parallel.make_recorder(resumed, worker, time.time() + 60, {name}, {})
    with pytest.raises(ValueError, match="another worker"):
        recorder.request("other_question__J0__r00|round0|opinions0", [])


def test_conflicting_publication_never_overwrites_a_saved_file(tmp_path):
    first, other, target = [tmp_path / p for p in ["first", "other", "target"]]
    first.write_text("original")
    other.write_text("conflict")
    assert parallel.publish_immutable(first, target)
    assert not parallel.publish_immutable(first, target)
    with pytest.raises(ValueError, match="Conflicting"):
        parallel.publish_immutable(other, target)
    assert target.read_text() == "original"


def test_workers_share_guards_but_exclude_serial_collection_and_replanning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with parallel.collection_locks(shared=True), parallel.collection_locks(shared=True):
        with pytest.raises(RuntimeError, match="still active"):
            with parallel.collection_locks():
                pytest.fail("Cannot replan while any worker is active")
    with parallel.collection_locks():
        with pytest.raises(RuntimeError, match="still active"):
            with parallel.collection_locks(shared=True):
                pytest.fail("Cannot start a worker during serial collection or handoff")


def test_shared_guards_open_readable_descriptors_for_linux_nfs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = parallel.fcntl.flock
    seen = []
    def nfs_flock(stream, operation):
        if operation & fcntl.LOCK_SH:
            mode = fcntl.fcntl(stream.fileno(), fcntl.F_GETFL) & os.O_ACCMODE
            assert mode != os.O_WRONLY, "NFS shared locks require a readable descriptor"
            seen.append(mode)
        return original(stream, operation)
    monkeypatch.setattr(parallel.fcntl, "flock", nfs_flock)
    with parallel.collection_locks(shared=True):
        pass
    assert len(seen) == 4


def prepared_fixture(tmp_path, monkeypatch):
    q, personas, graph, fmt = inputs()
    monkeypatch.chdir(tmp_path)
    cfg = {"source": {"n_agents": 32, "update_rounds": 1, "update_votes_per_agent": 5},
           "questions": {"splits": {"train": 8, "validation": 4, "assessment": 4}}}
    questions = [{**q, "qid": f"q{i}", "split": split} for i, split in enumerate(
                 ["train"] * 8 + ["validation"] * 4 + ["assessment"] * 4)]
    write(STUDY / "source_protocol.json", {"config": cfg})
    write(STUDY / "selection.json", {"questions": questions})
    write("data/raw/upstream/data/J_matrices.json", {"graphs": [{"id": "J0", "J": graph}]})
    write("data/raw/upstream/data/obj/personas.json", [{"persona": p} for p in personas])
    for code in parallel.CODE:
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_text("test fixture")
    manifest = {"production_export": True, "mode": "handoff"}
    write("hpc3/manifest.json", manifest)
    write("hpc3/fixtures.json", {"full_coverage": True})
    report = Path("hpc3/report.json")
    write(report, {"runtime": {"gpu": "fixture"}, "selected_batch_size": 4,
                  "numerical_gate_passed": True, "raw_results": {},
                  "candidates": [{"batch_size": 4, "passed": True, "checks": [{"passed": True}]}]})
    monkeypatch.setattr(parallel.cuda, "verify_package", lambda: manifest)
    monkeypatch.setattr(parallel.cuda, "runtime_identity", lambda: {"gpu": "fixture"})
    monkeypatch.setattr(parallel, "replay_worker", lambda *args: None)
    monkeypatch.setattr(parallel.signal, "signal", lambda *args: None)
    monkeypatch.setattr(prompts, "formatter", lambda: fmt)
    sources = []
    def load(batch, execution):
        sources.append(Source(execution))
        return sources[-1]
    monkeypatch.setattr(parallel.cuda, "load_source", load)
    return cfg, questions, personas, graph, fmt, report, sources


def test_handoff_preserves_partial_work_four_workers_finish_and_resume_without_duplicates(tmp_path, monkeypatch):
    cfg, questions, personas, graph, fmt, report, sources = prepared_fixture(tmp_path, monkeypatch)
    old = Source(stop_after=5)
    with pytest.raises(TimeLimit):
        run_episode(Recorder(old, STUDY, time.time() + 60), questions[0], personas, graph, fmt, 1)
    original = {p: sha(p) for p in (STUDY / "chunks").glob("*.json")}
    result = parallel.prepare("test", report)
    plan = Path(result["plan"])
    assert result["preserved_chunks"] == 5
    for worker in range(4):
        monkeypatch.setenv("SLURM_JOB_ID", str(1000 + worker))
        assert parallel.collect_worker(plan, worker, .1) == 0
    assert parallel.status(plan)["complete_episodes"] == {"train": 8, "validation": 4, "assessment": 0}
    all_ids = [sid for src in [old] + sources for _, ids, _ in src.calls for sid in ids]
    assert len(all_ids) == len(set(all_ids))
    assert all(sha(p) == digest for p, digest in original.items())
    monkeypatch.setenv("SLURM_JOB_ID", "2000")
    assert parallel.collect_worker(plan, 0, .1) == 0
    assert sources[-1].calls == []
    assert (plan.parent / "worker_00/attempts/1000.json").exists()
    assert (plan.parent / "worker_00/attempts/2000.json").exists()


def test_changed_gpu_or_source_blocks_inference(tmp_path, monkeypatch):
    *_, report, sources = prepared_fixture(tmp_path, monkeypatch)
    plan = Path(parallel.prepare("test", report)["plan"])
    monkeypatch.setenv("SLURM_JOB_ID", "1000")
    monkeypatch.setattr(parallel.cuda, "runtime_identity", lambda: {"gpu": "different"})
    with pytest.raises(RuntimeError, match="changed since the replay"):
        parallel.collect_worker(plan, 0, .1)
    assert sources == []
    assert read(plan.parent / "worker_00/state.json")["status"] == "failed"
    parallel.CODE[0].write_text("changed source")
    with pytest.raises(ValueError, match="parallel input changed"):
        parallel.verify_plan(plan, 0)


def test_handoff_rejects_assessment_chunks(tmp_path, monkeypatch):
    cfg, questions, personas, graph, fmt, report, _ = prepared_fixture(tmp_path, monkeypatch)
    q = questions[-1]
    group = next(parallel.group_ids(q, graph, cfg))
    write(parallel.chunk_path(STUDY, group), {"group_id": group})
    with pytest.raises(ValueError, match="Assessment interaction chunks"):
        parallel.prepare("test", report)
    assert not (parallel.PARALLEL / "test").exists()
