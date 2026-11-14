"""Automatic generation must stay behind the numerical gate and assessment lock."""
from pathlib import Path
from types import SimpleNamespace
import pytest

from scripts import hpc3_auto_collect as auto
from scripts.hpc3_prepare import read, write
from scripts.hpc3_prepare import sha
from scripts.hpc3_promote_handoff import promote


def fixture(tmp_path, monkeypatch, passed=True):
    monkeypatch.chdir(tmp_path)
    report = Path("hpc3/report.json")
    write(report, {"numerical_gate_passed": passed})
    return report


def test_failed_replay_never_starts_collection(tmp_path, monkeypatch):
    report = fixture(tmp_path, monkeypatch, False)
    monkeypatch.setattr(auto.subprocess, "run", lambda *a, **k: pytest.fail("Must not start a collector"))
    with pytest.raises(RuntimeError, match="passing"):
        auto.continue_collection(report, 1)


def test_only_train_then_validation_share_one_time_budget(tmp_path, monkeypatch):
    report = fixture(tmp_path, monkeypatch)
    commands = []
    def run(command, **kwargs):
        commands.append((command, kwargs["timeout"]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(auto.subprocess, "run", run)
    assert auto.continue_collection(report, 1) == 0
    assert [c[c.index("--split") + 1] for c, _ in commands] == ["train", "validation"]
    assert commands[1][1] <= commands[0][1] <= 3600
    state = read("hpc3/automatic_collection_state.json")
    assert state["status"] == "complete" and not state["assessment_included"]


def test_failure_stops_before_starting_validation(tmp_path, monkeypatch):
    report = fixture(tmp_path, monkeypatch)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=75)
    monkeypatch.setattr(auto.subprocess, "run", run)
    assert auto.continue_collection(report, 1) == 75
    assert len(commands) == 1
    state = read("hpc3/automatic_collection_state.json")
    assert state["status"] == "time_limit" and state["completed_splits"] == []


def make_promotion_fixture(tmp_path):
    source, target = tmp_path / "staging", tmp_path / "benchmark"
    for root in [source, target]:
        (root / "hpc3").mkdir(parents=True)
        (root / "fixed.txt").write_text("frozen input")
    common = {"original_repository": "/original/repo", "source_protocol_sha256": "protocol",
              "files": {"fixed.txt": {"sha256": sha(source / "fixed.txt"), "bytes": 12}}}
    write(target / "hpc3/manifest.json", {**common, "mode": "benchmark", "production_export": False})
    (source / "chunk.json").write_text("saved response")
    write(source / "hpc3/manifest.json", {**common, "mode": "handoff", "production_export": True,
        "files": {**common["files"], "chunk.json": {"sha256": sha(source / "chunk.json"), "bytes": 14}}})
    return source, target


def test_handoff_adds_saved_data_without_modifying_benchmark(tmp_path):
    source, target = make_promotion_fixture(tmp_path)
    old = sha(target / "fixed.txt")
    result = promote(source, target)
    assert result["new_files"] == 1 and sha(target / "fixed.txt") == old
    assert (target / "chunk.json").read_text() == "saved response"
    assert read(target / "hpc3/manifest.json")["production_export"]
    assert promote(source, target)["new_files"] == 0


def test_conflicting_handoff_changes_nothing(tmp_path):
    source, target = make_promotion_fixture(tmp_path)
    old = sha(target / "hpc3/manifest.json")
    (target / "chunk.json").write_text("different result")
    with pytest.raises(ValueError, match="conflicts"):
        promote(source, target)
    assert sha(target / "hpc3/manifest.json") == old
    assert (target / "chunk.json").read_text() == "different result"
