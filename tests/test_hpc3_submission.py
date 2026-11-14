"""HPC transfer and experiment-boundary tests; no CUDA or source inference."""
import copy
import fcntl
import json
from pathlib import Path
import tarfile
import time

import pytest

from scripts import hpc3_breadth as remote
from scripts import hpc3_prepare as export


def reference():
    return {"prompts": [{"prompt": "argument", "prompt_sha256": "p", "token_ids": [1, 2]}],
            "responses": [{"sample_id": "s", "generated_token_ids": [4, 5, 6],
                           "finish_reason": "stop", "first_literal_A_probability": .2,
                           "first_literal_B_probability": .3, "other_first_token_probability": .5}]}


def test_portable_paths_reject_escape():
    assert export.relative_path("/old/repo/scripts/a.py", "/old/repo") == Path("scripts/a.py")
    for path in ["../secret", "/outside/secret", "scripts/../../secret"]:
        with pytest.raises(ValueError):
            export.relative_path(path, "/old/repo")


def test_replay_checks_argument_continuations_tokenization_and_repeatability():
    old = reference()
    assert remote.compare_replay(old, old, old, .001)["passed"]
    changed = copy.deepcopy(old)
    changed["responses"][0]["generated_token_ids"][1] = 7
    assert not remote.compare_replay(old, changed, changed, .001)["passed"]
    assert not remote.compare_replay(old, old, changed, .001)["passed"]
    changed = copy.deepcopy(old)
    changed["prompts"][0]["token_ids"] = [1, 3]
    assert not remote.compare_replay(old, changed, changed, .001)["passed"]


def test_replay_rejects_large_probability_drift_and_nan():
    old = reference()
    changed = copy.deepcopy(old)
    changed["responses"][0]["first_literal_A_probability"] += .01
    assert not remote.compare_replay(old, changed, changed, .001)["passed"]
    changed["responses"][0]["first_literal_A_probability"] = float("nan")
    with pytest.raises(ValueError, match="probability"):
        remote.compare_replay(old, changed, changed, .001)


def test_replay_preserves_repeated_prompt_samples():
    old = reference()
    old["responses"].append({**old["responses"][0], "sample_id": "s2"})
    prompts, ids = remote.requests_from_reference(old)
    assert prompts == ["argument", "argument"] and ids == ["s", "s2"]
    changed = copy.deepcopy(old)
    changed["responses"].reverse()
    with pytest.raises(ValueError, match="identity"):
        remote.compare_replay(old, changed, changed, .001)


def test_production_requires_handoff_complete_replay_and_matching_runtime():
    manifest = {"mode": "handoff", "production_export": True}
    report = {"numerical_gate_passed": True, "runtime": {"driver": "one"},
              "selected_batch_size": 4, "candidates": [{"batch_size": 4, "passed": True,
                                                       "checks": [{"passed": True}]}]}
    fixtures = {"full_coverage": True}
    assert remote.check_collection_gate(manifest, report, report["runtime"], fixtures) == 4
    for bad in [{"mode": "benchmark", "production_export": False}, {"mode": "handoff", "production_export": False}]:
        with pytest.raises(RuntimeError, match="handoff"):
            remote.check_collection_gate(bad, report, report["runtime"], fixtures)
    with pytest.raises(RuntimeError, match="complete"):
        remote.check_collection_gate(manifest, report, report["runtime"], {"full_coverage": False})
    with pytest.raises(RuntimeError, match="changed"):
        remote.check_collection_gate(manifest, report, {"driver": "two"}, fixtures)
    report["candidates"][0]["checks"][0]["passed"] = False
    with pytest.raises(ValueError):
        remote.check_collection_gate(manifest, report, report["runtime"], fixtures)


def test_handoff_refuses_a_live_collector(tmp_path):
    with (tmp_path / ".source.lock").open("a") as busy:
        fcntl.flock(busy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="still active"):
            with export.idle_collection(tmp_path):
                pytest.fail("Live collector handoff must be rejected")
    with export.idle_collection(tmp_path):
        pass


def test_assessment_requires_frozen_models_and_checks_portable_hashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = export.STUDY
    root.mkdir(parents=True)
    manifest = {"original_repository": "/old/repo"}
    remote.authorize_split("train", manifest)
    with pytest.raises(RuntimeError, match="locked"):
        remote.authorize_split("assessment", manifest)
    model = tmp_path / "model.bin"
    model.write_bytes(b"frozen model")
    export.write(root / "models_frozen.json", {"artifacts": {"/old/repo/model.bin": export.sha(model)}})
    remote.authorize_split("assessment", manifest)
    model.write_bytes(b"different model")
    with pytest.raises(ValueError, match="changed"):
        remote.authorize_split("assessment", manifest)


def test_export_allowlist_and_portable_protocol_integrity(tmp_path, monkeypatch):
    repo = tmp_path / "original"
    for name in ["scripts/hpc3_prepare.py", "scripts/hpc3_breadth.py", "scripts/breadth_collect.py",
                 "hpc3/README.md", "runs/question_breadth_v1/source_benchmark.json",
                 ".env", ".ssh/private_key", "hpc3/.env", "hpc3/credentials.json", "unrelated_data.txt"]:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture content")
    protected = repo / "scripts/breadth_collect.py"
    export.write(repo / export.STUDY / "source_protocol.json", {"source_files": {str(protected): export.sha(protected)}})
    monkeypatch.setattr(export, "fixture_groups", lambda _: {"groups": [], "full_coverage": False})
    result = export.export_package(tmp_path / "transfer", repo=repo)
    remote.verify_package(Path(result["directory"]))
    with tarfile.open(result["archive"]) as tar:
        names = tar.getnames()
    assert not any(".env" in n or ".ssh" in n or "credentials" in n or "unrelated_data" in n for n in names)
    (Path(result["directory"]) / "scripts/breadth_collect.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        remote.verify_package(Path(result["directory"]))


def test_saved_request_groups_resume_without_another_model_call(tmp_path):
    from scripts.breadth_collect import Recorder
    class Source:
        calls = 0
        def generate_many(self, prompts, identities, **kwargs):
            self.calls += 1
            return {"responses": [{"sample_id": sid, "raw_response": "A", "parsed_sign": 1} for sid in identities]}
    export.write(tmp_path / "source_protocol.json", {"fixture": True})
    source = Source()
    requests = [{"sample_id": "sample0", "prompt": "fixture", "metadata": {"agent": 0}}]
    first = Recorder(source, tmp_path, time.time() + 1).request("group", requests)
    second = Recorder(source, tmp_path, time.time() - 1).request("group", requests)
    assert first == second and source.calls == 1
    requests[0]["prompt"] = "changed"
    with pytest.raises(AssertionError):
        Recorder(source, tmp_path, time.time() + 1).request("group", requests)
