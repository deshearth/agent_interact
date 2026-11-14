"""Disjoint, resumable source-generation units for the 3,200/800 expansion."""
from argparse import ArgumentParser
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import hpc3_scale_controller as control
from scripts import breadth_scale as study
from scripts import hpc3_breadth as cuda
from scripts import hpc3_parallel_collect as parallel
from scripts.hpc3_prepare import read, sha, write


class Stopped(Exception):
    pass


def prepare_deployment():
    protocol = study.verify()
    manifest = cuda.verify_package()
    report_path = protocol["config"]["collection"]["benchmark_report"]
    report, fixtures = read(report_path), read("hpc3/fixtures.json")
    cuda.check_collection_gate(manifest, report, report["runtime"], fixtures)
    parallel.check_replay_files(report)
    for item in read(study.ROOT / "retained_episodes.json"):
        source = Path(item["source_path"])
        assert sha(source) == item["sha256"]
        parallel.publish_immutable(source, study.ROOT / "episodes" / item["split"] / source.name)
    record = {"protocol_sha256": sha(study.ROOT / "protocol.json"), "retained_sha256": sha(study.ROOT / "retained_episodes.json"),
              "base_report": report_path, "base_report_sha256": sha(report_path), "assessment_included": False}
    path = study.ROOT / "deployment.json"
    if path.exists():
        assert read(path) == record
    else:
        write(path, record)
    return record


@contextmanager
def claim_next(units, execution):
    key = control.runtime_key(execution)
    for unit in units:
        if control.unit_complete(unit):
            continue
        root = control.ROOT / "units" / unit["id"]
        try:
            with control.lock(root / ".lock"):
                if control.unit_complete(unit):
                    continue
                path = root / "runtime_binding.json"
                if path.exists():
                    binding = read(path)
                    if binding["execution_key"] != key:
                        continue
                    assert binding["execution"] == execution
                else:
                    write(path, {"created_utc": control.now(), "execution_key": key, "execution": execution,
                                 "unit": unit, "protocol_sha256": sha(study.ROOT / "protocol.json")})
                parallel.publish_immutable(study.ROOT / "source_protocol.json", root / "source_protocol.json")
                yield unit
                return
        except BlockingIOError:
            continue
    yield None


def make_recorder(source, root, deadline, owned, phase, execution):
    from scripts.breadth_collect import Recorder
    allowed = {q["qid"] for q in owned}

    class UnitRecorder(Recorder):
        def request(self, group_id, requests, max_tokens=256):
            if control.stopped():
                raise Stopped("Collection was stopped")
            if not requests or any(r["metadata"].get("qid") not in allowed for r in requests):
                raise ValueError("Request outside this unit")
            if phase == "screen" and not group_id.startswith(("construct|", "screen|")):
                raise ValueError("Interaction request during screening")
            if phase == "collect" and any(r["metadata"].get("kind") not in ("message", "opinion")
                                         or r["metadata"].get("phase") == "screen" for r in requests):
                raise ValueError("Non-interaction request during collection")
            if group_id.startswith("construct|"):
                group_id = "scale_construct|" + requests[0]["metadata"]["qid"]
            path = parallel.chunk_path(self.root, group_id)
            if path.exists() and read(path).get("hpc3_execution") != execution:
                raise ValueError("Execution settings changed on resume")
            rows = super().request(group_id, requests, max_tokens)
            parallel.publish_immutable(path, parallel.chunk_path(study.ROOT, group_id))
            return rows
    return UnitRecorder(source, root, deadline)


def run_unit(unit, source, deadline, execution, bank, selected, personas, graphs, fmt):
    from scripts.breadth_collect import construct, screen, run_episode
    root = control.ROOT / "units" / unit["id"]
    owned = [bank[q] if unit["phase"] == "screen" else selected[q] for q in unit["qids"]]
    recorder = make_recorder(source, root, deadline, owned, unit["phase"], execution)
    if unit["phase"] == "screen":
        construct(recorder, owned)
        for q in owned:
            relative = Path("constructed") / (q["qid"] + ".json")
            parallel.publish_immutable(root / relative, study.ROOT / relative)
        screen(recorder, owned, personas, fmt)
        for q in owned:
            relative = Path("screen") / (q["qid"] + ".json")
            result = read(root / relative)
            assert result["qid"] == q["qid"] and result["status"] in ("complete", "invalid")
            assert result["protocol_sha256"] == recorder.protocol_sha256
            parallel.publish_immutable(root / relative, study.ROOT / relative)
    else:
        assert len(owned) == 1 and owned[0]["split"] in ("train", "validation")
        q = owned[0]
        relative = Path("episodes") / q["split"] / (parallel.episode_name(q) + ".json")
        run_episode(recorder, q, personas, graphs[q["graph_id"]], fmt, 8)
        study.episode_summary(q, root / relative, graphs, recorder.protocol_sha256)
        parallel.publish_immutable(root / relative, study.ROOT / relative)
    write(root / "complete.json", {"completed_utc": control.now(), "unit": unit["id"],
        "job_id": os.environ["SLURM_JOB_ID"], "source_protocol_sha256": recorder.protocol_sha256})
    return recorder.new_samples


def record_load(slot, job, error=None):
    path = control.ROOT / "load_retries" / f"slot_{slot:02d}.json"
    with control.lock(path.with_suffix(".lock"), nonblocking=False):
        data = read(path) if path.exists() else {"consecutive_failures": 0, "attempts": []}
        data["consecutive_failures"] = data["consecutive_failures"] + 1 if error else 0
        data["attempts"].append({"job_id": job, "utc": control.now(), "error": error})
        write(path, data)
        return data["consecutive_failures"]


def worker(slot):
    plan = control.verify_plan()
    study.verify()
    deployment = read(study.ROOT / "deployment.json")
    assert deployment["protocol_sha256"] == sha(study.ROOT / "protocol.json")
    if not 0 <= slot < sum(plan["partition_limits"].values()):
        raise ValueError("Invalid worker slot")
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise RuntimeError("Use a Slurm GPU allocation")
    units = control.available_units()
    if control.stopped() or all(control.unit_complete(u) for u in units):
        return 0
    root = control.ROOT / "attempts" / job
    state = {"job_id": job, "slot": slot, "started_utc": control.now(), "completed_units": [], "new_samples": 0}

    def save(**values):
        state.update(values, updated_utc=control.now())
        write(root / "state.json", state)

    def interrupted(signum, frame):
        raise InterruptedError(f"Signal {signum}; completed groups are saved")

    signal.signal(signal.SIGTERM, interrupted)
    deadline = time.time() + plan["worker_hours"] * 3600
    save(status="verifying")
    from scripts.breadth_collect import TimeLimit
    try:
        cache = root / "cache/mlx"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MLX_PTX_CACHE_DIR"] = str(cache.resolve())
        manifest, identity = cuda.verify_package(), cuda.runtime_identity()
        entries = [r for r in read(control.ROOT / "runtime_registry.json")["entries"] if r["runtime"] == identity]
        if len(entries) != 1:
            raise RuntimeError("This GPU/software environment lacks a certified replay")
        entry = entries[0]
        assert sha(entry["report"]) == entry["report_sha256"]
        report, fixtures = read(entry["report"]), read("hpc3/fixtures.json")
        batch = cuda.check_collection_gate(manifest, report, identity, fixtures)
        parallel.check_replay_files(report)
        execution = parallel.execution_record(entry["report"], report)
        save(status="loading_model", runtime_key=entry["runtime_key"], gpu=entry["gpu"])
        source = cuda.load_source(batch, execution)
        record_load(slot, job)
        save(status="checking_replay")
        parallel.replay_worker(source, fixtures, root / "replay")
        bank = {q["qid"]: q for q in read(study.ROOT / "candidate_bank.json")}
        selected = {}
        if units[0]["phase"] == "collect":
            selected = {q["qid"]: q for q in study.verify_selection()["new_questions"]}
            assert not set(selected) & set(plan["reserved_assessment_qids"])
        from scripts.local_conversation_pilot import formatter
        personas = [p["persona"] for p in read("data/raw/upstream/data/obj/personas.json")]
        graphs = {g["id"]: g["J"] for g in read("data/raw/upstream/data/J_matrices.json")["graphs"]}
        fmt = formatter()
        while time.time() < deadline and not control.stopped():
            if shutil.disk_usage(study.ROOT).free < 2 * 1024**3:
                raise RuntimeError("Less than 2 GiB storage remains; stop before incomplete writes")
            with claim_next(units, execution) as unit:
                if unit is None:
                    save(status="drained", active_unit=None)
                    return 0
                save(status="generating", active_unit=unit["id"], phase=unit["phase"])
                samples = run_unit(unit, source, deadline, execution, bank, selected, personas, graphs, fmt)
                state["completed_units"].append(unit["id"])
                save(new_samples=state["new_samples"] + samples)
        save(status="stopped" if control.stopped() else "time_limit")
        return 0 if control.stopped() else 75
    except (InterruptedError, TimeLimit) as exc:
        save(status="interrupted", error=str(exc))
        return 75
    except Stopped as exc:
        save(status="stopped", error=str(exc))
        return 0
    except Exception as exc:
        msg = str(exc).lower()
        if state["status"] == "loading_model" and "cudamalloc" in msg and "out of memory" in msg:
            if record_load(slot, job, str(exc)) <= plan["startup_oom_retries"]:
                save(status="retryable_startup_failure", error=str(exc))
                return 75
        save(status="failed", error=f"{type(exc).__name__}: {exc}")
        control.halt({"job_id": job, "unit": state.get("active_unit"), "error": state["error"]})
        raise


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "worker"])
    parser.add_argument("--slot", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        print(json.dumps(prepare_deployment(), indent=2))
    else:
        raise SystemExit(worker(args.slot))
