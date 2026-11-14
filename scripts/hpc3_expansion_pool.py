"""Dynamic free-GPU pool using the unchanged scientific generator and RNG IDs."""
from argparse import ArgumentParser
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import hpc3_recovery as control
from scripts import hpc3_breadth_expand as original
from scripts import hpc3_parallel_collect as parallel
from scripts.hpc3_prepare import read, sha, write


class Stopped(Exception):
    pass


def startup_oom(status, error):
    message = str(error).lower()
    return status == "loading_model" and "cudamalloc" in message and "out of memory" in message


def record_load_attempt(slot, job, error=None):
    path = control.ROOT / "load_retries" / f"slot_{slot:02d}.json"
    with control.lock(path.with_suffix(".lock"), nonblocking=False):
        previous = read(path) if path.exists() else {"consecutive_failures": 0, "attempts": []}
        previous["consecutive_failures"] = previous["consecutive_failures"] + 1 if error else 0
        previous["attempts"].append({"job_id": job, "utc": control.now(), "error": error})
        write(path, previous)
        return previous["consecutive_failures"]


def inherit_file(relative, destination, origins):
    """Adopt saved immutable files; conflicting copies must stop the run."""
    target = Path(destination) / relative
    for origin in origins:
        source = Path(origin) / relative
        if source.exists():
            parallel.publish_immutable(source, target)
    return target


def make_recorder(source, root, deadline, owned, phase, execution, legacy_root):
    base = type(original.make_recorder(source, root, deadline, owned, phase, execution))
    allowed = {q["qid"] for q in owned}

    class RestoringRecorder(base):
        def request(self, group_id, requests, max_tokens=256):
            if control.is_stopped():
                raise Stopped("Recovery controller or user stopped this workflow")
            if not requests or any(r["metadata"].get("qid") not in allowed for r in requests):
                raise ValueError("Request belongs to another unit")
            normalized = ("expand_construct|" + requests[0]["metadata"]["qid"]
                          if group_id.startswith("construct|") else group_id)
            relative = parallel.chunk_path(Path("."), normalized)
            inherit_file(relative, self.root, [original.ROOT, legacy_root])
            # The frozen recorder checks protocol/request hashes, sample IDs,
            # runtime identity and max_tokens before reusing any cached group.
            return super().request(group_id, requests, max_tokens)

    return RestoringRecorder(source, root, deadline)


@contextmanager
def claim_next(units):
    for unit in units:
        if control.unit_complete(unit):
            continue
        stack = ExitStack()
        try:
            stack.enter_context(control.lock(control.ROOT / "units" / unit["id"] / ".unit.lock"))
            # A legacy worker holds this exclusively; several independent new
            # units may hold it shared, preventing old/new writers from racing.
            stack.enter_context(control.lock(Path(unit["legacy_root"]) / ".worker.lock", shared=True))
        except BlockingIOError:
            stack.close()
            continue
        with stack:
            if control.unit_complete(unit):
                continue
            yield unit
            return
    yield None


def available_units():
    path = control.ROOT / "collection_units.json"
    if path.exists() and (control.ROOT / "collection_units_frozen.json").exists():
        frozen = read(control.ROOT / "collection_units_frozen.json")
        if sha(path) != frozen["sha256"] or sha(original.ROOT / "selection.json") != frozen["selection_sha256"]:
            raise ValueError("Collection mapping changed")
        return read(path)
    return read(control.ROOT / "screen_units.json")


def run_unit(unit, source, deadline, execution, bank, selected, personas, graphs, fmt):
    from scripts.breadth_collect import construct, screen, run_episode
    root = control.ROOT / "units" / unit["id"]
    parallel.publish_immutable(original.ROOT / "source_protocol.json", root / "source_protocol.json")
    phase = unit["phase"]
    lookup = bank if phase == "screen" else selected
    owned = [lookup[qid] for qid in unit["qids"]]
    recorder = make_recorder(source, root, deadline, owned, phase, execution, unit["legacy_root"])
    origins = [original.ROOT, Path(unit["legacy_root"])]
    if phase == "screen":
        for q in owned:
            for folder in ["constructed", "screen"]:
                inherit_file(Path(folder) / (q["qid"] + ".json"), root, origins)
        # The original groups of four, including the final two, are unchanged.
        construct(recorder, owned)
        for q in owned:
            relative = Path("constructed") / (q["qid"] + ".json")
            parallel.publish_immutable(root / relative, original.ROOT / relative)
        for q in owned:
            screen(recorder, [q], personas, fmt)
            relative = Path("screen") / (q["qid"] + ".json")
            record = read(root / relative)
            if (record["qid"] != q["qid"] or record["protocol_sha256"] != recorder.protocol_sha256
                    or record["question_sha256"] != sha(root / "constructed" / (q["qid"] + ".json"))
                    or record["status"] not in {"complete", "invalid"}):
                raise ValueError("Screen identity or completion mismatch")
            parallel.publish_immutable(root / relative, original.ROOT / relative)
    else:
        if len(owned) != 1 or owned[0]["split"] not in {"train", "validation"}:
            raise ValueError("Only one development conversation per collection unit")
        q = owned[0]
        relative = Path("episodes") / q["split"] / (parallel.episode_name(q) + ".json")
        inherit_file(relative, root, origins)
        run_episode(recorder, q, personas, graphs[q["graph_id"]], fmt, 8)
        from scripts.breadth_expand import episode_summary
        episode_summary(q, root / relative, graphs)
        parallel.publish_immutable(root / relative, original.ROOT / relative)
    write(root / "complete.json", {"completed_utc": control.now(), "unit": unit["id"],
          "job_id": os.environ["SLURM_JOB_ID"], "new_samples": recorder.new_samples,
          "source_protocol_sha256": recorder.protocol_sha256})
    return recorder.new_samples


def worker(slot, hours):
    plan = control.verify_plan()
    if not 0 <= slot < plan["maximum_gpu_jobs"] or not 0 < hours <= plan["worker_hours"]:
        raise ValueError("Invalid slot or time budget")
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise RuntimeError("Source generation requires a Slurm GPU allocation")
    protocol, deployment = original.verify_deployment()
    units = available_units()
    if control.is_stopped() or all(control.unit_complete(unit) for unit in units):
        return 0
    root = control.ROOT / "pool_attempts" / job
    state = {"job_id": job, "slot": slot, "started_utc": control.now(),
             "completed_units": [], "new_samples": 0, "assessment_included": False}

    def save(**values):
        state.update(values, updated_utc=control.now())
        write(root / "state.json", state)

    def interrupted(signum, frame):
        raise InterruptedError(f"Signal {signum}; saved groups retained")

    signal.signal(signal.SIGTERM, interrupted)
    deadline = time.time() + hours * 3600
    save(status="verifying")
    from scripts.breadth_collect import TimeLimit
    try:
        from scripts import hpc3_breadth as cuda
        from scripts.local_conversation_pilot import formatter
        cache = root / "cache/mlx"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MLX_PTX_CACHE_DIR"] = str(cache.resolve())
        manifest = cuda.verify_package()
        report = read(deployment["benchmark_report"])
        fixtures = read("hpc3/fixtures.json")
        identity = cuda.runtime_identity()
        batch = cuda.check_collection_gate(manifest, report, identity, fixtures)
        parallel.check_replay_files(report)
        execution = parallel.execution_record(deployment["benchmark_report"], report)
        save(status="loading_model")
        source = cuda.load_source(batch, execution)
        record_load_attempt(slot, job)
        save(status="checking_replay")
        parallel.replay_worker(source, fixtures, root / "replay")
        bank = {q["qid"]: q for q in read(original.ROOT / "candidate_bank.json")}
        selected = {}
        if units[0]["phase"] == "collect":
            from scripts.breadth_expand import verify_selection
            selected = {q["qid"]: q for q in verify_selection()["new_questions"]}
            if set(selected) & set(protocol["reserved_assessment_qids"]):
                raise ValueError("Reserved questions entered collection")
        personas = [p["persona"] for p in read("data/raw/upstream/data/obj/personas.json")]
        graphs = {g["id"]: g["J"] for g in read("data/raw/upstream/data/J_matrices.json")["graphs"]}
        fmt = formatter()
        while time.time() < deadline and not control.is_stopped():
            with claim_next(units) as unit:
                if unit is None:
                    save(status="drained", active_unit=None)
                    return 0
                save(status="generating", active_unit=unit["id"])
                samples = run_unit(unit, source, deadline, execution, bank, selected, personas, graphs, fmt)
                state["completed_units"].append(unit["id"])
                save(new_samples=state["new_samples"] + samples)
        if control.is_stopped():
            save(status="stopped")
            return 0
        save(status="time_limit")
        return 75
    except (InterruptedError, TimeLimit) as exc:
        save(status="interrupted", error=str(exc))
        return 75
    except Stopped as exc:
        save(status="stopped", error=str(exc))
        return 0
    except Exception as exc:
        if startup_oom(state["status"], exc):
            failures = record_load_attempt(slot, job, str(exc))
            if failures <= plan["startup_cuda_oom_retries_per_slot"]:
                save(status="retryable_startup_failure", error=str(exc), consecutive_load_failures=failures)
                return 75
        save(status="failed", error=f"{type(exc).__name__}: {exc}")
        control.halt("expansion", {"job_id": job, "unit": state.get("active_unit"), "error": state["error"]})
        raise


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--hours", type=float, default=7.5)
    args = parser.parse_args()
    raise SystemExit(worker(args.slot, args.hours))
