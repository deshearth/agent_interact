"""Disjoint HPC3 workers around the unchanged, frozen conversation generator.

Only scheduling and output ownership change. Each worker has a private recorder
root and publishes immutable chunks/episodes into the original study directory.
All workers share exclusion locks against the old serial collector; each also
holds its own exclusive lock. Preparing a different plan requires all writers
to be stopped. Assessment questions cannot enter this launcher.
"""
from argparse import ArgumentParser
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import hpc3_breadth as cuda
from scripts.hpc3_prepare import STUDY, read, relative_path, sha, write

PARALLEL = Path("hpc3/parallel_runs")
ACTIVE = STUDY / "parallel_active.json"
CODE = [Path("scripts/hpc3_parallel_collect.py"), Path("hpc3/parallel_collect.sbatch")]


def now():
    return datetime.now(timezone.utc).isoformat()


def episode_name(q):
    return f"{q['qid']}__{q['graph_id']}__r00"


def assignments(selection, splits, workers):
    if not 1 <= workers <= 24:
        raise ValueError("Worker count must be between 1 and 24")
    questions = selection["questions"]
    if len({q["qid"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    if Counter(q["split"] for q in questions) != Counter(splits):
        raise ValueError("Selection does not match the frozen split counts")
    output = [[] for _ in range(workers)]
    for split in ["train", "validation"]:
        for i, q in enumerate(q for q in questions if q["split"] == split):
            output[i % workers].append({k: q[k] for k in ["qid", "graph_id", "split"]})
    return output


@contextmanager
def collection_locks(shared=False):
    """Shared by disjoint workers, exclusive for handoff and serial writers."""
    paths = [Path("hpc3/.automatic_collection.lock")]
    paths += [STUDY / name for name in [".background.lock", ".pipeline.lock", ".source.lock"]]
    with ExitStack() as stack:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Linux NFS emulates flock with byte-range locks: a shared lock
            # requires a readable descriptor (write-only append fails EBADF).
            stream = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("An incompatible collector is still active") from exc
        yield


def publish_immutable(source, target):
    """Atomic no-overwrite publication on the shared filesystem, even on races."""
    source, target = Path(source), Path(target)
    if source.is_symlink() or not source.is_file() or target.is_symlink():
        raise ValueError("Expected immutable regular files")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return True
    except FileExistsError:
        if sha(source) != sha(target):
            raise ValueError(f"Conflicting saved output: {target}")
        return False


def group_ids(q, graph, cfg):
    """Enumerate existing cache names, including parse retries, without sampling."""
    name = episode_name(q)
    for step in range(cfg["source"]["update_rounds"] + 1):
        if step:
            for start in range(0, sum(any(row) for row in graph), 4):
                yield f"{name}|round{step}|messages{start}"
        for start in range(0, cfg["source"]["n_agents"], 4):
            group = f"{name}|round{step}|opinions{start}"
            yield group
            yield group + "|retry1"
            yield group + "|retry2"


def chunk_path(root, group):
    return Path(root) / "chunks" / (hashlib.sha256(group.encode()).hexdigest() + ".json")


def validate_episode(path, q, protocol_sha, cfg):
    episode = read(path)
    expected = {**q, "name": episode_name(q), "protocol_sha256": protocol_sha,
                "n": cfg["source"]["n_agents"], "k": cfg["source"]["update_votes_per_agent"],
                "steps": cfg["source"]["update_rounds"], "replica": 0}
    if any(episode["meta"].get(k) != v for k, v in expected.items()):
        raise ValueError(f"Episode identity mismatch: {path}")
    n, k, steps = expected["n"], expected["k"], expected["steps"]
    if (len(episode["spins_history"]) != steps + 1
            or any(len(row) != n for row in episode["spins_history"])
            or len(episode["spins_raw_history"]) != steps + 1
            or any(len(row) != n or any(len(v) != k for v in row) for row in episode["spins_raw_history"])
            or len(episode["messages_history"]) != steps):
        raise ValueError(f"Incomplete episode: {path}")


def check_replay_files(report):
    for path, digest in report["raw_results"].items():
        if sha(relative_path(path, ".")) != digest:
            raise ValueError(f"Replay output changed: {path}")


def prepare(run_id, report_path, workers=4):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Use a simple run ID")
    report_path = relative_path(report_path, ".")
    directory = PARALLEL / run_id
    if directory.exists():
        raise FileExistsError("Use a new plan ID; resume existing plans with the worker command")
    with collection_locks():
        manifest = cuda.verify_package()
        protocol = read(STUDY / "source_protocol.json")
        cfg, selection = protocol["config"], read(STUDY / "selection.json")
        report = read(report_path)
        cuda.check_collection_gate(manifest, report, report["runtime"], read("hpc3/fixtures.json"))
        check_replay_files(report)
        shards = assignments(selection, cfg["questions"]["splits"], workers)
        graphs = {g["id"]: g["J"] for g in read("data/raw/upstream/data/J_matrices.json")["graphs"]}
        protocol_sha = sha(STUDY / "source_protocol.json")
        existing_chunks = {p.name for p in (STUDY / "chunks").glob("*.json")}
        for q in selection["questions"]:
            if q["split"] == "assessment":
                if (STUDY / "episodes/assessment" / (episode_name(q) + ".json")).exists():
                    raise ValueError("This launcher requires untouched assessment conversations")
                if any(chunk_path(STUDY, g).name in existing_chunks for g in group_ids(q, graphs[q["graph_id"]], cfg)):
                    raise ValueError("Assessment interaction chunks already exist")
        saved = []
        for worker, shard in enumerate(shards):
            for q in shard:
                for group in group_ids(q, graphs[q["graph_id"]], cfg):
                    path = chunk_path(STUDY, group)
                    if path.name not in existing_chunks:
                        continue
                    chunk = read(path)
                    if chunk.get("group_id") != group or chunk.get("protocol_sha256") != protocol_sha:
                        raise ValueError(f"Saved chunk identity mismatch: {path}")
                    if "hpc3_execution" in chunk and chunk["hpc3_execution"] != execution_record(report_path, report):
                        raise ValueError(f"Saved CUDA execution mismatch: {path}")
                    saved.append({"worker": worker, "path": str(path.relative_to(STUDY)), "sha256": sha(path)})
                path = STUDY / "episodes" / q["split"] / (episode_name(q) + ".json")
                if path.exists():
                    validate_episode(path, q, protocol_sha, cfg)
                    saved.append({"worker": worker, "path": str(path.relative_to(STUDY)), "sha256": sha(path)})
        plan = {"schema": 1, "created_utc": now(), "run_id": run_id, "workers": workers,
                "assignments": shards, "assessment_included": False, "report_path": str(report_path),
                "benchmark_report_sha256": sha(report_path), "selection_sha256": sha(STUDY / "selection.json"),
                "source_protocol_sha256": protocol_sha, "package_manifest_sha256": sha("hpc3/manifest.json"),
                "code_sha256": {str(p): sha(p) for p in CODE}, "saved_files": saved,
                "policy": "Round-robin whole questions within each fixed development split; unchanged prompts, graphs and request RNG identities. No assessment generation or predictor training."}
        for worker in range(workers):
            root = directory / f"worker_{worker:02d}"
            publish_immutable(STUDY / "source_protocol.json", root / "source_protocol.json")
        for item in saved:
            publish_immutable(STUDY / item["path"], directory / f"worker_{item['worker']:02d}" / item["path"])
        write(directory / "plan.json", plan)
        write(ACTIVE, {"plan_path": str(directory / "plan.json"), "plan_sha256": sha(directory / "plan.json")})
        write(directory / "prepared.json", {"prepared_utc": now(), "plan_sha256": sha(directory / "plan.json"),
              "preserved_chunks": sum(x["path"].startswith("chunks/") for x in saved),
              "preserved_episodes": sum(x["path"].startswith("episodes/") for x in saved)})
        return {"plan": str(directory / "plan.json"), **read(directory / "prepared.json"),
                "worker_counts": [dict(Counter(q["split"] for q in shard)) for shard in shards]}


def execution_record(report_path, report):
    return {"backend": "MLX CUDA 12", "benchmark_report_sha256": sha(report_path),
            "runtime": report["runtime"], "batch_size": report["selected_batch_size"],
            "scientific_source_protocol_sha256": sha(STUDY / "source_protocol.json")}


def verify_plan(plan_path, worker):
    plan_path = relative_path(plan_path, ".")
    plan = read(plan_path)
    if not 0 <= worker < plan["workers"] or plan.get("assessment_included") is not False:
        raise ValueError("Invalid worker or assessment policy")
    if plan_path != PARALLEL / plan["run_id"] / "plan.json":
        raise ValueError("Unexpected plan location")
    if read(ACTIVE) != {"plan_path": str(plan_path), "plan_sha256": sha(plan_path)}:
        raise ValueError("Plan is not the active collection plan")
    if read(plan_path.parent / "prepared.json")["plan_sha256"] != sha(plan_path):
        raise ValueError("Plan handoff is incomplete")
    checks = {STUDY / "source_protocol.json": plan["source_protocol_sha256"],
              STUDY / "selection.json": plan["selection_sha256"],
              Path("hpc3/manifest.json"): plan["package_manifest_sha256"],
              relative_path(plan["report_path"], "."): plan["benchmark_report_sha256"]}
    checks.update({relative_path(p, "."): d for p, d in plan["code_sha256"].items()})
    for path, digest in checks.items():
        if sha(path) != digest:
            raise ValueError(f"Frozen parallel input changed: {path}")
    cfg = read(STUDY / "source_protocol.json")["config"]
    if assignments(read(STUDY / "selection.json"), cfg["questions"]["splits"], plan["workers"]) != plan["assignments"]:
        raise ValueError("Question ownership changed")
    root = plan_path.parent / f"worker_{worker:02d}"
    if sha(root / "source_protocol.json") != plan["source_protocol_sha256"]:
        raise ValueError("Worker protocol changed")
    for item in plan["saved_files"]:
        if item["worker"] == worker:
            rel = relative_path(item["path"], ".")
            if sha(root / rel) != item["sha256"] or sha(STUDY / rel) != item["sha256"]:
                raise ValueError(f"Preserved output changed: {rel}")
    return plan, root, cfg


def make_recorder(source, root, deadline, allowed, execution):
    from scripts.breadth_collect import Recorder

    class WorkerRecorder(Recorder):
        def request(self, group_id, requests, max_tokens=256):
            if group_id.split("|", 1)[0] not in allowed:
                raise ValueError("Request belongs to another worker or a reserved question")
            path = chunk_path(self.root, group_id)
            if path.exists():
                old = read(path)
                if "hpc3_execution" in old and old["hpc3_execution"] != execution:
                    raise ValueError("Refusing different CUDA execution settings on resume")
            rows = super().request(group_id, requests, max_tokens)
            publish_immutable(path, chunk_path(STUDY, group_id))
            return rows

    return WorkerRecorder(source, root, deadline)


def replay_worker(source, fixtures, output):
    checks = []
    for index, group in enumerate(fixtures["groups"]):
        reference = group["reference"]
        prompts, ids = cuda.requests_from_reference(reference)
        result = source.generate_many(prompts, ids, max_tokens=reference["max_tokens"])
        repeat = source.generate_many(prompts, ids, max_tokens=reference["max_tokens"])
        check = cuda.compare_replay(reference, result, repeat, fixtures["gate"]["max_first_token_probability_difference"])
        write(output / f"group{index}.json", {"result": result, "repeat": repeat, "check": check})
        checks.append({"category": group["category"], **check})
        if not check["passed"]:
            raise RuntimeError("Worker numerical replay failed; no collection permitted")
    write(output / "passed.json", {"checks": checks, "passed_utc": now()})


def collect_worker(plan_path, worker, hours=47.5):
    if not 0 < hours <= 71:
        raise ValueError("Invalid collection budget")
    plan_path = relative_path(plan_path, ".")
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise RuntimeError("Run workers inside Slurm allocations")
    with collection_locks(shared=True), (plan_path.parent / f"worker_{worker:02d}" / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, root, cfg = verify_plan(plan_path, worker)
        state = {"job_id": job, "worker": worker, "plan_sha256": sha(plan_path), "started_utc": now(),
                 "hours": hours, "assessment_included": False, "completed_episodes": [], "status": "verifying"}

        def save(**values):
            state.update(values, updated_utc=now())
            write(root / "state.json", state)
            write(root / "attempts" / f"{job}.json", state)

        def interrupted(signum, frame):
            raise InterruptedError(f"Worker received signal {signum}; atomic chunks remain resumable")

        signal.signal(signal.SIGTERM, interrupted)
        deadline = time.time() + hours * 3600
        save()
        try:
            # Each worker has its own compilation cache, while model weights and
            # the validated software environment remain shared and unchanged.
            os.environ["MLX_PTX_CACHE_DIR"] = str((root / "cache/mlx").resolve())
            (root / "cache/mlx").mkdir(parents=True, exist_ok=True)
            manifest = cuda.verify_package()
            report, fixtures = read(plan["report_path"]), read("hpc3/fixtures.json")
            identity = cuda.runtime_identity()
            batch = cuda.check_collection_gate(manifest, report, identity, fixtures)
            check_replay_files(report)
            execution = execution_record(plan["report_path"], report)
            save(status="loading_model")
            source = cuda.load_source(batch, execution)
            save(status="checking_replay")
            replay_worker(source, fixtures, root / "replay" / job)
            from scripts.breadth_collect import TimeLimit, run_episode
            from scripts.local_conversation_pilot import formatter
            from scripts.question_breadth import UPSTREAM
            personas = [p["persona"] for p in read(UPSTREAM / "obj/personas.json")]
            graphs = {g["id"]: g["J"] for g in read(UPSTREAM / "J_matrices.json")["graphs"]}
            questions = {q["qid"]: q for q in read(STUDY / "selection.json")["questions"]}
            owned = plan["assignments"][worker]
            recorder = make_recorder(source, root, deadline, {episode_name(q) for q in owned}, execution)
            fmt = formatter()
            try:
                for item in owned:
                    save(status="generating", active_episode=episode_name(item), active_split=item["split"])
                    run_episode(recorder, questions[item["qid"]], personas, graphs[item["graph_id"]], fmt, cfg["source"]["update_rounds"])
                    path = Path("episodes") / item["split"] / (episode_name(item) + ".json")
                    validate_episode(root / path, item, plan["source_protocol_sha256"], cfg)
                    publish_immutable(root / path, STUDY / path)
                    state["completed_episodes"].append(episode_name(item))
                    save(new_samples_this_attempt=recorder.new_samples)
            except TimeLimit:
                save(status="time_limit", new_samples_this_attempt=recorder.new_samples)
                return 75
            save(status="complete", active_episode=None, active_split=None, completed_utc=now())
            return 0
        except InterruptedError as exc:
            save(status="interrupted", error=str(exc))
            return 75
        except Exception as exc:
            save(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise


def status(plan_path):
    plan_path = relative_path(plan_path, ".")
    plan = read(plan_path)
    workers = []
    for i, shard in enumerate(plan["assignments"]):
        root = plan_path.parent / f"worker_{i:02d}"
        complete = [q for q in shard if (STUDY / "episodes" / q["split"] / (episode_name(q) + ".json")).is_file()]
        workers.append({"worker": i, "assigned": dict(Counter(q["split"] for q in shard)),
                        "complete": dict(Counter(q["split"] for q in complete)),
                        "state": read(root / "state.json") if (root / "state.json").exists() else None,
                        "progress": read(root / "source_progress.json") if (root / "source_progress.json").exists() else None})
    return {"checked_utc": now(), "run_id": plan["run_id"], "workers": workers,
            "complete_episodes": {split: len(list((STUDY / "episodes" / split).glob("*.json")))
                                  for split in ["train", "validation", "assessment"]},
            "note": "Worker state is the last saved checkpoint; use squeue/sacct to check live job status."}


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--run-id", required=True)
    prep.add_argument("--benchmark-report", type=Path, required=True)
    prep.add_argument("--workers", type=int, default=4)
    worker = sub.add_parser("worker")
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument("--worker", type=int, required=True)
    worker.add_argument("--hours", type=float, default=47.5)
    check = sub.add_parser("status")
    check.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.run_id, args.benchmark_report, args.workers), indent=2))
    elif args.command == "status":
        print(json.dumps(status(args.plan), indent=2))
    else:
        raise SystemExit(collect_worker(args.plan, args.worker, args.hours))


if __name__ == "__main__":
    main()
