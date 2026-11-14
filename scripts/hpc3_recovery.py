"""Cluster-side recovery and scheduling; all scientific drivers stay frozen.

The cron entry only submits a small free-CPU controller. Scientific validation,
selection and generation run inside Slurm allocations. Pool workers claim
disjoint construction batches/questions under filesystem locks.
"""
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hpc3_prepare import read, sha, write

ROOT = Path("runs/hpc3_recovery_v1")
EXPANSION = Path("runs/question_breadth_320_v1")
UNSEEN = Path("runs/unseen_graph_v1")
CODE = ["scripts/hpc3_recovery.py", "scripts/hpc3_expansion_pool.py",
        "hpc3/expansion_pool.sbatch", "hpc3/recovery_controller.sbatch",
        "hpc3/recovery_tick.sh"]
CRON_MARKER = "# agent-interact-hpc3-recovery-v1"
RETRY_STATES = {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "TIMEOUT"}


def now():
    return datetime.now(timezone.utc).isoformat()


def command(argv, **kwargs):
    return subprocess.run(argv, text=True, capture_output=True, check=True,
                          timeout=60, **kwargs).stdout


@contextmanager
def lock(path, shared=False, nonblocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(stream, mode | (fcntl.LOCK_NB if nonblocking else 0))
        yield stream


def screen_units(bank, assignments):
    lookup = {q["qid"]: q for q in bank}
    flat = sum(assignments, [])
    if len(flat) != len(set(flat)) or set(flat) != set(lookup):
        raise ValueError("Screening ownership differs from the frozen candidate bank")
    result = []
    for owner, qids in enumerate(assignments):
        for begin in range(0, len(qids), 4):
            result.append({"id": f"screen-w{owner:02d}-g{begin // 4:03d}",
                           "phase": "screen", "qids": qids[begin:begin + 4],
                           "legacy_root": str(EXPANSION / "workers" / f"screen_{owner:02d}")})
    return result


def collection_units(selection):
    questions = selection["new_questions"]
    if len({q["qid"] for q in questions}) != 300:
        raise ValueError("Expected 300 distinct added development questions")
    if any(q["split"] not in {"train", "validation"} for q in questions):
        raise ValueError("Assessment must not enter the pool")
    if sum(q["split"] == "train" for q in questions) != 240:
        raise ValueError("Wrong development split")
    # Preserve the original worker mapping solely to find any saved cache.
    owners = {}
    for split in ["train", "validation"]:
        for i, q in enumerate(sorted((q for q in questions if q["split"] == split),
                                     key=lambda q: q["qid"])):
            owners[q["qid"]] = i % 8
    return [{"id": "collect-" + q["qid"], "phase": "collect", "qids": [q["qid"]],
             "split": q["split"], "graph_id": q["graph_id"],
             "legacy_root": str(EXPANSION / "workers" / f"collect_{owners[q['qid']]:02d}")}
            for q in sorted(questions, key=lambda q: q["qid"])]


def unit_complete(unit):
    if unit["phase"] == "screen":
        return all((EXPANSION / "screen" / (qid + ".json")).is_file() for qid in unit["qids"])
    qid = unit["qids"][0]
    return (EXPANSION / "episodes" / unit["split"] /
            f"{qid}__{unit['graph_id']}__r00.json").is_file()


def prepare():
    from scripts import hpc3_breadth_expand as expand
    from scripts import hpc3_unseen_graph as unseen
    protocol, deployment = expand.verify_deployment()
    graph_protocol, graph_deployment = unseen.verify_deployment()
    if (ROOT / "plan.json").exists():
        return verify_plan()
    bank = read(EXPANSION / "candidate_bank.json")
    units = screen_units(bank, deployment["screening_assignments"])
    previous_expansion = read(read(EXPANSION / "active_submission.json")["submission_path"])
    previous_unseen = read(read(UNSEEN / "active_submission.json")["submission_path"])
    write(ROOT / "screen_units.json", units)
    plan = {"created_utc": now(), "account": "yifengc7", "maximum_gpu_jobs": 24,
            "cron_interval_minutes": 5, "gpu_partition": "free-gpu", "gpu_type": "A30",
            "gpu_hours": 8, "worker_hours": 7.5, "cpu_partition": "free",
            "repository": str(Path.cwd()), "cron_host": socket.gethostname(),
            "code_sha256": {p: sha(p) for p in CODE},
            "screen_units_sha256": sha(ROOT / "screen_units.json"),
            "protocol_sha256": sha(EXPANSION / "protocol.json"),
            "unseen_protocol_sha256": sha(UNSEEN / "protocol.json"),
            "legacy_expansion_jobs": previous_expansion["jobs"],
            "legacy_unseen_jobs": previous_unseen["jobs"],
            "assessment_included": False, "automatic_retraining": False,
            "retry_states": sorted(RETRY_STATES), "retry_exit_codes": [75],
            "startup_cuda_oom_retries_per_slot": 3,
            "scientific_changes": False,
            "policy": "Restart scheduler interruptions; allow three retries for CUDA allocation failure during initial model loading before sampling. Preserve cached request identities; stop on scientific/code errors or explicit cancellation."}
    write(ROOT / "plan.json", plan)
    write(ROOT / "plan_frozen.json", {"created_utc": now(), "sha256": sha(ROOT / "plan.json")})
    records = {f"unseen-{j['worker']}": {**j, "name": f"ag-unseen-r2-{j['worker']}",
               "adopted": True} for j in previous_unseen["jobs"]}
    write(ROOT / "jobs.json", {"latest": records, "history": [], "created_utc": now()})
    write(ROOT / "state.json", {"status": "prepared", "updated_utc": now()})
    return plan


def verify_plan():
    plan = read(ROOT / "plan.json")
    if sha(ROOT / "plan.json") != read(ROOT / "plan_frozen.json")["sha256"]:
        raise ValueError("Recovery plan changed")
    for path, digest in plan["code_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"Recovery code changed: {path}")
    for path, key in [(ROOT / "screen_units.json", "screen_units_sha256"),
                      (EXPANSION / "protocol.json", "protocol_sha256"),
                      (UNSEEN / "protocol.json", "unseen_protocol_sha256")]:
        if sha(path) != plan[key]:
            raise ValueError(f"Frozen input changed: {path}")
    if plan["maximum_gpu_jobs"] != 24 or plan["assessment_included"]:
        raise ValueError("Invalid resource or assessment policy")
    return plan


def queue(account):
    raw = command(["squeue", "-h", "-u", account, "-o", "%i|%j|%T|%P"])
    rows = {}
    for line in raw.splitlines():
        job_id, name, state, partition = line.strip().split("|", 3)
        rows[job_id] = {"job_id": job_id, "name": name, "state": state, "partition": partition}
    return rows


def accounting(job_ids):
    if not job_ids:
        return {}
    raw = command(["sacct", "-X", "-n", "-P", "-j", ",".join(sorted(set(job_ids))),
                   "--format=JobIDRaw,State,ExitCode"])
    output = {}
    for line in raw.splitlines():
        values = line.split("|")
        if len(values) >= 3:
            output[values[0]] = {"state": values[1].split()[0].rstrip("+"), "exit_code": values[2]}
    return output


def terminal_action(result):
    """Do not confuse explicit cancellation or code failure with preemption."""
    if result is None:
        return "wait"
    state = result["state"]
    if state in RETRY_STATES or (state == "FAILED" and result["exit_code"] == "75:0"):
        return "retry"
    if state == "COMPLETED" and result["exit_code"] == "0:0":
        return "finished"
    if state in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "REQUEUED", "SUSPENDED"}:
        return "wait"
    return "halt"


def halt(workflow, reason):
    path = ROOT / f"halt_{workflow}.json"
    with lock(ROOT / f".halt_{workflow}.lock", nonblocking=False):
        if not path.exists():
            write(path, {"created_utc": now(), "reason": reason})


def is_stopped(workflow="expansion"):
    return (ROOT / "STOP").exists() or (ROOT / f"halt_{workflow}.json").exists()


def controller():
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("Run the controller inside a Slurm CPU allocation")
    with lock(ROOT / ".controller.lock"):
        plan = verify_plan()
        if (ROOT / "STOP").exists():
            write(ROOT / "state.json", {"status": "stopped", "updated_utc": now()})
            return
        current = queue(plan["account"])
        legacy_active = [j["job_id"] for j in plan["legacy_expansion_jobs"] if j["job_id"] in current]
        if legacy_active:
            raise RuntimeError(f"Legacy expansion writers must finish stopping before handoff: {legacy_active}")
        jobs = read(ROOT / "jobs.json")
        latest = jobs["latest"]
        history = jobs["history"]
        # Recover an accepted submission if a controller died before recording it.
        for row in current.values():
            name = row["name"]
            key = name.removeprefix("ag-auto-") if name.startswith("ag-auto-") else None
            if key and (key.startswith(("pool-", "unseen-")) or key in {"select", "finalize", "score"}):
                if key not in latest or latest[key]["job_id"] != row["job_id"]:
                    if key in latest:
                        history.append({"key": key, **latest[key]})
                    latest[key] = {**row, "adopted": True, "submitted_utc": now()}
        write(ROOT / "jobs.json", jobs)
        ended = accounting([r["job_id"] for r in latest.values() if r["job_id"] not in current])

        def submit(key, name, script, args=()):
            matches = [row for row in current.values() if row["name"] == name]
            if matches:
                if len(matches) != 1:
                    raise RuntimeError(f"Duplicate active job name: {name}")
                row = {**matches[0], "adopted": True, "submitted_utc": now()}
            else:
                argv = ["sbatch", "--parsable", "--account=" + plan["account"],
                        "--job-name=" + name, script, *map(str, args)]
                job_id = command(argv).strip().split(";")[0]
                if not job_id.isdigit():
                    raise RuntimeError("Invalid sbatch job ID")
                row = {"job_id": job_id, "name": name, "submitted_utc": now(), "command": argv}
                partition = "free-gpu" if key.startswith(("pool-", "unseen-")) else "free"
                current[job_id] = {**row, "state": "PENDING", "partition": partition}
            if key in latest:
                history.append({"key": key, **latest[key], "terminal": ended.get(latest[key]["job_id"])})
            latest[key] = row
            jobs["updated_utc"] = now()
            write(ROOT / "jobs.json", jobs)
            return row["job_id"]

        def ready(key, workflow, allow_finished=True):
            previous = latest.get(key)
            if previous is None:
                return True
            if previous["job_id"] in current:
                return False
            result = ended.get(previous["job_id"])
            action = terminal_action(result)
            if action == "halt" or (action == "finished" and not allow_finished):
                halt(workflow, {"key": key, "job_id": previous["job_id"], "terminal": result,
                                "detail": "Job did not produce the required completion marker"})
            return action == "retry" or (action == "finished" and allow_finished)

        def gpu_count():
            return sum(row["partition"].startswith(("free-gpu", "gpu")) for row in current.values())

        # The original four unseen-graph workers keep their exact driver and caches.
        unseen_protocol = read(UNSEEN / "protocol.json")
        unseen_deployment = read(UNSEEN / "deployment.json")
        by_id = {q["qid"]: q for q in unseen_protocol["questions"]}
        unseen_remaining = []
        for worker, ids in enumerate(unseen_deployment["assignments"]):
            complete = all((UNSEEN / "episodes/validation" /
                            f"{qid}__{by_id[qid]['graph_id']}__r00.json").is_file() for qid in ids)
            if not complete:
                unseen_remaining.append(worker)
                if (not is_stopped("unseen") and ready(f"unseen-{worker}", "unseen", False)
                        and gpu_count() < plan["maximum_gpu_jobs"]):
                    submit(f"unseen-{worker}", f"ag-auto-unseen-{worker}", "hpc3/unseen_graph.sbatch", [worker])
        if not unseen_remaining and not (UNSEEN / "results.json").exists() and not is_stopped("unseen"):
            if ready("score", "unseen", False):
                submit("score", "ag-auto-score", "hpc3/recovery_controller.sbatch", ["score"])

        units = read(ROOT / "screen_units.json")
        incomplete_screens = [unit for unit in units if not unit_complete(unit)]
        stage = "screen"
        remaining = incomplete_screens
        if not incomplete_screens:
            stage = "select"
            remaining = []
            if not (EXPANSION / "selection_ready.json").exists():
                if not is_stopped() and ready("select", "expansion", False):
                    submit("select", "ag-auto-select", "hpc3/breadth_expand_finish.sbatch", ["select"])
            else:
                from scripts.breadth_expand import verify_selection
                selection = verify_selection()
                path = ROOT / "collection_units.json"
                new_units = collection_units(selection)
                if path.exists():
                    if read(path) != new_units:
                        raise RuntimeError("Collection unit mapping changed")
                else:
                    write(path, new_units)
                if not (ROOT / "collection_units_frozen.json").exists():
                    write(ROOT / "collection_units_frozen.json", {
                        "created_utc": now(), "sha256": sha(path),
                        "selection_sha256": sha(EXPANSION / "selection.json")})
                else:
                    frozen = read(ROOT / "collection_units_frozen.json")
                    if sha(path) != frozen["sha256"] or sha(EXPANSION / "selection.json") != frozen["selection_sha256"]:
                        raise RuntimeError("Frozen collection units changed")
                stage = "collect"
                remaining = [unit for unit in new_units if not unit_complete(unit)]
                if not remaining:
                    stage = "finalize"
                    if not (EXPANSION / "collection_complete.json").exists():
                        if not is_stopped() and ready("finalize", "expansion", False):
                            submit("finalize", "ag-auto-finalize", "hpc3/breadth_expand_finish.sbatch", ["finalize"])
                    else:
                        stage = "complete"

        # Inspect ended pool jobs even if a phase just completed, to catch real errors.
        for key in list(latest):
            if key.startswith("pool-") and latest[key]["job_id"] not in current:
                ready(key, "expansion")
        pool_rows = [row for row in current.values() if row["name"].startswith("ag-auto-pool-")]
        other_gpus = gpu_count() - len(pool_rows)
        target = min(len(remaining), max(0, plan["maximum_gpu_jobs"] - other_gpus))
        if not is_stopped():
            for slot in range(plan["maximum_gpu_jobs"]):
                if len(pool_rows) >= target or is_stopped():
                    break
                key, name = f"pool-{slot:02d}", f"ag-auto-pool-{slot:02d}"
                if any(row["name"] == name for row in pool_rows):
                    continue
                if ready(key, "expansion"):
                    job_id = submit(key, name, "hpc3/expansion_pool.sbatch", [slot])
                    pool_rows.append(current[job_id])

        expansion_done = (EXPANSION / "collection_complete.json").exists()
        unseen_done = (UNSEEN / "results.json").exists()
        terminal = (expansion_done or is_stopped()) and (unseen_done or is_stopped("unseen"))
        status = "complete" if expansion_done and unseen_done else "needs_attention" if terminal else "running"
        progress = {"updated_utc": now(), "status": status, "stage": stage,
                    "controller_job_id": os.environ["SLURM_JOB_ID"],
                    "screened": len(list((EXPANSION / "screen").glob("*.json"))),
                    "candidate_count": 1200, "remaining_units": len(remaining),
                    "episodes": {split: len(list((EXPANSION / "episodes" / split).glob("*.json")))
                                 for split in ["train", "validation", "assessment"]},
                    "unseen_completed": len(list((UNSEEN / "episodes/validation").glob("*.json"))),
                    "unseen_scored": unseen_done, "expansion_complete": expansion_done,
                    "pool_jobs": [{k: row[k] for k in ["job_id", "name", "state"]} for row in pool_rows],
                    "maximum_gpu_jobs": plan["maximum_gpu_jobs"],
                    "halted_workflows": [w for w in ["expansion", "unseen"] if is_stopped(w)]}
        write(ROOT / "state.json", progress)
        write(ROOT / "controller_checks" / (os.environ["SLURM_JOB_ID"] + ".json"), progress)
        if status == "complete":
            write(ROOT / "complete.json", {"completed_utc": now(),
                "artifacts": {str(p): sha(p) for p in [EXPANSION / "collection_complete.json",
                    EXPANSION / "episode_index.json", UNSEEN / "results.json"]},
                "assessment_included": False, "predictors_trained": False})
        print(json.dumps(progress, indent=2))


def cron_text(previous, entry):
    kept = [line for line in previous.splitlines() if CRON_MARKER not in line]
    if entry is not None:
        kept.append(entry)
    return "\n".join(kept).rstrip() + "\n"


def existing_cron():
    result = subprocess.run(["crontab", "-l"], text=True, capture_output=True, timeout=30)
    if result.returncode and "no crontab" not in result.stderr:
        raise RuntimeError(result.stderr)
    return result.stdout


def install_cron():
    plan = verify_plan()
    if socket.gethostname() != plan["cron_host"]:
        raise RuntimeError("Install on the recorded login host")
    previous = existing_cron()
    write(ROOT / "cron_before.json", {"saved_utc": now(), "host": socket.gethostname(), "text": previous})
    script = str(Path(plan["repository"]) / "hpc3/recovery_tick.sh")
    entry = f"*/5 * * * * /bin/bash {shlex.quote(script)} {CRON_MARKER}"
    updated = cron_text(previous, entry)
    command(["crontab", "-"], input=updated)
    if existing_cron() != updated:
        raise RuntimeError("Crontab installation did not verify")
    write(ROOT / "cron_installed.json", {"installed_utc": now(), "host": socket.gethostname(),
          "entry": entry, "interval_minutes": 5, "previous_preserved": True})
    print(json.dumps(read(ROOT / "cron_installed.json")))


def tick(from_cron=False):
    # Only scheduling and tiny metadata reads here; no model/data computation.
    with lock(ROOT / ".tick.lock"):
        plan = read(ROOT / "plan.json")
        state = read(ROOT / "state.json")
        heartbeat = {"checked_utc": now(), "host": socket.gethostname(),
                     "origin": "cron" if from_cron else "manual", "status": state["status"]}
        if (ROOT / "STOP").exists() or state["status"] in {"complete", "needs_attention", "stopped"}:
            command(["crontab", "-"], input=cron_text(existing_cron(), None))
            heartbeat["cron_removed"] = True
        else:
            active = [r for r in queue(plan["account"]).values() if r["name"] == "ag-auto-control"]
            if len(active) > 1:
                raise RuntimeError("Duplicate controller jobs")
            if active:
                heartbeat["controller_job_id"] = active[0]["job_id"]
            else:
                job_id = command(["sbatch", "--parsable", "--account=" + plan["account"],
                                  "--job-name=ag-auto-control", "hpc3/recovery_controller.sbatch", "controller"]).strip().split(";")[0]
                if not job_id.isdigit():
                    raise RuntimeError("Invalid controller job ID")
                heartbeat["controller_job_id"] = job_id
                heartbeat["submitted"] = True
        write(ROOT / ("cron_heartbeat.json" if from_cron else "manual_tick.json"), heartbeat)
        print(json.dumps(heartbeat), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "controller", "tick", "install-cron", "score", "status"])
    parser.add_argument("--cron", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        plan = prepare()
        print(json.dumps({"prepared": True, "maximum_gpu_jobs": plan["maximum_gpu_jobs"],
                          "screen_units": len(read(ROOT / "screen_units.json"))}))
    elif args.stage == "controller":
        controller()
    elif args.stage == "tick":
        tick(args.cron)
    elif args.stage == "install-cron":
        install_cron()
    elif args.stage == "score":
        verify_plan()
        from scripts.hpc3_unseen_graph import maybe_score
        if not maybe_score():
            raise RuntimeError("Unseen-graph cohort is incomplete")
    else:
        print(json.dumps(read(ROOT / "state.json"), indent=2))


if __name__ == "__main__":
    main()
