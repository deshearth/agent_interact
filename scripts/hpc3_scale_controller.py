"""Independent HPC3 scheduler for the 3,200/800 corpus; no Mac dependency."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hpc3_prepare import read, sha, write
from scripts.hpc3_recovery import lock, command, accounting, terminal_action

ROOT = Path("runs/hpc3_scale_3200_v1")
STUDY = Path("runs/question_breadth_3200_v1")
MARKER = "# agent-interact-hpc3-scale-3200-v1"
PREFIX = "ag3200-"


def now():
    return datetime.now(timezone.utc).isoformat()


def stopped():
    return (ROOT / "STOP").exists() or (ROOT / "halt.json").exists()


def halt(reason):
    with lock(ROOT / ".halt.lock", nonblocking=False):
        if not (ROOT / "halt.json").exists():
            write(ROOT / "halt.json", {"utc": now(), "reason": reason})


def runtime_key(execution):
    return hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()


def queue(account):
    raw = command(["squeue", "-r", "-h", "-u", account, "-o", "%i|%j|%T|%P|%b"])
    rows = {}
    for line in raw.splitlines():
        job, name, state, partition, tres = line.strip().split("|", 4)
        gpu = sum(int(t.rsplit(":", 1)[-1]) for t in tres.split(",") if "gpu" in t and t.rsplit(":", 1)[-1].isdigit())
        rows[job] = {"job_id": job, "name": name, "state": state, "partition": partition,
                     "gpu_count": gpu or int(partition.startswith("free-gpu"))}
    return rows


def screen_units(bank, seed):
    qids = sorted(q["qid"] for q in bank)
    assert len(qids) == len(set(qids))
    random.Random(seed).shuffle(qids)
    return [{"id": f"screen-{i // 4:05d}", "phase": "screen", "qids": qids[i:i + 4]} for i in range(0, len(qids), 4)]


def collect_units(selection, plan):
    questions = selection["new_questions"]
    if Counter(q["split"] for q in questions) != plan["new_questions"]:
        raise ValueError("Unexpected development counts or assessment in collection")
    assert len({q["qid"] for q in questions}) == len(questions)
    assert not {q["qid"] for q in questions} & set(plan["reserved_assessment_qids"])
    rows = [{"id": "collect-" + q["qid"], "phase": "collect", "qids": [q["qid"]],
             "split": q["split"], "graph_id": q["graph_id"]} for q in sorted(questions, key=lambda q: q["qid"])]
    random.Random(plan["unit_seed"] + 1).shuffle(rows)
    return rows


def unit_complete(unit):
    if unit["phase"] == "screen":
        return all((STUDY / "screen" / (q + ".json")).is_file() for q in unit["qids"])
    return (STUDY / "episodes" / unit["split"] / f"{unit['qids'][0]}__{unit['graph_id']}__r00.json").is_file()


def verify_plan():
    plan = read(ROOT / "plan.json")
    assert sha(ROOT / "plan.json") == read(ROOT / "plan_frozen.json")["sha256"]
    assert sha(STUDY / "protocol.json") == plan["protocol_sha256"]
    assert sha(ROOT / "screen_units.json") == plan["screen_units_sha256"]
    protocol = read(STUDY / "protocol.json")
    for group in ("code_sha256", "source_files_sha256"):
        for path, digest in protocol[group].items():
            assert sha(path) == digest, path
    assert not plan["assessment_included"]
    assert plan["partition_limits"] == {"free-gpu": 24, "free-gpu32": 8}
    return plan


def register_report(path, gpu):
    """Only append a fully passed original replay; never edit existing entries."""
    report = read(path)
    if not report.get("numerical_gate_passed") or not report.get("full_reference_coverage"):
        return False
    selected = [c for c in report["candidates"] if c["batch_size"] == report["selected_batch_size"]]
    if len(selected) != 1 or not selected[0]["passed"] or not all(c["passed"] for c in selected[0]["checks"]):
        return False
    runtime = report["runtime"]
    assert runtime["adapter_sha256"] == sha("scripts/hpc3_breadth.py")
    assert runtime["fixtures_sha256"] == sha("hpc3/fixtures.json")
    for name, digest in report["raw_results"].items():
        # Older replay manifests may record their absolute container path.
        name = name.removeprefix("/workspace/")
        assert sha(name) == digest, name
    with lock(ROOT / ".registry.lock", nonblocking=False):
        registry = read(ROOT / "runtime_registry.json") if (ROOT / "runtime_registry.json").exists() else {"entries": []}
        identity = runtime_key(runtime)
        previous = [r for r in registry["entries"] if r["runtime_key"] == identity]
        if previous:
            assert sha(previous[0]["report"]) == previous[0]["report_sha256"]
            return True
        registry["entries"].append({"gpu": gpu, "runtime_key": identity, "runtime": runtime,
            "report": str(path), "report_sha256": sha(path), "batch_size": report["selected_batch_size"], "registered_utc": now()})
        write(ROOT / "runtime_registry.json", registry)
    return True


def prepare(cron_host=None):
    if (ROOT / "plan.json").exists():
        return verify_plan()
    protocol = read(STUDY / "protocol.json")
    settings = protocol["config"]
    seed = settings["selection"]["unit_seed"]
    units = screen_units(read(STUDY / "candidate_bank.json"), seed)
    write(ROOT / "screen_units.json", units)
    plan = {"created_utc": now(), "repository": str(Path.cwd()), "cron_host": cron_host or socket.gethostname(),
        "account": "yifengc7", "partition_limits": settings["collection"]["partitions"],
        "worker_hours": settings["collection"]["worker_hours"], "walltime_hours": settings["collection"]["walltime_hours"],
        "protocol_sha256": sha(STUDY / "protocol.json"), "screen_units_sha256": sha(ROOT / "screen_units.json"),
        "unit_seed": seed, "new_questions": settings["new_questions"], "target_questions": settings["target_splits"],
        "candidate_count": protocol["candidate_count"], "reserved_assessment_qids": protocol["reserved_assessment_qids"],
        "assessment_included": False, "automatic_retraining": False, "startup_oom_retries": 3,
        "retry_policy": "Resume scheduler interruptions and explicit checkpoint exits; stop on cancellation or scientific/code failure.",
        "hardware_policy": "24 A30 workers in free-gpu, plus up to 8 numerically certified L40S/RTX6000 workers in free-gpu32; no paid partitions. Account for other active GPU allocations. Each unit is pinned to one certified runtime."}
    write(ROOT / "plan.json", plan)
    write(ROOT / "plan_frozen.json", {"created_utc": now(), "sha256": sha(ROOT / "plan.json")})
    write(ROOT / "jobs.json", {"latest": {}, "history": []})
    write(ROOT / "state.json", {"status": "prepared", "updated_utc": now()})
    register_report(settings["collection"]["benchmark_report"], "A30")
    return verify_plan()


def refresh_hardware():
    probes = ROOT / "hardware_probes.json"
    if probes.exists():
        for probe in read(probes)["jobs"]:
            job = probe.get("stdout", "").split(";")[0]
            path = Path("hpc3/results") / ("benchmark_" + job) / "report.json"
            if probe["returncode"] == 0 and job.isdigit() and path.exists():
                register_report(path, probe["gpu"])
    registry = read(ROOT / "runtime_registry.json")
    for entry in registry["entries"]:
        assert sha(entry["report"]) == entry["report_sha256"]
    return registry


def available_units():
    if (STUDY / "selection_ready.json").exists():
        plan = verify_plan()
        ready = read(STUDY / "selection_ready.json")
        assert ready["selection_sha256"] == sha(STUDY / "selection.json")
        return collect_units(read(STUDY / "selection.json"), plan)
    return read(ROOT / "screen_units.json")


def controller():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Controller work requires a Slurm allocation")
    plan = verify_plan()
    with lock(ROOT / ".controller.lock"):
        registry = refresh_hardware()
        current = queue(plan["account"])
        jobs = read(ROOT / "jobs.json")
        latest, history = jobs["latest"], jobs["history"]
        for row in current.values():
            key = row["name"].removeprefix(PREFIX)
            if row["name"].startswith(PREFIX) and (key.startswith("pool-") or key in ("select", "finalize")):
                if key not in latest or latest[key]["job_id"] != row["job_id"]:
                    if key in latest:
                        history.append({"key": key, **latest[key]})
                    latest[key] = {**row, "adopted": True, "submitted_utc": now()}
        ended = accounting([r["job_id"] for r in latest.values() if r["job_id"] not in current])

        def ready(key, require_marker=False):
            old = latest.get(key)
            if old is None:
                return True
            if old["job_id"] in current:
                return False
            action = terminal_action(ended.get(old["job_id"]))
            if action == "halt" or (action == "finished" and require_marker):
                halt({"key": key, "job_id": old["job_id"], "terminal": ended.get(old["job_id"])})
            return action == "retry" or (action == "finished" and not require_marker)

        def submit(key, script, args, extra=(), partition="free"):
            name = PREFIX + key
            matches = [r for r in current.values() if r["name"] == name]
            if matches:
                assert len(matches) == 1
                row = {**matches[0], "adopted": True}
            else:
                argv = ["sbatch", "--parsable", "--account=" + plan["account"], "--job-name=" + name, *extra, script, *map(str, args)]
                job = command(argv).strip().split(";")[0]
                assert job.isdigit()
                row = {"job_id": job, "name": name, "partition": partition, "state": "PENDING",
                       "gpu_count": int(partition.startswith("free-gpu")), "command": argv, "submitted_utc": now()}
                current[job] = row
            if key in latest:
                history.append({"key": key, **latest[key], "terminal": ended.get(latest[key]["job_id"])})
            latest[key] = row
            write(ROOT / "jobs.json", jobs)
            return row

        for key in list(latest):
            if key.startswith("pool-") and latest[key]["job_id"] not in current:
                ready(key)
        units = read(ROOT / "screen_units.json")
        remaining = [u for u in units if not unit_complete(u)]
        stage = "screen"
        if not remaining:
            stage = "select"
            if not (STUDY / "selection_ready.json").exists():
                if not stopped() and ready("select", True):
                    submit("select", "hpc3/scale_controller.sbatch", ["select"])
            else:
                units = available_units()
                units_path = ROOT / "collection_units.json"
                if units_path.exists():
                    assert read(units_path) == units
                else:
                    write(units_path, units)
                    write(ROOT / "collection_units_frozen.json", {"sha256": sha(units_path), "selection_sha256": sha(STUDY / "selection.json")})
                frozen = read(ROOT / "collection_units_frozen.json")
                assert frozen == {"sha256": sha(units_path), "selection_sha256": sha(STUDY / "selection.json")}
                remaining = [u for u in units if not unit_complete(u)]
                stage = "collect"
                if not remaining:
                    stage = "finalize"
                    if not (STUDY / "collection_complete.json").exists():
                        if not stopped() and ready("finalize", True):
                            submit("finalize", "hpc3/scale_controller.sbatch", ["finalize"])
                    else:
                        stage = "complete"
        pool = [r for r in current.values() if r["name"].startswith(PREFIX + "pool-")]
        gpu_types = {r["gpu"] for r in registry["entries"]}
        for partition, limit in plan["partition_limits"].items():
            types = ["A30"] if partition == "free-gpu" else sorted(gpu_types & {"L40S", "RTX6000"})
            if not types or stopped():
                continue
            other = sum(r["gpu_count"] for r in current.values() if partition in r["partition"].split(",") and not r["name"].startswith(PREFIX + "pool-"))
            existing = [r for r in pool if r["partition"] == partition]
            runtime_keys = {r["runtime_key"] for r in registry["entries"] if r["gpu"] in types}
            eligible = 0
            for unit in remaining:
                binding = ROOT / "units" / unit["id"] / "runtime_binding.json"
                if not binding.exists() or runtime_key(read(binding)["execution"]["runtime"]) in runtime_keys:
                    eligible += 1
            other_pool = len(pool) - len(existing)
            target = min(eligible, max(0, len(remaining) - other_pool), max(0, limit - other))
            offset = 0 if partition == "free-gpu" else 24
            for slot in range(offset, offset + limit):
                if len(existing) >= target or stopped():
                    break
                key = f"pool-{slot:02d}"
                if any(r["name"] == PREFIX + key for r in pool) or not ready(key):
                    continue
                gres = "gpu:" + (types[0] + ":" if len(types) == 1 else "") + "1"
                row = submit(key, "hpc3/scale_worker.sbatch", [slot],
                    ["--partition=" + partition, "--qos=low", "--gres=" + gres,
                     "--cpus-per-task=" + ("4" if partition == "free-gpu" else "6")], partition)
                pool.append(row)
                existing.append(row)
        done = (STUDY / "collection_complete.json").exists()
        status = "complete" if done else "needs_attention" if stopped() else "running"
        progress = {"updated_utc": now(), "status": status, "stage": stage, "controller_job_id": os.environ["SLURM_JOB_ID"],
            "candidate_count": plan["candidate_count"], "screened": len(list((STUDY / "screen").glob("*.json"))),
            "remaining_units": len(remaining), "episodes": {s: len(list((STUDY / "episodes" / s).glob("*.json"))) for s in ("train", "validation", "assessment")},
            "target_questions": plan["target_questions"], "certified_gpu_types": sorted(gpu_types),
            "partition_limits": plan["partition_limits"], "pool_jobs": [{k: r[k] for k in ("job_id", "name", "partition", "state")} for r in pool]}
        write(ROOT / "jobs.json", jobs)
        write(ROOT / "state.json", progress)
        write(ROOT / "controller_checks" / (os.environ["SLURM_JOB_ID"] + ".json"), progress)
        if done:
            write(ROOT / "complete.json", {"completed_utc": now(), "source_completion_sha256": sha(STUDY / "collection_complete.json"),
                "index_sha256": sha(STUDY / "episode_index.json"), "assessment_included": False, "predictors_trained": False})
        print(json.dumps(progress, indent=2), flush=True)


def cron_text(previous, entry):
    kept = [line for line in previous.splitlines() if MARKER not in line]
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
    assert socket.gethostname() == plan["cron_host"]
    before = existing_cron()
    if not (ROOT / "cron_before.json").exists():
        write(ROOT / "cron_before.json", {"saved_utc": now(), "host": socket.gethostname(), "text": before})
    script = str(Path(plan["repository"]) / "hpc3/scale_tick.sh")
    entry = f"*/5 * * * * /bin/bash {shlex.quote(script)} {MARKER}"
    updated = cron_text(before, entry)
    command(["crontab", "-"], input=updated)
    assert existing_cron() == updated
    write(ROOT / "cron_installed.json", {"installed_utc": now(), "host": socket.gethostname(), "entry": entry})


def tick(from_cron=False):
    # Login-node work is restricted to small metadata reads and Slurm submission.
    with lock(ROOT / ".tick.lock"):
        plan, state = read(ROOT / "plan.json"), read(ROOT / "state.json")
        heartbeat = {"checked_utc": now(), "host": socket.gethostname(),
                     "origin": "cron" if from_cron else "manual", "status": state["status"]}
        if stopped() or state["status"] in ("complete", "needs_attention", "stopped"):
            command(["crontab", "-"], input=cron_text(existing_cron(), None))
            heartbeat["cron_removed"] = True
        else:
            active = [r for r in queue(plan["account"]).values() if r["name"] == PREFIX + "control"]
            assert len(active) <= 1
            if active:
                heartbeat["controller_job_id"] = active[0]["job_id"]
            else:
                job = command(["sbatch", "--parsable", "--account=" + plan["account"],
                               "--job-name=" + PREFIX + "control", "hpc3/scale_controller.sbatch", "controller"]).strip().split(";")[0]
                assert job.isdigit()
                heartbeat.update(controller_job_id=job, submitted=True)
        write(ROOT / ("cron_heartbeat.json" if from_cron else "manual_tick.json"), heartbeat)
        print(json.dumps(heartbeat), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "controller", "install-cron", "tick", "status"])
    parser.add_argument("--cron", action="store_true")
    parser.add_argument("--cron-host")
    args = parser.parse_args()
    if args.stage == "prepare":
        print(json.dumps({"prepared": bool(prepare(args.cron_host))}))
    elif args.stage == "controller":
        controller()
    elif args.stage == "install-cron":
        install_cron()
    elif args.stage == "tick":
        tick(args.cron)
    else:
        print(json.dumps(read(ROOT / "state.json"), indent=2))
