"""Continue train/validation collection after a successful HPC3 replay.

The existing collector verifies the stopped-Mac export, exact source protocol,
CUDA runtime and replay results before generating anything. Assessment is never
part of this launcher. A failed or interrupted invocation remains resumable.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hpc3_prepare import read, write, sha


def continue_collection(report_path, hours=47.5):
    if not 0 < hours <= 71:
        raise ValueError("Invalid total collection time budget")
    report_path = Path(report_path)
    if not read(report_path).get("numerical_gate_passed"):
        raise RuntimeError("Automatic collection requires a passing numerical replay")
    state_path = Path("hpc3/automatic_collection_state.json")
    state = {"started_utc": datetime.now(timezone.utc).isoformat(),
             "benchmark_report": str(report_path), "benchmark_sha256": sha(report_path),
             "launcher_sha256": sha(__file__), "hours": hours,
             "planned_splits": ["train", "validation"], "assessment_included": False,
             "completed_splits": [], "status": "running"}
    deadline = time.monotonic() + 3600 * hours
    with Path("hpc3/.automatic_collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for split in ["train", "validation"]:
            remaining = deadline - time.monotonic()
            if remaining < 60:
                write(state_path, {**state, "status": "time_limit", "next_split": split})
                return 75
            write(state_path, {**state, "active_split": split})
            command = [sys.executable, "scripts/hpc3_breadth.py", "collect", "--split", split,
                       "--benchmark-report", str(report_path), "--hours", str(remaining / 3600)]
            try:
                result = subprocess.run(command, timeout=remaining)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 75
            if code:
                write(state_path, {**state, "status": "time_limit" if code == 75 else "failed",
                                   "active_split": split, "exit_code": code})
                return code
            state["completed_splits"].append(split)
        write(state_path, {**state, "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()})
        return 0


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=47.5)
    args = parser.parse_args()
    raise SystemExit(continue_collection(args.benchmark_report, args.hours))
