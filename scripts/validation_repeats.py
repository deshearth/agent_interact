"""Repeat the complete validation matrix and evaluate autonomous validation rollouts.

Does not select, unlock, encode, or evaluate any controlled test partition.
Run after the one-seed screen. An interrupted invocation can be resumed.
"""
from argparse import ArgumentParser, Namespace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from mac_agents.common import code_digest, json_write
from mac_agents.evaluate import evaluate
from mac_agents.train import train


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=60)
    args = parser.parse_args()
    if not 0 < args.minutes <= 60:
        raise ValueError("Validation follow-up is capped at 60 minutes")
    screen = json.loads(Path("runs/screen_plan_paper_binary.json").read_text())
    if any(x["status"] != "complete" for x in screen["runs"]):
        raise ValueError("Finish the one-seed screen first")
    path = Path("runs/validation_repeats_plan.json")
    if path.exists():
        plan = json.loads(path.read_text())
        if plan["code_sha256"] != code_digest():
            raise ValueError("Source changed; do not mix validation runs")
    else:
        plan = {
            "purpose": "Full validation matrix, seeds 42/43/44; test remains locked",
            "rationale": "Repeat all state and routing controls; no single-seed winner selection",
            "code_sha256": code_digest(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runs": [dict(x, seed=42) for x in screen["runs"]] + [
                {"model": x["model"], "state": x["state"], "seed": seed, "status": "pending"}
                for seed in (43, 44) for x in screen["runs"] if x["model"].startswith("A")
            ],
            "rollouts": [],
        }
        json_write(path, plan)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=args.minutes)
    for item in plan["runs"]:
        if item["status"] == "complete":
            continue
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds() / 60
        if remaining <= 0:
            print("Budget reached; pending training saved", flush=True)
            return
        if item["status"] == "running" and not item.get("run"):
            candidates = [p.parent for p in Path("runs").glob("*/manifest.json")
                          if p.parent.name not in item.get("prior_directories", [])]
            matching = []
            for candidate in candidates:
                meta = json.loads((candidate / "manifest.json").read_text())
                if (meta.get("kind"), meta.get("state"), meta.get("seed"), meta.get("code_sha256")) == (
                        item["model"], item["state"], item["seed"], plan["code_sha256"]):
                    matching.append(candidate)
            if len(matching) != 1:
                raise ValueError("Ambiguous interrupted run; inspect checkpoints before resuming")
            item["run"] = str(matching[0])
        if item.get("run"):
            meta = json.loads((Path(item["run"]) / "manifest.json").read_text())
            if meta["status"] == "complete":
                item["status"] = "complete"
                json_write(path, plan)
                continue
        item["status"] = "running"
        item["prior_directories"] = [p.name for p in Path("runs").iterdir() if p.is_dir()]
        json_write(path, plan)
        result = train(Namespace(config=screen["config"], model=item["model"], state=item["state"],
            seed=item["seed"], device="mps", max_steps=None, wall_minutes=remaining,
            limit_episodes=None, match_state=None, resume=Path(item["run"]) if item.get("run") else None,
            smoke=False))
        item["run"] = str(result)
        meta = json.loads((result / "manifest.json").read_text())
        item["status"] = "pending" if meta["stopping_reason"] == "wall_time_limit" else "complete"
        item.pop("prior_directories", None)
        json_write(path, plan)
        if item["status"] != "complete":
            return
    for item in plan["runs"]:
        if item["state"] not in ("opinion", "history") or item["run"] in plan["rollouts"]:
            continue
        if datetime.now(timezone.utc) >= deadline:
            print("Budget reached; pending rollouts saved", flush=True)
            return
        evaluate(Namespace(run=Path(item["run"]), partition="validation", mode="closed_loop",
                           selection=None, rollouts=32))
        plan["rollouts"].append(item["run"])
        json_write(path, plan)
    plan["completed_utc"] = datetime.now(timezone.utc).isoformat()
    json_write(path, plan)
    print("Validation repeats and autonomous rollouts complete; test remains locked", flush=True)


if __name__ == "__main__":
    main()
