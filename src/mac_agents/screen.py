"""Bounded validation screen; never chooses or evaluates controlled test runs."""
from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import time

from .common import json_write
from .train import train

MATRIX = [("P0", "opinion"), ("P1", "opinion"), ("I0", "opinion"),
          ("A0", "opinion"), ("A1", "opinion"), ("A2", "opinion"),
          ("A2", "vote_counts"), ("A2", "history"), ("A2", "existing_messages"),
          ("A0", "vote_counts"), ("A0", "history"), ("A0", "existing_messages")]


def screen(config_path, minutes=120, device="auto"):
    if minutes <= 0 or minutes > 120:
        raise ValueError("Initial screen must use a positive aggregate budget of at most 120 minutes")
    from .common import config
    from .data.prepare import load_prepared
    from .data.features import Snapshots
    cfg = config(config_path)
    arrays, index, _ = load_prepared(cfg)
    for partition in ("train", "validation"):
        data = Snapshots(arrays, index, cfg, partition, "existing_messages")
        if data.incomplete_messages:
            raise ValueError("Complete development message features before starting the full screen")
    plan_path = Path("runs") / f"screen_plan_{Path(config_path).stem}.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["config"] != config_path:
            raise ValueError("Existing screen uses a different config")
        if any(item["status"] == "running" for item in plan["runs"]):
            raise ValueError("A screen was interrupted mid-run; inspect its checkpoint and resume that run before updating the plan")
    else:
        plan = {"config": config_path, "purpose": "validation only; one seed; no test selection",
                "runs": [{"model": kind, "state": state, "status": "pending"} for kind, state in MATRIX]}
    started = time.monotonic()
    for item in plan["runs"]:
        if item["status"] == "complete":
            continue
        remaining = minutes - (time.monotonic()-started)/60
        if remaining <= 0:
            break
        item["status"] = "running"
        json_write(plan_path, plan)
        args = Namespace(config=config_path, model=item["model"], state=item["state"], seed=42,
                         device=device, max_steps=None, wall_minutes=remaining, limit_episodes=None,
                         match_state=None, resume=Path(item["run"]) if item.get("run") else None, smoke=False)
        try:
            run = train(args)
        except Exception:
            item["status"] = "pending"
            json_write(plan_path, plan)
            raise
        item["run"] = str(run)
        manifest = json.loads((run / "manifest.json").read_text())
        item["status"] = "pending" if manifest["stopping_reason"] == "wall_time_limit" else "complete"
        json_write(plan_path, plan)
        if item["status"] == "pending":
            break
    print(f"Screen progress saved to {plan_path}; test data remains locked.")
