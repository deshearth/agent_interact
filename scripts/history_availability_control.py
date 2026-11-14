"""Validation-only follow-up: distinguish history values from availability cues.

Uses exactly the existing six-feature history architecture and optimizer. Only
the two preceding-opinion indicator values are zeroed; availability and current
opinion features are retained. Runs are stored separately, with this adapter's
checksum, and are deliberately not standard CLI evaluation/freeze artifacts.
"""
from argparse import Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from mac_agents.common import code_digest, digest, json_write, new_run
from mac_agents.data.features import Snapshots, history_features
from mac_agents import train as training


def availability_only(extra):
    if extra.shape[-1] != 6:
        raise ValueError("This matched control requires the two-snapshot history configuration")
    result = extra.clone()
    result[..., :2] = 0
    return result


class AvailabilitySnapshots(Snapshots):
    def __getitem__(self, item):
        batch = super().__getitem__(item)
        batch["extra"] = availability_only(batch["extra"])
        return batch


def verify_transform():
    import torch
    a = np.array([[-1, 1], [1, -1]], dtype=np.float32)
    b = a.copy()
    b[0] *= -1
    x = availability_only(torch.from_numpy(history_features(a, 1, 2)))
    y = availability_only(torch.from_numpy(history_features(b, 1, 2)))
    assert torch.equal(x, y), "Past opinions leaked into the control"
    expected = torch.from_numpy(history_features(a, 1, 2))
    assert torch.equal(x[..., 2:], expected[..., 2:])
    t0 = torch.from_numpy(history_features(a, 0, 2))
    assert torch.equal(availability_only(t0), t0)
    assert torch.all(x[..., 2] == 1) and torch.all(t0[..., 2] == 0)


def main():
    verify_transform()
    repeats = json.loads(Path("runs/validation_repeats_plan.json").read_text())
    if "completed_utc" not in repeats:
        raise ValueError("Finish the original matrix and rollouts before this GPU stage")
    root = Path("runs/history_availability_control")
    root.mkdir(exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["adapter_sha256"] != digest(__file__) or plan["base_code_sha256"] != code_digest():
            raise ValueError("Control implementation changed; do not mix runs")
    else:
        plan = {"purpose": "Post-screen validation diagnosis; no test evaluation",
                "trigger": "Most A2 history NLL improvement occurred at initialization, when past opinions are absent",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "base_code_sha256": code_digest(), "adapter_sha256": digest(__file__),
                "feature_rule": "Zero history extra[..., :2]; retain previous availability and all current-state features",
                "runs": [{"model": model, "seed": seed, "status": "pending"}
                         for seed in (42, 43, 44) for model in ("A0", "A2")]}
        json_write(plan_path, plan)
    original_snapshots, original_new_run = training.Snapshots, training.new_run
    training.Snapshots = AvailabilitySnapshots
    started = time.monotonic()
    try:
        for item in plan["runs"]:
            if item["status"] == "complete":
                continue
            if item.get("run") and (Path(item["run"]) / "control_manifest.json").exists():
                saved = json.loads((Path(item["run"]) / "control_manifest.json").read_text())
                if saved["adapter_sha256"] != plan["adapter_sha256"] or saved["status"] != "complete":
                    raise ValueError("Interrupted control manifest needs review")
                item["status"] = "complete"
                json_write(plan_path, plan)
                continue
            remaining = 20 - (time.monotonic()-started)/60
            if remaining <= 0:
                print("Control cap reached; pending work saved", flush=True)
                return

            def allocate(label):
                allocated = new_run(label.replace("-history-", "-history_availability-"))
                destination = root / allocated.name
                allocated.rename(destination)
                item["run"] = str(destination)
                json_write(plan_path, plan)
                return destination

            training.new_run = allocate
            item["status"] = "running"
            json_write(plan_path, plan)
            result = training.train(Namespace(config="configs/paper_binary.yaml", model=item["model"],
                state="history", seed=item["seed"], device="mps", max_steps=None,
                wall_minutes=remaining, limit_episodes=None, match_state=None,
                resume=Path(item["run"]) if item.get("run") else None, smoke=False))
            meta = json.loads((result / "manifest.json").read_text())
            if meta["stopping_reason"] == "wall_time_limit":
                item["status"] = "pending"
                json_write(plan_path, plan)
                return
            meta.update(control="history_availability_only", base_state="history",
                        adapter_sha256=plan["adapter_sha256"], feature_rule=plan["feature_rule"],
                        evaluation_scope="validation only; standard CLI evaluation does not implement this adapter")
            json_write(result / "control_manifest.json", meta)
            (result / "manifest.json").unlink()
            item["status"] = "complete"
            json_write(plan_path, plan)
        plan["completed_utc"] = datetime.now(timezone.utc).isoformat()
        json_write(plan_path, plan)
        print("Six matched availability-only controls completed; test remains locked", flush=True)
    finally:
        training.Snapshots, training.new_run = original_snapshots, original_new_run


if __name__ == "__main__":
    main()
