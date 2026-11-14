"""Exercise each runnable model/state on small real development subsets."""
from argparse import Namespace
import json
from pathlib import Path

from mac_agents.common import json_write
from mac_agents.train import train
from mac_agents.evaluate import evaluate

matrix = [("P0", "opinion"), ("P1", "opinion"), ("I0", "opinion"),
          ("A0", "opinion"), ("A1", "opinion"), ("A2", "opinion"),
          ("A2", "vote_counts"), ("A2", "history"), ("A2", "existing_messages"),
          ("A0", "vote_counts"), ("A0", "history"), ("A0", "existing_messages")]
results = []
for kind, state in matrix:
    run = train(Namespace(config="configs/paper_binary.yaml", model=kind, state=state, seed=42,
                          device="cpu", max_steps=4, wall_minutes=2, limit_episodes=2,
                          match_state=None, resume=None, smoke=True))
    metrics = json.loads((run / "validation_metrics.json").read_text())
    results.append({"kind": kind, "state": state, "run": str(run), "validation_nll": metrics["nll"]})
    if kind == "A2" and state in ("opinion", "history"):
        evaluate(Namespace(run=run, partition="validation", mode="closed_loop", selection=None, rollouts=2))
json_write("reports/preparation_runs.json", results)
print("Completed", len(results), "preparation runs. These runs cannot be selected for test evaluation.")
