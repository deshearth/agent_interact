"""Short CPU/MPS correctness and compute check on one real training batch."""
import copy
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from mac_agents.common import config, json_write, seed_all
from mac_agents.data.features import Snapshots, to_device
from mac_agents.data.prepare import load_prepared
from mac_agents.models.graph import make_graph

cfg = config()
torch.set_num_threads(cfg["training"]["torch_threads"])
seed_all(42)
arrays, index, _ = load_prepared(cfg)
dataset = Snapshots(arrays, index, cfg, "train", "opinion")
batch = next(iter(DataLoader(dataset, batch_size=32)))
original = make_graph(cfg, "opinion")
devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
results, reference = {}, None
for device in devices:
    model = copy.deepcopy(original).to(device)
    b = to_device(batch, device)
    model.eval()
    with torch.no_grad():
        logits = model(b["s"], b["context"], b["J"], b["extra"]).cpu()
    if reference is None:
        reference = logits
    else:
        torch.testing.assert_close(logits, reference, atol=2e-5, rtol=2e-4)
    model.train()
    def step():
        model.zero_grad(set_to_none=True)
        outputs = model(b["s"], b["context"], b["J"], b["extra"])
        loss = torch.nn.functional.cross_entropy(outputs.reshape(-1, 2), b["target"].reshape(-1))
        loss.backward()
    for _ in range(3):
        step()
    if device == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    for _ in range(20):
        step()
    if device == "mps":
        torch.mps.synchronize()
    results[device] = {"mean_forward_backward_seconds": (time.perf_counter()-started)/20,
                       "max_absolute_logit_difference_from_cpu": float((logits-reference).abs().max()),
                       "dtype": "float32", "batch_snapshots": 32, "agents_per_snapshot": 32,
                       "parameters": sum(p.numel() for p in model.parameters())}
json_write("reports/predictor_benchmark.json", results)
Path("reports/predictor_benchmark.md").write_text(
    "# Predictor microbenchmark\n\nReal training batch; A2/opinion; 3 warm-up and 20 timed forward/backward steps.\n\n"
    "Timing excludes the optimizer, data loading, and text encoding. This is not an LLM speedup measurement.\n\n" +
    "\n".join(f"- {device}: {record['mean_forward_backward_seconds']:.6f} seconds/step; maximum CPU logit difference {record['max_absolute_logit_difference_from_cpu']:.3g}."
              for device, record in results.items()) + "\n")
print(results)
