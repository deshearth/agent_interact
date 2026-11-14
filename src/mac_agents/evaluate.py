from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from .common import code_digest, device_for, digest, json_write
from .data.features import Snapshots, history_features
from .data.prepare import load_prepared
from .metrics import score
from .models.baselines import baseline_logp
from .models.graph import make_graph
from .train import predict


def freeze(runs, output, rationale):
    output = Path(output)
    if output.exists():
        raise ValueError("Selection is immutable; use a distinct experiment file for a new study")
    selected = {}
    for run in runs:
        run = Path(run)
        meta = json.loads((run / "manifest.json").read_text())
        if meta["smoke"] or meta["stopping_reason"] == "wall_time_limit":
            raise ValueError(f"Preparation/unfinished runs are ineligible for final selection: {run}")
        if meta["code_sha256"] != code_digest():
            raise ValueError("Code changed after training; review and retrain before final selection")
        if list(run.glob("test_*")):
            raise ValueError("Test evaluation already exists; cannot reselect based on test outcomes")
        checkpoint = run / ("best.pt" if meta["kind"].startswith("A") else "baseline.json")
        selected[str(run.resolve())] = {"manifest_sha256": digest(run / "manifest.json"),
                                       "checkpoint_sha256": digest(checkpoint),
                                       "validation": json.loads((run / "validation_metrics.json").read_text())}
    record = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "rationale": rationale,
              "selection_source": "validation only", "runs": selected}
    json_write(output, record)
    print(f"Frozen {len(selected)} runs in {output}")
    return record


def check_selection(run, selection):
    if selection is None:
        raise ValueError("Controlled test evaluation requires --selection with a saved freeze record")
    record = json.loads(Path(selection).read_text())
    entry = record["runs"].get(str(Path(run).resolve()))
    if entry is None or entry["manifest_sha256"] != digest(Path(run) / "manifest.json"):
        raise ValueError("Run absent from selection or changed since freezing")
    meta = json.loads((Path(run) / "manifest.json").read_text())
    checkpoint = Path(run) / ("best.pt" if meta["kind"].startswith("A") else "baseline.json")
    if digest(checkpoint) != entry["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")


def rollout(model, s0, context, graph, cfg, steps, device, uniforms=None):
    """Only initial state/static data enter; no future observation is accepted."""
    state = cfg["state"]["condition"]
    if state not in ("opinion", "history"):
        raise ValueError(f"{state} has no autonomous state update")
    history = [np.asarray(s0, dtype=np.float32).copy()]
    if isinstance(model, torch.nn.Module):
        model.eval()
    c = torch.as_tensor(context, dtype=torch.float32, device=device)
    j = torch.as_tensor(graph, dtype=torch.float32, device=device)
    with torch.inference_mode():
        for t in range(steps):
            s = history[-1]
            if isinstance(model, torch.nn.Module):
                if state == "history":
                    past = np.stack(history, 1)
                    extra = np.stack([history_features(trajectory, t, cfg["state"]["history_window"])
                                      for trajectory in past])
                else:
                    extra = np.zeros((*s.shape, 0), dtype=np.float32)
                p = model(torch.as_tensor(s, device=device), c, j,
                          torch.as_tensor(extra, device=device)).softmax(-1).cpu().numpy()[..., 1]
            else:
                p = np.exp(baseline_logp(model, s, context, graph)[..., 1])
            next_state = np.where(p > .5, 1, -1) if uniforms is None else np.where(uniforms[:, t] < p, 1, -1)
            history.append(next_state.astype(np.float32))
    return np.stack(history[1:], 1).astype(np.int8)


def _switches(trajectory):
    return (trajectory[:, 1:] != trajectory[:, :-1]).sum(axis=1)


def _consensus_time(trajectory):
    unanimous = np.abs(trajectory.mean(-1)) == 1
    return np.where(unanimous.any(1), unanimous.argmax(1), trajectory.shape[1])


def evaluate(args):
    run = Path(args.run)
    meta = json.loads((run / "manifest.json").read_text())
    cfg = meta["config"]
    if args.partition == "test":
        check_selection(run, args.selection)
        if args.rollouts is not None and args.rollouts != cfg["evaluation"]["rollouts"]:
            raise ValueError("Test rollout count must match the frozen configuration")
    if meta["code_sha256"] != code_digest():
        raise ValueError("Evaluation code differs from the trained run; review before continuing")
    name = f"{args.partition}_{args.mode}"
    result_path = run / f"{name}_metrics.json"
    if args.partition == "test" and result_path.exists():
        print(result_path.read_text())
        return json.loads(result_path.read_text())
    arrays, index, directory = load_prepared(cfg)
    for key, path in (("episodes_sha256", "episodes.npz"), ("index_sha256", "index.json"), ("split_sha256", "splits.json")):
        if digest(directory / path) != meta["data_fingerprint"][key]:
            raise ValueError(f"Processed data changed: {path}")
    torch.set_num_threads(cfg["training"]["torch_threads"])
    device = device_for("auto")
    if meta["kind"].startswith("A"):
        if digest(run / "best.pt") != meta["best_sha256"]:
            raise ValueError("Best checkpoint checksum mismatch")
        model = make_graph(cfg, meta["state"]).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=False)["model"])
    else:
        model = json.loads((run / "baseline.json").read_text())
        device = torch.device("cpu")
    dataset = Snapshots(arrays, index, cfg, args.partition, meta["state"],
                        meta["limit_episodes"] if args.partition == "validation" else None, meta["match_state"])
    started = time.monotonic()
    if args.mode == "one_step":
        frame = predict(model, dataset, cfg, device)
        result, question = score(frame, meta["kind"])
        result["by_role"] = {}
        for keys, group in frame.groupby(["question_split", "graph_role", "graph_family"]):
            result["by_role"]["/".join(keys)] = score(group, meta["kind"])[0]
        frame.to_parquet(run / f"{name}_predictions.parquet", index=False)
        question.to_csv(run / f"{name}_by_question.csv", index=False)
    else:
        if not meta["autonomous"]:
            raise ValueError("Run requires observed evolving features and cannot roll out autonomously")
        episodes = [e for e in dataset.episode_ids if arrays["valid"][e].all()]
        if not episodes:
            raise ValueError("No complete valid trajectories for rollout")
        s0, phi, graph = arrays["spins"][episodes, 0], arrays["context"][episodes], arrays["J"][episodes]
        steps = arrays["spins"].shape[1]-1
        batch_size = cfg["training"]["batch_size"]
        def simulate(uniforms=None):
            return np.concatenate([rollout(model, s0[start:start+batch_size], phi[start:start+batch_size],
                graph[start:start+batch_size], cfg, steps, device,
                None if uniforms is None else uniforms[start:start+batch_size])
                for start in range(0, len(episodes), batch_size)])
        deterministic = simulate()
        truth = arrays["spins"][episodes, 1:]
        full_true = np.concatenate([s0[:, None], truth], axis=1)
        full_deterministic = np.concatenate([s0[:, None], deterministic], axis=1)
        draws = args.rollouts or cfg["evaluation"]["rollouts"]
        samples = []
        for draw in range(draws):
            uniforms = np.stack([np.random.default_rng(int.from_bytes(hashlib.sha256(
                f"{cfg['evaluation']['seed']}|{index[e]['episode_id']}|{draw}".encode()).digest()[:8], "big")).random((steps, s0.shape[-1]))
                for e in episodes])
            samples.append(simulate(uniforms))
        sampled = np.stack(samples)
        groups = {}
        for k, e in enumerate(episodes):
            groups.setdefault((index[e]["qid"], index[e]["graph_hash"]), []).append(k)
        condition_results = []
        for (qid, gh), indices in groups.items():
            ref = (truth[indices, -1] == 1).mean(-1)
            sim = (sampled[:, indices, -1] == 1).mean(-1).ravel()
            condition_results.append({"qid": qid, "graph_hash": gh, "reference_episodes": len(ref),
                                      "simulated_episodes": len(sim),
                                      "final_fraction_wasserstein": float(wasserstein_distance(ref, sim))})
        import pandas as pd
        condition_frame = pd.DataFrame(condition_results)
        q_wasserstein = condition_frame.groupby("qid").final_fraction_wasserstein.mean()
        full_sampled = np.concatenate([np.broadcast_to(s0, (draws, *s0.shape))[:, :, None], sampled], axis=2)
        flat_sampled = full_sampled.reshape(-1, steps+1, s0.shape[-1])
        result = {"episodes": len(episodes), "rollouts_per_initial_state": draws,
                  "horizon_accuracy": (deterministic == truth).mean(axis=(0, 2)).tolist(),
                  "horizon_group_mean_absolute_error": np.abs(deterministic.mean(-1)-truth.mean(-1)).mean(0).tolist(),
                  "question_mean_final_fraction_wasserstein": float(q_wasserstein.mean()),
                  "observed_mean_switches_per_agent": float(_switches(full_true).mean()),
                  "deterministic_mean_switches_per_agent": float(_switches(full_deterministic).mean()),
                  "sampled_mean_switches_per_agent": float(_switches(flat_sampled).mean()),
                  "consensus_criterion": "First unanimous state, including initialization; no event censored at horizon+1",
                  "observed_consensus_time_histogram": np.bincount(_consensus_time(full_true), minlength=steps+2).tolist(),
                  "sampled_consensus_time_histogram": np.bincount(_consensus_time(flat_sampled), minlength=steps+2).tolist(),
                  "limitation": "Few observed replicas per question/graph; extra Monte Carlo draws do not create more reference observations."}
        condition_frame.to_csv(run / f"{name}_conditions.csv", index=False)
        np.savez_compressed(run / f"{name}_trajectories.npz", deterministic=deterministic,
                            sampled=sampled, target=truth, initial=s0,
                            episode_ids=np.array([index[e]["episode_id"] for e in episodes]))
    result.update(actual_device=str(device), dtype=meta["dtype"], elapsed_seconds=time.monotonic()-started,
                  state_condition=meta["state"], forecast_boundary=cfg["state"]["forecast_boundary"],
                  preparation_run=meta["smoke"])
    json_write(result_path, result)
    print(json.dumps({"run": str(run), "evaluation": name, "result": result}, indent=2))
    return result
