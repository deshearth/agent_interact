"""Prepare and run the development-only argument response diagnostic.

Separate implementation namespace: the completed opinion/history study is not
changed. There is deliberately no test-set or autonomous rollout command here.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, config, device_for, digest, environment, json_write, seed_all
from mac_agents.data.features import to_device
from mac_agents.metrics import prediction_frame, score
from mac_agents.train import rng_state, restore_rng, save_checkpoint
from scripts.argument_features import ResponseSnapshots, load_cache, prepare
from scripts.argument_model import ResponseUpdate

CONFIG = "configs/argument_response.yaml"
ROOT = Path("runs/argument_response_v1")
CONDITIONS = ("stance_only", "full_arguments", "shuffled_arguments")
CORE_SHA = "92c500581c6d74e0d5effa4b89f9c402bb76f7800151ac6dbb3b3afcae939dda"


def implementation():
    if code_digest() != CORE_SHA:
        raise ValueError("The completed core study must remain unchanged")
    return {"core_sha256": code_digest(), **{name: digest(Path(__file__).with_name(name))
            for name in ("argument_features.py", "argument_model.py", "argument_response.py")}}


def implementation_digest():
    return hashlib.sha256(json.dumps(implementation(), sort_keys=True).encode()).hexdigest()


def make_model(cfg, kind):
    model = ResponseUpdate(kind, text_dim=cfg["data"]["encoder_dimension"],
                           **{k: v for k, v in cfg["model"].items() if k != "max_parameters"})
    if sum(p.numel() for p in model.parameters()) >= cfg["model"]["max_parameters"]:
        raise ValueError("Predictor exceeds parameter budget")
    return model


def cohort(cfg):
    return [{"kind": kind, "condition": condition, "seed": seed}
            for seed in cfg["training"]["seeds"] for kind in ("A0", "A2") for condition in CONDITIONS]


def ensure_protocol(cfg, metadata):
    ROOT.mkdir(parents=True, exist_ok=True)
    fixed = {"implementation": implementation(), "config_sha256": digest(CONFIG),
             "features_sha256": digest(Path(cfg["data"]["cache"]) / "metadata.json"),
             "cohort": cohort(cfg), "config": cfg}
    path = ROOT / "protocol.json"
    if path.exists():
        protocol = json.loads(path.read_text())
        if any(protocol[key] != value for key, value in fixed.items()):
            raise ValueError("Protocol, code, config or data changed; use a new study namespace")
    else:
        protocol = {**fixed, "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Does individual delivered message content add prediction beyond communicated stances, opinions, two-snapshot history and context?",
            "scope": "Exploratory development only: 16 fitting questions, 4 validation/early-stopping questions. These questions have been used in prior research. No fresh confirmatory assessment.",
            "forecast_boundary": "After messages_history[t] is delivered, before s[t+1] is chosen; all conditions predict the same receiver updates.",
            "primary_contrast": "A2/full_arguments minus A2/stance_only",
            "secondary_contrasts": ["A2/full_arguments minus A2/shuffled_arguments",
                                    "A0/full_arguments minus A0/stance_only",
                                    "A0/full_arguments minus A0/shuffled_arguments"],
            "metrics": "Question-weighted NLL primary; accuracy, two-class Brier, FCBA, flip/stay accuracy alongside. Report all seeds and question/round breakdowns.",
            "uncertainty": "Average seeds within question; 1000 paired question-bootstrap draws. Pointwise descriptive intervals on four selection questions, not confirmatory inference or independent transition samples.",
            "architecture_control": "Same architecture, parameters, initial weights and training budget within kind/seed; stance-only zeros text input. Text projection is inactive in stance-only, so equal nominal parameters do not imply equal effective capacity.",
            "inputs": "Receiver/sender current opinion, two-snapshot local histories, training-only-PCA persona/question context, degree, existing prescribed relationship, explicit message stance, and conditional individual frozen text embeddings. No arbitrary node ID embeddings or answer keys.",
            "shuffle": metadata["shuffle_note"], "shuffle_matching_fields": metadata["shuffle_matching_fields"],
            "shuffle_fit": "Train and validation shuffled independently within partition, question and matching strata. Each model seed has its own fixed bijection. Also evaluate full-text models under that substitution without retraining.",
            "shuffle_limitation": "Matching preserves observed stance/source/receiver cues, not all sender history or graph context. Unmatched messages retain their text; report matching and actual-change coverage. This is a predictive dependence diagnostic, not a causal argument intervention.",
            "content_limitation": "Full text includes reasoning, phrasing, numbers, length and other linguistic cues. Any gain concerns extra message content; it does not isolate logical reasoning. Frozen generic embeddings may miss important mathematical details.",
            "autonomous": False, "test_policy": "No test scoring. The prior controlled test outcomes have been inspected; fresh questions/outcomes are required for a future confirmatory assessment.",
            "graph_policy": "Preserve the source communication graph and prescribed signs. Emergent mutual credibility remains deferred."}
        json_write(path, protocol)
    return protocol


def predictions(model, dataset, cfg, device):
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False,
                                generator=torch.Generator().manual_seed(0), num_workers=0):
            logp = F.log_softmax(model(to_device(batch, device)), -1)
            values.append(logp.cpu().numpy())
    return np.concatenate(values)


def question_nll(logp, dataset):
    pairs = np.asarray(dataset.pairs)
    target = (dataset.arrays["spins"][pairs[:, 0], pairs[:, 1]+1] == 1).astype(np.int64)
    losses = -np.take_along_axis(logp, target[..., None], axis=-1)[..., 0].mean(-1)
    qids = np.array([dataset.index[e]["qid"] for e, _ in dataset.pairs])
    return float(np.mean([losses[qids == qid].mean() for qid in sorted(set(qids))]))


def save_predictions(output, name, model, dataset, cfg, device):
    logp = predictions(model, dataset, cfg, device)
    episodes, rounds = zip(*dataset.pairs)
    frame = prediction_frame(logp, episodes, rounds, dataset.arrays, dataset.index, "A2")
    frame["has_matched_substitution"] = dataset.support[np.array(episodes), np.array(rounds)].reshape(-1)
    frame["has_different_text"] = dataset.changed[np.array(episodes), np.array(rounds)].reshape(-1)
    frame.to_parquet(output / f"{name}_predictions.parquet", index=False)
    metrics, by_question = score(frame)
    json_write(output / f"{name}_metrics.json", metrics)
    by_question.to_csv(output / f"{name}_by_question.csv", index=False)
    return metrics


def train_one(item, cfg, cache, device, stop_at, smoke=False):
    arrays, index, vectors, metadata = cache
    kind, condition, seed = item["kind"], item["condition"], item["seed"]
    run = (ROOT / "smoke" if smoke else ROOT) / f"{kind}-{condition}-s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    identity = {**item, "implementation_sha256": implementation_digest(), "config_sha256": digest(CONFIG),
                "features_sha256": digest(Path(cfg["data"]["cache"]) / "metadata.json"),
                "protocol_sha256": digest(ROOT / "protocol.json"), "actual_device": str(device), "smoke": smoke}
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None:
        if any(manifest[key] != value for key, value in identity.items()):
            raise ValueError("Run identity changed; refusing reuse/resume")
        if manifest["status"] == "complete":
            for name, expected in manifest["artifact_sha256"].items():
                if digest(run / name) != expected:
                    raise ValueError(f"Completed run artifact changed: {name}")
            return manifest
    donors = np.load(Path(cfg["data"]["cache"]) / f"donors_s{seed}.npy", allow_pickle=False)
    train = ResponseSnapshots(arrays, index, vectors, "train", condition, donors)
    validation = ResponseSnapshots(arrays, index, vectors, "validation", condition, donors)
    seed_all(seed)
    model = make_model(cfg, kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                 weight_decay=cfg["training"]["weight_decay"])
    history, epoch, position, step, bad_epochs, best = [], 0, 0, 0, 0, float("inf")
    elapsed_before = 0.
    if manifest is not None:
        if not (run / "checkpoint.pt").exists():
            raise ValueError("Interrupted before first checkpoint; retain this run and use a new namespace")
        ck = torch.load(run / "checkpoint.pt", map_location=device, weights_only=False)
        if ck["identity"] != identity:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        history, epoch, position, step = ck["history"], ck["epoch"], ck["position"], ck["step"]
        bad_epochs, best, elapsed_before = ck["bad_epochs"], ck["best"], ck["elapsed_seconds"]
        restore_rng(ck["rng"])
    else:
        manifest = {**identity, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "inactive_text_parameters_in_stance_only": sum(p.numel() for p in model.text_projection.parameters()) if condition == "stance_only" else 0,
                    "frozen_encoder_parameters": metadata["encoder"]["frozen_parameters"],
                    "config": cfg, "dtype": "float32", "environment": environment(),
                    "train_snapshots": len(train), "validation_snapshots": len(validation),
                    "train_questions": metadata["questions"]["train"], "validation_questions": metadata["questions"]["validation"],
                    "forecast_boundary": "after_messages_before_next_opinion", "autonomous": False}
    json_write(manifest_path, manifest)
    started = time.monotonic()

    def checkpoint():
        save_checkpoint(run / "checkpoint.pt", {"identity": identity, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": rng_state(device), "epoch": epoch,
            "position": position, "step": step, "bad_epochs": bad_epochs, "best": best, "history": history,
            "elapsed_seconds": elapsed_before+time.monotonic()-started})

    def validate():
        nonlocal best, bad_epochs
        nll = question_nll(predictions(model, validation, cfg, device), validation)
        if not np.isfinite(nll):
            raise FloatingPointError("Nonfinite validation score")
        history.append({"epoch": epoch, "step": step, "validation_nll": nll})
        if nll < best - cfg["training"]["min_delta"]:
            best, bad_epochs = nll, 0
            save_checkpoint(run / "best.pt", {"model": model.state_dict(), "step": step, "validation_nll": nll})
        else:
            bad_epochs += 1

    checkpoint()
    reason = "epoch_limit"
    max_steps = 2 if smoke else cfg["training"]["max_steps"]
    while epoch < cfg["training"]["epochs"]:
        order = np.random.default_rng(seed+epoch).permutation(len(train))
        loader = DataLoader(Subset(train, order[position:].tolist()), batch_size=cfg["training"]["batch_size"],
                            generator=torch.Generator().manual_seed(seed+epoch), num_workers=0)
        for batch in loader:
            if step >= max_steps or time.monotonic() >= stop_at:
                reason = "step_limit" if step >= max_steps else "wall_time_limit"
                break
            model.train()
            batch = to_device(batch, device)
            loss = F.cross_entropy(model(batch).reshape(-1, 2), batch["target"].reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            optimizer.step()
            step += 1
            position += len(batch["s"])
            if step % 100 == 0:
                checkpoint()
        if position == len(train):
            epoch += 1
            position = 0
            validate()
            checkpoint()
            print(json.dumps({"run": run.name, **history[-1]}), flush=True)
            if bad_epochs >= cfg["training"]["patience"]:
                reason = "early_stopping"
                break
        if reason in ("step_limit", "wall_time_limit"):
            break
    if reason == "wall_time_limit":
        checkpoint()
        manifest.update(status="bounded", stopping_reason=reason, steps=step,
                        elapsed_seconds=elapsed_before+time.monotonic()-started,
                        checkpoint_sha256=digest(run / "checkpoint.pt"))
        json_write(manifest_path, manifest)
        return manifest
    if not history or history[-1]["step"] != step:
        validate()
    checkpoint()
    selected = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    metrics = save_predictions(run, "validation", model, validation, cfg, device)
    if condition == "full_arguments":
        perturbed = ResponseSnapshots(arrays, index, vectors, "validation", "shuffled_arguments", donors)
        save_predictions(run, "validation_substitution", model, perturbed, cfg, device)
    json_write(run / "history.json", history)
    artifact_names = ["checkpoint.pt", "best.pt", "history.json", "validation_predictions.parquet", "validation_metrics.json"]
    if condition == "full_arguments":
        artifact_names.extend(["validation_substitution_predictions.parquet", "validation_substitution_metrics.json"])
    manifest.update(status="complete", stopping_reason=reason, epochs=epoch, steps=step, best_step=selected["step"],
                    elapsed_seconds=elapsed_before+time.monotonic()-started,
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    artifact_sha256={name: digest(run / name) for name in artifact_names})
    json_write(manifest_path, manifest)
    print(json.dumps({"run": str(run), "status": "complete", "reason": reason,
                      "seconds": manifest["elapsed_seconds"], "metrics": metrics}), flush=True)
    return manifest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "smoke", "validation"])
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = parser.parse_args()
    cfg = config(CONFIG)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    device = device_for(args.device)
    if args.stage == "prepare":
        prepare(cfg, device)
        return
    cache = load_cache(cfg, verify=True)
    ensure_protocol(cfg, cache[-1])
    if args.stage == "smoke":
        train_one({"kind": "A2", "condition": "full_arguments", "seed": 42}, cfg, cache,
                  device, time.monotonic()+300, smoke=True)
        return
    plan_path = ROOT / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["protocol_sha256"] != digest(ROOT / "protocol.json"):
            raise ValueError("Plan protocol changed")
    else:
        plan = {"protocol_sha256": digest(ROOT / "protocol.json"), "elapsed_seconds": 0., "runs": []}
        json_write(plan_path, plan)
    started = time.monotonic()
    stop_at = started + cfg["training"]["aggregate_minutes"]*60 - plan["elapsed_seconds"]
    try:
        for item in cohort(cfg):
            if time.monotonic() >= stop_at:
                print("Aggregate budget reached; checkpoints retained", flush=True)
                break
            manifest = train_one(item, cfg, cache, device, stop_at)
            row = {**item, "run": str(ROOT / f"{item['kind']}-{item['condition']}-s{item['seed']}"),
                   "status": manifest["status"]}
            plan["runs"] = [r for r in plan["runs"] if (r["kind"], r["condition"], r["seed"]) !=
                            (item["kind"], item["condition"], item["seed"])] + [row]
            json_write(plan_path, plan)
            if manifest["status"] != "complete":
                break
        plan["complete"] = len(plan["runs"]) == len(cohort(cfg)) and all(r["status"] == "complete" for r in plan["runs"])
        if plan["complete"]:
            plan["completed_utc"] = datetime.now(timezone.utc).isoformat()
    finally:
        plan["elapsed_seconds"] += time.monotonic()-started
        json_write(plan_path, plan)


if __name__ == "__main__":
    main()
