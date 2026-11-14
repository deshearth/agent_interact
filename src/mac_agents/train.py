from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .common import code_digest, config, device_for, digest, environment, json_write, new_run, seed_all
from .data.features import Snapshots, to_device
from .data.prepare import load_prepared
from .metrics import prediction_frame, score
from .models.baselines import baseline_logp, fit_baseline
from .models.graph import make_graph


def predict(model, dataset, cfg, device):
    logs, episodes, rounds = [], [], []
    if isinstance(model, torch.nn.Module):
        model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(0)):
            if isinstance(model, torch.nn.Module):
                b = to_device(batch, device)
                logp = F.log_softmax(model(b["s"], b["context"], b["J"], b["extra"]), -1).cpu().numpy()
            else:
                logp = baseline_logp(model, batch["s"].numpy(), batch["context"].numpy(), batch["J"].numpy())
            logs.append(logp)
            episodes.extend(batch["episode"].tolist())
            rounds.extend(batch["round"].tolist())
    return prediction_frame(np.concatenate(logs), episodes, rounds, dataset.arrays, dataset.index, cfg["model"]["kind"])


def save_checkpoint(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def rng_state(device="cpu"):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "mps": torch.mps.get_rng_state() if str(device) == "mps" else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["mps"] is not None:
        if not torch.backends.mps.is_available():
            raise ValueError("Checkpoint requires MPS RNG; exact resume needs the original backend")
        torch.mps.set_rng_state(state["mps"].cpu())


def train(args):
    if args.resume:
        output = Path(args.resume)
        manifest = json.loads((output / "manifest.json").read_text())
        cfg = copy.deepcopy(manifest["config"])
        if manifest["code_sha256"] != code_digest():
            raise ValueError("Code changed since checkpoint; start a new run instead of silently resuming")
        if args.model and args.model != cfg["model"]["kind"] or args.state and args.state != cfg["state"]["condition"]:
            raise ValueError("Resume cannot change architecture or state")
        if args.seed is not None and args.seed != cfg["seed"]:
            raise ValueError("Resume cannot change seed")
        if args.device and str(device_for(args.device)) != manifest["actual_device"]:
            raise ValueError("Exact resume requires the original device")
        limit, match = manifest["limit_episodes"], manifest["match_state"]
    else:
        cfg = config(args.config)
        if args.model:
            cfg["model"]["kind"] = args.model
        if args.state:
            cfg["state"]["condition"] = args.state
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.device:
            cfg["device"] = args.device
        limit, match = args.limit_episodes, args.match_state
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.wall_minutes is not None:
        cfg["training"]["wall_minutes"] = args.wall_minutes
    if cfg["training"]["max_steps"] <= 0 or cfg["training"]["wall_minutes"] <= 0:
        raise ValueError("Training caps must be positive")
    torch.set_num_threads(cfg["training"]["torch_threads"])
    device = device_for(cfg["device"])
    seed_all(cfg["seed"])
    arrays, index, directory = load_prepared(cfg)
    state, kind = cfg["state"]["condition"], cfg["model"]["kind"]
    if kind in ("P0", "P1", "I0") and state != "opinion":
        raise ValueError("Original baselines are defined on opinion state")
    training = Snapshots(arrays, index, cfg, "train", state, limit, match)
    validation = Snapshots(arrays, index, cfg, "validation", state, limit, match)
    is_smoke = manifest["smoke"] if args.resume else bool(args.smoke or limit is not None)
    if not is_smoke and (training.incomplete_messages or validation.incomplete_messages):
        raise ValueError("Message cache is incomplete; encode the development partition or explicitly request a smoke run")
    fingerprint = {"split_sha256": digest(directory / "splits.json"),
                   "episodes_sha256": digest(directory / "episodes.npz"),
                   "index_sha256": digest(directory / "index.json"),
                   "train_pairs_sha256": hashlib.sha256(json.dumps(training.pairs).encode()).hexdigest(),
                   "validation_pairs_sha256": hashlib.sha256(json.dumps(validation.pairs).encode()).hexdigest()}
    if state == "existing_messages" or match == "existing_messages":
        from .data.features import message_directory
        message_root = message_directory(cfg)
        cache_files = [message_root / f"{index[e]['episode_id']}.npz"
                       for e in sorted(set(training.episode_ids + validation.episode_ids))]
        fingerprint["message_metadata_sha256"] = digest(message_root / "metadata.json")
        fingerprint["message_features_sha256"] = hashlib.sha256(
            json.dumps([(p.name, digest(p)) for p in cache_files]).encode()).hexdigest()
    if args.resume:
        if fingerprint != manifest["data_fingerprint"]:
            raise ValueError("Data, splits, or matched examples changed since checkpoint")
        if str(device) != manifest["actual_device"]:
            raise ValueError("Original backend unavailable; exact resume refused")
        manifest.setdefault("resume_caps", []).append({"max_steps": cfg["training"]["max_steps"], "wall_minutes": cfg["training"]["wall_minutes"]})
    else:
        output = new_run(f"{kind}-{state}-s{cfg['seed']}")
        manifest = {"config": cfg, "kind": kind, "state": state, "seed": cfg["seed"],
                    "code_sha256": code_digest(), "data_fingerprint": fingerprint,
                    "dependency_lock_sha256": digest("requirements-lock.txt") if Path("requirements-lock.txt").exists() else None,
                    "actual_device": str(device), "dtype": "float32", "environment": environment(),
                    "smoke": bool(args.smoke or limit is not None), "limit_episodes": limit, "match_state": match,
                    "train_snapshots": len(training), "validation_snapshots": len(validation),
                    "train_questions": sorted({index[e]["qid"] for e, _ in training.pairs}),
                    "validation_questions": sorted({index[e]["qid"] for e, _ in validation.pairs}),
                    "autonomous": state in ("opinion", "history"), "status": "running",
                    "protocol_deviations": ["Bounded preparation run; excluded from final selection"] if args.smoke or limit is not None else []}
    json_write(output / "manifest.json", manifest)
    started = time.monotonic()
    if kind in ("P0", "P1", "I0"):
        if args.resume:
            raise ValueError("Analytic/author baseline fits are restarted, not neural-checkpoint resumed")
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=cfg["training"]["torch_threads"]):
            model = fit_baseline(kind, training, cfg)
        json_write(output / "baseline.json", model)
        frame = predict(model, validation, cfg, torch.device("cpu"))
        summary, question = score(frame, kind)
        manifest.update(parameters=model["parameters"], status="complete", stopping_reason="baseline_fitted", actual_device="cpu", dtype="float64")
    else:
        model = make_graph(cfg, state).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                     weight_decay=cfg["training"]["weight_decay"])
        epoch, position, step, bad_epochs, best = 0, 0, 0, 0, float("inf")
        history, prior_seconds = [], 0.0
        manifest["parameters"] = sum(p.numel() for p in model.parameters())
        if args.resume:
            if digest(output / "checkpoint.pt") != manifest["checkpoint_sha256"]:
                raise ValueError("Checkpoint checksum mismatch")
            ck = torch.load(output / "checkpoint.pt", map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            epoch, position, step = ck["epoch"], ck["position"], ck["step"]
            bad_epochs, best, history = ck["bad_epochs"], ck["best"], ck["history"]
            prior_seconds = ck["elapsed_seconds"]
            restore_rng(ck["rng"])

        def checkpoint():
            save_checkpoint(output / "checkpoint.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "position": position, "step": step, "bad_epochs": bad_epochs, "best": best,
                "history": history, "rng": rng_state(device), "elapsed_seconds": prior_seconds + time.monotonic()-started})
            manifest["checkpoint_sha256"] = digest(output / "checkpoint.pt")
            json_write(output / "manifest.json", manifest)

        def validate():
            nonlocal best, bad_epochs
            frame = predict(model, validation, cfg, device)
            metrics, _ = score(frame, kind)
            value = metrics["nll"]
            history.append({"epoch": epoch, "step": step, "validation_nll": value})
            if value < best - cfg["training"]["min_delta"]:
                best, bad_epochs = value, 0
                save_checkpoint(output / "best.pt", {"model": model.state_dict(), "step": step, "validation_nll": value})
            else:
                bad_epochs += 1
            return metrics

        reason = "epoch_limit"
        while epoch < cfg["training"]["epochs"]:
            order = np.random.default_rng(cfg["seed"] + epoch).permutation(len(training))
            loader = DataLoader(Subset(training, order[position:].tolist()),
                                batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(cfg["seed"] + epoch))
            interrupted = False
            for batch in loader:
                if step >= cfg["training"]["max_steps"]:
                    reason, interrupted = "step_limit", True
                    break
                if time.monotonic()-started >= cfg["training"]["wall_minutes"]*60:
                    reason, interrupted = "wall_time_limit", True
                    break
                model.train()
                batch = to_device(batch, device)
                logits = model(batch["s"], batch["context"], batch["J"], batch["extra"])
                loss = F.cross_entropy(logits.reshape(-1, 2), batch["target"].reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
                optimizer.step()
                step += 1
                position += len(batch["s"])
                if step % 50 == 0:
                    checkpoint()
            if position == len(training):
                epoch += 1
                position = 0
                validate()
                checkpoint()
                if bad_epochs >= cfg["training"]["patience"]:
                    reason = "early_stopping"
                    break
            if interrupted:
                break
        # Interim validation is RNG-neutral; early-stopping patience counts full epochs only.
        if not history or history[-1]["step"] != step:
            previous_bad = bad_epochs
            validate()
            bad_epochs = previous_bad
        checkpoint()
        best_checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
        frame = predict(model, validation, cfg, device)
        summary, question = score(frame, kind)
        json_write(output / "history.json", history)
        manifest.update(steps=step, epochs=epoch, status="bounded" if reason in ("step_limit", "wall_time_limit") else "complete",
                        stopping_reason=reason, best_step=best_checkpoint["step"],
                        best_sha256=digest(output / "best.pt"))
    frame.to_parquet(output / "validation_predictions.parquet", index=False)
    question.to_csv(output / "validation_by_question.csv", index=False)
    json_write(output / "validation_metrics.json", summary)
    manifest["elapsed_seconds_this_session"] = time.monotonic()-started
    manifest["config"] = cfg
    json_write(output / "manifest.json", manifest)
    print(json.dumps({"run": str(output), "kind": kind, "state": state, "device": manifest["actual_device"],
                      "parameters": manifest["parameters"], "validation": summary,
                      "stopping_reason": manifest["stopping_reason"]}, indent=2))
    return output
