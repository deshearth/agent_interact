"""Training-only capacity diagnostics of the frozen GPT-data comparison.

Use the exact original GraphUpdate, with dropout disabled. No test predictions,
new source responses, encoder training or modifications to prior experiments.
"""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, config, digest, environment, json_write, seed_all
from mac_agents.data.features import Snapshots
from mac_agents.data.prepare import load_prepared
from mac_agents.models.baselines import baseline_logp
from mac_agents.models.graph import make_graph
from mac_agents.train import save_checkpoint
from scripts.inductive_study import study_code_digest

CONFIG = Path("configs/ising_transformer_capacity.yaml")
CORE_SHA = "92c500581c6d74e0d5effa4b89f9c402bb76f7800151ac6dbb3b3afcae939dda"


def now():
    return datetime.now(timezone.utc).isoformat()


def model_sha(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    return -p*np.log(p.clip(1e-300))-(1-p)*np.log((1-p).clip(1e-300))


def probability_metrics(logp, observed, teacher):
    logp = np.asarray(logp, dtype=np.float64)
    p = np.exp(logp[..., 1])
    pred = logp.argmax(-1)
    nll = -np.where(observed == 1, logp[..., 1], logp[..., 0])
    teacher_ce = -(teacher*logp[..., 1]+(1-teacher)*logp[..., 0])
    return {"observed_error": float(np.mean(pred != observed)),
            "observed_nll": float(nll.mean()),
            "observed_brier_two_class": float(2*np.mean((p-observed)**2)),
            "teacher_cross_entropy": float(teacher_ce.mean()),
            "teacher_entropy": float(entropy(teacher).mean()),
            "teacher_kl": float((teacher_ce-entropy(teacher)).mean()),
            "teacher_probability_mae": float(np.abs(p-teacher).mean()),
            "teacher_probability_rmse": float(np.sqrt(np.mean((p-teacher)**2))),
            "teacher_probability_p99_error": float(np.quantile(np.abs(p-teacher), .99)),
            "teacher_label_disagreement": float(np.mean(pred != (teacher > .5))),
            "transitions": int(observed.size)}


def soft_cross_entropy(logits, targets):
    logp = logits.log_softmax(-1)
    return -(targets*logp[..., 1]+(1-targets)*logp[..., 0]).mean()


def snapshot_arrays(arrays, index, dataset, baseline):
    pairs = np.asarray(dataset.pairs, dtype=np.int64)
    e, t = pairs.T
    # Source target opinions are never passed to baseline_logp or the model.
    result = {"s": arrays["spins"][e, t].astype(np.float32),
              "context": arrays["context"][e].astype(np.float32),
              "J": arrays["J"][e].astype(np.float32),
              "observed": (arrays["spins"][e, t+1] == 1).astype(np.int64),
              "pairs": pairs}
    lp = baseline_logp(baseline, result["s"], result["context"], result["J"])
    result["teacher_logp"] = lp
    result["teacher"] = np.exp(lp[..., 1])
    rows = [{"episode": int(a), "round": int(b), "qid": index[a]["qid"],
             "episode_id": index[a]["episode_id"]} for a, b in pairs]
    return result, rows


def structural_features(s, context, J):
    state = np.stack([s == -1, s == 1], -1).astype(np.float32)
    structure = np.log1p(np.stack([(J > 0).sum(-1), (J < 0).sum(-1),
                                   np.abs(J).sum(-1)], -1).astype(np.float32))
    return np.concatenate([state, context, structure], -1)


def receiver_key(features, J, receiver):
    """A conservative equality key for the actual one-hop, order-free inputs.

    No graph IDs, agent IDs, question IDs, outcomes, or unobserved history.
    Equal keys imply equal outputs for every setting of this model's weights.
    Distinct keys may still be indistinguishable to this architecture.
    """
    f = np.asarray(features, dtype=np.float32).copy()
    f[f == 0] = 0.  # Canonicalize negative zero.
    neighbors = np.flatnonzero(J[receiver])
    edges = sorted((np.float32(J[receiver, j]).tobytes()+f[j].tobytes()) for j in neighbors)
    return (f[receiver].tobytes(), tuple(edges))


def group_bounds(counts):
    values = np.asarray(counts, dtype=np.int64)
    sizes = values.sum(-1)
    p = values[:, 1]/sizes
    return {"groups": len(values), "transitions": int(sizes.sum()),
            "conflicting_groups": int((values.min(-1) > 0).sum()),
            "unavoidable_errors": int(values.min(-1).sum()),
            "classification_error_lower_bound": float(values.min(-1).sum()/sizes.sum()),
            "cross_entropy_lower_bound": float((sizes*entropy(p)).sum()/sizes.sum())}


def audit_collisions(arrays):
    f = structural_features(arrays["s"], arrays["context"], arrays["J"])
    groups, counts = {}, []
    assignment = np.empty_like(arrays["observed"], dtype=np.int32)
    for row in range(len(f)):
        for receiver in range(f.shape[1]):
            key = receiver_key(f[row], arrays["J"][row], receiver)
            if key not in groups:
                groups[key] = len(groups)
                counts.append([0, 0])
            g = groups[key]
            assignment[row, receiver] = g
            counts[g][int(arrays["observed"][row, receiver])] += 1
    return assignment, np.asarray(counts), group_bounds(counts)


def references(cfg):
    assert code_digest() == CORE_SHA, "Frozen core changed"
    study = Path(cfg["reference_study"])
    selected = json.loads((study/"selection.json").read_text())
    result, baseline = {}, None
    for directory, entry in selected["runs"].items():
        run = Path(directory)
        assert digest(run/"manifest.json") == entry["manifest_sha256"]
        m = json.loads((run/"manifest.json").read_text())
        assert m["code_sha256"] == study_code_digest()
        p = run/("best.pt" if m["kind"].startswith("A") else "baseline.json")
        assert digest(p) == entry["checkpoint_sha256"]
        # Hash existing evaluations without re-running or selecting on them.
        for artifact in run.iterdir():
            if artifact.is_file():
                result[str(artifact)] = digest(artifact)
        if m["kind"] == "I0" and m["state"] == "opinion":
            assert baseline is None
            baseline = (m, json.loads(p.read_text()), str(p))
    assert baseline is not None
    return result, baseline


def prepare(cfg):
    root = Path(cfg["output"])
    if (root/"prepared.json").exists():
        validate_prepared(cfg)
        print(json.dumps({"prepared": True, "reused": True}))
        return
    root.mkdir(parents=True, exist_ok=True)
    old, (meta, baseline, baseline_path) = references(cfg)
    arrays, index, directory = load_prepared(meta["config"])
    for key, name in (("episodes_sha256", "episodes.npz"), ("index_sha256", "index.json"), ("split_sha256", "splits.json")):
        assert digest(directory/name) == meta["data_fingerprint"][key]
    training = Snapshots(arrays, index, meta["config"], "train", "opinion")
    assert len(training) == 4096
    a, rows = snapshot_arrays(arrays, index, training, baseline)
    qids, qcounts = np.unique([r["qid"] for r in rows], return_counts=True)
    assert len(qids) == 16 and (qcounts == 256).all()
    model_cfg = copy.deepcopy(config(cfg["reference_config"]))
    model_cfg["model"]["kind"] = "A2"
    model_cfg["model"]["dropout"] = cfg["training"]["dropout"]
    identity = {"config": cfg, "config_sha256": digest(CONFIG), "code_sha256": digest(__file__),
        "core_sha256": code_digest(), "study_implementation_sha256": study_code_digest(),
        "selection_sha256": digest(Path(cfg["reference_study"])/"selection.json"),
        "original_model_config": meta["config"], "model_config": model_cfg,
        "source_data": meta["data_fingerprint"], "baseline_path": baseline_path,
        "baseline_sha256": digest(baseline_path), "train_questions": qids.tolist(),
        "reference_comparison_sha256": digest(cfg["reference_comparison"]),
        "reference_followup_sha256": digest(cfg["reference_followup"])}
    protocol = {**identity, "created_utc": now(),
        "primary_stages": ["imitation", "observed"], "primary_fits": 18,
        "targets": {"imitation": "Frozen fitted Ising P(A) on identical observed input states, not resampled labels. Cross-entropy minus teacher entropy is KL.",
                    "observed": "Original next majority-of-five opinion, exactly the prior GPT-data training target."},
        "inputs": "Original current opinions, 16-dimensional persona/question context, signed graph and degrees. No history/messages, new encoder, answer keys or learnable agent/question IDs.",
        "model": "Unmodified original GraphUpdate A2, 135458 parameters, dropout=0. Same initial weights within seed across rates and primary targets.",
        "optimization": "AdamW, zero weight decay, 128 epochs (16384 updates) per fit, deterministic epoch shuffles, three initial learning rates. Cosine decay to 1% of initial rate. No validation stopping. Initial, epoch1,2, then every4 epochs evaluated on full training set. Best training objective checkpoint and lowest observed training-error checkpoint retained separately.",
        "plateau": "Descriptive: best training objective improvement in final32 epochs <0.0001. No claim of global capacity optimum if this holds, and explicitly report budget-limited nonplateau runs.",
        "control_criterion": "Ising imitation considered close on the measured input states if training probability MAE<=0.005 and KL<=0.0005 nats. Report continuous errors and seed/rate variation, not only pass/fail.",
        "collision_bound": "Group exact input feature equality up to neighbor permutation, including receiver/sender state, context and degree. Classification min(countA,countB) and empirical entropy give lower bounds. Not a guaranteed attainable minimum for this architecture or source-model Bayes error.",
        "selection": "Training objective only, never validation/test. Report all seeds and rates. Best rate per seed may be summarized as a capacity diagnostic, clearly labeled train-selected.",
        "validation": "Only after fitting, score all selected and terminal checkpoints on the existing four development validation questions. These are repeatedly used exploratory questions, not a fresh confirmatory assessment.",
        "conditional_residual": "An optional later stage may add the frozen Ising logit to an otherwise identical neural correction initialized with zero output layer. Enabling it requires a separate frozen decision record after the primary diagnostics.",
        "no_test_evaluation": True, "source_llm_calls": 0, "autonomous_rollouts": False}
    if (root/"protocol.json").exists():
        existing = json.loads((root/"protocol.json").read_text())
        assert all(existing[k] == v for k, v in identity.items())
    else:
        json_write(root/"protocol.json", protocol)
    print(json.dumps({"stage": "prepare", "snapshots": len(rows), "transitions": a["observed"].size}), flush=True)
    assignment, counts, bounds = audit_collisions(a)
    np.savez_compressed(root/"train.npz", **a)
    np.savez_compressed(root/"input_groups.npz", assignment=assignment, counts=counts)
    json_write(root/"train_rows.json", rows)
    json_write(root/"prior_artifacts.json", old)
    json_write(root/"bounds.json", bounds)
    json_write(root/"ising_train_metrics.json", probability_metrics(a["teacher_logp"], a["observed"], a["teacher"]))
    names = ("train.npz", "input_groups.npz", "train_rows.json", "prior_artifacts.json", "bounds.json", "ising_train_metrics.json")
    json_write(root/"prepared.json", {"created_utc": now(), "protocol_sha256": digest(root/"protocol.json"),
        "artifacts": {name: digest(root/name) for name in names}})
    print(json.dumps({"bounds": bounds, "prepared": True}), flush=True)


def validate_prepared(cfg):
    root = Path(cfg["output"])
    p = json.loads((root/"protocol.json").read_text())
    assert p["code_sha256"] == digest(__file__) and p["config_sha256"] == digest(CONFIG)
    assert p["core_sha256"] == code_digest() == CORE_SHA
    assert p["study_implementation_sha256"] == study_code_digest()
    prepared = json.loads((root/"prepared.json").read_text())
    assert prepared["protocol_sha256"] == digest(root/"protocol.json")
    for name, sha in prepared["artifacts"].items():
        assert digest(root/name) == sha, name
    assert digest(p["baseline_path"]) == p["baseline_sha256"]
    return p


class TensorData:
    def __init__(self, arrays, device):
        self.arrays = arrays
        self.device = device
        self.tensors = {k: torch.as_tensor(arrays[k], dtype=torch.float32, device=device)
                        for k in ("s", "context", "J", "teacher")}
        self.tensors["observed"] = torch.as_tensor(arrays["observed"], dtype=torch.float32, device=device)
        self.tensors["teacher_logit"] = torch.as_tensor(arrays["teacher_logp"][..., 1]-arrays["teacher_logp"][..., 0], dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.arrays["s"])

    def batch(self, ids):
        b = {k: v[ids] for k, v in self.tensors.items()}
        b["extra"] = torch.empty((*b["s"].shape, 0), device=self.device)
        return b


def logits_for(model, batch, residual=False):
    logits = model(batch["s"], batch["context"], batch["J"], batch["extra"])
    if residual:
        logits = logits + torch.stack([torch.zeros_like(batch["teacher_logit"]), batch["teacher_logit"]], -1)
    return logits


def make_model(protocol, residual=False):
    model = make_graph(protocol["model_config"], "opinion")
    assert sum(p.numel() for p in model.parameters()) == 135458
    assert all(m.p == 0 for m in model.modules() if isinstance(m, nn.Dropout))
    if residual:
        nn.init.zeros_(model.classifier[-1].weight)
        nn.init.zeros_(model.classifier[-1].bias)
    return model


def predictions(model, data, batch_size=256, residual=False):
    result = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(data), batch_size):
            batch = data.batch(slice(offset, offset+batch_size))
            lp = logits_for(model, batch, residual).log_softmax(-1)
            result.append(lp.cpu().numpy())
    result = np.concatenate(result)
    assert np.isfinite(result).all()
    return result


def get_session(root, cfg):
    p = root/"session.json"
    if not p.exists():
        json_write(p, {"started_utc": now(), "training_deadline_epoch_seconds": time.time()+60*cfg["training"]["aggregate_minutes"],
                       "scope": "Aggregate training bound, resumable per epoch. Preparation and final analysis outside this training timer."})
    return json.loads(p.read_text())


def train_one(cfg, protocol, data, stage, seed, lr, deadline):
    root = Path(cfg["output"])
    residual = stage == "residual"
    objective = "imitation" if stage == "imitation" else "observed"
    rate = f"{lr:.0e}".replace("-0", "-").replace("+0", "+")
    run = root/stage/f"lr{rate}-s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    identity = {"stage": stage, "objective": objective, "seed": seed, "initial_learning_rate": lr,
        "protocol_sha256": digest(root/"protocol.json"), "prepared_sha256": digest(root/"prepared.json"),
        "residual": residual, "device": data.device}
    manifest_path = run/"manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        assert all(old[k] == v for k, v in identity.items())
        if old["status"] == "complete":
            for name, sha in old["artifacts"].items():
                assert digest(run/name) == sha
            return old
    seed_all(seed)
    model = make_model(protocol, residual).to(data.device)
    initial_sha = model_sha(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
    epochs, batch_size = cfg["training"]["epochs"], cfg["training"]["batch_size"]
    history, start_epoch, best, best_error, elapsed_before = [], 0, float("inf"), float("inf"), 0.
    created = now()
    if (run/"resume.pt").exists():
        ck = torch.load(run/"resume.pt", map_location=data.device, weights_only=False)
        assert ck["identity"] == identity
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        history, start_epoch = ck["history"], ck["epoch"]
        best, best_error, elapsed_before, created = ck["best"], ck["best_error"], ck["elapsed_seconds"], ck["created_utc"]
        torch.set_rng_state(ck["cpu_rng"].cpu())
        if data.device == "mps":
            torch.mps.set_rng_state(ck["mps_rng"].cpu())
    started = time.monotonic()
    objective_key = "teacher_cross_entropy" if objective == "imitation" else "observed_nll"
    base_manifest = {**identity, "created_utc": created, "initial_weights_sha256": initial_sha,
        "parameters": 135458, "training_snapshots": len(data), "training_transitions": data.arrays["observed"].size,
        "dropout": 0., "weight_decay": 0., "uses_validation_for_selection": False}
    json_write(manifest_path, {**base_manifest, "status": "running"})

    def evaluate(epoch, current_lr):
        nonlocal best, best_error
        lp = predictions(model, data, cfg["training"]["evaluation_batch_size"], residual)
        metrics = probability_metrics(lp, data.arrays["observed"], data.arrays["teacher"])
        record = {"epoch": epoch, "learning_rate": current_lr, "elapsed_seconds": elapsed_before+time.monotonic()-started, **metrics}
        history.append(record)
        if metrics[objective_key] < best:
            best = metrics[objective_key]
            save_checkpoint(run/"best.pt", {"identity": identity, "model": model.state_dict(), "epoch": epoch, "metrics": metrics})
        if metrics["observed_error"] < best_error:
            best_error = metrics["observed_error"]
            save_checkpoint(run/"best_error.pt", {"identity": identity, "model": model.state_dict(), "epoch": epoch, "metrics": metrics})
        json_write(run/"history.json", history)
        json_write(root/"progress.json", {"stage": stage, "run": run.name, "updated_utc": now(), **record})
        if epoch % 16 == 0 or epoch in (1, 2):
            print(json.dumps({"stage": stage, "run": run.name, "epoch": epoch,
                "train_error": metrics["observed_error"], "train_nll": metrics["observed_nll"],
                "teacher_mae": metrics["teacher_probability_mae"], "teacher_kl": metrics["teacher_kl"],
                "elapsed_seconds": record["elapsed_seconds"]}), flush=True)

    def checkpoint(epoch):
        save_checkpoint(run/"resume.pt", {"identity": identity, "model": model.state_dict(), "optimizer": opt.state_dict(),
            "history": history, "epoch": epoch, "best": best, "best_error": best_error, "created_utc": created,
            "elapsed_seconds": elapsed_before+time.monotonic()-started, "cpu_rng": torch.get_rng_state(),
            "mps_rng": torch.mps.get_rng_state() if data.device == "mps" else None})

    if not history:
        evaluate(0, lr)
        checkpoint(0)
    for epoch in range(start_epoch, epochs):
        if time.time() >= deadline:
            checkpoint(epoch)
            json_write(manifest_path, {**base_manifest, "status": "bounded", "completed_epochs": epoch})
            return {"status": "bounded"}
        fraction = epoch/max(1, epochs-1)
        min_fraction = cfg["training"]["cosine_minimum_fraction"]
        current_lr = lr*(min_fraction+(1-min_fraction)*.5*(1+math.cos(math.pi*fraction)))
        for group in opt.param_groups:
            group["lr"] = current_lr
        order = np.random.default_rng(seed+epoch).permutation(len(data))
        model.train()
        for offset in range(0, len(order), batch_size):
            batch = data.batch(order[offset:offset+batch_size])
            target = batch["teacher"] if objective == "imitation" else batch["observed"]
            opt.zero_grad(set_to_none=True)
            loss = soft_cross_entropy(logits_for(model, batch, residual), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"], error_if_nonfinite=True)
            opt.step()
        if epoch+1 in (1, 2) or (epoch+1) % cfg["training"]["evaluate_every_epochs"] == 0 or epoch+1 == epochs:
            evaluate(epoch+1, current_lr)
        checkpoint(epoch+1)
    terminal_sha = model_sha(model)
    terminal_metrics = history[-1]
    chosen = torch.load(run/"best.pt", map_location=data.device, weights_only=False)
    model.load_state_dict(chosen["model"])
    lp = predictions(model, data, cfg["training"]["evaluation_batch_size"], residual)
    selected = probability_metrics(lp, data.arrays["observed"], data.arrays["teacher"])
    for k in selected:
        np.testing.assert_allclose(selected[k], chosen["metrics"][k], atol=2e-7, rtol=2e-6)
    np.save(run/"train_logp.npy", lp, allow_pickle=False)
    assert model_sha(model) != initial_sha and terminal_sha != initial_sha
    window = cfg["diagnostic"]["plateau_window_epochs"]
    earlier = min(h[objective_key] for h in history if h["epoch"] <= epochs-window)
    improvement = earlier-best
    result = {**base_manifest, "status": "complete", "completed_utc": now(), "completed_epochs": epochs,
        "optimizer_steps": epochs*math.ceil(len(data)/batch_size), "selected_epoch": chosen["epoch"],
        "selected_weights_sha256": model_sha(model), "terminal_weights_sha256": terminal_sha,
        "selected_train": selected, "terminal_train": terminal_metrics,
        "minimum_training_error_observed": min(h["observed_error"] for h in history),
        "final_window_training_objective_improvement": improvement,
        "descriptive_training_plateau": improvement < cfg["diagnostic"]["plateau_nll_improvement"],
        "close_ising_imitation": selected["teacher_probability_mae"] <= cfg["diagnostic"]["imitation_mae_threshold"] and selected["teacher_kl"] <= cfg["diagnostic"]["imitation_kl_threshold"],
        "seconds": elapsed_before+time.monotonic()-started,
        "stopping_reason": "fixed_training_budget",
        "artifacts": {name: digest(run/name) for name in ("best.pt", "best_error.pt", "resume.pt", "history.json", "train_logp.npy")}}
    json_write(manifest_path, result)
    print(json.dumps({"stage": stage, "run": run.name, "status": "complete", "selected_epoch": chosen["epoch"],
                      "train_error": selected["observed_error"], "teacher_mae": selected["teacher_probability_mae"],
                      "teacher_kl": selected["teacher_kl"], "seconds": result["seconds"]}), flush=True)
    return result


def train_stage(cfg, stage, device):
    root = Path(cfg["output"])
    protocol = validate_prepared(cfg)
    if stage == "residual":
        decision = json.loads((root/"residual_decision.json").read_text())
        assert decision["enabled"] and decision["protocol_sha256"] == digest(root/"protocol.json")
    data = TensorData(dict(np.load(root/"train.npz", allow_pickle=False)), device)
    session = get_session(root, cfg)
    done = []
    for seed in cfg["training"]["seeds"]:
        for lr in cfg["training"]["learning_rates"]:
            record = train_one(cfg, protocol, data, stage, seed, lr, session["training_deadline_epoch_seconds"])
            if record["status"] != "complete":
                print(json.dumps({"stage": stage, "status": "bounded"}), flush=True)
                return
            done.append(record)
    json_write(root/f"{stage}_complete.json", {"completed_utc": now(), "stage": stage, "fits": len(done), "environment": environment(),
        "manifests": {str(p.parent.relative_to(root)): digest(p) for p in sorted((root/stage).glob("*/manifest.json"))}})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "imitation", "observed", "residual"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    cfg = config(CONFIG)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    if args.stage == "prepare":
        prepare(cfg)
        return
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    train_stage(cfg, args.stage, args.device)


if __name__ == "__main__":
    main()
