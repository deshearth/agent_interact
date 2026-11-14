"""Retrain on shared argument centroids or individual residuals, familiar questions only."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import argument_mechanisms as previous
from scripts import argument_response as original
from scripts import question_context as question
from scripts import joint_argument_context as driver
from scripts import joint_argument_encoding as session
from scripts.argument_features import load_cache

ROOT = session.SESSION/"residual_diagnostic"
CONFIG = "configs/argument_residual_diagnostic.yaml"
VARIANTS = ("question_mean", "question_stance_mean", "individual_residual")


def implementation():
    return {**question.implementation(), "residual_diagnostic_sha256": digest(__file__),
            "generic_driver_sha256": digest(driver.__file__)}


def make_model(cfg, kind):
    if kind not in VARIANTS:
        raise ValueError(kind)
    return question.make_model(cfg, "full_embedding")


def prepare(cfg):
    arrays, index, vectors, metadata = load_cache(cfg, verify=True)
    parent = json.loads((question.ROOT/"protocol.json").read_text())
    assert parent["implementation"] == question.implementation()
    for field in ("data", "model", "training", "evaluation", "diagnostic", "question"):
        assert cfg[field] == parent["config"][field]
    for file, sha in parent["prepared_sha256"].items():
        assert digest(question.ROOT/file) == sha
    index = previous.split_index(index, cfg)
    arrays = question.augmented_arrays(arrays, index, cfg["residual"]["question_features"], "full_embedding")
    fitting_qids = sorted({r["qid"] for r in index if r["diagnostic_role"] == "train"})
    identity = {"implementation": implementation(), "config": cfg, "config_sha256": digest(CONFIG),
        "parent_protocol_sha256": digest(question.ROOT/"protocol.json"),
        "features_sha256": digest(Path(cfg["data"]["cache"])/"metadata.json"), "fit_qids": fitting_qids}
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/"protocol.json"
    if path.exists():
        saved = json.loads(path.read_text())
        assert all(saved[k] == v for k, v in identity.items())
        for file, sha in saved["prepared_sha256"].items():
            assert digest(ROOT/file) == sha
    else:
        means = np.zeros((len(fitting_qids), 384), dtype=np.float64)
        signed = np.zeros((len(fitting_qids), 2, 384), dtype=np.float64)
        qids = np.zeros_like(arrays["text_ids"])
        qsids = np.zeros_like(qids)
        residual = np.lib.format.open_memmap(ROOT/"individual_residual.npy", mode="w+", dtype=np.float32, shape=vectors.shape)
        residual[:] = 0
        counts, squared = 0, 0.
        assignment = np.full(len(vectors), -1, dtype=np.int16)
        for qi, qid in enumerate(fitting_qids):
            es = [e for e, r in enumerate(index) if r["qid"] == qid and r["diagnostic_role"] == "train"]
            ids = arrays["text_ids"][es].reshape(-1)
            stance = arrays["stance"][es].reshape(-1)[ids != 0]
            ids = ids[ids != 0]
            x = np.asarray(vectors[ids], dtype=np.float64)
            means[qi] = x.mean(0)
            for sign in (-1, 1):
                mask = stance == sign
                centroid = x[mask].mean(0)
                signed[qi, int(sign == 1)] = centroid
                squared += float(np.square(x[mask]-centroid).sum())
                counts += int(mask.sum())
            # Transform all familiar-role inputs with fitting-only centroids.
            # No unseen-question vectors enter this branch or its evaluations.
            es = [e for e, r in enumerate(index) if r["qid"] == qid]
            for e in es:
                present = arrays["text_ids"][e] != 0
                qids[e, present] = qi+1
                qsids[e, present] = 1+2*qi+(arrays["stance"][e, present] == 1)
                for sign in (-1, 1):
                    use = np.unique(arrays["text_ids"][e][present & (arrays["stance"][e] == sign)])
                    code = 2*qi+int(sign == 1)
                    assert ((assignment[use] == -1) | (assignment[use] == code)).all()
                    assignment[use] = code
                    residual[use] = vectors[use]-signed[qi, int(sign == 1)]
        scale = float(np.sqrt(squared/counts))
        assert scale>0
        residual[:] /= scale
        residual.flush()
        del residual
        qm = means/np.linalg.norm(means, axis=-1, keepdims=True)
        qsm = signed.reshape(-1, 384)
        qsm = qsm/np.linalg.norm(qsm, axis=-1, keepdims=True)
        np.save(ROOT/"question_mean.npy", np.vstack([np.zeros((1, 384)), qm]).astype(np.float32))
        np.save(ROOT/"question_stance_mean.npy", np.vstack([np.zeros((1, 384)), qsm]).astype(np.float32))
        np.savez(ROOT/"ids.npz", question_mean=qids, question_stance_mean=qsids)
        np.savez(ROOT/"centroids.npz", qids=np.array(fitting_qids), question=means, question_stance=signed,
                 residual_rms_scale=np.array(scale), fitting_message_count=np.array(counts))
        files = [f"{k}.npy" for k in VARIANTS]+["ids.npz", "centroids.npz"]
        json_write(path, {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "prepared_sha256": {file: digest(ROOT/file) for file in files},
            "purpose": "Does familiar-question argument benefit survive retraining with no individual message variation, and does isolating that variation help?",
            "centroid_fit": "Only message occurrences from fitting replicas 0/1 of 16 questions; no next-state labels. Question means and question/current-stance means are L2-normalized after fitting.",
            "residual": "Subtract the unnormalized fitting question/stance centroid from each original message embedding, then divide by one global fitting RMS L2 residual norm. No per-vector normalization. Full question context and explicit stance inputs remain available.",
            "split": "Same fitting and selection replicas as all preceding controlled studies. Evaluate train, selection and familiar replica 3 only. No unseen-question scores because a new question lacks a fitting-derived centroid.",
            "architecture": "Identical full-question-context 382498-parameter predictor and optimizer/stopping. Train only the full_arguments condition using the named transformed representation; frozen original references supply stance/full/shuffled comparators.",
            "limitations": "Centroids retain shared argument content, potentially common reasoning. This isolates individual variation within known questions, not all semantics or transfer. Reused development data; no causal LLM intervention."})
    with np.load(ROOT/"ids.npz", allow_pickle=False) as saved:
        ids = {kind: saved[kind] for kind in VARIANTS if kind != "individual_residual"}
    adapted = deepcopy(metadata)
    adapted["questions"] = {"train": fitting_qids, "validation": fitting_qids}
    return {kind: ({**arrays, "text_ids": ids.get(kind, arrays["text_ids"])}, index,
        np.load(ROOT/f"{kind}.npy", mmap_mode="r", allow_pickle=False), adapted) for kind in VARIANTS}


def evaluate_run(item, cfg, cache, device, smoke=False):
    name = f"{item['kind']}-{item['condition']}-s{item['seed']}"
    run = (ROOT/"smoke" if smoke else ROOT)/name
    out = (ROOT/"smoke_evaluation" if smoke else ROOT/"evaluation")/name
    out.mkdir(parents=True, exist_ok=True)
    identity = {"run_manifest_sha256": digest(run/"manifest.json"), "best_sha256": digest(run/"best.pt"),
                "protocol_sha256": digest(ROOT/"protocol.json")}
    record = out/"identity.json"
    if record.exists():
        saved = json.loads(record.read_text())
        assert all(saved[k] == v for k, v in identity.items())
        for file, sha in saved["artifacts"].items():
            assert digest(out/file) == sha
        return
    arrays, index, vectors, _ = cache
    assert all(r["diagnostic_role"] != "unseen" for r in index if r["diagnostic_role"] in ("train", "familiar"))
    donors = np.load(previous.ROOT/f"donors_s{item['seed']}.npy", allow_pickle=False)
    model = make_model(cfg, item["kind"]).to(device)
    model.load_state_dict(torch.load(run/"best.pt", map_location=device, weights_only=False)["model"])
    names = []
    for role in ("train", "familiar"):
        ds = previous.SplitSnapshots(arrays, index, vectors, role, item["condition"], donors)
        original.save_predictions(out, role, model, ds, cfg, device)
        names.extend([f"{role}_{s}" for s in ("predictions.parquet", "metrics.json", "by_question.csv")])
    json_write(record, {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
                        "artifacts": {file: digest(out/file) for file in names}})


@contextmanager
def adapted_driver():
    # Adapt the frozen generic loop in memory; no prior source file is edited.
    changes = {"CONFIG": CONFIG, "ROOT": ROOT, "VARIANTS": VARIANTS, "CONDITIONS": ("full_arguments",),
               "implementation": implementation, "make_model": make_model, "prepare": prepare, "evaluate_run": evaluate_run}
    before = {name: getattr(driver, name) for name in changes}
    for name, value in changes.items():
        setattr(driver, name, value)
    try:
        yield
    finally:
        for name, value in before.items():
            setattr(driver, name, value)


if __name__ == "__main__":
    with adapted_driver():
        driver.main()
