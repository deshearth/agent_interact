"""Measure frozen predictor sensitivity to all four pilot conditions, locally."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from scripts import argument_intervention_pilot as pilot
from scripts import argument_mechanisms as previous
from scripts import question_context as question
from scripts import joint_argument_context as joint
from scripts import joint_argument_encoding as source
from scripts import joint_argument_encoding_windowed as encoding
from scripts.probe_argument_quality_predictors import standalone
from scripts.argument_features import load_cache

ROOT = pilot.ROOT / "predictor_probe"


def main():
    torch.set_num_threads(2)
    _, inputs, _ = pilot.verify_prepared(pilot.ROOT)
    cases = {c["case_id"]: c for c in json.loads((pilot.ROOT / "cases.json").read_text())}
    cfg = config(question.CONFIG)
    arrays, index, old_vectors, _ = load_cache(cfg, verify=True)
    index = previous.split_index(index, cfg)
    episodes = {r["episode_id"]: e for e, r in enumerate(index)}
    unique = sorted({(r["qid"], r["condition"]) for r in inputs})
    lookup = {k: i for i, k in enumerate(unique)}
    by_key = {(r["qid"], r["condition"]): r["argument"] for r in inputs}
    texts = [by_key[k] for k in unique]
    tokenizer, encoder = source.load_encoder("cpu")
    basic = standalone(tokenizer, encoder, texts)
    qstrings = json.loads((source.ROOT / "questions.json").read_text())
    qt = [tokenizer(qstrings[q], add_special_tokens=False, truncation=False, verbose=False)["input_ids"] for q, _ in unique]
    conditioned, _ = encoding.encode_records(tokenizer, encoder, qt, texts, "cpu", microbatch=12)
    del encoder
    specs = [("original_pca3", previous.ROOT, "A2", previous.make_model),
             ("full_question", question.ROOT, "full_embedding", question.make_model),
             ("question_conditioned", joint.ROOT, "question_conditioned", joint.make_model)]
    provenance = {}
    for label, root, kind, _ in specs:
        for seed in (42, 43, 44):
            run = root / f"{kind}-full_arguments-s{seed}"
            meta = json.loads((run / "manifest.json").read_text())
            assert meta["status"] == "complete" and digest(run / "best.pt") == meta["artifact_sha256"]["best.pt"]
            provenance[f"{label}/s{seed}"] = {"best_sha256": digest(run / "best.pt"), "manifest_sha256": digest(run / "manifest.json")}
    old_path = source.SESSION / "quality_uncertain_probe/predictions.csv"
    old_meta = json.loads((old_path.parent / "complete.json").read_text())
    assert digest(old_path) == old_meta["artifacts"][old_path.name]
    identity = {"implementation_sha256": digest(__file__), "pilot_manifest_sha256": digest(pilot.ROOT / "manifest.json"),
                "encoding_protocol_sha256": digest(encoding.ROOT / "protocol.json"),
                "parent_predictions_sha256": digest(old_path), "models": provenance}
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "complete.json").exists():
        saved = json.loads((ROOT / "complete.json").read_text())
        assert all(saved[k] == v for k, v in identity.items())
        for name, checksum in saved["artifacts"].items():
            assert digest(ROOT / name) == checksum
        print(json.dumps({"complete": True, "reused": True, "source_calls": 0})); return
    np.savez(ROOT / "argument_vectors.npz", keys=np.array([q + "/" + c for q, c in unique]),
             standalone=basic, question_conditioned=conditioned)
    geometry = []
    for name, vectors in (("standalone", basic), ("question_conditioned", conditioned)):
        for qid in sorted({q for q, _ in unique}):
            for left, right in pilot.CONTRASTS:
                a, b = vectors[lookup[qid, left]], vectors[lookup[qid, right]]
                geometry.append({"encoding": name, "qid": qid, "contrast": left + "_minus_" + right,
                                 "cosine": float(a @ b), "l2": float(np.linalg.norm(a - b))})
    pd.DataFrame(geometry).to_csv(ROOT / "embedding_geometry.csv", index=False)
    output = []
    for label, root, kind, factory in specs:
        a, vec, new = arrays, old_vectors, basic
        if label != "original_pca3":
            a = question.augmented_arrays(arrays, index, question.ROOT / "question_features.npz", "full_embedding")
        if label == "question_conditioned":
            with np.load(source.ROOT / "ids.npz", allow_pickle=False) as mapping:
                a = {**a, "text_ids": mapping["text_ids"]}
            vec = np.load(encoding.ROOT / "question_conditioned.npy", mmap_mode="r", allow_pickle=False)
            new = conditioned
        donors = np.load(previous.ROOT / "donors_s42.npy", allow_pickle=False)
        ds = previous.SplitSnapshots(a, index, vec, "unseen", "full_arguments", donors)
        positions = {p: i for i, p in enumerate(ds.pairs)}
        batches, receivers = [], []
        for row in inputs:
            case = cases[row["case_id"]]
            e, t, receiver = episodes[case["episode_id"]], case["round"], case["receiver"]
            assert index[e]["diagnostic_role"] == "unseen"
            batch = ds[positions[e, t]]
            batch.pop("target")
            slots = np.flatnonzero((a["neighbors"][e, receiver] == case["sender"]) &
                                  (a["signs"][e, receiver] == case["relationship_sign"]))
            assert len(slots) == 1
            slot = int(slots[0])
            assert a["stance"][e, t, receiver, slot] == case["message_stance"]
            before = batch["text"].clone()
            batch["text"][receiver, slot] = torch.from_numpy(new[lookup[row["qid"], row["condition"]]])
            changed = torch.any(before != batch["text"], dim=-1)
            assert changed.sum().item() == 1 and changed[receiver, slot]
            batches.append(batch); receivers.append(receiver)
        batch = default_collate(batches)
        for seed in (42, 43, 44):
            run = root / f"{kind}-full_arguments-s{seed}"
            model = factory(cfg, kind).eval()
            model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=False)["model"])
            with torch.inference_mode():
                probabilities = model(batch)[torch.arange(len(inputs)), torch.tensor(receivers)].softmax(-1)[:, 1].numpy()
            for row, p in zip(inputs, probabilities):
                output.append({"model": label, "seed": seed, "case_id": row["case_id"], "qid": row["qid"],
                               "condition": row["condition"], "relationship_sign": row["relationship_sign"],
                               "message_stance": row["message_stance"], "p_majority_A": float(p),
                               "p_majority_message_stance": float(p if row["message_stance"] == 1 else 1 - p)})
    frame = pd.DataFrame(output)
    old = pd.read_csv(old_path)
    matched = frame.merge(old, on=["model", "seed", "case_id", "qid", "condition", "relationship_sign", "message_stance"], validate="one_to_one")
    assert len(matched) == len(old) == 648
    error = np.abs(matched.p_majority_A - matched.p_A)
    assert error.max() < 1e-5
    frame.to_csv(ROOT / "predictions.csv", index=False)
    pairs = []
    keys = ["model", "seed", "case_id", "qid", "relationship_sign"]
    pivot = frame.pivot(index=keys, columns="condition", values="p_majority_message_stance").reset_index()
    for left, right in pilot.CONTRASTS:
        for _, row in pivot.iterrows():
            pairs.append({**{k: row[k] for k in keys}, "contrast": left + "_minus_" + right,
                          "difference": row[left] - row[right], "absolute_difference": abs(row[left] - row[right]),
                          "binary_prediction_changed": (row[left] >= .5) != (row[right] >= .5)})
    paired = pd.DataFrame(pairs)
    paired.to_csv(ROOT / "paired_predictions.csv", index=False)
    summary = paired.groupby(["model", "contrast", "relationship_sign"])[["difference", "absolute_difference", "binary_prediction_changed"]].mean().reset_index()
    summary.to_csv(ROOT / "summary.csv", index=False)
    files = ("argument_vectors.npz", "embedding_geometry.csv", "predictions.csv", "paired_predictions.csv", "summary.csv")
    json_write(ROOT / "complete.json", {**identity, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "source_calls": 0, "optimizer_steps": 0, "counterfactual_accuracy_evaluations": 0,
               "models": provenance, "prediction_rows": len(frame), "comparison_rows": len(paired),
               "reproduced_parent_predictions": len(matched), "maximum_parent_prediction_difference": float(error.max()),
               "interpretation": "Frozen predictor sensitivity only. No counterfactual source outcomes. Removing a rationale also removes question-specific text and changes length, so it is not a pure validity intervention.",
               "artifacts": {name: digest(ROOT / name) for name in files}})
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
