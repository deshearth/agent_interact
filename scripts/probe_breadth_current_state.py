"""Train-only compact state corrections; separate from the frozen comparison."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import breadth_train as study
from scripts.probe_initial_opinion_state import fit, standardize
from scripts.diagnose_ising_gap import score_metrics


def main():
    study.validate_prepared()
    study.authorize_split(study.ROOT, "assessment")
    original_freeze = digest(study.ROOT / "models_frozen.json")
    output = study.ROOT / "error_diagnosis_v1/current_state_probe"
    output.mkdir(parents=True, exist_ok=True)
    common = dict(np.load(study.ROOT / "predictors/development.npz", allow_pickle=False))
    saved = np.load(study.ROOT / "error_diagnosis_v1/predictions.npz", allow_pickle=False)
    prior = saved["ising_logp"][..., 1] - saved["ising_logp"][..., 0]
    e, t = common["pairs"].T
    target = (common["spins"][e, t + 1] == 1).astype(float)
    current = common["spins"][e, t].astype(float)
    current_logodds = logit(((common["raw_votes"][e, t] == 1).sum(-1) + .5) / 6)
    qlookup = {q["qid"]: q for q in json.loads((study.ROOT / "selection.json").read_text())["questions"]
               if q["split"] in ["train", "validation"]}
    split = np.array([qlookup[q]["split"] for q in common["episode_qids"][e]])
    train, validation = split == "train", split == "validation"
    assert train.sum() == 640 and validation.sum() == 160
    specification = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": digest(__file__),
        "models_frozen_sha256": original_freeze,
        "models": {"calibration": ["intercept", "prior_logit"],
                   "current_opinion": ["intercept", "prior_logit", "current_own_spin"],
                   "current_votes": ["intercept", "prior_logit", "current_own_spin", "current_vote_smoothed_logodds"]},
        "penalty": .001, "objective": "Mean BCE plus 0.001/2 times squared correction coefficients; original Ising logit is an offset.",
        "fitting": "80 training questions only, including standardization; no validation tuning or model selection.",
        "scope": "Post-hoc development diagnostic. Original frozen predictors and future assessment comparison are unchanged.",
        "state_boundary": "Only current opinions and current-round five votes; no target votes or future messages enter features."}
    json_write(output / "protocol.json", specification)
    records, predictions = [], {}
    for name, columns in specification["models"].items():
        components = [np.ones_like(prior), prior, current, current_logodds][:len(columns)]
        x = np.stack(components, -1)
        xfit, _, mean, scale = standardize(x[train].reshape(-1, len(columns)), x[validation].reshape(-1, len(columns)))
        with threadpool_limits(limits=1):
            w, details = fit(xfit, prior[train].ravel(), target[train].ravel(), specification["penalty"])
        z = prior + ((x - mean) / scale) @ w
        predictions[name] = z
        raw_weights = w / scale
        intercept = raw_weights[0] - np.sum(w * mean / scale)
        record = {"model": name, "additional_coefficients": len(columns), "weights": w.tolist(),
                  "mean": mean.tolist(), "scale": scale.tolist(), "intercept": float(intercept),
                  "ising_logit_scale": float(1 + raw_weights[1]), **details}
        if len(columns) >= 3: record["current_spin_coefficient"] = float(raw_weights[2])
        if len(columns) == 4: record["current_vote_logodds_coefficient"] = float(raw_weights[3])
        for label, mask in [("train", train), ("validation", validation)]:
            record[label] = score_metrics(z[mask], target[mask])
        records.append(record)
    np.savez_compressed(output / "predictions.npz", target=target, **predictions)
    json_write(output / "results.json", {"protocol_sha256": digest(output / "protocol.json"), "models": records})
    study.validate_prepared()
    study.authorize_split(study.ROOT, "assessment")
    assert digest(study.ROOT / "models_frozen.json") == original_freeze
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
