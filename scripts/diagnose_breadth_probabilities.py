"""Audit logged source probabilities as a retrospective sampling reference."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import xlogy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import breadth_train as study
from scripts.local_conversation_pilot import deliver, formatter, opinion_prompt


def majority_probability(p):
    p = np.asarray(p, dtype=np.float64)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    # Polynomial roundoff near one must not produce a negative 1-q in entropy.
    return np.clip(p**3 * (10 - 15*p + 6*p**2), 0., 1.)


def probability_metrics(logp, target, q):
    predicted = logp.argmax(-1)
    reference = np.minimum(q, 1-q)
    expected_error = np.where(predicted, 1-q, q)
    entropy = -xlogy(q, q) - xlogy(1-q, 1-q)
    expected_nll = -(1-q)*logp[..., 0] - q*logp[..., 1]
    return {"observed_error": float((predicted != target).mean()),
            "expected_error": float(expected_error.mean()),
            "sampling_reference_error": float(reference.mean()),
            "excess_expected_error": float((expected_error-reference).mean()),
            "expected_nll": float(expected_nll.mean()), "source_entropy": float(entropy.mean()),
            "excess_expected_nll": float((expected_nll-entropy).mean()),
            "probability_mean_squared_error": float(((np.exp(logp[..., 1])-q)**2).mean())}


def main():
    study.validate_prepared()
    study.authorize_split(study.ROOT, "assessment")
    freeze_sha = digest(study.ROOT / "models_frozen.json")
    output = study.ROOT / "error_diagnosis_v1"
    source_path = output / "development_source_probabilities_20261110.json.gz"
    assert digest(source_path) == "b7c6fd634be77e6fd2a86eb80783066bce3f57f35970e1f9216493dc34a1c0f5"
    with gzip.open(source_path, "rt") as stream:
        source = json.load(stream)
    assert source["source_protocol_sha256"] == digest(study.ROOT / "source_protocol.json")
    assert source["selection_sha256"] == digest(study.ROOT / "selection.json")
    common = dict(np.load(study.ROOT / "predictors/development.npz", allow_pickle=False))
    saved = np.load(output / "predictions.npz", allow_pickle=False)
    frame = pd.read_csv(output / "transitions.csv")
    lookup = {qid: i for i, qid in enumerate(common["episode_qids"])}
    questions = {q["qid"]: q for q in json.loads((study.ROOT / "selection.json").read_text())["questions"] if q["qid"] in lookup}
    shape = common["spins"].shape
    p = np.full(shape, np.nan)
    other = np.full(shape, np.nan)
    hashes = {}
    for row in source["rows"]:
        key = (lookup[row["qid"]], row["round"], row["agent"])
        assert np.isnan(p[key])
        np.testing.assert_array_equal(row["raw_votes"], common["raw_votes"][key])
        pa = np.array(row["first_literal_A_probabilities"])
        pb = np.array(row["first_literal_B_probabilities"])
        assert np.ptp(pa) == 0 and np.ptp(pb) == 0
        p[key] = pa[0] / (pa[0] + pb[0])
        other[key] = max(0, 1-pa[0]-pb[0])
        hashes[key] = row["prompt_sha256"]
    assert np.isfinite(p).all()
    # Independently reconstruct all full prompts from the saved messages.
    fmt = formatter()
    for qi, qid in enumerate(common["episode_qids"]):
        q = questions[qid]
        episode = json.loads((study.ROOT / "episodes" / q["split"] / f"{qid}__{q['graph_id']}__r00.json").read_text())
        agree, disagree = [[] for _ in range(32)], [[] for _ in range(32)]
        for step in range(9):
            if step:
                senders = {int(key.split(",")[0]): text for key, text in episode["messages_history"][step-1].items()}
                agree, disagree, edges = deliver(episode["J"], senders, episode["meta"]["name"], step)
                assert edges == episode["messages_history"][step-1]
            for agent in range(32):
                prompt = opinion_prompt(fmt, episode["personas"][agent], q, agree[agent], disagree[agent])
                assert hashlib.sha256(prompt.encode()).hexdigest() == hashes[qi, step, agent]
    e, t = common["pairs"].T
    q = majority_probability(p[e, t+1])
    target = saved["target"]
    frame["source_p_A"] = p[e, t+1].ravel()
    frame["source_majority_p_A"] = q.ravel()
    state_ids = []
    for qi, step in common["pairs"]:
        for agent in range(32):
            relevant = np.unique(np.r_[agent, np.flatnonzero(common["J"][qi, agent])])
            key = (str(common["episode_qids"][qi]), agent, tuple(int(x) for x in common["spins"][qi, step, relevant]))
            state_ids.append(str(key))
    frame["supplied_state_id"] = state_ids
    state = frame.groupby("supplied_state_id").source_majority_p_A
    frame["state_count"] = state.transform("size")
    frame["state_q_mean"] = state.transform("mean")
    frame["state_q_range"] = state.transform("max") - state.transform("min")
    records = {}
    for split in ["train", "validation"]:
        take = np.array([questions[qid]["split"] == split for qid in common["episode_qids"][e]])
        qs, ys = q[take], target[take]
        metrics = {"I0": probability_metrics(saved["ising_logp"][take], ys, qs)}
        seeds = [probability_metrics(lp[take], ys, qs) for lp in saved["nano_logp"]]
        metrics["nano"] = {key: float(np.mean([r[key] for r in seeds])) for key in seeds[0]}
        part = frame[frame.split == split]
        repeated = part[(part.state_count > 1) & (part.degree > 0)]
        sampling_reference = np.minimum(qs, 1-qs).mean()
        state_reference = np.minimum(part.state_q_mean, 1-part.state_q_mean).mean()
        assert state_reference >= sampling_reference - 1e-10
        actual_noise_errors = ((qs > .5) != ys)
        records[split] = {
            "models": metrics,
            "full_prompt_reference_observed_error": float(actual_noise_errors.mean()),
            "supplied_state_empirical_reference_error": float(state_reference),
            "additional_error_from_collapsing_observed_prompts_to_state": float(state_reference-sampling_reference),
            "connected_updates_in_repeated_states": len(repeated),
            "repeated_state_q_range_median": float(repeated.state_q_range.median()),
            "repeated_state_updates_with_q_range_over_025": float((repeated.state_q_range > .25).mean()),
            "majority_probability_mean": float(qs.mean()), "observed_majority_A_fraction": float(ys.mean()),
            "majority_calibration_residual_z": float((ys-qs).sum()/np.sqrt((qs*(1-qs)).sum())),
        }
    bins = pd.cut(frame.source_majority_p_A, np.linspace(0, 1, 11), include_lowest=True)
    calibration = []
    for (split, interval), part in frame.groupby(["split", bins], observed=True):
        calibration.append({"split": split, "bin": str(interval), "count": len(part),
                            "predicted_majority_A": float(part.source_majority_p_A.mean()),
                            "observed_majority_A": float(part.target_A.mean())})
    examples = []
    validation = frame[(frame.split == "validation") & (frame.degree > 0) & (frame.state_count > 1)]
    for key in validation.groupby("supplied_state_id").state_q_range.max().sort_values(ascending=False).head(6).index:
        part = validation[validation.supplied_state_id == key]
        rows = part.loc[[part.source_majority_p_A.idxmin(), part.source_majority_p_A.idxmax()]]
        examples.append({"state": key, "probability_range": float(part.state_q_range.iloc[0]),
                         "updates": rows[["qid", "agent", "round", "current_A", "target_A", "source_p_A", "source_majority_p_A", "ising_probability_A", "nano_probability_A_mean"]].to_dict("records")})
    frame.to_csv(output / "transitions_with_source_probabilities.csv", index=False)
    json_write(output / "same_state_probability_examples.json", examples)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": digest(__file__),
        "source_sha256": digest(source_path), "models_frozen_sha256": freeze_sha,
        "scope": "Retrospective train/validation diagnosis. Full source prompts include newly generated messages, unavailable at the original forecasting boundary.",
        "approximation": "Normalize first bare A/B token probabilities, then compute P(majority A) for five independent votes. This approximates full parsed-response majority probabilities; it is not an exact population Bayes bound for the supplied state.",
        "quality": {"reconstructed_prompt_hashes_verified": len(hashes), "source_responses": source["response_count"],
                    "strict_responses": source["strict_response_count"], "retries": source["retry_response_count"],
                    "not_single_bare_token_responses": source["not_single_bare_token_response_count"],
                    "mean_other_first_token_probability": float(other.mean()),
                    "maximum_other_first_token_probability": float(other.max()),
                    "other_first_token_probability_quantiles": np.quantile(other, [.5, .9, .99, .999]).tolist(),
                    "maximum_within_prompt_probability_spread": source["maximum_within_prompt_probability_spread"]},
        "results": records, "calibration": calibration,
        "interpretation": "The full-prompt sampling reference separates approximate sampling variability from excess model error. The supplied-state reference is an optimistic finite-sample memorization reference over recorded prompts, not a validated achievable model or global irreducible error estimate."}
    json_write(output / "source_probability_diagnosis.json", result)
    study.validate_prepared()
    study.authorize_split(study.ROOT, "assessment")
    assert digest(study.ROOT / "models_frozen.json") == freeze_sha
    print(json.dumps({"quality": result["quality"], "results": records, "examples": examples[:2]}, indent=2))


if __name__ == "__main__":
    main()
