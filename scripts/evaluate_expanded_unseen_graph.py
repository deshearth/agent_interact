"""Evaluate already-frozen 320-question predictors on the completed graph study.

No fitting, source generation, context refitting or assessment-question access.
The original graph study and both sets of predictor artifacts remain immutable.
"""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from mac_agents.models.baselines import baseline_logp
from scripts import breadth_expand_train as training
from scripts import transformer_generalization as base
from scripts import unseen_graph_experiment as graph_study

ROOT = Path("runs/unseen_graph_320_v1")
SOURCE = Path("runs/unseen_graph_v1")
METRICS = ("accuracy", "error", "nll", "expected_error", "expected_nll",
           "sampling_reference_error", "excess_expected_error", "source_entropy", "excess_expected_nll")


def paired_contrast(rows, terms, qids, seed, replicates):
    """Combine models/regimes within question before resampling whole questions."""
    lookup = {(r["training_questions"], r["regime"], r["model"], r["qid"]): r for r in rows}
    assert len(lookup) == len(rows)
    qids = sorted(qids)
    draws = np.random.default_rng(seed).integers(0, len(qids), (replicates, len(qids)))
    result = {}
    for metric in METRICS:
        delta = np.array([sum(coefficient * lookup[(count, regime, model, q)][metric]
                              for coefficient, count, regime, model in terms) for q in qids])
        low, high = np.quantile(delta[draws].mean(1), [.025, .975])
        result[metric] = {"difference": float(delta.mean()), "lower": float(low), "upper": float(high)}
    return result


def aggregate_records(records, qids):
    assert len(qids) == len(set(qids)) and {r["qid"] for r in records} == set(qids)
    result = []
    settings = sorted({(r["training_questions"], r["regime"], r["model"]) for r in records})
    for count, regime, model in settings:
        for qid in sorted(qids):
            selected = [r for r in records if (r["training_questions"], r["regime"], r["model"], r["qid"]) == (count, regime, model, qid)]
            assert len(selected) == (3 if model == "nano" else 1)
            assert len({r["seed"] for r in selected}) == len(selected)
            assert len({r["graph_id"] for r in selected}) == 1
            result.append({"training_questions": count, "regime": regime, "model": model,
                "qid": qid, "graph_id": selected[0]["graph_id"],
                **{k: float(np.mean([r[k] for r in selected])) for k in METRICS}})
    return result


def main():
    cfg = training.config()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    frozen = training.validate_frozen(cfg)
    assert [r["seed"] for r in frozen["refits"]] == [142, 143, 144]
    source_protocol = graph_study.validate(SOURCE)
    source_result = training.read(SOURCE / "results.json")
    assert source_result["protocol_sha256"] == digest(SOURCE / "protocol.json")
    assert source_result["new_episodes"] == 20 and source_result["new_agent_transitions"] == 5120
    assert not source_result["assessment_generated_or_scored"]
    sources = {str(SOURCE / name): digest(SOURCE / name) for name in
               ("protocol.json", "source_protocol.json", "results.json", "graphs.json", "evaluation_inputs.npz")}
    for path, sha in source_result["scored_episode_sha256"].items():
        assert digest(path) == sha
        sources[path] = sha
    for name, sha in source_result["output_sha256"].items():
        assert digest(SOURCE / name) == sha
        sources[str(SOURCE / name)] = sha

    model_root = Path(cfg["output"])
    model_protocol = training.read(model_root / "protocol.json")
    qids = source_protocol["validation_question_ids"]
    assert len(qids) == len(set(qids)) == 20
    assert set(qids) <= set(model_protocol["fold"]["validation_questions"])
    assert not set(qids) & set(model_protocol["fold"]["train_questions"])
    assert not set(qids) & set(model_protocol["fold"]["assessment_questions"])
    assert Counter(q["graph_id"] for q in source_protocol["questions"]) == {g: 5 for g in source_protocol["unseen_graph_ids"]}
    policy = {
        "script_sha256": digest(__file__), "source_artifacts": sources,
        "graph_protocol_sha256": digest(SOURCE / "protocol.json"),
        "new_models_frozen_sha256": digest(model_root.parent / "models_frozen.json"),
        "new_model_protocol_sha256": digest(model_root / "protocol.json"),
        "original_models_frozen_sha256": digest(Path(cfg["parent_root"]) / "models_frozen.json"),
        "questions": qids, "models": {"nano": 818, "I0": 19}, "seeds": [142, 143, 144],
        "design": "Post-training evaluation of all frozen 320-question refits and Ising on exactly the completed 20-question/four-graph study. All weights, PCA, thresholds and seed aggregation remain fixed. Original 80-question results are the reference.",
        "pairing": "Same questions and all five round-zero source votes; different signed graphs and later stochastic trajectories.",
        "analysis": "Equal question weighting, mean individual-seed scores. Paired bootstrap of 20 questions conditional on four graphs; report transformer-minus-Ising, unseen-minus-seen, 320-minus-80 and paired differences in degradation. No tuning on these trajectories.",
        "limitations": "Known development validation questions; four new graph instances within the same 32-node signed random-graph family. Not unseen topology families or new agent populations. Source-probability references use unavailable full-prompt information and approximate the parsed/retry process.",
        "model_refitting": False, "source_generation": False, "assessment_loaded": False,
        "bootstrap_seed": source_protocol["config"]["evaluation"]["bootstrap_seed"],
        "bootstrap_replicates": source_protocol["config"]["evaluation"]["bootstrap_replicates"],
    }
    if (ROOT / "protocol.json").exists():
        assert training.read(ROOT / "protocol.json") == policy
    else:
        json_write(ROOT / "protocol.json", policy)
    if (ROOT / "results.json").exists():
        result = training.read(ROOT / "results.json")
        assert result["protocol_sha256"] == digest(ROOT / "protocol.json")
        for name, sha in result["output_sha256"].items():
            assert digest(ROOT / name) == sha
        print("Existing frozen evaluation verified")
        return

    original = dict(np.load(SOURCE / "evaluation_inputs.npz", allow_pickle=False))
    reference = dict(np.load(SOURCE / "predictions.npz", allow_pickle=False))
    graphs = training.read(SOURCE / "graphs.json")
    np.testing.assert_array_equal(original["qids"], qids)
    np.testing.assert_array_equal(reference["qids"], qids)
    new_spins, new_graphs = [], []
    for i, q in enumerate(source_protocol["questions"]):
        assert q["qid"] == qids[i]
        episode = training.read(SOURCE / "episodes/validation" / (graph_study.episode_name(q) + ".json"))
        assert episode["statement"] == q["question"] and episode["choices"] == q["choices"]
        spins, _ = graph_study.episode_arrays(episode, q, digest(SOURCE / "source_protocol.json"), graphs[q["graph_id"]], original["seen_votes"][i, 0])
        new_spins.append(spins); new_graphs.append(graphs[q["graph_id"]])
    common = dict(np.load(model_root / "development.npz", allow_pickle=False))
    features = dict(np.load(model_root / "folds/main/features.npz", allow_pickle=False))
    lookup = {str(q): i for i, q in enumerate(common["episode_qids"])}
    indices = [lookup[q] for q in qids]
    context = features["context"][indices]
    np.testing.assert_array_equal(common["spins"][indices], original["seen_spins"])
    np.testing.assert_array_equal(common["J"][indices], original["seen_graph"])
    for graph in new_graphs:
        assert not any(np.array_equal(graph, g) for g in common["J"])
    regimes = {"seen": (original["seen_spins"], original["seen_graph"], original["seen_soft_targets"]),
               "unseen": (np.array(new_spins), np.array(new_graphs), reference["unseen_source_majority_A"])}
    baseline = training.read(model_root / "folds/main/ising.json")
    records = [{"training_questions": 80, **r} for r in training.read(SOURCE / "by_seed_question.json")]
    predictions, reproduction = {}, []
    for regime, (spins, graph, source_probability) in regimes.items():
        assert source_probability.shape == (20, 8, 32) and np.isfinite(source_probability).all()
        assert ((source_probability >= 0) & (source_probability <= 1)).all()
        pairs = np.array([(e, t) for e in range(20) for t in range(8)])
        e, t = pairs.T
        ising_lp = baseline_logp(baseline, spins[e, t], context[e], graph[e])
        data = base.FoldData({"spins": spins, "J": graph, "episode_qids": np.array(qids), "pairs": pairs},
                            {"context": context, "prior_logit": ising_lp[..., 1] - ising_lp[..., 0]}, "cpu")
        target = data.y.reshape(20, 8, 32)
        np.testing.assert_array_equal(target, reference[regime + "_target"])
        estimates = [("I0", -1, ising_lp)]
        for run in frozen["refits"]:
            checkpoint = Path(run["directory"]) / "best.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
            model = base.make_model(cfg, "nano")
            model.load_state_dict(state)
            manifest = training.read(Path(run["directory"]) / "manifest.json")
            assert base.model_weights_sha(model) == manifest["selected_weights_sha256"]
            lp = base.predictions(model, data, np.arange(160), False)
            # An independent float64 adapter checks the new graph masks and weights.
            alternate = graph_study.nano_logp({k: v.numpy() for k, v in state.items()}, spins[e, t], context[e], graph[e])
            np.testing.assert_allclose(np.exp(lp), np.exp(alternate), atol=2e-5, rtol=0)
            np.testing.assert_array_equal(lp.argmax(-1), alternate.argmax(-1))
            reproduction.append({"regime": regime, "seed": run["seed"], "all_classes_equal": True,
                "max_probability_difference": float(np.abs(np.exp(lp) - np.exp(alternate)).max())})
            estimates.append(("nano", run["seed"], lp))
        for model, seed, flat_lp in estimates:
            lp = flat_lp.reshape(20, 8, 32, 2)
            predictions[f"{regime}_{model}_{seed}_logp"] = lp
            for i, q in enumerate(source_protocol["questions"]):
                records.append({"training_questions": 320, "regime": regime, "model": model, "seed": seed,
                    "qid": qids[i], "graph_id": q["graph_id"] if regime == "unseen" else q["seen_graph_id"],
                    **graph_study.metrics(lp[i].astype(np.float64), target[i], source_probability[i])})
    rows = aggregate_records(records, qids)
    summary, by_graph = [], []
    for count in (80, 320):
        for regime in regimes:
            for model in ("I0", "nano"):
                selected = [r for r in rows if (r["training_questions"], r["regime"], r["model"]) == (count, regime, model)]
                assert len(selected) == 20
                summary.append({"training_questions": count, "regime": regime, "model": model,
                                **{k: float(np.mean([r[k] for r in selected])) for k in METRICS}})
                if regime == "unseen":
                    for g in sorted(graphs):
                        subset = [r for r in selected if r["graph_id"] == g]
                        assert len(subset) == 5
                        by_graph.append({"training_questions": count, "model": model, "graph_id": g,
                                         "questions": 5, **{k: float(np.mean([r[k] for r in subset])) for k in METRICS}})
    # The familiar-graph side must reproduce both earlier development reports.
    expanded = training.read(model_root.parent / "development_results/summary.json")
    for row in summary:
        if row["training_questions"] == 80:
            old = next(r for r in source_result["summary"] if (r["regime"], r["model"]) == (row["regime"], row["model"]))
            for k in METRICS:
                assert abs(row[k] - old[k]) < 1e-12
        elif row["regime"] == "seen":
            old = expanded["summary"]["retained_validation"][row["model"]]
            for k in ("error", "nll"):
                assert abs(row[k] - old[k]) < 1e-6
    terms = {}
    for count in (80, 320):
        for regime in regimes:
            terms[f"{count}_{regime}_nano_minus_I0"] = [(1, count, regime, "nano"), (-1, count, regime, "I0")]
        for model in ("I0", "nano"):
            terms[f"{count}_{model}_unseen_minus_seen"] = [(1, count, "unseen", model), (-1, count, "seen", model)]
        terms[f"{count}_difference_in_graph_effect"] = [(1, count, "unseen", "nano"), (-1, count, "seen", "nano"), (-1, count, "unseen", "I0"), (1, count, "seen", "I0")]
    for model in ("I0", "nano"):
        terms[f"{model}_unseen_320_minus_80"] = [(1, 320, "unseen", model), (-1, 80, "unseen", model)]
    comparisons = {key: paired_contrast(rows, value, qids, policy["bootstrap_seed"], policy["bootstrap_replicates"]) for key, value in terms.items()}
    np.savez_compressed(ROOT / "predictions.npz", **predictions)
    json_write(ROOT / "by_question.json", rows)
    json_write(ROOT / "by_seed_question.json", records)
    result = {"completed_utc": training.now(), "protocol_sha256": digest(ROOT / "protocol.json"),
        "summary": summary, "by_graph": by_graph, "paired_contrasts": comparisons,
        "numpy_cpu_reproduction": reproduction, "questions": 20, "graphs": 4, "transitions_per_regime": 5120,
        "models_refitted": False, "pca_refitted": False, "source_calls": 0, "assessment_loaded": False,
        "original_scores_reproduced": True, "expanded_familiar_scores_reproduced": True,
        "output_sha256": {name: digest(ROOT / name) for name in ("predictions.npz", "by_question.json", "by_seed_question.json")},
        "interpretation": policy["limitations"]}
    training.validate_frozen(cfg)
    graph_study.validate(SOURCE)
    for path, sha in sources.items():
        assert digest(path) == sha
    json_write(ROOT / "results.json", result)
    print(json.dumps({"summary": summary, "selected_contrasts": {k: {m: v for m, v in comparisons[k].items() if m in ("accuracy", "nll")} for k in ("320_unseen_nano_minus_I0", "nano_unseen_320_minus_80", "320_nano_unseen_minus_seen", "320_I0_unseen_minus_seen", "320_difference_in_graph_effect")}, "output": str(ROOT / "results.json")}, indent=2))


if __name__ == "__main__":
    main()
