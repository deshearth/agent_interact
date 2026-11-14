"""Deferred fixed-model scoring and capacity diagnostics for the 320 size sweep."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from mac_agents.models.baselines import baseline_logp
from scripts import breadth_320_size_sweep as study
from scripts import breadth_expand_train as parent
from scripts import transformer_generalization as base
from scripts import unseen_graph_experiment as graph_study

METRICS = ("nll", "error", "brier_two_class")


def prepare_unseen(cfg, common, features, baseline):
    source = Path(cfg["unseen_source"])
    protocol = parent.read(source / "protocol.json")
    original = dict(np.load(source / "evaluation_inputs.npz", allow_pickle=False))
    graph_map = parent.read(source / "graphs.json")
    result = parent.read(source / "results.json")
    qids = protocol["validation_question_ids"]
    assert len(qids) == len(set(qids)) == cfg["evaluation"]["unseen_questions"]
    assert len(graph_map) == cfg["evaluation"]["unseen_graphs"]
    np.testing.assert_array_equal(qids, original["qids"])
    lookup = {str(q): i for i, q in enumerate(common["episode_qids"])}
    positions = [lookup[q] for q in qids]
    np.testing.assert_array_equal(common["spins"][positions], original["seen_spins"])
    np.testing.assert_array_equal(common["J"][positions], original["seen_graph"])
    spins, graphs = [], []
    for i, q in enumerate(protocol["questions"]):
        assert q["qid"] == qids[i]
        path = source / "episodes/validation" / (graph_study.episode_name(q) + ".json")
        assert digest(path) == result["scored_episode_sha256"][str(path)]
        episode = parent.read(path)
        assert episode["statement"] == q["question"] and episode["choices"] == q["choices"]
        s, _ = graph_study.episode_arrays(episode, q, digest(source / "source_protocol.json"), graph_map[q["graph_id"]], original["seen_votes"][i, 0])
        graph = np.asarray(graph_map[q["graph_id"]], dtype=np.int8)
        assert not any(np.array_equal(graph, g) for g in common["J"])
        spins.append(s); graphs.append(graph)
    pairs = np.array([(e, t) for e in range(len(qids)) for t in range(8)], dtype=np.int64)
    new = {"spins": np.array(spins), "J": np.array(graphs), "pairs": pairs, "episode_qids": np.array(qids)}
    context = features["context"][positions]
    e, t = pairs.T
    lp = baseline_logp(baseline, new["spins"][e, t], context[e], new["J"][e])
    transformed = {"context": context, "prior_logit": lp[..., 1] - lp[..., 0]}
    data = base.FoldData(new, transformed, "cpu")
    return data, lp, {q["qid"]: q["graph_id"] for q in protocol["questions"]}


def question_means(rows, split, model, qids, seed_count=3):
    selected = [r for r in rows if r["split"] == split and r["model"] == model]
    expected = 1 if model == "I0" else seed_count
    assert len(selected) == len(qids) * expected
    assert {r["qid"] for r in selected} == set(qids)
    for q in qids:
        subset = [r for r in selected if r["qid"] == q]
        assert len(subset) == expected and len({r["seed"] for r in subset}) == expected
    return parent.collapse(selected, model, qids)


def comparisons(cfg, rows, groups):
    result = {}
    for split in ("validation", "graph_seen", "graph_unseen"):
        selected = [r for r in rows if r["split"] == split]
        result[split] = {}
        for kind in cfg["model_order"]:
            for reference in cfg["evaluation"]["compare_against"]:
                if kind == reference:
                    continue
                result[split][f"{kind}_minus_{reference}"] = parent.paired(selected, kind, reference, groups[split], cfg)
    return result


def evaluate(cfg):
    root = Path(cfg["output"])
    frozen = study.validate_frozen(cfg)
    protocol = parent.read(root / "protocol.json")
    selected = parent.read(root / "selection.json")
    output = root.parent / "results"
    if (output / "complete.json").exists():
        done = parent.read(output / "complete.json")
        assert done["models_frozen_sha256"] == digest(root.parent / "models_frozen.json")
        for name, sha in done["artifacts"].items():
            assert digest(output / name) == sha
        print("Existing size-sweep results verified")
        return parent.read(output / "summary.json")
    output.mkdir(parents=True, exist_ok=True)
    common = dict(np.load(root / "development.npz", allow_pickle=False))
    features = dict(np.load(root / "folds/main/features.npz", allow_pickle=False))
    baseline = parent.read(root / "folds/main/ising.json")
    assert baseline["parameters"] == 19
    data = base.FoldData(common, features, "cpu")
    fold = protocol["fold"]
    unseen, unseen_ising, graph_map = prepare_unseen(cfg, common, features, baseline)
    graph_qids = sorted(graph_map)
    assert set(graph_qids) <= set(fold["validation_questions"])
    assert not set(graph_qids) & set(fold["train_questions"])
    groups = {"train": fold["train_questions"], "validation": fold["validation_questions"],
              "graph_seen": graph_qids, "graph_unseen": graph_qids}
    ids = {split: data.ids(qids) for split, qids in groups.items() if split != "graph_unseen"}
    ids["graph_unseen"] = np.arange(len(unseen.qids))
    assert not set(data.qids) & set(fold["assessment_questions"])
    assert not set(unseen.qids) & set(fold["assessment_questions"])
    records, rows, reproduction, diagnostics, tuning = [], [], [], [], []

    def score(model, seed, split, lp, save=True):
        values = unseen if split == "graph_unseen" else data
        positions = ids[split]
        aggregate, details = base.metrics(lp, values.y[positions], values.qids[positions])
        # Independent flat computation is equivalent because every question has 256 cases.
        target = values.y[positions]
        assert abs(aggregate["error"] - float((lp.argmax(-1) != target).mean())) < 1e-12
        assert abs(aggregate["nll"] + float(np.take_along_axis(lp.astype(float), target[..., None], -1).mean())) < 1e-12
        records.append({"model": model, "seed": seed, "split": split, **aggregate})
        rows.extend({"model": model, "seed": seed, "split": split, **r} for r in details)
        if save:
            np.save(output / f"{model}_s{seed}_{split}_logp.npy", lp, allow_pickle=False)
        return aggregate

    e, t = common["pairs"].T
    baseline_lp = baseline_logp(baseline, common["spins"][e, t], features["context"][e], common["J"][e])
    for split in groups:
        score("I0", -1, split, unseen_ising if split == "graph_unseen" else baseline_lp[ids[split]])
    refits = frozen["refits"]
    for kind in cfg["model_order"]:
        runs = sorted((r for r in refits if r["model"] == kind), key=lambda r: r["seed"])
        assert [r["seed"] for r in runs] == cfg["training"]["refit_seeds"]
        choice = next(r for r in selected["choices"] if r["model"] == kind)
        for run in runs:
            directory = Path(run["directory"])
            manifest = parent.read(directory / "manifest.json")
            assert manifest["selected_step"] == manifest["steps"] == choice["refit_steps"]
            assert manifest["validation"] is None and manifest["candidate"] == choice["candidate"]
            model = base.make_model(cfg, kind)
            model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=False)["model"])
            assert base.model_weights_sha(model) == manifest["selected_weights_sha256"]
            assert sum(p.numel() for p in model.parameters()) == protocol["parameters"][kind]
            for split in groups:
                values = unseen if split == "graph_unseen" else data
                lp = base.predictions(model, values, ids[split], False, cfg["training"]["evaluation_batch_size"])
                score(kind, run["seed"], split, lp)
                saved = directory / f"{split}_logp.npy"
                if split in ("train", "validation") and saved.exists():
                    expected = np.load(saved, allow_pickle=False)
                    np.testing.assert_allclose(np.exp(lp), np.exp(expected), atol=5e-5, rtol=5e-5)
                    reproduction.append({"model": kind, "seed": run["seed"], "split": split,
                        "max_probability_difference": float(np.abs(np.exp(lp) - np.exp(expected)).max()),
                        "class_disagreements": int((lp.argmax(-1) != expected.argmax(-1)).sum())})
            print(json.dumps({"scored_model": kind, "seed": run["seed"], "reused": run["reused"]}), flush=True)
        # Diagnose stopping versus fitting without selecting any further model.
        candidate_dir = Path(choice["tuning_directory"])
        manifest = parent.read(candidate_dir / "manifest.json")
        entry = {"model": kind, "seed": cfg["training"]["tuning_seed"], "candidate": choice["candidate"],
                 "selected_step": manifest["selected_step"], "terminal_step": manifest["steps"],
                 "stopping_reason": manifest["stopping_reason"], "selected_at_cap": manifest["selected_step"] == cfg["training"]["max_steps"]}
        for checkpoint, label, sha_key in (("best.pt", "selected", "selected_weights_sha256"), ("resume.pt", "terminal", "terminal_weights_sha256")):
            model = base.make_model(cfg, kind)
            model.load_state_dict(torch.load(candidate_dir / checkpoint, map_location="cpu", weights_only=False)["model"])
            assert base.model_weights_sha(model) == manifest[sha_key]
            entry[label] = {}
            for split in ("train", "validation"):
                lp = base.predictions(model, data, ids[split], False, cfg["training"]["evaluation_batch_size"])
                entry[label][split], _ = base.metrics(lp, data.y[ids[split]], data.qids[ids[split]])
                if label == "selected":
                    expected = np.load(candidate_dir / f"{split}_logp.npy", allow_pickle=False)
                    np.testing.assert_allclose(np.exp(lp), np.exp(expected), atol=5e-5, rtol=5e-5)
                    reproduction.append({"model": kind, "seed": cfg["training"]["tuning_seed"], "split": split,
                        "max_probability_difference": float(np.abs(np.exp(lp) - np.exp(expected)).max()),
                        "class_disagreements": int((lp.argmax(-1) != expected.argmax(-1)).sum())})
        diagnostics.append(entry)
        source = Path(cfg["reference"]) if kind == cfg["reused_model"] else root
        for candidate in protocol["candidates"]:
            path = source / "tune/main" / f"{kind}-{candidate['id']}-s{cfg['training']['tuning_seed']}"
            m = parent.read(path / "manifest.json")
            tuning.append({"model": kind, "candidate": candidate, **{k: m[k] for k in ("selected_step", "steps", "train", "validation", "seconds", "stopping_reason")}})

    summary = {}
    for kind in ["I0"] + cfg["model_order"]:
        summary[kind] = {}
        for split, qids in groups.items():
            values = question_means(rows, split, kind, qids, len(cfg["training"]["refit_seeds"]))
            summary[kind][split] = {m: float(np.mean([r[m] for r in values.values()])) for m in METRICS}
            summary[kind][split].update(questions=len(qids), transitions=len(qids) * 256)
    previous = parent.read(Path(cfg["reference"]).parent / "development_results/summary.json")
    for kind in ("I0", cfg["reused_model"]):
        for split in ("train", "validation"):
            for metric in METRICS:
                assert abs(summary[kind][split][metric] - previous["summary"][split][kind][metric]) < 1e-6
    previous_graph = parent.read(Path(cfg["unseen_reference"]) / "results.json")
    for kind in ("I0", cfg["reused_model"]):
        for regime in ("seen", "unseen"):
            expected = next(r for r in previous_graph["summary"] if r["training_questions"] == cfg["counts"]["train"] and r["model"] == kind and r["regime"] == regime)
            for metric in ("error", "nll"):
                assert abs(summary[kind]["graph_" + regime][metric] - expected[metric]) < 1e-6
    paired = comparisons(cfg, rows, groups)
    by_graph = []
    for kind in ["I0"] + cfg["model_order"]:
        values = question_means(rows, "graph_unseen", kind, groups["graph_unseen"], len(cfg["training"]["refit_seeds"]))
        for graph in sorted(set(graph_map.values())):
            members = [q for q in graph_map if graph_map[q] == graph]
            by_graph.append({"model": kind, "graph_id": graph, "questions": len(members),
                             **{m: float(np.mean([values[q][m] for q in members])) for m in METRICS}})
    costs = {}
    for kind in cfg["model_order"]:
        seconds = sum(r["seconds"] for r in tuning if r["model"] == kind)
        seconds += sum(parent.read(Path(r["directory"]) / "manifest.json")["seconds"] for r in refits if r["model"] == kind)
        costs[kind] = {"fit_seconds": seconds, "reused": kind == cfg["reused_model"]}
    result = {"completed_utc": parent.now(), "protocol_sha256": digest(root / "protocol.json"),
        "models_frozen_sha256": digest(root.parent / "models_frozen.json"), "parameters": {"I0": 19, **protocol["parameters"]},
        "summary": summary, "by_seed": records, "paired_differences": paired, "by_unseen_graph": by_graph,
        "choices": selected["choices"], "selected_tuning_terminal_diagnostic": diagnostics, "tuning": tuning,
        "cpu_reproduction": reproduction, "training_cost": costs, "reference_scores_reproduced": True,
        "aggregation": "Equal question means; average individual scores over three refit seeds, not an ensemble.",
        "scope": cfg["evaluation"]["interpretation"], "terminal_diagnostic_scope": "Same selected candidate's later validation-stopped checkpoint, seed42 only. This is not a maximum-capacity or unregularized memorization experiment.",
        "unseen_scores_selected_models": False, "source_calls": 0, "assessment_loaded": False}
    json_write(output / "by_question.json", rows)
    json_write(output / "summary.json", result)
    study.validate_frozen(cfg)
    json_write(output / "complete.json", {"completed_utc": parent.now(), "models_frozen_sha256": digest(root.parent / "models_frozen.json"),
        "artifacts": {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "complete.json"},
        "new_fits": protocol["new_neural_fits"], "reused_fits": protocol["reused_neural_fits"],
        "assessment_loaded": False, "source_calls": 0, "reference_scores_reproduced": True})
    print(json.dumps({"parameters": result["parameters"], "summary": summary, "output": str(output / "summary.json")}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    cfg = study.config()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    evaluate(cfg)
