"""Frozen-model evaluation for the 320-question learning-curve study."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from mac_agents.models.baselines import baseline_logp
from scripts import breadth_320_learning_curve as study
from scripts import breadth_expand_train as parent
from scripts import transformer_generalization as base

METRICS = ("nll", "error", "brier_two_class")


def validation_vectors(cfg, rows, qids):
    """Average seeds, then balanced subset paths; never replicate the endpoint."""
    result = {}
    for n in cfg["curve"]["training_question_counts"]:
        result[n] = {}
        paths = 1 if n == cfg["counts"]["train"] else len(cfg["curve"]["subset_seeds"])
        for kind in ["I0"] + cfg["model_order"]:
            seeds = 1 if kind == "I0" else len(cfg["training"]["refit_seeds"])
            selected = [r for r in rows if r["n_questions"] == n and r["model"] == kind and r["split"] == "validation"]
            assert len(selected) == len(qids) * paths * seeds
            values = []
            for q in qids:
                matches = [r for r in selected if r["qid"] == q]
                assert len(matches) == len({(r["fold"], r["seed"]) for r in matches}) == paths * seeds
                assert len({r["fold"] for r in matches}) == paths
                values.append([np.mean([r[m] for r in matches]) for m in METRICS])
            result[n][kind] = np.asarray(values)
    return result


def paired_vector(delta, draws):
    assert delta.shape[1] == len(METRICS)
    sampled = delta[draws].mean(1)
    lo, hi = np.quantile(sampled, [.025, .975], axis=0)
    return {metric: {"difference": float(delta[:, j].mean()), "lower": float(lo[j]),
        "upper": float(hi[j]), "questions": len(delta), "questions_better": int((delta[:, j] < 0).sum())}
        for j, metric in enumerate(METRICS)}


def contrasts(cfg, vectors):
    counts = cfg["curve"]["training_question_counts"]
    qcount = len(next(iter(vectors[counts[0]].values())))
    draws = np.random.default_rng(cfg["evaluation"]["bootstrap_seed"]).integers(
        0, qcount, size=(cfg["evaluation"]["bootstrap_replicates"], qcount))
    pairs = list(zip(counts[:-1], counts[1:]))
    if (counts[0], counts[-1]) not in pairs:
        pairs.append((counts[0], counts[-1]))
    reference = cfg["evaluation"]["reference_model"]
    growth, interaction, sizes, baseline = {}, {}, {}, {}
    for low, high in pairs:
        key = f"{low}_to_{high}"
        changes = {k: vectors[high][k] - vectors[low][k] for k in ["I0"] + cfg["model_order"]}
        growth[key] = {k: paired_vector(v, draws) for k, v in changes.items()}
        interaction[key] = {k + "_minus_" + reference: paired_vector(changes[k] - changes[reference], draws)
                            for k in cfg["model_order"] if k != reference}
    for n in counts:
        sizes[str(n)] = {k + "_minus_" + reference: paired_vector(vectors[n][k] - vectors[n][reference], draws)
                         for k in cfg["model_order"] if k != reference}
        baseline[str(n)] = {k + "_minus_I0": paired_vector(vectors[n][k] - vectors[n]["I0"], draws)
                            for k in cfg["model_order"]}
    return {"data_growth": growth, "data_growth_interaction": interaction, "model_size": sizes,
            "versus_Ising": baseline,
            "directions": "All contrasts are differences in loss/error. Data growth is higher-count minus lower-count. Interaction subtracts the 818-model change; negative values mean the larger model benefits more from added questions. Accuracy changes are negative error changes."}


def evaluate(cfg):
    root = Path(cfg["output"])
    frozen = study.validate_frozen(cfg)
    protocol = parent.read(root / "protocol.json")
    selection = parent.read(root / "selection.json")
    output = root.parent / "results"
    if (output / "complete.json").exists():
        done = parent.read(output / "complete.json")
        assert done["models_frozen_sha256"] == digest(root.parent / "models_frozen.json")
        for name, sha in done["artifacts"].items():
            assert digest(output / name) == sha
        print("Existing learning-curve results verified")
        return parent.read(output / "summary.json")
    output.mkdir(parents=True, exist_ok=True)
    common = dict(np.load(root / "development.npz", allow_pickle=False))
    e, t = common["pairs"].T
    records, rows, reproduction, tuning = [], [], [], []
    folds = protocol["folds"]
    qids = sorted(folds[0]["validation_questions"])
    assert len(qids) == cfg["counts"]["validation"]

    def score(fold, data, kind, seed, split, ids, lp):
        metrics, details = base.metrics(lp, data.y[ids], data.qids[ids])
        assert abs(metrics["error"] - float((lp.argmax(-1) != data.y[ids]).mean())) < 1e-12
        assert abs(metrics["nll"] + float(np.take_along_axis(lp.astype(float), data.y[ids][..., None], -1).mean())) < 1e-12
        labels = {"fold": fold["id"], "path": fold["path"], "n_questions": fold["n_questions"],
                  "model": kind, "seed": seed, "split": split}
        records.append({**labels, **metrics})
        rows.extend({**labels, **r} for r in details)
        np.save(output / f"{fold['id']}_{kind}_s{seed}_{split}_logp.npy", lp, allow_pickle=False)

    for fold in folds:
        assert sorted(fold["validation_questions"]) == qids
        assert not set(fold["train_questions"]) & set(qids)
        assert not set(common["episode_qids"]) & set(fold["assessment_questions"])
        features = dict(np.load(root / "folds" / fold["id"] / "features.npz", allow_pickle=False))
        baseline = parent.read(root / "folds" / fold["id"] / "ising.json")
        data = base.FoldData(common, features, "cpu")
        ids = {s: data.ids(fold[s + "_questions"]) for s in ("train", "validation")}
        lp = baseline_logp(baseline, common["spins"][e, t], features["context"][e], common["J"][e])
        for split in ids:
            score(fold, data, "I0", -1, split, ids[split], lp[ids[split]])
        for kind in cfg["model_order"]:
            choice = next(c for c in selection["choices"] if c["fold"] == fold["id"] and c["model"] == kind)
            runs = sorted((r for r in frozen["refits"] if r["fold"] == fold["id"] and r["model"] == kind), key=lambda r: r["seed"])
            assert [r["seed"] for r in runs] == cfg["training"]["refit_seeds"]
            for run in runs:
                directory = Path(run["directory"])
                manifest = parent.read(directory / "manifest.json")
                assert manifest["steps"] == manifest["selected_step"] == choice["refit_steps"]
                assert manifest["validation"] is None and manifest["candidate"] == choice["candidate"]
                assert manifest["training_questions"] == fold["train_questions"]
                model = base.make_model(cfg, kind)
                model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=False)["model"])
                assert base.model_weights_sha(model) == manifest["selected_weights_sha256"]
                assert sum(p.numel() for p in model.parameters()) == protocol["parameters"][kind]
                for split in ids:
                    lp = base.predictions(model, data, ids[split], False, cfg["training"]["evaluation_batch_size"])
                    score(fold, data, kind, run["seed"], split, ids[split], lp)
                    if split == "train":
                        expected = np.load(directory / "train_logp.npy", allow_pickle=False)
                        np.testing.assert_allclose(np.exp(lp), np.exp(expected), atol=5e-5, rtol=5e-5)
                        reproduction.append({"fold": fold["id"], "model": kind, "seed": run["seed"], "split": split,
                            "max_probability_difference": float(abs(np.exp(lp) - np.exp(expected)).max()),
                            "class_disagreements": int((lp.argmax(-1) != expected.argmax(-1)).sum())})
            # Independently reproduce the checkpoint that selected this condition.
            directory = Path(choice["tuning_directory"])
            manifest = parent.read(directory / "manifest.json")
            model = base.make_model(cfg, kind)
            model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=False)["model"])
            assert base.model_weights_sha(model) == manifest["selected_weights_sha256"]
            for split in ids:
                lp = base.predictions(model, data, ids[split], False, cfg["training"]["evaluation_batch_size"])
                expected = np.load(directory / f"{split}_logp.npy", allow_pickle=False)
                np.testing.assert_allclose(np.exp(lp), np.exp(expected), atol=5e-5, rtol=5e-5)
                reproduction.append({"fold": fold["id"], "model": kind, "seed": cfg["training"]["tuning_seed"], "split": split,
                    "max_probability_difference": float(abs(np.exp(lp) - np.exp(expected)).max()),
                    "class_disagreements": int((lp.argmax(-1) != expected.argmax(-1)).sum())})
            for candidate in protocol["candidates"]:
                path = directory.parent / f"{kind}-{candidate['id']}-s{cfg['training']['tuning_seed']}"
                m = parent.read(path / "manifest.json")
                tuning.append({"fold": fold["id"], "n_questions": fold["n_questions"], "model": kind,
                    "candidate": candidate, **{k: m[k] for k in ("selected_step", "steps", "train", "validation", "seconds", "stopping_reason")}})
        print(json.dumps({"scored_fold": fold["id"], "training_questions": fold["n_questions"]}), flush=True)

    summary, by_subset = {}, []
    for fold in folds:
        for kind in ["I0"] + cfg["model_order"]:
            for split in ("train", "validation"):
                rs = [r for r in records if r["fold"] == fold["id"] and r["model"] == kind and r["split"] == split]
                assert len(rs) == (1 if kind == "I0" else len(cfg["training"]["refit_seeds"]))
                by_subset.append({"fold": fold["id"], "path": fold["path"], "n_questions": fold["n_questions"],
                    "model": kind, "split": split, **{m: float(np.mean([r[m] for r in rs])) for m in METRICS},
                    "seed_ranges": {m: [min(r[m] for r in rs), max(r[m] for r in rs)] for m in METRICS}})
    vectors = validation_vectors(cfg, rows, qids)
    for n in cfg["curve"]["training_question_counts"]:
        summary[str(n)] = {}
        for kind in ["I0"] + cfg["model_order"]:
            summary[str(n)][kind] = {}
            for split in ("train", "validation"):
                rs = [r for r in by_subset if r["n_questions"] == n and r["model"] == kind and r["split"] == split]
                expected = 1 if n == cfg["counts"]["train"] else len(cfg["curve"]["subset_seeds"])
                assert len(rs) == expected
                summary[str(n)][kind][split] = {m: float(np.mean([r[m] for r in rs])) for m in METRICS}
                summary[str(n)][kind][split].update(subset_paths=expected,
                    questions=n if split == "train" else len(qids),
                    transitions=(n if split == "train" else len(qids)) * 256,
                    subset_ranges={m: [min(r[m] for r in rs), max(r[m] for r in rs)] for m in METRICS})
                if split == "validation":
                    for j, m in enumerate(METRICS):
                        assert abs(summary[str(n)][kind][split][m] - vectors[n][kind][:, j].mean()) < 1e-12
    previous = parent.read(Path(cfg["size_reference"]) / "results/summary.json")
    for kind in ["I0"] + cfg["model_order"]:
        for split in ("train", "validation"):
            for metric in METRICS:
                assert abs(summary[str(cfg["counts"]["train"])][kind][split][metric] - previous["summary"][kind][split][metric]) < 1e-8
    costs = {"new_tuning_seconds": sum(r["seconds"] for r in tuning if r["n_questions"] < cfg["counts"]["train"]),
             "new_refit_seconds": sum(parent.read(Path(r["directory"]) / "manifest.json")["seconds"] for r in frozen["refits"] if not r["reused"])}
    result = {"completed_utc": parent.now(), "protocol_sha256": digest(root / "protocol.json"),
        "models_frozen_sha256": digest(root.parent / "models_frozen.json"),
        "parameters": {"I0": 19, **protocol["parameters"]}, "summary": summary, "by_seed": records,
        "by_subset": by_subset, "paired_differences": contrasts(cfg, vectors), "choices": selection["choices"],
        "tuning": tuning, "cpu_reproduction": reproduction, "training_cost": costs,
        "aggregation": "Equal question means within a fit; average three seed scores within each subset, then three subset paths at smaller counts. Full data is one shared endpoint with three seeds. No ensembling or duplicated endpoint.",
        "scope": cfg["evaluation"], "reference_scores_reproduced": True,
        "source_calls": 0, "assessment_loaded": False, "unseen_graphs_scored": False}
    json_write(output / "by_question.json", rows)
    json_write(output / "summary.json", result)
    study.validate_frozen(cfg)
    json_write(output / "complete.json", {"completed_utc": parent.now(), "models_frozen_sha256": digest(root.parent / "models_frozen.json"),
        "artifacts": {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "complete.json"},
        "new_tuning_fits": protocol["new_tuning_fits"], "new_refits": protocol["new_refits"],
        "reference_scores_reproduced": True, "assessment_loaded": False, "source_calls": 0})
    print(json.dumps({"completed": True, "output": str(output / "summary.json"), "summary": summary}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    cfg = study.config()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    evaluate(cfg)
