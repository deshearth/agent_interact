"""Report frozen breadth predictors on development questions only."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from mac_agents.models.baselines import baseline_logp
from scripts import breadth_train as study
from scripts import transformer_generalization as base


def main():
    protocol = study.validate_prepared()
    study.authorize_split(study.ROOT, "assessment")  # Verify frozen hashes; do not load assessment.
    frozen_path = study.ROOT / "models_frozen.json"
    frozen = json.loads(frozen_path.read_text())
    cfg = study.predictor_config()
    torch.set_num_threads(cfg["training"]["torch_threads"])
    root = Path(cfg["output"])
    common = dict(np.load(root / "development.npz", allow_pickle=False))
    features = dict(np.load(root / "folds/main/features.npz", allow_pickle=False))
    data = base.FoldData(common, features, "cpu")
    fold = protocol["fold"]
    assert not set(common["episode_qids"]) & set(fold["assessment_questions"])
    ids = {split: data.ids(fold[split + "_questions"]) for split in ["train", "validation"]}
    selected = json.loads((root / "selection.json").read_text())
    baseline = json.loads((root / "folds/main/ising.json").read_text())
    summaries, by_question, reproduction, tuning = {}, [], [], []

    for split, indices in ids.items():
        e, t = common["pairs"][indices].T
        lp = baseline_logp(baseline, common["spins"][e, t], features["context"][e], common["J"][e])
        metrics, rows = base.metrics(lp, data.y[indices], data.qids[indices])
        summaries.setdefault("I0", {})[split] = metrics
        by_question.extend({"model": "I0", "seed": None, "split": split, **row} for row in rows)

    for candidate in protocol["candidates"]:
        directory = root / "tune/main" / f"nano-{candidate['id']}-s{cfg['training']['tuning_seed']}"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        tuning.append({"candidate": candidate, "steps": manifest["steps"],
                       "selected_step": manifest["selected_step"], "train": manifest["train"],
                       "validation": manifest["validation"], "stopping_reason": manifest["stopping_reason"]})
    best = min(tuning, key=lambda x: (x["validation"]["nll"], x["candidate"]["id"]))
    assert best["candidate"]["id"] == selected["candidate"]
    assert best["selected_step"] == selected["refit_steps"]
    assert [run["seed"] for run in frozen["refits"]] == cfg["training"]["refit_seeds"]

    by_seed = []
    runs = [{"directory": str(root / "tune/main" / f"nano-{selected['candidate']}-s{cfg['training']['tuning_seed']}"),
             "seed": cfg["training"]["tuning_seed"], "role": "validation_selected_tuning_model"}]
    runs += [{**run, "role": "fresh_seed_refit"} for run in frozen["refits"]]
    for run in runs:
        directory = Path(run["directory"])
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["initial_weights_sha256"] != manifest["selected_weights_sha256"]
        model = base.make_model(cfg, "nano")
        model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=False)["model"])
        assert base.model_weights_sha(model) == manifest["selected_weights_sha256"]
        record = {"seed": run["seed"], "role": run["role"], "directory": str(directory)}
        for split, indices in ids.items():
            lp = base.predictions(model, data, indices, False, cfg["training"]["evaluation_batch_size"])
            record[split], rows = base.metrics(lp, data.y[indices], data.qids[indices])
            saved = directory / f"{split}_logp.npy"
            if saved.exists():
                expected = np.exp(np.load(saved, allow_pickle=False))
                actual = np.exp(lp)
                np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5)
                reproduction.append({"seed": run["seed"], "role": run["role"], "split": split,
                                     "maximum_probability_difference": float(np.abs(actual - expected).max())})
            if run["role"] == "fresh_seed_refit":
                by_question.extend({"model": "nano", "seed": run["seed"], "split": split, **row} for row in rows)
        by_seed.append(record)

    refits = [run for run in by_seed if run["role"] == "fresh_seed_refit"]
    summaries["nano"] = {}
    for split in ids:
        summaries["nano"][split] = {
            metric: float(np.mean([run[split][metric] for run in refits]))
            for metric in ["nll", "error", "brier_two_class"]}
        summaries["nano"][split].update(questions=len(fold[split + "_questions"]),
                                        transitions=int(data.y[ids[split]].size))
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Development results only; validation selected hyperparameters and duration. These are not untouched assessment results.",
        "target": "Next agent opinion from five fresh source-model votes, not correctness against the math answer.",
        "aggregation": "Equal question weight; average three fresh-seed model scores, not ensemble predictions.",
        "models_frozen_sha256": digest(frozen_path),
        "report_code_sha256": digest(__file__),
        "protocol_sha256": digest(root / "protocol.json"),
        "parameters": protocol["parameters"],
        "shared_frozen_encoder": json.loads((root / "encoding.json").read_text()),
        "context_policy": protocol["encoder_policy"],
        "inputs": cfg["inputs"],
        "selected_candidate": best["candidate"],
        "refit_steps": selected["refit_steps"],
        "summary": summaries, "by_seed": by_seed, "tuning": tuning,
        "cpu_reproduction": reproduction,
        "invalid_vote_count": int((common["raw_votes"] == 0).sum()),
        "valid_transition_snapshots": len(common["pairs"]),
        "assessment_loaded": False,
    }
    # Reporting outputs stay outside the frozen predictor artifact directory.
    output = study.ROOT / "development_results"
    json_write(output / "by_question.json", by_question)
    result["by_question_sha256"] = digest(output / "by_question.json")
    json_write(output / "summary.json", result)
    study.validate_prepared()
    study.authorize_split(study.ROOT, "assessment")
    print(json.dumps({"summary": summaries, "selected_candidate": best["candidate"],
                      "refit_steps": selected["refit_steps"],
                      "maximum_cpu_probability_difference": max(x["maximum_probability_difference"] for x in reproduction),
                      "output": str(output / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
