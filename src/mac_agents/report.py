from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import code_digest, json_write
from .metrics import paired_bootstrap


def report(experiment="paper_binary"):
    rows, runs = [], []
    for path in sorted(Path("runs").glob("*/manifest.json")):
        run = path.parent
        if not (run / "validation_metrics.json").exists():
            continue
        meta = json.loads(path.read_text())
        if meta["smoke"] and meta["code_sha256"] != code_digest():
            continue
        scores = json.loads((run / "validation_metrics.json").read_text())
        rows.append({"run": run.name, "model": meta["kind"], "state": meta["state"], "context": meta["config"]["data"]["context"], "seed": meta["seed"],
                     "preparation_only": meta["smoke"], "parameters": meta["parameters"],
                     "validation_nll": scores["nll"], "validation_brier": scores["brier"],
                     "validation_fcba": scores["fcba"], "device": meta["actual_device"],
                     "seconds": meta["elapsed_seconds_this_session"], "stopping_reason": meta["stopping_reason"]})
        runs.append((run, meta))
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(f"reports/{experiment}_runs.csv", index=False)
    comparisons = []
    for run, meta in runs:
        if meta["smoke"] or meta["state"] == "opinion":
            continue
        for reference, refmeta in runs:
            if refmeta["smoke"] or refmeta["kind"] != meta["kind"] or refmeta["seed"] != meta["seed"] or refmeta["state"] != "opinion":
                continue
            if refmeta["data_fingerprint"]["split_sha256"] != meta["data_fingerprint"]["split_sha256"]:
                continue
            a = pd.read_parquet(run / "validation_predictions.parquet")
            b = pd.read_parquet(reference / "validation_predictions.parquet")
            keys = ["episode_id", "round", "agent"]
            if set(map(tuple, a[keys].values)) != set(map(tuple, b[keys].values)):
                continue
            qa = pd.read_csv(run / "validation_by_question.csv").set_index("qid").nll
            qb = pd.read_csv(reference / "validation_by_question.csv").set_index("qid").nll
            comparisons.append({"run": str(run), "reference": str(reference), "metric": "NLL",
                                "paired_question_bootstrap": paired_bootstrap(qa, qb)})
    json_write(f"reports/{experiment}_paired_validation.json", comparisons)
    text = ["# Experiment preparation and results", "", "Preparation runs validate mechanics and are excluded from scientific model selection. Earlier preparation artifacts from superseded code remain in runs/ but are omitted here.", "",
            "Author reproduction is reported separately in `author_reproduction.md`. Controlled test evaluation requires a saved selection record.", "",
            "| Model | State | Context | Seed | Preparation only | Parameters | Validation NLL | Stop reason |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | --- |"]
    for r in rows:
        value = f"{r['validation_nll']:.5f}" if r["validation_nll"] is not None else "not scored"
        text.append(f"| {r['model']} | {r['state']} | {r['context']} | {r['seed']} | {r['preparation_only']} | {r['parameters']} | {value} | {r['stopping_reason']} |")
    text += ["", "No comparative scientific conclusion follows from short preparation runs.",
             "Full selected runs need question-level uncertainty, seed variation, graph-family results, and autonomous-rollout evaluation."]
    Path(f"reports/{experiment}_results.md").write_text("\n".join(text) + "\n")
    print(f"Wrote reports/{experiment}_results.md ({len(rows)} runs)")
