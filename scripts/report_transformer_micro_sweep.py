"""Deferred assessment and reproducible report for the transformer size sweep."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from scripts import transformer_generalization as base
from scripts.transformer_micro_sweep import CONFIG, reference_runs, validate

LABELS = {"I0": "Ising", "I0_current": "Ising + current-opinion correction",
          "pico": "Transformer 314", "nano": "Transformer 818", "micro": "Transformer 2,626", "tiny": "Transformer 7,370"}
ORDER = ["I0", "I0_current", "pico", "nano", "micro", "tiny"]


def now():
    return datetime.now(timezone.utc).isoformat()


def log_prob(score):
    score = np.asarray(score, dtype=np.float64)
    return np.stack([-np.logaddexp(0, score), -np.logaddexp(0, -score)], -1)


def evaluate(cfg, device):
    root, reference = Path(cfg["output"]), Path(cfg["reference"])
    protocol = validate(cfg)
    tuning = base.completed_runs(root, "tune")
    new = base.completed_runs(root, "refit")
    reused = reference_runs(cfg, "refit")
    assert len(tuning) == 216 and len(new) == 36 and len(reused) == 12
    selection = json.loads((root/"selection.json").read_text())
    refit_done = json.loads((root/"refit_complete.json").read_text())
    assert selection["frozen_utc"] <= refit_done["completed_utc"]
    assert not selection["selection_uses_outer_outcomes"]
    strengthened = Path(cfg["strengthened_baseline"])
    corrections = json.loads((strengthened/"refits.json").read_text())
    baseline_previous = pd.read_csv(strengthened/"by_question.csv")
    identity = {"protocol_sha256": digest(root/"protocol.json"), "selection_sha256": digest(root/"selection.json"),
        "refit_complete_sha256": digest(root/"refit_complete.json"), "report_code_sha256": digest(__file__),
        "reference_refits_sha256": digest(reference/"refit_complete.json"),
        "strengthened_baseline_sha256": digest(strengthened/"refits.json")}
    if (root/"assessment_freeze.json").exists():
        freeze = json.loads((root/"assessment_freeze.json").read_text())
        assert all(freeze[k] == v for k, v in identity.items())
    else:
        json_write(root/"assessment_freeze.json", {**identity, "frozen_utc": now(),
            "new_refits": 36, "reused_refits": 12, "fresh_confirmatory_dataset": False})
    common = dict(np.load(root/"development.npz", allow_pickle=False))
    index = json.loads((root/"episode_index.json").read_text())
    assert all(r["official_split"] == "train" and r["graph_role"] == "fit_graph" for r in index)
    records, by_question, reproduction = [], [], []
    for outer in range(4):
        fold = next(f for f in protocol["folds"] if f["id"] == f"o{outer}_refit")
        features = dict(np.load(root/"folds"/fold["id"]/"features.npz", allow_pickle=False))
        data = base.FoldData(common, features, device)
        cpu_data = base.FoldData(common, features, "cpu")
        train, assessment = data.ids(fold["train_questions"]), data.ids(fold["assessment_questions"])
        correction = next(c for c in corrections if c["outer"] == outer and c["model"] == "current_state")
        e, t = common["pairs"].T
        z = features["prior_logit"]
        x = np.stack([np.ones_like(z), z, common["spins"][e, t].astype(float)], -1)
        augmented = z+np.sum((x-np.array(correction["mean"]))/np.array(correction["scale"])*np.array(correction["weights"]), -1)
        for kind, scores in (("I0", z), ("I0_current", augmented)):
            for split, ids in (("train", train), ("assessment", assessment)):
                result, rows = base.metrics(log_prob(scores[ids]), data.y[ids], data.qids[ids])
                fields = {"model": kind, "outer": outer, "seed": -1, "split": split, "reused": True}
                records.append({**fields, **result}); by_question.extend({**fields, **r} for r in rows)
                if kind == "I0_current" and split == "assessment":
                    for row in rows:
                        old = baseline_previous[(baseline_previous.model == "current_state") & (baseline_previous.qid == row["qid"])].iloc[0]
                        for metric in ("nll", "error", "brier_two_class"):
                            np.testing.assert_allclose(row[metric], old[metric], atol=1e-10)
        for directory, manifest in [(p, m) for p, m in new+reused if m["outer"] == outer]:
            kind, is_reused = manifest["model"], manifest["model"] == cfg["reused_model"]
            choices = selection["reused_choices"] if is_reused else selection["choices"]
            chosen = next(c for c in choices if c["outer"] == outer and c["model"] == kind)
            assert chosen["candidate"] == manifest["candidate"]
            assert chosen["refit_steps"] == manifest["steps"] == manifest["selected_step"]
            if not is_reused:
                assert selection["frozen_utc"] <= manifest["created_utc"] <= manifest["completed_utc"]
            name = f"{fold['id']}__{directory.name}"
            output = root/"assessment"/name
            output.mkdir(parents=True, exist_ok=True)
            local_identity = {**identity, "run_directory": str(directory), "run_manifest_sha256": digest(directory/"manifest.json")}
            if (output/"complete.json").exists():
                previous = json.loads((output/"complete.json").read_text())
                assert all(previous[k] == v for k, v in local_identity.items())
                for path, sha in previous["artifacts"].items():
                    assert digest(output/path) == sha
                records += previous["records"]; by_question += previous["by_question"]
                reproduction.append(previous["reproduction"])
                continue
            model = base.make_model(cfg, kind).to(device)
            checkpoint = torch.load(directory/"best.pt", map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            assert base.model_weights_sha(model) == manifest["selected_weights_sha256"]
            assert sum(p.numel() for p in model.parameters()) == protocol["parameters"][kind]
            if is_reused:
                source = reference/"assessment"/name/"assessment_logp.npy"
                lp = np.load(source, allow_pickle=False)
            else:
                lp = base.predictions(model, data, assessment, False, 256)
            np.save(output/"assessment_logp.npy", lp, allow_pickle=False)
            train_lp = np.load(directory/"train_logp.npy", allow_pickle=False)
            own, own_q = [], []
            for split, ids, values in (("train", train, train_lp), ("assessment", assessment, lp)):
                result, rows = base.metrics(values, data.y[ids], data.qids[ids])
                fields = {"model": kind, "outer": outer, "seed": manifest["seed"], "split": split,
                          "run": name, "reused": is_reused}
                own.append({**fields, **result}); own_q.extend({**fields, **r} for r in rows)
                if split == "train":
                    for metric in ("nll", "error", "brier_two_class"):
                        np.testing.assert_allclose(result[metric], manifest["train"][metric], atol=1e-7, rtol=1e-6)
            cpu = base.make_model(cfg, kind)
            cpu.load_state_dict(torch.load(directory/"best.pt", map_location="cpu", weights_only=False)["model"])
            maximum = 0.
            for ids, values in ((train, train_lp), (assessment, lp)):
                positions = np.linspace(0, len(ids)-1, 64, dtype=int)
                actual = np.exp(base.predictions(cpu, cpu_data, ids[positions], False, 64)[..., 1])
                expected = np.exp(values[positions, ..., 1])
                np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5)
                maximum = max(maximum, float(np.max(np.abs(actual-expected))))
            repro = {"model": kind, "outer": outer, "seed": manifest["seed"], "reused": is_reused,
                     "train_snapshots": 64, "assessment_snapshots": 64, "maximum_probability_difference": maximum}
            json_write(output/"complete.json", {**local_identity, "records": own, "by_question": own_q,
                "reproduction": repro, "artifacts": {"assessment_logp.npy": digest(output/"assessment_logp.npy")}})
            records += own; by_question += own_q; reproduction.append(repro)
            print(json.dumps({"assessed": name, "reused": is_reused, "error": own[-1]["error"], "nll": own[-1]["nll"]}), flush=True)
        del data, cpu_data
    frame, qframe, repro = pd.DataFrame(records), pd.DataFrame(by_question), pd.DataFrame(reproduction)
    assert len(frame) == 112 and len(repro) == 48
    for kind in ORDER:
        q = qframe[(qframe.model == kind) & (qframe.split == "assessment")]
        assert q.qid.nunique() == 20
        assert (q.groupby("qid").outer.nunique() == 1).all()
        assert (q.groupby("qid").size() == (1 if kind in ("I0", "I0_current") else 3)).all()
    for fold in (f for f in protocol["folds"] if f["phase"] == "tune"):
        for kind in cfg["new_models"]:
            entries = [m for p, m in tuning if m["fold"] == fold["id"] and m["model"] == kind]
            assert len(entries) == 6 and len({m["initial_weights_sha256"] for m in entries}) == 1
    prior = json.loads((root/"prior_artifacts.json").read_text())
    for path, sha in prior.items():
        assert digest(path) == sha, path
    frame.to_csv(root/"outer_metrics.csv", index=False)
    qframe.to_csv(root/"outer_by_question.csv", index=False)
    repro.to_csv(root/"checkpoint_reproduction.csv", index=False)
    files = ("outer_metrics.csv", "outer_by_question.csv", "checkpoint_reproduction.csv")
    json_write(root/"assessment_complete.json", {**identity, "completed_utc": now(),
        "artifacts": {name: digest(root/name) for name in files}, "reproduced_checkpoints": 48,
        "maximum_cpu_mps_probability_difference": float(repro.maximum_probability_difference.max()),
        "prior_artifacts_verified_unchanged": len(prior), "new_official_test_evaluations": 0,
        "new_source_model_calls": 0, "fresh_confirmatory_dataset": False})


def paired(assessment, cfg):
    names = [(kind,"tiny") for kind in cfg["new_models"]] + [("pico","nano"),("nano","micro")]
    names += [(kind, baseline) for kind in ("pico", "nano", "micro", "tiny") for baseline in ("I0", "I0_current")]
    rng = np.random.default_rng(cfg["evaluation"]["bootstrap_seed"])
    qids = sorted(assessment.qid.unique())
    draws = rng.integers(0, len(qids), (cfg["evaluation"]["bootstrap_replicates"], len(qids)))
    records = []
    for first, second in names:
        a = assessment[assessment.model == first].set_index("qid").loc[qids]
        b = assessment[assessment.model == second].set_index("qid").loc[qids]
        for metric in ("error", "nll", "brier_two_class"):
            difference = (a[metric]-b[metric]).to_numpy()
            lower, upper = np.quantile(difference[draws].mean(1), [.025, .975])
            records.append({"first": first, "second": second, "metric": metric, "difference": float(difference.mean()),
                "descriptive_lower": float(lower), "descriptive_upper": float(upper),
                "questions_better": int((difference < -1e-6).sum()),
                "questions_tied": int((np.abs(difference) <= 1e-6).sum())})
    return pd.DataFrame(records)


def summarize(cfg):
    root = Path(cfg["output"])
    protocol = validate(cfg)
    completed = json.loads((root/"assessment_complete.json").read_text())
    assert completed["report_code_sha256"] == digest(__file__)
    for path, sha in completed["artifacts"].items():
        assert digest(root/path) == sha
    qframe = pd.read_csv(root/"outer_by_question.csv")
    averaged = qframe.groupby(["model", "split", "outer", "qid"])[["error", "nll", "brier_two_class"]].mean().reset_index()
    summary = averaged.groupby(["model", "split"])[["error", "nll", "brier_two_class"]].mean().reset_index()
    outer = averaged[averaged.split == "assessment"]
    contrasts = paired(outer, cfg)
    fold_means = outer.groupby(["model", "outer"])[["error", "nll", "brier_two_class"]].mean().reset_index()
    averaged.to_csv(root/"question_means.csv", index=False)
    summary.to_csv(root/"summary.csv", index=False)
    contrasts.to_csv(root/"paired_contrasts.csv", index=False)
    fold_means.to_csv(root/"outer_fold_summary.csv", index=False)
    # Confirm null differences remain exact rather than float-induced wins.
    null = outer.copy()
    for kind in ORDER:
        mask = null.model == kind
        for metric in ("error", "nll", "brier_two_class"):
            null.loc[mask, metric] = .3
    controls = paired(null, cfg)
    assert (controls[["difference", "descriptive_lower", "descriptive_upper", "questions_better"]] == 0).all().all()
    assert (controls.questions_tied == 20).all()
    tuning = base.completed_runs(root, "tune")
    refits = base.completed_runs(root, "refit")
    reference_tuning = reference_runs(cfg, "tune")
    selected = json.loads((root/"selection.json").read_text())
    seconds = sum(m["seconds"] for p, m in tuning+refits)
    stopping = pd.Series([m["stopping_reason"] for p, m in tuning]).value_counts().to_dict()
    tuning_records, curves = [], []
    for directory, m in tuning+reference_tuning:
        choice = next(c for c in selected["choices"]+selected["reused_choices"] if c["outer"] == m["outer"] and c["model"] == m["model"])
        tuning_records.append({"model": m["model"], "fold": m["fold"], "outer": m["outer"],
            "candidate": m["candidate"], "steps": m["steps"], "selected_step": m["selected_step"],
            "stopping_reason": m["stopping_reason"], "seconds": m["seconds"],
            "chosen_candidate": choice["candidate"] == m["candidate"],
            "checkpoint_at_cap": m["selected_step"] == cfg["training"]["max_steps"],
            "inner_nll": m["validation"]["nll"], "reused": m["model"] == cfg["reused_model"]})
        for point in json.loads((directory/"history.json").read_text()):
            curves.append({"model": m["model"], "fold": m["fold"], "candidate": m["candidate"],
                           "step": point["step"], **point["validation"]})
    tuning_frame = pd.DataFrame(tuning_records)
    tuning_frame.to_csv(root/"tuning_runs.csv", index=False)
    pd.DataFrame(curves).to_csv(root/"tuning_curves.csv", index=False)
    lookup = summary.set_index(["model", "split"])
    lines = ["# Transformer comparison below 7,370 parameters", "",
        "Completed three new transformer sizes on the existing twenty-question GPT development pool. The 7,370-parameter reference and strengthened Ising baseline are reused after identity/checksum checks. All four neural sizes have the same inputs, four attention heads, six tuning candidates, inner stopping rule and final seeds. Only attention/hidden/feedforward widths change.", "",
        "These are exploratory nested whole-question results on previously studied questions, not fresh confirmation. NLL is primary. Scores average seeds within question, then weight questions equally; predictions are not ensembled.", "",
        "| Model | Neural parameters | Other fitted coefficients | Training error | Outer error | Outer NLL | Outer Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind in ORDER:
        train, assessment = lookup.loc[(kind, "train")], lookup.loc[(kind, "assessment")]
        coefficients = "19" if kind == "I0" else ("19 baseline + 3 correction" if kind == "I0_current" else "0")
        lines.append(f"| {LABELS[kind]} | {protocol['parameters'].get(kind,0):,} | {coefficients} | {100*train.error:.2f}% | {100*assessment.error:.2f}% | {assessment.nll:.5f} | {assessment.brier_two_class:.5f} |")
    lines += ["", "## Paired comparisons", "",
        "Differences are first minus reference, so negative favors the first model. Intervals resample twenty paired question differences 3,000 times. They are pointwise and descriptive: only four overlapping outer training folds, prior question exposure and multiple contrasts limit inference. An interval crossing zero does not establish equivalence or noninferiority.", "",
        "| Comparison | Error difference, pp [95% interval] | NLL difference [95% interval] | Questions with lower NLL |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for first, second in ([(kind,"tiny") for kind in cfg["new_models"]] + [("pico","nano"),("nano","micro")] + [(kind,"I0_current") for kind in ("pico","nano","micro","tiny")]):
        rows = contrasts[(contrasts['first'] == first) & (contrasts['second'] == second)].set_index("metric")
        err, nll = rows.loc["error"], rows.loc["nll"]
        lines.append(f"| {LABELS[first]} − {LABELS[second]} | {100*err.difference:+.2f} [{100*err.descriptive_lower:+.2f}, {100*err.descriptive_upper:+.2f}] | {nll.difference:+.5f} [{nll.descriptive_lower:+.5f}, {nll.descriptive_upper:+.5f}] | {int(nll.questions_better)}/20 |")
    lines += ["", "## Matched procedure", "",
        "Four outer folds each hold out five questions; the other fifteen support three inner folds of ten fitting/five selection questions. Every question's eight graphs, four replicas, eight updates and 32 agents stay together. All sixteen training-fold question PCA and Ising fits are exactly reused from the verified earlier study. No official test question enters the new fits or scoring.", "",
        "Inputs remain current opinions, 16-dimensional persona/question context, signed graph and degree features. No arguments, raw text, additional history or question IDs are added. Architectures:", "",
        "| Parameters | Hidden width | Heads | Key/head | Value/head | Feedforward |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind in ("pico", "nano", "micro", "tiny"):
        m = cfg["models"][kind]
        lines.append(f"| {protocol['parameters'][kind]:,} | {m['d_model']} | {m['heads']} | {m['d_k']} | {m['d_v']} | {m['ffn_dim']} |")
    lines += ["", "Every size starts with a zero output head and receives learning rates 0.0001/0.0003 crossed with logit penalties 0/0.001/0.01, dropout 0.1, AdamW decay 0.001, batch32 and clipping1. The unchanged training loop evaluates inner NLL every 128 updates, allows initialization selection, uses six-check patience and a 3072-update cap. Within each outer fold/size, average inner NLL chooses a candidate; the median selected data passes sets the outer refit length, scaled from ten to fifteen questions. Refits use seeds 142/143/144. There is no outer-label early stopping or best-seed selection.", "",
        "Settings are separately selected for each size. This compares equally tuned procedures rather than holding selected optimizer settings or update counts identical. Smaller models can receive different selected training durations, within the same search policy.", "",
        "The strengthened Ising benchmark uses three correction coefficients (intercept, prior-logit scaling and current own opinion) atop its nineteen fold-fitted baseline coefficients. Its earlier nested penalty grid has three choices versus six for each neural model; it is a stronger benchmark, not a matched architecture/compute ablation. Its parameterization includes intercept/scale redundancy, so this is not a minimal coefficient count.", "",
        f"Executed 216 new inner fits and 36 new refits; reused 72 reference inner fits and 12 reference refits. New recorded fitting time: {seconds/60:.1f} minutes. Inner stopping counts: {stopping}. Reused fits are not counted as newly executed work.", "",
        "## Selected settings", "",
        "| Outer fold | Parameters | Learning rate | Output penalty | Refit updates | Origin |", "| ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in sorted(selected["choices"]+selected["reused_choices"], key=lambda r:(r["outer"], protocol["parameters"][r["model"]])):
        candidate = next(c for c in protocol["candidates"] if c["id"] == row["candidate"])
        lines.append(f"| {row['outer']} | {protocol['parameters'][row['model']]:,} | {candidate['learning_rate']} | {candidate['penalty']} | {row['refit_steps']} | {'reused' if row['model']==cfg['reused_model'] else 'new'} |")
    lines += ["", "## Training-cap diagnostic", "",
        "| Parameters | Inner fits reaching update cap | Best checkpoint at cap | Selected candidate's inner checkpoints at cap |",
        "| ---: | ---: | ---: | ---: |"]
    for kind in ("pico", "nano", "micro", "tiny"):
        part = tuning_frame[tuning_frame.model == kind]
        chosen = part[part.chosen_candidate]
        lines.append(f"| {protocol['parameters'][kind]:,} | {int(part.stopping_reason.eq('step_cap').sum())}/72 | {int(part.checkpoint_at_cap.sum())}/72 | {int(chosen.checkpoint_at_cap.sum())}/12 |")
    lines += ["", "A selected checkpoint at the update cap indicates that the search may be budget-limited; it does not establish convergence. The cap and candidate learning rates are matched to the reference, so this study estimates performance under that common search policy rather than each architecture's fully optimized potential. Full inner curves are saved in `tuning_curves.csv`.", "",
        "## Verification and limits", "",
        f"Eighteen focused tests passed, including exact stochastic resume, fold/input separation, objective equivalence, learning and agent-permutation checks at the new widths, selection isolation and exact reference initialization. GPU smoke tests pass for all three new sizes; replaying the reference's first 128 updates reproduces its original validation metrics exactly. All 48 final checkpoints, including the reused 12, were independently reloaded on CPU on 64 training and 64 assessment snapshots each. Maximum probability difference: {completed['maximum_cpu_mps_probability_difference']:.2e}. The strengthened baseline's per-question scores reproduce independently.", "",
        f"All {completed['prior_artifacts_verified_unchanged']:,} tracked prior immutable artifacts remain unchanged. Input arrays and all 16 preprocessing folds are identical to their originals. All 36 new final models were frozen before new outer assessment. No new source-model calls, official test scoring or autonomous rollouts occurred. No claims are made about new graph families, personas, source LLMs or a universal interaction law.", "",
        "The size sweep tests whether less capacity improves transfer with these inputs and this finite search. It cannot determine an irreducible accuracy ceiling, repair missing state or establish generalization on new independent questions. A final model choice still requires fresh question assessment.", "",
        "![Transformer size comparison](/Users/melancholymirth/deshearthGit/github/agent_interact/reports/transformer_micro_sweep.png)", "",
        "Artifacts: `runs/transformer_micro_sweep_v1/`; exact driver commands:", "",
        "```sh\n.venv/bin/python scripts/transformer_micro_sweep.py prepare\n.venv/bin/python scripts/transformer_micro_sweep.py fit --device mps\n.venv/bin/python scripts/report_transformer_micro_sweep.py evaluate --device mps\n.venv/bin/python scripts/report_transformer_micro_sweep.py summarize\n```"]
    Path("reports/transformer_micro_sweep.md").write_text("\n".join(lines)+"\n")
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/agent_interact_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, NullFormatter
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    kinds = ["pico", "nano", "micro", "tiny"]
    x = np.array([protocol["parameters"][k] for k in kinds])
    for ax, metric, title, scale in ((axes[0], "error", "Prediction error (%)", 100), (axes[1], "nll", "Negative log likelihood", 1)):
        for split, label, color in (("train", "Training", "#b5651d"), ("assessment", "Held-out questions", "#1d6394")):
            y = [scale*lookup.loc[(k,split),metric] for k in kinds]
            ax.plot(x,y,"o-",color=color,label=label,lw=2)
            for xx, yy in zip(x,y):
                ax.annotate(f"{yy:.2f}" if metric=="error" else f"{yy:.3f}",(xx,yy),xytext=(0,-14 if split=="train" else 8),textcoords="offset points",ha="center",fontsize=8,bbox={"facecolor":"white","edgecolor":"none","pad":.5,"alpha":.95})
        for baseline, label, style in (("I0", "Ising", ":"), ("I0_current", "Ising + current opinion", "--")):
            ax.axhline(scale*lookup.loc[(baseline,"assessment"),metric],color="0.45",ls=style,label=label)
        ax.set_xscale("log"); ax.set_xticks(x, [f"{int(v):,}" for v in x])
        ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Transformer parameters"); ax.set_ylabel(title)
        ax.grid(alpha=.15); ax.margins(x=.18,y=.22)
    axes[1].legend(fontsize=8,loc="best")
    fig.suptitle("Transformer size: same 20 development questions and nested folds",fontsize=12)
    fig.text(.5,.01,"Exploratory means; paired uncertainty intervals are in the accompanying report.",ha="center",fontsize=8,color="0.35")
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig("reports/transformer_micro_sweep.png",dpi=180)
    plt.close(fig)
    suites = ET.parse("reports/transformer_micro_sweep_checks.xml").getroot()
    assert len(list(suites.iter("testcase"))) == 18
    assert not list(suites.iter("failure")) and not list(suites.iter("error"))
    session = json.loads((root/"session.json").read_text())
    final = json.loads((root/"refit_complete.json").read_text())
    end = datetime.fromisoformat(final["completed_utc"])
    assert end.timestamp() <= session["deadline"]
    freeze = json.loads((root/"assessment_freeze.json").read_text())
    assert final["completed_utc"] <= freeze["frozen_utc"] <= completed["completed_utc"]
    prior = json.loads((root/"prior_artifacts.json").read_text())
    for path, sha in prior.items():
        assert digest(path) == sha, path
    files = [p for p in root.rglob("*") if p.is_file() and p.name != "completion_audit.json"]
    files += [CONFIG, Path(__file__), Path("scripts/transformer_micro_sweep.py"), Path("tests/test_transformer_micro_sweep.py"),
              Path("reports/transformer_micro_sweep.md"), Path("reports/transformer_micro_sweep.png"),
              Path("reports/transformer_micro_sweep_checks.xml")]
    json_write(root/"completion_audit.json", {"completed_utc": now(), "status": "complete",
        "new_tuning_fits": 216, "new_refits": 36, "reused_tuning_fits": 72, "reused_refits": 12,
        "focused_tests_passed": 18, "selected_checkpoints_cpu_reproduced": 48,
        "reference_128_step_gpu_replay_exact": True, "paired_null_control_passed": True,
        "prior_artifacts_verified_unchanged": len(prior), "core_sha256": base.CORE_SHA,
        "maximum_cpu_mps_probability_difference": completed["maximum_cpu_mps_probability_difference"],
        "new_fit_minutes": seconds/60, "within_training_time_guard": True,
        "selection_refit_assessment_order_verified": True, "fresh_confirmatory_dataset": False,
        "new_source_model_calls": 0, "new_official_test_evaluations": 0, "autonomous_rollouts": 0,
        "artifacts": {str(p): digest(p) for p in files}})
    print(summary.to_string(index=False))
    print(contrasts[(contrasts.metric.isin(["error","nll"])) & contrasts['second'].eq("tiny")].to_string(index=False))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evaluate", "summarize"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    cfg = config(CONFIG)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    if args.stage == "evaluate":
        evaluate(cfg, args.device)
    else:
        summarize(cfg)


if __name__ == "__main__":
    main()
