"""Independently verify and summarize the fixed Llama probability learning curve."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from mac_agents.models.baselines import baseline_logp
from scripts import breadth_probability_learning_curve as curve
from scripts import transformer_generalization as base

ROOT = curve.ROOT
REPORT = Path("reports/llama_probability_learning_curve.md")
FIGURE = Path("reports/llama_probability_learning_curve.png")
METRICS = ["accuracy", "error", "nll", "expected_error", "expected_nll", "probability_mean_squared_error"]


def contrast(frame, first, second, metric, draws):
    """Second minus first; first average seeds/subsets within each question."""
    def values(spec):
        part = frame
        for key, value in spec.items():
            part = part[part[key] == value]
        assert len(part)
        return part.groupby("qid")[metric].mean().sort_index()
    a, b = values(first), values(second)
    assert a.index.tolist() == b.index.tolist() and len(a) == 20
    delta = (b-a).to_numpy()
    distribution = delta[draws].mean(1)
    low, high = np.quantile(distribution, [.025, .975])
    return {"first": first, "second": second, "metric": metric, "difference": float(delta.mean()),
            "lower": float(low), "upper": float(high), "questions": len(delta),
            "questions_increased": int((delta > 1e-12).sum()),
            "questions_decreased": int((delta < -1e-12).sum()),
            "questions_equal": int((np.abs(delta) <= 1e-12).sum())}


def main():
    protocol = curve.validate()
    cfg = protocol["config"]
    complete = json.loads((ROOT / "run_complete.json").read_text())
    assert complete["protocol_sha256"] == digest(ROOT / "protocol.json")
    assert complete["neural_fits"] == 60
    for path, sha in complete["manifests"].items():
        assert digest(path) == sha
    torch.set_num_threads(cfg["training"]["torch_threads"])
    common = dict(np.load(curve.SOURCE / "predictors/development.npz", allow_pickle=False))
    targets = dict(np.load(ROOT / "targets.npz", allow_pickle=False))
    e, t = common["pairs"].T
    qids = common["episode_qids"][e]
    runs, questions, checks = [], [], []
    artifacts = {}
    for fold in protocol["folds"]:
        meta = {"subset_id": fold["subset_id"], "path": -1 if fold["outer"] is None else fold["outer"],
                "n_questions": fold["n_questions"], "condition": fold["condition"]}
        fm = json.loads((ROOT / "folds" / fold["id"] / "manifest.json").read_text())
        features = dict(np.load(ROOT / fm["features"], allow_pickle=False))
        data = curve.TargetData(common, features, "cpu", targets[fold["condition"]])
        predictions = []
        record = json.loads((ROOT / "folds" / fold["id"] / "ising.json").read_text())
        lp = baseline_logp(record, common["spins"][e,t], features["context"][e], common["J"][e])
        np.testing.assert_array_equal(lp, np.load(ROOT / "folds" / fold["id"] / "ising_logp.npy"))
        predictions.append(({**meta, "model": "I0", "seed": -1}, lp, None))
        for seed in cfg["training"]["refit_seeds"]:
            scored = ROOT / "scored" / fold["id"] / f"s{seed}"
            saved = json.loads((scored / "scores.json").read_text())
            assert digest(saved["checkpoint"]) == saved["checkpoint_sha256"]
            assert digest(saved["logp_path"]) == saved["logp_sha256"]
            lp = np.load(saved["logp_path"], allow_pickle=False)
            model = base.make_model(cfg, "nano")
            ck = torch.load(saved["checkpoint"], map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            cpu = base.predictions(model, data, np.arange(len(qids)), False, cfg["training"]["evaluation_batch_size"])
            difference = float(np.abs(np.exp(cpu)-np.exp(lp)).max())
            # Subset PCA can produce steeper logits than the full-pool model.
            # Use an absolute cross-device probability tolerance, while requiring
            # identical decisions on every recorded development transition.
            np.testing.assert_allclose(np.exp(cpu), np.exp(lp), atol=2e-5, rtol=0,
                                       err_msg=f"CPU replay: {fold['id']}, seed {seed}")
            np.testing.assert_array_equal(cpu.argmax(-1), lp.argmax(-1))
            checks.append({"fold": fold["id"], "seed": seed,
                "cpu_max_probability_difference": difference,
                "original_endpoint_max_probability_difference": saved["original_endpoint_max_probability_difference"]})
            print(json.dumps({"verified":fold["id"], "seed":seed,
                              "cpu_max_probability_difference":difference}), flush=True)
            for path in (Path(saved["checkpoint"]), Path(saved["logp_path"]), scored / "scores.json"):
                artifacts[str(path)] = digest(path)
            predictions.append(({**meta, "model": "nano", "seed": seed}, lp, saved))
        for identity, lp, saved in predictions:
            for split in ("train", "validation"):
                ids = data.ids(fold[f"{split}_questions"])
                aggregate, per_question = curve.score(lp[ids], data.y[ids], targets["soft"][ids], qids[ids])
                if saved is not None:
                    for key, value in aggregate.items():
                        np.testing.assert_allclose(value, saved["metrics"][split][key], rtol=0, atol=1e-12)
                runs.append({**identity, "split": split, **aggregate})
                questions += [{**identity, "split": split, **row} for row in per_question]
    run_frame, question_frame = pd.DataFrame(runs), pd.DataFrame(questions)
    run_frame.to_csv(ROOT / "run_metrics.csv", index=False)
    question_frame.to_csv(ROOT / "question_metrics.csv", index=False)
    summary = []
    for (model, condition, n, split), part in run_frame.groupby(["model", "condition", "n_questions", "split"]):
        row = {"model": model, "condition": condition, "n_questions": int(n), "split": split,
               "fits": len(part), "subset_draws": part.subset_id.nunique(), "metrics": {}, "variability": {}}
        path_means = part.groupby("subset_id")[METRICS].mean()
        for key in METRICS:
            row["metrics"][key] = float(part[key].mean())
            row["variability"][key] = {"run_min": float(part[key].min()), "run_max": float(part[key].max()),
                "subset_mean_min": float(path_means[key].min()), "subset_mean_max": float(path_means[key].max()),
                "subset_mean_sample_sd": float(path_means[key].std()) if len(path_means) > 1 else None,
                "mean_within_subset_seed_sample_sd": float(part.groupby("subset_id")[key].std().mean()) if model == "nano" else None}
        summary.append(row)
    validation = question_frame[question_frame.split == "validation"]
    rng = np.random.default_rng(cfg["evaluation"]["bootstrap_seed"])
    draws = rng.integers(0, 20, (cfg["evaluation"]["bootstrap_replicates"], 20))
    contrasts = []
    for model in ("nano", "I0"):
        for condition in ("hard", "soft"):
            for start, end in ((10,20), (20,40), (40,80), (10,80)):
                for metric in ("accuracy", "expected_nll"):
                    contrasts.append(contrast(validation, {"model":model,"condition":condition,"n_questions":start},
                        {"model":model,"condition":condition,"n_questions":end}, metric, draws))
        for n in (10,20,40,80):
            for metric in ("accuracy", "expected_error", "expected_nll", "probability_mean_squared_error"):
                contrasts.append(contrast(validation, {"model":model,"condition":"hard","n_questions":n},
                    {"model":model,"condition":"soft","n_questions":n}, metric, draws))
    for condition in ("hard","soft"):
        for n in (10,20,40,80):
            for metric in ("accuracy", "expected_nll"):
                contrasts.append(contrast(validation, {"model":"I0","condition":condition,"n_questions":n},
                    {"model":"nano","condition":condition,"n_questions":n}, metric, draws))
    json_write(ROOT / "summary.json", {"protocol_sha256":digest(ROOT / "protocol.json"), "cells":summary,
        "paired_contrasts":contrasts, "independence_warning":"Descriptive intervals resample the 20 reused validation questions after averaging model seeds and subset paths within question. They condition on chosen subsets and do not include full training-set-selection uncertainty or multiple-comparison correction."})
    render(run_frame, summary, contrasts, checks, protocol, complete)
    curve.verify_originals()
    for path in [ROOT / "summary.json", ROOT / "run_metrics.csv", ROOT / "question_metrics.csv", REPORT, FIGURE,
                 FIGURE.with_suffix(".pdf"), Path("reports/llama_probability_curve_checks.xml")]:
        artifacts[str(path)] = digest(path)
    json_write(ROOT / "completion_audit.json", {"completed_utc":curve.now(), "neural_models_checked":len(checks),
        "baseline_models_checked":20, "cpu_checks":checks,
        "max_cpu_probability_difference":max(c["cpu_max_probability_difference"] for c in checks),
        "source_llm_calls":0, "assessment_generated_or_scored":False,
        "original_models_frozen_sha256":digest(curve.SOURCE / "models_frozen.json"),
        "protocol_sha256":digest(ROOT / "protocol.json"),
        "script_sha256":digest(__file__), "artifacts":artifacts})
    print(json.dumps({"cells":summary, "maximum_cpu_difference":max(c["cpu_max_probability_difference"] for c in checks)}, indent=2))


def render(frame, summary, contrasts, checks, protocol, complete):
    def cell(model, condition, n, split="validation"):
        return next(s for s in summary if (s["model"],s["condition"],s["n_questions"],s["split"]) == (model,condition,n,split))
    def change(model, condition, start, end, metric="accuracy"):
        return next(c for c in contrasts if c["first"] == {"model":model,"condition":condition,"n_questions":start}
                    and c["second"] == {"model":model,"condition":condition,"n_questions":end} and c["metric"] == metric)
    def target_change(model, n, metric):
        return next(c for c in contrasts if c["first"] == {"model":model,"condition":"hard","n_questions":n}
                    and c["second"] == {"model":model,"condition":"soft","n_questions":n} and c["metric"] == metric)
    counts = [10,20,40,80]
    styles = [("nano","hard","Transformer · majority labels","#245dc1","-"),
              ("nano","soft","Transformer · probability labels","#245dc1","--"),
              ("I0","hard","Ising · majority labels","#c4661c","-"),
              ("I0","soft","Ising · probability labels","#c4661c","--")]
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1,3,figsize=(15,4.7))
    panels = [("validation","accuracy",100,"Validation accuracy","Next-majority opinion accuracy (%)"),
              ("train","accuracy",100,"Training accuracy","Next-majority opinion accuracy (%)"),
              ("validation","expected_nll",1,"Validation probability loss","Expected negative log likelihood")]
    for ax, (split, metric, scale, title, ylabel) in zip(axes, panels):
        for model, condition, label, color, linestyle in styles:
            part = frame[(frame.model==model)&(frame.condition==condition)&(frame.split==split)]
            stats = part.groupby("n_questions")[metric].agg(["mean","min","max"]).loc[counts]
            ax.plot(counts,stats["mean"]*scale,marker="o",markersize=4,linestyle=linestyle,color=color,label=label)
            ax.fill_between(counts,stats["min"]*scale,stats["max"]*scale,color=color,alpha=.07)
        ax.set_xscale("log",base=2)
        ax.set_xticks(counts, labels=counts)
        ax.set_xlabel("Training questions")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=.18)
    fig.suptitle("Llama data: fixed 818-parameter transformer vs. 19-coefficient Ising", y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5,.055), ncol=2, frameon=False)
    fig.text(.5,.01,"Bands: min–max across fitted subsets and seeds, not confidence intervals. Same 20 development validation questions; assessment untouched.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.17,1,.94])
    fig.savefig(FIGURE,dpi=180)
    fig.savefig(FIGURE.with_suffix(".pdf"))
    plt.close(fig)
    hard_change = change("nano","hard",10,80)
    full_soft_change = target_change("nano",80,"accuracy")
    lines = ["# Llama question-count and probability-target experiment", "",
        "Development-only comparison of 10, 20, 40 and 80 training questions. Accuracy means predicting an agent's next five-vote majority opinion, not answering the math question correctly. Every model predicts the same 20 validation questions (5,120 agent updates). All 40 assessment conversations remain ungenerated and unscored.", "",
        f"The hard-label 818-parameter transformer's mean validation accuracy changes from {cell('nano','hard',10)['metrics']['accuracy']:.2%} at 10 questions to {cell('nano','hard',80)['metrics']['accuracy']:.2%} at 80: {100*hard_change['difference']:+.2f} percentage points. At 80 questions, replacing sampled majority labels with logged majority probabilities changes accuracy by {100*full_soft_change['difference']:+.2f} points. These are fixed-procedure development results, not independently tuned sample-size optima.", "",
        "## Observed validation accuracy", "",
        "| Training questions | Agent updates | Transformer, majority labels | Transformer, probability labels | Ising, majority labels | Ising, probability labels |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in counts:
        scores = [cell(m,c,n)["metrics"]["accuracy"] for m,c,_,_,_ in styles]
        lines.append(f"| {n} | {n*256:,} | " + " | ".join(f"{v:.2%}" for v in scores) + " |")
    lines += ["", "Each 10/20/40 point averages three nested subset paths and three transformer seeds; Ising has one deterministic fit per subset. The 80-question endpoint uses the full pool once with three transformer seeds. Scores are averaged, not prediction probabilities. Smaller subsets are balanced by topic, correct-answer label and graph; none is selected using performance.", "",
              "![Learning curves](llama_probability_learning_curve.png)", "", "## Training accuracy", "",
              "| Questions | Transformer, majority labels | Transformer, probability labels | Ising, majority labels | Ising, probability labels |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for n in counts:
        scores = [cell(m,c,n,"train")["metrics"]["accuracy"] for m,c,_,_,_ in styles]
        lines.append(f"| {n} | " + " | ".join(f"{v:.2%}" for v in scores) + " |")
    lines += ["", "Training scores are evaluated with dropout disabled on the actual subset used for fitting. Training sets differ in size and composition, so falling training accuracy alone does not mean optimization deteriorated.", "",
              "## Paired size comparisons", "",
              "Changes below are larger minus smaller training set; positive accuracy changes favor more questions. Brackets are descriptive 95% paired validation-question bootstrap intervals.", "",
              "| Transformer targets | Question counts | Accuracy change, points [interval] | Expected NLL change [interval] |",
              "| --- | --- | ---: | ---: |"]
    for condition in ("hard","soft"):
        for start,end in ((10,20),(20,40),(40,80),(10,80)):
            a,b = change("nano",condition,start,end),change("nano",condition,start,end,"expected_nll")
            lines.append(f"| {condition} | {start} → {end} | {100*a['difference']:+.2f} [{100*a['lower']:+.2f}, {100*a['upper']:+.2f}] | {b['difference']:+.5f} [{b['lower']:+.5f}, {b['upper']:+.5f}] |")
    lines += ["", "These intervals average subset/seed scores within each validation question, then resample the 20 questions 5,000 times. They condition on the sampled training subsets, omit multiple-comparison adjustment, and use validation questions already involved in earlier tuning and diagnosis. They are not confirmatory significance tests.", "",
              "## Does probability supervision help?", "",
              "The soft target is the next-round majority-A probability q = p³(10 − 15p + 6p²), where p is the source's normalized first A/B-token probability. The input still contains only current opinions, signed graph, and question/persona context. Future probabilities are training labels only. This removes five-vote sampling noise from supervision conditional on the recorded prompt; it does not remove uncertainty about unobserved incoming messages.", "",
              "| Questions | Transformer accuracy change, soft − hard (points) | Expected error change (points) | Expected NLL change | Probability MSE change |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for n in counts:
        cs = [target_change("nano",n,m) for m in ("accuracy","expected_error","expected_nll","probability_mean_squared_error")]
        lines.append(f"| {n} | {100*cs[0]['difference']:+.2f} | {100*cs[1]['difference']:+.2f} | {cs[2]['difference']:+.5f} | {cs[3]['difference']:+.5f} |")
    lines += ["", "Higher observed accuracy is better; lower expected error/NLL/MSE is better. Expected metrics average over the logged majority distribution rather than the one realized sample. Source probabilities are an approximation to parsed-answer probabilities: 12 of 144,000 responses were not a single bare A/B token, with mean outside-A/B first-token mass 2.18e−7. These probabilities condition on realized messages and do not give a universal irreducible-error bound for the predictor's input state.", "",
              "### Full 80-question comparison", "",
              "| Model and targets | Observed accuracy | Observed NLL | Expected accuracy | Expected NLL | Probability MSE |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for m,c,label,_,_ in styles:
        s=cell(m,c,80)["metrics"]
        lines.append(f"| {label} | {s['accuracy']:.2%} | {s['nll']:.5f} | {1-s['expected_error']:.2%} | {s['expected_nll']:.5f} | {s['probability_mean_squared_error']:.5f} |")
    a=full_soft_change
    b=target_change("nano",80,"expected_nll")
    lines += ["",f"At 80 questions, soft-minus-hard transformer accuracy is {100*a['difference']:+.2f} points [{100*a['lower']:+.2f}, {100*a['upper']:+.2f}], and expected NLL changes by {b['difference']:+.5f} [{b['lower']:+.5f}, {b['upper']:+.5f}].", "",
              "## Sensitivity to question selection", "",
              "Ranges below first average the three optimizer seeds within each selected subset, then take the minimum and maximum over the three subsets. They are not uncertainty intervals. There is only one full-80 composition, so composition variation cannot be estimated there.", "",
              "| Questions | Hard-label transformer subset-mean accuracy range | Soft-label transformer subset-mean accuracy range |",
              "| ---: | ---: | ---: |"]
    for n in counts[:-1]:
        row=[]
        for c in ("hard","soft"):
            v=cell("nano",c,n)["variability"]["accuracy"]
            row.append(f"{v['subset_mean_min']:.2%}–{v['subset_mean_max']:.2%}")
        lines.append(f"| {n} | " + " | ".join(row) + " |")
    lines += ["", "## Interpretation", "",
        "More training questions improve the transformer's validation results within this pipeline. With majority labels, training accuracy falls from 88.17% to 84.59% while validation accuracy rises from 83.52% to 84.89%. The training/validation gap falls from 4.66 points to −0.29 points, consistent with less overfitting. These sets differ in composition, and validation being slightly easier than training does not establish an absence of overfitting.", "",
        "The Ising baseline changes less, from 82.71% to 83.32% validation accuracy. The transformer's advantage increases from 0.81 to 1.57 points. Most of the hard-label accuracy increase occurs before 40 questions; the 40→80 difference is only +0.21 points, with an interval spanning zero. Probability loss continues improving. These four points cannot establish a performance ceiling or predict the effect of hundreds of questions.", "",
        "Probability targets substantially improve probability loss at 10 and 20 questions, but do not consistently improve classification accuracy across sizes. At 80 questions, both the accuracy and expected-NLL soft-minus-hard intervals include zero. Vote-sampling noise therefore does not appear sufficient to explain the remaining error under the tested architecture and training recipe. Different regularization or duration for soft targets remains untested. The earlier matched-state/message evidence still motivates investigating omitted argument information with an explicitly defined observation boundary.", "",
        "## Controls and limits", "",
        "- Architecture stays at 818 trainable parameters. Frozen MiniLM embeddings are reused; question PCA is refitted on each selected training subset only. The same 32 known personas are used throughout. No arguments or probability features are supplied to either predictor.",
        "- Each transformer receives 1,280 optimizer updates with learning rate 0.001, output-logit penalty 0.001, dropout 0.1, AdamW weight decay 0.001, batch size up to 32, and seeds 142/143/144. Settings and duration come from the earlier full-80 validation selection, so this is a fixed-procedure curve favoring no newly tuned subset. Equal step counts are not equal epochs; the 10-question condition also has partial minibatches.",
        "- Hard/soft pairs start from identical weights and use identical minibatch order. Ising uses the unmodified pinned author's two-stage fit, including its regularization, with the same subset context and either binary or fractional BCE targets.",
        "- A seeded constrained selection uses only training metadata: exactly balanced A/B correct-answer labels and topic/graph counts within the floor/ceiling of full-pool proportions. Three paths are nested at 10/20/40, and all reach the same 80. Difficulty is recorded, not constrained. The sampler is not uniform over all feasible subsets.",
        "- Increasing question count simultaneously increases unique transitions, question variety, and PCA fitting data. This does not isolate diversity from raw sample count or representation estimation. It provides no evidence about data sizes above 80.",
        "- The predictor forecasts the next step on observed trajectories. Probability targets are conditioned on the actually generated future messages. No new message generation, autonomous rollouts, or untouched assessment evaluation is performed.", "",
        "## Execution and verification", "",
        f"Completed 60 new transformer fits and 20 Ising fits. The neural run took {complete['seconds_this_session']/60:.2f} minutes. All seven focused checks passed (subset nesting/strata, holdout exclusion, next-round labels, label/input separation, paired training, PCA isolation and source-probability checks). Full-80 hard-label PCA and Ising weights reproduce the original artifacts. All three newly trained hard-label endpoint models reproduce the frozen predictions within 2e−5 probability tolerance. CPU inference independently verifies all 60 neural checkpoints and exact class predictions; maximum probability difference is {max(c['cpu_max_probability_difference'] for c in checks):.3g}. Original primary model and diagnostic artifacts pass their stored hashes.", "",
        "Protocol, selected question IDs, checkpoints, run and per-question metrics, all contrasts and completion audit are in `runs/llama_probability_curve_v1/`. The primary frozen study remains in `runs/question_breadth_v1/`.", "",
        "```bash", ".venv/bin/python -m pytest tests/test_breadth_probability_learning_curve.py tests/test_breadth_error_diagnosis.py tests/test_breadth_train.py::test_training_pca_applies_to_new_questions_without_refitting", ".venv/bin/python -u scripts/breadth_probability_learning_curve.py prepare", ".venv/bin/python -u scripts/breadth_probability_learning_curve.py run --device mps", ".venv/bin/python scripts/report_breadth_probability_curve.py", "```", ""]
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
