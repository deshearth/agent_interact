"""Audit the question-context comparison and report every declared condition."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from mac_agents.metrics import paired_bootstrap, score
from scripts.argument_features import load_cache
from scripts import argument_mechanisms as previous
from scripts.question_context import CONFIG, ROOT, BASE_ROOT, VARIANTS, cohort, implementation, prepare

PREFIX = Path("reports/question_context")
KEYS = ["episode_id", "round", "agent"]
METRICS = ["nll", "accuracy", "brier", "fcba", "flip_accuracy", "stay_accuracy"]
REPRESENTATIONS = ("original_pca3", *VARIANTS)
CONDITIONS = ("stance_only", "full_arguments", "shuffled_arguments")
LABELS = {"original_pca3": "Original PCA3", "pca3_matched": "PCA3 + matched branch",
          "full_embedding": "PCA3 + full question embedding"}


def fmt_interval(row, metric):
    scale, places = (100, 2) if metric == "accuracy" else (1, 5)
    return f"{scale*row['difference']:+.{places}f} [{scale*row['ci95'][0]:+.{places}f}, {scale*row['ci95'][1]:+.{places}f}]"


def main():
    cfg = config(CONFIG)
    arrays, index, _, _ = prepare(cfg, load_cache(cfg, verify=True))
    protocol = json.loads((ROOT/"protocol.json").read_text())
    plan = json.loads((ROOT/"plan.json").read_text())
    assert plan["complete"] and plan["protocol_sha256"] == digest(ROOT/"protocol.json")
    expected = {(r["kind"], r["condition"], r["seed"]) for r in cohort(cfg)}
    assert {(r["kind"], r["condition"], r["seed"]) for r in plan["runs"]} == expected
    impl = hashlib.sha256(json.dumps(implementation(), sort_keys=True).encode()).hexdigest()
    checks = ET.parse("reports/question_context_checks.xml").getroot()
    assert not any(checks.findall(f".//{tag}") for tag in ("failure", "error", "skipped"))
    for seed in cfg["training"]["seeds"]:
        audit = previous.audit_donors(arrays, index, np.load(BASE_ROOT/f"donors_s{seed}.npy", allow_pickle=False))
        assert audit == protocol["shuffle_audit"][str(seed)]
    with np.load(ROOT/"question_features.npz", allow_pickle=False) as data:
        retained = float(data["pca3_variance_fraction"])
        fit_mask = np.isin(data["qids"], data["fit_qids"])
        geometry = {"pca3_training_variance_fraction": retained,
                    "fit_questions": data["fit_qids"].tolist(), "transformed_questions": data["qids"].tolist(),
                    "training_rank": int(np.linalg.matrix_rank(data["full_embedding"][fit_mask].astype(np.float64), tol=1e-6)),
                    "pca3_rank": int(np.linalg.matrix_rank(data["pca3_matched"][fit_mask].astype(np.float64), tol=1e-6)),
                    "normalization": {kind: {"fitting_rms_l2": float(np.sqrt((data[kind][fit_mask]**2).sum(-1).mean())),
                        "unseen_rms_l2": float(np.sqrt((data[kind][~fit_mask]**2).sum(-1).mean()))} for kind in VARIANTS}}
    json_write(f"{PREFIX}_geometry.json", geometry)
    ep_lookup = {r["episode_id"]: e for e, r in enumerate(index)}
    references, rows, qrows, round_rows, sensitivities, provenance = {}, [], [], [], [], {}
    items = [("original_pca3", {"kind": "A2", "condition": c, "seed": s}, BASE_ROOT)
             for s in cfg["training"]["seeds"] for c in CONDITIONS]
    items += [(r["kind"], r, ROOT) for r in cohort(cfg)]
    for representation, item, root in items:
        name = f"{item['kind']}-{item['condition']}-s{item['seed']}"
        run, out = root/name, root/"evaluation"/name
        manifest = json.loads((run/"manifest.json").read_text())
        assert manifest["status"] == "complete" and not manifest["smoke"]
        assert manifest["train_snapshots"] == 2048 and manifest["validation_snapshots"] == 1024
        assert manifest["train_questions"] == protocol["roles"]["train"]["questions"]
        assert manifest["validation_questions"] == protocol["roles"]["validation"]["questions"]
        assert manifest["protocol_sha256"] == digest(root/"protocol.json")
        for file, sha in manifest["artifact_sha256"].items():
            assert digest(run/file) == sha
        evaluation = json.loads((out/"identity.json").read_text())
        assert evaluation["run_manifest_sha256"] == digest(run/"manifest.json")
        assert evaluation["best_sha256"] == digest(run/"best.pt")
        assert evaluation["protocol_sha256"] == digest(root/"protocol.json")
        for file, sha in evaluation["artifacts"].items():
            assert digest(out/file) == sha
        projection_norm = None
        if root == ROOT:
            assert manifest["implementation_sha256"] == impl
            assert manifest["parameters"] == 382498
            ck = torch.load(run/"best.pt", map_location="cpu", weights_only=False)
            projection = ck["model"]["encoder.question_projection.weight"]
            assert torch.isfinite(projection).all() and projection.abs().sum() > 0
            projection_norm = float(projection.norm())
        else:
            baseline = protocol["baselines"][name]
            assert baseline["manifest_sha256"] == digest(run/"manifest.json")
            assert baseline["evaluation_sha256"] == digest(out/"identity.json")
            assert manifest["parameters"] == 185890
        provenance[f"{representation}/{name}"] = {"manifest_sha256": digest(run/"manifest.json"),
                                                   "evaluation_sha256": digest(out/"identity.json")}
        history = json.loads((run/"history.json").read_text())
        assert all(r["step"] == 64*r["epoch"] for r in history)
        label = {"representation": representation, "condition": item["condition"], "seed": item["seed"]}
        for role in previous.ROLES:
            where = run if role == "validation" else out
            frame = pd.read_parquet(where/f"{role}_predictions.parquet").sort_values(KEYS).reset_index(drop=True)
            assert not frame.duplicated(KEYS).any()
            truth = frame[KEYS+["qid", "current_state", "target"]]
            if role in references:
                pd.testing.assert_frame_equal(references[role], truth)
            else:
                references[role] = truth
                es = frame.episode_id.map(ep_lookup).to_numpy()
                ts, agents = frame["round"].to_numpy(), frame.agent.to_numpy()
                assert set(frame.episode_id) == set(protocol["roles"][role]["episode_ids"])
                assert {index[e]["diagnostic_role"] for e in es} == {role}
                np.testing.assert_array_equal(frame.target, arrays["spins"][es, ts+1, agents])
                np.testing.assert_array_equal(frame.current_state, arrays["spins"][es, ts, agents])
            metrics, questions = score(frame)
            saved = json.loads((where/f"{role}_metrics.json").read_text())
            for metric in METRICS:
                np.testing.assert_allclose(metrics[metric], saved[metric], rtol=0, atol=1e-12)
            rows.append({**label, "role": role, **metrics, "error": 1-metrics["accuracy"],
                         "parameters": manifest["parameters"], "steps": manifest["steps"],
                         "best_step": manifest["best_step"], "stopping_reason": manifest["stopping_reason"],
                         "question_projection_norm": projection_norm})
            qrows.extend(questions.assign(**label, role=role).to_dict("records"))
            for t, group in frame.groupby("round", sort=True):
                _, qs = score(group)
                round_rows.extend(qs.assign(**label, role=role, round=t).to_dict("records"))
            if item["condition"] == "full_arguments" and role in ("familiar", "unseen"):
                swapped = pd.read_parquet(out/f"{role}_substitution_predictions.parquet").sort_values(KEYS).reset_index(drop=True)
                pd.testing.assert_frame_equal(truth, swapped[truth.columns])
                for qid, group in frame.groupby("qid", sort=True):
                    other = swapped.loc[group.index]
                    m, _ = score(group)
                    altered, _ = score(other)
                    sensitivities.append({**label, "role": role, "qid": qid,
                        "nll_change": altered["nll"]-m["nll"], "accuracy_change": altered["accuracy"]-m["accuracy"],
                        "choice_change_fraction": float((group.prediction != other.prediction).mean()),
                        "mean_absolute_probability_change": float((group.p_positive-other.p_positive).abs().mean())})
    runs, questions = pd.DataFrame(rows), pd.DataFrame(qrows)
    runs.to_csv(f"{PREFIX}_runs.csv", index=False)
    questions.to_csv(f"{PREFIX}_by_question.csv", index=False)
    pd.DataFrame(round_rows).to_csv(f"{PREFIX}_by_question_round.csv", index=False)
    pd.DataFrame(sensitivities).to_csv(f"{PREFIX}_sensitivity.csv", index=False)
    means = runs.groupby(["role", "representation", "condition"], sort=True)[METRICS+["error"]].mean().reset_index()
    means.to_csv(f"{PREFIX}_summary.csv", index=False)
    qm = questions.groupby(["role", "representation", "condition", "qid"], sort=True)[METRICS].mean()
    paired = {}

    def contrast(name, first, second):
        paired[name] = {metric: paired_bootstrap(first[metric], second[metric],
            cfg["evaluation"]["bootstrap_replicates"], cfg["evaluation"]["bootstrap_seed"])
            for metric in ("nll", "accuracy")}

    for role in previous.ROLES:
        for condition in CONDITIONS:
            for control in ("pca3_matched", "original_pca3"):
                contrast(f"{role}/{condition}: full_embedding minus {control}",
                         qm.loc[(role, "full_embedding", condition)], qm.loc[(role, control, condition)])
            contrast(f"{role}/{condition}: pca3_matched minus original_pca3",
                     qm.loc[(role, "pca3_matched", condition)], qm.loc[(role, "original_pca3", condition)])
        for representation in REPRESENTATIONS:
            for control in ("stance_only", "shuffled_arguments"):
                contrast(f"{role}/{representation}: arguments minus {control}",
                    qm.loc[(role, representation, "full_arguments")], qm.loc[(role, representation, control)])
        full_delta = qm.loc[(role, "full_embedding", "full_arguments")]-qm.loc[(role, "full_embedding", "stance_only")]
        matched_delta = qm.loc[(role, "pca3_matched", "full_arguments")]-qm.loc[(role, "pca3_matched", "stance_only")]
        contrast(f"{role}: change in argument contribution, full_embedding minus pca3_matched", full_delta, matched_delta)
    json_write(f"{PREFIX}_paired.json", paired)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    colors = {"original_pca3": "#5c6470", "pca3_matched": "#4d8baf", "full_embedding": "#176e61"}
    for row, role in enumerate(("train", "familiar", "unseen")):
        for col, metric in enumerate(("error", "nll")):
            ax = axes[row, col]
            scale = 100 if metric == "error" else 1
            for offset, rep in zip((-.14, 0., .14), REPRESENTATIONS):
                values = []
                for x, condition in enumerate(CONDITIONS):
                    series = runs[(runs.role == role)&(runs.representation == rep)&(runs.condition == condition)][metric].to_numpy()*scale
                    values.append(series.mean())
                    ax.scatter(x+offset+np.linspace(-.025, .025, 3), series, color=colors[rep], s=15, alpha=.35)
                ax.plot(np.arange(3)+offset, values, color=colors[rep], marker="o", markersize=5,
                        label=LABELS[rep], linewidth=1.6, linestyle="--" if rep == "original_pca3" else "-")
            ax.set_title({"train": "Fitting conversations", "familiar": "Held-out conversations, 16 familiar questions",
                          "unseen": "4 unseen development questions"}[role], fontsize=11)
            ax.set_ylabel("Error (%)" if metric == "error" else "Negative log-likelihood")
            ax.set_xticks(range(3), ["Stance / history", "Full arguments", "Shuffled arguments"])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .958), ncol=3, frameon=False)
    fig.suptitle("Does fuller question context help predict argument responses?", y=.995, fontsize=15)
    fig.text(.5, .913, "Bold points: three-seed means. Faint points: individual seeds. Lower is better. Both new branches have equal parameter counts.", ha="center", fontsize=9)
    fig.text(.5, .012, "Same conversation split and stopping rule throughout. Frozen embeddings; exploratory reused development data.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .035, 1, .9])
    fig.savefig(f"{PREFIX}.png", dpi=180)
    fig.savefig(f"{PREFIX}.pdf")
    plt.close(fig)

    m = means.set_index(["role", "representation", "condition"])
    lines = ["# Fuller question-and-choice context", "",
        "This experiment restores the cached 1,536-dimensional question/choice embedding in both the no-argument and argument predictors. An equally sized control receives only a reconstruction of the original three principal components. All comparisons use the previous conversation split and stopping rule. This is exploratory research on objective mathematical answer revisions.", "",
        "**Finding:** fuller frozen question features modestly improve fitting, but do not improve argument-model accuracy over the matched three-component control. Argument-model training error falls from 5.74% to 5.55%; familiar-question error changes from 6.01% to 6.02%, and unseen-question error from 10.35% to 10.36%. The unseen accuracy difference is -0.02 percentage points, descriptive interval [-0.85, +0.76]; the NLL difference is -0.00283 [-0.02022, +0.01700]. These intervals are too broad to claim equivalence, but no clear gain is demonstrated.", "",
        "With the full question embedding, adding arguments improves mean accuracy by only 0.07 points on familiar questions and 0.29 points on unseen questions; both descriptive intervals include zero, and unseen NLL is slightly worse. Its improvement over the older argument model (10.84% to 10.36% unseen error) is also achieved by the matched PCA3 branch (10.35%), so that change cannot be attributed to the extra question dimensions. The original no-text reference still has the lowest mean unseen error, 9.84%. This tested removal of PCA compression does not resolve the missing argument benefit.", "",
        "## Design and interpretation", "",
        "The source LLM sees the complete problem and A/B choices. Earlier predictors received a three-component question PCA, plus persona/context interactions. This study adds a direct learned question projection before the local encoder nonlinearity; the original context is retained. It tests a compression bottleneck in a frozen embedding, not raw-token mathematical reasoning or a jointly encoded question/argument pair.", "",
        "| Question representation | Information added | Trainable parameters | Fits |",
        "| --- | --- | ---: | ---: |",
        "| Original PCA3 | None; frozen reference | 185,890 | 9 reused |",
        "| PCA3 + matched branch | Same three components reconstructed into 1,536 coordinates | 382,498 | 9 new |",
        "| PCA3 + full question embedding | All 1,536 cached coordinates | 382,498 | 9 new |", "",
        "The two added branches have the same parameters, initialization, optimizer and training budget. Their zero-initialized projections preserve the original initial predictions and shared RNG. The matched branch adds no question information beyond the existing context, but differs from the original model in parameterization and optimization. Equal nominal parameters do not equalize effective statistical capacity: the new inputs differ in rank by design.", "",
        f"Question features use the existing pinned `text-embedding-3-small` cache. Every cached input string was checked against the complete problem and its two choices. Centering, PCA and scalar RMS normalization fit only the 16 fitting questions; no response labels are used. The original three components retain {100*retained:.2f}% of centered fitting-question embedding variance. The full fitting feature rank is {geometry['training_rank']}, versus {geometry['pca3_rank']} for the matched control. Both fitting RMS norms are 1.0; unseen RMS norms are 1.05 for full embeddings and 0.48 for the PCA3 reconstruction. No held-out normalization is fitted. Variance and rank are not measures of retained mathematical meaning.", "",
        "Fit replicas 0/1 of the 16 questions: 256 conversations, 65,536 agent targets. Replica 2 selects stopping: 128 conversations, 32,768 targets. Replica 3 evaluates familiar questions: 128 conversations, 32,768 targets. All 128 conversations on the four unseen development questions form the other evaluation cohort. Neither evaluation cohort selects checkpoints. These are exactly the same identities as the preceding fusion study.", "",
        "Stance/history, full arguments and matched-shuffled arguments each use seeds 42–44. The argument encoder and original additive message fusion remain unchanged. Shuffles preserve the earlier matching constraints and separate roles; actual text-change coverage remains 53.3% in fitting, 14.5% in familiar evaluation, and 77.3% in unseen evaluation. The familiar-question shuffle comparison is therefore weak, and some substitutions exchange paraphrases.", "",
        "## Error rates and loss", "",
        "Means use all three seeds, weighting questions equally. Error is one minus next-choice accuracy; it is not the source agent's mathematical error rate.", "",
        "| Question context | Inputs | Training error | Familiar error | Unseen error | Familiar NLL | Unseen NLL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for rep in REPRESENTATIONS:
        for condition in CONDITIONS:
            train, fam, unseen = [m.loc[(role, rep, condition)] for role in ("train", "familiar", "unseen")]
            lines.append(f"| {LABELS[rep]} | {condition} | {100*train.error:.2f}% | {100*fam.error:.2f}% | {100*unseen.error:.2f}% | {fam.nll:.5f} | {unseen.nll:.5f} |")
    lines += ["", "![Question context comparison](question_context.png)", "",
        "## Controlled contrasts", "",
        "The primary representation comparison is full embedding versus the matched three-component branch. Negative NLL differences and positive accuracy differences favor the first condition. The interaction asks whether richer context increases the added contribution from argument text, beyond changes in the no-argument control.", "",
        "| Cohort / comparison | Accuracy difference, points [95% interval] | NLL difference [95% interval] |",
        "| --- | ---: | ---: |"]
    for role in ("familiar", "unseen"):
        names = [f"{role}/{condition}: full_embedding minus pca3_matched" for condition in ("stance_only", "full_arguments")]
        names += [f"{role}/{rep}: arguments minus stance_only" for rep in REPRESENTATIONS]
        names += [f"{role}: change in argument contribution, full_embedding minus pca3_matched"]
        for name in names:
            p = paired[name]
            lines.append(f"| {name} | {fmt_interval(p['accuracy'], 'accuracy')} | {fmt_interval(p['nll'], 'nll')} |")
    lines += ["", "All declared pairwise comparisons, including the unchanged original model and shuffled inputs, are in `question_context_paired.json`. Intervals average seeds within question and use 1,000 paired question bootstrap draws. They are descriptive and pointwise after many development analyses. The 16 familiar and four unseen questions are different cohorts and are not paired with each other. The fixed replica split does not estimate split-to-split variability.", "",
        "## Each unseen development question", "",
        "These are seed-mean next-choice error rates. The original questions and choices are preserved in `runs/question_context_v1/questions.json`; no correct-answer labels enter the added features.", "",
        "| Question | Matched PCA3: stance/history | Matched PCA3: arguments | Full embedding: stance/history | Full embedding: arguments |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for qid in protocol["roles"]["unseen"]["questions"]:
        errors = [100*(1-qm.loc[("unseen", rep, cond, qid), "accuracy"])
                  for rep in VARIANTS for cond in ("stance_only", "full_arguments")]
        lines.append(f"| {qid} | "+" | ".join(f"{v:.2f}%" for v in errors)+" |")
    lines += ["",
        "## Training and audit", "",
        "Both new representations use the frozen AdamW trainer (learning rate 0.0003, batch 32 communities), 64 updates per epoch, a 50-epoch limit, and six-epoch patience. Checkpoints are selected by question-mean NLL on replica 2, never by evaluation accuracy. Reported training errors are at those selected checkpoints, not necessarily the minimum attainable training error. Every selected new checkpoint has finite, nonzero learned question-projection weights.", "",
        "The report reconstructs preprocessing, verifies frozen data/code/baseline hashes, checks identical examples and targets across all 27 scored model runs, audits donor partitions, and recomputes saved metrics. Six focused checks cover exclusion of held-out question information from fitted transforms, the information restriction in the matched control, preservation of dynamic inputs, CPU/GPU initialization and gradients, padding, and permutation equivariance.", "",
        f"All 18 new fits completed at `{plan['completed_utc']}`. The driver recorded {plan['elapsed_seconds']:.1f} seconds including evaluation. Stopping reasons: {dict(Counter(r['stopping_reason'] for r in rows if r['role']=='train' and r['representation']!='original_pca3'))}. No new LLM calls, encoder downloads, test scoring, autonomous rollouts, or changes to prescribed graph signs were made.", "",
        "## Scope and artifacts", "",
        "A positive result would support retaining richer frozen question features for this prediction task. A negative result would not show that complete question meaning is unimportant: the cached embedding can discard numbers or logical distinctions, and the frozen message encoder remains unconditioned on the full problem at token level. Only 16 distinct fitting questions are available. No result here establishes a causal influence law or argument irrelevance in the source LLM.", "",
        "Configuration and driver: `configs/question_context.yaml`, `scripts/question_context.py`. Frozen protocol, transformed question features, model checkpoints, predictions and resume plan: `runs/question_context_v1/`. Tables include per-run, per-question/round, paired contrasts, sensitivity and geometry. Integrity audit: `reports/question_context_integrity.json`. The PNG and PDF are standalone figures.", ""]
    Path(f"{PREFIX}.md").write_text("\n".join(lines))
    json_write(f"{PREFIX}_integrity.json", {"protocol_sha256": digest(ROOT/"protocol.json"),
        "new_runs": 18, "reused_baselines": 9, "scientific_checks_passed": len(checks.findall('.//testcase')),
        "fit_only_transforms_recomputed": True, "complete_question_and_choices_verified": True,
        "identical_examples_targets_and_splits": True, "donors_confined_to_roles": True,
        "original_data_code_and_checkpoints_unchanged": True, "new_question_projections_nonzero": True,
        "metrics_recomputed": True, "hashes": provenance})
    print(means.to_string(index=False))


if __name__ == "__main__":
    main()
