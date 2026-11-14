"""Verify and summarize the exploratory Ising/transformer error diagnosis."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, digest, json_write
from scripts.diagnose_ising_gap import crossfit_offsets, score_metrics

GAP = Path("runs/ising_gap_diagnosis_v1")
PROBE = Path("runs/ising_initial_state_probe_v1")
SOURCE = Path("runs/transformer_generalization_v1")
REPORT = Path("reports/ising_gap_diagnosis.md")


def main():
    inputs = dict(np.load(GAP/"assessment_inputs.npz", allow_pickle=False))
    target, replica, qi = inputs["target"], inputs["replica"], inputs["qindex"]
    adjusted = dict(np.load(GAP/"adaptation_predictions.npz", allow_pickle=False))
    fits = json.loads((GAP/"adaptation_fits.json").read_text())
    maximum_adaptation_difference = 0.0
    for fit in fits:
        seed = 0 if fit["model"] == "I0" else fit["seed"]-142
        z = inputs[f"score_{fit['model']}"][seed]
        group = np.broadcast_to(qi[:, None], target.shape).copy()
        if fit["condition"] == "question_persona_offset":
            group = group*32 + np.arange(32)[None, :]
        train = np.isin(replica, fit["fit_replicas"])
        w = np.array(fit["offsets"])
        gradient = np.bincount(group[train], weights=expit(z[train]+w[group[train]])-target[train],
                               minlength=len(w))+w
        assert np.max(np.abs(gradient)) < 1e-8
        expected = z[~train]+w[group[~train]]
        stored = adjusted[f"{fit['model']}_{seed}_{fit['condition']}"][~train]
        maximum_adaptation_difference = max(maximum_adaptation_difference, float(np.max(np.abs(expected-stored))))
        np.testing.assert_allclose(stored, expected, atol=5e-6, rtol=1e-6)
    # Perturb real held-out-half labels; predictions for that half must stay identical.
    subset = qi == 0
    y = target[subset].copy()
    z, rep = inputs["score_I0"][0, subset], replica[subset]
    group = np.zeros_like(y, dtype=int)
    a, _ = crossfit_offsets(z, y, group, rep)
    y[rep >= 2] = ~y[rep >= 2]
    b, _ = crossfit_offsets(z, y, group, rep)
    np.testing.assert_array_equal(a[rep >= 2], b[rep >= 2])

    protocol = json.loads((SOURCE/"protocol.json").read_text())
    common = dict(np.load(SOURCE/"development.npz", allow_pickle=False))
    index = json.loads((SOURCE/"episode_index.json").read_text())
    original_index = json.loads(Path("data/processed/train_only_pca/index.json").read_text())
    lookup = {r["episode_id"]: i for i, r in enumerate(original_index)}
    original = np.load("data/processed/train_only_pca/episodes.npz", allow_pickle=False)
    votes0 = original["votes"][[lookup[r["episode_id"]] for r in index], 0]
    count = (votes0 == 1).sum(-1)
    own = np.log((count+.5)/(5-count+.5))
    total = count.sum(-1)
    network = np.log((total+.5)/(160-total+.5))
    e, t = common["pairs"].T
    qids = common["episode_qids"][e]
    predictions = dict(np.load(PROBE/"predictions.npz", allow_pickle=False))
    records = json.loads((PROBE/"refits.json").read_text())
    maximum_probe_difference = 0.0
    own_coefficients = []
    for record in records:
        fold = next(f for f in protocol["folds"] if f["id"] == record["fold"])
        assert not set(fold["train_questions"]) & set(fold["assessment_questions"])
        ids = np.flatnonzero(np.isin(qids, fold["assessment_questions"]))
        z = np.load(SOURCE/"folds"/fold["id"]/"features.npz")["prior_logit"][ids].astype(float)
        columns = [np.ones_like(z), z]
        if record["model"] in ("current_state", "initial_state"):
            columns.append(common["spins"][e[ids], t[ids]].astype(float))
        if record["model"] == "initial_state":
            columns += [own[e[ids]], np.broadcast_to(network[e[ids], None], z.shape)]
        x = np.stack(columns, -1)
        w, mean, scale = (np.asarray(record[k]) for k in ("weights", "mean", "scale"))
        reproduced = z+np.sum((x-mean)/scale*w, axis=-1)
        stored = predictions[record["model"]][ids]
        maximum_probe_difference = max(maximum_probe_difference, float(np.max(np.abs(stored-reproduced))))
        np.testing.assert_allclose(stored, reproduced, atol=1e-10, rtol=1e-10)
        if record["model"] == "current_state":
            own_coefficients.append(float((w/scale)[2]))
    selection = json.loads((PROBE/"selection_freeze.json").read_text())
    freeze = json.loads((PROBE/"refit_freeze.json").read_text())
    probe_audit = json.loads((PROBE/"audit.json").read_text())
    assert selection["created_utc"] <= freeze["created_utc"] <= probe_audit["completed_utc"]
    assert selection["selected_sha256"] == digest(PROBE/"selected_settings.csv")
    assert freeze["refits_sha256"] == digest(PROBE/"refits.json")
    assert not selection["outer_outcomes_scored"] and not freeze["outer_outcomes_scored"]
    summaries = pd.read_csv(GAP/"summary.csv").set_index(["model", "condition"])
    probe = pd.read_csv(PROBE/"summary.csv").set_index("model")
    for kind in probe.index:
        for metric, value in score_metrics(predictions[kind], predictions["target"]).items():
            np.testing.assert_allclose(value, probe.loc[kind, metric], atol=1e-10)
    q = pd.read_csv(PROBE/"by_question.csv")
    old = pd.read_csv(SOURCE/"question_means.csv")
    old = old[(old.split == "assessment") & old.model.isin(["I0", "small"])]
    frames = {m: q[q.model == m].set_index("qid")[["error", "nll"]].sort_index() for m in q.model.unique()}
    frames.update({m: old[old.model == m].set_index("qid")[["error", "nll"]].sort_index() for m in ("I0", "small")})
    rng = np.random.default_rng(447)
    draws = rng.integers(0, 20, (3000, 20))
    contrasts = []
    for first, second in (("current_state", "I0"), ("initial_state", "current_state"), ("small", "current_state")):
        for metric in ("error", "nll"):
            d = (frames[first][metric]-frames[second][metric]).to_numpy()
            lo, hi = np.quantile(d[draws].mean(1), [.025, .975])
            contrasts.append({"first": first, "second": second, "metric": metric,
                "difference": float(d.mean()), "lower": float(lo), "upper": float(hi),
                "questions_better": int((d < 0).sum())})
    pd.DataFrame(contrasts).to_csv(PROBE/"descriptive_contrasts.csv", index=False)
    groups = pd.read_csv(GAP/"subgroups.csv").groupby(["model", "variable", "group"])[
        ["transition_share", "error", "nll", "fraction_of_errors"]].mean()
    overlaps = pd.read_csv(GAP/"error_overlap.csv").groupby("model")[
        ["both_correct", "fixed_ising_error", "introduced_error", "shared_error"]].mean()
    overlap = overlaps.loc["small"]
    share = groups.loc[("I0", "ising_confidence", "below_0.8"), "transition_share"]
    old_error = summaries.loc[("I0", "original"), "error"]
    small_error = summaries.loc[("small", "original"), "error"]
    gain_fraction = (old_error-probe.loc["current_state", "error"])/(old_error-small_error)
    lines = ["# Why are the transformer and Ising scores close?", "",
        "The new diagnosis points to a large easy subset, partial cancellation of the transformer's corrections, and question-specific response biases. A compact persistence correction recovers part of the transformer's advantage. These explain aspects of the measured gap; they do not identify an irreducible prediction ceiling or a unique causal mechanism.", "",
        "All analyses use the same twenty previously studied GPT development questions, four outer folds and 163,840 agent transitions. Original predictors and preprocessing remain frozen. Scores average neural seeds rather than ensemble their probabilities. Accuracy means predicting the agent's next majority-of-five opinion, not answering the mathematical question correctly.", "",
        "## Where the overall score comes from", "",
        "Subgroups below are chosen after earlier overall results and are descriptive. Ising confidence is the larger of its two predicted probabilities. The pooled conditional rates weight transitions within each subgroup; full-cohort rates weight all twenty equally sized questions equally.", "",
        "| Subset | Share of updates | Ising error | Small transformer error |", "| --- | ---: | ---: | ---: |"]
    for label, group in (("Ising confidence < 0.80", "below_0.8"), ("0.80–0.95", "0.8_to_0.95"), ("At least 0.95", "at_least_0.95")):
        i = groups.loc[("I0", "ising_confidence", group)]
        s = groups.loc[("small", "ising_confidence", group)]
        lines.append(f"| {label} | {100*i.transition_share:.2f}% | {100*i.error:.2f}% | {100*s.error:.2f}% |")
    lines += ["", f"The difficult {100*share:.2f}% contains {100*groups.loc[('I0','ising_confidence','below_0.8'),'fraction_of_errors']:.2f}% of Ising errors. The small transformer improves classification there but has worse NLL: {groups.loc[('small','ising_confidence','below_0.8'),'nll']:.5f} versus {groups.loc[('I0','ising_confidence','below_0.8'),'nll']:.5f}. Its probability estimates on difficult cases still need work; classification gain alone is insufficient.", "",
        f"Across all updates the small transformer corrects {100*overlap.fixed_ising_error:.2f} percentage points of Ising errors, introduces {100*overlap.introduced_error:.2f} points, and shares {100*overlap.shared_error:.2f} points of errors. The net gain is {100*(old_error-small_error):.2f} points. These categories describe paired predictions, not a deployable oracle that knows which prediction is correct.", "",
        "## Sampling variation does not explain everything", "",
        "| Target vote agreement | Share of updates | Small transformer error | Share of its errors |",
        "| --- | ---: | ---: | ---: |"]
    for n in (3, 4, 5):
        v = groups.loc[("small", "target_votes", str(n))]
        lines.append(f"| {n}/5 | {100*v.transition_share:.2f}% | {100*v.error:.2f}% | {100*v.fraction_of_errors:.2f}% |")
    lines += ["", "Votes are used retrospectively here, never as next-outcome predictor inputs. Five agreeing samples do not establish a deterministic response or a Bayes accuracy ceiling. However, attributing all residual errors to split votes is inconsistent with this breakdown.", "",
        "## Question-specific adaptation diagnostic", "",
        "For each question, fit offsets on complete conversations in replicas 0/1 and evaluate replicas 2/3, then reverse. All eight graphs and all eight updates of each conversation stay together. Base predictors were originally fitted on the other fifteen questions. Offsets minimize summed binary cross-entropy plus one half the sum of squared offsets; this penalty was fixed before the adaptation fits. No hyperparameter search is performed.", "",
        "This intentionally supplies outcomes from other conversations about the same question. It measures familiar-question adaptation and must not be presented as new-question generalization or a better model using the original information budget.", "",
        "| Predictor | Original error | One offset per question | One offset per question and persona |",
        "| --- | ---: | ---: | ---: |"]
    for kind, label in (("I0", "Ising"), ("small", "Small transformer")):
        values = [100*summaries.loc[(kind, condition), "error"] for condition in ("original", "question_offset", "question_persona_offset")]
        lines.append(f"| {label} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% |")
    lines += ["", "A single question offset accounts for most of the adaptation improvement. This demonstrates a stable question-dependent prediction bias across recorded conversations. It does not by itself identify whether that bias comes from question representation, insufficient question diversity, or unrepresented message content.", "",
        "## A new, inexpensive state experiment", "",
        "Three small corrections were tested with the existing nested whole-question folds. Each uses the same frozen, fold-fitted Ising logit as an offset. The calibration control adds an intercept and a global logit coefficient (two correction coefficients). The current-state control adds the receiver's current spin (three). The initialization model adds two more inputs: smoothed log-odds from that receiver's five time-zero responses and from all 32 agents' time-zero responses in the same episode (five). Pooling the initial network is explicitly an additional global input. No later votes, future opinions or other same-question episodes construct these features.", "",
        "All three receive the same penalty grid [0.0001, 0.001, 0.01], applied to standardized correction coefficients. Standardization uses fitting questions only. Three inner folds choose mean validation NLL for each outer fold/model; twelve outer refits are all frozen before their outcomes are scored. This adds 108 inner fits and 12 outer fits, with no new transformer training or source generation.", "",
        "| Predictor | Outer error | Outer NLL |", "| --- | ---: | ---: |"]
    entries = [("Original Ising", summaries.loc[("I0", "original")]),
               ("Ising + global calibration", probe.loc["calibration"]),
               ("Ising + calibration + current own opinion", probe.loc["current_state"]),
               ("Same + initial response tendencies", probe.loc["initial_state"]),
               ("Small transformer (previous frozen fits)", summaries.loc[("small", "original")])]
    for name, v in entries:
        lines.append(f"| {name} | {100*v.error:.2f}% | {v.nll:.5f} |")
    lines += ["", f"The simple current-opinion correction recovers about {100*gain_fraction:.0f}% of the small transformer's net accuracy advantage over Ising. Its coefficient is positive in all four folds ({min(own_coefficients):.3f}–{max(own_coefficients):.3f} log-odds per signed spin). The remaining transformer advantage is {100*(probe.loc['current_state','error']-small_error):.2f} percentage points. Initial response features do not improve this tested linear correction; this does not rule out other estimators or nonlinear uses of initialization.", "",
        "The current-opinion coefficient measures a predictive persistence association. The released spin prompt does not explicitly feed back the receiver's own previous answer, so it should not be interpreted as a proven causal self-influence mechanism. A compact candidate expression is:", "",
        "```text\nlogit P(next majority = A) = b + a * Ising_score + c * current_spin\n```", "",
        "The original 19 Ising coefficients are fitted separately within each training fold; three correction coefficients are then fitted. This parameterization contains redundancy with the original intercept/scale and is not claimed to be a minimal parameter count.", "",
        "Paired differences below use 3,000 question-bootstrap draws (seed 447), with neural seeds averaged within questions. Intervals are descriptive and pointwise: repeated exposure to this development pool, overlapping training folds and multiple comparisons prevent treating them as fresh confirmation.", "",
        "| Comparison | Error difference, pp [descriptive 95% interval] | NLL difference [interval] |",
        "| --- | ---: | ---: |"]
    for first, second, label in (("current_state", "I0", "Current-state correction − Ising"),
                                 ("initial_state", "current_state", "Initialization − current-state correction"),
                                 ("small", "current_state", "Small transformer − current-state correction")):
        d = next(v for v in contrasts if v["first"] == first and v["second"] == second and v["metric"] == "error")
        n = next(v for v in contrasts if v["first"] == first and v["second"] == second and v["metric"] == "nll")
        lines.append(f"| {label} | {100*d['difference']:+.2f} [{100*d['lower']:+.2f}, {100*d['upper']:+.2f}] | {n['difference']:+.5f} [{n['lower']:+.5f}, {n['upper']:+.5f}] |")
    lines += ["", "## Interpretation and next discriminating experiment", "",
        "The previous capacity study already showed that the same transformer can fit observed training labels much better than Ising (4.41% versus 9.15% error), while transferring worse. The present evidence therefore favors a transfer/information problem over a demonstrated inability to express the Ising rule. It remains possible that the transferable rule conditional on the supplied opinions and compressed context is relatively simple.", "",
        "The source protocol explicitly asks for an A/B snap judgment without working through the problem, and asks each sender to justify its already selected option. This makes a strong opinion-aggregation baseline plausible, but the protocol alone does not prove that arguments are irrelevant or that influence is linear. Earlier full-question and argument-encoding experiments also failed to establish a transferable gain; simply repeating those changes is not a new explanation.", "",
        "The next useful experiment should distinguish stochastic source responses, omitted argument information and across-question transfer. On a broader fixed-source question bank, retain representative natural updates and a separately reported low-confidence subset. For identical complete prompts, collect disjoint repeated groups of five responses to estimate reproducibility of the actual majority target. For matched opinions/personas/signs, vary the incoming argument text and compare the source's response distribution. Fit both the compact current-state correction and the small transformer on matched data, with opinion-only versus argument-aware inputs. Vary question breadth at a controlled transition budget. Freeze all procedures before fresh question assessment; keep question/template families together. Hard-case sampling should be reported separately or weighted back to the target distribution.", "",
        "This would test whether useful information exists beyond the compact law and whether the predictor can learn it. Repeated prompts alone cannot establish causal argument effects; argument interventions alone cannot establish predictive generalization. If an open-source LLM is used, regenerate a matched training/assessment study for that source rather than combining it with the GPT results.", "",
        "## Verification", "",
        "Eight focused tests passed, covering independent numerical optimization, objective gradients, finite single-class offsets, error bookkeeping, initial/future vote separation, training-only normalization and exclusion of each assessed replica's labels from its calibration. All old aggregate scores were reproduced. Saved offsets and all twelve new outer predictions were independently reconstructed. A real-data assessment-label perturbation left its own assessed predictions unchanged. Selection/refit timestamps and source checksums were verified. No official test evaluation, source-model calls, paid API usage or autonomous rollout was added.", "",
        "Artifacts: `runs/ising_gap_diagnosis_v1/`, `runs/ising_initial_state_probe_v1/`, and `reports/ising_gap_diagnosis_checks.xml`. Drivers: `scripts/diagnose_ising_gap.py`, `scripts/probe_initial_opinion_state.py`, and `scripts/report_ising_gap_diagnosis.py`."]
    REPORT.write_text("\n".join(lines)+"\n")
    suites = ET.parse("reports/ising_gap_diagnosis_checks.xml").getroot()
    cases = list(suites.iter("testcase"))
    assert len(cases) == 8 and not list(suites.iter("failure")) and not list(suites.iter("error"))
    sources = {}
    for folder in (GAP, PROBE):
        sources.update(json.loads((folder/"audit.json").read_text())["source_artifacts"])
    for path, sha in sources.items():
        assert digest(path) == sha, path
    files = [p for folder in (GAP, PROBE) for p in folder.rglob("*") if p.is_file() and p.name != "completion_audit.json"]
    files += [REPORT, Path(__file__), Path("scripts/diagnose_ising_gap.py"), Path("scripts/probe_initial_opinion_state.py"),
              Path("tests/test_ising_gap_diagnosis.py"), Path("tests/test_initial_opinion_state_probe.py"),
              Path("reports/ising_gap_diagnosis_checks.xml"), Path("reports/ising_initial_state_probe.log")]
    json_write(GAP/"completion_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "focused_tests_passed": 8, "source_files_verified_unchanged": len(sources),
        "calibration_fits_reconstructed": len(fits), "new_outer_fits_reconstructed": len(records),
        "maximum_adaptation_prediction_difference": maximum_adaptation_difference,
        "maximum_initial_state_prediction_difference": maximum_probe_difference,
        "real_assessment_label_perturbation_passed": True,
        "selection_refit_assessment_order_verified": True,
        "core_sha256": code_digest(), "new_official_test_evaluations": 0, "source_llm_calls": 0,
        "new_transformer_training": 0, "fresh_confirmatory_dataset": False,
        "artifacts": {str(p): digest(p) for p in files}})
    print(json.dumps({"report": str(REPORT), "tests": 8, "source_files_unchanged": len(sources),
        "maximum_adaptation_difference": maximum_adaptation_difference,
        "maximum_probe_difference": maximum_probe_difference}, indent=2))


if __name__ == "__main__":
    main()
