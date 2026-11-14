"""Recompute the development-only response diagnostic from saved predictions."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from mac_agents.metrics import paired_bootstrap, score
from scripts.argument_response import CONFIG, ROOT, cohort, implementation_digest
from scripts.argument_features import load_cache

REPORT = Path("reports/argument_response_results.md")
METRICS = ("nll", "accuracy", "brier", "fcba", "flip_accuracy", "stay_accuracy")
KEYS = ["episode_id", "round", "agent"]


def format_interval(value, percent=False):
    scale = 100 if percent else 1
    places = 2 if percent else 5
    return f"{scale*value['difference']:+.{places}f} [{scale*value['ci95'][0]:+.{places}f}, {scale*value['ci95'][1]:+.{places}f}]"


def main():
    cfg = config(CONFIG)
    arrays, index, vectors, feature_meta = load_cache(cfg, verify=True)
    source_manifest = json.loads(Path("data/data_manifest.json").read_text())
    for row in index:
        path = Path(row["raw_path"])
        name = str(path.relative_to("data/raw/upstream"))
        if not (feature_meta["raw_files_sha256"][row["episode_id"]] ==
                source_manifest["files"][name]["sha256"] == digest(path)):
            raise ValueError("Raw message differs from pinned source")
    ids = arrays["text_ids"]
    real = np.flatnonzero(ids)
    e,t,r,k = np.unravel_index(real,ids.shape)
    qids = np.array([row["qid"] for row in index])
    partitions = np.array([row["question_split"] for row in index])
    sender = arrays["neighbors"][e,r,k]
    for seed in cfg["training"]["seeds"]:
        donor = np.load(Path(cfg["data"]["cache"])/f"donors_s{seed}.npy",allow_pickle=False).reshape(-1)
        np.testing.assert_array_equal(np.sort(donor[real]),real)
        de,dt,dr,dk = np.unravel_index(donor[real],ids.shape)
        ds = arrays["neighbors"][de,dr,dk]
        for first,second in ((t,dt),(r,dr),(sender,ds),(qids[e],qids[de]),(partitions[e],partitions[de]),
            (arrays["signs"][e,r,k],arrays["signs"][de,dr,dk]),
            (arrays["stance"][e,t,r,k],arrays["stance"][de,dt,dr,dk]),
            (arrays["spins"][e,t,sender],arrays["spins"][de,dt,ds]),
            (arrays["spins"][e,t,r],arrays["spins"][de,dt,dr])):
            np.testing.assert_array_equal(first,second)
        assert np.all((donor[real]==real)|(e!=de))
    checks=ET.parse("reports/argument_checks.xml").getroot()
    if checks.findall(".//failure") or checks.findall(".//error") or checks.findall(".//skipped"):
        raise ValueError("Scientific/GPU checks must all pass before reporting")
    check_count=len(checks.findall(".//testcase"))
    del arrays, index, vectors
    plan = json.loads((ROOT / "plan.json").read_text())
    if not plan.get("complete") or len(plan["runs"]) != len(cohort(cfg)):
        raise ValueError("Complete all declared runs before making the cohort report")
    if plan["protocol_sha256"] != digest(ROOT / "protocol.json"):
        raise ValueError("Protocol changed")
    expected = {(x["kind"],x["condition"],x["seed"]) for x in cohort(cfg)}
    if {(x["kind"],x["condition"],x["seed"]) for x in plan["runs"]} != expected:
        raise ValueError("Cohort differs from protocol")
    rows, questions, rounds, reference_rows = [], [], [], []
    reference = None
    fingerprints = {}
    frames = {}
    for item in plan["runs"]:
        run = Path(item["run"])
        meta = json.loads((run / "manifest.json").read_text())
        if (meta["implementation_sha256"] != implementation_digest() or meta["smoke"] or
            meta["status"] != "complete" or meta["protocol_sha256"] != digest(ROOT / "protocol.json")):
            raise ValueError("Ineligible run")
        for name, expected_sha in meta["artifact_sha256"].items():
            if digest(run / name) != expected_sha:
                raise ValueError(f"Saved artifact changed: {run / name}")
        fingerprints[str(run)] = {"manifest_sha256": digest(run / "manifest.json"), **meta["artifact_sha256"]}
        variants = ["validation"] + (["validation_substitution"] if item["condition"] == "full_arguments" else [])
        for variant in variants:
            frame = pd.read_parquet(run / f"{variant}_predictions.parquet").sort_values(KEYS).reset_index(drop=True)
            if frame.duplicated(KEYS).any() or set(frame.question_split) != {"validation"}:
                raise ValueError("Duplicate predictions or non-development evaluation")
            truth = frame[KEYS+["qid","current_state","target"]]
            if reference is None:
                reference = truth
            else:
                pd.testing.assert_frame_equal(reference,truth)
            summary, q = score(frame)
            saved = json.loads((run / f"{variant}_metrics.json").read_text())
            for metric in METRICS:
                np.testing.assert_allclose(summary[metric],saved[metric],rtol=0,atol=1e-12)
            label = item["condition"] if variant == "validation" else "full_arguments_substituted_at_evaluation"
            frames[(item["kind"],label,item["seed"])] = frame
            row = {"kind":item["kind"],"condition":label,"seed":item["seed"],"scope":"all",
                   "parameters":meta["parameters"],"inactive_parameters":meta["inactive_text_parameters_in_stance_only"],
                   "training_seconds":meta["elapsed_seconds"],"best_step":meta["best_step"], **summary}
            rows.append(row)
            for scope, subset in (("all", frame), ("matched_substitution",frame[frame.has_matched_substitution]),
                                  ("different_text",frame[frame.has_different_text])):
                if subset.empty:
                    continue
                _, q = score(subset)
                q = q.assign(kind=item["kind"],condition=label,seed=item["seed"],scope=scope)
                questions.extend(q.to_dict("records"))
            for t, part in frame.groupby("round"):
                _, q = score(part)
                q = q.assign(kind=item["kind"],condition=label,seed=item["seed"],round=int(t))
                rounds.extend(q.to_dict("records"))
    run_frame = pd.DataFrame(rows)
    qframe = pd.DataFrame(questions)
    round_frame = pd.DataFrame(rounds)
    run_frame.to_csv("reports/argument_response_runs.csv",index=False)
    qframe.to_csv("reports/argument_response_by_question.csv",index=False)
    round_frame.to_csv("reports/argument_response_by_question_round.csv",index=False)
    means = qframe.groupby(["kind","condition","scope","qid"])[list(METRICS)].mean()
    contrasts = []
    for kind in ("A0","A2"):
        for first, second in (("full_arguments","stance_only"),("full_arguments","shuffled_arguments"),
                              ("full_arguments_substituted_at_evaluation","full_arguments")):
            for scope in ("all","matched_substitution","different_text"):
                for metric in METRICS:
                    paired = paired_bootstrap(means.loc[(kind,first,scope),metric],
                                              means.loc[(kind,second,scope),metric],
                                              cfg["evaluation"]["bootstrap_replicates"],cfg["evaluation"]["bootstrap_seed"])
                    contrasts.append({"kind":kind,"first":first,"second":second,"scope":scope,"metric":metric,**paired})
    json_write("reports/argument_response_paired.json",contrasts)
    sensitivity = []
    for kind in ("A0","A2"):
        for seed in cfg["training"]["seeds"]:
            original=frames[(kind,"full_arguments",seed)]
            replaced=frames[(kind,"full_arguments_substituted_at_evaluation",seed)]
            sensitivity.append({"kind":kind,"seed":seed,
                                "mean_absolute_probability_change":float(np.abs(original.p_positive-replaced.p_positive).mean()),
                                "fraction_of_choices_changed":float((original.prediction!=replaced.prediction).mean())})
    sensitivity=pd.DataFrame(sensitivity)
    sensitivity.to_csv("reports/argument_response_substitution_sensitivity.csv",index=False)
    # The prior history fits provide a descriptive reference on precisely these examples.
    old_plan = json.loads(Path("runs/inductive_study/plan.json").read_text())
    old_items = old_plan["runs"]
    for item in old_items:
        if item["condition"] != "history" or item["model"] not in ("A0","A2"):
            continue
        path = Path(item["run"])
        frame = pd.read_parquet(path / "validation_predictions.parquet").sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(reference,frame[KEYS+["qid","current_state","target"]])
        metrics,_=score(frame)
        reference_rows.append({"kind":item["model"],"seed":item["seed"],**metrics})
    refs = pd.DataFrame(reference_rows)
    refs.to_csv("reports/argument_response_history_reference.csv",index=False)
    integrity = {"implementation_sha256":implementation_digest(), "protocol_sha256":digest(ROOT/"protocol.json"),
                 "plan_sha256":digest(ROOT/"plan.json"),"feature_metadata_sha256":digest(Path(cfg["data"]["cache"])/"metadata.json"),
                 "runs":fingerprints,"all_predictions_rescored":True,"identical_examples_and_targets":True,
                 "only_development_predictions":True,"history_reference_examples_match":True,
                 "raw_message_hashes_match_pinned_source":True,"matched_donor_cues_independently_verified":True,
                 "scientific_checks_passed":check_count,"checks_sha256":digest("reports/argument_checks.xml"),
                 "questions":int(reference.qid.nunique()),"agent_transitions_per_run":len(reference)}
    json_write("reports/argument_response_integrity.json",integrity)
    summary = run_frame[run_frame.condition.isin(["stance_only","full_arguments","shuffled_arguments"])].groupby(["kind","condition"])
    means_runs = summary[list(METRICS)+["parameters","training_seconds"]].mean()
    short = {"stance_only":"Stance only","full_arguments":"Full text","shuffled_arguments":"Matched text substitutions"}
    colors = {"stance_only":"#64748b","full_arguments":"#007f83","shuffled_arguments":"#c2802c"}
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(11.5,4.4))
    for ax,metric,title,scale in zip(axes,["nll","accuracy"],["NLL · lower is better","Accuracy (%) · higher is better"],[1,100]):
        for group_idx,kind in enumerate(("A0","A2")):
            for j,condition in enumerate(("stance_only","full_arguments","shuffled_arguments")):
                x = group_idx*4+j
                values = run_frame[(run_frame.kind==kind)&(run_frame.condition==condition)][metric].to_numpy()*scale
                ax.scatter(x+np.linspace(-.1,.1,len(values)),values,color=colors[condition],s=26,alpha=.75)
                ax.plot([x-.25,x+.25],[values.mean()]*2,color=colors[condition],lw=3,label=short[condition] if group_idx==0 else None)
        ax.set_xticks([1,5],["Fixed aggregation (A0)","Attention (A2)"])
        ax.set_title(title)
        ax.grid(axis="y",alpha=.18)
    axes[0].legend(loc="best",fontsize=9)
    fig.suptitle("Delivered-message response prediction · 4 development questions\nDots: three training seeds; lines: means. Validation also chooses stopping time.",fontsize=12)
    fig.tight_layout()
    fig.savefig("reports/argument_response_results.png",dpi=170)
    plt.close(fig)
    primary_nll = next(r for r in contrasts if (r["kind"],r["first"],r["second"],r["scope"],r["metric"])==
                       ("A2","full_arguments","stance_only","all","nll"))
    primary_acc = next(r for r in contrasts if (r["kind"],r["first"],r["second"],r["scope"],r["metric"])==
                       ("A2","full_arguments","stance_only","all","accuracy"))
    text = ["# Argument-aware response prediction: development diagnostic", "",
        "**Exploratory results, not a fresh test.** The study uses 16 fitting questions and four validation questions. Validation chooses stopping time and has been used in previous studies. Paired intervals below describe these four questions; they are not confirmatory evidence.","",
        "The predictor observes `messages_history[t]` after delivery and predicts `s[t+1]`. This starts later than the completed opinion-only forecast. It is response prediction with real incoming messages, not autonomous simulation.","",
        f"Primary attention comparison, full text minus stance only: NLL **{format_interval(primary_nll)}**; accuracy **{format_interval(primary_acc,True)} percentage points** (descriptive pointwise 95% question-bootstrap intervals).", "",
        "In this completed cohort, adding full text improves neither mean accuracy nor mean NLL over stance alone in either architecture. Original and matched-substituted messages give very similar results. This representation has not demonstrated an added predictive benefit; the experiment does not establish that arguments are irrelevant.","",
        "## Matched conditions", "",
        "Every condition retains opinions, two-snapshot history, training-only-PCA context, sender/recipient associations, and the prescribed friendly/unfriendly labels. The source messages explicitly favor the sender's current option in all audited cases. The frozen 384-dimensional encoder is reused from the earlier screen, but each delivered message is now attached to its own edge; inboxes are not pre-averaged.","",
        "- **Stance only:** zero text embedding; all other inputs retained.",
        "- **Full text:** the original individual message embedding.",
        "- **Matched substitutions:** text is permuted between different episodes within question, round, sender, recipient, relationship, communicated stance, sender opinion, and receiver opinion. Training and validation are kept separate. Singleton strata retain their original message.","",
        "The text projection and all other parameter shapes are identical within each architecture, with the same initialization for each seed. The text projection is inactive in stance only. Equal nominal parameter counts therefore do not mean equal effective capacity.","",
        "## Results", "", "| Aggregation | Input | NLL | Accuracy | Flip accuracy | Parameters |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for kind in ("A0","A2"):
        for condition in ("stance_only","full_arguments","shuffled_arguments"):
            r=means_runs.loc[(kind,condition)]
            text.append(f"| {kind} | {short[condition]} | {r.nll:.5f} | {100*r.accuracy:.2f}% | {100*r.flip_accuracy:.2f}% | {int(r.parameters):,} |")
    text += ["", "Question means are averaged over seeds 42, 43, and 44. Each run has 4,096 fitting snapshots (131,072 agent transitions) and 1,024 validation snapshots (32,768 transitions).", "",
             "![Development results](argument_response_results.png)","", "## Paired contrasts", "",
             "Negative NLL differences and positive accuracy differences favor the first condition. Evaluation-time substitution uses a full-text model without retraining; positive NLL there means prediction deteriorated.","",
             "| Aggregation | First minus second | NLL difference [95% interval] | Accuracy difference, points [95% interval] |",
             "| --- | --- | ---: | ---: |"]
    labels={**short,"full_arguments_substituted_at_evaluation":"Full model, substituted at evaluation"}
    for kind in ("A0","A2"):
        for first,second in (("full_arguments","stance_only"),("full_arguments","shuffled_arguments"),
                             ("full_arguments_substituted_at_evaluation","full_arguments")):
            selected={r["metric"]:r for r in contrasts if (r["kind"],r["first"],r["second"],r["scope"])==(kind,first,second,"all")}
            text.append(f"| {kind} | {labels[first]} − {labels[second]} | {format_interval(selected['nll'])} | {format_interval(selected['accuracy'],True)} |")
    text += ["", "Average seeds within each question, then take 1,000 paired bootstrap resamples of questions, seed 1729. No agent-level pseudo-replication or multiplicity-adjusted claims. Subset analyses for messages that can be exchanged and for actual text changes are saved in `argument_response_paired.json`.","",
             "As a descriptive check on model sensitivity, the same full-text models were evaluated with the matched substitutions, without retraining:","",
             "| Model | Mean absolute probability change | Fraction of predicted choices changed |",
             "| --- | ---: | ---: |"]
    for kind,r in sensitivity.groupby("kind")[["mean_absolute_probability_change","fraction_of_choices_changed"]].mean().iterrows():
        text.append(f"| {kind} | {100*r.mean_absolute_probability_change:.3f} percentage points | {100*r.fraction_of_choices_changed:.3f}% |")
    text += ["",
             "### Primary comparison by question", "", "| Question | Full-text NLL − stance NLL | Full-text accuracy − stance accuracy |", "| --- | ---: | ---: |"]
    a=means.loc[("A2","full_arguments","all")]; b=means.loc[("A2","stance_only","all")]
    for qid in a.index:
        text.append(f"| {qid} | {a.loc[qid,'nll']-b.loc[qid,'nll']:+.5f} | {100*(a.loc[qid,'accuracy']-b.loc[qid,'accuracy']):+.2f} points |")
    text += ["", "## Substitution coverage", "",
             "| Seed | Partition | Matched messages | Actually different text | Receiver updates with different text |", "| ---: | --- | ---: | ---: | ---: |"]
    for seed,partitions in feature_meta["shuffle"].items():
        for partition,r in partitions.items():
            denominator=131072 if partition=="train" else 32768
            text.append(f"| {seed} | {partition} | {r['matched_messages']:,}/{r['messages']:,} | {r['different_text_messages']:,}/{r['messages']:,} | {r['receiver_updates_with_different_text']:,}/{denominator:,} |")
    text += ["", "Graph identity and complete prior inbox/history are not matched. Thus substitutions can alter other contextual associations besides reasoning quality. Unchanged or repeated text weakens the perturbation. Neither model-only substitutions nor attention weights establish causal influence on the source LLM.","",
             "## Earlier history-model reference", "",
             "These saved models used the same examples and compact context, with no delivered-message input. Their architecture differs from the new edge-message model, so the primary matched comparison above is the appropriate test of added text for this implementation.","",
             "| Earlier model | NLL | Accuracy |", "| --- | ---: | ---: |"]
    for kind,r in refs.groupby("kind")[["nll","accuracy"]].mean().iterrows():
        text.append(f"| {kind}/history | {r.nll:.5f} | {100*r.accuracy:.2f}% |")
    seconds=run_frame[run_frame.condition.isin(short)].training_seconds.sum()
    text += ["", "## Provenance, costs and limits", "",
        f"All {check_count} scientific checks passed, including CPU/MPS agreement, message timing, future-state exclusion, shuffle partition/cue preservation, graph locality, isolated agents and permutation equivariance. Reporting independently verifies raw-source hashes and donor matching, recomputes every saved score, and confirms identical prediction keys and targets.","",
        f"All 18 declared fits completed. Predictor fitting plus final evaluation took {seconds:.1f} recorded seconds; the cohort driver recorded {plan['elapsed_seconds']:.1f} seconds. Feature preparation took {feature_meta['elapsed_seconds']:.1f} seconds, including {feature_meta['encoder']['new_encoding_seconds']:.1f} seconds in the new-encoding stage. These are local preprocessing/training costs, not complete autonomous-simulation speedups.","",
        f"The encoder has {feature_meta['encoder']['frozen_parameters']:,} frozen parameters, revision `{cfg['data']['encoder_revision']}`. {feature_meta['encoder']['reused_vectors']:,} vectors were reused and {feature_meta['encoder']['new_vectors']:,} new vectors encoded. Of {feature_meta['unique_texts']:,} unique messages, {feature_meta['encoder']['truncated_unique_texts']} exceed its 256-token cap ({feature_meta['encoder']['truncated_message_occurrences']} message occurrences). No new source-LLM calls or paid APIs were used.","",
        "Full text contains reasoning and other linguistic cues. A gain would show useful extra content for the tested representation, not specifically logical reasoning or a universal law. A null result would not establish that arguments are irrelevant: the generic encoder, compact context, update architecture and four-question validation can limit this comparison.","",
        "No test messages were encoded or scored in this study. Existing test outcomes were inspected in the preceding study, so a future confirmatory experiment needs fresh questions/outcomes. No autonomous rollout is reported because this model requires observed messages. A compact memory with its own learned update remains a separate next step.","",
        "Prescribed graph labels remain intact. The deferred emergent-credibility experiment was not run.","",
        "## Reproduce", "", "```bash", ".venv/bin/python -u scripts/argument_response.py prepare --device mps",
        ".venv/bin/python -u scripts/argument_response.py validation --device mps",
        ".venv/bin/python scripts/report_argument_response.py", "```", "",
        "The independent implementation, config, data and protocol hashes must match before runs can resume. Completed artifacts are verified before reuse. Core training/evaluation and previous frozen study files are unchanged. The protocol is in `runs/argument_response_v1/protocol.json`; all predictions, checkpoints and manifests are stored under the same run root. Audit results are in `argument_response_integrity.json` and `argument_feature_audit.json`."]
    REPORT.write_text("\n".join(text)+"\n")
    print(json.dumps({"report":str(REPORT),"primary_nll":primary_nll,"primary_accuracy":primary_acc,
                      "runs":len(plan["runs"]),"verified":True},indent=2))


if __name__=="__main__":
    main()
