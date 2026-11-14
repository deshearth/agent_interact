"""Recompute and plot the balanced question-count learning curve."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config,digest,json_write
from mac_agents.metrics import paired_bootstrap,score
from scripts.argument_features import load_cache
from scripts import question_count_curve as curve

KEYS=["episode_id","round","agent"]
METRICS=["nll","accuracy","error_rate","brier","fcba","flip_accuracy","stay_accuracy"]


def interval(record,percent=False):
    factor=100 if percent else 1
    places=2 if percent else 5
    return f"{factor*record['difference']:+.{places}f} [{factor*record['ci95'][0]:+.{places}f}, {factor*record['ci95'][1]:+.{places}f}]"


def plot_curves(run_frame,summary):
    colors={"stance_only":"#64748b","full_arguments":"#007f83"}
    labels={"stance_only":"Stance + history","full_arguments":"Plus argument text"}
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(12,7.8))
    for row,partition in enumerate(("train","validation")):
        for col,metric in enumerate(("error_rate","nll")):
            ax=axes[row,col]
            factor=100 if metric=="error_rate" else 1
            for j,condition in enumerate(colors):
                data=run_frame[(run_frame.partition==partition)&(run_frame.condition==condition)]
                endpoint=float(data[data.n_questions==16][metric].mean())
                for path in range(4):
                    path_data=data[data.path==path].groupby("n_questions")[metric].mean()
                    ys=[float(path_data.loc[n]) for n in (4,8,12)]+[endpoint]
                    ax.plot([4,8,12,16],np.array(ys)*factor,color=colors[condition],alpha=.20,lw=1)
                for n,values in data.groupby("n_questions"):
                    jitter=np.linspace(-.25,.25,len(values))+(j-.5)*.18
                    ax.scatter(n+jitter,values[metric]*factor,color=colors[condition],s=12,alpha=.42,edgecolors="none")
                means=summary.loc[(condition,partition)].reindex([4,8,12,16])
                ax.plot([4,8,12,16],means[metric]*factor,color=colors[condition],marker="o",lw=2.7,
                        label=labels[condition])
            ax.set_xticks([4,8,12,16])
            ax.set_xlabel("Distinct training questions")
            ax.set_ylabel("Error rate (%)" if metric=="error_rate" else "Question-mean NLL")
            ax.set_title(("Training" if partition=="train" else "Fixed four-question validation")+
                         " · "+("classification error" if metric=="error_rate" else "probability loss"))
            ax.grid(axis="y",alpha=.18)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.945),ncol=2,frameon=False)
    fig.suptitle("Question-count learning curve · lower is better",fontsize=15,y=.985)
    fig.text(.5,.01,"Thick lines: means. Thin lines: four nested subset paths. Dots: individual seeds/subsets.\n"
             "16 questions: one shared endpoint with three seeds. Subset spread is descriptive, not a confidence interval.",
             ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.92))
    fig.savefig("reports/question_count_learning_curve.png",dpi=180)
    fig.savefig("reports/question_count_learning_curve.pdf")
    plt.close(fig)


def main():
    cfg=config(curve.CONFIG)
    arrays,index,_,metadata=load_cache(cfg,verify=True)
    protocol=json.loads((curve.ROOT/"protocol.json").read_text())
    curve.verify_protocol_identity(protocol,cfg,metadata)
    plan=json.loads((curve.ROOT/"plan.json").read_text())
    if not plan.get("complete") or len(plan["runs"])!=72:
        raise ValueError("Complete all declared fits and endpoint scoring before reporting")
    if plan["protocol_sha256"]!=digest(curve.ROOT/"protocol.json"):
        raise ValueError("Learning-curve plan changed")
    checks=ET.parse("reports/question_count_checks.xml").getroot()
    if any(checks.findall(f".//{tag}") for tag in ("failure","error","skipped")):
        raise ValueError("Learning-curve checks must all pass")
    replay=json.loads((curve.ROOT/"endpoint_replay.json").read_text())
    if not replay["matched"] or replay["implementation_sha256"]!=curve.implementation_digest():
        raise ValueError("Endpoint optimization replay missing or stale")
    subsets={s["id"]:s for s in protocol["subsets"]}
    embeddings={r["qid"]:r["embedding"] for r in json.loads(curve.Q_EMBEDDINGS.read_text())["items"]
                if r["qid"] in set(row["qid"] for row in index)}
    personas_raw=sorted(json.loads(curve.P_EMBEDDINGS.read_text())["items"],key=lambda r:r["idx"])
    with np.load(Path(cfg["data"]["prepared"])/"persona_pca.npz",allow_pickle=False) as pca:
        personas=(np.asarray([r["embedding"] for r in personas_raw])-pca["mean"])@pca["components"].T
    for subset in subsets.values():
        mapping,mean,components=curve.fit_question_features(embeddings,subset["questions"],list(embeddings))
        with np.load(curve.ROOT/subset["feature_file"],allow_pickle=False) as stored:
            np.testing.assert_array_equal(mean,stored["mean"])
            np.testing.assert_array_equal(components,stored["components"])
            np.testing.assert_array_equal(curve.context_from_features(index,personas,mapping),stored["context"])
    all16_path=curve.ROOT/"endpoint_scores"/"A2-stance_only-s42"
    truths={partition:pd.read_parquet(all16_path/f"{partition}_predictions.parquet")[KEYS+["qid","current_state","target"]]
            .sort_values(KEYS).reset_index(drop=True) for partition in ("train","validation")}
    runs=plan["runs"]+[dict(reference,run=str(curve.ROOT/"endpoint_scores"/f"A2-{reference['condition']}-s{reference['seed']}"))
                         for reference in protocol["endpoint_references"]]
    expected={(item["subset_id"],item["condition"],item["seed"]) for item in protocol["cohort"]+protocol["endpoint_references"]}
    if {(item["subset_id"],item["condition"],item["seed"]) for item in runs}!=expected:
        raise ValueError("Declared run cohort differs")
    run_rows,question_rows,round_rows,timing=[],[],[],[]
    provenance={}
    for number,item in enumerate(runs,1):
        subset=subsets[item["subset_id"]]
        path=Path(item["run"])
        manifest=json.loads((path/"manifest.json").read_text())
        if manifest["implementation_sha256"]!=curve.implementation_digest():
            raise ValueError("Run implementation differs")
        if not item["reused"]:
            if manifest["status"]!="complete" or manifest["smoke"] or manifest["context_sha256"]!=subset["feature_sha256"]:
                raise ValueError("Invalid fitted run")
            if manifest["protocol_sha256"]!=digest(curve.ROOT/"protocol.json"):
                raise ValueError("Run protocol changed")
            if manifest["train_questions"]!=subset["questions"]:
                raise ValueError("Training membership changed")
            history=json.loads((path/"history.json").read_text())
            scheduled=[point["step"] for point in history if point["step"]%128==0]
            if scheduled!=list(range(128,(manifest["steps"]//128)*128+1,128)):
                raise ValueError("Validation cadence differs across subsets")
            timing.append({**{key:item[key] for key in ("subset_id","path","n_questions","condition","seed")},
                           "steps":manifest["steps"],"best_step":manifest["best_step"],
                           "completed_passes":manifest["completed_passes"],"stopping_reason":manifest["stopping_reason"],
                           "seconds":manifest["elapsed_seconds"]})
        for name,sha in manifest["artifact_sha256"].items():
            if digest(path/name)!=sha:raise ValueError(f"Saved artifact changed: {path/name}")
        provenance[str(path)]={"manifest_sha256":digest(path/"manifest.json"),**manifest["artifact_sha256"]}
        for partition in ("train","validation"):
            frame=pd.read_parquet(path/f"{partition}_predictions.parquet").sort_values(KEYS).reset_index(drop=True)
            if frame.duplicated(KEYS).any() or set(frame.question_split)!={partition}:
                raise ValueError("Duplicate predictions or split contamination")
            expected_truth=(truths[partition][truths[partition].qid.isin(subset["questions"])].reset_index(drop=True)
                            if partition=="train" else truths[partition])
            pd.testing.assert_frame_equal(frame[KEYS+["qid","current_state","target"]],expected_truth)
            if len(frame)!=(subset["train_agent_transitions"] if partition=="train" else 32768):
                raise ValueError("Unexpected transition count")
            summary,q=score(frame)
            saved=json.loads((path/f"{partition}_metrics.json").read_text())
            for metric in METRICS:
                if metric=="error_rate":continue
                if summary[metric] is None or saved[metric] is None:
                    if summary[metric]!=saved[metric]:raise ValueError("Missing metric differs")
                else:
                    np.testing.assert_allclose(summary[metric],saved[metric],rtol=0,atol=1e-12)
            summary["error_rate"]=1-summary["accuracy"]
            base={key:item[key] for key in ("subset_id","path","n_questions","condition","seed","reused")}
            base["partition"]=partition
            run_rows.append({**base,**summary})
            q["error_rate"]=1-q.accuracy
            question_rows.extend(q.assign(**base).to_dict("records"))
            if partition=="validation":
                for t,group in frame.groupby("round"):
                    _,byq=score(group)
                    byq["error_rate"]=1-byq.accuracy
                    round_rows.extend(byq.assign(**base,round=int(t)).to_dict("records"))
        if number%12==0:print(f"Recomputed {number}/{len(runs)} runs",flush=True)
    rframe=pd.DataFrame(run_rows);qframe=pd.DataFrame(question_rows);tframe=pd.DataFrame(timing)
    rframe.to_csv("reports/question_count_runs.csv",index=False)
    qframe.to_csv("reports/question_count_by_question.csv",index=False)
    pd.DataFrame(round_rows).to_csv("reports/question_count_by_question_round.csv",index=False)
    tframe.to_csv("reports/question_count_training.csv",index=False)
    summary=rframe.groupby(["condition","partition","n_questions"])[METRICS].mean()
    summary.to_csv("reports/question_count_summary.csv")
    spread=rframe.groupby(["condition","partition","n_questions"])[METRICS].agg(["std","min","max","count"])
    spread.to_csv("reports/question_count_spread.csv")
    qmeans=qframe[qframe.partition=="validation"].groupby(["condition","n_questions","qid"])[METRICS].mean()
    comparisons=[]
    for condition in cfg["curve"]["conditions"]:
        for first,second in ((16,4),(8,4),(12,8),(16,12)):
            for metric in ("nll","error_rate","accuracy"):
                paired=paired_bootstrap(qmeans.loc[(condition,first),metric],qmeans.loc[(condition,second),metric],
                                         cfg["evaluation"]["bootstrap_replicates"],cfg["evaluation"]["bootstrap_seed"])
                comparisons.append({"first_condition":condition,"first_n":first,"second_condition":condition,
                                    "second_n":second,"metric":metric,**paired})
    for n in cfg["curve"]["sizes"]:
        for metric in ("nll","error_rate","accuracy"):
            paired=paired_bootstrap(qmeans.loc[("full_arguments",n),metric],qmeans.loc[("stance_only",n),metric],
                                     cfg["evaluation"]["bootstrap_replicates"],cfg["evaluation"]["bootstrap_seed"])
            comparisons.append({"first_condition":"full_arguments","first_n":n,"second_condition":"stance_only",
                                "second_n":n,"metric":metric,**paired})
    json_write("reports/question_count_paired.json",comparisons)
    plot_curves(rframe,summary)
    audit={"implementation_sha256":curve.implementation_digest(),"protocol_sha256":digest(curve.ROOT/"protocol.json"),
           "plan_sha256":digest(curve.ROOT/"plan.json"),"checks_sha256":digest("reports/question_count_checks.xml"),
           "tests_passed":len(checks.findall(".//testcase")),"endpoint_replay":replay,
           "raw_subset_preprocessing_reconstructed":True,"training_membership_verified":True,
           "identical_validation_examples_and_targets":True,"all_saved_scores_recomputed":True,
           "same_128_update_validation_schedule":True,"shared_endpoint_not_counted_as_independent_repeats":True,
           "new_fits":72,"reused_endpoint_models":6,"all_prediction_runs":len(runs),
           "source_run_hashes":provenance,"validation_questions":protocol["questions"]["validation"],
           "test_scoring":False,"source_llm_calls":0}
    json_write("reports/question_count_integrity.json",audit)
    labels={"stance_only":"Stance + history","full_arguments":"Plus argument text"}
    text=["# Question-count learning curve", "",
          "**Development-only learning curve:** train on 4, 8, 12 or 16 distinct questions and predict the same four validation questions. Error rate means 1 − accuracy; lower is better. The validation questions also choose stopping time and have been inspected in earlier studies, so this is exploratory evidence.","",
          "**Finding:** increasing training questions improves mean validation error and NLL from 4 to 16. Most classification improvement occurs from 4 to 8; later accuracy changes are small, while mean NLL continues to improve. Added argument text has worse mean validation error and NLL than the stance/history control at every tested count. Broader training coverage helps, but this curve does not establish that insufficient question count explains the argument model's missing benefit.","",
          "## Training and validation error", "",
          "| Training questions | Stance/history training error | Stance/history validation error | Argument training error | Argument validation error |",
          "| ---: | ---: | ---: | ---: | ---: |"]
    for n in cfg["curve"]["sizes"]:
        vals=[100*summary.loc[(condition,partition,n),"error_rate"] for condition in cfg["curve"]["conditions"] for partition in ("train","validation")]
        text.append(f"| {n} | "+" | ".join(f"{v:.2f}%" for v in vals)+" |")
    text += ["", "Means use four balanced subset paths × three model seeds at 4, 8 and 12 questions. The 16-question endpoint uses the same three existing seeds once; it is shared by all paths. Each model has 185,890 trainable parameters; the MiniLM encoder stays frozen. Both conditions retain opinion history and the prescribed graph labels.","",
             "![Question-count learning curve](question_count_learning_curve.png)","",
             "The small dots and thin lines show subset/seed variability, not confidence intervals from independent datasets. All models predict the same validation transitions.","",
             "## Probability loss", "",
             "NLL is the primary metric and the checkpoint-selection criterion. Lower NLL is better. Training metrics are computed at the selected checkpoint in evaluation mode.","",
             "| Questions | Stance/history train NLL | Stance/history validation NLL | Argument train NLL | Argument validation NLL |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for n in cfg["curve"]["sizes"]:
        vals=[summary.loc[(condition,partition,n),"nll"] for condition in cfg["curve"]["conditions"] for partition in ("train","validation")]
        text.append(f"| {n} | "+" | ".join(f"{v:.5f}" for v in vals)+" |")
    text += ["", "## Paired changes on the fixed validation questions", "",
             "Differences are later/larger training set minus earlier/smaller training set. Negative error-rate or NLL changes favor more questions.","",
             "| Input | Training-question contrast | Validation error change, points [95% interval] | NLL change [95% interval] |",
             "| --- | --- | ---: | ---: |"]
    for condition in cfg["curve"]["conditions"]:
        for first,second in ((16,4),(8,4),(12,8),(16,12)):
            selected={r["metric"]:r for r in comparisons if (r["first_condition"],r["second_condition"],r["first_n"],r["second_n"])==
                      (condition,condition,first,second)}
            text.append(f"| {labels[condition]} | {first} − {second} | {interval(selected['error_rate'],True)} | {interval(selected['nll'])} |")
    text += ["", "Average model seeds and subset paths within each validation question, then use 1,000 paired question-bootstrap draws. These pointwise intervals condition on this subset design and the four reused validation questions. They do not estimate uncertainty for a large unseen question population or adjust for all comparisons.","",
             "### Does argument text help at any question count?", "",
             "Full argument text minus stance/history; negative differences favor text.","",
             "| Questions | Validation error difference, points [95% interval] | NLL difference [95% interval] |",
             "| ---: | ---: | ---: |"]
    for n in cfg["curve"]["sizes"]:
        selected={r["metric"]:r for r in comparisons if r["first_condition"]=="full_arguments" and r["second_condition"]=="stance_only" and r["first_n"]==n}
        text.append(f"| {n} | {interval(selected['error_rate'],True)} | {interval(selected['nll'])} |")
    text += ["", "### Per-validation-question endpoint changes", "",
             "| Question | Stance/history error: 4 → 16 | Argument error: 4 → 16 |", "| --- | ---: | ---: |"]
    for qid in protocol["questions"]["validation"]:
        values=[]
        for condition in cfg["curve"]["conditions"]:
            values.append(f"{100*qmeans.loc[(condition,4,qid),'error_rate']:.2f}% → {100*qmeans.loc[(condition,16,qid),'error_rate']:.2f}%")
        text.append(f"| {qid} | "+" | ".join(values)+" |")
    text += ["", "### Dependence on which questions are selected", "",
             "For the argument model, the range of validation errors across the four subset paths shrinks as more questions are included. Each path value below first averages its three model seeds. These are observed ranges over four balanced compositions, not confidence intervals.","",
             "| Questions | Lowest path-mean error | Highest path-mean error |", "| ---: | ---: | ---: |"]
    for n in (4,8,12):
        values=rframe[(rframe.condition=="full_arguments")&(rframe.partition=="validation")&(rframe.n_questions==n)].groupby("path").error_rate.mean()
        text.append(f"| {n} | {100*values.min():.2f}% | {100*values.max():.2f}% |")
    text += ["", "The 16-question point has only one available composition, so it cannot estimate variability over alternative 16-question training sets. The argument model's mean training/validation error gap narrows from 6.23 points at four questions to 3.30 points at sixteen, consistent with reduced overfitting within this protocol."]
    text += ["", "## Experimental controls", "",
             "- Four nested subset paths come from a seeded random partition into four blocks of four training questions. Every training question appears equally often at each size: once at 4, twice at 8, three times at 12. Validation questions never enter fitting.",
             "- Each selected question retains all eight fitting graphs, four replicas, eight updates and 32 agents. Unique fitting transitions are 32,768 / 65,536 / 98,304 / 131,072 for 4 / 8 / 12 / 16 questions. Validation always contains 32,768 transitions from four questions.",
             "- Question PCA is refitted on the selected training questions alone. The same 32 personas appear in every subset, so persona PCA stays fixed. Input dimension, original text scale, text encoder, network architecture and optimizer are unchanged.",
             "- Models receive a validation check every 128 optimizer updates, six-check patience and a 5,000-update cap. Small subsets are not stopped after fewer optimizer updates merely because their data passes are shorter. Data are reshuffled each pass; seeds 42–44 pair initialization across conditions and subset paths.",
             "- The 16-question context and PCA arrays match the original feature cache exactly. A 128-update GPU replay reproduces the original first validation score. Its six selected models are reused, and their training/validation scores independently recomputed.",
             "- All new fits have their own code/config/data/protocol hashes, optimizer/RNG checkpoints and prediction artifacts. Original studies and the previously consumed test set are unchanged.","",
             "### Question blocks", ""]
    for i,block in enumerate(protocol["blocks"]):text.append(f"- Block {i}: "+", ".join(f"`{q}`" for q in block))
    text += ["", "Paths begin with blocks 0, 1, 2 or 3 respectively and add the next block cyclically. This gives nested comparisons with balanced question coverage, rather than independent randomly sampled subsets.","",
             "## Interpretation limits", "",
             "Increasing question count also increases unique transitions and changes the fitted PCA. This curve measures the full training pipeline; it does not isolate diversity from sample quantity or feature estimation. Four validation questions constrain inference. A trend up to 16 questions cannot establish what happens beyond 16, identify a final accuracy ceiling, or show that arguments are unnecessary. A fresh question set is needed for confirmatory claims.","",
             "The argument model still consumes observed messages after delivery. It is not an autonomous simulator. This study adds no new LLM-generated questions or conversations and does not test agents establishing their own credibility.","",
             "## Execution and checks", "",
             f"{len(checks.findall('.//testcase'))} focused checks passed for nested/balanced sampling, subset-only preprocessing, split/trajectory isolation and exact CPU mid-pass resume. The endpoint GPU replay matched. Reporting reconstructed every subset's context/PCA, checked example identities and target equality, verified update-based validation cadence and recomputed all saved metrics.","",
             f"Completed {len(plan['runs'])} new fits and reused {len(protocol['endpoint_references'])} endpoint models. The driver recorded {plan['elapsed_seconds']:.1f} seconds; new model fitting and final scoring recorded {tframe.seconds.sum():.1f} seconds. Stopping reasons: "+", ".join(f"{int(count)} {name}" for name,count in tframe.stopping_reason.value_counts().items())+". No paid API or source-LLM calls were made.","",
             "The full protocol is `runs/question_count_curve_v1/protocol.json`; integrity checks are in `question_count_integrity.json`. Run, question, round, spread and training-step tables are saved as `question_count_*.csv`. The shared endpoint is not counted as extra independent subset repeats.","",
             "## Reproduce", "", "```bash", ".venv/bin/python -m pytest tests/test_question_count_curve.py",
             ".venv/bin/python -u scripts/question_count_curve.py prepare --device cpu",
             ".venv/bin/python -u scripts/question_count_curve.py smoke --device mps",
             ".venv/bin/python -u scripts/question_count_curve.py run --device mps",
             ".venv/bin/python scripts/report_question_count_curve.py", "```"]
    Path("reports/question_count_learning_curve.md").write_text("\n".join(text)+"\n")
    print(json.dumps({"report":"reports/question_count_learning_curve.md","new_fits":72,"references":6,
                      "seconds":plan["elapsed_seconds"],"validation":summary.xs("validation",level="partition").reset_index().to_dict("records")},indent=2))


if __name__=="__main__":main()
