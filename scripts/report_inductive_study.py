"""Audit and summarize the training-only PCA study, separately by evaluation role."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR",str(Path(".cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import inductive_study as study
from mac_agents.common import config, digest, json_write
from mac_agents.data.prepare import load_prepared
from mac_agents.metrics import score, paired_bootstrap
from mac_agents.evaluate import check_selection

REPORTS=Path("reports")
METRICS=["nll","brier","accuracy","fcba","flip_accuracy","stay_accuracy"]
ROLES={"validation/fit_graph/random":"Validation questions / familiar random graphs",
       "test/fit_graph/random":"New questions / familiar random graphs",
       "test/heldout_random/random":"New questions / fresh random graphs",
       "test/heldout_family/square":"New questions / square graph",
       "test/heldout_family/triangular":"New questions / triangular graph",
       "test/heldout_family/lowrank":"New questions / low-rank graphs",
       "train/heldout_family/square":"Known training questions / square graph",
       "train/heldout_family/triangular":"Known training questions / triangular graph"}
CONTRASTS=[(("A2","opinion"),("I0","opinion")),
           (("A2","opinion"),("A0","opinion")),
           (("A2","opinion"),("A1","opinion")),
           (("A2","history"),("A2","opinion")),
           (("A0","history"),("A0","opinion")),
           (("A2","history"),("A2","availability_only")),
           (("A0","history"),("A0","availability_only")),
           (("A2","availability_only"),("A2","opinion")),
           (("A0","availability_only"),("A0","opinion"))]


def role(row):
    return "/".join(row[k] for k in ("question_split","graph_role","graph_family"))


def collect(partition,plan):
    _,index,_=load_prepared(config(study.CONFIG))
    lookup={x["episode_id"]:x for x in index}
    pair_roles={}
    for x in index:
        key=(x["qid"],x["graph_hash"])
        if key in pair_roles and pair_roles[key]!=role(x):
            raise ValueError("Ambiguous graph evaluation role")
        pair_roles[key]=role(x)
    rows,questions,rollrows,rollquestions=[],[],[],[]
    reference,reference_trajectory=None,None
    expected_n=32768 if partition=="validation" else 268288
    for item in plan["runs"]:
        if item["status"]!="complete" or not item["validation_rollout"]:
            raise ValueError("Incomplete validation cohort")
        if partition=="test" and not (item["test_one_step"] and item["test_closed_loop"]):
            raise ValueError("Complete all frozen test evaluations before reporting")
        run=Path(item["run"])
        meta=study.validate_run(item)
        if (meta["kind"],meta["seed"])!=(item["model"],item["seed"]):
            raise ValueError("Manifest model/seed differs from the declared run")
        if partition=="test":
            check_selection(run,study.ROOT/"selection.json")
        elif meta["kind"].startswith("A") and digest(run/"best.pt")!=meta["best_sha256"]:
            raise ValueError("Validation checkpoint changed")
        prefix="validation" if partition=="validation" else "test_one_step"
        frame=pd.read_parquet(run/f"{prefix}_predictions.parquet")
        if len(frame)!=expected_n or frame.duplicated(["episode_id","round","agent"]).any():
            raise ValueError("Missing or duplicate prediction keys")
        observed=frame.set_index(["episode_id","round","agent"])[["qid","target","current_state"]].sort_index()
        if reference is None:
            reference=observed
        else:
            pd.testing.assert_frame_equal(reference,observed)
        rescored,_=score(frame,item["model"])
        stored=json.loads((run/f"{prefix}_metrics.json").read_text())
        for metric in METRICS:
            if stored[metric] is not None:
                np.testing.assert_allclose(rescored[metric],stored[metric],rtol=0,atol=1e-12)
        identity={"model":item["model"],"condition":item["condition"],"seed":item["seed"],"run":str(run)}
        for keys,group in frame.groupby(["question_split","graph_role","graph_family"],sort=True):
            r="/".join(keys)
            s,q=score(group,item["model"])
            rows.append(dict(identity,role=r,**{k:s[k] for k in METRICS},questions=s["questions"],
                             transitions=s["agent_transitions"],parameters=meta["parameters"],
                             fit_seconds=meta["elapsed_seconds_this_session"],epochs=meta.get("epochs"),
                             stopping_reason=meta["stopping_reason"]))
            questions.append(q.assign(**identity,role=r))
        rollout_prefix=f"{partition}_closed_loop"
        ev=json.loads((run/f"{rollout_prefix}_metrics.json").read_text())
        if ev["rollouts_per_initial_state"]!=32 or ev["preparation_run"]:
            raise ValueError("Rollout protocol mismatch")
        conditions=pd.read_csv(run/f"{rollout_prefix}_conditions.csv")
        conditions["role"]=[pair_roles[(q,g)] for q,g in zip(conditions.qid,conditions.graph_hash)]
        with np.load(run/f"{rollout_prefix}_trajectories.npz",allow_pickle=False) as z:
            ids=z["episode_ids"]
            truth,initial,deterministic,sampled=z["target"],z["initial"],z["deterministic"],z["sampled"]
        if reference_trajectory is None:
            reference_trajectory=(ids.copy(),truth.copy(),initial.copy())
        else:
            for actual,ref in zip((ids,truth,initial),reference_trajectory):
                np.testing.assert_array_equal(actual,ref)
        np.testing.assert_allclose((deterministic==truth).mean(axis=(0,2)),ev["horizon_accuracy"],rtol=0,atol=1e-12)
        episode_meta=pd.DataFrame([dict(qid=lookup[e]["qid"],role=role(lookup[e]),position=i) for i,e in enumerate(ids)])
        for (r,qid),group in episode_meta.groupby(["role","qid"]):
            positions=group.position.to_numpy()
            target=truth[positions]
            pred=deterministic[positions]
            full_true=np.concatenate([initial[positions,None],target],axis=1)
            full_sampled=np.concatenate([np.broadcast_to(initial[positions],(32,len(positions),32))[:,:,None],sampled[:,positions]],axis=2)
            h_accuracy=(pred==target).mean(axis=(0,2))
            w=conditions[(conditions.role==r)&(conditions.qid==qid)].final_fraction_wasserstein.mean()
            rollquestions.append(dict(identity,role=r,qid=qid,episodes=len(positions),
                final_accuracy=float(h_accuracy[-1]),final_wasserstein=float(w),
                observed_switches=float((full_true[:,1:]!=full_true[:,:-1]).sum(axis=1).mean()),
                sampled_switches=float((full_sampled[:,:,1:]!=full_sampled[:,:,:-1]).sum(axis=2).mean()),
                observed_consensus=float((np.abs(full_true.mean(-1))==1).any(axis=1).mean()),
                sampled_consensus=float((np.abs(full_sampled.mean(-1))==1).any(axis=2).mean()),
                **{f"accuracy_h{h+1}":float(value) for h,value in enumerate(h_accuracy)}))
        rollrows.append(dict(identity,seconds=ev["elapsed_seconds"],episodes=len(ids),device=ev["actual_device"]))
    frame=pd.DataFrame(rows)
    qframe=pd.concat(questions,ignore_index=True)
    rq=pd.DataFrame(rollquestions)
    timing=pd.DataFrame(rollrows)
    for name,data in (("runs",frame),("by_question",qframe),("rollouts_by_question",rq),("rollout_timing",timing)):
        data.to_csv(REPORTS/f"inductive_{partition}_{name}.csv",index=False)
    comparisons=[]
    for r in sorted(frame.role.unique()):
        for first,second in CONTRASTS:
            a=qframe[(qframe.role==r)&(qframe.model==first[0])&(qframe.condition==first[1])].groupby("qid")[METRICS].mean()
            b=qframe[(qframe.role==r)&(qframe.model==second[0])&(qframe.condition==second[1])].groupby("qid")[METRICS].mean()
            ra=rq[(rq.role==r)&(rq.model==first[0])&(rq.condition==first[1])].groupby("qid")[["final_accuracy","final_wasserstein"]].mean()
            rb=rq[(rq.role==r)&(rq.model==second[0])&(rq.condition==second[1])].groupby("qid")[["final_accuracy","final_wasserstein"]].mean()
            comparisons.append({"role":r,"first":list(first),"reference":list(second),
                "metrics":{m:paired_bootstrap(a[m],b[m]) for m in METRICS},
                "rollout_metrics":{m:paired_bootstrap(ra[m],rb[m]) for m in ("final_accuracy","final_wasserstein")}})
    json_write(REPORTS/f"inductive_{partition}_paired.json",comparisons)
    audit={"partition":partition,"implementation":study.implementation(),"protocol_sha256":plan["protocol_sha256"],
           "runs":len(plan["runs"]),"prediction_rows_per_run":len(reference),
           "identical_prediction_keys_and_targets":True,"all_saved_scores_recomputed":True,
           "identical_rollout_initializations_targets_and_order":True,"sampled_rollouts_per_initialization":32,
           "fit_seconds":float(frame.drop_duplicates('run').fit_seconds.sum()),
           "rollout_seconds":float(timing.seconds.sum()),"study_elapsed_seconds":plan["elapsed_seconds"]}
    if partition=="test":
        audit["selection_sha256"]=digest(study.ROOT/"selection.json")
        audit["frozen_protocol_sha256"]=digest(study.ROOT/"frozen_protocol.json")
        audit["all_frozen_manifests_and_checkpoints_verified"]=True
    json_write(REPORTS/f"inductive_{partition}_integrity.json",audit)
    return frame,qframe,rq,comparisons,audit


def mean_sd(values,scale=1,precision=4):
    values=values.dropna()
    if not len(values):
        return "—"
    out=f"{scale*values.mean():.{precision}f}"
    if len(values)>1:
        out+=f" ± {scale*values.std():.{precision}f}"
    return out


def author_comparison(frame):
    old=pd.concat([pd.read_csv(REPORTS/"validation_runs.csv"),pd.read_csv(REPORTS/"history_control_runs.csv")],ignore_index=True)
    old=old.rename(columns={"state":"condition"})
    old=old.groupby(["model","condition"])[["accuracy","nll"]].mean()
    new=frame.groupby(["model","condition"])[["accuracy","nll"]].mean()
    joined=new.join(old,lsuffix="_inductive",rsuffix="_author",how="inner")
    joined.to_csv(REPORTS/"inductive_preprocessing_comparison.csv")
    lines=["## Matched preprocessing comparison", "",
           "Same examples, architecture, and seeds. The new context uses raw question-plus-options embeddings projected with PCA fitted only on training questions. The authors' projection provenance is unverified, so differences cannot be attributed solely to removal of leakage; the feature representation itself changes.", "",
           "| Model / state | Author-context NLL | Training-only NLL | Author accuracy % | Training-only accuracy % |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for (model,condition),row in joined.iterrows():
        if model in ("P0","P1"):
            continue
        lines.append(f"| {model} / {condition} | {row.nll_author:.4f} | {row.nll_inductive:.4f} | {100*row.accuracy_author:.2f} | {100*row.accuracy_inductive:.2f} |")
    return lines


def plots(partition,frame,rq,comparisons):
    primary="validation/fit_graph/random" if partition=="validation" else "test/fit_graph/random"
    data=[x for x in comparisons if x["role"]==primary and (tuple(x["first"]),tuple(x["reference"])) in CONTRASTS[:7]]
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),layout="constrained")
    labels=[f"{' / '.join(x['first'])}\nvs {' / '.join(x['reference'])}" for x in data]
    for ax,metric,scale,title in zip(axes,["nll","accuracy"],[1,100],["NLL difference · lower favors first","Accuracy difference (percentage points) · higher favors first"]):
        for i,item in enumerate(data):
            value=item["metrics"][metric]
            low,high=np.array(value["ci95"])*scale
            mean=value["difference"]*scale
            ax.plot([low,high],[i,i],color="#41688c",lw=2)
            ax.scatter([mean],[i],color="#41688c",s=25,zorder=3)
        ax.axvline(0,color="#666666",ls="--",lw=1)
        ax.set_yticks(range(len(labels)),labels if ax is axes[0] else [""]*len(labels))
        ax.invert_yaxis()
        ax.set_title(title,fontsize=10)
        ax.grid(axis="x",alpha=.15)
    fig.suptitle(f"Training-only PCA · {ROLES[primary]}\nPointwise 95% paired question-bootstrap intervals; seeds averaged within question",fontsize=11)
    fig.savefig(REPORTS/f"inductive_{partition}_paired.png",dpi=180)
    plt.close(fig)
    selected_roles=[primary] if partition=="validation" else [primary,"test/heldout_random/random"]
    fig,axes=plt.subplots(1,len(selected_roles),figsize=(7*len(selected_roles),4.4),squeeze=False,layout="constrained")
    choices=[("I0","opinion","#444444"),("A0","opinion","#497999"),("A2","opinion","#c3683d"),
             ("A2","history","#a68b29"),("A2","availability_only","#489485")]
    for ax,r in zip(axes[0],selected_roles):
        for model,condition,color in choices:
            per_seed=rq[(rq.role==r)&(rq.model==model)&(rq.condition==condition)].groupby("seed")[[f"accuracy_h{h}" for h in range(1,9)]].mean().to_numpy()*100
            avg=per_seed.mean(0)
            ax.plot(range(1,9),avg,color=color,marker="o",ms=3,label=f"{model} / {condition}")
            if len(per_seed)>1:
                sd=per_seed.std(0,ddof=1)
                ax.fill_between(range(1,9),avg-sd,avg+sd,color=color,alpha=.1)
        ax.set(title=ROLES[r],xlabel="Autonomous prediction round",ylabel="Accuracy (%)",xticks=range(1,9))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Generated states only after initialization · bands show training-seed SD",fontsize=11)
    fig.savefig(REPORTS/f"inductive_{partition}_rollouts.png",dpi=180)
    plt.close(fig)


def write_report(partition,frame,rq,comparisons,audit):
    primary="validation/fit_graph/random" if partition=="validation" else "test/fit_graph/random"
    primary_contrast=next(x for x in comparisons if x["role"]==primary and x["first"]==["A2","opinion"] and x["reference"]==["I0","opinion"])
    a=primary_contrast["metrics"]["accuracy"]
    n=primary_contrast["metrics"]["nll"]
    decision="The interval includes zero; an accuracy advantage is not established." if a["ci95"][0]<=0<=a["ci95"][1] else (
        "The interval is above zero for this predeclared comparison; the magnitude and other evaluation roles still matter." if a["ci95"][0]>0 else
        "The interval is below zero: the transformer has lower accuracy in this comparison.")
    title="Training-only PCA validation" if partition=="validation" else "Frozen test evaluation with training-only PCA"
    lines=[f"# {title}","",
        f"A2/opinion minus I0 on **{ROLES[primary].lower()}**: accuracy difference {100*a['difference']:+.2f} percentage points, paired 95% interval [{100*a['ci95'][0]:+.2f}, {100*a['ci95'][1]:+.2f}]. {decision}","",
        f"The NLL difference is {n['difference']:+.5f}, interval [{n['ci95'][0]:+.5f}, {n['ci95'][1]:+.5f}] (negative favors A2). NLL was the predeclared primary probability metric; accuracy is reported alongside it. Intervals average seeds within each question before resampling questions. They are pointwise, not adjusted for the multiple secondary comparisons.","",
        "The declared cohort contains 24 fits: three baselines and seven neural configurations at seeds 42/43/44. Neural configurations are A0/A1/A2 with opinions, A0/A2 with history, and A0/A2 with the matched availability-only control. Every candidate was retained; no best-seed selection was performed.","",
        "PCA is fitted on raw frozen question-plus-options embeddings from the 16 training questions only. Its mean/components and every context vector were reconstructed independently. Perturbing held-out embeddings did not change fitted PCA. All other arrays, example identities, targets, and graphs match the earlier author-context study. All 32 personas appear in training. Architectures, hyperparameters, stopping rules, graph signs, and forecast boundaries are unchanged.","",
        "The availability-only control uses the same six-feature history architecture, zeroing past-opinion values while retaining availability and current-opinion features. The transform is applied consistently in fitting, one-step evaluation, and generated-history rollout. Study-specific implementation hashes prevent ordinary CLI evaluation from silently omitting it.",""]
    if partition=="validation":
        lines += ["Only four validation questions are available and were also used for early stopping. These comparisons are developmental. The test set is not used in this report.",""]
        lines += author_comparison(frame)+[""]
    else:
        frozen=json.loads((study.ROOT/"selection.json").read_text())
        lines += [f"Selection was frozen at `{frozen['frozen_utc']}` before any controlled test scoring. All checkpoints, manifests, preprocessing files, and the study implementation were verified against the freeze record. No retraining, preprocessing changes, or reselection followed test evaluation.","",
            "New-question tests include 20 questions on familiar random graphs, fresh random graphs, and square/triangular graphs; low-rank graphs have ten questions. Known-training-question tests on square/triangular graphs have 16 questions and are reported separately. The primary role was fixed as new questions on familiar random graphs.",""]
        def comparison(r,first,second):
            return next(x for x in comparisons if x["role"]==r and x["first"]==list(first) and x["reference"]==list(second))
        fresh="test/heldout_random/random"
        baseline_fresh=comparison(fresh,("A2","opinion"),("I0","opinion"))["metrics"]["accuracy"]
        control_seen=comparison(primary,("A2","opinion"),("A0","opinion"))["metrics"]["accuracy"]
        control_fresh=comparison(fresh,("A2","opinion"),("A0","opinion"))["metrics"]["accuracy"]
        hist_seen=comparison(primary,("A2","history"),("A2","availability_only"))["metrics"]["accuracy"]
        hist_fresh=comparison(fresh,("A2","history"),("A2","availability_only"))["metrics"]["accuracy"]
        roll_seen=comparison(primary,("A2","opinion"),("I0","opinion"))["rollout_metrics"]["final_accuracy"]
        roll_fresh=comparison(fresh,("A2","opinion"),("I0","opinion"))["rollout_metrics"]["final_accuracy"]
        def formatted(value):
            return f"{100*value['difference']:+.2f} pp [{100*value['ci95'][0]:+.2f}, {100*value['ci95'][1]:+.2f}]"
        summary=["## Findings for the research question","",
            f"The held-out study supports a small one-step advantage over the Ising baseline on random graphs: {formatted(a)} on familiar graphs and {formatted(baseline_fresh)} on fresh graphs. Both NLL and accuracy improve in these comparisons. This is more evidence than the earlier four-question validation screen, while the practical accuracy gain remains modest.","",
            f"A transformer-specific accuracy advantage is unresolved: A2 minus the simpler A0 control is {formatted(control_seen)} on familiar graphs and {formatted(control_fresh)} on fresh graphs. The pointwise NLL advantage over A0 is clearer on familiar graphs and unresolved on fresh graphs. The fixed-aggregation/history model has mean accuracy 89.35% and 88.47% on those two roles, compared with 89.20% and 87.89% for attention/history; these descriptive means do not establish a universal winner.","",
            f"Actual past opinion values provide a small additional one-step benefit over the matched initialization-only control: A2/history minus A2/availability-only is {formatted(hist_seen)} on familiar graphs and {formatted(hist_fresh)} on fresh graphs, with lower NLL in both roles. This separates part of the history benefit from the initialization cue. It does not establish that two snapshots form a sufficient state, and the accuracy gain does not extend uniformly to the structured graph families.","",
            f"There is no demonstrated general improvement in autonomous simulation. A2/opinion minus I0 at round eight is {formatted(roll_seen)} on familiar graphs and {formatted(roll_fresh)} on fresh graphs. Final-fraction distribution distances are higher on average for A2 in both roles, with paired intervals spanning zero. History's one-step improvements do not yield consistent rollout gains. The next scientific target is explaining the trajectory/distribution mismatch using development data, rather than further tuning against these now-observed test outcomes.",""]
        lines=lines[:6]+summary+lines[6:]
    roles=[primary]+[r for r in ROLES if r!=primary and r in set(frame.role)]
    for r in roles:
        local=frame[frame.role==r]
        count=local.iloc[0]
        lines += [f"## {ROLES[r]}","",f"{int(count.questions)} questions; {int(count.transitions):,} agent transitions per run. Means ± sample SD across neural training seeds. Accuracy means matching agent opinion, not task correctness.","",
            "| Model | State | Parameters | NLL ↓ | Brier ↓ | Accuracy % | FCBA % |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for (model,condition),sub in local.groupby(["model","condition"],sort=False):
            lines.append(f"| {model} | {condition} | {int(sub.parameters.iloc[0]):,} | {mean_sd(sub.nll)} | {mean_sd(sub.brier)} | {mean_sd(sub.accuracy,100,2)} | {mean_sd(sub.fcba,100,2)} |")
        lines += ["", "Autonomous simulation from the observed initialization: deterministic trajectories and 32 samples per initial state, eight rounds. Each model uses generated states after initialization.","",
            "| Model | State | Round-8 accuracy % | Final-fraction Wasserstein ↓ | Sampled switches / agent | Observed switches / agent |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
        for (model,condition),sub in rq[rq.role==r].groupby(["model","condition"],sort=False):
            per_seed=sub.groupby("seed").mean(numeric_only=True)
            lines.append(f"| {model} | {condition} | {mean_sd(per_seed.final_accuracy,100,2)} | {mean_sd(per_seed.final_wasserstein)} | {per_seed.sampled_switches.mean():.3f} | {per_seed.observed_switches.mean():.3f} |")
        lines += ["","Paired differences: first model minus reference. Positive accuracy and negative NLL favor the first model.","",
            "| First | Reference | Δ accuracy pp [95% interval] | Δ NLL [95% interval] |",
            "| --- | --- | --- | --- |"]
        for x in comparisons:
            if x["role"]!=r:
                continue
            ac,nl=x["metrics"]["accuracy"],x["metrics"]["nll"]
            lines.append(f"| {' / '.join(x['first'])} | {' / '.join(x['reference'])} | {100*ac['difference']:+.2f} [{100*ac['ci95'][0]:+.2f}, {100*ac['ci95'][1]:+.2f}] | {nl['difference']:+.5f} [{nl['ci95'][0]:+.5f}, {nl['ci95'][1]:+.5f}] |")
        lines += [""]
    lines += ["## Plots", "", f"![Paired uncertainty](inductive_{partition}_paired.png)","",
              f"![Autonomous horizons](inductive_{partition}_rollouts.png)","", "## Interpretation limits and reproducibility", "",
        "Question bootstrap intervals preserve all associated agents, rounds, and episodes. They condition on the tested graph collection and source model; they do not establish generalization across arbitrary graphs, agents, or LLMs. Three training seeds quantify optimization variability, not independent new questions. The low-rank subset contains fewer questions. Each question/graph has few observed replicas, so Monte Carlo samples do not provide additional reference observations.","",
        "Question features still come from a frozen externally produced embedding model; this study establishes training-only fitting of the PCA projection, not absence of any upstream knowledge of the questions. A small predictive model need not recover a causal interaction law. None of these experiments changes the source agents' prescribed relationship graph or tests emergent credibility.","",
        f"Recorded fitting/validation time for the 24 models: {audit['fit_seconds']:.1f} seconds. {partition.capitalize()} autonomous evaluation time: {audit['rollout_seconds']:.1f} seconds. Neural stages use MPS float32; baseline stages use CPU float64. These timings are not LLM speedups. No text encoder or generator is called by these opinion/history models. Peak memory was not instrumented.","",
        f"[Run scores](inductive_{partition}_runs.csv), [question scores](inductive_{partition}_by_question.csv), [rollout question scores](inductive_{partition}_rollouts_by_question.csv), [paired intervals](inductive_{partition}_paired.json), [integrity audit](inductive_{partition}_integrity.json).", "",
        "Protocol, plans, exact configs, source/data/checkpoint hashes, predictions, and rollout trajectories are under `runs/inductive_study/`. The test selection and its linked protocol are immutable.", "",
        f"Regenerate: `.venv/bin/python scripts/report_inductive_study.py --partition {partition}`.",""]
    (REPORTS/f"inductive_{partition}_results.md").write_text("\n".join(lines))


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--partition",choices=["validation","test"],default="validation")
    args=parser.parse_args()
    plan=json.loads((study.ROOT/"plan.json").read_text())
    protocol=json.loads((study.ROOT/"protocol.json").read_text())
    if protocol["implementation"]!=study.implementation() or plan["protocol_sha256"]!=digest(study.ROOT/"protocol.json"):
        raise ValueError("Study implementation or protocol changed")
    if [{k:item[k] for k in ("model","condition","seed")} for item in plan["runs"]]!=protocol["cohort"]:
        raise ValueError("Run cohort differs from the predeclared protocol")
    if args.partition=="test":
        frozen=json.loads((study.ROOT/"frozen_protocol.json").read_text())
        if frozen["selection_sha256"]!=digest(study.ROOT/"selection.json") or frozen["protocol_sha256"]!=plan["protocol_sha256"]:
            raise ValueError("Frozen selection/protocol changed")
    frame,questions,rq,comparisons,audit=collect(args.partition,plan)
    plots(args.partition,frame,rq,comparisons)
    write_report(args.partition,frame,rq,comparisons,audit)
    primary="validation/fit_graph/random" if args.partition=="validation" else "test/fit_graph/random"
    print(frame[frame.role==primary].groupby(["model","condition"])[["nll","accuracy"]].mean().to_string())
    print("Primary contrast:",json.dumps(next(x for x in comparisons if x["role"]==primary),indent=2))
    print(f"Wrote reports/inductive_{args.partition}_results.md; {len(plan['runs'])} runs audited")


if __name__=="__main__":
    main()
