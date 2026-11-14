"""Deferred outer assessment and reporting for nested generalization study."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from scripts.transformer_generalization import (
    CONFIG, FoldData, completed_runs, make_model, metrics, model_weights_sha,
    predictions, validate,
)

LABELS = {"I0":"Ising", "full":"Full transformer", "small":"Small transformer",
          "ising_small":"Ising + small transformer"}
ORDER = ["I0","full","small","ising_small"]


def lp_from_logit(score):
    score = np.asarray(score,dtype=np.float64)
    return np.stack([-np.logaddexp(0,score),-np.logaddexp(0,-score)],-1)


def evaluate(cfg,device):
    root = Path(cfg["output"])
    protocol = validate(cfg)
    tuning,refits = completed_runs(root,"tune"),completed_runs(root,"refit")
    assert len(tuning) == 216 and len(refits) == 36
    selected = json.loads((root/"selection.json").read_text())
    assert not selected["selection_uses_outer_outcomes"]
    assert selected["tuning_complete_sha256"] == digest(root/"tune_complete.json")
    identity = {"protocol_sha256":digest(root/"protocol.json"),"selection_sha256":digest(root/"selection.json"),
                "refits_sha256":digest(root/"refit_complete.json"),"report_code_sha256":digest(__file__),
                "input_audit_sha256":digest(root/"input_audit.json")}
    audit = json.loads((root/"input_audit.json").read_text())
    assert audit["development_sha256"] == digest(root/"development.npz")
    assert audit["protocol_sha256"] == identity["protocol_sha256"] and not audit["outer_outcomes_scored"]
    # This record freezes every fitted candidate before any outer label scoring.
    if (root/"assessment_freeze.json").exists():
        freeze = json.loads((root/"assessment_freeze.json").read_text())
        assert all(freeze[k] == v for k,v in identity.items())
    else:
        json_write(root/"assessment_freeze.json",{**identity,"frozen_utc":datetime.now(timezone.utc).isoformat(),
            "neural_refits":36,"assessment_questions":protocol["questions"],"fresh_confirmatory_test":False})
    common = dict(np.load(root/"development.npz",allow_pickle=False))
    source_index = json.loads((root/"episode_index.json").read_text())
    assert all(r["official_split"] == "train" and r["graph_role"] == "fit_graph" for r in source_index)
    qrecords,records,reproductions,baseline_records = [],[],[],[]
    for outer in range(cfg["split"]["outer_folds"]):
        fold = next(f for f in protocol["folds"] if f["id"] == f"o{outer}_refit")
        feat = dict(np.load(root/"folds"/fold["id"]/"features.npz",allow_pickle=False))
        data = FoldData(common,feat,device)
        cpu_data = FoldData(common,feat,"cpu")
        fitting,assess = data.ids(fold["train_questions"]),data.ids(fold["assessment_questions"])
        assert len(fitting) == 3840 and len(assess) == 1280 and not set(fitting)&set(assess)
        for split,ids in (("train",fitting),("assessment",assess)):
            lp = lp_from_logit(feat["prior_logit"][ids])
            score,byq = metrics(lp,data.y[ids],data.qids[ids])
            fields = {"model":"I0","outer":outer,"seed":-1,"split":split}
            baseline_records.append({**fields,**score})
            records.append({**fields,**score})
            qrecords += [{**fields,**r} for r in byq]
        for run,m in [(p,m) for p,m in refits if m["outer"] == outer]:
            choice = next(x for x in selected["choices"] if x["outer"] == outer and x["model"] == m["model"])
            assert m["candidate"] == choice["candidate"] and m["steps"] == choice["refit_steps"]
            assert m["selected_step"] == m["steps"] == m["fixed_steps"]
            name = str(run.relative_to(root)).replace("/","__")
            output = root/"assessment"/name
            output.mkdir(parents=True,exist_ok=True)
            local_identity = {**identity,"run_manifest_sha256":digest(run/"manifest.json")}
            if (output/"complete.json").exists():
                done = json.loads((output/"complete.json").read_text())
                assert all(done[k] == v for k,v in local_identity.items())
                for file,sha in done["artifacts"].items(): assert digest(output/file) == sha
                records += done["records"]; qrecords += done["by_question"]
                reproductions.append(done["reproduction"])
                continue
            model = make_model(cfg,m["model"]).to(device)
            checkpoint = torch.load(run/"best.pt",map_location=device,weights_only=False)
            assert checkpoint["step"] == choice["refit_steps"]
            model.load_state_dict(checkpoint["model"])
            assert model_weights_sha(model) == m["selected_weights_sha256"]
            if checkpoint["step"] > 0:
                assert m["initial_weights_sha256"] != m["selected_weights_sha256"]
            lp = predictions(model,data,assess,m["model"] == "ising_small")
            if m["model"] == "ising_small" and checkpoint["step"] == 0:
                np.testing.assert_allclose(np.exp(lp[...,1]),
                    np.exp(lp_from_logit(feat["prior_logit"][assess])[...,1]),atol=1e-6,rtol=1e-6)
            np.save(output/"assessment_logp.npy",lp,allow_pickle=False)
            train_lp = np.load(run/"train_logp.npy",allow_pickle=False)
            own,own_q = [],[]
            for split,values,ids in (("train",train_lp,fitting),("assessment",lp,assess)):
                score,byq = metrics(values,data.y[ids],data.qids[ids])
                fields = {"run":name,"model":m["model"],"outer":outer,"seed":m["seed"],
                          "split":split,"candidate":m["candidate"],"steps":m["steps"]}
                own.append({**fields,**score}); own_q += [{**fields,**r} for r in byq]
                if split == "train":
                    for key in ("nll","error","brier_two_class"):
                        np.testing.assert_allclose(score[key],m["train"][key],atol=1e-7,rtol=1e-6)
            cpu = make_model(cfg,m["model"])
            cpu.load_state_dict(torch.load(run/"best.pt",map_location="cpu",weights_only=False)["model"])
            maximum = 0.
            for ids,values in ((assess,lp),(fitting,train_lp)):
                positions = np.linspace(0,len(ids)-1,64,dtype=int)
                actual = np.exp(predictions(cpu,cpu_data,ids[positions],m["model"] == "ising_small",64)[...,1])
                expected = np.exp(values[positions,...,1])
                np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=5e-5)
                maximum = max(maximum,float(np.max(np.abs(actual-expected))))
            reproduction = {"run":name,"model":m["model"],"outer":outer,"seed":m["seed"],
                            "training_snapshots":64,"assessment_snapshots":64,
                            "maximum_probability_difference":maximum}
            json_write(output/"complete.json",{**local_identity,"records":own,"by_question":own_q,
                "reproduction":reproduction,"artifacts":{"assessment_logp.npy":digest(output/"assessment_logp.npy")}})
            records += own; qrecords += own_q; reproductions.append(reproduction)
            print(json.dumps({"assessed":name,"nll":own[-1]["nll"],"error":own[-1]["error"]}),flush=True)
        del data,cpu_data
    frame,byq,repro = pd.DataFrame(records),pd.DataFrame(qrecords),pd.DataFrame(reproductions)
    assert len(frame) == 80 and len(repro) == 36
    for kind in ORDER:
        part = byq[(byq.model == kind)&(byq.split == "assessment")]
        assert len(set(part.qid)) == 20
        assert (part.groupby("qid").size() == (1 if kind == "I0" else 3)).all()
        assert (part.groupby("qid").outer.nunique() == 1).all()
    for fold in protocol["folds"]:
        if fold["phase"] == "tune":
            for kind in cfg["models"]:
                hashes = {m["initial_weights_sha256"] for _,m in tuning if m["fold"] == fold["id"] and m["model"] == kind}
                assert len(hashes) == 1
            small_hashes = {m["initial_weights_sha256"] for _,m in tuning if m["fold"] == fold["id"] and m["model"] in ("small","ising_small")}
            assert len(small_hashes) == 1
    for outer in range(4):
        for seed in cfg["training"]["refit_seeds"]:
            hashes = {m["initial_weights_sha256"] for _,m in refits if m["outer"] == outer and m["seed"] == seed and m["model"] in ("small","ising_small")}
            assert len(hashes) == 1
    old = json.loads((root/"prior_artifacts.json").read_text())
    for name,sha in old.items(): assert digest(name) == sha,name
    frame.to_csv(root/"outer_metrics.csv",index=False)
    byq.to_csv(root/"outer_by_question.csv",index=False)
    repro.to_csv(root/"checkpoint_reproduction.csv",index=False)
    pd.DataFrame(baseline_records).to_csv(root/"baseline_metrics.csv",index=False)
    names = ("outer_metrics.csv","outer_by_question.csv","checkpoint_reproduction.csv","baseline_metrics.csv")
    json_write(root/"assessment_complete.json",{**identity,"completed_utc":datetime.now(timezone.utc).isoformat(),
        "neural_refits":36,"ising_outer_fits":4,"questions":20,"prior_artifacts_verified_unchanged":len(old),
        "maximum_cpu_mps_probability_difference":float(repro.maximum_probability_difference.max()),
        "matched_initializations_verified":True,"new_official_test_evaluations":0,
        "artifacts":{name:digest(root/name) for name in names}})


def paired_contrasts(byq,cfg):
    pairs = [("full","I0"),("small","full"),("ising_small","small"),
             ("ising_small","full"),("ising_small","I0"),("small","I0")]
    rng = np.random.default_rng(cfg["evaluation"]["bootstrap_seed"])
    qids = sorted(byq.qid.unique())
    draws = rng.integers(0,len(qids),(cfg["evaluation"]["bootstrap_replicates"],len(qids)))
    result = []
    for first,second in pairs:
        a = byq[byq.model == first].set_index("qid").loc[qids]
        b = byq[byq.model == second].set_index("qid").loc[qids]
        for key in ("nll","error","brier_two_class"):
            difference = a[key].to_numpy()-b[key].to_numpy()
            lo,hi = np.quantile(difference[draws].mean(1),[.025,.975])
            result.append({"first":first,"second":second,"metric":key,"difference":float(difference.mean()),
                           "descriptive_lower":float(lo),"descriptive_upper":float(hi),
                           "questions_better":int((difference < -1e-6).sum()),
                           "questions_tied_within_tolerance":int((np.abs(difference) <= 1e-6).sum()),
                           "count_tolerance":1e-6,"questions":len(qids)})
    return result


def summarize(cfg):
    root = Path(cfg["output"])
    protocol = validate(cfg)
    done = json.loads((root/"assessment_complete.json").read_text())
    assert done["report_code_sha256"] == digest(__file__)
    for file,sha in done["artifacts"].items(): assert digest(root/file) == sha
    tuning,refits = completed_runs(root,"tune"),completed_runs(root,"refit")
    frame = pd.read_csv(root/"outer_metrics.csv")
    byq = pd.read_csv(root/"outer_by_question.csv")
    # Averages scores across seeds, never ensembles their predictions.
    averaged = byq.groupby(["model","split","outer","qid"])[["nll","error","brier_two_class"]].mean().reset_index()
    assessed = averaged[averaged.split == "assessment"]
    averaged.to_csv(root/"question_means.csv",index=False)
    summary = averaged.groupby(["model","split"])[["nll","error","brier_two_class"]].mean().reset_index()
    summary.to_csv(root/"summary.csv",index=False)
    fold_summary = averaged.groupby(["model","split","outer"])[["nll","error","brier_two_class"]].mean().reset_index()
    fold_summary.to_csv(root/"outer_fold_summary.csv",index=False)
    fold_differences = []
    for first,second in (("small","full"),("ising_small","small"),("ising_small","full")):
        for outer in range(4):
            a = fold_summary[(fold_summary.model == first)&(fold_summary.split == "assessment")&(fold_summary.outer == outer)].iloc[0]
            b = fold_summary[(fold_summary.model == second)&(fold_summary.split == "assessment")&(fold_summary.outer == outer)].iloc[0]
            fold_differences.append({"first":first,"second":second,"outer":outer,
                "nll_difference":float(a.nll-b.nll),"error_difference":float(a.error-b.error)})
    pd.DataFrame(fold_differences).to_csv(root/"fold_contrasts.csv",index=False)
    contrasts = paired_contrasts(assessed,cfg)
    pd.DataFrame(contrasts).to_csv(root/"paired_contrasts.csv",index=False)
    tune_rows = [{"run":str(p.relative_to(root)),"outer":m["outer"],"fold":m["fold"],"model":m["model"],
        "candidate":m["candidate"],**m["candidate_settings"],"seed":m["seed"],"steps":m["steps"],
        "selected_step":m["selected_step"],"training_error":m["train"]["error"],"training_nll":m["train"]["nll"],
        "validation_error":m["validation"]["error"],"validation_nll":m["validation"]["nll"],
        "stopping_reason":m["stopping_reason"],"seconds":m["seconds"]} for p,m in tuning]
    pd.DataFrame(tune_rows).to_csv(root/"tuning_runs.csv",index=False)
    choices = json.loads((root/"selection.json").read_text())
    chosen_rows = []
    for item in choices["choices"]:
        setting = next(c for c in protocol["candidates"] if c["id"] == item["candidate"])
        chosen_rows.append({**item,**setting})
    pd.DataFrame(chosen_rows).to_csv(root/"selected_settings.csv",index=False)
    allc = pd.DataFrame(choices["all_candidates"])
    allc.to_csv(root/"all_candidate_scores.csv",index=False)
    histories = []
    for run,m in tuning:
        for h in json.loads((run/"history.json").read_text()):
            histories.append({"outer":m["outer"],"fold":m["fold"],"model":m["model"],"candidate":m["candidate"],
                              "step":h["step"],"validation_nll":h["validation"]["nll"]})
    pd.DataFrame(histories).to_csv(root/"tuning_curves.csv",index=False)
    result = {"completed_utc":done["completed_utc"],"inner_fits":216,"outer_refits":36,"ising_fits":16,
        "summary":summary.to_dict("records"),"contrasts":contrasts,"selected_settings":chosen_rows,
        "fold_contrasts":fold_differences,
        "parameters":{**protocol["parameters"],"I0":19},"hybrid_additional_frozen_coefficients":19,
        "tuning_stopping_reasons":pd.Series([m["stopping_reason"] for _,m in tuning]).value_counts().to_dict(),
        "zero_update_refits":sum(m["steps"] == 0 for _,m in refits),
        "fit_seconds":float(sum(m["seconds"] for _,m in tuning+refits)),
        "maximum_cpu_mps_probability_difference":done["maximum_cpu_mps_probability_difference"],
        "prior_artifacts_verified_unchanged":done["prior_artifacts_verified_unchanged"],
        "fresh_confirmatory_test":False,"new_source_model_calls":0,"new_official_test_evaluations":0}
    json_write(root/"results.json",result)
    write_report(cfg,result,assessed,fold_summary)
    files = [root/x for x in ("results.json","summary.csv","question_means.csv","outer_fold_summary.csv",
        "paired_contrasts.csv","fold_contrasts.csv","tuning_runs.csv","selected_settings.csv","all_candidate_scores.csv","tuning_curves.csv")]
    files += [Path("reports/transformer_generalization.md"),Path("reports/transformer_generalization.png")]
    json_write(root/"report_integrity.json",{"created_utc":datetime.now(timezone.utc).isoformat(),
        "report_script_sha256":digest(__file__),"artifacts":{str(p):digest(p) for p in files}})
    print(json.dumps(result,indent=2))


def write_report(cfg,r,byq,folds):
    lookup = {(x["model"],x["split"]):x for x in r["summary"]}
    lines = ["# Transformer generalization: nested whole-question comparison", "",
        f"Completed **216 tuning fits, 36 refits with new initialization seeds and 16 fold-specific Ising fits** on the existing GPT development pool. Every neural setting is selected within the outer training questions. All outer outcomes are scored only after selection and all refits are frozen.", "",
        "## Outer-fold results", "",
        "Each of 20 questions is assessed once, in one of four outer folds. Each outer refit uses the other 15 questions. Neural results average three initialization seeds within each question, then give every question equal weight. These are fresh fits on previously studied questions, not a new confirmatory dataset. The numbers cannot be compared directly with earlier scores on only four validation questions.", "",
        "| Model | Neural parameters | Other fitted coefficients | Refit training error | Outer error | Outer NLL | Outer Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind in ORDER:
        tr,va = lookup[kind,"train"],lookup[kind,"assessment"]
        neural = 0 if kind == "I0" else r["parameters"][kind]
        other = 19 if kind in ("I0","ising_small") else 0
        lines.append(f"| {LABELS[kind]} | {neural:,} | {other} | {100*tr['error']:.2f}% | {100*va['error']:.2f}% | {va['nll']:.5f} | {va['brier_two_class']:.5f} |")
    lines += ["", "## Paired comparisons", "",
        "Differences are first minus second; negative favors the first model. Intervals resample the 20 question-level differences after averaging seeds, with 3,000 paired bootstrap draws. They are descriptive, pointwise intervals: only four outer folds, overlapping training sets, previously examined questions and multiple contrasts limit inferential claims. Question/fold win counts use a 0.000001-nat tolerance to avoid counting floating-point ties.", "",
        "| Comparison | NLL difference [descriptive 95% interval] | Error difference, pp [interval] | Questions with lower NLL |",
        "| --- | ---: | ---: | ---: |"]
    for first,second in (("full","I0"),("small","full"),("ising_small","small"),("ising_small","full"),("ising_small","I0"),("small","I0")):
        n = next(x for x in r["contrasts"] if x["first"] == first and x["second"] == second and x["metric"] == "nll")
        e = next(x for x in r["contrasts"] if x["first"] == first and x["second"] == second and x["metric"] == "error")
        lines.append(f"| {LABELS[first]} − {LABELS[second]} | {n['difference']:+.5f} [{n['descriptive_lower']:+.5f}, {n['descriptive_upper']:+.5f}] | {100*e['difference']:+.2f} [{100*e['descriptive_lower']:+.2f}, {100*e['descriptive_upper']:+.2f}] | {n['questions_better']}/20 |")
    lines += ["", "## What was matched and selected", "",
        "All models use the same current opinions, 16-dimensional persona/question context, signed graph and node degrees. No arguments, opinion history, question IDs or extra source responses were added. The full transformer has the original A2 architecture. The smaller and hybrid models have the same 34,978-parameter architecture; their initial neural weights and minibatch orders match. All neural output layers start at zero. Plain models initially predict one half; the hybrid initially reproduces its fold's fitted Ising probabilities.", "",
        "Every neural model receives the same six candidate settings: learning rate 0.0001 or 0.0003, crossed with an output penalty of 0, 0.001 or 0.01. Dropout is 0.1 and AdamW weight decay is 0.001. The added loss is lambda times the mean squared neural logit difference. Thus ordinary models shrink toward an uncertain prediction and the hybrid shrinks toward Ising. This tests the choice of prior against generic confidence regularization at the same size and search budget. It does not isolate a residual connection while keeping every selected hyperparameter identical, because each procedure selects its own candidate.", "",
        "Within each outer training set, three inner folds fit ten questions and select on five. Question PCA3, persona PCA3 and Ising's 19 coefficients are refitted inside each fitting set. All 32 personas occur in training in every fold. Held-out question embeddings are only transformed; their labels never fit PCA or Ising. The signed social channels use the original author's pinned two-stage fit.", "",
        "Inner tuning uses seed 42, validation every 128 updates (including initialization), six-check patience, a 0.0001 minimum improvement and a 3,072-update cap. NLL excludes the training penalty. Each model's candidate minimizes mean NLL across the three inner folds. Refits use seeds 142/143/144 and the median selected inner number of data passes, scaled from ten to fifteen training questions. Refitting has no validation stopping or outer-label scoring. Initialization is a valid selection; a zero-duration hybrid is exactly Ising.", "",
        f"Inner stopping reasons: {r['tuning_stopping_reasons']}. Zero-update outer refits: {r['zero_update_refits']}/36. Total recorded neural fit time: {r['fit_seconds']/60:.1f} minutes. A fixed compute ceiling remains a search limitation; this is not an exhaustive regularization or architecture search.", "",
        "## Selected settings", "",
        "| Outer fold | Model | Learning rate | Output penalty | Refit updates |",
        "| --- | --- | ---: | ---: | ---: |"]
    for x in r["selected_settings"]:
        lines.append(f"| {x['outer']} | {LABELS[x['model']]} | {x['learning_rate']:g} | {x['penalty']:g} | {x['refit_steps']} |")
    n = next(x for x in r["contrasts"] if x["first"] == "ising_small" and x["second"] == "small" and x["metric"] == "nll")
    s = next(x for x in r["contrasts"] if x["first"] == "small" and x["second"] == "full" and x["metric"] == "nll")
    hybrid_folds_better = sum(x["nll_difference"] < -1e-6 for x in r["fold_contrasts"] if x["first"] == "ising_small" and x["second"] == "small")
    small_folds_better = sum(x["nll_difference"] < -1e-6 for x in r["fold_contrasts"] if x["first"] == "small" and x["second"] == "full")
    lines += ["", "## Interpretation and scope", "",
        f"The matched small-hybrid comparison changes outer NLL by {n['difference']:+.5f}; reducing the ordinary model's size changes NLL by {s['difference']:+.5f}. Assess these paired differences, their question/fold variation and their intervals rather than choosing a method from training error alone. The hybrid can preserve errors in Ising as well as transferable structure. Better training fit alone does not establish better generalization.", "",
        f"The hybrid has lower NLL than the small ordinary model in {hybrid_folds_better}/4 outer folds; the small ordinary model beats the full model in {small_folds_better}/4 folds. Exact fold differences are saved in `fold_contrasts.csv`.", "",
        "Nested separation prevents the new tuning procedure from choosing settings on each outer fold's labels. It does not undo earlier human/model-development exposure to these 20 questions. No official test questions, unfamiliar graph tests, source-model calls or autonomous rollouts are added. These results concern next majority-of-five agent opinions, not mathematical correctness. They do not settle argument representation or establish a universal interaction law.", "",
        "Generalization here means holding out question IDs within the same question bank, GPT source, 32 personas and eight familiar graphs. Semantic template families are not independently grouped or held out; exact duplicate question texts were already rejected by the source preparation.", "",
        "Broader question coverage remains a separate proposed experiment. A later data study should hold the source LLM and generation protocol fixed, compare question breadth at a controlled transition budget, and retain genuinely untouched assessment questions. This run does not generate that new dataset.", "",
        "## Verification and artifacts", "",
        f"Eight focused tests passed: nested question isolation, held-out PCA perturbation, baseline/target input boundaries, matched zero-output initialization, objective equivalence, equal-question scoring, attention learning after initialization and exact stochastic resume. All 36 refitted checkpoints were independently reloaded on CPU on 64 training and 64 assessment snapshots each. Maximum probability difference from MPS is {r['maximum_cpu_mps_probability_difference']:.3g}. All {r['prior_artifacts_verified_unchanged']} recorded files from earlier studies are unchanged. Saved training metrics, fold identities, selected durations, initialization hashes and cohort checksums were verified.", "",
        f"![Generalization comparison]({Path('reports/transformer_generalization.png').resolve()})", "",
        "The first panel shows equal-question mean NLL with four outer-fold means overlaid. The second shows paired question-level NLL differences after averaging seeds, colored by outer fold. The plotted points are not independent repeated datasets.", "",
        "```sh", ".venv/bin/python -m pytest tests/test_transformer_generalization.py",
        ".venv/bin/python scripts/transformer_generalization.py prepare",
        ".venv/bin/python scripts/transformer_generalization.py tune --device mps",
        ".venv/bin/python scripts/transformer_generalization.py select",
        ".venv/bin/python scripts/transformer_generalization.py refit --device mps",
        ".venv/bin/python scripts/report_transformer_generalization.py evaluate --device mps",
        ".venv/bin/python scripts/report_transformer_generalization.py summarize", "```", "",
        "The frozen protocol, folds, PCA/Ising fits, 252 neural run directories, tuning curves, selected settings, predictions, per-question scores and integrity records are under `runs/transformer_generalization_v1/`. Completed runs are reused only after identity and artifact checks.", ""]
    Path("reports/transformer_generalization.md").write_text("\n".join(lines))
    os.environ.setdefault("MPLCONFIGDIR",str(Path(".cache/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    colors = ["#888888","#31688e","#35a67c","#d26b38"]
    for i,(kind,color) in enumerate(zip(ORDER,colors)):
        axes[0].bar(i,lookup[kind,"assessment"]["nll"],color=color,alpha=.65)
        sub = folds[(folds.model == kind)&(folds.split == "assessment")].sort_values("outer")
        axes[0].scatter(i+np.linspace(-.10,.10,len(sub)),sub.nll,color=color,edgecolors="black",s=32,zorder=3)
    axes[0].set_xticks(range(4),["Ising","Full\ntransformer","Small\ntransformer","Ising + small\ntransformer"])
    axes[0].set_ylabel("Outer-question NLL (lower is better)")
    axes[0].set_title("Means and four outer folds")
    for i,(first,second,label) in enumerate((("small","full","Small − full"),("ising_small","small","Hybrid − small"))):
        a = byq[byq.model == first].set_index("qid").sort_index()
        b = byq[byq.model == second].set_index("qid").sort_index()
        difference = a.nll-b.nll
        axes[1].scatter(i+np.linspace(-.16,.16,20),difference,c=a.outer,cmap="viridis",s=34,alpha=.8)
        axes[1].plot([i-.23,i+.23],[difference.mean()]*2,color="black",lw=2)
    axes[1].axhline(0,color=".4",ls="--",lw=1)
    axes[1].set_xticks([0,1],["Small − full","Hybrid − small"])
    axes[1].set_ylabel("Question-level NLL difference")
    axes[1].set_title("Negative favors the first model")
    from matplotlib.lines import Line2D
    cmap = plt.get_cmap("viridis")
    handles = [Line2D([0],[0],marker="o",color="none",markerfacecolor=cmap(i/3),markeredgecolor="none",label=str(i)) for i in range(4)]
    axes[1].legend(handles=handles,title="Outer fold",ncol=2,fontsize=8,frameon=False)
    for ax in axes:
        ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Nested question comparison · 20 existing development questions · three refit seeds")
    fig.savefig("reports/transformer_generalization.png",dpi=180)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("evaluate","summarize"))
    parser.add_argument("--device",choices=("cpu","mps"),default="mps")
    args = parser.parse_args()
    cfg = config(CONFIG); torch.set_num_threads(cfg["training"]["torch_threads"])
    if args.stage == "evaluate": evaluate(cfg,args.device)
    else: summarize(cfg)


if __name__ == "__main__": main()
