"""Post-hoc diagnostic of whether aggregate argument effects hide useful subgroups."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from mac_agents.metrics import score, paired_bootstrap
from scripts import argument_mechanisms as previous
from scripts import question_context as question
from scripts import argument_residual_diagnostic as residual
from scripts import argument_residual_original_context as residual_pca3
from scripts import joint_argument_encoding as session
from scripts.argument_features import load_cache

ROOT = session.SESSION/"stratified_diagnostic"
KEYS = ["episode_id","round","agent"]
COMPARISONS = [
    ("pca3_arguments_vs_stance",previous.ROOT,"A2-full_arguments",previous.ROOT,"A2-stance_only","pca3"),
    ("full_question_arguments_vs_stance",question.ROOT,"full_embedding-full_arguments",question.ROOT,"full_embedding-stance_only","full"),
    ("pca3_question_mean_vs_individual",residual_pca3.ROOT,"pca3_question_mean-full_arguments",previous.ROOT,"A2-full_arguments","pca3"),
    ("pca3_residual_vs_individual",residual_pca3.ROOT,"pca3_individual_residual-full_arguments",previous.ROOT,"A2-full_arguments","pca3"),
    ("full_question_mean_vs_individual",residual.ROOT,"question_mean-full_arguments",question.ROOT,"full_embedding-full_arguments","full"),
    ("full_question_residual_vs_individual",residual.ROOT,"individual_residual-full_arguments",question.ROOT,"full_embedding-full_arguments","full"),
]


def load(root,prefix,seed,role):
    out = root/"evaluation"/f"{prefix}-s{seed}"
    path = out/f"{role}_predictions.parquet"
    manifest = json.loads((out/"identity.json").read_text())
    assert digest(path) == manifest["artifacts"][path.name]
    return pd.read_parquet(path).sort_values(KEYS).reset_index(drop=True)


def main():
    arrays,index,_,_ = load_cache(config(question.CONFIG),verify=True)
    index = previous.split_index(index,config(question.CONFIG))
    ep = {r["episode_id"]:e for e,r in enumerate(index)}
    for root in (residual.ROOT,residual_pca3.ROOT):
        assert json.loads((root/"plan.json").read_text())["complete"]
    ROOT.mkdir(parents=True,exist_ok=True)
    identity = {"code_sha256":digest(__file__),"feature_metadata_sha256":digest("data/processed/argument_response_v1/metadata.json"),
        "source_plans":{str(p):digest(p/"plan.json") for p in (previous.ROOT,question.ROOT,residual.ROOT,residual_pca3.ROOT)}}
    if (ROOT/"protocol.json").exists():
        p = json.loads((ROOT/"protocol.json").read_text());assert all(p[k] == v for k,v in identity.items())
    else:
        json_write(ROOT/"protocol.json",{**identity,"created_utc":datetime.now(timezone.utc).isoformat(),
            "purpose":"Check whether weak aggregate argument gains hide improvements for mixed incoming conclusions, balanced signed votes, uncertain baseline predictions, or the first update.",
            "strata":"All; isolated; messages all share A/B conclusion; both A and B present; degree>=2 and absolute signed opinion sum<=1; absolute signed sum>1; no-text three-seed mean confidence<=0.75 versus >0.75; first update versus later. These sets overlap.",
            "no_outcome_stratification":"Group membership uses observed pre-update inputs or frozen baseline probabilities only. No next opinion enters the masks. Uncertainty masks are common across seeds and each paired comparison.",
            "analysis":"Equal question weights within each subgroup; average seeds before paired question bootstrap. Report every group and both error and NLL. Only two argument-versus-stance comparisons extend to unseen questions; centroid/residual comparisons use familiar questions only.",
            "limitations":"Post-hoc exploratory subgroup analysis on repeatedly used development data. Small subgroups and many pointwise comparisons; no confirmatory significance claims or causal interpretations."})
    rows, masks_saved = [], []
    for label,root_a,prefix_a,root_b,prefix_b,context in COMPARISONS:
        roles = ("familiar","unseen") if label.endswith("vs_stance") else ("familiar",)
        baseline_root,baseline_prefix = (previous.ROOT,"A2-stance_only") if context == "pca3" else (question.ROOT,"full_embedding-stance_only")
        for role in roles:
            baseline = [load(baseline_root,baseline_prefix,s,role) for s in (42,43,44)]
            ref = baseline[0]
            p = np.stack([f.p_positive.to_numpy() for f in baseline]).mean(0)
            es = ref.episode_id.map(ep).to_numpy();ts = ref["round"].to_numpy();js = ref.agent.to_numpy()
            signs, stances = arrays["signs"][es,js],arrays["stance"][es,ts,js]
            degree = (signs != 0).sum(-1)
            positive = ((stances == 1)&(signs != 0)).sum(-1)
            negative = ((stances == -1)&(signs != 0)).sum(-1)
            field = np.abs((signs*stances).sum(-1))
            masks = {"all":np.ones(len(ref),dtype=bool),"isolated":degree == 0,
                "single_conclusion":(degree>0)&((positive == 0)|(negative == 0)),
                "mixed_conclusions":(positive>0)&(negative>0),"balanced_signed_votes":(degree>=2)&(field<=1),
                "unbalanced_signed_votes":field>1,"no_text_uncertain":np.maximum(p,1-p)<=.75,
                "no_text_confident":np.maximum(p,1-p)>.75,"first_update":ts == 0,"later_updates":ts>0}
            for name,mask in masks.items():
                masks_saved.append({"comparison":label,"role":role,"group":name,"examples":int(mask.sum()),
                    "questions":int(ref.loc[mask,"qid"].nunique()),"fraction":float(mask.mean())})
            for seed in (42,43,44):
                a,b = load(root_a,prefix_a,seed,role),load(root_b,prefix_b,seed,role)
                for frame in (a,b):
                    pd.testing.assert_frame_equal(ref[KEYS+["qid","target","current_state"]],frame[KEYS+["qid","target","current_state"]])
                for group,mask in masks.items():
                    for side,frame in (("first",a),("second",b)):
                        _,q = score(frame.loc[mask])
                        rows.extend(q.assign(comparison=label,role=role,group=group,side=side,seed=seed).to_dict("records"))
    questions = pd.DataFrame(rows)
    questions.to_csv(ROOT/"by_question.csv",index=False)
    pd.DataFrame(masks_saved).to_csv(ROOT/"support.csv",index=False)
    grouped = questions.groupby(["comparison","role","group","side","qid"],sort=True)[["nll","accuracy"]].mean()
    results = []
    for (comparison,role,group),part in questions.groupby(["comparison","role","group"],sort=True):
        a = grouped.loc[(comparison,role,group,"first")];b = grouped.loc[(comparison,role,group,"second")]
        support = next(r for r in masks_saved if (r["comparison"],r["role"],r["group"]) == (comparison,role,group))
        result = {**support,"first_error":float(1-a.accuracy.mean()),"second_error":float(1-b.accuracy.mean()),
            "first_nll":float(a.nll.mean()),"second_nll":float(b.nll.mean())}
        for metric in ("nll","accuracy"):
            paired = paired_bootstrap(a[metric],b[metric],1000,1729)
            result[f"{metric}_difference"] = paired["difference"]
            result[f"{metric}_ci_low"],result[f"{metric}_ci_high"] = paired["ci95"]
        results.append(result)
    summary = pd.DataFrame(results);summary.to_csv(ROOT/"summary.csv",index=False)
    files = ("by_question.csv","support.csv","summary.csv")
    json_write(ROOT/"complete.json",{"protocol_sha256":digest(ROOT/"protocol.json"),"completed_utc":datetime.now(timezone.utc).isoformat(),
        "artifacts":{f:digest(ROOT/f) for f in files},"groups_per_comparison":10,"paired_subgroup_comparisons":len(summary)})
    print(summary[summary.group.isin(["isolated","mixed_conclusions","no_text_uncertain"])][["comparison","role","group","examples","first_error","second_error","accuracy_difference"]].to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
