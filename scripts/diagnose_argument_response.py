"""Post-hoc diagnostics of frozen development runs; no training or test scoring.

Reconstruct train/validation scores at initialization, selected and terminal
checkpoints. Audit actual parameter updates, branch scale and input sensitivity.
All findings are descriptive and leave the completed experiment unchanged.
"""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, device_for, digest, json_write, seed_all
from mac_agents.data.features import to_device
from scripts.argument_features import ResponseSnapshots, load_cache
from scripts.argument_response import CONFIG, ROOT, implementation_digest, make_model

OUT = Path("runs/argument_diagnostics_v1")


def dataset_targets(dataset):
    pairs = np.asarray(dataset.pairs)
    target = (dataset.arrays["spins"][pairs[:,0],pairs[:,1]+1] == 1).astype(np.int64)
    current = (dataset.arrays["spins"][pairs[:,0],pairs[:,1]] == 1).astype(np.int64)
    qids = np.array([dataset.index[e]["qid"] for e,_ in dataset.pairs])
    return target,current,qids


def score_arrays(logp, dataset):
    target,current,qids = dataset_targets(dataset)
    positive = np.exp(logp[...,1])
    prediction = logp.argmax(-1)
    loss = -np.take_along_axis(logp,target[...,None],axis=-1)[...,0]
    rows = []
    for qid in sorted(set(qids)):
        which = qids == qid
        y,s,p = target[which],current[which],prediction[which]
        correct,flip = p==y,s!=y
        rows.append({"qid":qid,"count":int(y.size),"nll":float(loss[which].astype(np.float64).mean()),
                     "accuracy":float(correct.mean()),
                     "brier":float((2*(positive[which].astype(np.float64)-y)**2).mean()),
                     "flip_accuracy":float(correct[flip].mean()) if flip.any() else None,
                     "stay_accuracy":float(correct[~flip].mean()) if (~flip).any() else None})
    qframe = pd.DataFrame(rows)
    return {key:float(qframe[key].mean()) for key in ("nll","accuracy","brier","flip_accuracy","stay_accuracy")},rows


def predict(model, dataset, cfg, device, zero_text=False, branch_scales=False):
    model.eval()
    values,scales = [],defaultdict(list)
    captured,handles = {},[]
    if branch_scales:
        def hook(name):
            def capture(module,inputs,output):
                captured[name]=output
            return capture
        for name in ("norm","text_projection","stance_projection"):
            handles.append(getattr(model,name).register_forward_hook(hook(name)))
    try:
        with torch.inference_mode():
            for batch in DataLoader(dataset,batch_size=cfg["training"]["batch_size"],shuffle=False,
                                    generator=torch.Generator().manual_seed(0),num_workers=0):
                b=to_device(batch,device)
                if zero_text:
                    b["text"].zero_()
                logp=F.log_softmax(model(b),-1)
                values.append(logp.cpu().numpy())
                if branch_scales:
                    sender=captured["norm"][torch.arange(len(b["s"]),device=device)[:,None,None],b["neighbors"]]
                    text=captured["text_projection"]
                    stance=captured["stance_projection"]
                    mask=b["signs"]!=0
                    base=sender+stance
                    named={"sender_norm":sender.norm(dim=-1),"stance_norm":stance.norm(dim=-1),
                           "text_norm":text.norm(dim=-1),"base_norm":base.norm(dim=-1),
                           "text_to_base_norm_ratio":text.norm(dim=-1)/base.norm(dim=-1).clamp_min(1e-12),
                           "edge_change_without_text":(model.edge_norm(base+text)-model.edge_norm(base)).norm(dim=-1)/model.edge_norm(base).norm(dim=-1).clamp_min(1e-12)}
                    for key,value in named.items():
                        scales[key].append(value[mask].cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
    summary={key:{"mean":float(np.concatenate(v).mean()),"median":float(np.median(np.concatenate(v))),
                  "p05":float(np.quantile(np.concatenate(v),.05)),"p95":float(np.quantile(np.concatenate(v),.95))}
             for key,v in scales.items()}
    return np.concatenate(values),summary


def parameter_updates(model, initial, checkpoint, run):
    rows=[]
    names=[name for name,_ in model.named_parameters()]
    ids=checkpoint["optimizer"]["param_groups"][0]["params"]
    if len(names)!=len(ids):
        raise ValueError("Optimizer/model parameter order differs")
    for name,param_id in zip(names,ids):
        tensor=checkpoint["model"][name].cpu()
        state=checkpoint["optimizer"]["state"].get(param_id)
        rows.append({"run":run,"parameter":name,"elements":tensor.numel(),
                     "initial_norm":float(initial[name].norm()),"terminal_norm":float(tensor.norm()),
                     "relative_weight_change":float((tensor-initial[name]).norm()/initial[name].norm().clamp_min(1e-12)),
                     "absolute_weight_change":float((tensor-initial[name]).norm()),
                     "optimizer_steps":int(state["step"].item()) if state else 0,
                     "optimizer_momentum_norm":float(state["exp_avg"].norm()) if state else 0.})
    return rows


def gradient_check(model,dataset,cfg,device,run):
    chosen=np.random.default_rng(87123).choice(len(dataset),size=32,replace=False)
    batch=next(iter(DataLoader(Subset(dataset,chosen.tolist()),batch_size=32,
                              generator=torch.Generator().manual_seed(0))))
    model.eval()
    model.zero_grad(set_to_none=True)
    batch=to_device(batch,device)
    loss=F.cross_entropy(model(batch).reshape(-1,2),batch["target"].reshape(-1))
    loss.backward()
    result={"run":run,"loss":float(loss.detach().cpu()),"parameter_gradients":{}}
    for name,param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                raise FloatingPointError("Nonfinite diagnostic gradient")
            result["parameter_gradients"][name]=float(param.grad.norm().cpu())
    model.zero_grad(set_to_none=True)
    return result


def embedding_geometry(arrays,index,vectors,donors,root):
    ids=arrays["text_ids"]
    shuffled=ids.reshape(-1)[donors]
    eligible=(ids!=0)&(ids!=shuffled)
    rng=np.random.default_rng(48216)
    rows,samples=[],[]
    raw_cache={}
    for partition in ("train","validation"):
        belongs=np.array([row["question_split"]==partition for row in index])
        flat=np.flatnonzero(eligible & belongs[:,None,None,None])
        selected=rng.choice(flat,size=min(20000,len(flat)),replace=False)
        first=ids.reshape(-1)[selected]
        second=shuffled.reshape(-1)[selected]
        cosine=np.einsum("ij,ij->i",vectors[first],vectors[second])
        distance=np.linalg.norm(vectors[first]-vectors[second],axis=-1)
        rows.append({"partition":partition,"different_text_edges":len(flat),"sampled_edges":len(selected),
                     "cosine_mean":float(cosine.mean()),"cosine_p05":float(np.quantile(cosine,.05)),
                     "cosine_median":float(np.median(cosine)),"cosine_p95":float(np.quantile(cosine,.95)),
                     "l2_mean":float(distance.mean()),"identical_vectors":int((distance<1e-7).sum())})
        if partition!="validation":continue
        e,t,r,k=np.unravel_index(selected,ids.shape)
        qids=np.array([index[i]["qid"] for i in e])
        for qid in sorted(set(qids)):
            candidates=np.flatnonzero(qids==qid)
            ordered=candidates[np.argsort(cosine[candidates])]
            for quantile in (.05,.5,.95):
                pos=int(ordered[int(quantile*(len(ordered)-1))])
                origin=int(selected[pos]); other=int(donors.reshape(-1)[origin])
                a=np.unravel_index(origin,ids.shape); b=np.unravel_index(other,ids.shape)
                texts=[]
                for ee,tt,rr,kk in (a,b):
                    sender=int(arrays["neighbors"][ee,rr,kk])
                    if ee not in raw_cache:
                        raw_cache[ee]=json.loads(Path(index[ee]["raw_path"]).read_text())
                    texts.append({"episode":index[ee]["name"],"round":int(tt),"sender":sender,"receiver":int(rr),
                                  "text":raw_cache[ee]["messages_history"][tt][f"{sender},{rr}"]})
                samples.append({"qid":qid,"selection_quantile":quantile,"cosine":float(cosine[pos]),
                                "statement":index[e[pos]]["statement"],"choices":index[e[pos]]["choices"],
                                "pair":texts})
    json_write(root/"argument_pair_examples.json",samples)
    return rows


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cpu","mps"],default="mps")
    args=parser.parse_args()
    cfg=config(CONFIG)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    device=device_for(args.device)
    cache=load_cache(cfg,verify=True)
    arrays,index,vectors,metadata=cache
    OUT.mkdir(parents=True,exist_ok=True)
    diagnostic_identity={"script_sha256":digest(__file__),"study_implementation_sha256":implementation_digest(),
                         "study_protocol_sha256":digest(ROOT/"protocol.json"),"feature_metadata_sha256":digest(Path(cfg["data"]["cache"])/"metadata.json")}
    if (OUT/"identity.json").exists() and json.loads((OUT/"identity.json").read_text())!=diagnostic_identity:
        raise ValueError("Diagnostic implementation changed; preserve its results in a new namespace")
    json_write(OUT/"identity.json",diagnostic_identity)
    started=time.monotonic()
    rows,questions,scale_rows,weights,gradients,sensitivity,learning=[],[],[],[],[],[],[]
    provenance={}
    for seed in cfg["training"]["seeds"]:
        donors=np.load(Path(cfg["data"]["cache"])/f"donors_s{seed}.npy",allow_pickle=False)
        for condition in ("stance_only","full_arguments"):
            run=ROOT/f"A2-{condition}-s{seed}"
            manifest=json.loads((run/"manifest.json").read_text())
            if manifest["implementation_sha256"]!=implementation_digest() or manifest["status"]!="complete":
                raise ValueError("Invalid source run")
            for name,expected in manifest["artifact_sha256"].items():
                if digest(run/name)!=expected:
                    raise ValueError("Source run artifact changed")
            provenance[run.name]={"manifest_sha256":digest(run/"manifest.json"),**manifest["artifact_sha256"]}
            seed_all(seed)
            model=make_model(cfg,"A2").to(device)
            initial={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            terminal=torch.load(run/"checkpoint.pt",map_location="cpu",weights_only=False)
            best=torch.load(run/"best.pt",map_location="cpu",weights_only=False)
            weights.extend(parameter_updates(model,initial,terminal,run.name))
            for point in json.loads((run/"history.json").read_text()):
                learning.append({"condition":condition,"seed":seed,**point})
            datasets={partition:ResponseSnapshots(arrays,index,vectors,partition,condition,donors)
                      for partition in ("train","validation")}
            selected_logs={}
            for stage,state,step in (("initial",initial,0),("selected",best["model"],best["step"]),
                                     ("terminal",terminal["model"],terminal["step"])):
                model.load_state_dict(state)
                for partition,dataset in datasets.items():
                    logp,scales=predict(model,dataset,cfg,device,
                                        branch_scales=condition=="full_arguments" and partition=="validation")
                    metrics,q=score_arrays(logp,dataset)
                    record={"condition":condition,"seed":seed,"stage":stage,"partition":partition,"step":step}
                    rows.append({**record,**metrics})
                    questions.extend({**record,**value} for value in q)
                    for feature,stats in scales.items():
                        scale_rows.append({**record,"feature":feature,**stats})
                    np.savez_compressed(OUT/f"{condition}-s{seed}-{stage}-{partition}.npz",logp=logp,
                                        pairs=np.asarray(dataset.pairs),qid=np.asarray([index[e]["qid"] for e,_ in dataset.pairs]))
                    if stage=="selected":
                        selected_logs[partition]=logp
                        if partition=="validation":
                            saved=json.loads((run/"validation_metrics.json").read_text())
                            for metric in ("nll","accuracy","brier"):
                                np.testing.assert_allclose(metrics[metric],saved[metric],rtol=0,atol=2e-7)
                print(json.dumps({"run":run.name,"stage":stage,"last_score":rows[-1]}),flush=True)
            if condition=="full_arguments":
                model.load_state_dict(best["model"])
                gradients.append(gradient_check(model,datasets["train"],cfg,device,run.name))
                for partition,dataset in datasets.items():
                    original=selected_logs[partition]
                    for perturbation in ("zero_text","matched_substitution"):
                        supplied=dataset if perturbation=="zero_text" else ResponseSnapshots(
                            arrays,index,vectors,partition,"shuffled_arguments",donors)
                        altered,_=predict(model,supplied,cfg,device,zero_text=perturbation=="zero_text")
                        metrics,q=score_arrays(altered,dataset)
                        original_metrics,_=score_arrays(original,dataset)
                        sensitivity.append({"seed":seed,"partition":partition,"perturbation":perturbation,**metrics,
                            "nll_change":metrics["nll"]-original_metrics["nll"],
                            "accuracy_change":metrics["accuracy"]-original_metrics["accuracy"],
                            "mean_absolute_probability_change":float(np.abs(np.exp(altered[...,1])-np.exp(original[...,1])).mean()),
                            "fraction_choices_changed":float((altered.argmax(-1)!=original.argmax(-1)).mean())})
                if not all(g["parameter_gradients"][name]>0 for g in gradients for name in
                           ("text_projection.weight","q.weight","k.weight","v.weight")):
                    raise ValueError("A supposedly active branch is disconnected")
    donors=np.load(Path(cfg["data"]["cache"])/"donors_s42.npy",allow_pickle=False)
    geometry=embedding_geometry(arrays,index,vectors,donors,OUT)
    pd.DataFrame(rows).to_csv("reports/argument_diagnostic_scores.csv",index=False)
    pd.DataFrame(questions).to_csv("reports/argument_diagnostic_by_question.csv",index=False)
    pd.DataFrame(weights).to_csv("reports/argument_diagnostic_parameter_updates.csv",index=False)
    pd.DataFrame(scale_rows).to_csv("reports/argument_diagnostic_branch_scales.csv",index=False)
    pd.DataFrame(sensitivity).to_csv("reports/argument_diagnostic_sensitivity.csv",index=False)
    pd.DataFrame(geometry).to_csv("reports/argument_diagnostic_embedding_geometry.csv",index=False)
    json_write("reports/argument_diagnostic_gradients.json",gradients)
    # Verify all source artifacts remain unchanged after inference/gradient inspection.
    for name,record in provenance.items():
        for file,expected in record.items():
            path=ROOT/name/("manifest.json" if file=="manifest_sha256" else file)
            if digest(path)!=expected:
                raise ValueError("Diagnostics altered a frozen artifact")
    result={**diagnostic_identity,"created_utc":datetime.now(timezone.utc).isoformat(),"device":str(device),
            "elapsed_seconds":time.monotonic()-started,"runs":provenance,
            "frozen_run_scores_reproduced":True,"frozen_run_artifacts_unchanged":True,
            "gradient_paths_nonzero":True,"no_optimizer_steps_or_test_scoring":True,
            "scope":"Post-hoc descriptive analysis on the existing 16/4 development questions",
            "representation_geometry":"20,000 sampled different-text matched pairs per partition, seed 48216; no target labels used",
            "zero_text_caveat":"Deleting text from a text-trained model is a distribution shift, not a fair replacement for the separately trained stance-only baseline"}
    json_write("reports/argument_diagnostic_integrity.json",result)
    make_plot(pd.DataFrame(rows),pd.DataFrame(learning))
    print(json.dumps(result,indent=2),flush=True)


def make_plot(scores,learning):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
    colors={"stance_only":"#64748b","full_arguments":"#007f83"}
    names={"stance_only":"Stance + history","full_arguments":"Full argument text"}
    for condition,color in colors.items():
        for seed in (42,43,44):
            one=learning[(learning.condition==condition)&(learning.seed==seed)]
            axes[0].plot(one.epoch,one.validation_nll,color=color,alpha=.65,
                         label=names[condition] if seed==42 else None)
        for j,partition in enumerate(("train","validation")):
            one=scores[(scores.condition==condition)&(scores.partition==partition)]
            means=one.groupby("stage").nll.mean().reindex(["selected","terminal"])
            x=np.arange(2)+(0 if condition=="stance_only" else .06)
            axes[1].plot(x,means,marker="o",color=color,ls="-" if partition=="train" else "--",
                         label=names[condition]+" · "+partition)
    axes[0].set(title="Validation curves · three seeds",xlabel="Completed epoch",ylabel="Question-mean NLL")
    axes[1].set(title="Selected versus terminal checkpoints",xticks=[0,1],xticklabels=["Best validation","End of training"],ylabel="Question-mean NLL")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Diagnosing the trained argument transformer · development data",fontsize=12)
    fig.tight_layout()
    fig.savefig("reports/argument_diagnostics.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
