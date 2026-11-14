"""Balanced question-count learning curves on the existing development data.

Question PCA is fitted inside each subset. Validation cadence and patience are
defined in optimizer updates, so small subsets do not get less time to learn.
The original argument study and all original artifacts remain immutable.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, device_for, digest, environment, json_write, seed_all
from mac_agents.data.features import to_device
from mac_agents.data.prepare import pca_fit_transform
from mac_agents.metrics import prediction_frame, score
from mac_agents.train import restore_rng, rng_state, save_checkpoint
from scripts.argument_features import ResponseSnapshots, load_cache
from scripts import argument_response as original

CONFIG="configs/question_count_curve.yaml"
ROOT=Path("runs/question_count_curve_v1")
Q_EMBEDDINGS=Path("data/raw/upstream/data/obj/question_embeddings_choices.json")
P_EMBEDDINGS=Path("data/raw/upstream/data/obj/persona_embeddings.json")


def implementation():
    return {**original.implementation(),"question_count_curve.py":digest(__file__)}


def implementation_digest():
    return hashlib.sha256(json.dumps(implementation(),sort_keys=True).encode()).hexdigest()


def balanced_subsets(qids,seed=64017):
    qids=sorted(qids)
    if len(qids)!=16 or len(set(qids))!=16:
        raise ValueError("Expected the fixed 16 fitting questions")
    order=np.random.default_rng(seed).permutation(qids).tolist()
    blocks=[order[i:i+4] for i in range(0,16,4)]
    subsets=[]
    for path in range(4):
        nested=sum(blocks[path:]+blocks[:path],[])
        for n in (4,8,12):
            subsets.append({"id":f"path{path}_n{n:02d}","path":path,"n_questions":n,
                            "questions":sorted(nested[:n])})
    subsets.append({"id":"all_n16","path":None,"n_questions":16,"questions":qids})
    return blocks,subsets


def fit_question_features(embeddings,fit_qids,transform_qids):
    """Only fit_qids determine mean/components; unused questions only transform."""
    transform_qids=sorted(transform_qids)
    fit_qids=sorted(fit_qids)
    if len(fit_qids)<4 or not set(fit_qids)<=set(transform_qids):
        raise ValueError("Three-component PCA requires at least four fitting questions")
    values=np.asarray([embeddings[q] for q in transform_qids],dtype=np.float64)
    transformed,mean,components=pca_fit_transform(values,[transform_qids.index(q) for q in fit_qids])
    return dict(zip(transform_qids,transformed)),mean,components


def context_from_features(index,personas,qfeatures):
    e,n=len(index),len(personas)
    q=np.stack([qfeatures[row["qid"]] for row in index])
    return np.concatenate([np.ones((e,n,1)),np.broadcast_to(personas,(e,n,3)),
        np.broadcast_to(q[:,None,:],(e,n,3)),
        (personas[None,:,:,None]*q[:,None,None,:]).reshape(e,n,9)],axis=-1).astype(np.float32)


class SubsetSnapshots(ResponseSnapshots):
    def __init__(self,arrays,index,vectors,partition,condition,donors,train_questions):
        eligible={row["qid"] for row in index if row["question_split"]=="train"}
        validation={row["qid"] for row in index if row["question_split"]=="validation"}
        if not set(train_questions)<=eligible or set(train_questions)&validation:
            raise ValueError("Training question selection crosses the fixed split")
        super().__init__(arrays,index,vectors,partition,condition,donors)
        if partition=="train":
            self.pairs=[(e,t) for e,t in self.pairs if index[e]["qid"] in set(train_questions)]
        if not self.pairs:raise ValueError("Empty subset")


def verify_protocol_identity(protocol,cfg,metadata):
    expected={"implementation":implementation(),"config_sha256":digest(CONFIG),
              "feature_metadata_sha256":digest(Path(cfg["data"]["cache"])/"metadata.json"),
              "raw_question_embeddings_sha256":digest(Q_EMBEDDINGS),
              "raw_persona_embeddings_sha256":digest(P_EMBEDDINGS)}
    if any(protocol[key]!=value for key,value in expected.items()):
        raise ValueError("Learning-curve code, config or inputs changed; use a new namespace")
    if protocol["questions"]!=metadata["questions"]:
        raise ValueError("Development question split changed")
    for subset in protocol["subsets"]:
        if digest(ROOT/subset["feature_file"])!=subset["feature_sha256"]:
            raise ValueError("Subset preprocessing changed")
    for baseline in protocol["endpoint_references"]:
        path=Path(baseline["run"])
        if digest(path/"manifest.json")!=baseline["manifest_sha256"]:
            raise ValueError("16-question reference manifest changed")
        for name,sha in baseline["artifact_sha256"].items():
            if digest(path/name)!=sha:raise ValueError("16-question reference changed")


def prepare(cfg,cache):
    arrays,index,_,metadata=cache
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/"protocol.json").exists():
        protocol=json.loads((ROOT/"protocol.json").read_text())
        verify_protocol_identity(protocol,cfg,metadata)
        return protocol
    blocks,subsets=balanced_subsets(metadata["questions"]["train"],cfg["curve"]["subset_seed"])
    qids=sorted(set(row["qid"] for row in index))
    embeddings={row["qid"]:row["embedding"] for row in json.loads(Q_EMBEDDINGS.read_text())["items"]
                if row["qid"] in qids}
    raw_personas=sorted(json.loads(P_EMBEDDINGS.read_text())["items"],key=lambda x:x["idx"])
    with np.load(Path(cfg["data"]["prepared"])/"persona_pca.npz",allow_pickle=False) as pca:
        personas=(np.asarray([p["embedding"] for p in raw_personas])-pca["mean"])@pca["components"].T
    for subset in subsets:
        qfeatures,mean,components=fit_question_features(embeddings,subset["questions"],qids)
        # An independent holdout perturbation checks PCA fitting, not transformed held-out inputs.
        altered={q:np.array(value).copy() for q,value in embeddings.items()}
        for q in set(qids)-set(subset["questions"]):altered[q]*=1000
        _,other_mean,other_components=fit_question_features(altered,subset["questions"],qids)
        np.testing.assert_array_equal(mean,other_mean)
        np.testing.assert_array_equal(components,other_components)
        context=context_from_features(index,personas,qfeatures)
        if subset["n_questions"]==16:
            np.testing.assert_array_equal(context,arrays["context"])
            with np.load(Path(cfg["data"]["prepared"])/"question_pca.npz",allow_pickle=False) as pca:
                np.testing.assert_array_equal(mean,pca["mean"])
                np.testing.assert_array_equal(components,pca["components"])
        path=ROOT/"subsets"/f"{subset['id']}.npz"
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,context=context,mean=mean,components=components,
                            train_questions=np.array(subset["questions"]),development_qids=np.array(qids))
        subset.update(feature_file=str(path.relative_to(ROOT)),feature_sha256=digest(path),
                      train_episodes=32*subset["n_questions"],train_snapshots=256*subset["n_questions"],
                      train_agent_transitions=8192*subset["n_questions"])
    references=[]
    for condition in cfg["curve"]["conditions"]:
        for seed in cfg["training"]["seeds"]:
            run=original.ROOT/f"A2-{condition}-s{seed}"
            meta=json.loads((run/"manifest.json").read_text())
            if meta["status"]!="complete" or meta["implementation_sha256"]!=original.implementation_digest():
                raise ValueError("Invalid original endpoint")
            old=meta["config"]
            for key in ("batch_size","max_steps","learning_rate","weight_decay","gradient_clip","patience","min_delta"):
                if old["training"][key]!=cfg["training"][key]:raise ValueError("Training settings differ from endpoint")
            if old["model"]!=cfg["model"] or old["data"]!=cfg["data"]:
                raise ValueError("Model or cache differ from endpoint")
            if meta["train_snapshots"]//cfg["training"]["batch_size"]!=cfg["training"]["validation_every_steps"]:
                raise ValueError("Endpoint validation cadence mismatch")
            for name,sha in meta["artifact_sha256"].items():
                if digest(run/name)!=sha:raise ValueError("Original endpoint artifact changed")
            references.append({"subset_id":"all_n16","n_questions":16,"path":None,"condition":condition,
                               "seed":seed,"run":str(run),"manifest_sha256":digest(run/"manifest.json"),
                               "artifact_sha256":meta["artifact_sha256"],"reused":True})
    cohort=[{"subset_id":subset["id"],"path":subset["path"],"n_questions":subset["n_questions"],
             "condition":condition,"seed":seed}
            for subset in subsets if subset["n_questions"]<16
            for seed in cfg["training"]["seeds"] for condition in cfg["curve"]["conditions"]]
    protocol={"created_utc":datetime.now(timezone.utc).isoformat(),"implementation":implementation(),
              "config_sha256":digest(CONFIG),"config":cfg,
              "feature_metadata_sha256":digest(Path(cfg["data"]["cache"])/"metadata.json"),
              "raw_question_embeddings_sha256":digest(Q_EMBEDDINGS),"raw_persona_embeddings_sha256":digest(P_EMBEDDINGS),
              "questions":metadata["questions"],"blocks":blocks,"subsets":subsets,"cohort":cohort,
              "endpoint_references":references,
              "hypothesis":"Does increasing the number of distinct training questions improve generalization of the fixed argument-response predictor?",
              "design":"Four cyclic paths through a seeded partition into four blocks of four questions. Paths are nested. Each question appears once at size 4, twice at size 8 and three times at size 12. Size 16 is a shared endpoint, not four independent repetitions.",
              "preprocessing":"Fit three question-PCA components using only the selected training questions; transform the fixed validation questions. Persona PCA uses the same 32 personas present in every subset. Frozen text embeddings are unchanged.",
              "optimization":"Validation every 128 optimizer updates, patience six checks, max 5000 updates for every subset. Shuffle unique community snapshots each data pass. The same seeds pair initialization across sizes and input conditions. This avoids giving small subsets fewer updates before early stopping.",
              "endpoint_validation":"All-16 context and PCA arrays must equal the original cache exactly. Replay its first 128-update validation for seed 42 before launching the curve. Existing 16-question model checkpoints are reused only after these checks.",
              "metrics":"Equal-question mean validation NLL primary; training/validation accuracy and error rate (1-accuracy), Brier and flip/stay scores alongside. Training scores use selected checkpoints in evaluation mode.",
              "contrasts":["16 minus 4, 8 minus 4, 12 minus 8, 16 minus 12 for each input condition",
                           "full_arguments minus stance_only at each question count"],
              "uncertainty":"Show individual subset/seed runs and nested-path means; spread is descriptive, not independent-sample confidence. Average seeds and subset paths within each validation question before 1000 paired question-bootstrap draws. Only four reused validation questions; pointwise intervals are exploratory.",
              "limits":"Question count changes both training examples and the fitted PCA. This is a full-pipeline learning curve, not a pure diversity intervention at fixed sample count. Cannot establish behavior above 16 training questions. No previously consumed test data are scored.",
              "audit":{"subset_only_pca_verified":True,"unused_embedding_perturbation_invariant":True,
                       "all16_context_exact_match":True,"n_new_fits":len(cohort),"n_endpoint_references":len(references)}}
    json_write(ROOT/"protocol.json",protocol)
    print(json.dumps({"prepared":True,"new_fits":len(cohort),"endpoint_references":len(references),"subsets":len(subsets)}),flush=True)
    return protocol


def dataset_for(cache,subset,cfg,partition,condition,seed):
    arrays,index,vectors,_=cache
    with np.load(ROOT/subset["feature_file"],allow_pickle=False) as saved:
        if saved["train_questions"].tolist()!=subset["questions"]:
            raise ValueError("Subset feature/question identity mismatch")
        adapted={**arrays,"context":saved["context"]}
    donors=np.load(Path(cfg["data"]["cache"])/f"donors_s{seed}.npy",allow_pickle=False)
    dataset=SubsetSnapshots(adapted,index,vectors,partition,condition,donors,subset["questions"])
    expected=subset["train_snapshots"] if partition=="train" else 1024
    if len(dataset)!=expected:raise ValueError("Unexpected learning-curve sample count")
    return dataset


def save_scores(path,partition,model,dataset,cfg,device):
    logp=original.predictions(model,dataset,cfg,device)
    episodes,rounds=zip(*dataset.pairs)
    frame=prediction_frame(logp,episodes,rounds,dataset.arrays,dataset.index,"A2")
    summary,q=score(frame)
    frame.to_parquet(path/f"{partition}_predictions.parquet",index=False)
    json_write(path/f"{partition}_metrics.json",summary)
    q.to_csv(path/f"{partition}_by_question.csv",index=False)
    return summary


def train_run(item,subset,cfg,cache,device,stop_at,smoke=False):
    path=(ROOT/"smoke" if smoke else ROOT/"fits")/item["subset_id"]/f"A2-{item['condition']}-s{item['seed']}"
    path.mkdir(parents=True,exist_ok=True)
    train=dataset_for(cache,subset,cfg,"train",item["condition"],item["seed"])
    validation=dataset_for(cache,subset,cfg,"validation",item["condition"],item["seed"])
    identity={**item,"implementation_sha256":implementation_digest(),"protocol_sha256":digest(ROOT/"protocol.json"),
              "config_sha256":digest(CONFIG),"context_sha256":subset["feature_sha256"],
              "train_questions":subset["questions"],"actual_device":str(device),"smoke":smoke,
              "train_pairs_sha256":hashlib.sha256(json.dumps(train.pairs).encode()).hexdigest(),
              "validation_pairs_sha256":hashlib.sha256(json.dumps(validation.pairs).encode()).hexdigest()}
    meta_path=path/"manifest.json"
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else None
    if meta is not None:
        if any(meta[k]!=v for k,v in identity.items()):raise ValueError("Run identity changed")
        if meta["status"]=="complete":
            for name,sha in meta["artifact_sha256"].items():
                if digest(path/name)!=sha:raise ValueError("Completed curve artifact changed")
            return path,meta
    seed_all(item["seed"])
    model=original.make_model(cfg,"A2").to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    epoch,position,step,bad,best,history,elapsed=0,0,0,0,float("inf"),[],0.
    if meta is not None:
        ck=torch.load(path/"checkpoint.pt",map_location=device,weights_only=False)
        if ck["identity"]!=identity:raise ValueError("Curve checkpoint identity mismatch")
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"])
        epoch,position,step,bad,best=ck["epoch"],ck["position"],ck["step"],ck["bad"],ck["best"]
        history,elapsed=ck["history"],ck["elapsed_seconds"]
        restore_rng(ck["rng"])
    else:
        meta={**identity,"status":"running","created_utc":datetime.now(timezone.utc).isoformat(),
              "parameters":sum(p.numel() for p in model.parameters()),"config":cfg,
              "train_snapshots":len(train),"validation_snapshots":len(validation),
              "train_agent_transitions":len(train)*32,"validation_agent_transitions":len(validation)*32,
              "validation_questions":cache[-1]["questions"]["validation"],"environment":environment(),
              "dtype":"float32","autonomous":False,"forecast_boundary":"after_messages_before_next_opinion"}
    json_write(meta_path,meta)
    started=time.monotonic()
    def checkpoint():
        save_checkpoint(path/"checkpoint.pt",{"identity":identity,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "rng":rng_state(device),"epoch":epoch,"position":position,"step":step,"bad":bad,"best":best,
            "history":history,"elapsed_seconds":elapsed+time.monotonic()-started})
    def validate():
        nonlocal best,bad
        value=original.question_nll(original.predictions(model,validation,cfg,device),validation)
        if not np.isfinite(value):raise FloatingPointError("Nonfinite validation loss")
        history.append({"step":step,"completed_passes":epoch,"position":position,"validation_nll":value})
        if value<best-cfg["training"]["min_delta"]:
            best,bad=value,0
            save_checkpoint(path/"best.pt",{"model":model.state_dict(),"step":step,"validation_nll":value})
        else:bad+=1
        print(json.dumps({"subset":subset["id"],"condition":item["condition"],"seed":item["seed"],**history[-1]}),flush=True)
    checkpoint()
    limit=128 if smoke else cfg["training"]["max_steps"]
    reason="step_limit"
    interrupted=False
    while step<limit and bad<cfg["training"]["patience"]:
        order=np.random.default_rng(item["seed"]+epoch).permutation(len(train))
        loader=DataLoader(Subset(train,order[position:].tolist()),batch_size=cfg["training"]["batch_size"],
                          generator=torch.Generator().manual_seed(item["seed"]+epoch),num_workers=0)
        for batch in loader:
            if time.monotonic()>=stop_at:
                reason,interrupted="wall_time_limit",True
                break
            model.train()
            batch=to_device(batch,device)
            loss=F.cross_entropy(model(batch).reshape(-1,2),batch["target"].reshape(-1))
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["gradient_clip"])
            optimizer.step()
            step+=1;position+=len(batch["s"])
            if position==len(train):epoch+=1;position=0
            if step%cfg["training"]["validation_every_steps"]==0:
                validate();checkpoint()
            elif step%100==0:checkpoint()
            if bad>=cfg["training"]["patience"]:
                reason="early_stopping"
                break
            if step>=limit:break
        if interrupted or bad>=cfg["training"]["patience"]:break
    if interrupted:
        checkpoint()
        meta.update(status="bounded",stopping_reason=reason,steps=step,
                    elapsed_seconds=elapsed+time.monotonic()-started,checkpoint_sha256=digest(path/"checkpoint.pt"))
        json_write(meta_path,meta)
        return path,meta
    if not history or history[-1]["step"]!=step:validate()
    checkpoint()
    selected=torch.load(path/"best.pt",map_location=device,weights_only=False)
    model.load_state_dict(selected["model"])
    training_score=save_scores(path,"train",model,train,cfg,device)
    validation_score=save_scores(path,"validation",model,validation,cfg,device)
    json_write(path/"history.json",history)
    files=["checkpoint.pt","best.pt","history.json"]+[f"{part}_{suffix}" for part in ("train","validation")
             for suffix in ("predictions.parquet","metrics.json","by_question.csv")]
    meta.update(status="complete",stopping_reason=reason,steps=step,completed_passes=epoch,position=position,
                best_step=selected["step"],elapsed_seconds=elapsed+time.monotonic()-started,
                completed_utc=datetime.now(timezone.utc).isoformat(),artifact_sha256={name:digest(path/name) for name in files})
    json_write(meta_path,meta)
    print(json.dumps({"run":str(path),"status":"complete","seconds":meta["elapsed_seconds"],
                      "train":training_score,"validation":validation_score}),flush=True)
    return path,meta


def endpoint_scores(cfg,cache,protocol,device):
    subset=next(s for s in protocol["subsets"] if s["n_questions"]==16)
    for reference in protocol["endpoint_references"]:
        output=ROOT/"endpoint_scores"/f"A2-{reference['condition']}-s{reference['seed']}"
        if (output/"manifest.json").exists():
            meta=json.loads((output/"manifest.json").read_text())
            if meta["source"]!=reference or meta["implementation_sha256"]!=implementation_digest():
                raise ValueError("Endpoint score provenance changed")
            for name,sha in meta["artifact_sha256"].items():
                if digest(output/name)!=sha:raise ValueError("Endpoint scores changed")
            continue
        output.mkdir(parents=True,exist_ok=True)
        model=original.make_model(cfg,"A2").to(device)
        selected=torch.load(Path(reference["run"])/"best.pt",map_location=device,weights_only=False)
        model.load_state_dict(selected["model"])
        for partition in ("train","validation"):
            dataset=dataset_for(cache,subset,cfg,partition,reference["condition"],reference["seed"])
            summary=save_scores(output,partition,model,dataset,cfg,device)
            if partition=="validation":
                previous=json.loads((Path(reference["run"])/"validation_metrics.json").read_text())
                for key in ("nll","accuracy","brier"):
                    np.testing.assert_allclose(summary[key],previous[key],rtol=0,atol=2e-7)
        files=[f"{part}_{suffix}" for part in ("train","validation")
               for suffix in ("predictions.parquet","metrics.json","by_question.csv")]
        json_write(output/"manifest.json",{"source":reference,"implementation_sha256":implementation_digest(),
                   "source_scores_reproduced":True,"artifact_sha256":{name:digest(output/name) for name in files}})


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","smoke","run"])
    parser.add_argument("--device",choices=["cpu","mps"],default="mps")
    args=parser.parse_args()
    cfg=config(CONFIG)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    cache=load_cache(cfg,verify=True)
    protocol=prepare(cfg,cache)
    if args.stage=="prepare":return
    device=device_for(args.device)
    if args.stage=="smoke":
        subset=next(s for s in protocol["subsets"] if s["n_questions"]==16)
        item={"subset_id":subset["id"],"path":None,"n_questions":16,"condition":"full_arguments","seed":42}
        path,meta=train_run(item,subset,cfg,cache,device,time.monotonic()+300,smoke=True)
        recorded=json.loads((path/"history.json").read_text())[0]["validation_nll"]
        previous=json.loads((original.ROOT/"A2-full_arguments-s42"/"history.json").read_text())[0]["validation_nll"]
        np.testing.assert_allclose(recorded,previous,rtol=0,atol=2e-6)
        json_write(ROOT/"endpoint_replay.json",{"device":str(device),"step":128,"curve_nll":recorded,
                   "original_nll":previous,"matched":True,"implementation_sha256":implementation_digest()})
        print("First 128-update endpoint replay matched the original run",flush=True)
        return
    replay=json.loads((ROOT/"endpoint_replay.json").read_text())
    if not replay["matched"] or replay["implementation_sha256"]!=implementation_digest():
        raise ValueError("Run the endpoint replay before fitting subsets")
    plan_path=ROOT/"plan.json"
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else {
        "protocol_sha256":digest(ROOT/"protocol.json"),"elapsed_seconds":0.,"runs":[]}
    if plan["protocol_sha256"]!=digest(ROOT/"protocol.json"):raise ValueError("Curve plan changed")
    started=time.monotonic()
    stop_at=started+cfg["training"]["aggregate_minutes"]*60-plan["elapsed_seconds"]
    subsets={subset["id"]:subset for subset in protocol["subsets"]}
    try:
        for item in protocol["cohort"]:
            if time.monotonic()>=stop_at:break
            path,meta=train_run(item,subsets[item["subset_id"]],cfg,cache,device,stop_at)
            row={**item,"run":str(path),"status":meta["status"],"reused":False}
            key=lambda r:(r["subset_id"],r["condition"],r["seed"])
            plan["runs"]=[r for r in plan["runs"] if key(r)!=key(item)]+[row]
            json_write(plan_path,plan)
            if meta["status"]!="complete":break
        plan["fits_complete"]=len(plan["runs"])==len(protocol["cohort"]) and all(r["status"]=="complete" for r in plan["runs"])
        if plan["fits_complete"]:
            endpoint_scores(cfg,cache,protocol,device)
            plan.update(complete=True,completed_utc=datetime.now(timezone.utc).isoformat())
    finally:
        plan["elapsed_seconds"]+=time.monotonic()-started
        json_write(plan_path,plan)


if __name__=="__main__":main()
