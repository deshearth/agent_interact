"""One post-hoc scale control using the frozen argument experiment's trainer.

Only multiply text inputs by 32. Same initialized weights, architecture, data,
optimizer and stopping rule. Preserve the original study and its feature cache.
"""
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, device_for, digest, json_write
from scripts import argument_response as study
from scripts.argument_features import ResponseSnapshots, load_cache

FACTOR=32
ROOT=Path("runs/argument_scale_diagnostic_v1")
CONFIG="configs/argument_scale_diagnostic.yaml"
BASE_ROOT=study.ROOT
BASE_IMPLEMENTATION=study.implementation


class ScaledSnapshots(ResponseSnapshots):
    def __getitem__(self,item):
        batch=super().__getitem__(item)
        batch["text"]=batch["text"]*FACTOR
        return batch


def implementation():
    return {**BASE_IMPLEMENTATION(),"scale_adapter_sha256":digest(__file__)}


@contextmanager
def runtime():
    names=("ROOT","CONFIG","implementation","ResponseSnapshots")
    before={name:getattr(study,name) for name in names}
    study.ROOT,study.CONFIG=ROOT,CONFIG
    study.implementation,study.ResponseSnapshots=implementation,ScaledSnapshots
    try:
        yield
    finally:
        for name,value in before.items():
            setattr(study,name,value)


def audit_features(cache,cfg):
    arrays,index,vectors,_=cache
    donors=np.load(Path(cfg["data"]["cache"])/"donors_s42.npy",allow_pickle=False)
    for condition in ("stance_only","full_arguments","shuffled_arguments"):
        for partition in ("train","validation"):
            base=ResponseSnapshots(arrays,index,vectors,partition,condition,donors)
            scaled=ScaledSnapshots(arrays,index,vectors,partition,condition,donors)
            assert base.pairs==scaled.pairs
            for item in (0,len(base)//2,len(base)-1):
                first,second=base[item],scaled[item]
                for key,value in first.items():
                    if isinstance(value,torch.Tensor):
                        torch.testing.assert_close(second[key],value*FACTOR if key=="text" else value,rtol=0,atol=0)
                    else:
                        assert second[key]==value
                if condition=="stance_only":assert not second["text"].any()
    return {"text_scaled_exactly":True,"other_fields_and_examples_unchanged":True,
            "stance_only_remains_identical":True,"partitions_checked":["train","validation"]}


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cpu","mps"],default="mps")
    args=parser.parse_args()
    cfg=config(CONFIG)
    assert cfg["diagnostic"]["text_input_multiplier"]==FACTOR
    torch.set_num_threads(cfg["training"]["torch_threads"])
    device=device_for(args.device)
    cache=load_cache(cfg,verify=True)
    audit=audit_features(cache,cfg)
    baselines={}
    for seed in cfg["training"]["seeds"]:
        for condition in ("stance_only","full_arguments","shuffled_arguments"):
            run=BASE_ROOT/f"A2-{condition}-s{seed}"
            meta=json.loads((run/"manifest.json").read_text())
            if meta["status"]!="complete":raise ValueError("Missing completed reference")
            for name,expected in meta["artifact_sha256"].items():
                if digest(run/name)!=expected:raise ValueError("Reference artifact changed")
            baselines[str(run)]={"manifest_sha256":digest(run/"manifest.json"),"best_sha256":digest(run/"best.pt"),
                                 "prediction_sha256":digest(run/"validation_predictions.parquet")}
    items=[{"kind":"A2","condition":condition,"seed":seed} for seed in cfg["training"]["seeds"]
           for condition in ("full_arguments","shuffled_arguments")]
    ROOT.mkdir(parents=True,exist_ok=True)
    identity={"implementation":implementation(),"config_sha256":digest(CONFIG),"config":cfg,
              "feature_metadata_sha256":digest(Path(cfg["data"]["cache"])/"metadata.json"),
              "baselines":baselines,"cohort":items,"audit":audit}
    protocol_path=ROOT/"protocol.json"
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text())
        if any(protocol[k]!=v for k,v in identity.items()):raise ValueError("Scale protocol changed")
    else:
        protocol={**identity,"created_utc":datetime.now(timezone.utc).isoformat(),
                  "hypothesis":"The original argument branch may be disadvantaged by its small magnitude relative to normalized opinion/history features.",
                  "choice_of_scale":"Single multiplier 32 selected from the measured initial text/base norm ratio 0.0275, aiming for comparable order of magnitude (about 0.88). No scale sweep or new encoder.",
                  "primary_comparison":"A2/full_arguments scale 32 versus the frozen matched A2/full_arguments scale 1, averaged over all three seeds",
                  "secondary_comparisons":["scale 32 full versus unchanged stance-only reference","scale 32 full versus scale 32 shuffled"],
                  "invariants":"Only text input magnitude changes. Identical parameter count, seed initialization, training/validation examples, batch order, learning rate, optimizer, patience and checkpoint-selection metric. Zero-text stance condition is exactly unchanged and reused.",
                  "assessment":"Question-mean NLL primary, accuracy alongside; paired four-question intervals descriptive only.",
                  "scope":"Post-hoc exploratory diagnosis after inspecting the original validation outcomes. No original checkpoint is overwritten. No test scoring, no new source LLM calls."}
        json_write(protocol_path,protocol)
    plan_path=ROOT/"plan.json"
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else {"protocol_sha256":digest(protocol_path),"elapsed_seconds":0.,"runs":[]}
    if plan["protocol_sha256"]!=digest(protocol_path):raise ValueError("Scale plan changed")
    started=time.monotonic()
    stop_at=started+cfg["training"]["aggregate_minutes"]*60-plan["elapsed_seconds"]
    try:
        with runtime():
            for item in items:
                if time.monotonic()>=stop_at:break
                manifest=study.train_one(item,cfg,cache,device,stop_at)
                row={**item,"run":str(ROOT/f"A2-{item['condition']}-s{item['seed']}"),"status":manifest["status"]}
                plan["runs"]=[r for r in plan["runs"] if (r["condition"],r["seed"])!=(item["condition"],item["seed"])]+[row]
                json_write(plan_path,plan)
                if manifest["status"]!="complete":break
        plan["complete"]=len(plan["runs"])==len(items) and all(r["status"]=="complete" for r in plan["runs"])
        if plan["complete"]:plan["completed_utc"]=datetime.now(timezone.utc).isoformat()
    finally:
        plan["elapsed_seconds"]+=time.monotonic()-started
        json_write(plan_path,plan)


if __name__=="__main__":main()
