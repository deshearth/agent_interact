"""Read-only comparison of selected versus already-saved terminal checkpoints.

No optimizer steps, checkpoint reselection, unseen-question or familiar-evaluation
rescoring. Terminal fitting scores clarify whether selected training errors bound
the model's ability to fit the data.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config,device_for,digest,json_write
from scripts import argument_response as original
from scripts import argument_mechanisms as previous
from scripts import question_context as question
from scripts import argument_residual_diagnostic as residual
from scripts import argument_residual_original_context as residual_pca3
from scripts import argument_question_identity as ids
from scripts import joint_argument_encoding as session
from scripts.argument_features import load_cache

ROOT = session.SESSION/"terminal_fit_audit"


def main():
    parser = ArgumentParser(description=__doc__);parser.add_argument("--device",choices=("cpu","mps"),default="cpu")
    args = parser.parse_args();torch.set_num_threads(4);device = device_for(args.device)
    cfg = config(question.CONFIG)
    arrays,index,vectors,meta = load_cache(cfg,verify=True)
    index = previous.split_index(index,cfg)
    base = (arrays,index,vectors,meta)
    full = (question.augmented_arrays(arrays,index,question.ROOT/"question_features.npz","full_embedding"),index,vectors,meta)
    res = residual.prepare(config(residual.CONFIG));pca = residual_pca3.prepare(config(residual_pca3.CONFIG));ident = ids.prepare(config(ids.CONFIG))
    families = [("pca3_stance",previous,"A2","stance_only",base),
        ("pca3_arguments",previous,"A2","full_arguments",base),
        ("full_question_stance",question,"full_embedding","stance_only",full),
        ("full_question_arguments",question,"full_embedding","full_arguments",full),
        ("full_question_residual",residual,"individual_residual","full_arguments",res["individual_residual"]),
        ("pca3_residual",residual_pca3,"pca3_individual_residual","full_arguments",pca["pca3_individual_residual"]),
        ("pca3_question_mean",residual_pca3,"pca3_question_mean","full_arguments",pca["pca3_question_mean"]),
        ("pca3_question_id",ids,"question_id","full_arguments",ident["question_id"])]
    provenance = {}
    for label,module,kind,condition,cache in families:
        for seed in (42,43,44):
            run = module.ROOT/f"{kind}-{condition}-s{seed}"
            manifest = json.loads((run/"manifest.json").read_text())
            assert manifest["status"] == "complete"
            for file in ("checkpoint.pt","best.pt"):
                assert digest(run/file) == manifest["artifact_sha256"][file]
            provenance[f"{label}/s{seed}"] = {"manifest_sha256":digest(run/"manifest.json"),
                "terminal_sha256":digest(run/"checkpoint.pt"),"selected_sha256":digest(run/"best.pt")}
    ROOT.mkdir(parents=True,exist_ok=True)
    identity = {"code_sha256":digest(__file__),"device":str(device),"source_checkpoints":provenance}
    path = ROOT/"protocol.json"
    if path.exists():
        p = json.loads(path.read_text());assert all(p[k] == v for k,v in identity.items())
    else:
        json_write(path,{**identity,"created_utc":datetime.now(timezone.utc).isoformat(),
            "purpose":"Separate fitting capacity from checkpoint selection using the terminal weights already saved by completed runs.",
            "scope":"Training predictions only for 24 terminal checkpoints. Selected training predictions and selection NLL histories are reused. No new training, model selection, familiar-evaluation or unseen predictions.",
            "interpretation":"Lower terminal training error with worse selection NLL indicates overfitting during the already-executed continuation. It does not determine unlimited model capacity or achievable generalization."})
    rows = []
    for label,module,kind,condition,cache in families:
        a,index,vec,_ = cache
        for seed in (42,43,44):
            if time.time() >= session.deadline(reserve=600):
                raise TimeoutError("Reserve time for final report; no further scoring")
            run = module.ROOT/f"{kind}-{condition}-s{seed}"
            out = ROOT/f"{label}-s{seed}";out.mkdir(exist_ok=True)
            record = out/"identity.json"
            identity_run = {"protocol_sha256":digest(path),"source":provenance[f"{label}/s{seed}"]}
            if record.exists():
                saved = json.loads(record.read_text());assert all(saved[k] == v for k,v in identity_run.items())
                for file,sha in saved["artifacts"].items():assert digest(out/file) == sha
                rows.append(saved["row"]);continue
            ck = torch.load(run/"checkpoint.pt",map_location=device,weights_only=False)
            selected = torch.load(run/"best.pt",map_location="cpu",weights_only=False)
            model = module.make_model(config(module.CONFIG),kind).to(device)
            model.load_state_dict(ck["model"])
            donors = np.load(previous.ROOT/f"donors_s{seed}.npy",allow_pickle=False)
            ds = previous.SplitSnapshots(a,index,vec,"train",condition,donors)
            terminal = original.save_predictions(out,"train_terminal",model,ds,cfg,device)
            selected_path = module.ROOT/"evaluation"/run.name/"train_metrics.json"
            selected_metrics = json.loads(selected_path.read_text())
            final_selection_nll = ck["history"][-1]["validation_nll"]
            assert ck["history"][-1]["step"] == ck["step"]
            row = {"family":label,"seed":seed,"selected_step":selected["step"],"terminal_step":ck["step"],
                "selected_train_error":1-selected_metrics["accuracy"],"terminal_train_error":1-terminal["accuracy"],
                "selected_train_nll":selected_metrics["nll"],"terminal_train_nll":terminal["nll"],
                "selected_selection_nll":selected["validation_nll"],"terminal_selection_nll":final_selection_nll}
            rows.append(row)
            files = ("train_terminal_predictions.parquet","train_terminal_metrics.json","train_terminal_by_question.csv")
            json_write(record,{**identity_run,"row":row,"artifacts":{f:digest(out/f) for f in files}})
            print(json.dumps(row),flush=True)
    frame = pd.DataFrame(rows);frame.to_csv(ROOT/"runs.csv",index=False)
    frame.groupby("family",sort=True).mean(numeric_only=True).to_csv(ROOT/"summary.csv")
    json_write(ROOT/"complete.json",{"protocol_sha256":digest(path),"completed_utc":datetime.now(timezone.utc).isoformat(),
        "checkpoints":len(rows),"optimizer_steps":0,"artifacts":{f:digest(ROOT/f) for f in ("runs.csv","summary.csv")},
        "per_run_identity":{p.parent.name:digest(p) for p in ROOT.glob("*/identity.json")}})


if __name__ == "__main__":main()
