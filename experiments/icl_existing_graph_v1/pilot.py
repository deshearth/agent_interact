"""Read-only source data -> blind ICL packet -> irrevocable forecasts -> scores."""
from argparse import ArgumentParser
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.hpc3_prepare import read,sha,write
from scripts import unseen_graph_experiment as baseline
from experiments.icl_forecast_v1.blind_pilot import visible_packet,rank

HERE=Path("experiments/icl_existing_graph_v1")
ROOT=Path("runs/icl_existing_graph_v1")
CONFIG=HERE/"protocol.json"


def now():return datetime.now(timezone.utc).isoformat()


def readable(packet,training=False):
    opinions={a["agent_id"]:a["current_opinion"] for a in packet["agents"]}
    rows=[]
    for a in packet["agents"]:
        neighbors=[e for e in packet["signed_edges"] if e["receiver"]==a["agent_id"]]
        counts={sign+op:sum(e["relationship"]==sign and opinions[e["sender"]]==op for e in neighbors) for sign in ["friendly","unfriendly"] for op in ["A","B"]}
        row={"agent":a["agent_id"],"current":a["current_opinion"],"fA":counts["friendlyA"],"fB":counts["friendlyB"],"uA":counts["unfriendlyA"],"uB":counts["unfriendlyB"]}
        if training:row["next"]=packet["training_outcomes"][a["agent_id"]]
        rows.append(row)
    return {"panel_id":packet["panel_id"],"question":packet["question"],"choices":packet["choices"],"rows":rows}


def prepare():
    if (ROOT/"ready.json").exists():verify();print(json.dumps({"ready":True,"reused":True}));return
    settings=read(CONFIG);original=Path(settings["parent"]);graph_root=Path(settings["graph_study"])
    graph_protocol=baseline.validate()
    availability=read(ROOT/"availability.json")
    complete=sorted(e for w in availability["workers"] for e in w["state"]["completed_episodes"])
    assert complete==settings["test_episodes"] and len(complete)==4
    questions=read(original/"selection.json")["questions"]
    train=sorted((q for q in questions if q["split"]=="train"),key=lambda q:rank(settings["seed"],"demonstration",q["qid"]))[:settings["training_questions"]]
    test_lookup={baseline.episode_name(q):q for q in graph_protocol["questions"]}
    test=[test_lookup[name] for name in complete]
    assert not {q["qid"] for q in train}&{q["qid"] for q in test}
    assert not {q["qid"] for q in train+test}&set(graph_protocol["reserved_assessment_question_ids"])
    inputs=[];demos=[];targets=[];panels=[];sources={}
    for training,part in [(True,train),(False,test)]:
        for i,q in enumerate(part):
            name=baseline.episode_name(q)
            path=original/"episodes/train"/(name+".json") if training else ROOT/"sealed/source"/(name+".json")
            episode=read(path);sources[str(path)]=sha(path)
            assert all(episode["meta"][k]==q[k] for k in ["qid","graph_id","split"])
            assert episode["meta"]["protocol_sha256"]==sha((original if training else graph_root)/"source_protocol.json")
            item={"qid":q["qid"],"graph_id":q["graph_id"],"split":q["split"],"t":settings["boundary_t"],"agents":list(range(32)),"panel_id":f"{'train' if training else 'test'}_{i:02d}"}
            packet=visible_packet(episode,item)
            next_opinions=episode["spins_history"][item["t"]+1]
            assert len(next_opinions)==32 and all(s in [-1,1] for s in next_opinions)
            if training:
                packet["training_outcomes"]=["A" if s==1 else "B" for s in next_opinions];demos.append(packet)
            else:
                inputs.append(packet)
                targets += [{"case_id":f"{item['panel_id']}_agent{a:02d}","panel_id":item["panel_id"],"target":int(s==1),"current":int(episode["spins_history"][item["t"]][a]==1)} for a,s in enumerate(next_opinions)]
                panels.append(item)
    write(ROOT/"inputs/demonstrations.json",demos)
    write(ROOT/"inputs/test_inputs.json",inputs)
    write(ROOT/"inputs/personas.json",[a["persona"] for a in demos[0]["agents"]])
    assert all([a["persona"] for a in p["agents"]]==read(ROOT/"inputs/personas.json") for p in demos+inputs)
    write(ROOT/"inputs/training_tables.json",[readable(p,True) for p in demos])
    write(ROOT/"inputs/test_tables.json",[readable(p) for p in inputs])
    write(ROOT/"sealed/targets.json",targets)
    write(ROOT/"sealed/panels.json",panels)
    tracked=[CONFIG,HERE/"pilot.py",Path("experiments/icl_forecast_v1/blind_pilot.py"),ROOT/"availability.json",original/"models_frozen.json",graph_root/"protocol.json"]
    tracked += sorted((ROOT/"inputs").glob("*.json"))+[ROOT/"sealed/targets.json",ROOT/"sealed/panels.json"]
    write(ROOT/"ready.json",{"prepared_utc":now(),"test_questions":4,"test_forecasts":128,"training_questions":8,"training_examples":256,"artifacts":{str(p):sha(p) for p in tracked},"source_sha256":sources,"test_outcomes_displayed":False})
    print(json.dumps({"ready":True,"training_examples":256,"test_forecasts":128,"test_outcomes_displayed":False,"allowed_input_directory":str(ROOT/"inputs")}))


def verify():
    ready=read(ROOT/"ready.json")
    for path,digest in {**ready["artifacts"],**ready["source_sha256"]}.items():assert sha(path)==digest,path
    return ready


def commit(path):
    ready=verify()
    if (ROOT/"committed.json").exists():raise RuntimeError("Forecasts already committed; no revisions allowed")
    probabilities=read(path)
    cases={r["case_id"] for p in read(ROOT/"inputs/test_inputs.json") for r in p["forecast_requests"]}
    assert set(probabilities)==cases and len(cases)==ready["test_forecasts"]
    assert all(type(p) in [int,float] and math.isfinite(p) and 0<p<1 for p in probabilities.values())
    strategy=ROOT/"strategy_before_test.md"
    assert strategy.is_file()
    write(ROOT/"predictions.json",probabilities)
    write(ROOT/"committed.json",{"committed_utc":now(),"predictions_sha256":sha(ROOT/"predictions.json"),"ready_sha256":sha(ROOT/"ready.json"),"strategy_sha256":sha(strategy),"test_forecasts":len(cases),"test_outcomes_revealed":False,"baseline_predictions_computed":False})
    print(json.dumps({"committed":True,"forecasts":len(cases),"predictions_sha256":sha(ROOT/"predictions.json")}))


def metrics(p,y):
    p,y=np.asarray(p,float),np.asarray(y,float)
    return {"accuracy":float(((p>=.5)==y).mean()),"error":float(((p>=.5)!=y).mean()),"nll":float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()),"brier":float(((p-y)**2).mean())}


def score():
    verify()
    if not (ROOT/"committed.json").exists():raise RuntimeError("Commit all128 forecasts before viewing outcomes or baselines")
    seal=read(ROOT/"committed.json")
    assert seal["predictions_sha256"]==sha(ROOT/"predictions.json") and seal["ready_sha256"]==sha(ROOT/"ready.json")
    assert seal["strategy_sha256"]==sha(ROOT/"strategy_before_test.md")
    if (ROOT/"results.json").exists():print(json.dumps(read(ROOT/"results.json"),indent=2));return
    protocol=baseline.validate();settings=read(CONFIG)
    truth=read(ROOT/"sealed/targets.json");panels=read(ROOT/"sealed/panels.json")
    forecasts=read(ROOT/"predictions.json");inputs=read(ROOT/"inputs/test_inputs.json")
    y=np.array([r["target"] for r in truth]).reshape(4,32)
    p=np.array([forecasts[r["case_id"]] for r in truth]).reshape(4,32)
    own=np.array([r["current"] for r in truth]).reshape(4,32)
    graph_root=Path(settings["graph_study"])
    with np.load(graph_root/"evaluation_inputs.npz",allow_pickle=False) as raw:
        lookup={str(q):i for i,q in enumerate(raw["qids"])}
        context=raw["context"][[lookup[q["qid"]] for q in panels]]
        ising_weights=raw["ising_weights"]
    graphs=read(graph_root/"graphs.json")
    graph=np.array([graphs[q["graph_id"]] for q in panels]);s=2*own-1
    assert all([int(a["current_opinion"]=="A") for a in packet["agents"]]==row.tolist() for packet,row in zip(inputs,own))
    candidates=[("assistant_icl",-1,p)]
    candidates.append(("ising19",-1,np.exp(baseline.ising_logp(ising_weights,s,context,graph)[...,1])))
    for seed in protocol["config"]["predictors"]["seeds"]:
        weights=dict(np.load(graph_root/"models"/f"nano_s{seed}.npz",allow_pickle=False))
        candidates.append(("transformer818",seed,np.exp(baseline.nano_logp(weights,s,context,graph)[...,1])))
    by_question=[];case_rows=[{**r,"pA_assistant":forecasts[r["case_id"]],"predicted_assistant":int(forecasts[r["case_id"]]>=.5)} for r in truth]
    for model,seed,prob in candidates:
        for i,q in enumerate(panels):by_question.append({"model":model,"seed":seed,"panel_id":q["panel_id"],**metrics(prob[i],y[i])})
        for row,value in zip(case_rows,prob.ravel()):row[f"pA_{model}_{seed}"]=float(value)
    names=["assistant_icl","ising19","transformer818"]
    averaged={model:np.array([[np.mean([r[k] for r in by_question if r["model"]==model and r["panel_id"]==q["panel_id"]]) for k in ["accuracy","error","nll","brier"]] for q in panels]) for model in names}
    summary={m:dict(zip(["accuracy","error","nll","brier"],x.mean(0).tolist())) for m,x in averaged.items()}
    summary["persistence"]={"accuracy":float((own==y).mean()),"error":float((own!=y).mean())}
    rng=np.random.default_rng(settings["seed"]);draws=rng.integers(0,4,size=(5000,4));contrasts=[]
    for other in ["ising19","transformer818"]:
        delta=averaged["assistant_icl"]-averaged[other]
        low,high=np.quantile(delta[draws].mean(1),[.025,.975],axis=0)
        contrasts += [{"contrast":"assistant_icl_minus_"+other,"metric":k,"difference":float(delta[:,j].mean()),"lower":float(low[j]),"upper":float(high[j])} for j,k in enumerate(["accuracy","error","nll","brier"])]
    write(ROOT/"case_results.json",case_rows);write(ROOT/"by_question.json",by_question)
    result={"scored_utc":now(),"commit_sha256":sha(ROOT/"committed.json"),"summary":summary,"contrasts":contrasts,"test_questions":4,"test_forecasts":128,"training_questions":8,"training_demonstrated_updates":256,"assistant_correct":int(((p>=.5)==y).sum()),"assistant_errors":int(((p>=.5)!=y).sum()),"test_outcomes_revealed_after_commit":True,"scope":settings["interpretation"],"weights_fitted_for_icl":False,"assessment_used":False,"official20_episode_graph_results_unchanged":True}
    write(ROOT/"results.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=ArgumentParser();parser.add_argument("stage",choices=["prepare","commit","score"]);parser.add_argument("--predictions",type=Path)
    args=parser.parse_args()
    if args.stage=="commit":commit(args.predictions)
    else:{"prepare":prepare,"score":score}[args.stage]()
