"""Prepare input-only packets, commit manual probabilities, then score once.

Only aggregate status is printed before predictions have been committed. The
truth file is for the scorer, never for the assistant while making forecasts.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.hpc3_prepare import read,sha,write

HERE=Path("experiments/icl_forecast_v1")
SETTINGS=HERE/"protocol.json"
ROOT=Path("runs/icl_forecast_v1")


def now():return datetime.now(timezone.utc).isoformat()


def rank(seed,role,value):
    return hashlib.sha256(f"{seed}|{role}|{value}".encode()).hexdigest()


def make_panel(questions,settings,training=False):
    split="train" if training else "validation"
    origin="retained" if training else "new"
    role="demonstration" if training else "test"
    count=settings["training_demonstration_questions"] if training else settings["test_questions"]
    agents=1 if training else settings["test_agents_per_question"]
    candidates=[q for q in questions if q["split"]==split and q.get("origin")==origin]
    candidates.sort(key=lambda q:rank(settings["seed"],role,q["qid"]))
    if len(candidates)<count:raise RuntimeError(f"Need {count} eligible {role} questions")
    output=[]
    for i,q in enumerate(candidates[:count]):
        ids=sorted(range(32),key=lambda a:rank(settings["seed"],role+"_agent",f"{q['qid']}|{a}"))[:agents]
        output.append({"qid":q["qid"],"graph_id":q["graph_id"],"split":split,"t":i%8,"agents":ids,"panel_id":f"{role}_{i:02d}"})
    assert len({q["qid"] for q in output})==count
    return output


def visible_packet(episode,item):
    """Whitelist inputs; future data and source metadata cannot enter the packet."""
    assert 0<=item["t"]<8 and item["split"] in ["train","validation"]
    current=episode["spins_history"][item["t"]]
    assert len(current)==32 and all(s in [-1,1] for s in current)
    graph=episode["J"]
    assert len(graph)==32 and all(len(row)==32 and all(v in [-1,0,1] for v in row) for row in graph)
    assert len(episode["personas"])==32 and set(episode["choices"])=={"A","B"}
    return {"panel_id":item["panel_id"],"boundary_t":item["t"],"question":episode["statement"],
        "choices":episode["choices"],"agents":[{"agent_id":i,"persona":p,"current_opinion":"A" if current[i]==1 else "B"} for i,p in enumerate(episode["personas"])],
        "signed_edges":[{"receiver":i,"sender":j,"relationship":"friendly" if v==1 else "unfriendly"} for i,row in enumerate(graph) for j,v in enumerate(row) if v],
        "forecast_requests":[{"case_id":f"{item['panel_id']}_agent{a:02d}","agent_id":a} for a in item["agents"]]}


def freeze():
    settings=read(SETTINGS)
    source=Path(settings["source"]);parent=Path(settings["parent"])
    paths=[SETTINGS,HERE/"blind_pilot.py",source/"protocol.json",parent/"selection.json",parent/"models_frozen.json"]
    expected={str(p):sha(p) for p in paths}
    path=ROOT/"protocol_frozen.json"
    if path.exists():
        assert read(path)["artifacts"]==expected
    else:write(path,{"frozen_utc":now(),"artifacts":expected,"test_outcomes_viewed":False,"predictions_created":False})
    print(json.dumps({"protocol_frozen":True,"test_questions":settings["test_questions"],"forecasts":settings["test_questions"]*settings["test_agents_per_question"],"status":"waiting_for_fresh_validation_conversations"}))


def verify_frozen():
    frozen=read(ROOT/"protocol_frozen.json")
    for path,digest in frozen["artifacts"].items():assert sha(path)==digest,path
    return read(SETTINGS)


def prepare():
    settings=verify_frozen();source=Path(settings["source"]);parent=Path(settings["parent"])
    if (ROOT/"ready.json").exists():
        verify_ready();print(json.dumps({"ready":True,"reused":True}));return
    if not (source/"selection_ready.json").exists():
        print(json.dumps({"ready":False,"reason":"Fresh screening and selection are not complete"}));return
    ready=read(source/"selection_ready.json")
    assert ready["selection_sha256"]==sha(source/"selection.json") and ready["quality_gate_passed"]
    questions=read(source/"selection.json")["questions"]
    test=make_panel(questions,settings)
    train=make_panel(questions,settings,True)
    assert not {q["qid"] for q in test}&{q["qid"] for q in train}
    panels=train+test
    paths={q["panel_id"]:(parent if q["split"]=="train" else source)/"episodes"/q["split"]/f"{q['qid']}__{q['graph_id']}__r00.json" for q in panels}
    missing=sum(not p.is_file() for p in paths.values())
    if missing:
        print(json.dumps({"ready":False,"missing_conversations":missing}));return
    inputs=[];demonstrations=[];truth=[]
    for item in panels:
        episode=read(paths[item["panel_id"]])
        assert all(episode["meta"][k]==item[k] for k in ["qid","graph_id","split"])
        expected_protocol=sha((parent if item["split"]=="train" else source)/"source_protocol.json")
        assert episode["meta"]["protocol_sha256"]==expected_protocol
        packet=visible_packet(episode,item)
        answers=[]
        for request in packet["forecast_requests"]:
            target=episode["spins_history"][item["t"]+1][request["agent_id"]]
            assert target in [-1,1],"Invalid target: stop without substituting easier cases"
            row={"case_id":request["case_id"],"observed_next":"A" if target==1 else "B"}
            answers.append(row)
            if item["split"]=="validation":truth.append({**row,"panel_id":item["panel_id"],"current":packet["agents"][request["agent_id"]]["current_opinion"]})
        if item["split"]=="train":demonstrations.append({**packet,"training_outcomes":answers})
        else:inputs.append(packet)
    write(ROOT/"inputs/demonstrations.json",demonstrations)
    write(ROOT/"inputs/test_inputs.json",inputs)
    instructions=("Predict each requested agent's next five-vote majority opinion after one synchronous exchange. "
        "Estimate P(A), not whether A is mathematically correct. A friendly edge presents the sender's message as someone to agree with; an unfriendly edge presents it as someone to disagree with. "
        "The next messages and all next votes are unknown at this boundary. Use the eight labeled training examples as context; no model weights are fitted. "
        "Read only this instructions file and the two input JSON files. Do not inspect source episodes, selection, sealed files, source probabilities or competing predictions. "
        "Write a JSON object {case_id: probability_of_A} for all160 cases, with each probability strictly between0and1. Commit all estimates before any scoring; no revision after revelation.\n")
    (ROOT/"inputs/instructions.txt").write_text(instructions)
    write(ROOT/"sealed/targets.json",truth)
    write(ROOT/"sealed/panel.json",{"panels":panels,"episodes_sha256":{str(p):sha(p) for p in paths.values()},"selection_sha256":sha(source/"selection.json")})
    files=[p for p in (ROOT/"inputs").iterdir()]+[ROOT/"sealed/targets.json",ROOT/"sealed/panel.json"]
    write(ROOT/"ready.json",{"prepared_utc":now(),"protocol_frozen_sha256":sha(ROOT/"protocol_frozen.json"),"questions":len(test),"forecasts":len(truth),"artifacts":{str(p.relative_to(ROOT)):sha(p) for p in files}})
    print(json.dumps({"ready":True,"questions":len(test),"forecasts":len(truth),"allowed_input_directory":str(ROOT/"inputs")}))


def verify_ready():
    verify_frozen();record=read(ROOT/"ready.json")
    assert record["protocol_frozen_sha256"]==sha(ROOT/"protocol_frozen.json")
    for path,digest in record["artifacts"].items():assert sha(ROOT/path)==digest,path
    return record


def commit(path):
    record=verify_ready()
    if (ROOT/"predictions_committed.json").exists():raise RuntimeError("Predictions already sealed; overwriting is forbidden")
    forecasts=read(path)
    cases={r["case_id"] for p in read(ROOT/"inputs/test_inputs.json") for r in p["forecast_requests"]}
    assert set(forecasts)==cases and len(cases)==record["forecasts"]
    assert all(type(p) in [int,float] and math.isfinite(p) and 0<p<1 for p in forecasts.values())
    write(ROOT/"predictions.json",forecasts)
    write(ROOT/"predictions_committed.json",{"committed_utc":now(),"predictions_sha256":sha(ROOT/"predictions.json"),"ready_sha256":sha(ROOT/"ready.json"),"forecasts":len(forecasts),"test_outcomes_revealed":False})
    print(json.dumps({"committed":True,"forecasts":len(forecasts)}))


def score():
    verify_ready()
    if not (ROOT/"predictions_committed.json").exists():raise RuntimeError("Seal all predictions before revealing outcomes")
    seal=read(ROOT/"predictions_committed.json")
    assert seal["ready_sha256"]==sha(ROOT/"ready.json") and seal["predictions_sha256"]==sha(ROOT/"predictions.json")
    if (ROOT/"results.json").exists():print(json.dumps(read(ROOT/"results.json")));return
    forecasts=read(ROOT/"predictions.json");truth=read(ROOT/"sealed/targets.json")
    grouped={}
    for row in truth:
        p=forecasts[row["case_id"]];y=int(row["observed_next"]=="A")
        grouped.setdefault(row["panel_id"],[]).append({"accuracy":float((p>=.5)==y),"nll":-math.log(p if y else 1-p),"brier":(p-y)**2,"persistence_accuracy":float(row["current"]==row["observed_next"])})
    per_question=[{k:sum(r[k] for r in rows)/len(rows) for k in rows[0]} for rows in grouped.values()]
    metrics={k:sum(q[k] for q in per_question)/len(per_question) for k in per_question[0]}
    rng=random.Random(read(SETTINGS)["seed"]);bootstrap={k:[] for k in metrics}
    for _ in range(5000):
        draw=rng.choices(per_question,k=len(per_question))
        for k in metrics:bootstrap[k].append(sum(q[k] for q in draw)/len(draw))
    intervals={k:[sorted(v)[124],sorted(v)[4874]] for k,v in bootstrap.items()}
    result={"scored_utc":now(),"questions":len(grouped),"forecasts":len(truth),"metrics":metrics,"question_bootstrap_95_percent":intervals,"predictions_commit_sha256":sha(ROOT/"predictions_committed.json"),"baseline_model_comparison":"Pending separate frozen-model evaluation on the exact committed panel","assessment_used":False,"scope":"Exploratory manual ICL pilot; one boundary per question"}
    write(ROOT/"results.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=ArgumentParser();p.add_argument("stage",choices=["freeze","prepare","commit","score"]);p.add_argument("--predictions",type=Path)
    args=p.parse_args()
    if args.stage=="commit":commit(args.predictions)
    else:{"freeze":freeze,"prepare":prepare,"score":score}[args.stage]()
