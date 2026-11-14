"""Paired fresh-random-graph evaluation of the frozen Llama predictors.

Preparation uses local PyTorch only to export existing weights and verify a
NumPy inference adapter. Cluster evaluation needs NumPy and the standard library;
it never fits a predictor, imports the source LLM, or observes assessment data.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hpc3_prepare import read, sha, write

ROOT = Path("runs/unseen_graph_v1")
SOURCE = Path("runs/question_breadth_v1")
CONFIG = Path("configs/unseen_graph.yaml")
CODE = [Path("scripts/unseen_graph_experiment.py"), Path("scripts/hpc3_unseen_graph.py"),
        Path("hpc3/unseen_graph.sbatch")]


def now():
    return datetime.now(timezone.utc).isoformat()


def episode_name(q):
    return f"{q['qid']}__{q['graph_id']}__r00"


def assignments(questions, graph_ids, seed):
    selected = sorted((q for q in questions if q["split"] == "validation"), key=lambda q:q["qid"])
    assert len(selected) == 20 and len({q["qid"] for q in selected}) == 20
    assert len(graph_ids) == len(set(graph_ids)) == 4
    schedule = list(graph_ids)*5
    random.Random(seed).shuffle(schedule)
    return [{**q, "seen_graph_id":q["graph_id"], "graph_id":g} for q,g in zip(selected,schedule)]


def nano_logp(weights, s, context, graph):
    """Exact architecture, evaluation mode, float64 numerical implementation."""
    s, context, graph = np.asarray(s), np.asarray(context,dtype=float), np.asarray(graph)
    b,n = s.shape
    assert context.shape == (b,n,16) and graph.shape == (b,n,n)
    w = {k:np.asarray(v,dtype=float) for k,v in weights.items()}
    def linear(x, name):
        result = x @ w[name+".weight"].T
        return result + w[name+".bias"] if name+".bias" in w else result
    def norm(x, name):
        return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)*w[name+".weight"]+w[name+".bias"]
    def gelu(x):
        return .5*x*(1+np.vectorize(math.erf,otypes=[float])(x/math.sqrt(2)))
    state = np.stack([s == -1,s == 1],-1).astype(float)
    structure = np.log1p(np.stack([(graph>0).sum(-1),(graph<0).sum(-1),np.abs(graph).sum(-1)],-1))
    h = linear(gelu(linear(np.concatenate([state,context,structure],-1),"encoder.0")),"encoder.2")
    hn = norm(h,"norm")
    q = linear(hn,"q").reshape(b,n,4,1).transpose(0,2,1,3)
    k = linear(hn,"k").reshape(b,n,4,1).transpose(0,2,1,3)
    v = linear(hn,"v").reshape(b,n,4,2).transpose(0,2,1,3)
    positive, negative = (graph>0)[:,None], (graph<0)[:,None]
    mask = positive | negative
    scores = q @ k.transpose(0,1,3,2)
    scores += positive*w["edge_bias"][None,:,None,None,0]+negative*w["edge_bias"][None,:,None,None,1]
    scores = np.where(mask,scores,-np.inf)
    scores = np.where(~mask.any(-1,keepdims=True),0.,scores)
    exp = np.exp(scores-scores.max(-1,keepdims=True))
    alpha = exp/exp.sum(-1,keepdims=True)*mask
    signed = positive*w["gamma"][None,:,None,None,0]+negative*w["gamma"][None,:,None,None,1]
    rho = np.logaddexp(0,linear(structure,"strength")).transpose(0,2,1)[...,None]
    aggregate = rho*((alpha*signed) @ v)
    h += linear(aggregate.transpose(0,2,1,3).reshape(b,n,8),"out")
    h += linear(gelu(linear(norm(h,"ffn.0"),"ffn.1")),"ffn.4")
    z = linear(norm(h,"classifier.0"),"classifier.1")
    delta = z[...,1]-z[...,0]
    return np.stack([-np.logaddexp(0,delta),-np.logaddexp(0,-delta)],-1)


def ising_logp(weights, s, context, graph):
    positive = np.einsum("bij,bj->bi",np.maximum(graph,0),s)
    negative = np.einsum("bij,bj->bi",np.minimum(graph,0),s)
    x = np.concatenate([context,np.stack([positive,negative,positive-negative],-1)],-1)
    delta = x @ weights
    return np.stack([-np.logaddexp(0,delta),-np.logaddexp(0,-delta)],-1)


def validate(root=ROOT):
    root=Path(root)
    protocol=read(root/"protocol.json")
    assert sha(root/"source_protocol.json") == sha(root/"protocol.json")
    for path,digest in protocol["code_sha256"].items():
        assert sha(path)==digest,path
    for path,digest in protocol["source_files_sha256"].items():
        assert sha(path)==digest,path
    assert sha(CONFIG)==protocol["config_sha256"]
    for name,digest in protocol["assets"].items():
        assert sha(root/name)==digest,name
    assert set(q["qid"] for q in protocol["questions"])==set(protocol["validation_question_ids"])
    assert not set(protocol["validation_question_ids"]) & set(protocol["reserved_assessment_question_ids"])
    assert all(q["split"]=="validation" and q["graph_id"] in protocol["unseen_graph_ids"] for q in protocol["questions"])
    assert Counter(q["graph_id"] for q in protocol["questions"])=={g:5 for g in protocol["unseen_graph_ids"]}
    return protocol


def prepare():
    import torch
    import yaml
    from scripts import breadth_probability_learning_curve as previous
    from scripts import transformer_generalization as base
    previous.validate()
    for path,digest in read(previous.ROOT/"completion_audit.json")["artifacts"].items():
        assert sha(path)==digest,path
    assert not (ROOT/"protocol.json").exists(), "Use validate to inspect an existing frozen plan"
    cfg=yaml.safe_load(CONFIG.read_text())
    all_questions=read(SOURCE/"selection.json")["questions"]
    selected=assignments(all_questions,cfg["unseen_graphs"],cfg["assignment_seed"])
    matrices=read("data/raw/upstream/data/J_matrices.json")["graphs"]
    graphs={g["id"]:g["J"] for g in matrices}
    for g in cfg["unseen_graphs"]:
        assert next(x for x in matrices if x["id"]==g)["group"]=="fresh"
        assert all(not np.array_equal(graphs[g],graphs[s]) for s in cfg["seen_graphs"])
    frozen=read(SOURCE/"models_frozen.json")
    common=dict(np.load(SOURCE/"predictors/development.npz",allow_pickle=False))
    features=dict(np.load(SOURCE/"predictors/folds/main/features.npz",allow_pickle=False))
    old_pred=dict(np.load(SOURCE/"error_diagnosis_v1/predictions.npz",allow_pickle=False))
    targets=dict(np.load(previous.ROOT/"targets.npz",allow_pickle=False))
    lookup={str(q):i for i,q in enumerate(common["episode_qids"])}
    episodes=np.array([lookup[q["qid"]] for q in selected])
    pair_ids=np.array([np.flatnonzero(common["pairs"][:,0]==e) for e in episodes])
    assert pair_ids.shape==(20,8)
    assert np.array_equal(common["pairs"][pair_ids,1],np.tile(np.arange(8),(20,1)))
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"models").mkdir(exist_ok=True)
    reproduction=[]
    e,t=common["pairs"].T
    for i,run in enumerate(frozen["refits"]):
        path=Path(run["directory"])/"best.pt"
        state=torch.load(path,map_location="cpu",weights_only=False)["model"]
        weights={key:value.numpy() for key,value in state.items()}
        assert sum(v.size for v in weights.values())==818
        np.savez_compressed(ROOT/"models"/f"nano_s{run['seed']}.npz",**weights)
        lp=nano_logp(weights,common["spins"][e,t],features["context"][e],common["J"][e])
        reference=old_pred["nano_logp"][i]
        difference=float(np.abs(np.exp(lp)-np.exp(reference)).max())
        np.testing.assert_allclose(np.exp(lp),np.exp(reference),atol=2e-5,rtol=0)
        np.testing.assert_array_equal(lp.argmax(-1),reference.argmax(-1))
        reproduction.append({"seed":run["seed"],"checkpoint_sha256":sha(path),
                             "maximum_probability_difference":difference,"all_classes_equal":True})
    weights=np.array(read(SOURCE/"predictors/folds/main/ising.json")["weights"])
    lp=ising_logp(weights,common["spins"][e,t],features["context"][e],common["J"][e])
    np.testing.assert_allclose(lp,old_pred["ising_logp"],atol=1e-12,rtol=0)
    np.savez_compressed(ROOT/"evaluation_inputs.npz",qids=np.array([q["qid"] for q in selected]),
        context=features["context"][episodes],ising_weights=weights,
        seen_spins=common["spins"][episodes],seen_votes=common["raw_votes"][episodes],
        seen_graph=common["J"][episodes],seen_soft_targets=targets["soft"][pair_ids],
        seen_ising_logp=old_pred["ising_logp"][pair_ids],seen_nano_logp=old_pred["nano_logp"][:,pair_ids])
    write(ROOT/"graphs.json",{g:graphs[g] for g in cfg["unseen_graphs"]})
    shutil.copyfile(SOURCE/"models_frozen.json",ROOT/"original_models_frozen.json")
    write(ROOT/"numpy_reproduction.json",{"models":reproduction,"ising_logp_tolerance":1e-12,
        "comparison_transitions_per_model":int(old_pred["target"].size)})
    assets=["evaluation_inputs.npz","graphs.json","original_models_frozen.json","numpy_reproduction.json"]
    assets += [f"models/nano_s{s}.npz" for s in cfg["predictors"]["seeds"]]
    protocol={"created_utc":now(),"config":cfg,"config_sha256":sha(CONFIG),
        "code_sha256":{str(p):sha(p) for p in CODE},"assets":{p:sha(ROOT/p) for p in assets},
        "questions":selected,"validation_question_ids":[q["qid"] for q in selected],
        "reserved_assessment_question_ids":[q["qid"] for q in all_questions if q["split"]=="assessment"],
        "unseen_graph_ids":cfg["unseen_graphs"],"seen_graph_ids":cfg["seen_graphs"],
        "original_source_protocol_sha256":sha(SOURCE/"source_protocol.json"),
        "original_selection_sha256":sha(SOURCE/"selection.json"),
        "original_models_frozen_sha256":sha(SOURCE/"models_frozen.json"),
        "paired_episode_sha256":{q["qid"]:sha(SOURCE/"episodes/validation"/f"{q['qid']}__{q['seen_graph_id']}__r00.json") for q in selected},
        "source_files_sha256":{p:sha(p) for p in ["scripts/breadth_collect.py","scripts/breadth_source.py",
            "scripts/local_conversation_pilot.py","scripts/hpc3_breadth.py","scripts/hpc3_parallel_collect.py","scripts/hpc3_prepare.py","hpc3/container.sh"]},
        "design":"Twenty previously used validation questions, one new conversation per question on one of four never-fitted random graphs (five questions per graph). Compare each new trajectory with its original seen-graph trajectory using identical frozen predictor weights and context.",
        "initial_pairing":"Reuse original round-zero source responses with matching prompts, preserving all five initial votes for every agent. Every reused response records original sample/chunk provenance. Later draws use new graph-specific identities; this does not isolate a deterministic causal graph effect.",
        "scope":"New random graph instances within the original 32-agent, 56-undirected-edge signed-graph family. Questions/personas are familiar to development evaluation but questions were excluded from predictor weight fitting. This is not new-topology-family testing or untouched-question confirmation.",
        "predictors":"Original majority-label 818-parameter transformer seeds142/143/144 and original Ising19 only. No fitting, model selection, PCA refitting, probability-target variant selection or argument inputs.",
        "analysis":"Equal question averages; average individual transformer seed scores, not ensemble predictions. Paired question bootstrap for unseen-minus-seen performance and transformer-minus-Ising. Source-probability expected/excess metrics distinguish sampled difficulty. Intervals are descriptive, conditional on four graph instances and one assignment/trajectory per question.",
        "assessment_generated_or_scored":False,"planned_new_episodes":20,
        "automatic_scoring":True,"automatic_resubmission":False}
    write(ROOT/"protocol.json",protocol)
    shutil.copyfile(ROOT/"protocol.json",ROOT/"source_protocol.json")
    validate()
    print(json.dumps({"prepared":True,"episodes":20,"graph_counts":dict(Counter(q['graph_id'] for q in selected)),
                      "numpy_reproduction":reproduction},indent=2))


def episode_arrays(episode, question, protocol_sha, graph, paired_votes):
    """Validate a complete paired trajectory, including majority/edge identity."""
    expected={"qid":question["qid"],"graph_id":question["graph_id"],"split":"validation",
              "name":episode_name(question),"n":32,"k":5,"steps":8,"replica":0,
              "protocol_sha256":protocol_sha}
    assert all(episode["meta"].get(k)==v for k,v in expected.items())
    s=np.asarray(episode["spins_history"],dtype=np.int8)
    votes=np.asarray(episode["spins_raw_history"],dtype=np.int8)
    assert s.shape==(9,32) and votes.shape==(9,32,5)
    assert np.isin(s,[-1,1]).all() and np.isin(votes,[-1,1]).all()
    np.testing.assert_array_equal(s,np.where(votes.sum(-1)>0,1,-1))
    np.testing.assert_array_equal(votes[0],paired_votes)
    np.testing.assert_array_equal(episode["J"],graph)
    assert len(episode["messages_history"])==len(episode["communicated_stances_history"] )==8
    edge_keys={f"{a},{b}" for a in range(32) for b in range(32) if graph[a][b]}
    for messages in episode["messages_history"]:
        assert set(messages)==edge_keys
    return s,votes


def logged_probabilities(root, question, votes):
    import hashlib
    probabilities=np.full((8,32),np.nan)
    quality={"responses":0,"outside_bare_AB_probability_sum":0.,"not_single_bare_tokens":0,
             "first_attempt_unparsed":0,"replacement_responses":0}
    for step in range(1,9):
        for start in range(0,32,4):
            group=f"{episode_name(question)}|round{step}|opinions{start}"
            path=root/"chunks"/(hashlib.sha256(group.encode()).hexdigest()+".json")
            chunk=read(path)
            assert chunk["group_id"]==group and chunk["protocol_sha256"]==sha(root/"source_protocol.json")
            resolved={(r["agent"],r["vote_position"]):r["parsed_sign"] for r in chunk["responses"]}
            for attempt in [1,2]:
                retry_group=f"{group}|retry{attempt}"
                retry=root/"chunks"/(hashlib.sha256(retry_group.encode()).hexdigest()+".json")
                if retry.exists():
                    replacement=read(retry)
                    assert replacement["group_id"]==retry_group and replacement["protocol_sha256"]==sha(root/"source_protocol.json")
                    for row in replacement["responses"]:
                        assert resolved[row["agent"],row["vote_position"]] not in [-1,1]
                        resolved[row["agent"],row["vote_position"]]=row["parsed_sign"]
                        quality["replacement_responses"]+=1
            for agent in range(start,start+4):
                rows=sorted((r for r in chunk["responses"] if r["agent"]==agent),key=lambda r:r["vote_position"])
                assert len(rows)==5 and [r["vote_position"] for r in rows]==list(range(5))
                assert all(r["parse_attempt"]==0 for r in rows)
                quality["first_attempt_unparsed"]+=sum(r["parsed_sign"] not in [-1,1] for r in rows)
                np.testing.assert_array_equal([resolved[agent,v] for v in range(5)],votes[step,agent])
                a=np.array([r["first_literal_A_probability"] for r in rows])
                b=np.array([r["first_literal_B_probability"] for r in rows])
                assert np.ptp(a)==np.ptp(b)==0 and a[0]+b[0]>0
                p=a[0]/(a[0]+b[0])
                probabilities[step-1,agent]=np.clip(p**3*(10-15*p+6*p*p),0,1)
                quality["responses"]+=5
                quality["outside_bare_AB_probability_sum"]+=float(np.maximum(0,1-a-b).sum())
                quality["not_single_bare_tokens"]+=sum(r["raw_response"] not in ["A","B"] for r in rows)
    assert np.isfinite(probabilities).all()
    return probabilities,quality


def metrics(lp,y,q):
    pred=lp.argmax(-1)
    expected_error=np.where(pred,1-q,q)
    ref=np.minimum(q,1-q)
    entropy=-(q*np.log(np.clip(q,1e-300,1))+(1-q)*np.log(np.clip(1-q,1e-300,1)))
    expected_nll=-(1-q)*lp[...,0]-q*lp[...,1]
    return {"accuracy":float((pred==y).mean()),"error":float((pred!=y).mean()),
        "nll":float(-np.take_along_axis(lp,y[...,None],-1).mean()),
        "expected_error":float(expected_error.mean()),"expected_nll":float(expected_nll.mean()),
        "sampling_reference_error":float(ref.mean()),"excess_expected_error":float((expected_error-ref).mean()),
        "source_entropy":float(entropy.mean()),"excess_expected_nll":float((expected_nll-entropy).mean())}


def evaluate(root=ROOT):
    root=Path(root)
    protocol=validate(root)
    paths=[root/"episodes/validation"/(episode_name(q)+".json") for q in protocol["questions"]]
    if not all(p.exists() for p in paths):
        raise RuntimeError(f"Collection incomplete: {sum(p.exists() for p in paths)}/20 episodes; no partial scoring")
    if (root/"results.json").exists():
        result=read(root/"results.json")
        assert result["protocol_sha256"]==sha(root/"protocol.json")
        for path,digest in result["scored_episode_sha256"].items():assert sha(path)==digest
        return result
    data=dict(np.load(root/"evaluation_inputs.npz",allow_pickle=False))
    graphs=read(root/"graphs.json")
    new_s,new_j,new_q,quality=[],[],[],[]
    for i,(question,path) in enumerate(zip(protocol["questions"],paths)):
        assert question["qid"]==data["qids"][i]
        s,v=episode_arrays(read(path),question,sha(root/"source_protocol.json"),graphs[question["graph_id"]],data["seen_votes"][i,0])
        new_s.append(s);new_j.append(graphs[question["graph_id"]])
        q,details=logged_probabilities(root,question,v)
        new_q.append(q);quality.append({"qid":question["qid"],**details})
    new_s,new_j,new_q=np.array(new_s),np.array(new_j),np.array(new_q)
    inputs={"seen":(data["seen_spins"],data["seen_graph"],data["seen_soft_targets"]),
            "unseen":(new_s,new_j,new_q)}
    weights={s:dict(np.load(root/"models"/f"nano_s{s}.npz",allow_pickle=False)) for s in protocol["config"]["predictors"]["seeds"]}
    predictions={"qids":data["qids"],"unseen_source_majority_A":new_q}
    records=[]
    for regime,(spins,graph,q) in inputs.items():
        context=np.repeat(data["context"],8,axis=0)
        current=spins[:,:-1].reshape(-1,32)
        expanded_graph=np.repeat(graph,8,axis=0)
        y=(spins[:,1:]==1).astype(int)
        predictions[regime+"_target"]=y
        all_predictions=[("I0",-1,ising_logp(data["ising_weights"],current,context,expanded_graph))]
        all_predictions += [("nano",seed,nano_logp(w,current,context,expanded_graph)) for seed,w in weights.items()]
        for model,seed,flat_lp in all_predictions:
            lp=flat_lp.reshape(20,8,32,2)
            if regime=="seen":
                reference=data["seen_ising_logp"] if model=="I0" else data["seen_nano_logp"][list(weights).index(seed)]
                np.testing.assert_allclose(np.exp(lp),np.exp(reference),atol=2e-5,rtol=0)
                np.testing.assert_array_equal(lp.argmax(-1),reference.argmax(-1))
            predictions[f"{regime}_{model}_{seed}_logp"]=lp
            for i,question in enumerate(protocol["questions"]):
                records.append({"regime":regime,"model":model,"seed":seed,"qid":question["qid"],
                    "graph_id":question["graph_id"] if regime=="unseen" else question["seen_graph_id"],
                    **metrics(lp[i],y[i],q[i])})
    metric_names=list(metrics(lp[0],y[0],q[0]))
    by_question=[]
    for regime in inputs:
        for model in ["I0","nano"]:
            for question in protocol["questions"]:
                part=[r for r in records if (r["regime"],r["model"],r["qid"])==(regime,model,question["qid"])]
                by_question.append({"regime":regime,"model":model,"qid":question["qid"],"graph_id":part[0]["graph_id"],
                    **{key:float(np.mean([r[key] for r in part])) for key in metric_names}})
    summary=[]
    for regime in inputs:
        for model in ["I0","nano"]:
            part=[r for r in by_question if (r["regime"],r["model"])==(regime,model)]
            summary.append({"regime":regime,"model":model,**{key:float(np.mean([r[key] for r in part])) for key in metric_names}})
    rng=np.random.default_rng(protocol["config"]["evaluation"]["bootstrap_seed"])
    draws=rng.integers(0,20,(protocol["config"]["evaluation"]["bootstrap_replicates"],20))
    contrasts=[]
    for first,second in [(('seen',m),('unseen',m)) for m in ['I0','nano']]+[((r,'I0'),(r,'nano')) for r in inputs]:
        for metric in metric_names:
            a=[r[metric] for r in by_question if (r['regime'],r['model'])==first]
            b=[r[metric] for r in by_question if (r['regime'],r['model'])==second]
            delta=np.array(b)-a
            low,high=np.quantile(delta[draws].mean(1),[.025,.975])
            contrasts.append({"first":list(first),"second":list(second),"metric":metric,
                "difference":float(delta.mean()),"lower":float(low),"upper":float(high)})
    np.savez_compressed(root/"predictions.npz",**predictions)
    write(root/"by_question.json",by_question)
    write(root/"by_seed_question.json",records)
    result={"completed_utc":now(),"protocol_sha256":sha(root/"protocol.json"),"summary":summary,
        "paired_contrasts":contrasts,"source_probability_quality":quality,
        "scored_episode_sha256":{str(p):sha(p) for p in paths},
        "output_sha256":{p:sha(root/p) for p in ["predictions.npz","by_question.json","by_seed_question.json"]},
        "new_episodes":20,"new_agent_transitions":5120,"models_refitted":False,
        "assessment_generated_or_scored":False,"interpretation":protocol["scope"]+" "+protocol["analysis"]}
    write(root/"results.json",result)
    print(json.dumps({"scoring_complete":True,"summary":summary}),flush=True)
    return result


def main():
    parser=ArgumentParser()
    parser.add_argument("stage",choices=["prepare","validate","evaluate"])
    args=parser.parse_args()
    if args.stage=="prepare":prepare()
    elif args.stage=="validate":print(json.dumps({"validated":True,"questions":len(validate()["questions"])}))
    else:evaluate()


if __name__=="__main__":main()
