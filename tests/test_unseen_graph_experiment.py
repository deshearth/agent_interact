"""Tests use synthetic examples; no source model or assessment access."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import unseen_graph_experiment as experiment
from scripts import hpc3_unseen_graph as collector
from scripts.hpc3_prepare import sha,write
from scripts import transformer_generalization as base
from mac_agents.common import config


def test_balanced_fresh_graph_assignment_excludes_train_and_assessment():
    questions=[{"qid":f"q{i:03d}","split":"validation" if i<20 else "assessment" if i<60 else "train",
                "graph_id":f"J{i%8}"} for i in range(140)]
    first=experiment.assignments(questions,[f"Jf{i}" for i in range(4)],918271)
    second=experiment.assignments(list(reversed(questions)),[f"Jf{i}" for i in range(4)],918271)
    assert first==second
    assert len(first)==20 and all(q["split"]=="validation" for q in first)
    assert all(sum(q["graph_id"]==g for q in first)==5 for g in [f"Jf{i}" for i in range(4)])
    assert all(q["seen_graph_id"] in [f"J{i}" for i in range(8)] for q in first)
    assert not any(q["graph_id"]==q["seen_graph_id"] for q in first)


def test_numpy_adapter_matches_torch_with_isolates_and_signed_new_graphs():
    torch.manual_seed(781)
    cfg=config("configs/unseen_graph.yaml")
    # Nonzero, random trained-like parameters exercise every computation branch.
    model=base.make_model(config("configs/breadth_probability_learning_curve.yaml"),"nano").eval()
    with torch.no_grad():
        model.classifier[-1].weight.normal_(0,.3)
        model.edge_bias.normal_(0,.2)
        model.gamma.normal_(0,1)
    rng=np.random.default_rng(782)
    graph=np.zeros((4,32,32),np.float32)
    for b in range(4):
        upper=np.triu(rng.choice([-1,0,1],(32,32),p=[.06,.88,.06]),1)
        graph[b]=upper+upper.T
    graph[:,0,:]=graph[:,:,0]=0
    graph[1]=0
    s=rng.choice([-1,1],(4,32)).astype(np.float32)
    context=rng.normal(size=(4,32,16)).astype(np.float32)
    with torch.inference_mode():
        z=model(torch.tensor(s),torch.tensor(context),torch.tensor(graph),torch.empty((4,32,0)))
        delta=z[...,1]-z[...,0]
        reference=torch.stack([-torch.nn.functional.softplus(delta),-torch.nn.functional.softplus(-delta)],-1).numpy()
    weights={k:v.detach().numpy() for k,v in model.state_dict().items()}
    actual=experiment.nano_logp(weights,s,context,graph)
    np.testing.assert_allclose(np.exp(actual),np.exp(reference),atol=2e-6,rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1),reference.argmax(-1))
    assert sum(v.size for v in weights.values())==818


def test_initial_reuse_keeps_true_responses_provenance_and_checks_prompts(tmp_path):
    q={"qid":"q0","graph_id":"Jf0","seen_graph_id":"J0","split":"validation"}
    group="q0__Jf0__r00|round0|opinions0"
    original="q0__J0__r00|round0|opinions0"
    requests,rows=[],[]
    for vote in range(5):
        for agent in range(4):
            prompt=f"Question q0; persona {agent}"
            metadata={"qid":"q0","episode":"q0__Jf0__r00","agent":agent,"vote_position":vote,"parse_attempt":0}
            sid=f"{original}|agent{agent}|vote{vote}|attempt0"
            requests.append({"sample_id":sid.replace("__J0__","__Jf0__"),"prompt":prompt,"metadata":metadata})
            rows.append({"sample_id":sid,"sample_seed":100+vote*4+agent,"agent":agent,"vote_position":vote,
                "parse_attempt":0,"parsed_sign":1 if vote%2 else -1,"raw_response":"A" if vote%2 else "B",
                "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"first_literal_A_probability":.4})
    path=collector.chunk_path(tmp_path,original)
    write(path,{"group_id":original,"protocol_sha256":"original_sha","responses":rows})
    expected={str(path):sha(path)}
    output,origin=collector.initial_responses(group,requests,q,{"original_source_protocol_sha256":"original_sha"},expected,tmp_path)
    assert [r["raw_response"] for r in output]==[r["raw_response"] for r in rows]
    assert [r["parsed_sign"] for r in output]==[r["parsed_sign"] for r in rows]
    assert [r["reused_source_sample_id"] for r in output]==[r["sample_id"] for r in rows]
    assert all(r["sample_id"]==request["sample_id"] for r,request in zip(output,requests))
    assert all("sample_seed" not in r for r in output)
    assert all(r["reused_source_sample_seed"]==old["sample_seed"] for r,old in zip(output,rows))
    assert sha(path)==expected[str(path)] and origin["reused_source_group"]==original
    requests[0]["prompt"]="different prompt"
    with pytest.raises(AssertionError):
        collector.initial_responses(group,requests,q,{"original_source_protocol_sha256":"original_sha"},expected,tmp_path)


def test_initial_pairing_and_graph_identity_are_required_in_scoring():
    q={"qid":"q0","graph_id":"Jf0"}
    graph=np.zeros((32,32),int).tolist()
    votes=np.ones((9,32,5),int)
    episode={"meta":{"qid":"q0","graph_id":"Jf0","name":"q0__Jf0__r00","split":"validation",
                    "n":32,"k":5,"steps":8,"replica":0,"protocol_sha256":"sha"},
        "spins_history":np.ones((9,32),int).tolist(),"spins_raw_history":votes.tolist(),"J":graph,
        "messages_history":[{} for _ in range(8)],"communicated_stances_history":[{} for _ in range(8)]}
    experiment.episode_arrays(episode,q,"sha",graph,votes[0])
    wrong=votes[0].copy();wrong[0,0]=-1
    with pytest.raises(AssertionError):experiment.episode_arrays(episode,q,"sha",graph,wrong)
    episode["spins_history"][1][0]=-1
    with pytest.raises(AssertionError):experiment.episode_arrays(episode,q,"sha",graph,votes[0])


def test_complete_collection_gate_runs_before_scoring(tmp_path,monkeypatch):
    monkeypatch.setattr(experiment,"validate",lambda root:{"questions":[{"qid":f"q{i}","graph_id":"Jf0"} for i in range(20)]})
    with pytest.raises(RuntimeError,match="Collection incomplete: 0/20"):
        experiment.evaluate(tmp_path)


def test_source_probability_reader_uses_all_next_rounds_and_records_retries(tmp_path):
    q={"qid":"q0","graph_id":"Jf0"}
    write(tmp_path/"source_protocol.json",{"synthetic":True})
    protocol_sha=sha(tmp_path/"source_protocol.json")
    votes=np.ones((9,32,5),int)
    for step in range(1,9):
        for start in range(0,32,4):
            group=f"q0__Jf0__r00|round{step}|opinions{start}"
            rows=[]
            for agent in range(start,start+4):
                for vote in range(5):
                    p=.1+.08*step
                    rows.append({"agent":agent,"vote_position":vote,"parse_attempt":0,"parsed_sign":1,
                        "raw_response":"A","first_literal_A_probability":p,"first_literal_B_probability":1-p})
            if step==1 and start==0:
                rows[0]["parsed_sign"]=None
                rows[0]["raw_response"]="undecided"
                retry_group=group+"|retry1"
                write(collector.chunk_path(tmp_path,retry_group),{"group_id":retry_group,"protocol_sha256":protocol_sha,
                    "responses":[{"agent":0,"vote_position":0,"parse_attempt":1,"parsed_sign":1}]})
            write(collector.chunk_path(tmp_path,group),{"group_id":group,"protocol_sha256":protocol_sha,"responses":rows})
    actual,quality=experiment.logged_probabilities(tmp_path,q,votes)
    p=.1+.08*np.arange(1,9)
    expected=np.repeat((p**3*(10-15*p+6*p*p))[:,None],32,axis=1)
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    assert quality["responses"]==1280 and quality["replacement_responses"]==1
    assert quality["first_attempt_unparsed"]==1 and quality["not_single_bare_tokens"]==1
