"""Expansion guards use metadata and synthetic responses, with no LLM calls."""
from collections import Counter
import copy
import time

import numpy as np
import pytest

from scripts import breadth_expand as study
from scripts import hpc3_breadth_expand as collector
from scripts import hpc3_parallel_collect as parallel
from scripts.hpc3_prepare import read,sha,write


@pytest.fixture
def planned():
    settings=study.cfg()
    parents=read(study.PARENT/"selection.json")["questions"]
    quotas=study.make_quotas(parents,settings)
    selected=[]
    for topic,labels in quotas["new_selected"].items():
        for label,n in labels.items():
            for i in range(n):
                selected.append({"qid":f"new{len(selected):04d}","type":topic,"answer":label,
                    "level":f"Level {i%5+1}","screen":{"pooled_entropy_bits":.98+.001*(i%20),
                    "within_agent_entropy_bits":.8+.002*(i%20)}})
    return settings,parents,quotas,selected


def test_quotas_retain_original_split_counts_and_select_one_in_four(planned):
    settings,parents,quotas,selected=planned
    assert Counter(q["split"] for q in parents)=={"train":80,"validation":20,"assessment":40}
    assert len(selected)==300
    for group,total in [("development",400),("validation",80),("new_selected",300),("new_validation",60),("candidates",1200)]:
        cells=quotas[group]
        assert sum(sum(labels.values()) for labels in cells.values())==total
        assert {a:sum(labels[a] for labels in cells.values()) for a in ["A","B"]}=={"A":total//2,"B":total//2}
    for t in quotas["candidates"]:
        for a in ["A","B"]:assert quotas["candidates"][t][a]==4*quotas["new_selected"][t][a]
    for group in ["development","validation"]:
        counts=[sum(labels.values()) for labels in quotas[group].values()]
        assert max(counts)-min(counts)==1


def test_new_splits_are_reproducible_balanced_and_preserve_old_assignments(planned):
    settings,parents,quotas,selected=planned
    original=copy.deepcopy(parents)
    first,diagnostic=study.choose_split(selected,parents,quotas,settings)
    second,again=study.choose_split(selected,parents,quotas,settings)
    assert first==second and diagnostic==again and parents==original
    assert Counter(q["split"] for q in first)=={"train":240,"validation":60}
    assert not {q["qid"] for q in first}&{q["qid"] for q in parents}
    for split,n in [("train",320),("validation",80)]:
        full=[q for q in parents+first if q["split"]==split]
        assert len(full)==n
        assert Counter(q["answer"] for q in full)=={"A":n//2,"B":n//2}
        assert Counter(q["graph_id"] for q in full)=={f"J{i}":n//8 for i in range(8)}


def test_workers_partition_questions_once_and_never_collect_assessment():
    bank=[{"qid":f"q{i:04d}"} for i in range(1200)]
    screen=collector.assignments(bank,8,"screen")
    assert [len(part) for part in screen]==[150]*8
    assert len({q["qid"] for part in screen for q in part})==1200
    selected=[{"qid":f"q{i:04d}","split":"train" if i<240 else "validation"} for i in range(300)]
    collect=collector.assignments(selected,8,"collect")
    assert len({q["qid"] for part in collect for q in part})==300
    for part in collect:
        assert sum(q["split"]=="train" for q in part)==30
        assert sum(q["split"]=="validation" for q in part) in [7,8]
    selected[-1]["split"]="assessment"
    with pytest.raises(AssertionError):collector.assignments(selected,8,"collect")


class FakeSource:
    def __init__(self,execution):self.calls=0;self.execution=execution
    def generate_many(self,prompts,sample_ids,batch_size,max_tokens):
        self.calls+=1
        return {"hpc3_execution":self.execution,"responses":[{"sample_id":sid,"raw_response":"1","finish_reason":"stop"} for sid in sample_ids]}


def test_construction_cache_is_unique_resumable_and_keeps_sampling_ids(tmp_path,monkeypatch):
    root=tmp_path/"worker";canonical=tmp_path/"canonical"
    write(root/"source_protocol.json",{"synthetic":True})
    monkeypatch.setattr(collector,"ROOT",canonical)
    execution={"runtime":"fixed"};source=FakeSource(execution)
    recorder=collector.make_recorder(source,root,time.time()+60,[{"qid":f"q{i}"} for i in range(8)],"screen",execution)
    groups=[]
    for start in [0,4]:
        requests=[{"sample_id":f"construct|q{i}","prompt":f"prompt{i}","metadata":{"qid":f"q{i}"}} for i in range(start,start+4)]
        rows=recorder.request("construct|group0",requests,64)
        assert [r["sample_id"] for r in rows]==[r["sample_id"] for r in requests]
        groups.append(requests)
    assert source.calls==2 and len(list((canonical/"chunks").glob("*.json")))==2
    recorder.request("construct|group0",groups[0],64)
    assert source.calls==2
    changed=collector.make_recorder(source,root,time.time()+60,[{"qid":f"q{i}"} for i in range(8)],"screen",{"runtime":"changed"})
    with pytest.raises(AssertionError,match="Changed execution"):
        changed.request("construct|group0",groups[0],64)
    with pytest.raises(ValueError,match="another worker"):
        recorder.request("construct|group0",[{"metadata":{"qid":"reserved"}}],64)
    with pytest.raises(ValueError,match="Interaction"):
        recorder.request("q0__J0__r00|round0|opinions0",groups[0])


@pytest.mark.parametrize("origin",["retained","new"])
def test_episode_audit_checks_own_protocol_majority_and_graph(tmp_path,monkeypatch,origin):
    parent=tmp_path/"parent";root=tmp_path/"new"
    write(parent/"source_protocol.json",{"old":True});write(root/"source_protocol.json",{"new":True})
    monkeypatch.setattr(study,"ROOT",root);monkeypatch.setattr(study,"PARENT",parent)
    q={"qid":"q0","graph_id":"J0","split":"train","origin":origin}
    protocol=parent if origin=="retained" else root
    graph=np.zeros((32,32),int).tolist()
    episode={"meta":{"qid":"q0","graph_id":"J0","split":"train","name":"q0__J0__r00","n":32,"k":5,"steps":8,"replica":0,"protocol_sha256":sha(protocol/"source_protocol.json")},
        "spins_history":np.ones((9,32),int).tolist(),"spins_raw_history":np.ones((9,32,5),int).tolist(),
        "messages_history":[{}]*8,"communicated_stances_history":[{}]*8,"J":graph}
    path=tmp_path/"episode.json";write(path,episode)
    result=study.episode_summary(q,path,{"J0":graph})
    assert result["valid_snapshots"]==8 and result["invalid_votes"]==0
    episode["spins_history"][1][0]=-1;write(path,episode)
    with pytest.raises(AssertionError):study.episode_summary(q,path,{"J0":graph})
    episode["spins_history"][1][0]=1;episode["meta"]["protocol_sha256"]="wrong";write(path,episode)
    with pytest.raises(AssertionError):study.episode_summary(q,path,{"J0":graph})
    with pytest.raises(AssertionError):study.episode_summary({**q,"split":"assessment"},path,{"J0":graph})
