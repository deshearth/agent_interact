"""Subset isolation, nested balance, preprocessing and resumable training."""
import copy
import json

import numpy as np
import pytest
import torch

from mac_agents.common import config,json_write
from scripts import question_count_curve as curve
from scripts.argument_model import ResponseUpdate
from test_argument_response import fixture_cache


def test_nested_paths_balance_question_composition_and_share_endpoint():
    qids=[f"q{i:02d}" for i in range(16)]
    blocks,subsets=curve.balanced_subsets(qids)
    assert len(blocks)==4 and len(subsets)==13
    for path in range(4):
        one=[set(s["questions"]) for s in subsets if s["path"]==path]
        assert one[0]<one[1]<one[2]<set(qids)
    for n in (4,8,12):
        counts={q:sum(q in s["questions"] for s in subsets if s["n_questions"]==n) for q in qids}
        assert set(counts.values())=={n//4}
    assert sum(s["n_questions"]==16 for s in subsets)==1
    assert curve.balanced_subsets(qids)==curve.balanced_subsets(qids[::-1])


def test_pca_fits_only_chosen_questions():
    rng=np.random.default_rng(12)
    embeddings={f"q{i}":rng.normal(size=9) for i in range(8)}
    selected=["q0","q1","q2","q3"]
    mapped,mean,components=curve.fit_question_features(embeddings,selected,list(embeddings))
    changed=copy.deepcopy(embeddings)
    for q in set(embeddings)-set(selected):changed[q]+=1000*rng.normal(size=9)
    second,m2,c2=curve.fit_question_features(changed,selected,list(embeddings))
    np.testing.assert_array_equal(mean,m2)
    np.testing.assert_array_equal(components,c2)
    for q in selected:np.testing.assert_array_equal(mapped[q],second[q])
    assert not np.allclose(mapped["q7"],second["q7"])


def toy_cache():
    arrays,index,vectors,donors=fixture_cache()
    for i,(row,q) in enumerate(zip(index,["train_a","train_b","val_a","val_b"])):
        row.update(qid=q,episode_id=f"episode_{i}",graph_hash="fixture_graph",graph_family="random")
    return arrays,index,vectors,donors


def test_subsetting_keeps_complete_question_trajectories_and_fixed_validation():
    arrays,index,vectors,donors=toy_cache()
    small=curve.SubsetSnapshots(arrays,index,vectors,"train","full_arguments",donors,["train_a"])
    large=curve.SubsetSnapshots(arrays,index,vectors,"train","full_arguments",donors,["train_a","train_b"])
    v1=curve.SubsetSnapshots(arrays,index,vectors,"validation","full_arguments",donors,["train_a"])
    v2=curve.SubsetSnapshots(arrays,index,vectors,"validation","full_arguments",donors,["train_a","train_b"])
    assert small.pairs==[(0,0),(0,1)] and len(large)==4
    assert v1.pairs==v2.pairs==[(2,0),(2,1),(3,0),(3,1)]
    changed=copy.deepcopy(arrays)
    changed["spins"][1]*=-1
    changed["text_ids"][1]=0
    other=curve.SubsetSnapshots(changed,index,vectors,"train","full_arguments",donors,["train_a"])
    for i in range(len(small)):
        for key,value in small[i].items():
            if isinstance(value,torch.Tensor):torch.testing.assert_close(value,other[i][key],rtol=0,atol=0)
    with pytest.raises(ValueError,match="fixed split"):
        curve.SubsetSnapshots(arrays,index,vectors,"train","full_arguments",donors,["val_a"])


def test_update_based_validation_and_exact_cpu_midpass_resume(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    arrays,index,vectors,donors=toy_cache()
    metadata={"questions":{"validation":["val_a","val_b"]}}
    cache=(arrays,index,vectors,metadata)
    subset={"id":"fixture","questions":["train_a","train_b"],"feature_sha256":"fixture"}
    cfg=config(curve.CONFIG)
    cfg["training"].update(batch_size=2,max_steps=6,validation_every_steps=2,patience=99)
    item={"subset_id":"fixture","path":0,"n_questions":2,"condition":"full_arguments","seed":42}
    calls={"train":0,"cutoff":None}
    class Tiny(ResponseUpdate):
        def forward(self,batch):
            if self.training:calls["train"]+=1
            return super().forward(batch)
    monkeypatch.setattr(curve.original,"make_model",lambda cfg,kind:Tiny(text_dim=8,d_model=16,heads=2,d_k=4,d_v=8,ffn_dim=32,dropout=.1))
    monkeypatch.setattr(curve,"dataset_for",lambda cache,subset,cfg,partition,condition,seed:
                        curve.SubsetSnapshots(arrays,index,vectors,partition,condition,donors,subset["questions"]))
    monkeypatch.setattr(curve.time,"monotonic",lambda:2. if calls["cutoff"] is not None and calls["train"]>=calls["cutoff"] else 0.)
    full_root=tmp_path/"full";json_write(full_root/"protocol.json",{"synthetic":True})
    monkeypatch.setattr(curve,"ROOT",full_root)
    full_path,full=curve.train_run(item,subset,cfg,cache,torch.device("cpu"),1.)
    assert full["status"]=="complete"
    split_root=tmp_path/"split";json_write(split_root/"protocol.json",{"synthetic":True})
    monkeypatch.setattr(curve,"ROOT",split_root)
    calls.update(train=0,cutoff=3)
    split_path,partial=curve.train_run(item,subset,cfg,cache,torch.device("cpu"),1.)
    assert partial["status"]=="bounded" and partial["steps"]==3
    calls["cutoff"]=None
    _,resumed=curve.train_run(item,subset,cfg,cache,torch.device("cpu"),1.)
    assert resumed["status"]=="complete"
    a=torch.load(full_path/"checkpoint.pt",map_location="cpu",weights_only=False)
    b=torch.load(split_path/"checkpoint.pt",map_location="cpu",weights_only=False)
    for key in a["model"]:torch.testing.assert_close(a["model"][key],b["model"][key],rtol=0,atol=0)
    assert [x["step"] for x in a["history"]]==[2,4,6]
    assert a["history"]==b["history"]
