"""Scientific invariants for the separate delivered-message diagnostic."""
import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.argument_features import (ResponseSnapshots, matched_donors, neighbor_slots,
                                       parse_stance, response_ids, shuffle_groups)
from scripts.argument_model import ResponseUpdate


def fixture_cache():
    rng = np.random.default_rng(12)
    graph = np.array([[0, 1, -1, 0], [1, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
    neighbors, signs = neighbor_slots(np.repeat(graph[None], 4, axis=0))
    spins = np.tile(np.array([[-1, 1, -1, 1], [1, -1, 1, -1], [-1, -1, -1, -1]]), (4, 1, 1))
    ids = np.zeros((4, 2, 4, neighbors.shape[-1]), dtype=np.int32)
    stance = np.zeros_like(ids, dtype=np.int8)
    for e, t, receiver, slot in np.ndindex(ids.shape):
        if signs[e, receiver, slot]:
            ids[e, t, receiver, slot] = 1+np.ravel_multi_index((e,t,receiver,slot), ids.shape)
            stance[e, t, receiver, slot] = spins[e, t, neighbors[e, receiver, slot]]
    arrays = {"spins": spins, "context": rng.normal(size=(4,4,16)).astype(np.float32),
              "J": np.repeat(graph[None],4,axis=0), "valid": np.ones((4,2),dtype=bool),
              "neighbors":neighbors, "signs":signs, "stance":stance, "text_ids":ids}
    index = [{"qid":"q", "question_split": "train" if e<2 else "validation", "graph_role":"fit_graph"} for e in range(4)]
    vectors = rng.normal(size=(ids.size+1, 8)).astype(np.float32)
    vectors[0] = 0
    donors = matched_donors(shuffle_groups(arrays,index,ids,stance,neighbors,signs), ids.shape,42)
    return arrays,index,vectors,donors


def test_delivered_message_timing_and_no_future_opinion_inputs():
    raw = {"J": [[0,1],[1,0]], "messages_history": [
        {"0,1":"I favor option A because first.","1,0":"I favor option B because first."},
        {"0,1":"I favor option B because second.","1,0":"I favor option A because second."}]}
    neighbors,signs=neighbor_slots(np.array(raw["J"])[None])
    lookup={text:i+1 for i,text in enumerate(m for row in raw["messages_history"] for m in row.values())}
    ids,stance=response_ids(raw,neighbors[0],signs[0],lookup.__getitem__)
    assert ids[0,1,0]==lookup[raw["messages_history"][0]["0,1"]]
    assert ids[1,1,0]==lookup[raw["messages_history"][1]["0,1"]]
    assert stance[:,1,0].tolist()==[1,-1]
    arrays,index,vectors,donors=fixture_cache()
    before=ResponseSnapshots(arrays,index,vectors,"train","full_arguments",donors)[0]
    changed=copy.deepcopy(arrays)
    changed["spins"][:,1:]*=-1
    changed["text_ids"][:,1:]=0
    changed["stance"][:,1:]*=-1
    after=ResponseSnapshots(changed,index,vectors,"train","full_arguments",donors)[0]
    assert not torch.equal(before["target"],after["target"])
    for key in ("s","context","history","neighbors","signs","stance","text"):
        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match="development-only"):
        ResponseSnapshots(arrays,index,vectors,"test","full_arguments",donors)


def test_shuffle_preserves_cues_partitions_and_ignores_targets():
    arrays,index,vectors,donors=fixture_cache()
    ids=arrays["text_ids"]
    valid=np.flatnonzero(ids)
    np.testing.assert_array_equal(np.sort(donors.reshape(-1)[valid]),valid)
    assert (donors.reshape(-1)[valid]!=valid).all()
    for flat in valid:
        e,t,r,k=np.unravel_index(flat,ids.shape)
        de,dt,dr,dk=np.unravel_index(donors.reshape(-1)[flat],ids.shape)
        assert e!=de and t==dt and r==dr
        assert index[e]["question_split"]==index[de]["question_split"]
        assert index[e]["qid"]==index[de]["qid"]
        assert arrays["neighbors"][e,r,k]==arrays["neighbors"][de,dr,dk]
        assert arrays["signs"][e,r,k]==arrays["signs"][de,dr,dk]
        assert arrays["stance"][e,t,r,k]==arrays["stance"][de,dt,dr,dk]
        assert arrays["spins"][e,t,r]==arrays["spins"][de,dt,dr]
    changed=copy.deepcopy(arrays)
    changed["spins"][:,-1]*=-1
    other=matched_donors(shuffle_groups(changed,index,ids,arrays["stance"],arrays["neighbors"],arrays["signs"]),ids.shape,42)
    np.testing.assert_array_equal(donors,other)
    # A one-member stratum cannot be changed; its padded and real slots remain fixed.
    singleton=matched_donors([np.array([1])],(3,),42)
    np.testing.assert_array_equal(singleton,np.arange(3))


def test_stance_only_ignores_text_and_preserves_every_other_feature():
    arrays,index,vectors,donors=fixture_cache()
    full=ResponseSnapshots(arrays,index,vectors,"train","full_arguments",donors)[0]
    stance=ResponseSnapshots(arrays,index,vectors,"train","stance_only",donors)[0]
    assert full["text"].abs().sum()>0 and not stance["text"].any()
    for key in full:
        if key=="text": continue
        if isinstance(full[key],torch.Tensor):
            torch.testing.assert_close(full[key],stance[key],rtol=0,atol=0)
        else: assert full[key]==stance[key]
    assert parse_stance("I favor option A because B is unlikely.")==1
    assert parse_stance("I support option B because it works.")==-1
    with pytest.raises(ValueError): parse_stance("Possibly either choice.")


@pytest.mark.parametrize("kind",["A0","A2"])
def test_message_locality_padding_isolation_and_permutation(kind):
    arrays,index,vectors,donors=fixture_cache()
    dataset=ResponseSnapshots(arrays,index,vectors,"train","full_arguments",donors)
    batch=next(iter(DataLoader(dataset,batch_size=2)))
    torch.manual_seed(7)
    model=ResponseUpdate(kind,text_dim=8,dropout=0).eval()
    with torch.no_grad():
        original=model(batch)
        altered={k:v.clone() for k,v in batch.items()}
        altered["text"][:,0]+=4
        predicted=model(altered)
        assert not torch.allclose(original[:,0],predicted[:,0])
        torch.testing.assert_close(original[:,1:],predicted[:,1:],rtol=0,atol=0)
        padding={k:v.clone() for k,v in batch.items()}
        padding["text"][padding["signs"]==0]=123
        torch.testing.assert_close(original,model(padding),rtol=0,atol=0)
        assert torch.isfinite(original).all()
        permutation=torch.tensor([2,0,3,1]); inverse=torch.argsort(permutation)
        reordered={k:(v[:,permutation].clone() if v.ndim>=2 and v.shape[1]==4 else v.clone()) for k,v in batch.items()}
        reordered["neighbors"]=inverse[reordered["neighbors"]]
        torch.testing.assert_close(model(reordered),original[:,permutation],rtol=1e-5,atol=1e-6)
        # The target labels are never consumed by forward.
        reordered["target"]*=0
        torch.testing.assert_close(model(reordered),original[:,permutation],rtol=1e-5,atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason="MPS is not exposed in this environment")
def test_argument_cpu_mps_agreement():
    arrays,index,vectors,donors=fixture_cache()
    dataset=ResponseSnapshots(arrays,index,vectors,"train","full_arguments",donors)
    batch=next(iter(DataLoader(dataset,batch_size=2)))
    model=ResponseUpdate(text_dim=8,dropout=0).eval()
    with torch.no_grad():
        expected=model(batch)
        actual=model.to("mps")({k:v.to("mps") for k,v in batch.items()}).cpu()
    torch.testing.assert_close(expected,actual,rtol=1e-4,atol=1e-5)
