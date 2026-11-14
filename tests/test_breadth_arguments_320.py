"""Information-boundary, graph, optimization and recovery tests for arguments."""
import copy
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from mac_agents.common import json_write, seed_all
from mac_agents.models.graph import GraphUpdate
from scripts import breadth_argument_features as features
from scripts import breadth_argument_study as study

def fixture_batch(n=5,b=3):
    seed_all(23)
    s=torch.where(torch.rand(b,n)>.5,1.,-1.)
    context=torch.randn(b,n,16)
    graph=torch.zeros(b,n,n)
    for i in range(n-2): graph[:,i,i+1]=graph[:,i+1,i]=(-1.)**i
    text=torch.nn.functional.normalize(torch.randn(b,n,384),dim=-1)
    return s,context,graph,text

def test_parameter_counts_and_exact_zero_text_graph_equivalence():
    cfg=features.config()
    assert {w:sum(p.numel() for p in study.make_model(cfg,w).parameters()) for w in cfg['models']}=={'nano':3890,'full':184610}
    args=fixture_batch()
    for width in cfg['models']:
        model=study.make_model(cfg,width,45).eval()
        # Nonzero output prevents a vacuous equality from zero initialization.
        torch.nn.init.normal_(model.classifier[-1].weight)
        old=GraphUpdate(kind='A2',dropout=cfg['training']['dropout'],**cfg['models'][width]).eval()
        old.load_state_dict({k:v for k,v in model.state_dict().items() if k!='text_projection.weight'})
        expected=old(*args[:3],torch.empty((*args[0].shape,0)))
        actual=model(*args[:3],torch.zeros_like(args[3]))
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)

def source_fixture():
    return {'J':[[0,1,0],[1,0,-1],[0,-1,0]],'spins_history':[[1,-1,1],[-1,1,-1],[1,1,1]],
        'communicated_stances_history':[{'0':1,'1':-1,'2':1},{'0':-1,'1':1,'2':-1}],
        'messages_history':[{'0,1':'A because first','1,0':'B because second','1,2':'B because second','2,1':'A because third'},
                            {'0,1':'later A','1,0':'later B','1,2':'later B','2,1':'later C'}],
        'spins_raw_history':[[[1]*5]*3]*3}

def test_message_inputs_exclude_future_votes_and_later_messages():
    source=source_fixture();before=features.snapshot_messages(source,0)
    changed=copy.deepcopy(source)
    changed['spins_history'][1:]=[[999]*3]*2
    changed['spins_raw_history']='not a predictor input'
    changed['messages_history'][1]={'irrelevant':'future'}
    changed['solution']='not a predictor input';changed['answer']='not a predictor input'
    assert features.snapshot_messages(changed,0)==before
    broken=copy.deepcopy(source);broken['communicated_stances_history'][0]['0']=-1
    with pytest.raises(AssertionError,match='Stance'):features.snapshot_messages(broken,0)
    broken=copy.deepcopy(source);broken['messages_history'][0].pop('0,1')
    with pytest.raises(AssertionError,match='Missing'):features.snapshot_messages(broken,0)
    broken=copy.deepcopy(source);broken['messages_history'][0]['1,2']='different recipient message'
    with pytest.raises(AssertionError,match='broadcast'):features.snapshot_messages(broken,0)

def test_one_hop_masking_isolation_and_agent_permutation():
    cfg=features.config();model=study.make_model(cfg,'nano',41).eval()
    torch.nn.init.normal_(model.classifier[-1].weight)
    args=fixture_batch();z,social,_=model(*args,return_social=True)
    assert torch.isfinite(z).all();assert torch.equal(social[:,-1],torch.zeros_like(social[:,-1]))
    changed=[a.clone() for a in args]
    changed[0][:,2:]*=-1;changed[1][:,2:]+=100;changed[3][:,2:]*=-7
    torch.testing.assert_close(model(*changed)[:,0],z[:,0],atol=1e-6,rtol=1e-6)
    # Receiver 0's own generated message is not delivered back to itself.
    changed=[a.clone() for a in args];changed[3][:,0]*=-11
    torch.testing.assert_close(model(*changed)[:,0],z[:,0],atol=1e-6,rtol=1e-6)
    order=torch.tensor([3,1,4,0,2])
    permuted=(args[0][:,order],args[1][:,order],args[2][:,order][:,:,order],args[3][:,order])
    torch.testing.assert_close(model(*permuted),z[:,order],atol=1e-5,rtol=1e-5)

def test_text_branch_receives_gradients_and_updates():
    cfg=features.config();model=study.make_model(cfg,'nano',71)
    args=fixture_batch();target=torch.zeros_like(args[0]);opt=torch.optim.AdamW(model.parameters(),lr=.01)
    initial=model.text_projection.weight.detach().clone()
    for _ in range(3):
        opt.zero_grad();z=model(*args);loss=torch.nn.functional.cross_entropy(z.reshape(-1,2),target.long().flatten());loss.backward()
        assert torch.isfinite(model.text_projection.weight.grad).all();opt.step()
    assert model.text_projection.weight.grad.abs().sum()>0
    assert not torch.equal(model.text_projection.weight,initial)
    model.eval();assert not torch.allclose(model(*args),model(*args[:3],torch.zeros_like(args[3])))

class ToyData:
    def __init__(self):
        s,c,j,t=fixture_batch(b=8)
        self.values={'s':s,'context':c,'J':j,'text':t,'target':((s.roll(1,-1)>0)).float()}
        self.y=self.values['target'].numpy().astype(np.int64)
        self.qids=np.array(['t0','t0','t1','t1','v0','v0','v1','v1'])
        self.device='cpu';self.condition='arguments'
    def ids(self,qs):return np.flatnonzero(np.isin(self.qids,qs))
    def batch(self,ids,condition=None):return {k:v[ids] for k,v in self.values.items()}

def test_training_checkpoint_resume_matches_uninterrupted(tmp_path,monkeypatch):
    cfg=features.config();cfg['output']=str(tmp_path)
    cfg['training'].update(batch_size=2,evaluation_batch_size=4,validation_every_steps=2,max_steps=6,patience=100)
    cfg['training']['learning_rates']=[.003];cfg['training']['output_penalties']=[0.]
    protocol={'fold':{'train_questions':['t0','t1'],'validation_questions':['v0','v1'],'assessment_questions':[]}}
    json_write(tmp_path/'protocol.json',protocol);json_write(tmp_path/'encoding.json',{});json_write(tmp_path/'selection.json',{})
    item={'phase':'refit','width':'nano','condition':'arguments','candidate':'c0','seed':142}
    real_clock=study.time.time;clock={'calls':0}
    def ticking():
        clock['calls']+=1
        return 0. if clock['calls']<=3 else 2.
    monkeypatch.setattr(study.time,'time',ticking)
    directory,partial=study.train_one(cfg,protocol,ToyData(),item,deadline=1.,fixed_steps=6)
    assert partial['status']=='bounded'
    partial_state=torch.load(directory/'resume.pt',weights_only=False)
    assert partial_state['state']['step']==3
    monkeypatch.setattr(study.time,'time',real_clock)
    directory,done=study.train_one(cfg,protocol,ToyData(),item,time.time()+60,fixed_steps=6)
    assert done['status']=='complete'
    weights=torch.load(directory/'best.pt',weights_only=False)['model']
    directory.rename(directory.with_name(directory.name+'-resumed'))
    fresh,finished=study.train_one(cfg,protocol,ToyData(),item,time.time()+60,fixed_steps=6)
    other=torch.load(fresh/'best.pt',weights_only=False)['model']
    for key in weights:torch.testing.assert_close(weights[key],other[key],atol=0,rtol=0)
    assert done['train']==finished['train']

def test_bootstrap_preserves_question_pairing_and_seed_averaging():
    cfg=features.config();cfg['evaluation']['bootstrap_replicates']=100
    names=['nano_arguments','nano_conclusion','full_arguments','full_conclusion','opinion_nano','I0']
    rows=[]
    for name in names:
        for seed in [1,2,3]:
            for q in ['a','b','c']:
                value=.2 if name.endswith('arguments') else .3
                rows.append({'model':name,'seed':seed,'qid':q,'split':'validation','nll':value,'error':value,'brier_two_class':value})
    comparisons=study.contrasts(rows,cfg)
    first=next(r for r in comparisons if r['left']=='nano_arguments' and r['right']=='nano_conclusion' and r['metric']=='error')
    assert first['mean_difference']==pytest.approx(-.1)
    assert first['ci95']==pytest.approx([-.1,-.1])

def test_full_synthetic_tuning_refit_freeze_and_input_controls(tmp_path,monkeypatch):
    cfg=features.config();cfg['output']=str(tmp_path)
    cfg['training'].update(batch_size=4,evaluation_batch_size=16,validation_every_steps=2,max_steps=4,
                           patience=5,learning_rates=[.003],output_penalties=[0.],refit_seeds=[142])
    root=tmp_path/'data';root.mkdir()
    rng=np.random.default_rng(74);s,c,j,t=fixture_batch(b=4)
    states=np.stack([s.numpy()*((-1)**k) for k in range(9)],axis=1)
    pairs=np.array([(e,k) for e in range(4) for k in range(8)])
    np.savez(root/'development.npz',spins=states,J=j.numpy(),pairs=pairs,episode_qids=np.array(['t0','t1','v0','v1']))
    np.savez(root/'features.npz',context=c.numpy())
    ids=rng.integers(3,20,(4,8,5));np.save(root/'message_ids.npy',ids)
    vectors=rng.normal(size=(20,384)).astype(np.float32);vectors/=np.linalg.norm(vectors,axis=1,keepdims=True);vectors[0]=0
    np.save(root/'vectors.npy',vectors)
    protocol={'fold':{'train_questions':['t0','t1'],'validation_questions':['v0','v1'],'assessment_questions':['unread']}}
    for name,obj in [('protocol.json',protocol),('encoding.json',{}),('prepared.json',{})]:json_write(tmp_path/name,obj)
    monkeypatch.setattr(features,'validate',lambda *args,**kwargs:protocol)
    data=study.Data(cfg,'cpu');idx=np.arange(3)
    control=data.batch(idx,'conclusion')['text'].clone();original=data.batch(idx,'arguments')['text'].clone()
    data.ids_text[:]=3
    assert torch.equal(control,data.batch(idx,'conclusion')['text'])
    assert not torch.equal(original,data.batch(idx,'arguments')['text'])
    study.fit(cfg,'cpu');frozen=study.validate_frozen(cfg)
    assert frozen['tuning_fits']==frozen['refits_completed']==4
    selection=study.read(tmp_path/'selection.json')
    assert len(selection['choices'])==4
    for choice in selection['choices']:assert choice['refit_steps'] in [0,2,4]
    for directory in frozen['refits']:
        manifest=study.read(Path(directory)/'manifest.json')
        assert manifest['train']['transitions']==80
        assert manifest['parameters'] in [3890,184610]

def test_cpu_mps_agreement_when_available():
    if not torch.backends.mps.is_available():pytest.skip('MPS unavailable in this process')
    cfg=features.config();args=fixture_batch()
    for width in cfg['models']:
        cpu=study.make_model(cfg,width,17).eval();torch.nn.init.normal_(cpu.classifier[-1].weight,std=.1)
        gpu=copy.deepcopy(cpu).to('mps')
        a=cpu(*args).softmax(-1);b=gpu(*(x.to('mps') for x in args)).softmax(-1).cpu()
        torch.testing.assert_close(a,b,atol=5e-5,rtol=1e-5)
