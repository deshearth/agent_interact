"""Non-vacuous model/input-boundary checks for compact argument adapters."""
import copy
import numpy as np
import torch
from mac_agents.models.graph import GraphUpdate
from scripts import compact_argument_study as study
from mac_agents.common import json_write

def batch(text_dim):
    torch.manual_seed(9)
    s=torch.where(torch.rand(3,5)>.5,1.,-1.)
    c=torch.randn(3,5,16);j=torch.zeros(3,5,5)
    for i in range(3):j[:,i,i+1]=j[:,i+1,i]=(-1.)**i
    text=torch.randn(3,5,text_dim)/text_dim**.5
    return s,c,j,text

def test_counts_and_zero_text_matches_original_nonzero_output():
    cfg=study.config()
    for width,m in cfg['models'].items():
        model=study.make_model(cfg,width,42).eval()
        assert sum(p.numel() for p in model.parameters())==cfg['expected_parameters'][width]
        torch.nn.init.normal_(model.classifier[-1].weight)
        old=GraphUpdate(kind='A2',dropout=.1,**{k:v for k,v in m.items() if k!='text_dim'}).eval()
        old.load_state_dict({k:v for k,v in model.state_dict().items() if k!='text_projection.weight'})
        s,c,j,t=batch(m['text_dim'])
        torch.testing.assert_close(model(s,c,j,torch.zeros_like(t)),old(s,c,j,torch.empty(3,5,0)),atol=1e-6,rtol=1e-6)

def test_pca_training_boundary_and_zero_padding():
    rng=np.random.default_rng(18);x=rng.normal(size=(40,384));x/=np.linalg.norm(x,axis=1,keepdims=True);x[0]=0
    ids=np.array([3,4,5,6,7,8,9,10,11,12,12])
    a,fit=study.fit_message_pca(x,ids)
    changed=x.copy();changed[13:]=rng.normal(size=changed[13:].shape)
    b,other=study.fit_message_pca(changed,ids)
    for key in fit:np.testing.assert_array_equal(fit[key],other[key])
    np.testing.assert_array_equal(a[:13],b[:13])
    np.testing.assert_array_equal(a[0],0)
    np.testing.assert_allclose((a[np.unique(ids)]**2).mean(),1/8,rtol=1e-6)

def test_substitution_preserves_sender_question_stance_without_targets():
    ids=np.arange(1,49).reshape(2,8,3);states=np.ones_like(ids);states[:,4:,1]=-1
    result,donors,audit=study.matched_round_substitution(ids,states,93)
    for e in range(2):
        for s in range(3):
            np.testing.assert_array_equal(result[e,:,s],ids[e,donors[e,:,s],s])
            np.testing.assert_array_equal(states[e,:,s],states[e,donors[e,:,s],s])
            assert set(result[e,:,s])==set(ids[e,:,s])
    assert audit['changed_fraction']==1 and audit['eligible_fraction']==1

def test_text_affects_only_linked_receivers_and_has_trainable_gradient():
    cfg=study.config()
    for width,m in cfg['models'].items():
        model=study.make_model(cfg,width,42).eval();torch.nn.init.normal_(model.classifier[-1].weight)
        s,c,j,t=batch(m['text_dim']);a=model(s,c,j,t)
        altered=t.clone();altered[:,0]*=-3;b=model(s,c,j,altered)
        torch.testing.assert_close(a[:,[0,2,3,4]],b[:,[0,2,3,4]],atol=0,rtol=0)
        assert not torch.allclose(a[:,1],b[:,1])
        torch.nn.functional.cross_entropy(a.reshape(-1,2),(s>0).long().flatten()).backward()
        assert model.text_projection.weight.grad.norm()>0
        perm=torch.tensor([4,2,0,3,1])
        torch.testing.assert_close(model(s[:,perm],c[:,perm],j[:,perm][:,:,perm],t[:,perm]),a[:,perm],atol=1e-6,rtol=1e-6)

def test_shared_adapter_leaves_original_factory_unchanged():
    original=study.shared.make_model
    with study.adapted_trainer():assert study.shared.make_model is study.make_model
    assert study.shared.make_model is original

def test_end_to_end_adapter_tuning_refit_and_frozen_artifacts(tmp_path,monkeypatch):
    cfg=study.config();cfg['output']=str(tmp_path/'study');cfg['message_reference']=str(tmp_path/'messages')
    cfg['training'].update(max_steps=4,validation_every_steps=2,batch_size=4,evaluation_batch_size=8,
        learning_rates=[.003],output_penalties=[0.],refit_seeds=[142],patience=5)
    root=tmp_path/'study';folder=root/'data';folder.mkdir(parents=True)
    source=tmp_path/'messages'/'data';source.mkdir(parents=True)
    rng=np.random.default_rng(8);spins=rng.choice([-1,1],size=(4,9,5));graph=np.ones((4,5,5),dtype=np.float32)
    for j in range(5):graph[:,j,j]=0
    np.savez(folder/'development.npz',spins=spins,J=graph,pairs=np.array([(e,t) for e in range(4) for t in range(8)]),
             episode_qids=np.array(['t0','t1','v0','v1']))
    np.savez(folder/'features.npz',context=rng.normal(size=(4,5,16)).astype(np.float32))
    vectors=rng.normal(size=(30,384)).astype(np.float32);vectors/=np.linalg.norm(vectors,axis=1,keepdims=True);vectors[0]=0
    np.save(source/'vectors.npy',vectors);ids=rng.integers(3,30,size=(4,8,5));np.save(folder/'message_ids.npy',ids)
    np.save(folder/'vectors_pca8.npy',study.fit_message_pca(vectors,ids[:2])[0])
    np.savez(folder/'matched_substitution.npz',message_ids=ids[:,::-1])
    protocol={'fold':{'train_questions':['t0','t1'],'validation_questions':['v0','v1'],'assessment_questions':['unread']},
              'parameters':cfg['expected_parameters']}
    for name,obj in [('protocol.json',protocol),('prepared.json',{}),('encoding.json',{})]:json_write(root/name,obj)
    monkeypatch.setattr(study,'validate',lambda config:protocol)
    data=study.Data(cfg,'cpu');original=study.shared.make_model
    for width,model_cfg in cfg['models'].items():
        data.set_width(width)
        for condition in ('arguments','conclusion','zero','matched'):
            assert data.batch([0,1],condition)['text'].shape==(2,5,model_cfg['text_dim'])
        a=data.batch([0,1],'conclusion')['text'].clone();data.ids_text[0]=3
        torch.testing.assert_close(a,data.batch([0,1],'conclusion')['text'])
    study.fit(cfg,'cpu');frozen=study.validate_frozen(cfg)
    assert frozen['tuning_fits']==frozen['refits_completed']==6
    assert study.shared.make_model is original
    for path in frozen['refits']:
        manifest=study.read(study.Path(path)/'manifest.json')
        assert manifest['parameters']==cfg['expected_parameters'][manifest['width']]
        assert manifest['selected_step'] in (0,2,4)
