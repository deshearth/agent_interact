"""Scientific boundaries and stochastic-resume checks for nested comparison."""
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from mac_agents.common import config, digest, seed_all
from mac_agents.models.baselines import design
from scripts.transformer_generalization import (
    CONFIG, FoldData, fit_context, fold_dataset, logits, loss_for, make_folds,
    make_model, metrics, model_weights_sha, predictions, train_one,
)


def test_nested_folds_hold_entire_questions_apart_and_cover_assessment_once():
    questions = [f"q{i:02d}" for i in range(20)]
    folds = make_folds(questions, 81943)
    held = []
    for outer in range(4):
        refit = next(f for f in folds if f['id'] == f'o{outer}_refit')
        held += refit['assessment_questions']
        validation = []
        for fold in (f for f in folds if f['outer'] == outer and f['phase'] == 'tune'):
            a,b,c = (set(fold[k]) for k in ('train_questions','validation_questions','assessment_questions'))
            assert [len(a),len(b),len(c)] == [10,5,5]
            assert not a&b and not a&c and not b&c
            assert a|b == set(refit['train_questions'])
            validation += list(b)
        assert sorted(validation) == sorted(refit['train_questions'])
    assert sorted(held) == questions
    assert folds == make_folds(list(reversed(questions)),81943)


def test_heldout_embeddings_cannot_change_fitted_pca_or_training_features():
    rng = np.random.default_rng(3)
    q = {f'q{i}':rng.normal(size=12) for i in range(20)}
    persona = rng.normal(size=(32,12))
    training = list(q)[:10]
    ids = np.array(list(q))
    first = fit_context(q,persona,training,ids)
    changed = {k:v.copy() for k,v in q.items()}
    for k in set(q)-set(training): changed[k] = changed[k]*1000+17
    second = fit_context(changed,persona,training,ids)
    for k in ('question_mean','question_components','persona_mean','persona_components'):
        np.testing.assert_array_equal(first[k],second[k])
    np.testing.assert_array_equal(first['context'][:10],second['context'][:10])
    assert not np.array_equal(first['context'][10:],second['context'][10:])


def tiny_data(device='cpu'):
    rng = np.random.default_rng(9)
    spins = rng.choice([-1,1],size=(3,9,4)).astype(np.int8)
    J = np.ones((3,4,4),dtype=np.float32)-np.eye(4,dtype=np.float32)[None]
    J[:,0,1] = J[:,1,0] = -1
    common = {'spins':spins,'J':J,'valid':np.ones((3,8),dtype=bool),
              'episode_qids':np.array(['train','validation','assessment']),
              'pairs':np.array([(e,t) for e in range(3) for t in range(8)])}
    features = {'context':rng.normal(size=(3,4,16)).astype(np.float32),
                'prior_logit':rng.normal(size=(24,4)).astype(np.float64)}
    return common,features,FoldData(common,features,device)


def test_heldout_labels_do_not_enter_baseline_fit_or_current_inputs():
    common,feat,data = tiny_data()
    ds = fold_dataset(common,feat['context'],['train'])
    e,t = np.array(ds.pairs).T
    x = design(ds.arrays['context'][e],ds.arrays['J'][e],ds.arrays['spins'][e,t])
    y = ds.arrays['spins'][e,t+1].copy()
    common['spins'][1:] *= -1
    altered = fold_dataset(common,feat['context'],['train'])
    e2,t2 = np.array(altered.pairs).T
    np.testing.assert_array_equal(x,design(altered.arrays['context'][e2],altered.arrays['J'][e2],altered.arrays['spins'][e2,t2]))
    np.testing.assert_array_equal(y,altered.arrays['spins'][e2,t2+1])
    # A batch at t=0 consumes current spins only, even when its next label changes.
    before = data.batch(np.array([0]))
    common['spins'][0,1] *= -1
    after = FoldData(common,feat,'cpu').batch(np.array([0]))
    for k in ('s','context','J','prior'):
        torch.testing.assert_close(before[k],after[k])
    assert not torch.equal(before['target'],after['target'])


def test_small_models_have_identical_weights_and_hybrid_starts_exactly_at_ising():
    cfg = config(CONFIG)
    seed_all(42); plain = make_model(cfg,'small')
    seed_all(42); hybrid = make_model(cfg,'ising_small')
    assert model_weights_sha(plain) == model_weights_sha(hybrid)
    assert sum(p.numel() for p in plain.parameters()) == 34978
    _,_,data = tiny_data()
    batch = data.batch(np.array([0,1]))
    z,delta = logits(hybrid,batch,True)
    torch.testing.assert_close(delta,torch.zeros_like(delta))
    torch.testing.assert_close(z.sigmoid(),batch['prior'].sigmoid())
    z0,_ = logits(plain,batch,False)
    torch.testing.assert_close(z0.sigmoid(),torch.full_like(z0,.5))


def test_binary_objective_matches_original_softmax_and_regularizes_correction():
    torch.manual_seed(3)
    raw = torch.randn(3,4,2,requires_grad=True)
    target = torch.randint(0,2,(3,4))
    delta = raw[...,1]-raw[...,0]
    actual = loss_for(delta,delta,target.float(),0.)
    expected = F.cross_entropy(raw.reshape(-1,2),target.reshape(-1))
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,raw,retain_graph=True)[0],
                               torch.autograd.grad(expected,raw,retain_graph=True)[0])
    torch.testing.assert_close(loss_for(delta,delta,target.float(),.01)-actual,.01*delta.square().mean())


def test_question_metrics_do_not_weight_large_questions_more():
    p = np.array([[.9],[.9],[.9],[.2]])
    lp = np.stack([np.log1p(-p),np.log(p)],-1)
    target = np.ones((4,1),dtype=np.int64)
    aggregate,rows = metrics(lp,target,np.array(['a','a','a','b']))
    assert aggregate['error'] == .5
    np.testing.assert_allclose(aggregate['nll'],(-np.log(.9)-np.log(.2))/2)
    assert [r['transitions'] for r in rows] == [3,1]


def test_learning_reaches_attention_after_zero_head_starts_up():
    cfg = config(CONFIG)
    seed_all(42); model = make_model(cfg,'ising_small')
    _,_,data = tiny_data()
    opt = torch.optim.AdamW(model.parameters(),lr=.001)
    initial = model.q.weight.detach().clone()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        b = data.batch(np.arange(8))
        score,delta = logits(model,b,True)
        loss_for(score,delta,b['target'],.001).backward()
        opt.step()
    assert model.q.weight.grad.abs().sum() > 0
    assert not torch.equal(initial,model.q.weight)


def test_interrupted_tuning_replays_dropout_optimizer_and_batch_order(tmp_path,monkeypatch):
    cfg = config(CONFIG)
    cfg['training'].update(batch_size=4,evaluation_batch_size=4,validation_every_steps=2,
                           max_steps=6,patience=99,min_delta=0.)
    fold = {'id':'fixture','train_questions':['train'],'validation_questions':['validation'],
            'assessment_questions':['assessment']}
    item = {'phase':'tune','fold':'fixture','outer':0,'model':'small','candidate':'c0','seed':42}
    protocol = {'candidates':[{'id':'c0','learning_rate':.001,'penalty':.001}]}
    def setup(name):
        root = tmp_path/name; (root/'folds/fixture').mkdir(parents=True)
        (root/'protocol.json').write_text('{}')
        (root/'folds/fixture/manifest.json').write_text('{}')
        return root
    first,second = setup('full'),setup('resume')
    _,_,data = tiny_data()
    cfg['output'] = str(first)
    out,complete = train_one(cfg,protocol,data,fold,item,'cpu',float('inf'))
    assert complete['status'] == 'complete'
    cfg['output'] = str(second)
    # Permit one update, interrupt at the next step, then resume from disk.
    counter = iter([0.,2.])
    with monkeypatch.context() as patch:
        patch.setattr('scripts.transformer_generalization.time.time',lambda:next(counter))
        _,interrupted = train_one(cfg,protocol,data,fold,item,'cpu',1.)
    assert interrupted['status'] == 'bounded'
    resumed,res = train_one(cfg,protocol,data,fold,item,'cpu',float('inf'))
    assert res['selected_weights_sha256'] == complete['selected_weights_sha256']
    assert res['terminal_weights_sha256'] == complete['terminal_weights_sha256']
    a = torch.load(out/'resume.pt',weights_only=False)
    b = torch.load(resumed/'resume.pt',weights_only=False)
    assert a['state']['history'] == b['state']['history']
    np.testing.assert_array_equal(np.load(out/'train_logp.npy'),np.load(resumed/'train_logp.npy'))
