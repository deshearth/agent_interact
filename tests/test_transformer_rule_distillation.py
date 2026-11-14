import copy

import numpy as np
from scipy.special import expit

from mac_agents.common import config
from scripts.distill_transformer_rule import (
    CONFIG,equal_question_weights,features,fidelity,fit_rule,log_prob,
    objective,predict,standardize,
)


def fixture():
    rng=np.random.default_rng(7)
    s=rng.choice([-1.,1.],size=(5,7))
    phi=rng.normal(size=(5,7,16)); phi[...,0]=1
    graph=rng.choice([-1,0,1],size=(5,7,7)); graph=np.triu(graph,1); graph+=graph.transpose(0,2,1)
    prior=rng.normal(size=(5,7))
    return s,phi,graph,prior


def test_rules_have_declared_sizes_and_respect_agent_permutations():
    s,phi,j,z=fixture(); order=np.array([3,1,0,5,6,4,2])
    for family,count in [('ising_own',3),('pooled',21),('contextual',39),('quadratic',42)]:
        x,names=features(s,phi,j,z,family)
        changed,names2=features(s[:,order],phi[:,order],j[:,order][:,:,order],z[:,order],family)
        assert x.shape==(5,7,count) and len(set(names))==count and names2==names
        np.testing.assert_allclose(changed,x[:,order],atol=1e-14)


def test_signed_neighbor_pools_use_raw_opinions_and_exclude_own_spin():
    s=np.array([[1.,1.,-1.]])
    j=np.array([[[0,1,-1],[1,0,0],[-1,0,0]]])
    phi=np.ones((1,3,16)); prior=np.zeros((1,3))
    x,names=features(s,phi,j,prior,'pooled')
    assert x[0,0,names.index('friendly_sum')]==1
    assert x[0,0,names.index('unfriendly_sum')]==-1
    changed=s.copy(); changed[0,0]=-1
    y,_=features(changed,phi,j,prior,'pooled')
    assert y[0,0,names.index('own_opinion')]==-1
    np.testing.assert_array_equal(y[0,0,17:],x[0,0,17:])


def test_question_weights_and_standardization_ignore_unpassed_heldout_rows():
    q=np.array(['a','a','a','b']); w=equal_question_weights(q,2)
    np.testing.assert_allclose(w[:6].sum(),.5)
    np.testing.assert_allclose(w[6:].sum(),.5)
    x=np.column_stack([np.ones(8),np.arange(8)])
    _,mean,scale=standardize(x,w)
    altered=np.vstack([x,[[1,1e10],[1,-1e10]]])
    _,mean2,scale2=standardize(altered[:8],w)
    np.testing.assert_array_equal(mean,mean2); np.testing.assert_array_equal(scale,scale2)
    assert mean[0]==0 and scale[0]==1


def test_soft_target_objective_gradient_matches_finite_difference():
    rng=np.random.default_rng(13)
    x=np.column_stack([np.ones(19),rng.normal(size=(19,4))])
    p=rng.uniform(.05,.95,19); w=np.full(19,1/19); beta=rng.normal(size=5)
    value,grad=objective(beta,x,p,w,.001)
    numeric=[]
    for k in range(5):
        step=np.zeros(5); step[k]=1e-5
        numeric.append((objective(beta+step,x,p,w,.001)[0]-objective(beta-step,x,p,w,.001)[0])/2e-5)
    np.testing.assert_allclose(grad,numeric,atol=1e-9,rtol=1e-7)
    expected=np.mean(-p*np.log(expit(x@beta))-(1-p)*np.log(expit(-x@beta)))+.0005*np.sum(beta[1:]**2)
    np.testing.assert_allclose(value,expected,atol=1e-12)


def test_surrogate_recovers_known_rule_without_observed_target_labels():
    rng=np.random.default_rng(19)
    x=rng.normal(size=(80,4,5)); x[...,0]=1
    truth=np.array([.1,.8,-.5,.3,.2]); teacher=log_prob(x@truth)
    fitting=copy.deepcopy(config(CONFIG)['fitting']); fitting['ridge']=0.
    rule=fit_rule(x,teacher,np.array([f'q{i%8}' for i in range(80)]),fitting)
    np.testing.assert_allclose(rule['raw_weights'],truth,atol=2e-5)
    lp=predict(rule,x)
    np.testing.assert_allclose(np.exp(lp[...,1]),np.exp(teacher[...,1]),atol=2e-6)
    rows=fidelity(lp,teacher,np.array([f'q{i%8}' for i in range(80)]))
    assert max(r['teacher_kl'] for r in rows)<1e-10


def test_identical_probabilities_have_zero_fidelity_loss():
    z=np.array([[0.,2.,-2.],[1.,-1.,.1]])
    rows=fidelity(log_prob(z),log_prob(z),np.array(['a','b']))
    for r in rows:
        assert r['probability_mae']==0 and r['teacher_label_disagreement']==0
        assert abs(r['teacher_kl'])<1e-14
