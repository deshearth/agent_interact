"""Scientific invariants for the within-question representation comparison."""
import numpy as np
from scripts import argument_variance_diagnosis as study

def test_covariance_matches_explicit_weighted_residuals_and_total_partition():
    rng=np.random.default_rng(123);vectors=rng.normal(size=(30,12));groups=np.array([[3,4,5,5,6],[5,7,8,9,9],[10,11,12,13,14]])
    output,fit=study.fit_within_question_pca(vectors,groups,4)
    membership=np.bincount(np.concatenate([np.unique(g) for g in groups]),minlength=len(vectors))
    expected=np.zeros((12,12))
    for group in groups:
        ids=np.unique(group);w=1/membership[ids];x=vectors[ids];mu=(x*w[:,None]).sum(0)/w.sum()
        expected+=(x-mu).T@((x-mu)*w[:,None])
    expected/=len(fit['training_ids'])
    np.testing.assert_allclose(fit['within_covariance'],expected,atol=2e-15)
    np.testing.assert_allclose(fit['within_covariance']+fit['between_covariance'],fit['total_covariance'],atol=1e-15)
    np.testing.assert_allclose((output[fit['training_ids']]**2).mean(),1/4,rtol=1e-6)
    np.testing.assert_array_equal(output[0],0)

def test_no_heldout_vectors_affect_fitted_transform_or_training_encoding():
    rng=np.random.default_rng(125);x=rng.normal(size=(40,16));groups=np.arange(3,23).reshape(4,5)
    a,fit=study.fit_within_question_pca(x,groups,8)
    changed=x.copy();changed[23:]=rng.normal(size=changed[23:].shape)*100
    b,other=study.fit_within_question_pca(changed,groups,8)
    for key in fit:np.testing.assert_array_equal(fit[key],other[key])
    np.testing.assert_array_equal(a[:23],b[:23])
    assert not np.array_equal(a[23:],b[23:])

def test_within_axes_ignore_constant_question_shifts():
    rng=np.random.default_rng(126);x=rng.normal(size=(83,12));groups=np.arange(3,83).reshape(8,10)
    _,a=study.fit_within_question_pca(x,groups,4)
    changed=x.copy()
    for group in groups:changed[group]+=rng.normal(size=12)*5
    _,b=study.fit_within_question_pca(changed,groups,4)
    np.testing.assert_allclose(a['within_covariance'],b['within_covariance'],atol=2e-14)
    np.testing.assert_allclose(a['components'],b['components'],atol=2e-12)

def test_corrections_regressions_and_question_pairing_are_consistent():
    y=np.array([[0,0],[1,1]])
    a=np.log(np.array([[[[.8,.2],[.2,.8]],[[.2,.8],[.7,.3]]]]))
    b=np.log(np.array([[[[.3,.7],[.3,.7]],[[.3,.7],[.3,.7]]]]))
    result=study.error_pair(a,b,y)
    assert result['corrected']==result['introduced']==.25
    assert result['disagreement']==.5
    assert result['right_error']-result['left_error']==result['corrected']-result['introduced']
    rng=np.random.default_rng(91);draws=rng.integers(0,2,(1000,2))
    ma,mb=study.metrics(a,y),study.metrics(b,y)
    paired=study.paired(ma['error'],mb['error'],draws)
    assert paired['mean_difference']==0
    assert paired['ci95'][0]<0<paired['ci95'][1]
