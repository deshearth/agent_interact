import numpy as np
import torch
from scripts import modal_change_rule as study


def test_previous_inbox_alignment_and_empty_initial():
    data=study.Data('inner'); e,t=data.common['pairs'].T
    for key in ('s','context','J','text_arguments','text_conclusion','neighbors','signs','positions'):
        np.testing.assert_array_equal(data.values['previous_'+key][t>0],data.values[key][np.flatnonzero(t>0)-1])
    for key in ('J','signs','positions','text_arguments','text_conclusion'): assert not data.values['previous_'+key][t==0].any()
    single=np.load('runs/generalization_investigation_320_v1/noise_audit_v1/probabilities.npz')['single']
    np.testing.assert_allclose(data.values['current_logit'],np.log(single[e,t])-np.log1p(-single[e,t]),atol=1e-6)


def test_fixed_gain_conserves_identical_inbox_and_ignores_target():
    data=study.Data('inner'); batch=data.batch(np.array([0,7,10,101])); model=study.make_model(study.config(),'arguments_fixed',42)
    with torch.no_grad(): model.potential.classifier[-1].weight.normal_()
    model.eval()
    for key in ('s','context','J','text_arguments','text_conclusion','neighbors','signs','positions'): batch['previous_'+key]=batch[key].clone()
    with torch.inference_mode():
        score=model.score(batch); torch.testing.assert_close(score,batch['current_logit'],atol=0,rtol=0)
        batch['target']=1-batch['target']; torch.testing.assert_close(model.score(batch),score,atol=0,rtol=0)


def test_two_inbox_branches_have_gradients():
    data=study.Data('inner'); batch=data.batch(np.array([2,3])); model=study.make_model(study.config(),'arguments_learned',42)
    with torch.no_grad(): model.potential.classifier[-1].weight.normal_()
    model.eval()
    for key in ('text_arguments','previous_text_arguments'): batch[key]=batch[key].clone().requires_grad_()
    model.score(batch).square().mean().backward()
    assert model.gain.grad is not None
    for key in ('text_arguments','previous_text_arguments'): assert batch[key].grad.abs().sum()>0
