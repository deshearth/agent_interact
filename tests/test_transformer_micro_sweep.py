import copy

import pytest
import torch

from mac_agents.common import config, seed_all
from scripts import transformer_generalization as base
from scripts.transformer_micro_sweep import CONFIG, expected
from scripts.transformer_size_sweep import verify_reference_compatibility


def test_parameter_counts_and_unchanged_reference_training_contract():
    cfg = config(CONFIG)
    verify_reference_compatibility(cfg, config(cfg['reference_config']))
    assert {k:sum(p.numel() for p in base.make_model(cfg,k).parameters()) for k in cfg['models']} == {
        'micro':2626,'nano':818,'pico':314,'tiny':7370}
    assert expected(cfg,'tune') == 216 and expected(cfg,'refit') == 36
    changed = copy.deepcopy(cfg); changed['training']['max_steps'] = 2048
    with pytest.raises(AssertionError):
        verify_reference_compatibility(changed, config(cfg['reference_config']))


@pytest.mark.parametrize('kind',['micro','nano','pico'])
def test_smallest_models_learn_and_preserve_agent_permutation_with_isolates(kind):
    seed_all(82)
    m = base.make_model(config(CONFIG),kind)
    s = torch.tensor([[-1.,1.,1.,-1.,1.],[1.,-1.,1.,-1.,-1.]])
    context = torch.randn(2,5,16)
    j = torch.zeros(2,5,5)
    j[:,0,1] = j[:,1,0] = 1
    j[:,1,2] = j[:,2,1] = -1
    j[:,2,3] = j[:,3,2] = 1
    batch = {'s':s,'context':context,'J':j,'extra':torch.empty(2,5,0)}
    initial = m.q.weight.detach().clone()
    opt = torch.optim.AdamW(m.parameters(),lr=.001)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        z,d = base.logits(m,batch,False)
        loss = base.loss_for(z,d,(s>0).float(),.001)
        assert torch.isfinite(loss)
        loss.backward(); opt.step()
    assert m.q.weight.grad.abs().sum() > 0
    assert not torch.equal(initial,m.q.weight)
    m.eval(); p = torch.tensor([4,2,0,3,1])
    permuted = {'s':s[:,p],'context':context[:,p],'J':j[:,p][:,:,p],'extra':batch['extra'][:,p]}
    torch.testing.assert_close(base.logits(m,permuted,False)[0],base.logits(m,batch,False)[0][:,p],atol=1e-6,rtol=1e-5)


def test_reused_reference_initialization_matches_exactly():
    cfg = config(CONFIG); original = config(cfg['reference_config'])
    seed_all(142); a = base.make_model(cfg,'tiny')
    seed_all(142); b = base.make_model(original,'tiny')
    assert base.model_weights_sha(a) == base.model_weights_sha(b)
