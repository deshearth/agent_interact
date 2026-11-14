import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from mac_agents.common import config, seed_all
from scripts import transformer_generalization as base
from scripts.transformer_size_sweep import CONFIG, choose, verify_reference_compatibility


def test_sizes_and_reference_training_contract():
    cfg = config(CONFIG)
    verify_reference_compatibility(cfg, config(base.CONFIG))
    assert {k: sum(p.numel() for p in base.make_model(cfg, k).parameters()) for k in cfg['models']} == {
        'tiny': 7370, 'medium': 16842, 'small': 34978}
    changed = copy.deepcopy(cfg)
    changed['training']['dropout'] = 0.2
    with pytest.raises(AssertionError):
        verify_reference_compatibility(changed, config(base.CONFIG))


@pytest.mark.parametrize('kind', ['tiny', 'medium'])
def test_smaller_attention_learns_and_respects_agent_permutations(kind):
    cfg = config(CONFIG)
    seed_all(31)
    model = base.make_model(cfg, kind)
    s = torch.tensor([[-1., 1., 1., -1.], [1., -1., 1., -1.]])
    context = torch.randn(2, 4, 16)
    graph = torch.tensor([[0.,1.,0.,-1.],[1.,0.,-1.,0.],[0.,-1.,0.,1.],[-1.,0.,1.,0.]]).repeat(2,1,1)
    batch = {'s':s, 'context':context, 'J':graph, 'extra':torch.empty(2,4,0)}
    initial = model.q.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        z,d = base.logits(model,batch,False)
        base.loss_for(z,d,(s>0).float(),.001).backward()
        opt.step()
    assert not torch.equal(model.q.weight,initial)
    model.eval()
    p = torch.tensor([2,0,3,1])
    permuted = {'s':s[:,p], 'context':context[:,p], 'J':graph[:,p][:,:,p], 'extra':batch['extra'][:,p]}
    torch.testing.assert_close(base.logits(model,permuted,False)[0],base.logits(model,batch,False)[0][:,p],rtol=1e-5,atol=1e-6)


def test_selection_ignores_outer_scores_and_scales_selected_data_passes():
    candidates = [{'id':'c0'},{'id':'c1'}]
    records = []
    for candidate in candidates:
        for inner, step in enumerate([128,256,512]):
            records.append((Path(f'fixture/{candidate["id"]}/{inner}'), {'outer':0,'fold':f'i{inner}',
                'model':'tiny','candidate':candidate['id'],'validation':{'nll':.3 if candidate['id']=='c0' else .4},
                'selected_step':step,'assessment':{'nll':0. if candidate['id']=='c1' else 100.}}))
    choices, _ = choose(records,candidates,['tiny'],outer_count=1)
    assert choices[0]['candidate'] == 'c0' and choices[0]['refit_steps'] == 384
    changed = copy.deepcopy(records)
    for _,m in changed:
        m['assessment']['nll'] *= -100
    assert choose(changed,candidates,['tiny'],outer_count=1)[0] == choices


def test_reference_initialization_is_exactly_reusable():
    cfg, old = config(CONFIG), config(base.CONFIG)
    seed_all(142); first = base.make_model(cfg,'small')
    seed_all(142); second = base.make_model(old,'small')
    assert base.model_weights_sha(first) == base.model_weights_sha(second)
