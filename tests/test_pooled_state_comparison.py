import copy

import numpy as np
import torch

from mac_agents.common import config, seed_all
from scripts import transformer_generalization as base
from scripts import distill_transformer_rule as rules
from scripts.pooled_state_comparison import (
    CONFIG, PooledUpdate, fit_linear, linear_predict, make_model, normalization, pooled_inputs,
)


def fixture():
    seed_all(441)
    s = (torch.randint(0, 2, (6, 5))*2-1).float()
    phi = torch.randn(6, 5, 16); phi[..., 0] = 1
    j = torch.randint(-1, 2, (6, 5, 5)).float()
    j = torch.triu(j, diagonal=1); j = j+j.transpose(-1, -2)
    j[:, 4] = 0; j[:, :, 4] = 0
    return s, phi, j


def test_pooling_matches_independent_numpy_signed_rule_and_permutation():
    s, phi, j = fixture()
    x = pooled_inputs(s, phi, j)
    expected, _ = rules.features(s.numpy(), phi.numpy(), j.numpy(), np.zeros(s.shape), 'pooled')
    np.testing.assert_allclose(x.numpy(), expected[..., 1:], atol=1e-7)
    p = [4, 2, 1, 0, 3]
    torch.testing.assert_close(pooled_inputs(s[:, p], phi[:, p], j[:, p][:, :, p]), x[:, p])
    assert (x[:, 4, -4:] == 0).all()


def test_models_have_declared_sizes_zero_start_and_learn_from_pool():
    cfg = config(CONFIG); s, phi, j = fixture(); x = pooled_inputs(s, phi, j)
    for kind, expected in [('pooled_small', 809), ('pooled_large', 3457)]:
        seed_all(41); m = make_model(cfg, kind)
        assert sum(p.numel() for p in m.parameters()) == expected
        assert (m(s, phi, j, x) == 0).all()
        initial = m.net[0].weight.detach().clone()
        opt = torch.optim.AdamW(m.parameters(), lr=.01)
        for _ in range(8):
            opt.zero_grad(); z = m(s, phi, j, x); score = z[..., 1]-z[..., 0]
            loss = base.loss_for(score, score, (s > 0).float(), .001)
            loss.backward(); opt.step()
        assert not torch.equal(m.net[0].weight, initial)
        m.eval(); torch.testing.assert_close(m(s, phi, j, x), m(s, phi, j, x[..., :0]))
        p = [4, 2, 1, 0, 3]
        torch.testing.assert_close(m(s[:, p], phi[:, p], j[:, p][:, :, p], x[:, p]), m(s, phi, j, x)[:, p])


def test_neighbor_persona_changes_do_not_enter_receivers_pool():
    s, phi, j = fixture(); x = pooled_inputs(s, phi, j)
    changed = phi.clone(); changed[:, 1:, 1:] += 100
    torch.testing.assert_close(pooled_inputs(s, changed, j)[:, 0], x[:, 0])


def test_normalization_fits_only_given_question_weighted_inputs():
    rng = np.random.default_rng(13); x = rng.normal(size=(4, 3, 20))
    qids = np.array(['a', 'a', 'a', 'b'])
    n = normalization(x, qids)
    np.testing.assert_allclose(n['mean'], (x[:3].mean((0, 1))+x[3].mean(0))/2)
    extra = np.concatenate([x, x*1e9]); changed = normalization(extra[:4], qids)
    assert n == changed


def test_linear_fit_ignores_unpassed_outcomes_and_recovers_known_probabilities():
    rng = np.random.default_rng(44); x = rng.normal(size=(50, 4, 20)); q = np.array([f'q{i%5}' for i in range(50)])
    truth = np.r_[.2, rng.normal(scale=.2, size=20)]
    p = 1/(1+np.exp(-(truth[0]+x@truth[1:])))
    record = fit_linear(x, p, q, 0.)
    np.testing.assert_allclose(record['raw_weights'], truth, atol=2e-5)
    np.testing.assert_allclose(np.exp(linear_predict(record, x)[..., 1]), p, atol=2e-6)
    future = np.concatenate([p, 1-p]); second = fit_linear(x, future[:50], q, 0.)
    assert record == second


def test_neural_training_settings_match_frozen_graph_reference():
    cfg = config(CONFIG); ref = config(cfg['reference_config'])
    for k, v in cfg['training'].items():
        if k not in ['torch_threads', 'aggregate_minutes']:
            assert v == ref['training'][k]
