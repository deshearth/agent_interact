import copy
import numpy as np
from scipy.special import expit
import torch
from scripts import distribution_prediction_study as study

def logp(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return np.stack([np.log1p(-p), np.log(p)], -1)

def test_distribution_metrics_preserve_fractional_targets_and_reward_matching():
    p = np.array([[.1, .6], [.7, .9]]); qids = np.array(['a', 'b'])
    good, _ = study.distribution_metrics(logp(p), p, qids)
    wrong, _ = study.distribution_metrics(logp(np.array([[.01, .99], [.99, .99]])), p, qids)
    assert good['kl'] < 1e-15 and good['mae'] < 1e-15 and good['within_05'] == 1
    assert good['mode_error'] == wrong['mode_error'] == 0 and wrong['kl'] > .1 and wrong['mae'] > .1
    z = torch.tensor(np.log(p/(1-p)), dtype=torch.float64, requires_grad=True)
    torch.nn.functional.binary_cross_entropy_with_logits(z, torch.tensor(p)).backward()
    torch.testing.assert_close(z.grad, torch.zeros_like(z.grad), atol=1e-15, rtol=0)

def test_current_and_next_probability_boundary():
    p = np.arange(2*9*4).reshape(2, 9, 4)/72
    pairs = np.array([[0, 2], [1, 4]])
    current, target = study.paired_probabilities(pairs, p)
    changed = p.copy(); changed[0, 3:] = .99; changed[1, 5:] = .01
    after, new_target = study.paired_probabilities(pairs, changed)
    np.testing.assert_array_equal(current, after); assert not np.array_equal(target, new_target)
    np.testing.assert_array_equal(current, np.stack([p[0, 2], p[1, 4]]))

def test_extra_input_preserves_initial_function_parameters_and_rng():
    cfg = study.config(); cfg['training']['dropout'] = 0.
    for old, new in [('graph_opinion', 'graph_probability'), ('ordered_choice_opinion', 'ordered_choice_probability')]:
        a = study.make_model(cfg, old, seed=41); rng_a = torch.random.get_rng_state().clone()
        b = study.make_model(cfg, new, seed=41); rng_b = torch.random.get_rng_state().clone()
        torch.testing.assert_close(rng_a, rng_b)
        torch.testing.assert_close(a.encoder[0].weight[:, :18], b.encoder[0].weight[:, :18])
        torch.testing.assert_close(a.encoder[0].weight[:, 18:], b.encoder[0].weight[:, 19:])
        assert b.encoder[0].weight[:, 18].abs().sum() == 0
        assert sum(p.numel() for p in b.parameters())-sum(p.numel() for p in a.parameters()) == 8
    x = torch.randn(2, 3, 21); probability = torch.rand(2, 3, 1)
    torch.testing.assert_close(a.encoder[0](x), b.encoder[0](torch.cat([x[..., :18], probability, x[..., 18:]], -1)))

def test_richer_ising_extends_original_design_with_only_current_probabilities():
    phi = np.ones((1, 3, 16)); J = np.array([[[0, 1, -1], [1, 0, 0], [-1, 0, 0]]]); spins = np.array([[1, -1, 1]])
    p = np.array([[.2, .4, .9]])
    a = study.ising_design(phi, J, spins, p, False); b = study.ising_design(phi, J, spins, p, True)
    assert a.shape[-1] == 19 and b.shape[-1] == 22
    np.testing.assert_array_equal(a[..., :18], b[..., :18]); np.testing.assert_array_equal(a[..., -1], b[..., -1])
    np.testing.assert_allclose(b[0, 0, 18:21], [-.6, -.2, -.8])

def test_trainer_adapter_restores_original_functions():
    original = study.trainer.make_model, study.trainer.score_model, study.base.metrics
    with study.adapted_trainer():
        assert study.base.metrics is study.distribution_metrics
        assert study.trainer.make_model is study.make_model
    assert (study.trainer.make_model, study.trainer.score_model, study.base.metrics) == original

def test_forward_ignores_next_labels_and_uses_only_declared_current_information():
    cfg = study.config(); cfg['training']['dropout'] = 0.
    torch.manual_seed(17)
    graph = np.array([[[0, 1, -1], [1, 0, 1], [-1, 1, 0]]], dtype=np.int8)
    neighbors, signs = study.ordered.pair.neighbor_slots(graph)
    batch = {'s': torch.tensor([[1., -1., 1.]]), 'context': torch.randn(1, 3, 16), 'J': torch.tensor(graph, dtype=torch.float32),
        'current_probability': torch.tensor([[.2, .6, .8]]), 'target': torch.tensor([[.9, .1, .4]]),
        'text_arguments': torch.randn(1, 3, 8), 'text_conclusion': torch.randn(1, 3, 8),
        'neighbors': torch.tensor(neighbors), 'signs': torch.tensor(signs), 'positions': torch.randn(1, 3, 2, 3)}
    for name in cfg['models']:
        model = study.make_model(cfg, name, seed=9).eval()
        with torch.no_grad():
            model.classifier[-1].weight[1] = torch.linspace(-.5, .5, 8)
            if model.probability_input: model.encoder[0].weight[:, 18] = torch.linspace(-1, 1, 8)
        first = study.score_model(model, batch)
        altered = dict(batch); altered['target'] = 1-batch['target']
        torch.testing.assert_close(first, study.score_model(model, altered), atol=0, rtol=0)
        altered = dict(batch); altered['current_probability'] = 1-batch['current_probability']
        changed = study.score_model(model, altered)
        if model.probability_input: assert (first-changed).abs().max() > 1e-6
        else: torch.testing.assert_close(first, changed, atol=0, rtol=0)
        altered = dict(batch); altered['text_arguments'] = -batch['text_arguments']
        changed = study.score_model(model, altered)
        if name.startswith('ordered_arguments'): assert (first-changed).abs().max() > 1e-6
        else: torch.testing.assert_close(first, changed, atol=0, rtol=0)
