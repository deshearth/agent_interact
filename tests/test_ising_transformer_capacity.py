"""Checks of objective, information boundary and capacity-audit controls."""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mac_agents.models.graph import GraphUpdate
from scripts.ising_transformer_capacity import (
    group_bounds, logits_for, probability_metrics, receiver_key,
    snapshot_arrays, soft_cross_entropy, structural_features,
)


def example():
    rng = np.random.default_rng(4)
    s = np.array([1., -1., 1., -1.], dtype=np.float32)
    J = np.array([[0, 1, -1, 0], [1, 0, 0, 0], [-1, 0, 0, 1], [0, 0, 1, 0]], dtype=np.float32)
    context = rng.normal(size=(4, 16)).astype(np.float32)
    return s, context, J


def test_soft_objective_matches_original_classification_loss():
    torch.manual_seed(8)
    logits = torch.randn(3, 4, 2, requires_grad=True)
    target = torch.randint(0, 2, (3, 4))
    actual = soft_cross_entropy(logits, target.float())
    expected = F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])


def test_imitation_has_nonzero_entropy_but_zero_kl_at_teacher():
    p = np.array([[.2, .8, .5, 1e-7]])
    lp = np.stack([np.log1p(-p), np.log(p)], -1)
    m = probability_metrics(lp, np.array([[0, 1, 0, 0]]), p)
    assert m["teacher_entropy"] > 0
    assert abs(m["teacher_kl"]) < 1e-12
    assert m["teacher_probability_mae"] < 1e-12
    assert m["teacher_label_disagreement"] == 0


def test_collision_bounds_are_count_weighted_and_handle_pure_groups():
    m = group_bounds([[3, 1], [0, 2], [1, 1]])
    assert m["transitions"] == 8
    assert m["unavoidable_errors"] == 2
    assert m["classification_error_lower_bound"] == .25
    expected = (-(3*np.log(.75)+np.log(.25))+2*np.log(2))/8
    np.testing.assert_allclose(m["cross_entropy_lower_bound"], expected)


def test_teacher_and_input_features_do_not_read_next_opinion():
    s, c, J = example()
    a = {"spins": np.stack([s, -s])[None], "context": c[None], "J": J[None]}
    index = [{"qid": "q", "episode_id": "e"}]
    dataset = SimpleNamespace(pairs=[(0, 0)])
    baseline = {"kind": "I0", "weights": np.linspace(-.2, .3, 19).tolist()}
    before, _ = snapshot_arrays(a, index, dataset, baseline)
    a["spins"][:, 1] *= -1
    after, _ = snapshot_arrays(a, index, dataset, baseline)
    for k in ("s", "context", "J", "teacher", "teacher_logp"):
        np.testing.assert_array_equal(before[k], after[k])
    assert not np.array_equal(before["observed"], after["observed"])


def test_input_key_respects_node_permutations_and_one_hop_boundary():
    s, c, J = example()
    f = structural_features(s, c, J)
    key = receiver_key(f, J, 0)
    order = np.array([2, 3, 0, 1])
    J2 = J[order][:, order]
    f2 = structural_features(s[order], c[order], J2)
    assert receiver_key(f2, J2, 2) == key
    s2 = s.copy(); s2[3] *= -1
    assert receiver_key(structural_features(s2, c, J), J, 0) == key
    # The actual architecture obeys these same invariances.
    torch.manual_seed(42)
    model = GraphUpdate(dropout=0.).eval()
    def pred(spins, context, graph):
        return model(torch.tensor(spins[None]), torch.tensor(context[None]),
                     torch.tensor(graph[None]), torch.empty(1, 4, 0)).detach()
    torch.testing.assert_close(pred(s, c, J)[:, 0], pred(s[order], c[order], J2)[:, 2])
    torch.testing.assert_close(pred(s, c, J)[:, 0], pred(s2, c, J)[:, 0])
    changed = J.copy(); changed[1, 3] = changed[3, 1] = 1
    assert receiver_key(structural_features(s, c, changed), changed, 0) != key


def test_zero_initialized_neural_correction_exactly_contains_ising():
    s, c, J = example()
    model = GraphUpdate(dropout=0.)
    nn.init.zeros_(model.classifier[-1].weight)
    nn.init.zeros_(model.classifier[-1].bias)
    teacher_logit = torch.tensor([[1., -2., 0., .7]])
    b = {"s": torch.tensor(s[None]), "context": torch.tensor(c[None]),
         "J": torch.tensor(J[None]), "extra": torch.empty(1, 4, 0), "teacher_logit": teacher_logit}
    lp = logits_for(model, b, residual=True).log_softmax(-1)
    torch.testing.assert_close(lp[..., 1].exp(), teacher_logit.sigmoid())


def test_unregularized_model_matches_old_evaluation_function_and_learns():
    s, c, J = example()
    torch.manual_seed(42)
    original = GraphUpdate(dropout=.1).eval()
    model = GraphUpdate(dropout=0.)
    model.load_state_dict(original.state_dict())
    args = (torch.tensor(s[None]), torch.tensor(c[None]), torch.tensor(J[None]), torch.empty(1, 4, 0))
    torch.testing.assert_close(model(*args), original(*args))
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
    initial = model.q.weight.detach().clone()
    loss = soft_cross_entropy(model(*args), torch.tensor([[.1, .9, .3, .8]]))
    loss.backward()
    for layer in (model.q, model.k, model.v, model.encoder[0], model.classifier[-1]):
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
    opt.step()
    assert not torch.equal(model.q.weight, initial)
