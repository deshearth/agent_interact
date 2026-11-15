import numpy as np
import torch
from scripts import modal_causal_history as study


def test_future_states_cannot_change_causal_features():
    rng = np.random.default_rng(42)
    spins = rng.choice([-1, 1], size=(2, 9, 3)); probability = rng.uniform(size=spins.shape)
    graph = np.array([[[0, 1, -1], [1, 0, 0], [-1, 0, 0]]]*2)
    pairs = np.array([[0, 0], [1, 3]])
    before = study.causal_states(spins, probability, graph, pairs)
    changed_s, changed_p = spins.copy(), probability.copy()
    changed_s[0, 1:] *= -1; changed_p[0, 1:] = 1-changed_p[0, 1:]
    changed_s[1, 4:] *= -1; changed_p[1, 4:] = 1-changed_p[1, 4:]
    np.testing.assert_array_equal(before, study.causal_states(changed_s, changed_p, graph, pairs))
    assert not before[0, :, 1:].any() and not before[1, :, 4:].any()


def test_real_history_gradients_and_matched_round_control():
    torch.set_num_threads(2); cfg = study.config(); data = study.Data('inner')
    p = study.core.validate(study.core.config()); ids = data.ids(p['inner']['fit'])[:2]
    batch = data.batch(ids, 'mode')
    a, b = [study.make_model(cfg, 'history_choice_'+state, 42) for state in ('observed', 'round')]
    for left, right in zip(a.parameters(), b.parameters()): torch.testing.assert_close(left, right, atol=0, rtol=0)
    for model in (a, b):
        model.eval()
        with torch.no_grad():
            model.encoder[0].weight[:, 18:34].fill_(.03)
            model.classifier[-1].weight[1].fill_(.1)
        score = study.core.score_model(model, batch); loss = torch.nn.functional.binary_cross_entropy_with_logits(score, batch['target']); loss.backward()
        assert torch.isfinite(loss)
    assert a.history.weight_ih_l0.grad[:, :6].abs().sum() > 0
    assert b.history.weight_ih_l0.grad[:, :6].abs().sum() == 0
    changed = dict(batch, observed_history=batch['observed_history'].clone())
    changed['observed_history'][..., :6] += .5
    with torch.no_grad():
        torch.testing.assert_close(study.core.score_model(b, batch), study.core.score_model(b, changed), atol=0, rtol=0)
        assert not torch.equal(study.core.score_model(a, batch), study.core.score_model(a, changed))
