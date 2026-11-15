"""Scientific checks for shared pair tokens and ordinary attention."""
import copy
import math
import time
import numpy as np
import torch
from mac_agents.common import json_write
from scripts import pair_token_attention_study as study

def fixture_batch(text_dim):
    torch.manual_seed(9)
    s = torch.where(torch.rand(3, 5) > .5, 1., -1.)
    c = torch.randn(3, 5, 16); j = torch.zeros(3, 5, 5)
    j[:, 0, 1] = 1; j[:, 2, 1] = -1; j[:, 0, 2] = -1; j[:, 1, 0] = 1
    text = torch.randn(3, 5, text_dim)/math.sqrt(text_dim)
    neighbors, signs = study.neighbor_slots(j.numpy())
    return s, c, j, text, torch.from_numpy(neighbors), torch.from_numpy(signs)

def active_model(cfg, width):
    model = study.make_model(cfg, width, 42).eval()
    torch.nn.init.normal_(model.classifier[-1].weight)
    return model

def test_counts_no_special_gains_and_same_common_initialization():
    cfg = study.config()
    for width in cfg['models']:
        model = study.make_model(cfg, width, 42)
        old = study.prior.make_model(cfg, width, 42)
        assert sum(p.numel() for p in model.parameters()) == cfg['expected_parameters'][width]
        assert not any(hasattr(model, name) for name in ('edge_bias', 'gamma', 'strength'))
        for name, weight in model.state_dict().items():
            if not name.startswith('pair_encoder.'):
                torch.testing.assert_close(weight, old.state_dict()[name], rtol=0, atol=0)

def test_packed_edges_match_dense_and_manual_standard_attention():
    cfg = study.config()
    for width, model_cfg in cfg['models'].items():
        model = active_model(cfg, width)
        s, c, j, t, neighbors, signs = fixture_batch(model_cfg['text_dim'])
        dense = model(s, c, j, t, True)
        packed = model(s, c, j, t, True, neighbors=neighbors, signs=signs)
        for a, b in zip(dense[:2], packed[:2]): torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
        state = torch.stack([s == -1, s == 1], -1).float()
        structure = torch.stack([(j > 0).sum(-1), (j < 0).sum(-1), j.abs().sum(-1)], -1).float().log1p()
        hn = model.norm(model.encoder(torch.cat([state, c, structure], -1)))
        sender = hn + model.text_projection(t*math.sqrt(model.text_dim))
        manual = torch.zeros_like(dense[1])
        for e in range(3):
            for receiver in range(5):
                nbr = torch.where(j[e, receiver] != 0)[0]
                if not len(nbr): continue
                tokens = model.pair_encoder(torch.cat([sender[e, nbr], j[e, receiver, nbr, None]], -1))
                headvalues = []
                for head in range(model.heads):
                    q = model.q(hn[e, receiver]).reshape(model.heads, model.d_k)[head]
                    k = model.k(tokens).reshape(len(nbr), model.heads, model.d_k)[:, head]
                    v = model.v(tokens).reshape(len(nbr), model.heads, model.d_v)[:, head]
                    weights = (k@q/math.sqrt(model.d_k)).softmax(0)
                    headvalues.append(weights@v)
                manual[e, receiver] = model.out(torch.cat(headvalues))
        torch.testing.assert_close(packed[1], manual, atol=2e-6, rtol=2e-6)
        assert torch.isfinite(packed[0]).all()
        torch.testing.assert_close(packed[1][:, 3:], torch.zeros_like(packed[1][:, 3:]), atol=0, rtol=0)
        torch.testing.assert_close(packed[2][:, :, 3:], torch.zeros_like(packed[2][:, :, 3:]), atol=0, rtol=0)

def test_pair_sign_text_locality_equivariance_and_gradients():
    cfg = study.config()
    for width, model_cfg in cfg['models'].items():
        model = active_model(cfg, width)
        s, c, j, text, neighbors, signs = fixture_batch(model_cfg['text_dim'])
        before = model(s, c, j, text)
        changed = text.clone(); changed[:, 1] *= -3
        after = model(s, c, j, changed)
        torch.testing.assert_close(before[:, [1, 3, 4]], after[:, [1, 3, 4]], rtol=0, atol=0)
        assert not torch.allclose(before[:, [0, 2]], after[:, [0, 2]])
        flipped = signs.clone(); flipped[:, 0, 0] *= -1
        sign_after = model(s, c, j, text, neighbors=neighbors, signs=flipped)
        assert not torch.allclose(before[:, 0], sign_after[:, 0])
        token_input = torch.randn(2, model.encoder[0].out_features + 1)
        token_input[1, :-1] = token_input[0, :-1]; token_input[:, -1] = torch.tensor([1., -1.])
        assert not torch.allclose(*model.pair_encoder(token_input).unbind(0))
        permutation = torch.tensor([4, 2, 0, 3, 1])
        actual = model(s[:, permutation], c[:, permutation], j[:, permutation][:, :, permutation], text[:, permutation])
        torch.testing.assert_close(actual, before[:, permutation], atol=2e-6, rtol=2e-6)
        torch.nn.functional.cross_entropy(before.reshape(-1, 2), (s > 0).long().flatten()).backward()
        assert model.text_projection.weight.grad.norm() > 0
        assert model.pair_encoder[0].weight.grad[:, -1].norm() > 0

def synthetic_data(cfg, root):
    cfg = copy.deepcopy(cfg); cfg['output'] = str(root); cfg['message_reference'] = str(root/'messages')
    cfg['training'].update(max_steps=6, validation_every_steps=2, batch_size=4, evaluation_batch_size=8,
        learning_rates=[.003], output_penalties=[0.], refit_seeds=[142], patience=5, torch_threads=1)
    folder = root/'data'; folder.mkdir(parents=True); source = root/'messages/data'; source.mkdir(parents=True)
    rng = np.random.default_rng(8); spins = rng.choice([-1, 1], size=(4, 9, 5))
    graph = np.zeros((4, 5, 5), dtype=np.float32); graph[:, 0, 1] = 1; graph[:, 1, 0] = -1
    np.savez(folder/'development.npz', spins=spins, J=graph,
        pairs=np.array([(e, t) for e in range(4) for t in range(8)]), episode_qids=np.array(['t0', 't1', 'v0', 'v1']))
    np.savez(folder/'features.npz', context=rng.normal(size=(4, 5, 16)).astype(np.float32))
    vectors = rng.normal(size=(30, 384)).astype(np.float32); vectors /= np.linalg.norm(vectors, axis=1, keepdims=True); vectors[0] = 0
    np.save(source/'vectors.npy', vectors); ids = rng.integers(3, 30, size=(4, 8, 5)); np.save(folder/'message_ids.npy', ids)
    np.save(folder/'vectors_pca8.npy', study.prior.fit_message_pca(vectors, ids[:2])[0])
    np.savez(folder/'matched_substitution.npz', message_ids=ids[:, ::-1])
    protocol = {'fold': {'train_questions': ['t0', 't1'], 'validation_questions': ['v0', 'v1'], 'assessment_questions': ['unread']},
                'parameters': cfg['expected_parameters']}
    for name, obj in [('protocol.json', protocol), ('prepared.json', {}), ('encoding.json', {})]: json_write(root/name, obj)
    return cfg, protocol

def test_synthetic_tune_refit_freeze_and_target_isolation(tmp_path, monkeypatch):
    cfg, protocol = synthetic_data(study.config(), tmp_path/'all')
    monkeypatch.setattr(study, 'validate', lambda cfg: protocol)
    data = study.Data(cfg, 'cpu'); original = study.shared.make_model, study.shared.score_model
    before = data.batch([0, 1]); data.y[:] = 1-data.y
    after = data.batch([0, 1])
    for key in before:
        if key != 'target': torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    data.condition = 'conclusion'; before = data.batch([0, 1])['text']; data.ids_text[:] = 3
    torch.testing.assert_close(before, data.batch([0, 1])['text'], atol=0, rtol=0)
    study.fit(cfg, 'cpu'); frozen = study.validate_frozen(cfg)
    assert frozen['tuning_fits'] == frozen['refits_completed'] == 8
    assert (study.shared.make_model, study.shared.score_model) == original
    for path in frozen['refits']:
        m = study.read(study.Path(path)/'manifest.json')
        assert m['parameters'] == cfg['expected_parameters'][m['width']]

def test_interrupted_optimizer_resume_equals_uninterrupted(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg, protocol = synthetic_data(study.config(), tmp_path/'resumed')
    whole, _ = synthetic_data(study.config(), tmp_path/'whole')
    item = dict(phase='tune', width='pico_pca8', condition='arguments', candidate='c0', seed=42)
    with study.adapted_trainer():
        calls = iter([0, 0, 0, 0, 101])
        with monkeypatch.context() as patch:
            patch.setattr(study.shared.time, 'time', lambda: next(calls, 101))
            path, partial = study.shared.train_one(cfg, protocol, study.Data(cfg, 'cpu'), item, 100)
        assert partial['status'] == 'bounded'
        resume = torch.load(path/'resume.pt', weights_only=False)
        assert 0 < resume['state']['step'] < cfg['training']['max_steps']
        p, resumed = study.shared.train_one(cfg, protocol, study.Data(cfg, 'cpu'), item, time.time()+60)
        q, full = study.shared.train_one(whole, protocol, study.Data(whole, 'cpu'), item, time.time()+60)
    assert resumed['terminal_weights_sha256'] == full['terminal_weights_sha256']
    assert resumed['selected_weights_sha256'] == full['selected_weights_sha256']
    assert resumed['selected_step'] == full['selected_step']
