import copy
import pickle
import numpy as np
from mac_agents.common import json_write
from scripts import generalization_state_models as study
from scripts.local_conversation_pilot import deliver

def fixture():
    rng = np.random.default_rng(6)
    spins = rng.choice([-1, 1], size=(2, 9, 5)); raw = np.repeat(spins[..., None], 5, -1)
    graph = np.zeros((2, 5, 5), dtype=np.int8)
    for i, j, sign in [(0, 1, 1), (0, 2, -1), (0, 3, 1), (1, 2, 1)]: graph[:, i, j] = graph[:, j, i] = sign
    common = {'spins': spins, 'raw_votes': raw, 'J': graph, 'pairs': np.array([(e, t) for e in range(2) for t in range(8)])}
    context = rng.normal(size=(2, 5, 16)).astype(np.float32)
    names = ['synthetic-a', 'synthetic-b']
    order = study.ordered_neighbors(graph, names, common['pairs'])
    control = study.ordered_neighbors(graph, names, common['pairs'], True)
    return common, context, order, control, names

def test_order_matches_actual_source_inboxes():
    common, _, order, control, names = fixture()
    for b, (e, t) in enumerate(common['pairs']):
        messages = {i: 'sender'+str(i) for i in range(5) if common['J'][e, i].any()}
        agree, disagree, _ = deliver(common['J'][e].tolist(), messages, names[e], t+1)
        for receiver in range(5):
            for k, actual in enumerate((agree[receiver], disagree[receiver])):
                mapped = order[b, receiver, k]; mapped = mapped[mapped >= 0]
                assert [messages[i] for i in mapped] == actual
                assert set(control[b, receiver, k][control[b, receiver, k] >= 0]) == set(mapped)
    assert not np.array_equal(order, control)

def test_future_votes_and_states_cannot_change_current_features():
    common, context, order, control, _ = fixture()
    a = study.state_features(common, context, order, control)
    changed = copy.deepcopy(common); changed['spins'][0, 4:] *= -1; changed['raw_votes'][0, 4:] *= -1
    b = study.state_features(changed, context, order, control)
    take = np.flatnonzero((common['pairs'][:, 0] == 0)&(common['pairs'][:, 1] <= 3))
    for j in (0, 2): np.testing.assert_array_equal(a[j][take], b[j][take])
    for j in (0, 1): np.testing.assert_array_equal(a[1][j][take], b[1][j][take])
    assert np.isfinite(a[0]).all() and np.isfinite(a[2]).all()
    assert not np.array_equal(a[2][4:8], b[2][4:8])

def test_opinion_summary_features_are_permutation_equivariant_and_isolated():
    common, context, order, control, names = fixture()
    a = study.state_features(common, context, order, control)[0]
    perm = np.array([3, 1, 4, 2, 0]); changed = copy.deepcopy(common)
    changed['spins'] = common['spins'][:, :, perm]; changed['raw_votes'] = common['raw_votes'][:, :, perm]
    changed['J'] = common['J'][:, perm][:, :, perm]
    oo = study.ordered_neighbors(changed['J'], names, changed['pairs'])
    b = study.state_features(changed, context[:, perm], oo, oo)[0]
    np.testing.assert_allclose(a[:, perm], b, atol=1e-7, rtol=1e-7)
    np.testing.assert_array_equal(a[:, 4, 17:], 0)

def test_internal_training_then_refit_ignores_outer_validation_targets(tmp_path, monkeypatch):
    cfg = study.config(); cfg['output'] = str(tmp_path); cfg['variants'] = ['base']
    cfg['training'].update(max_iter=8, max_leaf_nodes=[7], l2_regularization=[1.], min_samples_leaf=2, threads=1, n_iter_no_change=3)
    protocol = {'fold': {'train_questions': ['a', 'b', 'c', 'd'], 'validation_questions': ['v', 'w']},
                'inner': {'fit': ['a', 'b', 'c'], 'validation': ['d']}}
    rng = np.random.default_rng(9); x = rng.normal(size=(192, 5)).astype(np.float32)
    y = ((x[:, 0]+.4*rng.normal(size=192)) > 0).astype(int); q = np.repeat(['a', 'b', 'c', 'd', 'v', 'w'], 32)
    for stage in ('inner', 'full'):
        folder = tmp_path/'features'/stage; folder.mkdir(parents=True); np.save(folder/'base.npy', x)
    np.savez(tmp_path/'labels.npz', target=y, qids=q); json_write(tmp_path/'protocol.json', protocol)
    monkeypatch.setattr(study, 'validate', lambda cfg: protocol)
    study.worker(cfg, 'base', 'tune'); choice = study.read(tmp_path/'tune/base/leaf7-l21.json')
    assert 1 <= choice['selected_iterations'] <= 8
    json_write(tmp_path/'selection.json', {'choices': [choice]})
    study.worker(cfg, 'base', 'refit')
    with (tmp_path/'refit/base/model.pkl').open('rb') as f: first = pickle.load(f)
    expected = first.predict_proba(x)
    # A separate output root with identical training labels but altered outer labels.
    other = tmp_path/'changed'; other.mkdir(); altered = copy.deepcopy(cfg); altered['output'] = str(other)
    import shutil
    shutil.copytree(tmp_path/'features', other/'features')
    flipped = y.copy(); flipped[np.isin(q, ['v', 'w'])] = 1-flipped[np.isin(q, ['v', 'w'])]
    np.savez(other/'labels.npz', target=flipped, qids=q)
    for name in ('protocol.json', 'selection.json'): shutil.copyfile(tmp_path/name, other/name)
    study.worker(altered, 'base', 'refit')
    with (other/'refit/base/model.pkl').open('rb') as f: second = pickle.load(f)
    np.testing.assert_array_equal(expected, second.predict_proba(x))
