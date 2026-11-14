"""Causal-input and graph invariance checks for the external pilot."""
import copy

import numpy as np
import pytest
import torch

from experiments.external.common import freeze_split
from experiments.external.mallm_pilot_v1.prepare import answer, build_record
from experiments.external.coevolve_pilot_v1.models import Inputs, TinyAttention, PooledMLP


def conversation():
    messages = []
    for i, agent in enumerate(['a', 'b', 'c', 'a', 'b']):
        messages.append({'message_id': i, 'agent_id': agent, 'turn': i//3+1,
                         'message': f'message {i}', 'solution': 'A) Yes',
                         'agreement': False, 'memory_ids': list(range(i)),
                         'additional_args': {'current_draft': 'A) Yes'}})
    return {'input': ['Question? A) Yes B) No'],
            'personas': [{'agentId': x} for x in ['a', 'b', 'c']], 'globalMemory': messages}


def test_mallm_target_and_future_text_cannot_change_current_features():
    original = conversation(); changed = copy.deepcopy(original)
    changed['globalMemory'][3]['message'] = 'Leaky target text'
    changed['globalMemory'][3]['solution'] = 'B) No'
    changed['globalMemory'][4]['message'] = 'Leaky future text'
    a = build_record(original, 'train')[0][0]
    b = build_record(changed, 'train')[0][0]
    assert a['target'] != b['target']
    for key in ['states', 'own', 'draft', 'visible_ids', 'latest_message_ids', 'agent_index', 'turn']:
        assert a[key] == b[key]
    assert a['visible_ids'] == [0, 1, 2]


def test_mallm_rejects_future_inbox():
    record = conversation(); record['globalMemory'][3]['memory_ids'].append(4)
    with pytest.raises(AssertionError):
        build_record(record, 'train')


def test_agreement_is_not_answer_and_inconsistent_labels_are_rejected():
    assert answer('[AGREE]') is None
    assert answer('A) No') is None
    assert answer('B) Yes') is None
    assert answer('A) Yes') == 1 and answer('B) No') == 0
    rows = build_record(conversation(), 'train')[0]
    assert all(r['target'] == 1 for r in rows)  # agreement=False must not become No.


def arrays():
    return {'state': np.array([[0, 1, 2], [2, 1, 0]]),
            'target': np.array([[1, 1, 2], [0, 1, 0]]),
            'identity': np.array([[0, 1, 2], [2, 3, 4]]),
            'adjacency': np.array([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]*2, dtype=bool),
            'step': np.array([0, 1]), 'question': np.array(['q1', 'q2']),
            'partition': np.array(['train', 'validation'])}


def test_coevolve_features_ignore_targets_and_confidence():
    a = arrays(); b = copy.deepcopy(a); b['target'][:] = 2
    b['probability_before'] = np.ones((2, 3, 3))*999
    x, y = Inputs(a), Inputs(b)
    assert torch.equal(x.pooled, y.pooled)
    for c, d in zip(x.batch([0, 1]), y.batch([0, 1])):
        assert torch.equal(c, d)


def test_pooling_excludes_non_neighbors():
    a = arrays(); b = copy.deepcopy(a); b['state'][0, 2] = 0
    x, y = Inputs(a), Inputs(b)
    assert torch.equal(x.pooled[0], y.pooled[0])
    assert not torch.equal(x.pooled[1], y.pooled[1])


def active_model():
    torch.manual_seed(5); model = TinyAttention(dropout=0.)
    torch.nn.init.normal_(model.classifier[-1].weight)
    model.eval(); return model


def test_attention_is_invariant_to_sender_order_and_ignores_masked_nodes():
    torch.set_num_threads(1)
    data = Inputs(arrays()); r, s, m, p = data.batch([0, 1, 2])
    model = active_model(); expected = model(r, s, m, p)
    order = torch.tensor([2, 0, 1])
    torch.testing.assert_close(expected, model(r, s[:, order], m[:, order], p))
    changed = s.clone(); changed[~m] = 1000
    torch.testing.assert_close(expected, model(r, changed, m, p))


def test_isolated_nodes_are_finite_and_match_budget():
    data = Inputs(arrays()); r, s, m, p = data.batch([0, 1])
    model = active_model(); logits = model(r, s, torch.zeros_like(m), p)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
    pooled = PooledMLP(np.zeros(68), np.ones(68))
    assert sum(x.numel() for x in model.parameters()) == 795
    assert sum(x.numel() for x in pooled.parameters()) == 795


def test_question_split_is_disjoint_reproducible_and_keeps_inspected_question_in_train(tmp_path):
    qs = [f'{prefix}_{i}' for prefix in ['false', 'true'] for i in range(10)]
    cfg = {'seed': 20261109, 'train_fraction': .6, 'validation_fraction': .2}
    args = dict(forced_train=['false_0'], strata={q: q.split('_')[0] for q in qs})
    a = freeze_split(tmp_path/'split.json', qs, cfg, **args)
    b = freeze_split(tmp_path/'split.json', list(reversed(qs)), cfg, **args)
    assert a == b
    assert {k: len(v) for k, v in a['groups'].items()} == {'train': 12, 'validation': 4, 'assessment': 4}
    assert 'false_0' in a['groups']['train']
    assert len(set(sum(a['groups'].values(), []))) == 20


def test_cross_configuration_overlap_retains_reserved_questions(tmp_path):
    qs = [f'q{i}' for i in range(20)]
    cfg = {'seed': 20261109, 'train_fraction': .6, 'validation_fraction': .2}
    inherited = {'q1': 'assessment', 'q2': 'train', 'q3': 'validation'}
    split = freeze_split(tmp_path/'split.json', qs, cfg, inherited=inherited)
    for q, role in inherited.items():
        assert q in split['groups'][role]
