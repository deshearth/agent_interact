"""Independent whole-question scores, position construction, and model replay."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import generalization_ordered_transformer as study
from scripts.local_conversation_pilot import deliver

def main():
    cfg = study.config(); root = Path(cfg['output']); p = study.validate(cfg)
    frozen = study.read(root/'models_frozen.json'); results = study.read(root/'results.json')
    for path, sha in frozen['artifacts'].items(): assert digest(path) == sha, path
    assert results['models_frozen_sha256'] == digest(root/'models_frozen.json')
    assert frozen['frozen_utc'] < results['completed_utc']
    choices = study.read(root/'selection.json')['choices']; max_metric_difference = 0.
    for c in choices:
        records = [study.read(path) for path in (root/'tune').glob('*/manifest.json') if study.read(path)['condition'] == c['condition']]
        best = min(records, key=lambda r: (r['validation']['nll'], r['candidate']))
        assert c['candidate'] == best['candidate'] and c['refit_steps'] == best['selected_step']
        assert len({r['initial_weights_sha256'] for r in records}) == 1
    for path in frozen['refits']:
        r = study.read(Path(path)/'manifest.json'); c = next(c for c in choices if c['condition'] == r['condition'])
        assert r['selected_step'] == c['refit_steps'] and r['completed_utc'] < frozen['frozen_utc']
        assert r['parameters'] == 970 and r['validation'] is None
    common = dict(np.load(study.state.config()['common'], allow_pickle=False)); e, t = common['pairs'].T
    index = {i['qid']: i for i in study.read(study.state.config()['index'])}
    # Materialize once: NpzFile lookup would decompress the whole position array
    # for every edge in this independent verification loop.
    pos = dict(np.load(root/'data/positions.npz', allow_pickle=False)); verified_edges = 0
    for b, (episode, transition) in enumerate(common['pairs']):
        graph = common['J'][episode]; name = Path(index[common['episode_qids'][episode]]['path']).stem
        senders = {i: str(i) for i in range(32) if graph[i].any()}
        agree, disagree, _ = deliver(graph.tolist(), senders, name, int(transition)+1)
        for receiver in range(32):
            for slot, sender in enumerate(pos['neighbors'][episode, receiver]):
                sign = pos['signs'][episode, receiver, slot]
                if not sign:
                    np.testing.assert_array_equal(pos['actual'][b, receiver, slot], 0)
                    np.testing.assert_array_equal(pos['random'][b, receiver, slot], 0)
                    continue
                assert sign == graph[sender, receiver]
                sequence = agree[receiver] if sign == 1 else disagree[receiver]; rank = sequence.index(str(sender))
                np.testing.assert_allclose(pos['actual'][b, receiver, slot], [rank/max(len(sequence)-1, 1), rank == 0, rank == len(sequence)-1], atol=3e-8)
                verified_edges += 1
        if (b+1) % 800 == 0: print(json.dumps({'verified_transitions': b+1, 'verified_edges': verified_edges}), flush=True)
    torch.set_num_threads(2); data = study.Data(cfg, 'cpu', 'full')
    original_rows = study.read(root/'seed_by_question.json'); recomputed = {}; replays = []
    teacher = np.load('runs/generalization_investigation_320_v1/noise_audit_v1/probabilities.npz')['majority']
    expected_rows = []
    for item in results['prediction_files']:
        assert digest(item['path']) == item['sha256']
        d = np.load(item['path']); model, seed, split = item['model'], item['seed'], item['split']
        ids = data.ids(p['fold'][split+'_questions'])
        np.testing.assert_array_equal(d['pairs'], common['pairs'][ids]); np.testing.assert_array_equal(d['target'], data.y[ids])
        np.testing.assert_array_equal(d['qids'], data.qids[ids])
        lp = d['logp'].astype(float); y = d['target']; probability = np.exp(lp[..., 1])
        arrays = {'error': (lp.argmax(-1) != y).astype(float),
            'nll': -np.take_along_axis(lp, y[..., None], -1)[..., 0],
            'brier_two_class': 2*(probability-y)**2,
            'expected_accuracy': np.where(lp.argmax(-1) == 1, teacher[ids], 1-teacher[ids])}
        records = [r for r in original_rows if r['model'] == model and r['seed'] == seed and r['split'] == split]
        for row in records:
            take = data.qids[ids] == row['qid']; values = {m: float(v[take].mean()) for m, v in arrays.items()}
            for m in ('error', 'nll', 'brier_two_class'): max_metric_difference = max(max_metric_difference, abs(values[m]-row[m]))
            recomputed[model, split, row['qid'], seed] = values
            expected_rows.append(dict(model=model, split=split, qid=row['qid'], seed=seed, expected_accuracy=values['expected_accuracy']))
        if split == 'validation':
            checkpoint = torch.load(item['checkpoint'], map_location='cpu', weights_only=False)
            fitted = study.make_model(cfg, 'nano_pca8'); fitted.load_state_dict(checkpoint['model']); fitted.eval(); data.condition = model
            total_position_gradient = float(fitted.pair_encoder[0].weight[:, -3:].detach().abs().sum())
            with study.adapted_trainer(): replay = study.trainer.predictions(fitted, data, ids, 128)
            replays.append({'model': model, 'seed': seed, 'max_log_probability_difference': float(np.abs(replay-d['logp']).max()),
                'class_differences': int((replay.argmax(-1) != d['logp'].argmax(-1)).sum()),
                'position_weight_absolute_sum': total_position_gradient})
    averaged = {}
    for model in cfg['conditions']:
        for split in ('train', 'validation'):
            for q in p['fold'][split+'_questions']:
                averaged[model, split, q] = {m: float(np.mean([recomputed[model, split, q, seed][m] for seed in cfg['training']['refit_seeds']]))
                    for m in ('error', 'nll', 'brier_two_class', 'expected_accuracy')}
            for m, value in results['summary'][model][split].items():
                actual = np.mean([averaged[model, split, q][m] for q in p['fold'][split+'_questions']])
                max_metric_difference = max(max_metric_difference, abs(actual-value))
    for row in study.read(Path(cfg['state_reference'])/'by_question.json'):
        if row['model'] in ('I0', 'opinion_nano'): averaged[row['model'], row['split'], row['qid']] = row
    qs = sorted(p['fold']['validation_questions']); rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    draws = rng.integers(len(qs), size=(cfg['evaluation']['bootstrap_replicates'], len(qs)))
    for c in results['paired_differences']:
        delta = np.array([averaged[c['left'], 'validation', q][c['metric']]-averaged[c['right'], 'validation', q][c['metric']] for q in qs])
        max_metric_difference = max(max_metric_difference, abs(delta.mean()-c['difference']), max(abs(np.quantile(delta[draws].mean(1), [.025, .975])-c['ci95'])))
    assert max_metric_difference < 1e-10
    assert all(r['class_differences'] == 0 and r['max_log_probability_difference'] == 0 for r in replays)
    json_write(root/'expected_by_question.json', expected_rows)
    result = {'passed': True, 'completed_utc': study.now(), 'audit_code_sha256': digest(__file__),
        'results_sha256': digest(root/'results.json'), 'models_frozen_sha256': digest(root/'models_frozen.json'),
        'verified_directed_edge_positions': verified_edges, 'verified_prediction_files': len(results['prediction_files']),
        'verified_contrasts': len(results['paired_differences']), 'max_metric_difference': max_metric_difference,
        'checkpoint_replays': replays, 'assessment_loaded': False}
    json_write(root/'completion_audit.json', result); print(json.dumps(result), flush=True)

if __name__ == '__main__': main()
