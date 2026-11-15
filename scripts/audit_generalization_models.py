"""Independent recomputation of saved model metrics and selection barriers."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write

def read(path): return json.loads(Path(path).read_text())

def audit(root):
    root = Path(root); protocol = read(root/'protocol.json'); result = read(root/'results.json')
    frozen = read(root/'models_frozen.json'); selection = read(root/'selection.json')
    for path, sha in frozen['artifacts'].items(): assert digest(path) == sha, path
    for path, sha in protocol['code_files'].items(): assert digest(path) == sha, path
    assert result['models_frozen_sha256'] == digest(root/'models_frozen.json')
    assert frozen['frozen_utc'] < result['completed_utc']
    assert set(protocol['inner']['fit']).isdisjoint(protocol['inner']['validation'])
    assert set(protocol['inner']['fit']) | set(protocol['inner']['validation']) == set(protocol['fold']['train_questions'])
    assert set(protocol['fold']['train_questions']).isdisjoint(protocol['fold']['validation_questions'])
    assert len(protocol['fold']['assessment_questions']) == 40
    for choice in selection['choices']:
        records = [read(p) for p in (root/'tune'/choice['variant']).glob('*.json')]
        expected = min(records, key=lambda r: (r['inner_nll'], r['inner_error'], r['selected_iterations'], r['candidate']['id']))
        assert choice == expected
        refit = read(root/'refit'/choice['variant']/'complete.json')
        assert refit['choice'] == choice and refit['selection_sha256'] == digest(root/'selection.json')
        assert refit['completed_utc'] < frozen['frozen_utc']
    common = np.load('runs/question_breadth_320_train_v1/predictors/development.npz', allow_pickle=False)
    e, t = common['pairs'].T; qids = np.repeat(common['episode_qids'][e], 32)
    target = (common['spins'][e, t+1] == 1).ravel()
    teacher = np.load('runs/generalization_investigation_320_v1/noise_audit_v1/probabilities.npz')['majority'].ravel()
    by_question = read(root/'by_question.json'); fresh_rows = {}; differences = []; replay = []
    for item in result['prediction_files']:
        path = Path(item['path']); assert digest(path) == item['sha256']
        pred = np.load(path, allow_pickle=False); name, split = item['model'], item['split']
        take = np.isin(qids, protocol['fold'][split+'_questions'])
        np.testing.assert_array_equal(pred['qids'], qids[take]); np.testing.assert_array_equal(pred['target'], target[take])
        pp = pred['probability'].astype(np.float64); yy = target[take]
        assert np.isfinite(pp).all() and ((pp >= 0)&(pp <= 1)).all()
        clipped = np.clip(pp, 1e-10, 1-1e-10)
        measures = {'error': ((pp > .5) != yy).astype(float), 'nll': -np.where(yy, np.log(clipped), np.log1p(-clipped)),
            'brier_two_class': 2*(clipped-yy)**2, 'expected_accuracy': np.where(pp > .5, teacher[take], 1-teacher[take])}
        saved = {r['qid']: r for r in by_question if r['model'] == name and r['split'] == split}
        for q in np.unique(qids[take]):
            keep = qids[take] == q; row = {k: float(v[keep].mean()) for k, v in measures.items()}
            for metric in ('error', 'nll', 'brier_two_class'):
                differences.append(abs(row[metric]-saved[q][metric]))
            fresh_rows[name, split, q] = row
        for metric, value in result['summary'][name][split].items():
            expected = np.mean([fresh_rows[name, split, q][metric] for q in np.unique(qids[take])])
            differences.append(abs(expected-value))
        if split == 'validation':
            is_distillation = 'state_reference' in protocol['config']
            feature = name.rsplit('_', 1)[0] if is_distillation else name
            feature_root = Path(protocol['config']['state_reference']) if is_distillation else root
            x = np.load(feature_root/'features/full'/f'{feature}.npy', mmap_mode='r')
            with (root/'refit'/name/'model.pkl').open('rb') as stream: model = pickle.load(stream)
            with threadpool_limits(limits=4): again = model.predict_proba(x[take])[:, 1]
            replay.append({'model': name, 'max_probability_difference': float(np.abs(again-pp).max()),
                           'class_differences': int(((again > .5) != (pp > .5)).sum())})
    # Frozen comparison rows, including the old three-seed average, are also
    # checked against their prediction arrays instead of trusting summaries.
    old_manifest = read('runs/compact_arguments_320_v1/results/predictions.json')
    old_rows = {}
    for item in old_manifest:
        if item['model'] not in ('I0', 'opinion_nano'): continue
        assert digest(item['path']) == item['sha256']
        pred = np.load(item['path'], allow_pickle=False); lp = pred['logp'].astype(float); yy = pred['target']
        probability = np.exp(lp[..., 1]); prediction = lp.argmax(-1)
        for q in np.unique(pred['qids']):
            keep = pred['qids'] == q; y = yy[keep]; p = probability[keep]
            key = item['model'], item['split'], q
            old_rows.setdefault(key, []).append({'error': float((prediction[keep] != y).mean()),
                'nll': float(-np.take_along_axis(lp[keep], y[..., None], -1).mean()),
                'brier_two_class': float(((np.exp(lp[keep])-np.eye(2)[y])**2).sum(-1).mean())})
    for key, rows in old_rows.items(): fresh_rows[key] = {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
    if 'state_reference' in protocol['config']:
        reference = Path(protocol['config']['state_reference'])
        assert digest(reference/'prepared.json') == protocol['state_prepared_sha256']
        previous = read(reference/'completion_audit.json'); assert previous['passed']
        for r in read(reference/'by_question.json'):
            fresh_rows.setdefault((r['model'], r['split'], r['qid']), r)
        targets = np.load(root/'training_targets.npz', allow_pickle=False)
        assert set(targets['qids']) == set(protocol['fold']['train_questions'])
        assert set(targets['qids']).isdisjoint(protocol['fold']['validation_questions'])
        np.testing.assert_array_equal(targets['hard'], target[targets['row_ids']])
        assert len(targets['row_ids']) == 81920
    else:
        prepared = read(root/'prepared.json')
        for path, sha in prepared['artifacts'].items(): assert digest(root/path) == sha, path
        qvectors = np.load(protocol['config']['context_vectors'])['question_vectors']
        message_ids = np.load(protocol['config']['message_ids']); vectors = np.load(protocol['config']['message_vectors'], mmap_mode='r')
        for stage, questions in [('inner', protocol['inner']['fit']), ('full', protocol['fold']['train_questions'])]:
            fits = np.load(root/'features'/stage/'transforms.npz')
            fit_e = np.flatnonzero(np.isin(common['episode_qids'], questions))
            np.testing.assert_allclose(fits['question_mean'], qvectors[fit_e].astype(float).mean(0), atol=1e-8)
            np.testing.assert_allclose(fits['question32_mean'], qvectors[fit_e].astype(float).mean(0), atol=1e-8)
            expected_ids = np.unique(message_ids[fit_e]); expected_ids = expected_ids[expected_ids > 2]
            np.testing.assert_array_equal(expected_ids, fits['message_training_ids'])
            np.testing.assert_allclose(fits['message_mean'], vectors[expected_ids].astype(float).mean(0), atol=1e-10)
    if 'parent_state' in protocol['config']:
        parent = Path(protocol['config']['parent_state'])
        assert read(parent/'completion_audit.json')['passed']
        for path, sha in protocol['parent_artifacts'].items(): assert digest(path) == sha, path
        for r in read(parent/'by_question.json'):
            fresh_rows.setdefault((r['model'], r['split'], r['qid']), r)
        for stage in ('inner', 'full'):
            lexical = np.load(root/'features'/stage/'lexical.npy', mmap_mode='r')
            combined = np.load(root/'features'/stage/'combined.npy', mmap_mode='r')
            rich = np.load(parent/'features'/stage/'rich.npy', mmap_mode='r')
            arguments = np.load(parent/'features'/stage/'arguments.npy', mmap_mode='r')
            np.testing.assert_array_equal(lexical[:, :191], rich)
            np.testing.assert_array_equal(combined[:, :255], arguments)
            np.testing.assert_array_equal(lexical[:, -64:], combined[:, -64:])
        np.testing.assert_array_equal(np.load(root/'features/inner/lexical.npy', mmap_mode='r')[:, -64:],
                                      np.load(root/'features/full/lexical.npy', mmap_mode='r')[:, -64:])
    questions = sorted(protocol['fold']['validation_questions']); cfg = protocol['config']['evaluation']
    rng = np.random.default_rng(cfg['bootstrap_seed']); draws = rng.integers(len(questions), size=(cfg['bootstrap_replicates'], len(questions)))
    for contrast in result['paired_differences']:
        l, r, metric = contrast['left'], contrast['right'], contrast['metric']
        delta = np.array([fresh_rows[l, 'validation', q][metric]-fresh_rows[r, 'validation', q][metric] for q in questions])
        differences.append(abs(delta.mean()-contrast['difference']))
        differences.extend(abs(np.quantile(delta[draws].mean(1), [.025, .975])-contrast['ci95']))
    assert max(differences) < 1e-7, max(differences)
    assert max(r['max_probability_difference'] for r in replay) == 0
    output = {'passed': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'audit_code_sha256': digest(__file__),
        'results_sha256': digest(root/'results.json'), 'models_frozen_sha256': digest(root/'models_frozen.json'),
        'verified_prediction_files': len(result['prediction_files']), 'verified_contrasts': len(result['paired_differences']),
        'max_metric_difference': float(max(differences)), 'checkpoint_replay': replay,
        'training_only_transform_checks': True, 'selection_and_freeze_barrier_checks': True, 'assessment_loaded': False}
    json_write(root/'completion_audit.json', output); print(json.dumps(output), flush=True)

if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('root'); args = parser.parse_args(); audit(args.root)
