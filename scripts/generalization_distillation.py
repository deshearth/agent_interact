"""Matched hard/soft supervision, with training-only teacher probabilities."""
from argparse import ArgumentParser
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config as load_config, digest, json_write
from scripts import generalization_state_models as state

CONFIG = Path('configs/generalization_distillation_320.yaml')
def config(): return load_config(CONFIG)
read, now = state.read, state.now

def teacher_targets(q, hard, mass, bare, retries, maximum_mass):
    assert np.isfinite(q).all() and ((q >= -1e-12)&(q <= 1+1e-12)).all()
    q = np.clip(q, 0, 1)  # Polynomial evaluation can exceed 1 by ~1e-15.
    reliable = (mass <= maximum_mass)&(bare == 5)&(retries == 0)
    return np.where(reliable, q, hard), reliable

def expand(x, target):
    """Exact fractional Bernoulli loss via two equally located weighted rows.

    Total weight per original observation is one. Both hard and soft controls
    use this representation. Leaf row-count minimum is doubled accordingly.
    """
    n = len(target)
    assert x.shape[0] == n and np.isfinite(target).all() and ((target >= 0)&(target <= 1)).all()
    return np.concatenate([x, x]), np.concatenate([np.zeros(n), np.ones(n)]), np.concatenate([1-target, target])

def variants(cfg): return [(f+'_'+target, f, target) for f in cfg['features'] for target in cfg['targets']]

def prepare(cfg):
    root, reference = Path(cfg['output']), Path(cfg['state_reference'])
    if (root/'prepared.json').exists(): return validate(cfg)
    parent = state.validate(state.config()); assert reference == Path(state.config()['output'])
    noise = Path(cfg['noise_reference']); audit = read(noise/'audit.json')
    assert audit['actual_prompt_order_verified'] and audit['verified_prompts'] == 115200
    assert audit['probabilities_sha256'] == digest(noise/'probabilities.npz')
    labels = np.load(reference/'labels.npz', allow_pickle=False)
    teacher = np.load(noise/'probabilities.npz', allow_pickle=False)
    train_rows = np.flatnonzero(np.isin(labels['qids'], parent['fold']['train_questions']))
    e, t = teacher['pairs'].T
    target, reliable = teacher_targets(teacher['majority'].ravel()[train_rows], labels['target'][train_rows],
        teacher['other_mass'][e, t+1].ravel()[train_rows], teacher['initial_bare_count'][e, t+1].ravel()[train_rows],
        teacher['retry_count'][e, t+1].ravel()[train_rows], cfg['maximum_teacher_other_mass'])
    assert len(train_rows) == 81920 and set(labels['qids'][train_rows]) == set(parent['fold']['train_questions'])
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root/'training_targets.npz', row_ids=train_rows, qids=labels['qids'][train_rows],
             hard=labels['target'][train_rows], soft=target, teacher_reliable=reliable)
    code = [CONFIG, Path(__file__), Path('tests/test_generalization_distillation.py'), Path(state.__file__),
            Path('scripts/finish_generalization_distillation.py')]
    protocol = {'created_utc': now(), 'config': cfg, 'fold': parent['fold'], 'inner': parent['inner'],
        'code_files': {str(p): digest(p) for p in code}, 'state_prepared_sha256': digest(reference/'prepared.json'),
        'noise_audit_sha256': digest(noise/'audit.json'), 'training_targets_sha256': digest(root/'training_targets.npz'),
        'hypothesis': 'Precise source response probabilities as training labels reduce variance in supervision beyond enlarging state alone.',
        'feature_policy': 'Reuse frozen features and strictly training-fitted transforms from cycle one. No source probabilities as model inputs.',
        'teacher_policy': 'Only the 320 training questions appear in the training-target file. Replace majority outcome by source q when first-token '
            'outside-A/B mass <=.001, all five original responses are bare A/B and none needed retry. Otherwise keep the original hard label. '
            'This is label-quality fallback, not case exclusion; every training row remains. Validation teacher probabilities are evaluation only.',
        'training_policy': 'Both hard and soft conditions use exact duplicate-location weighted Bernoulli loss; row-count leaf minimum doubled to256. '
            'Early stopping and all six-recipe selection use OBSERVED hard-label NLL on the same internal64 training questions. '
            'Fit on internal256, then fixed-iteration refit on320. All six choices/refits freeze before any outer80 score. '
            'Report all variants; single deterministic booster per condition.',
        'reliable_training_targets': int(reliable.sum()), 'fallback_training_targets': int((~reliable).sum()),
        'assessment_loaded': False, 'source_calls': 0}
    json_write(root/'protocol.json', protocol); json_write(root/'prepared.json', {'protocol_sha256': digest(root/'protocol.json')})
    print(json.dumps({'prepared': True, 'reliable': int(reliable.sum()), 'fallback': int((~reliable).sum())}), flush=True)
    return protocol

def validate(cfg):
    root, reference = Path(cfg['output']), Path(cfg['state_reference']); p = read(root/'protocol.json')
    assert p['config'] == cfg and read(root/'prepared.json')['protocol_sha256'] == digest(root/'protocol.json')
    for path, sha in p['code_files'].items(): assert digest(path) == sha, path
    assert p['training_targets_sha256'] == digest(root/'training_targets.npz')
    assert p['state_prepared_sha256'] == digest(reference/'prepared.json')
    noise = Path(cfg['noise_reference'])
    assert p['noise_audit_sha256'] == digest(noise/'audit.json')
    assert read(noise/'audit.json')['probabilities_sha256'] == digest(noise/'probabilities.npz')
    for path, sha in read(reference/'prepared.json')['artifacts'].items(): assert digest(reference/path) == sha, path
    return p

def worker(cfg, variant, phase):
    p = validate(cfg); root, reference = Path(cfg['output']), Path(cfg['state_reference'])
    name, feature, target = next(v for v in variants(cfg) if v[0] == variant)
    targets = np.load(root/'training_targets.npz', allow_pickle=False)
    rows, qids, hard, y = targets['row_ids'], targets['qids'], targets['hard'], targets[target]
    folder = root/phase/variant; folder.mkdir(parents=True, exist_ok=True)
    if phase == 'tune':
        fit = np.isin(qids, p['inner']['fit']); val = np.isin(qids, p['inner']['validation'])
        x = np.load(reference/'features/inner'/f'{feature}.npy', mmap_mode='r')
        xfit, yfit, weights = expand(x[rows[fit]], y[fit]); xv, yv = x[rows[val]], hard[val]
        for candidate in state.candidates(cfg):
            path = folder/(candidate['id']+'.json')
            if path.exists(): continue
            start = time.monotonic(); model = state.model(cfg, candidate)
            with threadpool_limits(limits=cfg['training']['threads']):
                model.fit(xfit, yfit, sample_weight=weights, X_val=xv, y_val=yv)
                best, chosen = None, None
                for iteration, pred in enumerate(model.staged_predict_proba(xv), 1):
                    probability = np.clip(pred[:, 1], 1e-10, 1-1e-10)
                    nll = float((-yv*np.log(probability)-(1-yv)*np.log1p(-probability)).mean())
                    error = float(((probability > .5) != yv).mean()); key = (nll, error, iteration)
                    if best is None or key < best: best, chosen = key, probability.copy()
            np.save(folder/(candidate['id']+'.npy'), chosen)
            record = {'variant': variant, 'candidate': candidate, 'selected_iterations': best[2], 'inner_nll': best[0],
                'inner_error': best[1], 'fitted_iterations': model.n_iter_, 'seconds': time.monotonic()-start,
                'completed_utc': now(), 'protocol_sha256': digest(root/'protocol.json')}
            json_write(path, record); print(json.dumps(record), flush=True)
    else:
        if (folder/'complete.json').exists(): return
        choice = next(c for c in read(root/'selection.json')['choices'] if c['variant'] == variant)
        x = np.load(reference/'features/full'/f'{feature}.npy', mmap_mode='r')
        start = time.monotonic(); model = state.model(cfg, choice['candidate'], choice['selected_iterations'])
        xx, yy, weights = expand(x[rows], y)
        with threadpool_limits(limits=cfg['training']['threads']):
            model.fit(xx, yy, sample_weight=weights); probability = model.predict_proba(x[rows])[:, 1]
        with (folder/'model.pkl').open('wb') as stream: pickle.dump(model, stream)
        np.save(folder/'train_probability.npy', probability)
        metric, _ = state.score(probability, hard, qids)
        json_write(folder/'complete.json', {'variant': variant, 'choice': choice, 'train': metric,
            'seconds': time.monotonic()-start, 'completed_utc': now(), 'selection_sha256': digest(root/'selection.json'),
            'model_sha256': digest(folder/'model.pkl'), 'train_probability_sha256': digest(folder/'train_probability.npy'),
            'tree_count': len(model._predictors), 'tree_nodes': sum(len(tree[0].nodes) for tree in model._predictors)})

def evaluate(cfg):
    p = validate(cfg); root, reference = Path(cfg['output']), Path(cfg['state_reference'])
    frozen = read(root/'models_frozen.json')
    for path, sha in frozen['artifacts'].items(): assert digest(path) == sha, path
    if (root/'results.json').exists(): return
    labels = np.load(reference/'labels.npz', allow_pickle=False)
    teacher = np.load(Path(cfg['noise_reference'])/'probabilities.npz', allow_pickle=False)['majority'].ravel()
    y, qids = labels['target'], labels['qids']; all_rows, summary, prediction_files = [], {}, []
    for name, feature, target in variants(cfg):
        with (root/'refit'/name/'model.pkl').open('rb') as stream: model = pickle.load(stream)
        x = np.load(reference/'features/full'/f'{feature}.npy', mmap_mode='r'); summary[name] = {}
        for split in ('train', 'validation'):
            take = np.isin(qids, p['fold'][split+'_questions'])
            with threadpool_limits(limits=cfg['training']['threads']): probability = model.predict_proba(x[take])[:, 1]
            metrics, rows = state.score(probability, y[take], qids[take])
            expected = np.where(probability > .5, teacher[take], 1-teacher[take])
            for row in rows: row['expected_accuracy'] = float(expected[qids[take] == row['qid']].mean())
            metrics['expected_accuracy'] = float(np.mean([r['expected_accuracy'] for r in rows]))
            summary[name][split] = metrics; all_rows += [dict(model=name, split=split, **r) for r in rows]
            path = root/'predictions'/f'{name}-{split}.npz'; path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, probability=probability, target=y[take], qids=qids[take])
            prediction_files.append({'model': name, 'split': split, 'path': str(path), 'sha256': digest(path)})
    previous = read(reference/'by_question.json')
    all_rows += previous
    grouped = {(r['model'], r['qid']): r for r in all_rows if r['split'] == 'validation'}
    questions = sorted(p['fold']['validation_questions']); rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    draws = rng.integers(len(questions), size=(cfg['evaluation']['bootstrap_replicates'], len(questions)))
    contrasts = []
    pairs = [(f+'_soft', f+'_hard') for f in cfg['features']]
    pairs += [(name, 'opinion_nano') for name, _, _ in variants(cfg)]
    pairs += [(f+'_hard', f) for f in cfg['features']]
    for left, right in pairs:
        for metric in ('error', 'nll', 'brier_two_class'):
            delta = np.array([grouped[left, q][metric]-grouped[right, q][metric] for q in questions])
            contrasts.append({'left': left, 'right': right, 'metric': metric, 'difference': float(delta.mean()),
                'ci95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist()})
    json_write(root/'by_question.json', all_rows)
    json_write(root/'results.json', {'completed_utc': now(), 'summary': summary, 'paired_differences': contrasts,
        'prediction_files': prediction_files, 'models_frozen_sha256': digest(root/'models_frozen.json'),
        'assessment_loaded': False, 'scope': 'Exploratory reused80; all tuning limited to320 training questions.'})
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'worker', 'evaluate'])
    parser.add_argument('--variant'); parser.add_argument('--phase', choices=['tune', 'refit']); args = parser.parse_args(); cfg = config()
    if args.action == 'prepare': prepare(cfg)
    elif args.action == 'evaluate': evaluate(cfg)
    else: worker(cfg, args.variant, args.phase)
