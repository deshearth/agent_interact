"""Whole-question internal tuning for ordered/history/context state hypotheses."""
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config as load_config, digest, json_write, code_digest
from scripts import compact_argument_study as compact
from scripts import transformer_generalization as base

CONFIG = Path('configs/generalization_state_models_320.yaml')
def read(path): return json.loads(Path(path).read_text())
def config(): return load_config(CONFIG)
def now(): return datetime.now(timezone.utc).isoformat()

def ordered_neighbors(graph, names, pairs, independent=False):
    """Actual documented inbox permutation, split into friendly/unfriendly groups.

    Only position's sender opinion becomes a feature, never the question ID or
    shuffle seed. The independent permutation is a matched information control.
    """
    result = np.full((len(pairs), graph.shape[1], 2, 9), -1, dtype=np.int16)
    assert (graph != 0).sum(-1).max() <= 9
    for b, (e, t) in enumerate(pairs):
        for receiver in range(graph.shape[1]):
            incoming = np.flatnonzero(graph[e, :, receiver]).tolist()
            seed = '{}|{}|{}'.format(names[e], t+1, receiver)
            if independent: seed = 'independent-order-control|'+seed
            random.Random(seed).shuffle(incoming)
            for slot, sign in enumerate((1, -1)):
                group = [sender for sender in incoming if graph[e, sender, receiver] == sign]
                result[b, receiver, slot, :len(group)] = group
    return result

def aggregate(mask, values):
    return np.einsum('bij,bjf->bif', mask, values, optimize=True)/np.maximum(mask.sum(-1, keepdims=True), 1)

def state_features(common, context, order, random_order):
    """No target-round states/votes are read to form a transition's features."""
    e, t = common['pairs'].T; s = common['spins'][e, t].astype(np.float32)
    votes = (common['raw_votes'] == 1).mean(-1).astype(np.float32)
    current = votes[e, t]; initial = votes[e, 0]
    cumulative = votes.cumsum(1)[e, t]/(t[:, None]+1)
    flips = np.concatenate([np.zeros_like(common['spins'][:, :1]),
        (common['spins'][:, 1:] != common['spins'][:, :-1]).cumsum(1)], 1)[e, t]/np.maximum(t[:, None], 1)
    graph = common['J'][e]; positive, negative = (graph > 0).astype(np.float32), (graph < 0).astype(np.float32)
    c = context[e]; parts = [c, s[..., None]]; names = ['context_'+str(i) for i in range(c.shape[-1])] + ['own_spin']
    for label, mask in [('friendly', positive), ('unfriendly', negative)]:
        degree = mask.sum(-1, keepdims=True)
        parts += [degree, (mask*(s[:, None, :] == 1)).sum(-1, keepdims=True), aggregate(mask, s[..., None]),
                  aggregate(mask, c), aggregate(mask, c*s[..., None])]
        names += [label+'_degree', label+'_A_count', label+'_mean_spin']
        names += [label+'_mean_context_'+str(i) for i in range(c.shape[-1])]
        names += [label+'_spin_context_'+str(i) for i in range(c.shape[-1])]
    base_x = np.concatenate(parts, -1).astype(np.float32)
    positions = []
    for mapping in (order, random_order):
        live = mapping >= 0
        positions.append((s[np.arange(len(s))[:, None, None, None], np.maximum(mapping, 0)]*live).reshape(*s.shape, -1))
    extra = [current[..., None], (2*np.abs(current-.5))[..., None], initial[..., None], cumulative[..., None], flips[..., None]]
    extra_names = ['current_vote_A_fraction', 'current_vote_confidence', 'initial_vote_A_fraction', 'past_vote_A_mean', 'past_flip_fraction']
    for lag in (1, 2):
        previous = np.maximum(t-lag, 0)
        extra += [common['spins'][e, previous, :, None], votes[e, previous, :, None], np.broadcast_to((t >= lag)[:, None, None], (*s.shape, 1))]
        extra_names += [f'lag{lag}_spin', f'lag{lag}_vote_fraction', f'lag{lag}_available']
    extra += [np.broadcast_to(t[:, None, None]/8, (*s.shape, 1))]
    extra_names += ['round_fraction']
    for label, mask in [('friendly', positive), ('unfriendly', negative)]:
        for name, v in [('current_votes', current), ('initial_votes', initial), ('past_votes', cumulative), ('past_flips', flips)]:
            extra.append(aggregate(mask, v[..., None])); extra_names.append(label+'_'+name)
    history = np.concatenate(extra, -1).astype(np.float32)
    return base_x, positions, history, names, extra_names

class Data:
    def __init__(self, cfg):
        self.cfg = cfg; self.common = dict(np.load(cfg['common'], allow_pickle=False))
        self.qids = self.common['episode_qids']; self.e, self.t = self.common['pairs'].T
        self.y = (self.common['spins'][self.e, self.t+1] == 1).astype(np.int64).ravel()
        self.row_qids = np.repeat(self.qids[self.e], 32)
        self.index = {x['qid']: x for x in read(cfg['index'])}
        assert set(self.index) == set(self.qids)
        self.names = [Path(self.index[q]['path']).stem for q in self.qids]
        arrays = np.load(cfg['context_vectors'], allow_pickle=False)
        self.qvectors, self.pvectors = arrays['question_vectors'], arrays['persona_vectors']
        self.message_ids = np.load(cfg['message_ids'], allow_pickle=False)
        self.vectors = np.load(cfg['message_vectors'], mmap_mode='r', allow_pickle=False)
        self.order = ordered_neighbors(self.common['J'], self.names, self.common['pairs'])
        self.random_order = ordered_neighbors(self.common['J'], self.names, self.common['pairs'], True)
        assert np.array_equal(self.common['J'], self.common['J'].transpose(0, 2, 1))

    def rows(self, questions): return np.flatnonzero(np.isin(self.row_qids, questions))

    def features(self, fit_questions):
        context = base.fit_context(dict(zip(self.qids, self.qvectors)), self.pvectors, fit_questions, self.qids)
        b, positions, history, names, history_names = state_features(self.common, context['context'], self.order, self.random_order)
        _, qmean, qaxes = base.pca_fit_transform(self.qvectors, np.flatnonzero(np.isin(self.qids, fit_questions)), width=32)
        qfeatures = (self.qvectors-qmean)@qaxes.T
        persona = np.broadcast_to(np.eye(32, dtype=np.float32)[None], (len(self.e), 32, 32))
        question = np.broadcast_to(qfeatures[self.e, None], (len(self.e), 32, 32))
        current_votes = (self.common['raw_votes'][self.e, self.t] == 1).mean(-1)
        initial_votes = (self.common['raw_votes'][self.e, 0] == 1).mean(-1)
        globals_ = np.stack([current_votes.mean(-1), initial_votes.mean(-1)], -1)
        globals_ = np.broadcast_to(globals_[:, None], (len(self.e), 32, 2))
        rich = np.concatenate([b, positions[0], history, question, persona, globals_], -1).astype(np.float32)
        fit_e = np.flatnonzero(np.isin(self.qids, fit_questions))
        reduced, pca = compact.fit_message_pca(self.vectors, self.message_ids[fit_e], 8)
        messages = reduced[self.message_ids[self.e, self.t]]*math.sqrt(8)
        graph = self.common['J'][self.e]; s = self.common['spins'][self.e, self.t]
        text_parts = []
        for sign in (1, -1):
            for stance in (1, -1):
                mask = ((graph == sign)&(s[:, None, :] == stance)).astype(np.float32)
                text_parts.append(aggregate(mask, messages))
            group = self.order[:, :, 0 if sign == 1 else 1]
            counts = (group >= 0).sum(-1)
            for location in ('first', 'last'):
                sender = group[..., 0] if location == 'first' else np.take_along_axis(group, np.maximum(counts-1, 0)[..., None], -1)[..., 0]
                text_parts.append(messages[np.arange(len(s))[:, None], np.maximum(sender, 0)]*(counts > 0)[..., None])
        variants = {'base': b, 'order': np.concatenate([b, positions[0]], -1),
            'random_order': np.concatenate([b, positions[1]], -1), 'history': np.concatenate([b, history], -1),
            'rich': rich, 'arguments': np.concatenate([rich, *text_parts], -1)}
        result = {k: np.ascontiguousarray(v.reshape(-1, v.shape[-1]), dtype=np.float32) for k, v in variants.items()}
        assert all(np.isfinite(v).all() for v in result.values())
        fits = {**context, 'question32_mean': qmean, 'question32_axes': qaxes,
                'message_mean': pca['mean'], 'message_axes': pca['components'], 'message_scale': pca['scale'],
                'message_training_ids': pca['training_ids']}
        return result, fits, {'base': names, 'history_extra': history_names,
            'dimensions': {k: v.shape[-1] for k, v in result.items()}}

def score(p, target, qids):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-10, 1-1e-10)
    q = sorted(set(qids)); per = []
    for key in q:
        take = qids == key; yy = target[take]; pp = p[take]
        per.append({'qid': str(key), 'error': float(((pp > .5) != yy).mean()),
            'nll': float((-yy*np.log(pp)-(1-yy)*np.log1p(-pp)).mean()),
            'brier_two_class': float((2*(pp-yy)**2).mean())})
    return {m: float(np.mean([r[m] for r in per])) for m in ('error', 'nll', 'brier_two_class')}, per

def prepare(cfg):
    root = Path(cfg['output']); protocol_path = root/'protocol.json'
    if protocol_path.exists(): return validate(cfg)
    reference = compact.validate(compact.config()); root.mkdir(parents=True, exist_ok=True)
    data = Data(cfg); train = reference['fold']['train_questions']
    strata = [data.index[q]['graph_id']+'|'+data.index[q]['origin'] for q in train]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=cfg['inner_validation_questions'], random_state=cfg['split_seed'])
    fit, val = next(splitter.split(train, strata))
    inner = {'fit': sorted(np.array(train)[fit].tolist()), 'validation': sorted(np.array(train)[val].tolist())}
    code = [CONFIG, Path(__file__), Path(compact.__file__), Path(base.__file__), Path('tests/test_generalization_state_models.py'),
            Path('scripts/local_conversation_pilot.py'), Path('data/raw/upstream/lib/datagen/samplers.py')]
    inputs = [Path(cfg[k]) for k in ('common', 'context_vectors', 'message_vectors', 'message_ids', 'index')]
    inputs += [Path(cfg['reference'])/'results/summary.json', Path(cfg['reference'])/'results/by_question.json', Path(cfg['reference'])/'protocol.json']
    protocol = {'created_utc': now(), 'config': cfg, 'config_sha256': digest(CONFIG), 'core_sha256': code_digest(),
        'code_files': {str(p): digest(p) for p in code}, 'inputs': {str(p): digest(p) for p in inputs},
        'fold': reference['fold'], 'inner': inner,
        'hypotheses': ['Nonlinear decision boundaries missed by small attention.', 'Source inbox order omitted by permutation-invariant aggregation.',
            'Current vote confidence and past responses carry stable response tendencies.', 'Compressed question/persona features omit transferable context.',
            'Arguments may help with receiver-specific nonlinear summaries and exact order.'],
        'feature_policy': 'No target-round votes/opinions, question IDs, correct answers, source probabilities or RNG seeds in features. '
            'Order comes from the saved generator; original and independent-order control have equal dimensions. '
            'History is past/current only. Rich adds training-only PCA32 of frozen question vectors, known-persona one-hot and current/initial global vote means. '
            'Arguments adds training-only PCA8 means by friendship/stance and first/last messages. Frozen MiniLM.',
        'selection_policy': '256/64 whole-question internal training split, stratified by graph and retained/new origin. '
            'All transforms fit only 256 questions during tuning, then 320 for final refits. Six tree recipes per variant. '
            'Choose lowest internal NLL across staged predictions, ties by accuracy then shorter duration; all six variants reported. '
            'No outer-validation outcomes used for this cycle selection. Deterministic boosting uses one seed, not a neural three-seed ensemble.',
        'assessment_loaded': False, 'source_calls': 0}
    json_write(protocol_path, protocol)
    for stage, questions in [('inner', inner['fit']), ('full', train)]:
        with threadpool_limits(limits=cfg['training']['threads']): arrays, fitted, metadata = data.features(questions)
        folder = root/'features'/stage; folder.mkdir(parents=True, exist_ok=True)
        for name, x in arrays.items(): np.save(folder/(name+'.npy'), x, allow_pickle=False)
        np.savez(folder/'transforms.npz', **fitted); json_write(folder/'metadata.json', metadata)
    np.savez(root/'labels.npz', target=data.y, qids=data.row_qids, pairs=data.common['pairs'])
    json_write(root/'prepared.json', {'protocol_sha256': digest(protocol_path),
        'artifacts': {str(p.relative_to(root)): digest(p) for p in (root/'features').rglob('*') if p.is_file()},
        'labels_sha256': digest(root/'labels.npz')})
    print(json.dumps({'prepared': True, 'inner_questions': [len(inner['fit']), len(inner['validation'])],
                      'dimensions': metadata['dimensions']}), flush=True)
    return protocol

def validate(cfg):
    root = Path(cfg['output']); p = read(root/'protocol.json')
    assert p['config'] == cfg and p['config_sha256'] == digest(CONFIG) and p['core_sha256'] == code_digest()
    for group in ('code_files', 'inputs'):
        for path, sha in p[group].items(): assert digest(path) == sha, path
    prepared = read(root/'prepared.json')
    assert prepared['protocol_sha256'] == digest(root/'protocol.json') and prepared['labels_sha256'] == digest(root/'labels.npz')
    for path, sha in prepared['artifacts'].items(): assert digest(root/path) == sha, path
    return p

def candidates(cfg):
    return [{'id': f'leaf{leaf}-l2{int(l2)}', 'max_leaf_nodes': leaf, 'l2_regularization': l2}
            for leaf in cfg['training']['max_leaf_nodes'] for l2 in cfg['training']['l2_regularization']]

def model(cfg, candidate, maximum=None):
    tc = cfg['training']
    return HistGradientBoostingClassifier(learning_rate=tc['learning_rate'], max_iter=maximum or tc['max_iter'],
        max_leaf_nodes=candidate['max_leaf_nodes'], min_samples_leaf=tc['min_samples_leaf'],
        l2_regularization=candidate['l2_regularization'], early_stopping=maximum is None,
        n_iter_no_change=tc['n_iter_no_change'], tol=tc['tolerance'], random_state=cfg['random_seed'],
        scoring='loss', categorical_features=None)

def worker(cfg, variant, phase):
    p = validate(cfg); root = Path(cfg['output']); labels = np.load(root/'labels.npz', allow_pickle=False)
    y, qids = labels['target'], labels['qids']; folder = root/phase/variant; folder.mkdir(parents=True, exist_ok=True)
    if phase == 'tune':
        fit = np.isin(qids, p['inner']['fit']); val = np.isin(qids, p['inner']['validation'])
        x = np.load(root/'features/inner'/f'{variant}.npy', mmap_mode='r')
        for candidate in candidates(cfg):
            path = folder/(candidate['id']+'.json')
            if path.exists(): continue
            start = time.monotonic(); fitted = model(cfg, candidate)
            with threadpool_limits(limits=cfg['training']['threads']):
                fitted.fit(x[fit], y[fit], X_val=x[val], y_val=y[val])
                best, chosen = None, None
                yy = y[val]
                for iteration, probabilities in enumerate(fitted.staged_predict_proba(x[val]), start=1):
                    pp = np.clip(probabilities[:, 1], 1e-10, 1-1e-10)
                    nll = float((-yy*np.log(pp)-(1-yy)*np.log1p(-pp)).mean()); error = float(((pp > .5) != yy).mean())
                    key = (nll, error, iteration)
                    if best is None or key < best: best, chosen = key, pp.copy()
            np.save(folder/(candidate['id']+'.npy'), chosen)
            json_write(path, {'variant': variant, 'candidate': candidate, 'selected_iterations': best[2],
                'inner_nll': best[0], 'inner_error': best[1], 'fitted_iterations': fitted.n_iter_,
                'seconds': time.monotonic()-start, 'completed_utc': now(), 'protocol_sha256': digest(root/'protocol.json')})
            print(json.dumps(read(path)), flush=True)
    else:
        choice = next(c for c in read(root/'selection.json')['choices'] if c['variant'] == variant)
        if (folder/'complete.json').exists(): return
        fit = np.isin(qids, p['fold']['train_questions']); x = np.load(root/'features/full'/f'{variant}.npy', mmap_mode='r')
        start = time.monotonic(); fitted = model(cfg, choice['candidate'], choice['selected_iterations'])
        with threadpool_limits(limits=cfg['training']['threads']):
            fitted.fit(x[fit], y[fit]); train_p = fitted.predict_proba(x[fit])[:, 1]
        with (folder/'model.pkl').open('wb') as stream: pickle.dump(fitted, stream)
        np.save(folder/'train_probability.npy', train_p)
        metric, _ = score(train_p, y[fit], qids[fit])
        json_write(folder/'complete.json', {'variant': variant, 'choice': choice, 'train': metric,
            'seconds': time.monotonic()-start, 'completed_utc': now(), 'selection_sha256': digest(root/'selection.json'),
            'model_sha256': digest(folder/'model.pkl'), 'train_probability_sha256': digest(folder/'train_probability.npy'),
            'tree_count': len(fitted._predictors),
            'tree_nodes': sum(len(tree[0].nodes) for tree in fitted._predictors)})
        print(json.dumps(read(folder/'complete.json')), flush=True)

def evaluate(cfg):
    p = validate(cfg); root = Path(cfg['output']); frozen = read(root/'models_frozen.json')
    for path, sha in frozen['artifacts'].items(): assert digest(path) == sha, path
    if (root/'results.json').exists(): return
    labels = np.load(root/'labels.npz', allow_pickle=False); y, qids = labels['target'], labels['qids']
    summaries, per_question, sources = {}, [], []
    for variant in cfg['variants']:
        with (root/'refit'/variant/'model.pkl').open('rb') as stream: fitted = pickle.load(stream)
        x = np.load(root/'features/full'/f'{variant}.npy', mmap_mode='r')
        summaries[variant] = {}
        for split in ('train', 'validation'):
            take = np.isin(qids, p['fold'][split+'_questions'])
            with threadpool_limits(limits=cfg['training']['threads']): probabilities = fitted.predict_proba(x[take])[:, 1]
            metric, rows = score(probabilities, y[take], qids[take]); summaries[variant][split] = metric
            per_question += [dict(model=variant, split=split, **row) for row in rows]
            file = root/'predictions'/f'{variant}-{split}.npz'; file.parent.mkdir(exist_ok=True)
            np.savez_compressed(file, probability=probabilities, target=y[take], qids=qids[take])
            sources.append({'model': variant, 'split': split, 'path': str(file), 'sha256': digest(file)})
    # Frozen old scores average seed losses rather than forming an ensemble.
    old = read(Path(cfg['reference'])/'results/by_question.json')
    for oldname in ('I0', 'opinion_nano'):
        summaries[oldname] = {}
        for split in ('train', 'validation'):
            grouped = {}
            for row in old:
                if row['model'] == oldname and row['split'] == split: grouped.setdefault(row['qid'], []).append(row)
            rows = [dict(model=oldname, split=split, qid=q, **{m: float(np.mean([r[m] for r in group]))
                    for m in ('error', 'nll', 'brier_two_class')}) for q, group in sorted(grouped.items())]
            per_question += rows; summaries[oldname][split] = {m: float(np.mean([r[m] for r in rows])) for m in ('error', 'nll', 'brier_two_class')}
    questions = sorted(p['fold']['validation_questions']); rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    draws = rng.integers(len(questions), size=(cfg['evaluation']['bootstrap_replicates'], len(questions)))
    grouped = {(r['model'], r['qid']): r for r in per_question if r['split'] == 'validation'}
    pairs = [(v, 'opinion_nano') for v in cfg['variants']] + [('order', 'random_order'), ('order', 'base'),
        ('history', 'base'), ('rich', 'order'), ('arguments', 'rich')]
    contrasts = []
    for left, right in pairs:
        for metric in ('error', 'nll', 'brier_two_class'):
            delta = np.array([grouped[left, q][metric]-grouped[right, q][metric] for q in questions])
            contrasts.append({'left': left, 'right': right, 'metric': metric, 'difference': float(delta.mean()),
                             'ci95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist()})
    json_write(root/'by_question.json', per_question)
    json_write(root/'results.json', {'completed_utc': now(), 'summary': summaries, 'paired_differences': contrasts,
        'prediction_files': sources, 'models_frozen_sha256': digest(root/'models_frozen.json'),
        'assessment_loaded': False, 'scope': 'Exploratory reused validation; inner selection uses training questions only.'})
    print(json.dumps(summaries), flush=True)

if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'worker', 'evaluate'])
    parser.add_argument('--variant'); parser.add_argument('--phase', choices=['tune', 'refit']); args = parser.parse_args(); cfg = config()
    if args.action == 'prepare': prepare(cfg)
    elif args.action == 'evaluate': evaluate(cfg)
    else: worker(cfg, args.variant, args.phase)
