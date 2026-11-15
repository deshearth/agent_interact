"""Training-only hypothesis screen for deterministic source-mode prediction."""
from argparse import ArgumentParser
from pathlib import Path
import json
import pickle
import subprocess
import sys
import time

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write, code_digest
from scripts import distribution_prediction_study as dist
from scripts import generalization_state_models as state
from scripts import generalization_lexical as lexical
from scripts import compact_argument_study as compact

ROOT = Path('runs/modal_accuracy_investigation_320_v1/state_screen_v1')
VARIANTS = ['current', 'initial', 'rich', 'lexical', 'semantic8', 'semantic32']
SETTINGS = dict(max_iter=600, learning_rate=.08, max_leaf_nodes=31, min_samples_leaf=64,
                l2_regularization=5., n_iter_no_change=40, tol=.00005, random_state=92641,
                scoring='accuracy', categorical_features=None)


def slots(values, order):
    assert values.ndim == 3 and order.ndim == 4
    b, n, channels = values.shape
    gathered = values[np.arange(b)[:, None, None, None], np.maximum(order, 0)]
    return (gathered*(order >= 0)[..., None]).reshape(b, n, -1)


def probability_features(single, pairs, graph, order):
    e, t = pairs.T; p = single[e, t]; initial = single[e, 0]
    assert ((single > 0)&(single < 1)).all()
    def summarize(values):
        logs = logit(values); parts = [values[..., None], logs[..., None]]
        for sign in (1, -1):
            mask = graph == sign; degree = np.maximum(mask.sum(-1), 1)
            mean = (mask*values[:, None, :]).sum(-1)/degree
            variance = (mask*(values[:, None, :]-mean[..., None])**2).sum(-1)/degree
            minimum = np.where(mask, values[:, None, :], np.inf).min(-1)
            maximum = np.where(mask, values[:, None, :], -np.inf).max(-1)
            minimum[~mask.any(-1)] = 0; maximum[~mask.any(-1)] = 0
            parts.extend([v[..., None] for v in (mean, np.sqrt(variance), minimum, maximum,
                (mask*logs[:, None, :]).sum(-1)/degree, (mask*(2*values[:, None, :]-1)).sum(-1))])
        parts.append(slots(np.stack([values, logs], -1), order))
        global_values = np.stack([values.mean(-1), values.std(-1)], -1)
        parts.append(np.broadcast_to(global_values[:, None], (*values.shape, 2)))
        return np.concatenate(parts, -1).astype(np.float32)
    current = summarize(p); initial_features = summarize(initial)
    # Known-persona population priors are measured before interaction, not IDs or answers.
    population_initial = np.broadcast_to(initial[:, None, :], (*p.shape, p.shape[-1]))
    initial_features = np.concatenate([initial_features, population_initial], -1).astype(np.float32)
    history = [single.cumsum(1)[e, t]/(t[:, None]+1)]
    for lag in (1, 2):
        history.extend([single[e, np.maximum(t-lag, 0)], np.broadcast_to((t >= lag)[:, None], p.shape)])
    history.extend([p-initial, np.broadcast_to((t/8)[:, None], p.shape)])
    return current, initial_features, np.stack(history, -1).astype(np.float32)


def prepare():
    if (ROOT/'prepared.json').exists(): return validate()
    parent = dist.validate(dist.config()); state.validate(state.config()); lexical.validate(lexical.config())
    ROOT.mkdir(parents=True, exist_ok=True)
    cfg = state.config(); common = dict(np.load(cfg['common'], allow_pickle=False)); e, t = common['pairs'].T
    index = {v['qid']: v for v in state.read(cfg['index'])}
    names = [Path(index[q]['path']).stem for q in common['episode_qids']]
    order = state.ordered_neighbors(common['J'], names, common['pairs'])
    source = dict(np.load(Path(dist.config()['probability_reference'])/'probabilities.npz', allow_pickle=False))
    current, initial, history = probability_features(source['single'], common['pairs'], common['J'][e], order)
    raw_lexical = np.load(lexical.ROOT/'sender_lexical.npy', allow_pickle=False)[e, t]
    lexical_slots = slots(raw_lexical, order)
    ids = np.load(cfg['message_ids'], allow_pickle=False); vectors = np.load(cfg['message_vectors'], mmap_mode='r')
    dimensions = {}; pca_metadata = {}
    with threadpool_limits(limits=4):
        for stage, questions in [('inner', parent['inner']['fit']), ('full', parent['fold']['train_questions'])]:
            folder = ROOT/'features'/stage; folder.mkdir(parents=True, exist_ok=True)
            old = Path(cfg['output'])/'features'/stage
            base = np.load(old/'order.npy').reshape(len(e), 32, -1)
            rich = np.load(old/'rich.npy').reshape(len(e), 32, -1)
            probability_state = np.concatenate([current, initial, history], -1)
            arrays = {'current': np.concatenate([base, current], -1),
                      'initial': np.concatenate([base, current, initial], -1),
                      'rich': np.concatenate([rich, probability_state], -1),
                      'lexical': np.concatenate([rich, probability_state, lexical_slots], -1)}
            reduced8 = np.load(Path(dist.config()['ordered_reference'])/'data'/f'vectors_{stage}.npy')
            fit_e = np.flatnonzero(np.isin(common['episode_qids'], questions))
            reduced32, fitted = compact.fit_message_pca(vectors, ids[fit_e], 32)
            np.savez(folder/'pca32.npz', **fitted)
            np.save(folder/'vectors32.npy', reduced32)
            for dim, reduced in [(8, reduced8), (32, reduced32)]:
                arrays['semantic'+str(dim)] = np.concatenate([arrays['lexical'], slots(reduced[ids[e, t]], order)], -1)
            pca_metadata[stage] = float(fitted['explained_variance_fraction'])
            for name, array in arrays.items():
                x = np.ascontiguousarray(array.reshape(-1, array.shape[-1]), dtype=np.float32)
                assert np.isfinite(x).all(); np.save(folder/(name+'.npy'), x); dimensions[name] = x.shape[-1]
    np.savez_compressed(ROOT/'labels.npz', probability=source['single'][e, t+1].ravel(),
        target=(source['single'][e, t+1] > .5).astype(np.int8).ravel(), qids=np.repeat(common['episode_qids'][e], 32),
        pairs=common['pairs'], current=source['single'][e, t], initial=source['single'][e, 0])
    references = [Path(dist.config()['output'])/f for f in ('protocol.json', 'prepared.json', 'completion_audit.json', 'results.json')]
    references += [Path(cfg['output'])/'prepared.json', lexical.ROOT/'prepared.json', Path(cfg['common']),
        Path(cfg['message_vectors']), Path(cfg['message_ids']), Path(dist.config()['probability_reference'])/'probabilities.npz']
    protocol = dict(created_utc=state.now(), fold=parent['fold'], inner=parent['inner'], variants=VARIANTS, settings=SETTINGS,
        code_files={str(p): digest(p) for p in [Path(__file__), Path('tests/test_modal_state_screen.py')]},
        core_sha256=code_digest(), references={str(p): digest(p) for p in references}, dimensions=dimensions,
        pca32_variance=pca_metadata, target='A iff next single-response conditional A/B p>0.5; retain all cases and existing histories.',
        hypotheses=['Direct classification may recover decision boundaries neglected by probability fidelity.',
                    'Initial source probabilities capture the no-discussion question/persona prior.',
                    'Ordered full slots and richer semantics retain information lost by averaged/PCA8 arguments.'],
        feature_policy='Current and initial exact probabilities plus strictly past probability history. Current shared arguments are already delivered. '
            'No next probabilities, next votes, answer keys or question IDs enter features. PCA fits only internal256 or final320. '
            'Initial32 probabilities describe the known-persona population before discussion; no future-state estimation is used.',
        selection='Training-only screen: one fixed booster recipe per variant, internal256/64 whole questions. '
            'Choose staged maximum accuracy, tie by mode-label NLL then fewer iterations. These are hypothesis-screen scores, not outer-validation results.',
        assessment_loaded=False, source_calls=0)
    json_write(ROOT/'protocol.json', protocol)
    files = list((ROOT/'features').rglob('*.npy'))+list((ROOT/'features').rglob('*.npz'))+[ROOT/'labels.npz']
    json_write(ROOT/'prepared.json', dict(protocol_sha256=digest(ROOT/'protocol.json'), artifacts={str(p.relative_to(ROOT)): digest(p) for p in files}))
    print(json.dumps({'prepared': True, 'dimensions': dimensions, 'pca32_variance': pca_metadata}), flush=True)
    return protocol


def validate():
    p = state.read(ROOT/'protocol.json'); prepared = state.read(ROOT/'prepared.json')
    assert p['settings'] == SETTINGS and p['variants'] == VARIANTS and p['core_sha256'] == code_digest()
    assert digest(ROOT/'protocol.json') == prepared['protocol_sha256']
    for group in ('code_files', 'references'):
        for path, sha in p[group].items(): assert digest(path) == sha, path
    for path, sha in prepared['artifacts'].items(): assert digest(ROOT/path) == sha, path
    return p


def worker(name):
    p = validate(); labels = dict(np.load(ROOT/'labels.npz')); qids, y = labels['qids'], labels['target']
    fit, val = [np.isin(qids, p['inner'][k]) for k in ('fit', 'validation')]
    assert not set(qids[fit]) & set(qids[val]); assert len(set(qids[fit])) == 256 and len(set(qids[val])) == 64
    folder = ROOT/'screen'/name; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists(): return
    x = np.load(ROOT/'features/inner'/f'{name}.npy', mmap_mode='r'); start = time.monotonic()
    model = HistGradientBoostingClassifier(**SETTINGS, early_stopping=True)
    best = None; history = []; chosen = None
    with threadpool_limits(limits=2):
        model.fit(x[fit], y[fit], X_val=x[val], y_val=y[val])
        for iteration, pred in enumerate(model.staged_predict_proba(x[val]), 1):
            pp = np.clip(pred[:, 1], 1e-10, 1-1e-10)
            error = float(((pp > .5) != y[val]).mean()); nll = float((-y[val]*np.log(pp)-(1-y[val])*np.log1p(-pp)).mean())
            history.append(dict(iteration=iteration, error=error, nll=nll)); key = error, nll, iteration
            if best is None or key < best: best, chosen = key, pp.copy()
        terminal_train = model.predict_proba(x[fit])[:, 1]
    np.save(folder/'inner_probability.npy', chosen); np.save(folder/'terminal_train_probability.npy', terminal_train)
    with (folder/'terminal_model.pkl').open('wb') as stream: pickle.dump(model, stream)
    json_write(folder/'history.json', history)
    record = dict(model=name, selected_iterations=best[2], inner_accuracy=1-best[0], inner_nll=best[1],
        terminal_training_accuracy=float(((terminal_train > .5) == y[fit]).mean()),
        fitted_iterations=model.n_iter_, seconds=time.monotonic()-start, completed_utc=state.now(),
        protocol_sha256=digest(ROOT/'protocol.json'), artifacts={p.name: digest(p) for p in folder.iterdir() if p.is_file()})
    json_write(folder/'complete.json', record); print(json.dumps(record), flush=True)


def run():
    prepare(); pending = VARIANTS.copy(); active = {}; complete = {}
    while pending or active:
        while pending and len(active) < 2:
            name = pending.pop(0); stream = (ROOT/(name+'.log')).open('a', buffering=1)
            active[name] = subprocess.Popen([sys.executable, __file__, 'worker', '--model', name], stdout=stream, stderr=subprocess.STDOUT), stream
        for name, (process, stream) in list(active.items()):
            if process.poll() is not None:
                stream.close(); complete[name] = process.returncode; del active[name]
        json_write(ROOT/'progress.json', dict(phase='screen', updated_utc=state.now(), active={k: p.pid for k, (p, s) in active.items()}, pending=pending, complete=complete))
        if any(complete.values()): raise RuntimeError(str(complete))
        if active: time.sleep(2)
    json_write(ROOT/'screen_results.json', dict(completed_utc=state.now(), models={name: state.read(ROOT/'screen'/name/'complete.json') for name in VARIANTS}, outer_validation_scored=False))
    json_write(ROOT/'progress.json', dict(phase='complete', updated_utc=state.now()))


if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'screen', 'worker']); parser.add_argument('--model', choices=VARIANTS)
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'screen': run()
    else: worker(args.model)
