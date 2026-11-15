"""Test explicit option language against same-width PCA8 argument summaries."""
from argparse import ArgumentParser
import fcntl
import json
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write, environment
from scripts import generalization_state_models as state

ROOT = Path('runs/generalization_investigation_320_v1/lexical_v1')
READ = state.read
MENTION = re.compile(r'(?<![A-Za-z0-9])([AB])(?![A-Za-z0-9])')
EXPLICIT = re.compile(r'\b(?:option|choice|answer)\s*(?:is\s*|[:=]\s*)?[\(\*]*([AB])\b', re.I)

def config():
    cfg = state.config(); cfg.update(output=str(ROOT), parent_state=cfg['output'], variants=['lexical', 'combined'])
    return cfg

def text_features(text):
    mentions = MENTION.findall(text); explicit = [m.upper() for m in EXPLICIT.findall(text)]
    sign = lambda m: 1 if m == 'A' else -1
    return np.array([np.log1p(mentions.count('A')), np.log1p(mentions.count('B')),
        sign(mentions[0]) if mentions else 0, sign(mentions[-1]) if mentions else 0,
        np.log1p(explicit.count('A')), np.log1p(explicit.count('B')),
        sign(explicit[-1]) if explicit else 0, np.log1p(len(text.split()))], dtype=np.float32)

def prepare(cfg):
    if (ROOT/'prepared.json').exists(): return validate(cfg)
    parent = state.validate(state.config()); parent_root = Path(cfg['parent_state'])
    common = dict(np.load(cfg['common'], allow_pickle=False)); index = {i['qid']: i for i in READ(cfg['index'])}
    messages = np.zeros((400, 8, 32, 8), dtype=np.float32); names = []
    training_counts = {'messages': 0, 'explicit_mentions': 0, 'last_explicit_differs_current_stance': 0}
    for e, q in enumerate(common['episode_qids']):
        item = index[q]; path = Path(item['path']); assert digest(path) == item['sha256']
        ep = READ(path); names.append(ep['meta']['name'])
        for t, edges in enumerate(ep['messages_history']):
            senders = {}
            for edge, text in edges.items():
                sender = int(edge.split(',')[0]); assert sender not in senders or senders[sender] == text
                senders[sender] = text
            for sender, text in senders.items():
                messages[e, t, sender] = text_features(text)
                if item['split'] == 'train':
                    training_counts['messages'] += 1
                    explicit = messages[e, t, sender, 6]
                    training_counts['explicit_mentions'] += int(explicit != 0)
                    training_counts['last_explicit_differs_current_stance'] += int(explicit != 0 and explicit != common['spins'][e, t, sender])
    e, t = common['pairs'].T; graph = common['J'][e]; s = common['spins'][e, t]
    values = messages[e, t]; order = state.ordered_neighbors(common['J'], names, common['pairs']); parts = []
    for sign in (1, -1):
        for stance in (1, -1): parts.append(state.aggregate(((graph == sign)&(s[:, None, :] == stance)).astype(np.float32), values))
        group = order[:, :, 0 if sign == 1 else 1]; counts = (group >= 0).sum(-1)
        for last in (False, True):
            sender = np.take_along_axis(group, np.maximum(counts-1, 0)[..., None], -1)[..., 0] if last else group[..., 0]
            parts.append(values[np.arange(len(s))[:, None], np.maximum(sender, 0)]*(counts > 0)[..., None])
    lexical = np.concatenate(parts, -1).reshape(-1, 64)
    ROOT.mkdir(parents=True, exist_ok=True)
    for stage in ('inner', 'full'):
        folder = ROOT/'features'/stage; folder.mkdir(parents=True, exist_ok=True)
        for name, source in [('lexical', 'rich'), ('combined', 'arguments')]:
            features = np.load(parent_root/'features'/stage/f'{source}.npy', mmap_mode='r')
            np.save(folder/f'{name}.npy', np.concatenate([features, lexical], -1))
        shutil.copyfile(parent_root/'features'/stage/'transforms.npz', folder/'transforms.npz')
    np.save(ROOT/'sender_lexical.npy', messages)
    shutil.copyfile(parent_root/'labels.npz', ROOT/'labels.npz')
    protocol = {'created_utc': state.now(), 'config': cfg, 'fold': parent['fold'], 'inner': parent['inner'],
        'code_files': {str(p): digest(p) for p in [Path(__file__), Path(state.__file__), Path('tests/test_generalization_lexical.py')]},
        'parent_artifacts': {str(parent_root/p): digest(parent_root/p) for p in ('protocol.json', 'prepared.json', 'completion_audit.json', 'results.json', 'by_question.json')},
        'source_inputs': parent['inputs'], 'training_text_diagnostic': training_counts,
        'hypothesis': 'Option-specific wording is blurred by semantic PCA8, so explicitly representing it improves transfer.',
        'features': 'Eight fixed regex/length features per already-delivered message: log A count, log B count, first and last uppercase standalone A/B, '
            'log explicit option-A/B counts, last explicit option, log word count. Aggregate exactly as the prior PCA8 argument features, yielding64 features. '
            'Lexical replaces those64 dimensions; combined appends them. No target/source probabilities or correct answers; no text-fitting or validation examples inspected.',
        'selection_policy': 'Exactly the cycle1 six-recipe tree search, same256/64 internal questions and observed-label NLL; refit320, freeze both before scoring80. '
            'Compare all models against frozen richer-state/PCA8 controls. Both new models use hard labels.',
        'assessment_loaded': False, 'source_calls': 0}
    json_write(ROOT/'protocol.json', protocol)
    files = [p for p in (ROOT/'features').rglob('*') if p.is_file()]+[ROOT/'sender_lexical.npy', ROOT/'labels.npz']
    json_write(ROOT/'prepared.json', {'protocol_sha256': digest(ROOT/'protocol.json'), 'artifacts': {str(p.relative_to(ROOT)): digest(p) for p in files}})
    print(json.dumps({'prepared': True, 'dimensions': {'lexical': 255, 'combined': 319}, 'training_counts': training_counts}), flush=True)
    return protocol

def validate(cfg):
    p = READ(ROOT/'protocol.json'); assert p['config'] == cfg
    for group in ('code_files', 'parent_artifacts', 'source_inputs'):
        for path, sha in p[group].items(): assert digest(path) == sha, path
    prepared = READ(ROOT/'prepared.json'); assert prepared['protocol_sha256'] == digest(ROOT/'protocol.json')
    for path, sha in prepared['artifacts'].items(): assert digest(ROOT/path) == sha, path
    return p

def evaluate(cfg):
    p = validate(cfg); frozen = READ(ROOT/'models_frozen.json')
    for path, sha in frozen['artifacts'].items(): assert digest(path) == sha, path
    if (ROOT/'results.json').exists(): return
    labels = np.load(ROOT/'labels.npz'); y, qids = labels['target'], labels['qids']
    rows = READ(Path(cfg['parent_state'])/'by_question.json'); summary, files = {}, []
    for name in cfg['variants']:
        with (ROOT/'refit'/name/'model.pkl').open('rb') as stream: model = pickle.load(stream)
        x = np.load(ROOT/'features/full'/f'{name}.npy', mmap_mode='r'); summary[name] = {}
        for split in ('train', 'validation'):
            take = np.isin(qids, p['fold'][split+'_questions'])
            with threadpool_limits(limits=cfg['training']['threads']): probabilities = model.predict_proba(x[take])[:, 1]
            metric, byq = state.score(probabilities, y[take], qids[take]); summary[name][split] = metric
            rows += [dict(model=name, split=split, **r) for r in byq]
            path = ROOT/'predictions'/f'{name}-{split}.npz'; path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, probability=probabilities, target=y[take], qids=qids[take])
            files.append({'model': name, 'split': split, 'path': str(path), 'sha256': digest(path)})
    questions = sorted(p['fold']['validation_questions']); grouped = {(r['model'], r['qid']): r for r in rows if r['split'] == 'validation'}
    rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed']); draws = rng.integers(len(questions), size=(cfg['evaluation']['bootstrap_replicates'], len(questions)))
    contrasts = []
    pairs = [('lexical', 'arguments'), ('combined', 'arguments'), ('lexical', 'rich'), ('combined', 'lexical')]
    pairs += [(name, 'opinion_nano') for name in cfg['variants']]
    for left, right in pairs:
        for metric in ('error', 'nll', 'brier_two_class'):
            delta = np.array([grouped[left, q][metric]-grouped[right, q][metric] for q in questions])
            contrasts.append({'left': left, 'right': right, 'metric': metric, 'difference': float(delta.mean()),
                'ci95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist()})
    json_write(ROOT/'by_question.json', rows)
    json_write(ROOT/'results.json', {'completed_utc': state.now(), 'summary': summary, 'paired_differences': contrasts,
        'prediction_files': files, 'models_frozen_sha256': digest(ROOT/'models_frozen.json'), 'assessment_loaded': False})
    print(json.dumps(summary), flush=True)

def phase(cfg, phase):
    processes = []
    for name in cfg['variants']:
        stream = (ROOT/f'{phase}_{name}.log').open('a', buffering=1)
        process = subprocess.Popen([sys.executable, __file__, 'worker', '--variant', name, '--phase', phase], stdout=stream, stderr=subprocess.STDOUT)
        processes.append((process, stream))
    for process, stream in processes:
        code = process.wait(); stream.close(); assert code == 0

def main(cfg):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'controller.lock').open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); prepare(cfg)
        if not (ROOT/'models_frozen.json').exists():
            json_write(ROOT/'environment.json', environment()); json_write(ROOT/'progress.json', {'phase': 'tune', 'updated_utc': state.now()}); phase(cfg, 'tune')
            choices = []
            for variant in cfg['variants']:
                candidates = [READ(p) for p in (ROOT/'tune'/variant).glob('*.json')]; assert len(candidates) == 6
                choices.append(min(candidates, key=lambda r: (r['inner_nll'], r['inner_error'], r['selected_iterations'], r['candidate']['id'])))
            json_write(ROOT/'selection.json', {'protocol_sha256': digest(ROOT/'protocol.json'), 'choices': choices})
            json_write(ROOT/'progress.json', {'phase': 'refit', 'updated_utc': state.now()}); phase(cfg, 'refit')
            files = [ROOT/p for p in ('protocol.json', 'prepared.json', 'selection.json')]
            files += [p for folder in ('tune', 'refit') for p in (ROOT/folder).rglob('*') if p.is_file()]
            json_write(ROOT/'models_frozen.json', {'frozen_utc': state.now(), 'artifacts': {str(p): digest(p) for p in files},
                'new_tuning_fits': 12, 'new_refits': 2, 'assessment_loaded': False})
        evaluate(cfg); json_write(ROOT/'progress.json', {'phase': 'complete', 'updated_utc': state.now()})

if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'run', 'worker']); parser.add_argument('--variant'); parser.add_argument('--phase', choices=['tune', 'refit'])
    args = parser.parse_args(); cfg = config()
    if args.action == 'prepare': prepare(cfg)
    elif args.action == 'run': main(cfg)
    else:
        original = state.validate; state.validate = validate
        try: state.worker(cfg, args.variant, args.phase)
        finally: state.validate = original
