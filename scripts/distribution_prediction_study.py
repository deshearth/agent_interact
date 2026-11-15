"""Predict source opinion distributions, with and without current distributions.

New scientific namespace. Original data, encoders, predictors and source jobs
are read-only. Current probabilities are features only in the declared richer
state; next probabilities are supervised targets and never predictor inputs.
"""
from argparse import ArgumentParser
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.special import xlogy
import torch
from torch import nn
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config as load_config, code_digest, digest, json_write, seed_all
from mac_agents.models.graph import GraphUpdate
from mac_agents.models.baselines import author_module, design
from scripts import generalization_ordered_transformer as ordered
from scripts import generalization_state_models as state
from scripts import breadth_argument_study as trainer
from scripts import transformer_generalization as base

CONFIG = Path('configs/distribution_prediction_320.yaml')
def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def config(): return load_config(CONFIG)

def paired_probabilities(pairs, single):
    e, t = pairs.T
    return single[e, t].copy(), single[e, t+1].copy()

def distribution_metrics(logp, target, qids):
    lp, p = np.asarray(logp, dtype=np.float64), np.asarray(target, dtype=np.float64)
    assert lp.shape == (*p.shape, 2) and len(qids) == len(p)
    assert np.isfinite(lp).all() and np.isfinite(p).all() and ((p >= 0)&(p <= 1)).all()
    np.testing.assert_allclose(np.exp(lp).sum(-1), 1, atol=2e-6, rtol=0)
    predicted = np.exp(lp[..., 1]); nll = -(1-p)*lp[..., 0]-p*lp[..., 1]
    entropy = -xlogy(p, p)-xlogy(1-p, 1-p)
    kl = nll-entropy; assert kl.min() > -2e-6
    values = {'nll': nll, 'kl': np.maximum(kl, 0), 'mae': np.abs(predicted-p), 'mse': (predicted-p)**2,
              'within_05': (np.abs(predicted-p) <= .05).astype(float),
              'mode_error': (lp.argmax(-1) != (p > .5)).astype(float)}
    rows = [dict(qid=str(q), transitions=int(p[qids == q].size),
        **{k: float(v[qids == q].mean()) for k, v in values.items()}) for q in sorted(set(qids))]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in values}
    mean.update(questions=len(rows), transitions=int(p.size))
    return mean, rows

def add_probability_channel(model):
    """Embed the old function exactly, with one initially-zero extra input.

    Preserving the RNG state also preserves subsequent dropout draws between
    matched initializations. The new input sits after state2/context16.
    """
    old = model.encoder[0]; rng = torch.random.get_rng_state()
    new = nn.Linear(old.in_features+1, old.out_features)
    torch.random.set_rng_state(rng)
    with torch.no_grad():
        new.weight[:, :18].copy_(old.weight[:, :18]); new.weight[:, 18].zero_()
        new.weight[:, 19:].copy_(old.weight[:, 18:]); new.bias.copy_(old.bias)
    model.encoder[0] = new

class DistributionGraph(GraphUpdate):
    def __init__(self, probability_input=False, **kwargs):
        super().__init__(kind='A2', context_dim=16, extra_dim=0, **kwargs)
        self.probability_input = probability_input
        if probability_input: add_probability_channel(self)

    def forward(self, s, context, J, current_probability):
        extra = (2*current_probability-1)[..., None] if self.probability_input else torch.empty((*s.shape, 0), device=s.device, dtype=context.dtype)
        return super().forward(s, context, J, extra)

class DistributionOrdered(ordered.OrderedPairAttention):
    def __init__(self, probability_input=False, text_input='arguments', **kwargs):
        super().__init__(text_dim=8, **kwargs)
        self.probability_input, self.text_input = probability_input, text_input
        if probability_input: add_probability_channel(self)

    def forward(self, s, context, J, text, current_probability, **kwargs):
        if self.probability_input: context = torch.cat([context, (2*current_probability-1)[..., None]], -1)
        return super().forward(s, context, J, text, **kwargs)

def make_model(cfg, width, seed=None):
    if seed is not None: seed_all(seed)
    spec = cfg['models'][width]; kwargs = dict(dropout=cfg['training']['dropout'], **cfg['architecture'])
    if spec['family'] == 'graph': model = DistributionGraph(probability_input=spec['probability_input'], **kwargs)
    else: model = DistributionOrdered(probability_input=spec['probability_input'], text_input=spec['text_input'], **kwargs)
    nn.init.zeros_(model.classifier[-1].weight); nn.init.zeros_(model.classifier[-1].bias)
    assert sum(p.numel() for p in model.parameters()) == spec['parameters']
    return model

class Data:
    def __init__(self, cfg, stage, device='cpu'):
        root = Path(cfg['output']); self.device = str(device)
        self.common = dict(np.load(state.config()['common'], allow_pickle=False)); e, t = self.common['pairs'].T
        self.qids = self.common['episode_qids'][e]
        probabilities = dict(np.load(root/'data/probabilities.npz', allow_pickle=False))
        self.current = probabilities['current']; self.y = probabilities['target']
        self.hard = (self.common['spins'][e, t+1] == 1).astype(int)
        fitted = dict(np.load(Path(cfg['state_reference'])/'features'/stage/'transforms.npz', allow_pickle=False))
        graph = self.common['J'][e].astype(np.float32); self.s = self.common['spins'][e, t].astype(np.float32)
        positions = dict(np.load(Path(cfg['ordered_reference'])/'data/positions.npz', allow_pickle=False))
        ids = np.load(state.config()['message_ids'], allow_pickle=False)[e, t]
        vectors = np.load(Path(cfg['ordered_reference'])/'data'/f'vectors_{stage}.npy', allow_pickle=False)
        self.context = fitted['context'][e]
        values = {'s': self.s, 'context': self.context, 'J': graph,
            'current_probability': self.current.astype(np.float32), 'target': self.y.astype(np.float32),
            'text_arguments': vectors[ids], 'text_conclusion': vectors[np.where(self.s == 1, 2, 1)],
            'neighbors': positions['neighbors'][e], 'signs': positions['signs'][e], 'positions': positions['actual']}
        self.values = {k: torch.as_tensor(v, device=device) for k, v in values.items()}
        self.condition = 'distribution'

    def ids(self, questions): return np.flatnonzero(np.isin(self.qids, questions))

    def batch(self, ids, condition=None):
        assert condition in (None, 'distribution')
        return {k: v[ids] for k, v in self.values.items()}

def score_model(model, batch, opinion=False):
    assert not opinion
    if isinstance(model, DistributionGraph):
        z = model(batch['s'], batch['context'], batch['J'], batch['current_probability'])
    else:
        z = model(batch['s'], batch['context'], batch['J'], batch['text_'+model.text_input], batch['current_probability'],
            neighbors=batch['neighbors'], signs=batch['signs'], positions=batch['positions'])
    return z[..., 1]-z[..., 0]

@contextmanager
def adapted_trainer(progress_path=None):
    original = trainer.make_model, trainer.score_model, trainer.json_write, base.metrics
    def writer(path, value):
        if progress_path is not None and Path(path).name == 'progress.json': path = progress_path
        return original[2](path, value)
    trainer.make_model, trainer.score_model, trainer.json_write, base.metrics = make_model, score_model, writer, distribution_metrics
    try: yield
    finally: trainer.make_model, trainer.score_model, trainer.json_write, base.metrics = original

def ising_design(context, graph, spins, current_probability, enriched):
    old = design(context, graph, spins).astype(np.float64)
    if not enriched: return old
    mean_spin = 2*current_probability-1
    pos = np.einsum('bij,bj->bi', np.maximum(graph, 0), mean_spin)
    neg = np.einsum('bij,bj->bi', np.minimum(graph, 0), mean_spin)
    return np.concatenate([old[..., :18], mean_spin[..., None], pos[..., None], neg[..., None], old[..., 18:]], -1)

def prepare(cfg):
    root = Path(cfg['output'])
    if (root/'prepared.json').exists(): return validate(cfg)
    parent = state.validate(state.config()); ordered.validate(ordered.config())
    assert read(Path(cfg['ordered_reference'])/'completion_audit.json')['passed']
    noise = Path(cfg['probability_reference']); audit = read(noise/'audit.json')
    assert audit['probabilities_sha256'] == digest(noise/'probabilities.npz') and audit['actual_prompt_order_verified']
    raw = dict(np.load(noise/'probabilities.npz', allow_pickle=False))
    common = dict(np.load(state.config()['common'], allow_pickle=False)); np.testing.assert_array_equal(raw['pairs'], common['pairs'])
    e, t = common['pairs'].T; current, target = paired_probabilities(common['pairs'], raw['single'])
    assert target.shape == current.shape == (3200, 32)
    assert np.isfinite(current).all() and np.isfinite(target).all() and ((target >= 0)&(target <= 1)).all()
    folder = root/'data'; folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'probabilities.npz', current=current, target=target,
        current_other_mass=raw['other_mass'][e, t], target_other_mass=raw['other_mass'][e, t+1], pairs=common['pairs'], qids=common['episode_qids'][e])
    code = [CONFIG, Path(__file__), Path('scripts/finish_distribution_prediction.py'), Path('tests/test_distribution_prediction.py'),
        Path(ordered.__file__), Path(state.__file__), Path(trainer.__file__), Path(base.__file__), Path('data/raw/upstream/res/utils.py')]
    references = [Path(cfg['state_reference'])/n for n in ('protocol.json', 'prepared.json', 'completion_audit.json')]
    references += [Path(cfg['ordered_reference'])/n for n in ('protocol.json', 'prepared.json', 'completion_audit.json')]
    references += [noise/'audit.json', noise/'probabilities.npz', Path(state.config()['common']), Path(state.config()['message_ids'])]
    quality = {}
    for split in ('train', 'validation'):
        take = np.isin(common['episode_qids'][e], parent['fold'][split+'_questions']); quality[split] = {}
        for label, time_index in [('current', t), ('target', t+1)]:
            mass = raw['other_mass'][e[take], time_index[take]]
            quality[split][label] = {'rows': int(mass.size), 'other_mass_mean': float(mass.mean()), 'other_mass_max': float(mass.max()),
                'other_mass_above_001': int((mass > .001).sum())}
    parameters = {name: sum(p.numel() for p in make_model(cfg, name).parameters()) for name in cfg['models']}
    protocol = {'created_utc': now(), 'config': cfg, 'fold': parent['fold'], 'inner': parent['inner'], 'parameters': parameters,
        'core_sha256': code_digest(), 'code_files': {str(p): digest(p) for p in code}, 'references': {str(p): digest(p) for p in references},
        'source_artifacts': read(Path(cfg['ordered_reference'])/'protocol.json')['source_artifacts'],
        'target': 'Single-response conditional A/B probability p=P(first literal A)/(P(first literal A)+P(first literal B)) at the recorded temperature0.7. '
            'Target is p at t+1, not a sampled vote, majority-of-five q, or frequency from five votes. Keep all observations; no hard-label fallback or easy-case filtering. '
            'This is exactly the normalized recorded first-token distribution, an approximation to the complete parsed A/B opinion distribution; outside-token mass is reported.',
        'state': 'Two authorized input conditions: sampled current opinions s_t, or the same opinions plus current source probabilities p_t. '
            'Every model also receives the original signed graph and frozen question/persona context. Added neural feature is m_i=2p_i-1, visible to both sender and receiver encodings. '
            'Neither next-round probabilities nor next-round opinions enter inference inputs. Actual messages are available after delivery, before response sampling.',
        'models': 'Ising19 is the pinned author two-stage fitter on soft p targets; enriched Ising22 adds own m_i and friendly/unfriendly signed sums of m_j as coupling columns. '
            'Unordered graph transformers have818/826 parameters; ordered choice-only and argument controls have970/978. All ordered models use verified actual inbox positions. '
            'Argument effect is measured within equal-size ordered pairs. MiniLM22,713,216 parameters stays frozen, with PCA8 fitted only on inner256 or final320 training questions.',
        'selection': 'Nine learning-rate/output-penalty recipes per neural condition, seed42; same internal256/64 whole-question split. '
            'Train soft cross-entropy and select its internal mean, which is equivalent to KL minimization because target entropy is fixed. '
            'Refit the selected duration on320 questions with seeds142/143/144. Ising regularization/procedure is pinned and unselected. '
            'Freeze all eighteen neural refits and two Ising fits before scoring the external80; no averaging probabilities across seeds.',
        'metrics': 'Primary equal-question KL(source||prediction) in nats. Secondary absolute probability error, MSE, fraction within0.05, and modal-choice disagreement. '
            'Perfect distribution matching has KL/MAE/MSE zero; the old91.4% sampled-label ceiling does not apply to these metrics. '
            'Modal agreement is not sampled-opinion accuracy. Secondary majority-target probabilities are derived through M5 for continuity, not used for selection.',
        'uncertainty': '5000 paired question-bootstrap draws after averaging neural seed scores. The80 validation questions are reused; intervals are exploratory and pointwise.',
        'quality': quality, 'assessment_loaded': False, 'source_calls': 0}
    json_write(root/'protocol.json', protocol)
    json_write(root/'encoding.json', {'ordered_reference_prepared_sha256': digest(Path(cfg['ordered_reference'])/'prepared.json'),
        'frozen_encoder_parameters': 22713216, 'pca_dimensions': 8})
    json_write(root/'prepared.json', {'protocol_sha256': digest(root/'protocol.json'), 'encoding_sha256': digest(root/'encoding.json'),
        'artifacts': {str(p.relative_to(root)): digest(p) for p in folder.iterdir() if p.is_file()}})
    print(json.dumps({'prepared': True, 'parameters': parameters, 'ising_parameters': [19, 22], 'quality': quality}), flush=True)
    return protocol

def validate(cfg):
    root = Path(cfg['output']); p = read(root/'protocol.json'); prepared = read(root/'prepared.json')
    assert p['config'] == cfg and p['core_sha256'] == code_digest()
    for group in ('code_files', 'references', 'source_artifacts'):
        for path, sha in p[group].items(): assert digest(path) == sha, path
    for reference in ('state_reference', 'ordered_reference'):
        folder = Path(cfg[reference])
        for path, sha in read(folder/'prepared.json')['artifacts'].items(): assert digest(folder/path) == sha, path
    assert prepared['protocol_sha256'] == digest(root/'protocol.json') and prepared['encoding_sha256'] == digest(root/'encoding.json')
    for path, sha in prepared['artifacts'].items(): assert digest(root/path) == sha, path
    return p

def fit_baselines(cfg):
    p = validate(cfg); root = Path(cfg['output']); data = Data(cfg, 'full'); fit = data.ids(p['fold']['train_questions'])
    author = author_module({'data': {'root': 'data'}})
    for name, enriched in [('ising_opinion', False), ('ising_probability', True)]:
        folder = root/'baselines'/name; folder.mkdir(parents=True, exist_ok=True)
        if (folder/'complete.json').exists(): continue
        start = time.monotonic(); graph = data.common['J'][data.common['pairs'][fit, 0]]
        x = ising_design(data.context[fit], graph, data.s[fit], data.current[fit], enriched).reshape(-1, 22 if enriched else 19)
        with threadpool_limits(limits=cfg['training']['torch_threads']):
            weights = author.two_stage_fit(x[:, :16], x[:, 16:-1], x[:, -1], np.ones(len(x), dtype=bool), data.y[fit].ravel())
        logits = x@weights; logp = np.stack([-np.logaddexp(0, logits), -np.logaddexp(0, -logits)], -1).reshape(len(fit), 32, 2)
        metrics, rows = distribution_metrics(logp, data.y[fit], data.qids[fit])
        json_write(folder/'model.json', {'name': name, 'probability_input': enriched, 'weights': weights.tolist(), 'parameters': len(weights),
            'design': '[context16, signed current-opinion drives2, optional own/friendly/unfriendly mean-spin3, unsigned current-opinion drive1]'})
        np.save(folder/'train_logp.npy', logp)
        json_write(folder/'complete.json', {'completed_utc': now(), 'model': name, 'parameters': len(weights), 'train': metrics,
            'seconds': time.monotonic()-start, 'protocol_sha256': digest(root/'protocol.json'),
            'artifacts': {str(file): digest(file) for file in (folder/'model.json', folder/'train_logp.npy')}})
        print(json.dumps({'baseline_complete': name, 'train': metrics}), flush=True)

def worker(cfg, name, phase):
    p = validate(cfg); root = Path(cfg['output']); torch.set_num_threads(cfg['training']['torch_threads'])
    data = Data(cfg, 'inner' if phase == 'tune' else 'full'); protocol = copy.deepcopy(p)
    protocol['fold'] = {'train_questions': p['inner']['fit'], 'validation_questions': p['inner']['validation']} if phase == 'tune' else {'train_questions': p['fold']['train_questions'], 'validation_questions': []}
    deadline = time.time()+cfg['training']['aggregate_minutes']*60
    with adapted_trainer(root/'worker_progress'/f'{name}.json'):
        if phase == 'tune':
            for candidate in base.candidates(cfg):
                item = dict(phase=phase, width=name, condition='distribution', candidate=candidate['id'], seed=cfg['training']['tuning_seed'])
                _, record = trainer.train_one(cfg, protocol, data, item, deadline); assert record['status'] == 'complete'
        else:
            choice = next(c for c in read(root/'selection.json')['choices'] if c['width'] == name)
            for seed in cfg['training']['refit_seeds']:
                item = {k: choice[k] for k in ('width', 'condition', 'candidate')}; item.update(phase=phase, seed=seed)
                _, record = trainer.train_one(cfg, protocol, data, item, deadline, choice['refit_steps']); assert record['status'] == 'complete'

def majority(p): return np.clip(p**3*(10-15*p+6*p*p), 0, 1)

def evaluate(cfg):
    p = validate(cfg); root = Path(cfg['output']); frozen = read(root/'models_frozen.json')
    for path, sha in frozen['artifacts'].items(): assert digest(path) == sha, path
    if (root/'results.json').exists(): return
    torch.set_num_threads(cfg['training']['torch_threads']); data = Data(cfg, 'full'); records, files, secondary = [], [], []
    def save(name, seed, split, ids, lp, checkpoint):
        metrics, rows = distribution_metrics(lp, data.y[ids], data.qids[ids])
        records.extend(dict(model=name, seed=seed, split=split, **r) for r in rows)
        predicted = np.exp(lp[..., 1].astype(float)); qp, qt = majority(predicted), majority(data.y[ids])
        for qid in sorted(set(data.qids[ids])):
            mask = data.qids[ids] == qid
            secondary.append({'model': name, 'seed': seed, 'split': split, 'qid': str(qid),
                'majority_probability_mae': float(np.abs(qp[mask]-qt[mask]).mean()),
                'sampled_majority_accuracy': float(((predicted[mask] > .5) == data.hard[ids][mask]).mean())})
        path = root/'predictions'/f'{name}-s{seed}-{split}.npz'; path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, logp=lp, probability=predicted, target_probability=data.y[ids],
            current_probability=data.current[ids], hard_target=data.hard[ids], qids=data.qids[ids], pairs=data.common['pairs'][ids])
        files.append({'model': name, 'seed': seed, 'split': split, 'path': str(path), 'sha256': digest(path), 'checkpoint': str(checkpoint)})
    with adapted_trainer():
        for path in frozen['refits']:
            folder = Path(path); record = read(folder/'manifest.json'); name, seed = record['width'], record['seed']
            model = make_model(cfg, name); checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False); model.load_state_dict(checkpoint['model'])
            for split in ('train', 'validation'):
                ids = data.ids(p['fold'][split+'_questions']); lp = trainer.predictions(model, data, ids, cfg['training']['evaluation_batch_size'])
                save(name, seed, split, ids, lp, folder/'best.pt')
    for name in ('ising_opinion', 'ising_probability'):
        folder = root/'baselines'/name; fitted = read(folder/'model.json')
        for split in ('train', 'validation'):
            ids = data.ids(p['fold'][split+'_questions']); graph = data.common['J'][data.common['pairs'][ids, 0]]
            x = ising_design(data.context[ids], graph, data.s[ids], data.current[ids], fitted['probability_input'])
            score = x@np.asarray(fitted['weights']); lp = np.stack([-np.logaddexp(0, score), -np.logaddexp(0, -score)], -1)
            save(name, -1, split, ids, lp, folder/'model.json')
    metrics = ('nll', 'kl', 'mae', 'mse', 'within_05', 'mode_error'); models = ['ising_opinion', 'ising_probability']+list(cfg['models'])
    by_question, summary = [], {}
    for name in models:
        summary[name] = {}
        for split in ('train', 'validation'):
            for q in p['fold'][split+'_questions']:
                group = [r for r in records if r['model'] == name and r['split'] == split and r['qid'] == q]
                assert len(group) == (1 if name.startswith('ising') else 3)
                by_question.append(dict(model=name, split=split, qid=q, **{m: float(np.mean([r[m] for r in group])) for m in metrics}))
            summary[name][split] = {m: float(np.mean([r[m] for r in by_question if r['model'] == name and r['split'] == split])) for m in metrics}
    qs = sorted(p['fold']['validation_questions']); lookup = {(r['model'], r['qid']): r for r in by_question if r['split'] == 'validation'}
    rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed']); draws = rng.integers(len(qs), size=(cfg['evaluation']['bootstrap_replicates'], len(qs)))
    pairs = [('graph_opinion', 'ising_opinion'), ('graph_probability', 'ising_probability')]
    pairs += [(family+'_probability', family+'_opinion') for family in ('ising', 'graph', 'ordered_choice', 'ordered_arguments')]
    pairs += [('ordered_arguments_'+s, 'ordered_choice_'+s) for s in ('opinion', 'probability')]
    pairs += [('ordered_choice_'+s, 'graph_'+s) for s in ('opinion', 'probability')]
    contrasts = []
    for left, right in pairs:
        for metric in metrics:
            d = np.array([lookup[left, q][metric]-lookup[right, q][metric] for q in qs])
            contrasts.append({'left': left, 'right': right, 'metric': metric, 'difference': float(d.mean()), 'ci95': np.quantile(d[draws].mean(1), [.025, .975]).tolist()})
    json_write(root/'seed_by_question.json', records); json_write(root/'by_question.json', by_question); json_write(root/'majority_secondary.json', secondary)
    json_write(root/'results.json', {'completed_utc': now(), 'summary': summary, 'paired_differences': contrasts,
        'prediction_files': files, 'models_frozen_sha256': digest(root/'models_frozen.json'), 'assessment_loaded': False,
        'scope': 'Single-response distribution prediction. Internal question selection; outer80 reused exploratory validation. Neural metrics average three seeds, not an ensemble.'})
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'worker', 'baselines', 'evaluate']); parser.add_argument('--model'); parser.add_argument('--phase', choices=['tune', 'refit'])
    args = parser.parse_args(); cfg = config()
    if args.action == 'prepare': prepare(cfg)
    elif args.action == 'baselines': fit_baselines(cfg)
    elif args.action == 'evaluate': evaluate(cfg)
    else: worker(cfg, args.model, args.phase)
