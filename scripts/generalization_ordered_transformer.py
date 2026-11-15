"""Small ordinary attention with auditable receiver-specific inbox positions."""
from argparse import ArgumentParser
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config as load_config, code_digest, digest, json_write, seed_all
from scripts import pair_token_attention_study as pair
from scripts import breadth_argument_study as trainer
from scripts import transformer_generalization as base
from scripts import generalization_state_models as state

CONFIG = Path('configs/generalization_ordered_transformer_320.yaml')
read, now = state.read, state.now
def config(): return load_config(CONFIG)

def positions_for_slots(common, order, neighbors):
    e = common['pairs'][:, 0]
    positions = np.zeros((*neighbors[e].shape, 3), dtype=np.float32)
    for b in range(len(e)):
        for receiver in range(common['J'].shape[1]):
            degree = int(np.count_nonzero(common['J'][e[b], receiver]))
            # Padding also stores index zero: restrict to the live prefix so
            # padded zeros cannot overwrite the actual sender-zero slot.
            slots = {int(sender): j for j, sender in enumerate(neighbors[e[b], receiver, :degree])}
            for sign in range(2):
                sequence = order[b, receiver, sign]; sequence = sequence[sequence >= 0]
                for rank, sender in enumerate(sequence):
                    positions[b, receiver, slots[int(sender)]] = [rank/max(len(sequence)-1, 1), rank == 0, rank == len(sequence)-1]
    return positions

class OrderedPairAttention(pair.PairTokenAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        width = self.encoder[0].out_features
        self.pair_encoder = nn.Sequential(nn.Linear(width+4, width), nn.GELU(), nn.LayerNorm(width))

    def forward(self, s, context, J, text, *, neighbors, signs, positions):
        b, n = s.shape
        state_vector = torch.stack([s == -1, s == 1], -1).to(context.dtype)
        structure = torch.stack([(J > 0).sum(-1), (J < 0).sum(-1), J.abs().sum(-1)], -1).to(context.dtype).log1p()
        h = self.encoder(torch.cat([state_vector, context, structure], -1)); hn = self.norm(h)
        sender = hn+self.text_projection(text*math.sqrt(self.text_dim))
        slots = neighbors.shape[-1]
        sender_by_edge = sender[torch.arange(b, device=s.device)[:, None, None], neighbors]
        tokens = self.pair_encoder(torch.cat([sender_by_edge, signs[..., None], positions], -1))
        q = self.q(hn).view(b, n, self.heads, self.d_k)
        k = self.k(tokens).view(b, n, slots, self.heads, self.d_k).transpose(2, 3)
        v = self.v(tokens).view(b, n, slots, self.heads, self.d_v).transpose(2, 3)
        mask = signs.ne(0)[:, :, None, :]; isolated = ~mask.any(-1, keepdim=True)
        scores = (q.unsqueeze(-2)*k).sum(-1)/math.sqrt(self.d_k)
        scores = torch.where(isolated, torch.zeros_like(scores), scores.masked_fill(~mask, -torch.inf))
        alpha = scores.softmax(-1)*mask
        social = self.out((alpha[..., None]*v).sum(-2).reshape(b, n, -1))
        h = h+self.dropout(social); h = h+self.dropout(self.ffn(h))
        return self.classifier(h)

def make_model(cfg, width, seed=None):
    if seed is not None: seed_all(seed)
    model = OrderedPairAttention(dropout=cfg['training']['dropout'], **cfg['models'][width])
    nn.init.zeros_(model.classifier[-1].weight); nn.init.zeros_(model.classifier[-1].bias)
    return model

class Data:
    def __init__(self, cfg, device, stage):
        self.device = str(device); root = Path(cfg['output']); parent = state.config()
        self.common = dict(np.load(parent['common'], allow_pickle=False)); e, t = self.common['pairs'].T
        self.y = (self.common['spins'][e, t+1] == 1).astype(np.int64)
        self.qids = self.common['episode_qids'][e]; self.condition = cfg['conditions'][0]
        fitted = np.load(Path(cfg['state_reference'])/'features'/stage/'transforms.npz', allow_pickle=False)
        positions = np.load(root/'data/positions.npz', allow_pickle=False)
        ids = np.load(parent['message_ids'], allow_pickle=False)[e, t]
        vectors = np.load(root/'data'/f'vectors_{stage}.npy', allow_pickle=False)
        self.s = self.common['spins'][e, t].astype(np.float32)
        values = {'s': self.s, 'context': fitted['context'][e], 'J': self.common['J'][e].astype(np.float32),
            'target': self.y.astype(np.float32), 'neighbors': positions['neighbors'][e], 'signs': positions['signs'][e]}
        self.values = {k: torch.as_tensor(v, device=device) for k, v in values.items()}
        self.text = {'arguments': torch.as_tensor(vectors[ids], device=device),
            'conclusion': torch.as_tensor(vectors[np.where(self.s == 1, 2, 1)], device=device)}
        self.positions = {name: torch.as_tensor(positions[name], device=device) for name in ('actual', 'random')}

    def ids(self, questions): return np.flatnonzero(np.isin(self.qids, questions))

    def batch(self, ids, condition=None):
        order, text = (condition or self.condition).split('_')
        result = {k: value[ids] for k, value in self.values.items()}
        result['text'] = self.text[text][ids]; result['positions'] = self.positions[order][ids]
        return result

def score_model(model, batch, opinion=False):
    assert not opinion
    logits = model(batch['s'], batch['context'], batch['J'], batch['text'], neighbors=batch['neighbors'], signs=batch['signs'], positions=batch['positions'])
    return logits[..., 1]-logits[..., 0]

@contextmanager
def adapted_trainer(progress_path=None):
    factory, score, writer = trainer.make_model, trainer.score_model, trainer.json_write
    def separate_progress(path, value):
        if progress_path is not None and Path(path).name == 'progress.json': path = progress_path
        return writer(path, value)
    trainer.make_model, trainer.score_model = make_model, score_model
    trainer.json_write = separate_progress
    try: yield
    finally: trainer.make_model, trainer.score_model, trainer.json_write = factory, score, writer

def prepare(cfg):
    root = Path(cfg['output']); reference = Path(cfg['state_reference'])
    if (root/'prepared.json').exists(): return validate(cfg)
    parent = state.validate(state.config()); assert read(reference/'completion_audit.json')['passed']
    noise = read('runs/generalization_investigation_320_v1/noise_audit_v1/audit.json')
    assert noise['actual_prompt_order_verified']
    assert sum(p.numel() for p in make_model(cfg, 'nano_pca8').parameters()) == cfg['expected_parameters']
    root.mkdir(parents=True, exist_ok=True); folder = root/'data'; folder.mkdir(exist_ok=True)
    common = dict(np.load(state.config()['common'], allow_pickle=False)); neighbors, signs = pair.neighbor_slots(common['J'])
    index = {i['qid']: i for i in read(state.config()['index'])}
    names = [Path(index[q]['path']).stem for q in common['episode_qids']]
    positional = {}
    for name, random in [('actual', False), ('random', True)]:
        order = state.ordered_neighbors(common['J'], names, common['pairs'], random)
        positional[name] = positions_for_slots(common, order, neighbors)
    np.savez_compressed(folder/'positions.npz', neighbors=neighbors, signs=signs, **positional)
    vectors = np.load(state.config()['message_vectors'], mmap_mode='r')
    for stage in ('inner', 'full'):
        fitted = np.load(reference/'features'/stage/'transforms.npz', allow_pickle=False)
        with threadpool_limits(limits=cfg['training']['torch_threads']):
            reduced = ((vectors.astype(float)-fitted['message_mean'])@fitted['message_axes'].T/(float(fitted['message_scale'])*math.sqrt(8))).astype(np.float32)
        reduced[0] = 0; np.save(folder/f'vectors_{stage}.npy', reduced)
    code = [CONFIG, Path(__file__), Path('scripts/finish_generalization_ordered_transformer.py'), Path('tests/test_generalization_ordered_transformer.py'),
            Path(pair.__file__), Path(trainer.__file__), Path(base.__file__), Path(state.__file__)]
    protocol = {'created_utc': now(), 'config': cfg, 'fold': parent['fold'], 'inner': parent['inner'],
        'code_files': {str(p): digest(p) for p in code}, 'core_sha256': code_digest(),
        'state_prepared_sha256': digest(reference/'prepared.json'), 'state_audit_sha256': digest(reference/'completion_audit.json'),
        'noise_audit_sha256': digest('runs/generalization_investigation_320_v1/noise_audit_v1/audit.json'),
        'source_artifacts': parent['inputs'], 'parameters': cfg['expected_parameters'],
        'hypothesis': 'Exact inbox order contains transferable information that a compact pair-token transformer can exploit.',
        'architecture': 'Shared pair encoder Linear(d_model+4,d_model), GELU, LayerNorm on [sender_state+W_text*message, tau, within-sign rank, first, last]. '
            'Ordinary q*k/sqrt(dk) masked attention; no edge biases or signed-value/degree multipliers. All four conditions970 parameters.',
        'features': 'Original16-dimensional context; PCA8 frozen MiniLM message or choice-only vectors. Every fitted transform uses only inner256 during '
            'tuning, then320 for refitting. Source-derived positions contain order only, never question IDs or RNG seeds. No target-round opinions or probabilities.',
        'selection': 'Three learning rates, fixed .001 output penalty, tuning seed42, max16384 updates, evaluate each512, patience6. '
            'Use same inner256/64 questions as cycle1. Best internal NLL chooses rate/duration. Three fresh refits142/143/144 on all320. '
            'All choices and twelve refits freeze before external80 evaluation. Report mean seed metrics, not an ensemble or best seed.',
        'assessment_loaded': False, 'source_calls': 0}
    json_write(root/'protocol.json', protocol)
    json_write(root/'encoding.json', {'frozen_encoder_parameters': 22713216, 'state_prepared_sha256': digest(reference/'prepared.json')})
    json_write(root/'prepared.json', {'protocol_sha256': digest(root/'protocol.json'), 'encoding_sha256': digest(root/'encoding.json'),
        'artifacts': {str(p.relative_to(root)): digest(p) for p in folder.iterdir() if p.is_file()}})
    print(json.dumps({'prepared': True, 'parameters': cfg['expected_parameters'], 'conditions': cfg['conditions']}), flush=True)
    return protocol

def validate(cfg):
    root = Path(cfg['output']); p = read(root/'protocol.json'); prepared = read(root/'prepared.json')
    assert p['config'] == cfg and p['core_sha256'] == code_digest()
    for path, sha in p['code_files'].items(): assert digest(path) == sha, path
    for path, sha in p['source_artifacts'].items(): assert digest(path) == sha, path
    assert p['state_prepared_sha256'] == digest(Path(cfg['state_reference'])/'prepared.json')
    assert p['state_audit_sha256'] == digest(Path(cfg['state_reference'])/'completion_audit.json')
    assert p['noise_audit_sha256'] == digest('runs/generalization_investigation_320_v1/noise_audit_v1/audit.json')
    parent_prepared = read(Path(cfg['state_reference'])/'prepared.json')
    for path, sha in parent_prepared['artifacts'].items(): assert digest(Path(cfg['state_reference'])/path) == sha, path
    assert prepared['protocol_sha256'] == digest(root/'protocol.json') and prepared['encoding_sha256'] == digest(root/'encoding.json')
    for path, sha in prepared['artifacts'].items(): assert digest(root/path) == sha, path
    return p

def worker(cfg, condition, phase):
    p = validate(cfg); root = Path(cfg['output']); torch.set_num_threads(cfg['training']['torch_threads'])
    data = Data(cfg, 'cpu', 'inner' if phase == 'tune' else 'full'); protocol = copy.deepcopy(p)
    protocol['fold'] = {'train_questions': p['inner']['fit'], 'validation_questions': p['inner']['validation']} if phase == 'tune' else {'train_questions': p['fold']['train_questions'], 'validation_questions': []}
    deadline = time.time()+cfg['training']['aggregate_minutes']*60
    with adapted_trainer(root/'worker_progress'/f'{condition}.json'):
        if phase == 'tune':
            for candidate in base.candidates(cfg):
                item = dict(phase=phase, width='nano_pca8', condition=condition, candidate=candidate['id'], seed=cfg['training']['tuning_seed'])
                path, record = trainer.train_one(cfg, protocol, data, item, deadline)
                assert record['status'] == 'complete'
        else:
            choice = next(c for c in read(root/'selection.json')['choices'] if c['condition'] == condition)
            for seed in cfg['training']['refit_seeds']:
                item = {k: choice[k] for k in ('width', 'condition', 'candidate')}; item.update(phase=phase, seed=seed)
                path, record = trainer.train_one(cfg, protocol, data, item, deadline, choice['refit_steps'])
                assert record['status'] == 'complete'

def evaluate(cfg):
    p = validate(cfg); root = Path(cfg['output']); frozen = read(root/'models_frozen.json')
    for path, sha in frozen['artifacts'].items(): assert digest(path) == sha, path
    if (root/'results.json').exists(): return
    torch.set_num_threads(cfg['training']['torch_threads']); data = Data(cfg, 'cpu', 'full'); rows, prediction_files = [], []
    with adapted_trainer():
        for path in frozen['refits']:
            path = Path(path); record = read(path/'manifest.json'); condition, seed = record['condition'], record['seed']
            model = make_model(cfg, 'nano_pca8'); ck = torch.load(path/'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(ck['model']); data.condition = condition
            for split in ('train', 'validation'):
                ids = data.ids(p['fold'][split+'_questions']); lp = trainer.predictions(model, data, ids, cfg['training']['evaluation_batch_size'])
                metric, byq = base.metrics(lp, data.y[ids], data.qids[ids])
                rows += [dict(model=condition, seed=seed, split=split, **r) for r in byq]
                out = root/'predictions'/f'{condition}-s{seed}-{split}.npz'; out.parent.mkdir(exist_ok=True)
                np.savez_compressed(out, logp=lp, target=data.y[ids], qids=data.qids[ids], pairs=data.common['pairs'][ids])
                prediction_files.append({'model': condition, 'seed': seed, 'split': split, 'path': str(out), 'sha256': digest(out), 'checkpoint': str(path/'best.pt')})
    # Aggregate seeds within each question, preserving a question as the bootstrap unit.
    averaged = []
    for condition in cfg['conditions']:
        for split in ('train', 'validation'):
            for q in p['fold'][split+'_questions']:
                group = [r for r in rows if r['model'] == condition and r['split'] == split and r['qid'] == q]
                assert len(group) == 3
                averaged.append(dict(model=condition, split=split, qid=q, **{m: float(np.mean([r[m] for r in group])) for m in ('error', 'nll', 'brier_two_class')}))
    averaged += [r for r in read(Path(cfg['state_reference'])/'by_question.json') if r['model'] in ('I0', 'opinion_nano')]
    summary = {name: {split: {m: float(np.mean([r[m] for r in averaged if r['model'] == name and r['split'] == split]))
        for m in ('error', 'nll', 'brier_two_class')} for split in ('train', 'validation')} for name in cfg['conditions']+['I0', 'opinion_nano']}
    grouped = {(r['model'], r['qid']): r for r in averaged if r['split'] == 'validation'}
    questions = sorted(p['fold']['validation_questions']); rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    draws = rng.integers(len(questions), size=(cfg['evaluation']['bootstrap_replicates'], len(questions)))
    pairs = [('actual_conclusion', 'random_conclusion'), ('actual_arguments', 'random_arguments'),
             ('actual_arguments', 'actual_conclusion'), ('random_arguments', 'random_conclusion')]
    pairs += [(c, 'opinion_nano') for c in cfg['conditions']]
    contrasts = []
    for left, right in pairs:
        for metric in ('error', 'nll', 'brier_two_class'):
            delta = np.array([grouped[left, q][metric]-grouped[right, q][metric] for q in questions])
            contrasts.append({'left': left, 'right': right, 'metric': metric, 'difference': float(delta.mean()),
                'ci95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist()})
    json_write(root/'seed_by_question.json', rows); json_write(root/'by_question.json', averaged)
    json_write(root/'results.json', {'completed_utc': now(), 'summary': summary, 'paired_differences': contrasts,
        'prediction_files': prediction_files, 'models_frozen_sha256': digest(root/'models_frozen.json'),
        'assessment_loaded': False, 'scope': 'Exploratory reused80; all fitting/selection within320; three-seed mean metrics.'})
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'worker', 'evaluate']); parser.add_argument('--condition'); parser.add_argument('--phase', choices=['tune', 'refit'])
    args = parser.parse_args(); cfg = config()
    if args.action == 'prepare': prepare(cfg)
    elif args.action == 'evaluate': evaluate(cfg)
    else: worker(cfg, args.condition, args.phase)
