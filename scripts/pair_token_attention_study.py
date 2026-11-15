"""Receiver-specific relationship tokens with ordinary scaled-dot-product attention.

New experiment namespace; prior inputs, models and trainer source are immutable.
"""
from argparse import ArgumentParser
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config as load_config, code_digest, digest, json_write, seed_all, environment
from scripts import compact_argument_study as prior
from scripts import breadth_argument_study as shared
from scripts import transformer_generalization as base

CONFIG = Path('configs/pair_token_attention_320.yaml')

def read(path): return json.loads(Path(path).read_text())
def config(): return load_config(CONFIG)

def neighbor_slots(graph):
    """Exact signed incoming edges; padding has zero sign and is masked."""
    assert graph.ndim == 3 and graph.shape[1] == graph.shape[2]
    assert np.isin(graph, [-1, 0, 1]).all()
    width = max(1, int((graph != 0).sum(-1).max()))
    neighbors = np.zeros((*graph.shape[:2], width), dtype=np.int64)
    signs = np.zeros_like(neighbors, dtype=np.float32)
    for e in range(len(graph)):
        for receiver in range(graph.shape[1]):
            senders = np.flatnonzero(graph[e, receiver])
            neighbors[e, receiver, :len(senders)] = senders
            signs[e, receiver, :len(senders)] = graph[e, receiver, senders]
    return neighbors, signs

class PairTokenAttention(prior.CompactArgumentGraph):
    def __init__(self, **kwargs):
        # Preserve initial common weights of the same-width signed model.
        super().__init__(**kwargs)
        del self.edge_bias, self.gamma, self.strength
        width = self.encoder[0].out_features
        self.pair_encoder = nn.Sequential(nn.Linear(width + 1, width), nn.GELU(), nn.LayerNorm(width))

    def forward(self, s, context, J, text, return_social=False, *, neighbors=None, signs=None):
        b, n = s.shape
        state = torch.stack([s == -1, s == 1], -1).to(context.dtype)
        structure = torch.stack([(J > 0).sum(-1), (J < 0).sum(-1), J.abs().sum(-1)], -1).to(context.dtype).log1p()
        h = self.encoder(torch.cat([state, context, structure], -1))
        hn = self.norm(h)
        sender = hn + self.text_projection(text * math.sqrt(self.text_dim))
        if neighbors is None:
            # Dense equivalent is useful for independently checking packed edges.
            assert signs is None
            neighbors = torch.arange(n, device=s.device)[None, None].expand(b, n, n)
            signs = J
        else:
            assert signs is not None and neighbors.shape == signs.shape
        slots = neighbors.shape[-1]
        sender_by_edge = sender[torch.arange(b, device=s.device)[:, None, None], neighbors]
        tokens = self.pair_encoder(torch.cat([sender_by_edge, signs[..., None]], -1))
        q = self.q(hn).view(b, n, self.heads, self.d_k)
        k = self.k(tokens).view(b, n, slots, self.heads, self.d_k).transpose(2, 3)
        v = self.v(tokens).view(b, n, slots, self.heads, self.d_v).transpose(2, 3)
        mask = signs.ne(0)[:, :, None, :]
        isolated = ~mask.any(-1, keepdim=True)
        scores = (q.unsqueeze(-2) * k).sum(-1) / math.sqrt(self.d_k)
        scores = torch.where(isolated, torch.zeros_like(scores), scores.masked_fill(~mask, -torch.inf))
        alpha = scores.softmax(-1) * mask
        aggregate = (alpha[..., None] * v).sum(-2)
        social = self.out(aggregate.reshape(b, n, -1))
        h = h + self.dropout(social)
        h = h + self.dropout(self.ffn(h))
        logits = self.classifier(h)
        return (logits, social, alpha.transpose(1, 2)) if return_social else logits

def make_model(cfg, width, seed=None):
    if seed is not None: seed_all(seed)
    model = PairTokenAttention(dropout=cfg['training']['dropout'], **cfg['models'][width])
    nn.init.zeros_(model.classifier[-1].weight)
    nn.init.zeros_(model.classifier[-1].bias)
    return model

class Data(prior.Data):
    def __init__(self, cfg, device, width=None):
        super().__init__(cfg, device, width)
        self.neighbors, self.signs = neighbor_slots(self.common['J'])

    def batch(self, ids, condition=None):
        result = super().batch(ids, condition)
        result['neighbors'] = torch.as_tensor(self.neighbors[self.e[ids]], device=self.device)
        result['signs'] = torch.as_tensor(self.signs[self.e[ids]], device=self.device)
        return result

def score_model(model, batch, opinion=False):
    assert not opinion
    z = model(batch['s'], batch['context'], batch['J'], batch['text'],
              neighbors=batch['neighbors'], signs=batch['signs'])
    return z[..., 1] - z[..., 0]

@contextmanager
def adapted_trainer():
    original_factory, original_score = shared.make_model, shared.score_model
    shared.make_model, shared.score_model = make_model, score_model
    try: yield
    finally: shared.make_model, shared.score_model = original_factory, original_score

def prepare(cfg):
    root, reference = Path(cfg['output']), Path(cfg['compact_reference'])
    if (root/'prepared.json').exists(): validate(cfg); return
    old = prior.validate(prior.config())
    old_frozen = prior.validate_frozen(prior.config())
    assert Path(prior.config()['output']) == reference
    for key, value in cfg['training'].items():
        if key not in ('aggregate_minutes', 'torch_threads'): assert value == old['config']['training'][key], key
    assert not old['assessment_loaded']
    parameters = {w: sum(p.numel() for p in make_model(cfg, w).parameters()) for w in cfg['models']}
    assert parameters == cfg['expected_parameters']
    folder = root/'data'; folder.mkdir(parents=True, exist_ok=True)
    for path in (reference/'data').iterdir():
        if path.is_file(): shutil.copyfile(path, folder/path.name)
    code_paths = [CONFIG, Path(__file__), Path(prior.__file__), Path(shared.__file__), Path(base.__file__),
                  Path('scripts/report_pair_token_attention.py'), Path('tests/test_pair_token_attention.py')]
    input_paths = [reference/name for name in ('protocol.json', 'prepared.json', 'encoding.json', 'models_frozen.json',
                   'completion_audit.json', 'encoding_replay.json', 'results/summary.json', 'results/predictions.json', 'results/by_question.json')]
    input_paths += [Path(cfg['message_reference'])/'data/vectors.npy']
    input_paths += [Path(r['path']) for r in read(reference/'results/predictions.json')]
    immutable = {**old['prior_artifacts'], **old_frozen['artifacts']}
    protocol = {'created_utc': shared.now(), 'config': cfg, 'config_sha256': digest(CONFIG), 'core_sha256': code_digest(),
        'code_files': {str(p): digest(p) for p in code_paths}, 'inputs': {str(p): digest(p) for p in input_paths},
        'source_artifacts': old['source_artifacts'], 'prior_artifacts': immutable, 'fold': old['fold'],
        'parameters': parameters, 'candidates': base.candidates(cfg),
        'architecture': 'u_j=LayerNorm(E(state_j))+W_text*scaled_text_j; z_ij=LayerNorm(GELU(Linear([u_j,tau_ij]))); '
          'q_i=W_Q*LayerNorm(E(state_i)); k_ij=W_K*z_ij; v_ij=W_V*z_ij. Ordinary masked softmax(q_i*k_ij/sqrt(dk)) '
          'and weighted sum of values. No learned edge bias, signed value multiplier, or degree-dependent social multiplier. '
          'One shared pair encoder for all receivers and senders; no learned IDs. Existing degree inputs retained in E.',
        'comparison_policy': 'Architecture bundle change, not an isolated test of one mechanism. Same information, '
          'training policy and three refit seeds. Width-four parameter counts match old signed models; width-eight has 64 more. '
          'Actual messages versus choice-only within each architecture; compare every new condition with its old counterpart. '
          'Old predictions are reused byte-for-byte; no old fit is overwritten.',
        'feature_policy': old['feature_policy'], 'forecast_boundary': old['forecast_boundary'],
        'control_policy': old['control_policy'], 'selection_policy': old['selection_policy'],
        'shuffle_policy': old['shuffle_policy'], 'shuffle_audit': old['shuffle_audit'],
        'pca_training_unique_texts': old['pca_training_unique_texts'],
        'pca_explained_variance_fraction': old['pca_explained_variance_fraction'],
        'evaluation_policy': 'Equal-question metrics, averaging three seed scores, never an ensemble or best seed. '
          '5000 paired validation-question bootstrap draws after averaging seeds. Reused validation is exploratory; '
          'pointwise intervals exclude seed-resampling and tuning uncertainty. No assessment or new source calls.',
        'assessment_loaded': False, 'source_llm_calls': 0, 'new_tuning_fits': 72, 'new_refits': 24}
    assert [len(protocol['fold'][k+'_questions']) for k in ('train', 'validation', 'assessment')] == [320, 80, 40]
    json_write(root/'protocol.json', protocol)
    json_write(root/'encoding.json', {'reference_encoding_sha256': digest(reference/'encoding.json'),
        'pca_sha256': digest(folder/'message_pca.npz'), 'encoder_parameters': 22713216})
    json_write(root/'prepared.json', {'protocol_sha256': digest(root/'protocol.json'),
        'artifacts': {str(p.relative_to(root)): digest(p) for p in folder.iterdir() if p.is_file()},
        'encoding_sha256': digest(root/'encoding.json')})
    print(json.dumps({'prepared': True, 'parameters': parameters}), flush=True)

def validate(cfg):
    root = Path(cfg['output']); protocol = read(root/'protocol.json')
    assert protocol['config'] == cfg and digest(CONFIG) == protocol['config_sha256']
    assert code_digest() == protocol['core_sha256'] == base.CORE_SHA
    for group in ('code_files', 'inputs'):
        for path, sha in protocol[group].items(): assert digest(path) == sha, path
    manifest = read(root/'prepared.json')
    assert manifest['protocol_sha256'] == digest(root/'protocol.json')
    assert manifest['encoding_sha256'] == digest(root/'encoding.json')
    for path, sha in manifest['artifacts'].items(): assert digest(root/path) == sha, path
    return protocol

def fit(cfg, device):
    protocol = validate(cfg); root = Path(cfg['output'])
    if (root/'models_frozen.json').exists(): validate_frozen(cfg); return
    torch.set_num_threads(cfg['training']['torch_threads'])
    data = Data(cfg, device); deadline = time.time() + cfg['training']['aggregate_minutes']*60
    if not (root/'training_environment.json').exists(): json_write(root/'training_environment.json', environment())
    records, refits = [], []
    with adapted_trainer():
        for width in cfg['models']:
            data.set_width(width)
            for condition in cfg['conditions']:
                for candidate in base.candidates(cfg):
                    item = dict(phase='tune', width=width, condition=condition, candidate=candidate['id'], seed=cfg['training']['tuning_seed'])
                    path, record = shared.train_one(cfg, protocol, data, item, deadline)
                    if record['status'] != 'complete': raise SystemExit(75)
                    records.append((path, record))
        selected = {'protocol_sha256': digest(root/'protocol.json'), 'choices': shared.choose(cfg, records),
                    'tuning_manifests': {str(p/'manifest.json'): digest(p/'manifest.json') for p, r in records}}
        if (root/'selection.json').exists(): assert read(root/'selection.json') == selected
        else: json_write(root/'selection.json', selected)
        for choice in selected['choices']:
            data.set_width(choice['width'])
            for seed in cfg['training']['refit_seeds']:
                item = {k: choice[k] for k in ('width', 'condition', 'candidate')}; item.update(phase='refit', seed=seed)
                path, record = shared.train_one(cfg, protocol, data, item, deadline, choice['refit_steps'])
                if record['status'] != 'complete': raise SystemExit(75)
                refits.append((path, record))
    artifacts = {str(root/n): digest(root/n) for n in ('protocol.json', 'prepared.json', 'encoding.json', 'selection.json')}
    for path, record in records + refits:
        artifacts[str(path/'manifest.json')] = digest(path/'manifest.json')
        artifacts.update({str(path/name): sha for name, sha in record['artifacts'].items()})
    json_write(root/'models_frozen.json', {'frozen_utc': shared.now(), 'artifacts': artifacts,
        'refits': [str(p) for p, r in refits], 'parameters': protocol['parameters'], 'tuning_fits': len(records),
        'refits_completed': len(refits), 'fit_seconds': sum(r['seconds'] for p, r in records+refits), 'assessment_loaded': False})
    print(json.dumps({'stage': 'models_frozen', 'tuning_fits': len(records), 'refits': len(refits)}), flush=True)

def validate_frozen(cfg):
    validate(cfg); record = read(Path(cfg['output'])/'models_frozen.json')
    for path, sha in record['artifacts'].items(): assert digest(path) == sha, path
    return record

if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'fit']); parser.add_argument('--device', default='cpu')
    args = parser.parse_args(); cfg = config()
    if args.action == 'prepare': prepare(cfg)
    else: fit(cfg, args.device)
