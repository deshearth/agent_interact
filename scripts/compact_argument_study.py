"""Small argument-aware extensions; independent artifacts, shared frozen trainer.

The companion study owns its running fits. This adapter reads its frozen inputs
and imports its trainer without modifying its files or running process.
"""
from argparse import ArgumentParser
from collections import defaultdict
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
from scripts import breadth_argument_features as shared_features
from scripts import breadth_argument_study as shared
from scripts import transformer_generalization as base

CONFIG = Path('configs/compact_arguments_320.yaml')

def read(path): return json.loads(Path(path).read_text())
def config(): return load_config(CONFIG)

class CompactArgumentGraph(shared.ArgumentGraph):
    def __init__(self, text_dim=384, **kwargs):
        super().__init__(**kwargs)
        self.text_dim = text_dim
        if text_dim != 384:
            self.text_projection = nn.Linear(text_dim, self.encoder[0].out_features, bias=False)

    def forward(self, s, context, J, text, return_social=False):
        # The shared forward scales unit-norm full vectors by sqrt(384).
        # PCA vectors have training RMS 1/sqrt(text_dim), with padding zero.
        return super().forward(s, context, J, text * math.sqrt(self.text_dim / 384), return_social)

def make_model(cfg, width, seed=None):
    if seed is not None: seed_all(seed)
    model = CompactArgumentGraph(dropout=cfg['training']['dropout'], **cfg['models'][width])
    nn.init.zeros_(model.classifier[-1].weight)
    nn.init.zeros_(model.classifier[-1].bias)
    return model

def fit_message_pca(vectors, training_ids, dimensions=8):
    """No validation vectors or labels enter the fitted transform or scale."""
    training_ids = np.unique(training_ids)
    training_ids = training_ids[training_ids > 2]
    x = np.asarray(vectors[training_ids], dtype=np.float64)
    mean = x.mean(0)
    centered = x - mean
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(x))
    components = eigenvectors[:, -dimensions:][:, ::-1].T.copy()
    sign = np.sign(components[np.arange(dimensions), np.abs(components).argmax(1)])
    components *= sign[:, None]
    fitting_scores = centered @ components.T
    rms = float(np.sqrt(np.mean(fitting_scores ** 2)))
    assert rms > 0
    result = ((np.asarray(vectors, dtype=np.float64) - mean) @ components.T / (rms * math.sqrt(dimensions))).astype(np.float32)
    result[0] = 0
    return result, dict(mean=mean, components=components, scale=np.asarray(rms), training_ids=training_ids,
                        explained_variance_fraction=np.asarray(eigenvalues[-dimensions:].sum()/eigenvalues.sum()))

def matched_round_substitution(ids, states, seed):
    """Broadcast donor permutation within question, sender, current stance.

    This diagnostic deliberately does not match the receiver state or round.
    Donors are used only after fitting; target opinions are never inspected.
    """
    assert states.shape == ids.shape
    result = ids.copy()
    donor_round = np.broadcast_to(np.arange(ids.shape[1])[None, :, None], ids.shape).copy()
    rng = np.random.default_rng(seed)
    eligible = 0
    for e in range(len(ids)):
        for sender in range(ids.shape[2]):
            for stance in (-1, 1):
                positions = np.flatnonzero((states[e, :, sender] == stance) & (ids[e, :, sender] != 0))
                if len(positions) < 2: continue
                order = rng.permutation(positions)
                donors = np.roll(order, int(rng.integers(1, len(order))))
                result[e, order, sender] = ids[e, donors, sender]
                donor_round[e, order, sender] = donors
                eligible += len(order)
    nonzero = ids != 0
    return result, donor_round, {'sender_messages': int(nonzero.sum()), 'eligible_sender_messages': eligible,
        'changed_text_ids': int((result != ids).sum()), 'eligible_fraction': eligible / int(nonzero.sum()),
        'changed_fraction': float((result != ids)[nonzero].mean())}

def prepare(cfg):
    root = Path(cfg['output'])
    if (root/'prepared.json').exists(): validate(cfg); return
    reference = Path(cfg['message_reference'])
    original = shared_features.validate(shared_features.config(), True)
    assert Path(shared_features.config()['output']) == reference
    assert not original['assessment_loaded']
    fold = original['fold']
    assert [len(fold[k+'_questions']) for k in ('train', 'validation', 'assessment')] == [320, 80, 40]
    for key, value in cfg['training'].items():
        if key != 'aggregate_minutes': assert value == original['config']['training'][key], key
    parameters = {w: sum(p.numel() for p in make_model(cfg, w).parameters()) for w in cfg['models']}
    assert parameters == cfg['expected_parameters']
    data_root = root/'data'; data_root.mkdir(parents=True, exist_ok=True)
    for name in ('development.npz', 'features.npz', 'ising.json', 'message_ids.npy'):
        shutil.copyfile(reference/'data'/name, data_root/name)
    common = dict(np.load(data_root/'development.npz', allow_pickle=False))
    ids = np.load(data_root/'message_ids.npy', allow_pickle=False)
    vectors = np.load(reference/'data/vectors.npy', mmap_mode='r', allow_pickle=False)
    train = np.flatnonzero(np.isin(common['episode_qids'], fold['train_questions']))
    compressed, transform = fit_message_pca(vectors, ids[train], cfg['pca_dimensions'])
    np.save(data_root/'vectors_pca8.npy', compressed, allow_pickle=False)
    np.savez(data_root/'message_pca.npz', **transform)
    substituted, donor_round, shuffle_audit = matched_round_substitution(ids, common['spins'][:, :-1], cfg['shuffle_seed'])
    np.savez(data_root/'matched_substitution.npz', message_ids=substituted, donor_round=donor_round)
    input_paths = [reference/'protocol.json', reference/'prepared.json', reference/'encoding.json', reference/'data/vectors.npy']
    input_paths += [reference/p for p in read(reference/'prepared.json')['artifacts']]
    code_paths = [CONFIG, Path(__file__), Path(shared.__file__), Path(shared_features.__file__), Path(base.__file__),
                  Path('scripts/report_compact_arguments.py')]
    protocol = {'created_utc': shared.now(), 'config': cfg, 'config_sha256': digest(CONFIG), 'core_sha256': code_digest(),
        'code_files': {str(p): digest(p) for p in code_paths}, 'inputs': {str(p): digest(p) for p in input_paths},
        'source_artifacts': original['source_artifacts'], 'prior_artifacts': original['prior_artifacts'],
        'fold': fold, 'parameters': parameters, 'candidates': base.candidates(cfg),
        'feature_policy': 'Identical source message embeddings, context, graph and opinion inputs to companion study. Training-only PCA8 of unique training messages; global training RMS scaling; no encoder training. Message projection only enters sender keys/values.',
        'forecast_boundary': original['forecast_boundary'],
        'control_policy': original['control_policy'],
        'selection_policy': original['selection_policy'],
        'parameter_policy': 'Counts exclude frozen MiniLM (22,713,216), frozen PCA and context transforms. Width four is the smallest previously tested backbone, not a proven mathematical minimum.',
        'reuse_policy': 'Reuse companion width-eight full-vector fits after they freeze; never alter or duplicate those fits. Reuse original pico314, nano818 and Ising19 baselines.',
        'shuffle_policy': 'Post-fit sensitivity only. Bijective substitution from another round of the same question and sender with the same current stance, preserving broadcast, persona, signed graph and question. Receiver states, full histories and time are not matched. No targets used. Not a causal intervention.',
        'shuffle_audit': shuffle_audit, 'pca_training_unique_texts': len(transform['training_ids']),
        'pca_explained_variance_fraction': float(transform['explained_variance_fraction']),
        'evaluation_policy': 'Primary: paired validation-question NLL of arguments minus same-size conclusion control, then accuracy/Brier; 5000 question bootstraps after three-seed averaging. All sizes reported, no best-seed selection. Reused validation, exploratory pointwise intervals, no fresh assessment.',
        'assessment_loaded': False, 'source_llm_calls': 0, 'new_tuning_fits': 54, 'new_refits': 18}
    json_write(root/'protocol.json', protocol)
    # Shared trainer needs a frozen encoding identity, independent of its run.
    json_write(root/'encoding.json', {'source_encoding_sha256': digest(reference/'encoding.json'),
        'message_pca_sha256': digest(data_root/'message_pca.npz'), 'encoder_parameters': 22713216})
    json_write(root/'prepared.json', {'protocol_sha256': digest(root/'protocol.json'),
        'artifacts': {str(p.relative_to(root)): digest(p) for p in data_root.iterdir() if p.is_file()},
        'encoding_sha256': digest(root/'encoding.json')})
    print(json.dumps({'prepared': True, 'parameters': parameters, 'pca_variance_fraction': protocol['pca_explained_variance_fraction'],
        'training_unique_texts': protocol['pca_training_unique_texts'], 'shuffle': shuffle_audit}), flush=True)

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

class Data(shared.Data):
    def __init__(self, cfg, device, width=None):
        # The inherited constructor reads vectors from its own data directory.
        # Load common arrays manually, keeping the large frozen vector file shared.
        self.cfg, self.device = cfg, str(device)
        root = Path(cfg['output'])/'data'
        self.common = dict(np.load(root/'development.npz', allow_pickle=False))
        self.fitted = dict(np.load(root/'features.npz', allow_pickle=False))
        self.full_vectors = np.load(Path(cfg['message_reference'])/'data/vectors.npy', mmap_mode='r', allow_pickle=False)
        self.pca_vectors = np.load(root/'vectors_pca8.npy', allow_pickle=False)
        self.message_ids = np.load(root/'message_ids.npy', allow_pickle=False)
        self.substitutions = np.load(root/'matched_substitution.npz', allow_pickle=False)['message_ids']
        e,t = self.common['pairs'].T; self.e,self.t = e,t
        self.s = self.common['spins'][e,t].astype(np.float32)
        self.y = (self.common['spins'][e,t+1] == 1).astype(np.int64)
        self.qids = self.common['episode_qids'][e]
        self.context = self.fitted['context'][e]
        self.graph = self.common['J'][e].astype(np.float32)
        self.ids_text = self.message_ids[e,t]
        self.condition = 'arguments'
        self.set_width(width or next(iter(cfg['models'])))

    def set_width(self, width):
        self.width = width
        self.vectors = self.pca_vectors if self.cfg['models'][width]['text_dim'] == 8 else self.full_vectors

    def batch(self, ids, condition=None):
        condition = self.condition if condition is None else condition
        if condition != 'matched': return super().batch(ids, condition)
        text_ids = self.substitutions[self.e[ids], self.t[ids]]
        result = super().batch(ids, 'zero')
        result['text'] = torch.as_tensor(np.asarray(self.vectors[text_ids]), device=self.device)
        return result

@contextmanager
def adapted_trainer():
    # Module-local in this Python process only; shared on-disk files stay frozen.
    original = shared.make_model
    shared.make_model = make_model
    try: yield
    finally: shared.make_model = original

def fit(cfg, device):
    protocol = validate(cfg); root = Path(cfg['output'])
    if (root/'models_frozen.json').exists(): validate_frozen(cfg); return
    torch.set_num_threads(cfg['training']['torch_threads'])
    data = Data(cfg, device); deadline = time.time() + cfg['training']['aggregate_minutes']*60
    json_write(root/'training_environment.json', environment())
    records = []; refits = []
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
                    'tuning_manifests': {str(p/'manifest.json'): digest(p/'manifest.json') for p,r in records}}
        if (root/'selection.json').exists(): assert read(root/'selection.json') == selected
        else: json_write(root/'selection.json', selected)
        for choice in selected['choices']:
            data.set_width(choice['width'])
            for seed in cfg['training']['refit_seeds']:
                item = {k: choice[k] for k in ('width', 'condition', 'candidate')}; item.update(phase='refit', seed=seed)
                path, record = shared.train_one(cfg, protocol, data, item, deadline, choice['refit_steps'])
                if record['status'] != 'complete': raise SystemExit(75)
                refits.append((path, record))
    artifacts = {str(root/n): digest(root/n) for n in ('protocol.json','prepared.json','encoding.json','selection.json')}
    for path, record in records + refits:
        artifacts[str(path/'manifest.json')] = digest(path/'manifest.json')
        artifacts.update({str(path/name): sha for name,sha in record['artifacts'].items()})
    json_write(root/'models_frozen.json', {'frozen_utc': shared.now(), 'artifacts': artifacts,
        'refits': [str(p) for p,r in refits], 'parameters': protocol['parameters'], 'tuning_fits': len(records),
        'refits_completed': len(refits), 'fit_seconds': sum(r['seconds'] for p,r in records+refits), 'assessment_loaded': False})
    print(json.dumps({'stage':'models_frozen','tuning_fits':len(records),'refits':len(refits)}),flush=True)

def validate_frozen(cfg):
    validate(cfg); record = read(Path(cfg['output'])/'models_frozen.json')
    for path,sha in record['artifacts'].items(): assert digest(path) == sha, path
    return record

if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action',choices=['prepare','fit']); parser.add_argument('--device',default='cpu')
    args = parser.parse_args(); cfg = config()
    if args.action == 'prepare': prepare(cfg)
    else: fit(cfg,args.device)
