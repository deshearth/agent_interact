"""Direct mode classification, richer state, and matched argument attention."""
from argparse import ArgumentParser
from contextlib import contextmanager
import copy
import fcntl
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config as load_config, digest, code_digest, json_write, seed_all, environment
from mac_agents.models.baselines import author_module
from scripts import distribution_prediction_study as dist
from scripts import generalization_state_models as state
from scripts import generalization_lexical as lexical
from scripts import modal_state_screen as screen
from scripts import modal_neural_training as trainer

CONFIG = Path('configs/modal_accuracy_320.yaml')
read, now = state.read, state.now
def config(): return load_config(CONFIG)


class ModeAttention(dist.ordered.OrderedPairAttention):
    def __init__(self, spec, dropout):
        width = spec['width']
        super().__init__(d_model=width, heads=4, d_k=max(width//8, 1), d_v=width//4,
                         ffn_dim=width*2, text_dim=spec['text_dim'], dropout=dropout)
        self.spec = spec; extra = 1+int(spec['initial'])+96*int(spec['rich'])
        old = self.encoder[0]; rng = torch.random.get_rng_state()
        new = nn.Linear(old.in_features+extra, old.out_features); torch.random.set_rng_state(rng)
        with torch.no_grad():
            new.weight[:, :18] = old.weight[:, :18]; new.weight[:, 18:18+extra] = 0
            new.weight[:, 18+extra:] = old.weight[:, 18:]; new.bias.copy_(old.bias)
        self.encoder[0] = new

    def forward(self, batch):
        spec = self.spec; parts = [batch['context'], (2*batch['current_probability']-1)[..., None]]
        if spec['initial']: parts.append((2*batch['initial_probability']-1)[..., None])
        if spec['rich']: parts += [batch['question32'], batch['persona32'], batch['initial_population']]
        return super().forward(batch['s'], torch.cat(parts, -1), batch['J'], batch['text_'+spec['text']+str(spec['text_dim'])],
            neighbors=batch['neighbors'], signs=batch['signs'], positions=batch['positions'])


def make_model(cfg, width, seed=None):
    if seed is not None: seed_all(seed)
    model = ModeAttention(cfg['models'][width], cfg['training']['dropout'])
    nn.init.zeros_(model.classifier[-1].weight); nn.init.zeros_(model.classifier[-1].bias)
    return model


def score_model(model, batch, opinion=False):
    assert not opinion
    z = model(batch); return z[..., 1]-z[..., 0]


class Data(dist.Data):
    def __init__(self, stage):
        super().__init__(dist.config(), stage)
        self.source = self.y.copy(); self.y = (self.source > .5).astype(np.int64)
        common = self.common; e, t = common['pairs'].T
        probabilities = dict(np.load(Path(dist.config()['probability_reference'])/'probabilities.npz'))
        initial = probabilities['single'][e, 0].astype(np.float32); self.initial = initial
        self.values['target_soft'] = torch.as_tensor(self.source.astype(np.float32))
        self.values['target_mode'] = torch.as_tensor(self.y.astype(np.float32))
        self.values['initial_probability'] = torch.as_tensor(initial)
        self.values['initial_population'] = torch.as_tensor(np.broadcast_to(initial[:, None], (*initial.shape, 32)).copy())
        self.values['persona32'] = torch.as_tensor(np.broadcast_to(np.eye(32, dtype=np.float32)[None], (len(e), 32, 32)).copy())
        f = dict(np.load(Path(state.config()['output'])/'features'/stage/'transforms.npz'))
        q = np.load(state.config()['context_vectors'])['question_vectors']
        q32 = ((q-f['question32_mean'])@f['question32_axes'].T).astype(np.float32)
        self.values['question32'] = torch.as_tensor(np.broadcast_to(q32[e, None], (len(e), 32, 32)).copy())
        ids = np.load(state.config()['message_ids'])[e, t]
        vectors = np.load(screen.ROOT/'features'/stage/'vectors32.npy', allow_pickle=False)
        sender = np.load(lexical.ROOT/'sender_lexical.npy')[e, t]/4.
        choice = np.stack([lexical.text_features('B'), lexical.text_features('A')])/4.
        self.values['text_arguments40'] = torch.as_tensor(np.concatenate([vectors[ids], sender], -1))
        self.values['text_conclusion40'] = torch.as_tensor(np.concatenate([vectors[np.where(self.s == 1, 2, 1)], choice[(self.s == 1).astype(int)]], -1))
        for kind in ('arguments', 'conclusion'): self.values['text_'+kind+'8'] = self.values.pop('text_'+kind)
        self.condition = 'mode'

    def batch(self, ids, condition=None):
        target = self.condition if condition is None else condition; assert target in ('mode', 'soft')
        batch = {k: v[ids] for k, v in self.values.items()}
        batch['target'] = batch['target_'+target]
        return batch


@contextmanager
def adapted(progress_path=None):
    shared = dist.trainer
    original = trainer.make_model, trainer.score_model, trainer.json_write, shared.score_model
    def writer(path, value):
        if progress_path is not None and Path(path).name == 'progress.json': path = progress_path
        return original[2](path, value)
    trainer.make_model, trainer.score_model, trainer.json_write, shared.score_model = make_model, score_model, writer, score_model
    try: yield
    finally: trainer.make_model, trainer.score_model, trainer.json_write, shared.score_model = original


def prepare(cfg):
    root = Path(cfg['output'])
    if (root/'prepared.json').exists(): return validate(cfg)
    parent = dist.validate(dist.config()); screen.validate()
    assert read(screen.ROOT/'screen_results.json')['outer_validation_scored'] is False
    root.mkdir(parents=True, exist_ok=True)
    files = [CONFIG, Path(__file__), Path(trainer.__file__), Path('tests/test_modal_neural_study.py'), Path(dist.__file__), Path(screen.__file__)]
    references = [Path(cfg['distribution_reference'])/name for name in ('protocol.json', 'prepared.json', 'completion_audit.json', 'results.json', 'delivery_manifest.json')]
    references += [screen.ROOT/name for name in ('protocol.json', 'prepared.json', 'screen_results.json')]
    references += [screen.ROOT/'screen'/name/'complete.json' for name in screen.VARIANTS]
    params = {name: sum(p.numel() for p in make_model(cfg, name).parameters()) for name in cfg['models']}
    protocol = dict(created_utc=now(), config=cfg, fold=parent['fold'], inner=parent['inner'], parameters=params,
        core_sha256=code_digest(), code_files={str(p): digest(p) for p in files}, references={str(p): digest(p) for p in references},
        target='Most probable next single-response A/B opinion: A iff next p>0.5. Every update on existing histories is retained.',
        training='Matched soft-source-p and hard-mode training labels. Both select internal256/64 checkpoints by modal accuracy then modal-label NLL. '
            'Two learning rates; same dropout, decay, penalty, seeds, step cap and patience. Three fresh-seed320 refits per neural condition. '
            'All final models freeze before scoring outer80. No target probabilities or targets become inference inputs.',
        features='Nano current reproduces the978 architecture. Nano initial adds the own/sender initial pre-discussion probability, eight weights. '
            'Rich width32 adds training-only questionPCA32, known-persona one-hot32, and the initial32-persona population probabilities. '
            'Actual ordered sender text is PCA32 plus eight fixed lexical features scaled by1/4; matched controls use only A/B texts with identical dimensions. '
            'Frozen MiniLM remains22,713,216 pretrained parameters. Numerical histories never extend beyond the current round.',
        baselines='Four pinned Ising fits: current22 or initial25 features crossed with soft/mode labels, same author optimizer/regularization. '
            'Six tree screens refit their selected internal durations on320 with identical fixed settings. Trees are diagnostic larger models, not parameter-matched attention.',
        inference_scope='One-step mode prediction on existing stochastic histories, not a newly generated deterministic trajectory.',
        assessment_loaded=False, source_calls=0)
    json_write(root/'protocol.json', protocol)
    json_write(root/'encoding.json', dict(screen_prepared_sha256=digest(screen.ROOT/'prepared.json'), frozen_encoder_parameters=22713216))
    json_write(root/'prepared.json', dict(protocol_sha256=digest(root/'protocol.json'), encoding_sha256=digest(root/'encoding.json')))
    print({'prepared': True, 'parameters': params}, flush=True); return protocol


def validate(cfg):
    root = Path(cfg['output']); p = read(root/'protocol.json'); a = read(root/'prepared.json')
    assert p['config'] == cfg and p['core_sha256'] == code_digest()
    assert a['protocol_sha256'] == digest(root/'protocol.json') and a['encoding_sha256'] == digest(root/'encoding.json')
    for group in ('code_files', 'references'):
        for path, sha in p[group].items(): assert digest(path) == sha, path
    for path, sha in read(screen.ROOT/'prepared.json')['artifacts'].items(): assert digest(screen.ROOT/path) == sha, path
    return p


def ising_design(data, ids, initial):
    graph = data.common['J'][data.common['pairs'][ids, 0]]
    x = dist.ising_design(data.context[ids], graph, data.s[ids], data.current[ids], True)
    if initial:
        m = 2*data.initial[ids].astype(float)-1
        extra = np.stack([m, np.einsum('bij,bj->bi', np.maximum(graph, 0), m), np.einsum('bij,bj->bi', np.minimum(graph, 0), m)], -1)
        x = np.concatenate([x[..., :-1], extra, x[..., -1:]], -1)
    return x


def fit_baselines(cfg):
    root = Path(cfg['output']); p = validate(cfg); data = Data('full'); ids = data.ids(p['fold']['train_questions'])
    author = author_module({'data': {'root': 'data'}})
    for initial in (False, True):
        for condition in cfg['conditions']:
            name = 'ising_'+('initial' if initial else 'current')+'_'+condition
            folder = root/'baselines'/name; folder.mkdir(parents=True, exist_ok=True)
            if (folder/'complete.json').exists(): continue
            x = ising_design(data, ids, initial).reshape(-1, 25 if initial else 22)
            target = data.source[ids].ravel() if condition == 'soft' else data.y[ids].ravel()
            with threadpool_limits(limits=2):
                weights = author.two_stage_fit(x[:, :16], x[:, 16:-1], x[:, -1], np.ones(len(x), dtype=bool), target)
            json_write(folder/'model.json', dict(model=name, weights=weights.tolist(), initial=initial, condition=condition))
            json_write(folder/'complete.json', dict(completed_utc=now(), parameters=len(weights), model_sha256=digest(folder/'model.json')))
    labels = dict(np.load(screen.ROOT/'labels.npz')); fit = np.isin(labels['qids'], p['fold']['train_questions'])
    for name in screen.VARIANTS:
        folder = root/'baselines'/('tree_'+name); folder.mkdir(parents=True, exist_ok=True)
        if (folder/'complete.json').exists(): continue
        selected = read(screen.ROOT/'screen'/name/'complete.json')
        settings = dict(screen.SETTINGS); settings.update(max_iter=selected['selected_iterations'], early_stopping=False)
        model = HistGradientBoostingClassifier(**settings)
        x = np.load(screen.ROOT/'features/full'/f'{name}.npy', mmap_mode='r'); start = time.monotonic()
        with threadpool_limits(limits=4): model.fit(x[fit], labels['target'][fit])
        with (folder/'model.pkl').open('wb') as stream: pickle.dump(model, stream)
        json_write(folder/'complete.json', dict(model='tree_'+name, completed_utc=now(), seconds=time.monotonic()-start,
            model_sha256=digest(folder/'model.pkl'), selected_iterations=settings['max_iter'],
            tree_nodes=sum(len(t[0].nodes) for t in model._predictors)))


def worker(cfg, name, condition, phase):
    p = validate(cfg); root = Path(cfg['output']); torch.set_num_threads(2); data = Data('inner' if phase == 'tune' else 'full')
    runtime = copy.deepcopy(p)
    runtime['fold'] = dict(train_questions=p['inner']['fit'], validation_questions=p['inner']['validation']) if phase == 'tune' else dict(train_questions=p['fold']['train_questions'], validation_questions=[])
    deadline = time.time()+cfg['training']['aggregate_minutes']*60
    with adapted(root/'worker_progress'/f'{name}_{condition}.json'):
        if phase == 'tune':
            for candidate in dist.base.candidates(cfg):
                item = dict(phase=phase, width=name, condition=condition, candidate=candidate['id'], seed=42)
                _, result = trainer.train_one(cfg, runtime, data, item, deadline); assert result['status'] == 'complete'
        else:
            selected = next(c for c in read(root/'selection.json')['choices'] if c['width'] == name and c['condition'] == condition)
            for seed in cfg['training']['refit_seeds']:
                item = dict(phase=phase, width=name, condition=condition, candidate=selected['candidate'], seed=seed)
                _, result = trainer.train_one(cfg, runtime, data, item, deadline, selected['refit_steps']); assert result['status'] == 'complete'


def phase(cfg, label):
    root = Path(cfg['output']); pending = [(name, c) for name in cfg['models'] for c in cfg['conditions']]; active, completed = {}, {}
    while pending or active:
        while pending and len(active) < cfg['training']['max_concurrent_models']:
            name, condition = pending.pop(0); key = name+'_'+condition; stream = (root/f'{label}_{key}.log').open('a', buffering=1)
            active[key] = subprocess.Popen([sys.executable, __file__, 'worker', '--model', name, '--condition', condition, '--phase', label], stdout=stream, stderr=subprocess.STDOUT), stream
        for key, (process, stream) in list(active.items()):
            if process.poll() is not None: stream.close(); completed[key] = process.returncode; del active[key]
        json_write(root/'progress.json', dict(phase=label, updated_utc=now(), active={k: p.pid for k, (p, s) in active.items()}, pending=pending, completed=completed))
        if any(completed.values()):
            for process, stream in active.values():
                process.terminate(); process.wait(); stream.close()
            raise RuntimeError(str(completed))
        if active: time.sleep(2)


def run(cfg):
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    with (root/'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); prepare(cfg)
        if not (root/'models_frozen.json').exists():
            json_write(root/'environment.json', environment()); fit_baselines(cfg); phase(cfg, 'tune')
            records = [(p.parent, read(p)) for p in (root/'tune').glob('*/manifest.json')]; assert len(records) == 16
            choices = []
            for name in cfg['models']:
                for condition in cfg['conditions']:
                    group = [(path, r) for path, r in records if r['width'] == name and r['condition'] == condition]
                    assert len(group) == 2 and all(r['status'] == 'complete' for path, r in group)
                    path, best = min(group, key=lambda v: (v[1]['validation']['error'], v[1]['validation']['nll'], v[1]['selected_step'], v[1]['candidate']))
                    choices.append(dict(width=name, condition=condition, candidate=best['candidate'], refit_steps=best['selected_step'], tuning_directory=str(path)))
            selection = dict(protocol_sha256=digest(root/'protocol.json'), choices=choices)
            if (root/'selection.json').exists(): assert read(root/'selection.json') == selection
            else: json_write(root/'selection.json', selection)
            phase(cfg, 'refit'); refits = [(p.parent, read(p)) for p in (root/'refit').glob('*/manifest.json')]
            assert len(refits) == 24 and all(r['status'] == 'complete' for path, r in refits)
            files = [root/name for name in ('protocol.json', 'prepared.json', 'encoding.json', 'selection.json')]
            files += [p for stage in ('tune', 'refit', 'baselines') for p in (root/stage).rglob('*') if p.is_file()]
            json_write(root/'models_frozen.json', dict(frozen_utc=now(), artifacts={str(p): digest(p) for p in files}, refits=[str(p) for p, r in refits], assessment_loaded=False))
        json_write(root/'progress.json', dict(phase='frozen_ready_to_evaluate', updated_utc=now()))


if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'run', 'worker']); parser.add_argument('--model'); parser.add_argument('--condition', choices=['soft', 'mode']); parser.add_argument('--phase', choices=['tune', 'refit'])
    args = parser.parse_args(); cfg = config()
    if args.action == 'prepare': prepare(cfg)
    elif args.action == 'run': run(cfg)
    else: worker(cfg, args.model, args.condition, args.phase)
