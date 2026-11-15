"""Causal recurrent summaries of already observed probabilities and social states."""
from argparse import ArgumentParser
from pathlib import Path
import copy
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write, code_digest
from scripts import modal_neural_study as core

ROOT = Path('runs/modal_accuracy_investigation_320_v1/causal_history_v1')
BaseData, base_validate = core.Data, core.validate
read, now = core.read, core.now


def config():
    cfg = copy.deepcopy(core.config()); cfg['output'] = str(ROOT); cfg['conditions'] = ['mode']
    cfg['models'] = {f'history_{text}_{observed}': dict(width=8, text='conclusion' if text == 'choice' else 'arguments', text_dim=8,
        initial=False, rich=False, history_hidden=16, observed=observed == 'observed')
        for text in ('choice', 'arguments') for observed in ('round', 'observed')}
    cfg['training'].update(max_steps=16384, patience=6)
    return cfg


def causal_states(spins, probability, graph, pairs):
    """No observation later than t is delivered with the prediction of t+1."""
    positive, negative = (graph > 0).astype(float), (graph < 0).astype(float)
    def average(weights, values):
        return np.einsum('eij,etj->eti', weights, values)/np.maximum(weights.sum(-1), 1)[:, None]
    s = spins[:, :8]; m = 2*probability[:, :8]-1
    time_value = np.broadcast_to(np.arange(8)[None, :, None]/7., s.shape)
    features = np.stack([s, m, average(positive, s), average(negative, s), average(positive, m), average(negative, m), time_value], -1)
    e, t = pairs.T; values = features[e].transpose(0, 2, 1, 3).copy()
    values *= (np.arange(8)[None, :] <= t[:, None])[:, None, :, None]
    return values.astype(np.float32)


class Data(BaseData):
    def __init__(self, stage):
        super().__init__(stage)
        probabilities = np.load(Path(core.dist.config()['probability_reference'])/'probabilities.npz')['single']
        values = causal_states(self.common['spins'], probabilities, self.common['J'], self.common['pairs'])
        self.values['observed_history'] = torch.as_tensor(values)
        self.values['current_round'] = torch.as_tensor(self.common['pairs'][:, 1].copy(), dtype=torch.long)


class HistoryAttention(core.ModeAttention):
    def __init__(self, spec, dropout):
        super().__init__(spec, dropout)
        h = spec['history_hidden']; old = self.encoder[0]
        rng = torch.random.get_rng_state(); new = nn.Linear(old.in_features+h, old.out_features); torch.random.set_rng_state(rng)
        with torch.no_grad():
            new.weight[:, :18] = old.weight[:, :18]; new.weight[:, 18:18+h] = 0
            new.weight[:, 18+h:] = old.weight[:, 18:]; new.bias.copy_(old.bias)
        self.encoder[0] = new; self.history = nn.GRU(7, h, batch_first=True)

    def forward(self, batch):
        values = batch['observed_history']
        if not self.spec['observed']:
            mask = torch.zeros(7, device=values.device); mask[-1] = 1
            values = values*mask
        batch_size, agents, length, dimensions = values.shape
        output, _ = self.history(values.reshape(batch_size*agents, length, dimensions))
        times = batch['current_round'][:, None].expand(batch_size, agents).reshape(-1)
        encoded = output[torch.arange(len(times), device=times.device), times].reshape(batch_size, agents, -1)
        changed = dict(batch, context=torch.cat([batch['context'], encoded], -1))
        return super().forward(changed)


def make_model(cfg, width, seed=None):
    if seed is not None: core.seed_all(seed)
    model = HistoryAttention(cfg['models'][width], cfg['training']['dropout'])
    nn.init.zeros_(model.classifier[-1].weight); nn.init.zeros_(model.classifier[-1].bias)
    return model


def prepare():
    cfg = config(); ROOT.mkdir(parents=True, exist_ok=True)
    parent = core.validate(core.config())
    files = [Path(__file__), Path('tests/test_modal_causal_history.py'), Path(core.__file__), Path(core.trainer.__file__)]
    p = dict(config=cfg, fold=parent['fold'], inner=parent['inner'], core_sha256=code_digest(),
        parameters={n: sum(v.numel() for v in make_model(cfg, n).parameters()) for n in cfg['models']},
        code_files={str(f): digest(f) for f in files}, references={str(Path(core.config()['output'])/'models_frozen.json'): digest(Path(core.config()['output'])/'models_frozen.json')},
        hypothesis='Past probabilities and past social states can reveal question-specific influence that a one-step state misses.',
        features='Per-agent GRU16 observes own opinion and probability, friendly/unfriendly mean opinions and probabilities, and round number up to current t. '
            'No future observation is delivered. Matched control keeps only round numbers, with identical parameter count and initialization. '
            'Both use actual ordered choice or argument PCA8 tokens, current probabilities, graph and original question/persona context.',
        policy='Internal256/64 mode-accuracy/NLL selection, two rates, three320 refits, then frozen before outer scoring. No assessment or source calls.',
        assessment_loaded=False, source_calls=0)
    if (ROOT/'protocol.json').exists(): assert read(ROOT/'protocol.json') == p
    else:
        json_write(ROOT/'protocol.json', p); json_write(ROOT/'encoding.json', dict(parent_prepared_sha256=digest(Path(core.config()['output'])/'prepared.json')))
        json_write(ROOT/'prepared.json', dict(protocol_sha256=digest(ROOT/'protocol.json'), encoding_sha256=digest(ROOT/'encoding.json')))
    return p


def validate(cfg): return base_validate(cfg)


def phase(label):
    cfg = config(); pending = list(cfg['models']); active, done = {}, {}
    while pending or active:
        while pending and len(active) < 4:
            name = pending.pop(0); stream = (ROOT/f'{label}_{name}.log').open('a', buffering=1)
            active[name] = subprocess.Popen([sys.executable, __file__, 'worker', '--model', name, '--phase', label], stdout=stream, stderr=subprocess.STDOUT), stream
        for name, (process, stream) in list(active.items()):
            if process.poll() is not None: done[name] = process.returncode; stream.close(); del active[name]
        json_write(ROOT/'progress.json', dict(phase=label, updated_utc=now(), active={k: p.pid for k, (p, s) in active.items()}, pending=pending, completed=done))
        if any(done.values()):
            for process, stream in active.values(): process.terminate(); process.wait(); stream.close()
            raise RuntimeError(str(done))
        if active: time.sleep(2)


def run():
    p = prepare(); validate(config())
    if (ROOT/'models_frozen.json').exists(): return
    phase('tune'); records = [(f.parent, read(f)) for f in (ROOT/'tune').glob('*/manifest.json')]
    assert len(records) == 8 and all(m['status'] == 'complete' for f, m in records)
    choices = []
    for name in config()['models']:
        folder, best = min([(f, m) for f, m in records if m['width'] == name], key=lambda v: (v[1]['validation']['error'], v[1]['validation']['nll'], v[1]['selected_step'], v[1]['candidate']))
        choices.append(dict(width=name, condition='mode', candidate=best['candidate'], refit_steps=best['selected_step'], tuning_directory=str(folder)))
    selection = dict(protocol_sha256=digest(ROOT/'protocol.json'), choices=choices)
    if (ROOT/'selection.json').exists(): assert read(ROOT/'selection.json') == selection
    else: json_write(ROOT/'selection.json', selection)
    phase('refit'); refits = [(f.parent, read(f)) for f in (ROOT/'refit').glob('*/manifest.json')]
    assert len(refits) == 12 and all(m['status'] == 'complete' for f, m in refits)
    files = [ROOT/n for n in ('protocol.json', 'prepared.json', 'encoding.json', 'selection.json')]
    files += [f for stage in ('tune', 'refit') for f in (ROOT/stage).rglob('*') if f.is_file()]
    json_write(ROOT/'models_frozen.json', dict(frozen_utc=now(), artifacts={str(f): digest(f) for f in files}, refits=[str(f) for f, m in refits]))
    json_write(ROOT/'progress.json', dict(phase='frozen_ready_to_evaluate', updated_utc=now()))


if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'run', 'worker']); parser.add_argument('--model'); parser.add_argument('--phase', choices=['tune', 'refit']); args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'run': run()
    else:
        core.Data, core.make_model, core.validate = Data, make_model, validate
        core.worker(config(), args.model, 'mode', args.phase)
