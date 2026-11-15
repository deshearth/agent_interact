"""Normalize token attention keys and isolate question-conditioned message words."""
from argparse import ArgumentParser
from pathlib import Path
import copy
import subprocess
import sys
import time
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write, code_digest
from scripts import modal_token_pooling as tokens

ROOT = Path('runs/modal_accuracy_investigation_320_v1/token_focus_v1')
core = tokens.core
read, now = tokens.read, tokens.now


def config():
    cfg = copy.deepcopy(tokens.config()); cfg['output'] = str(ROOT)
    cfg['models'] = {f'focus_{text}_{region}': dict(width=16, text='conclusion' if text == 'choice' else 'arguments',
        text_dim=32, initial=False, rich=False, pooling='attention', message_only=region == 'message')
        for text in ('choice', 'arguments') for region in ('all', 'message')}
    return cfg


def focused_pool(projected, mask, query):
    if query is None: return tokens.pool_tokens(projected, mask)
    n, length, dimensions = projected.shape; values = projected.reshape(n, length, 4, dimensions//4)
    keys = F.layer_norm(values, (dimensions//4,))
    scores = torch.einsum('nlhd,hd->nlh', keys, query)/(dimensions//4)**.5
    scores = scores.masked_fill(~mask[..., None], -torch.inf)
    scores = torch.where(mask.any(1)[:, None, None], scores, torch.zeros_like(scores))
    weights = scores.softmax(1)*mask[..., None]
    return F.normalize((weights[..., None]*values).sum(1).reshape(n, dimensions), dim=-1)


class FocusAttention(tokens.TokenAttention):
    def __init__(self, spec, dropout):
        super().__init__(spec, dropout)
        # A/B controls have one message token, so there is no learnable token selection.
        if spec['message_only'] and spec['text'] == 'conclusion': self.register_parameter('token_query', None)

    def forward(self, batch):
        mask = batch['token_mask']
        if self.spec['message_only']: mask = mask & (batch['token_features'][..., 32] > .5)
        vector = focused_pool(F.gelu(self.token_projection(batch['token_features'])), mask, self.token_query)
        changed = dict(batch)
        changed['text_'+self.spec['text']+'32'] = vector[batch['token_inverse']]
        return core.ModeAttention.forward(self, changed)


def make_model(cfg, width, seed=None):
    if seed is not None: core.seed_all(seed)
    model = FocusAttention(cfg['models'][width], cfg['training']['dropout'])
    nn.init.zeros_(model.classifier[-1].weight); nn.init.zeros_(model.classifier[-1].bias)
    return model


Data = tokens.Data


def prepare():
    ROOT.mkdir(exist_ok=True); cfg = config(); parent = tokens.validate(tokens.config())
    files = [Path(__file__), Path('tests/test_modal_token_focus.py'), Path(tokens.__file__), Path(core.__file__), Path(core.trainer.__file__)]
    refs = [tokens.ROOT/'prepared.json', tokens.ROOT/'protocol.json', ROOT.parent/'diagnostics_v1/token_attention_scale.json']
    p = dict(config=cfg, fold=parent['fold'], inner=parent['inner'], core_sha256=code_digest(),
        parameters={n: sum(v.numel() for v in make_model(cfg, n).parameters()) for n in cfg['models']},
        code_files={str(f): digest(f) for f in files}, references={str(f): digest(f) for f in refs},
        hypothesis='A training-only probe found nearly uniform token attention (normalized entropy0.999997). Small key variance can keep query logits weak. '
            'Normalize each eight-dimensional attention key without affine parameters; keep the original projected values. '
            'Cross this with pooling all tokens versus only actual Message tokens. Question/options still condition each token through the frozen encoder.',
        controls='Same prepared token vectors, numerical state, two learning rates, duration policy, seeds and batching as token_pooling_v1. '
            'All query weights start at zero. The message-only A/B control has one token, so remove its32 inactive query parameters. No answer key, label or future state enters input.',
        policy='Internal256/64 selection, three320 refits, all frozen before outer80 scoring. This follows a training-only optimization diagnostic; no outer outcome selected these changes.',
        source_calls=0, assessment_loaded=False)
    if (ROOT/'protocol.json').exists(): assert read(ROOT/'protocol.json') == p
    else:
        json_write(ROOT/'protocol.json', p); json_write(ROOT/'encoding.json', dict(parent_prepared_sha256=digest(tokens.ROOT/'prepared.json')))
        json_write(ROOT/'prepared.json', dict(protocol_sha256=digest(ROOT/'protocol.json'), encoding_sha256=digest(ROOT/'encoding.json')))
    return p


def validate(cfg):
    p = tokens.base_validate(cfg); tokens.validate(tokens.config()); return p


def worker(name, label):
    class SelectedData(tokens.Data):
        def __init__(self, stage):
            super().__init__(stage); self.text_kind = config()['models'][name]['text']
    core.Data, core.make_model, core.validate = SelectedData, make_model, validate
    core.worker(config(), name, 'mode', label)


def phase(label):
    pending = list(config()['models']); active, done = {}, {}
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
    prepare(); validate(config())
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
    else: worker(args.model, args.phase)
