"""Task-trained token pooling instead of a fixed sentence average."""
from argparse import ArgumentParser
from pathlib import Path
import copy
import os
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write, code_digest
from scripts import modal_neural_study as core
from scripts import modal_question_arguments as paired

ROOT = Path('runs/modal_accuracy_investigation_320_v1/token_pooling_v1')
BaseData, base_validate = core.Data, core.validate
read, now = core.read, core.now


def config():
    cfg = copy.deepcopy(core.config()); cfg['output'] = str(ROOT); cfg['conditions'] = ['mode']
    cfg['models'] = {f'tokens_{text}_{pool}': dict(width=16, text='conclusion' if text == 'choice' else 'arguments', text_dim=32,
        initial=False, rich=False, pooling=pool) for text in ('choice', 'arguments') for pool in ('mean', 'attention')}
    cfg['training'].update(max_steps=16384, patience=6, batch_size=16, evaluation_batch_size=32)
    return cfg


def pool_tokens(projected, mask, query=None):
    """Four independent pooling heads; padding contributes neither keys nor values."""
    n, length, dimensions = projected.shape; values = projected.reshape(n, length, 4, dimensions//4)
    if query is None:
        pooled = (values*mask[..., None, None]).sum(1)/mask.sum(1).clamp(min=1)[:, None, None]
    else:
        scores = torch.einsum('nlhd,hd->nlh', values, query)/(dimensions//4)**.5
        scores = scores.masked_fill(~mask[..., None], -torch.inf)
        scores = torch.where(mask.any(1)[:, None, None], scores, torch.zeros_like(scores))
        weights = scores.softmax(1)*mask[..., None]
        pooled = (weights[..., None]*values).sum(1)
    return F.normalize(pooled.reshape(n, dimensions), dim=-1)


class TokenAttention(core.ModeAttention):
    def __init__(self, spec, dropout):
        super().__init__(spec, dropout)
        self.token_projection = nn.Linear(37, 32)
        if spec['pooling'] == 'attention': self.token_query = nn.Parameter(torch.zeros(4, 8))
        else: self.register_parameter('token_query', None)

    def forward(self, batch):
        vector = pool_tokens(F.gelu(self.token_projection(batch['token_features'])), batch['token_mask'], self.token_query)
        changed = dict(batch)
        changed['text_'+self.spec['text']+'32'] = vector[batch['token_inverse']]
        return super().forward(changed)


def make_model(cfg, width, seed=None):
    if seed is not None: core.seed_all(seed)
    model = TokenAttention(cfg['models'][width], cfg['training']['dropout'])
    nn.init.zeros_(model.classifier[-1].weight); nn.init.zeros_(model.classifier[-1].bias)
    return model


def prepare():
    cfg = config(); ROOT.mkdir(exist_ok=True); parent = paired.validate(paired.config())
    files = [Path(__file__), Path('tests/test_modal_token_pooling.py'), Path(core.__file__), Path(core.trainer.__file__)]
    protocol = dict(config=cfg, fold=parent['fold'], inner=parent['inner'], core_sha256=code_digest(),
        parameters={n: sum(v.numel() for v in make_model(cfg, n).parameters()) for n in cfg['models']},
        code_files={str(f): digest(f) for f in files}, references={str(Path(paired.config()['output'])/'models_frozen.json'): digest(Path(paired.config()['output'])/'models_frozen.json')},
        encoder_snapshot=paired.config()['encoder_snapshot'], pca_text_sample=2048, token_pca_dimensions=32,
        hypothesis='Sentence averaging may erase important conclusions or negations; train pooling over contextual wordpieces instead.',
        encoding='Frozen MiniLM contextual token states from the same full Question/Options/Message strings. Complete510-token windows. '
            'PCA32 fitted separately to tokens from2048 sampled training-only actual-message texts in inner256 or full320. '
            'Projected token vectors are unit-normalized and stored float16. Five fixed token features: message-region indicator, literal uppercase A/B, digit, relative position. '
            'No labels, answer keys or future messages enter token encoding or PCA. All tokens are retained.',
        pooling='Learned37-to32 token projection and GELU, followed by uniform mean or four learned attention queries. Query weights start at zero, '
            'so matched models initially implement uniform pooling. Attention adds32 trained parameters. Choice/argument controls otherwise match.',
        selection='Internal256/64 mode accuracy then NLL, same two rates; batch16 and evaluation32 bound token memory; three fresh-seed320 refits. Freeze before outer scoring.',
        assessment_loaded=False, source_calls=0)
    if (ROOT/'protocol.json').exists(): assert read(ROOT/'protocol.json') == protocol
    else: json_write(ROOT/'protocol.json', protocol)
    return protocol


def token_vectors(texts, tokenizer, encoder, device):
    parsed = tokenizer(texts, add_special_tokens=False, truncation=False, return_offsets_mapping=True, verbose=False)
    tokens = parsed['input_ids']; offsets = parsed['offset_mapping']
    chunks = [(j, start, values[start:start+510]) for j, values in enumerate(tokens) for start in range(0, len(values), 510)]
    outputs = [np.empty((len(t), 384), dtype=np.float32) for t in tokens]
    with torch.inference_mode():
        for begin in range(0, len(chunks), 32):
            pack = chunks[begin:begin+32]; width = min(512, ((max(len(v) for j, s, v in pack)+129)//128)*128)
            ids = torch.full((32, width), tokenizer.pad_token_id, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids); ids[:, 0] = tokenizer.cls_token_id; ids[:, 1] = tokenizer.sep_token_id; mask[:, :2] = 1
            for row, (j, start, values) in enumerate(pack):
                ids[row, :len(values)+2] = torch.as_tensor([tokenizer.cls_token_id]+values+[tokenizer.sep_token_id], device=device)
                mask[row, :len(values)+2] = 1
            hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state.cpu().numpy()
            for row, (j, start, values) in enumerate(pack): outputs[j][start:start+len(values)] = hidden[row, 1:len(values)+1]
    return outputs, offsets


def extra_features(text, offsets):
    start = text.index('\nMessage: ')+len('\nMessage: ')
    return np.asarray([[int(lo >= start), int(text[lo:hi] == 'A'), int(text[lo:hi] == 'B'),
        int(any(c.isdigit() for c in text[lo:hi])), (j+1)/len(offsets)] for j, (lo, hi) in enumerate(offsets)], dtype=np.float32)


def encode():
    p = prepare()
    if (ROOT/'prepared.json').exists(): return validate(config())
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2); device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(p['encoder_snapshot'], local_files_only=True)
    encoder = AutoModel.from_pretrained(p['encoder_snapshot'], local_files_only=True).to(device).eval()
    for parameter in encoder.parameters(): parameter.requires_grad_(False)
    parent = Path(paired.config()['output']); texts = read(parent/'data/texts.json')
    lengths = np.load(parent/'data/token_counts.npy').astype(np.int64); offsets = np.r_[0, lengths.cumsum()]
    common = dict(np.load(core.state.config()['common'])); message_ids = np.load(parent/'data/message_ids.npy')
    transforms = {}; started = time.monotonic(); (ROOT/'data').mkdir(exist_ok=True)
    for stage, questions, seed in [('inner', p['inner']['fit'], 92801), ('full', p['fold']['train_questions'], 92802)]:
        folder = ROOT/'features'/stage; folder.mkdir(parents=True, exist_ok=True)
        if (folder/'pca.npz').exists(): transforms[stage] = dict(np.load(folder/'pca.npz')); continue
        ids = np.unique(message_ids[np.isin(common['episode_qids'], questions)]); ids = ids[ids > 0]
        selected = np.sort(np.random.default_rng(seed).choice(ids, size=min(len(ids), p['pca_text_sample']), replace=False))
        total = 0; sums = np.zeros(384); cross = np.zeros((384, 384))
        with threadpool_limits(limits=2):
            for begin in range(0, len(selected), 64):
                hidden, _ = token_vectors([texts[i] for i in selected[begin:begin+64]], tokenizer, encoder, device)
                x = np.concatenate(hidden).astype(np.float64); total += len(x); sums += x.sum(0); cross += x.T@x
            mean = sums/total; covariance = cross/total-np.outer(mean, mean)
            values, vectors = np.linalg.eigh(covariance); axes = vectors[:, -32:][:, ::-1].T.copy()
        transforms[stage] = dict(mean=mean.astype(np.float32), axes=axes.astype(np.float32), training_text_ids=selected,
            fitted_tokens=np.array(total), explained_variance=np.array(values[-32:].sum()/values.sum()))
        np.savez(folder/'pca.npz', **transforms[stage])
        print({'pca': stage, 'texts': len(selected), 'tokens': total, 'variance': float(transforms[stage]['explained_variance'])}, flush=True)
    progress_path = ROOT/'encoding_progress.json'
    progress = read(progress_path) if progress_path.exists() else dict(next_text=1, seconds=0.)
    mode = 'r+' if progress_path.exists() else 'w+'
    features = {stage: np.lib.format.open_memmap(ROOT/'features'/stage/'tokens.npy', mode=mode, dtype=np.float16, shape=(int(offsets[-1]), 32)) for stage in transforms}
    extras = np.lib.format.open_memmap(ROOT/'data/extras.npy', mode=mode, dtype=np.float16, shape=(int(offsets[-1]), 5))
    prior = progress['seconds']; started_encoding = time.monotonic()
    with threadpool_limits(limits=2):
        for begin in range(progress['next_text'], len(texts), 64):
            end = min(begin+64, len(texts)); hidden, positions = token_vectors(texts[begin:end], tokenizer, encoder, device)
            for text_id, x, pos in zip(range(begin, end), hidden, positions):
                assert len(x) == lengths[text_id]
                sl = slice(offsets[text_id], offsets[text_id+1]); extras[sl] = extra_features(texts[text_id], pos)
                for stage, transform in transforms.items():
                    v = (x-transform['mean'])@transform['axes'].T
                    v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
                    features[stage][sl] = v.astype(np.float16)
            if (end-1) % 1024 == 0 or end == len(texts):
                for a in [extras, *features.values()]: a.flush()
                progress = dict(next_text=end, seconds=prior+time.monotonic()-started_encoding, updated_utc=now(), total_texts=len(texts))
                json_write(progress_path, progress); print(progress, flush=True)
                if device == 'mps': torch.mps.empty_cache()
    np.save(ROOT/'data/offsets.npy', offsets)
    json_write(ROOT/'encoding.json', dict(completed_utc=now(), device=device, tokens=int(offsets[-1]), tokens_dropped=0,
        seconds=progress['seconds'], frozen_encoder_parameters=sum(v.numel() for v in encoder.parameters()),
        pca_variance={s: float(v['explained_variance']) for s, v in transforms.items()}))
    files = [f for folder in ('features', 'data') for f in (ROOT/folder).rglob('*') if f.is_file()]
    json_write(ROOT/'prepared.json', dict(protocol_sha256=digest(ROOT/'protocol.json'), encoding_sha256=digest(ROOT/'encoding.json'), artifacts={str(f.relative_to(ROOT)): digest(f) for f in files}))


def validate(cfg):
    p = base_validate(cfg)
    for path, sha in read(ROOT/'prepared.json')['artifacts'].items(): assert digest(ROOT/path) == sha, path
    return p


class Data(BaseData):
    def __init__(self, stage):
        super().__init__(stage); parent = Path(paired.config()['output']); e, t = self.common['pairs'].T
        self.message_ids = np.load(parent/'data/message_ids.npy')[e, t]
        choices = np.load(parent/'data/choice_ids.npy')[e]
        self.choice_ids = np.take_along_axis(choices[:, None, :], (self.s == 1)[..., None].astype(np.int64), -1)[..., 0]
        self.offsets = np.load(ROOT/'data/offsets.npy')
        self.tokens = np.load(ROOT/'features'/stage/'tokens.npy', mmap_mode='r'); self.extras = np.load(ROOT/'data/extras.npy', mmap_mode='r')
        self.text_kind = 'arguments'

    def batch(self, ids, condition=None):
        result = super().batch(ids, condition)
        selected = (self.message_ids if self.text_kind == 'arguments' else self.choice_ids)[ids]
        unique, inverse = np.unique(selected, return_inverse=True); lengths = self.offsets[unique+1]-self.offsets[unique]
        length = max(int(lengths.max()), 1); values = np.zeros((len(unique), length, 37), dtype=np.float32)
        mask = np.arange(length)[None, :] < lengths[:, None]
        for j, text_id in enumerate(unique):
            lo, hi = self.offsets[text_id:text_id+2]; n = int(hi-lo)
            values[j, :n, :32] = self.tokens[lo:hi]; values[j, :n, 32:] = self.extras[lo:hi]
        result.update(token_features=torch.as_tensor(values), token_mask=torch.as_tensor(mask), token_inverse=torch.as_tensor(inverse.reshape(selected.shape)))
        return result


def worker(name, phase):
    class SelectedData(Data):
        def __init__(self, stage):
            super().__init__(stage); self.text_kind = config()['models'][name]['text']
    core.Data, core.make_model, core.validate = SelectedData, make_model, validate
    core.worker(config(), name, 'mode', phase)


def phase(label):
    active, done = {}, {}; pending = list(config()['models'])
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
    encode(); validate(config())
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
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'encode', 'run', 'worker']); parser.add_argument('--model'); parser.add_argument('--phase', choices=['tune', 'refit']); args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'encode': encode()
    elif args.action == 'run': run()
    else: worker(args.model, args.phase)
