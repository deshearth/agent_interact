"""Question-conditioned frozen text encoding, with matched choice controls."""
from argparse import ArgumentParser
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config as load_config, digest, json_write, code_digest
from scripts import modal_neural_study as core
from scripts import breadth_argument_features as encoding
from scripts import compact_argument_study as compact

CONFIG = Path('configs/modal_question_arguments_320.yaml')
read, now = core.read, core.now
BaseData, base_validate = core.Data, core.validate
def config(): return load_config(CONFIG)


def pair_text(statement, choices, message):
    return f"Question: {statement}\nOption A: {choices['A']}\nOption B: {choices['B']}\nMessage: {message}"


def prepare(cfg):
    root = Path(cfg['output'])
    if (root/'protocol.json').exists():
        p = read(root/'protocol.json'); assert p['config'] == cfg
        for path, sha in p['code_files'].items(): assert digest(path) == sha, path
        return p
    parent = core.dist.validate(core.dist.config()); core.screen.validate()
    common = dict(np.load(core.state.config()['common'])); index = {r['qid']: r for r in read(core.state.config()['index'])}
    original = np.load(core.state.config()['message_ids']); old_texts = read('runs/breadth_arguments_320_v1/data/texts.json')
    texts = ['']; lookup = {}; messages = np.zeros_like(original); choices = np.zeros((400, 2), dtype=np.int64)
    def add(text):
        if text not in lookup: lookup[text] = len(texts); texts.append(text)
        return lookup[text]
    for e, qid in enumerate(common['episode_qids']):
        item = index[qid]; assert item['split'] in ('train', 'validation') and digest(item['path']) == item['sha256']
        episode = read(item['path'])
        for spin, label in enumerate(('B', 'A')):
            choices[e, spin] = add(pair_text(episode['statement'], episode['choices'], label))
        for t in range(8):
            for sender in range(32):
                old = original[e, t, sender]
                if old: messages[e, t, sender] = add(pair_text(episode['statement'], episode['choices'], old_texts[old]))
    folder = root/'data'; folder.mkdir(parents=True, exist_ok=True)
    json_write(folder/'texts.json', texts); np.save(folder/'message_ids.npy', messages); np.save(folder/'choice_ids.npy', choices)
    files = [CONFIG, Path(__file__), Path(core.__file__), Path(core.trainer.__file__), Path(encoding.__file__), Path('tests/test_modal_question_arguments.py')]
    refs = [Path(core.config()['output'])/n for n in ('protocol.json', 'prepared.json')]
    refs += [core.screen.ROOT/'prepared.json', Path(core.state.config()['common']), Path(core.state.config()['index']),
             Path(core.state.config()['message_ids']), Path('runs/breadth_arguments_320_v1/data/texts.json')]
    encoder_files = read('runs/question_breadth_320_train_v1/predictors/protocol.json')['encoder_files']
    for path, sha in encoder_files.items(): assert digest(path) == sha
    p = dict(created_utc=now(), config=cfg, fold=parent['fold'], inner=parent['inner'], core_sha256=code_digest(),
        code_files={str(p): digest(p) for p in files}, references={str(p): digest(p) for p in refs}, encoder_files=encoder_files,
        parameters={name: sum(p.numel() for p in core.make_model(cfg, name).parameters()) for name in cfg['models']},
        input_artifacts={str(p.relative_to(root)): digest(p) for p in folder.iterdir()},
        hypothesis='Encoding question/options jointly with each current message preserves relationships lost by independent embeddings; full384 preserves more than PCA32.',
        features='Same rich numerical state and actual message order as the preceding wider model. Frozen MiniLM encodes Question/Options/Message jointly. '
            'Matched controls encode the same question/options with only current A/B. All tokens are retained using complete510-token windows and original token-sum pooling. '
            'PCA32 fits only actual-message IDs from inner256 or full320; full384 uses the frozen normalized vectors. Eight lexical fields retain matched effective scale.',
        selection='Mode labels only. Same two recipes, internal256/64 accuracy/NLL selection and three fresh-seed refits. All models freeze before outer80 scoring.',
        target='Most probable next A/B opinion on all existing histories; no confidence filtering, target tokens, solutions, or next opinions in features.',
        assessment_loaded=False, source_calls=0)
    json_write(root/'protocol.json', p); print({'prepared': True, 'texts': len(texts)-1, 'parameters': p['parameters']}, flush=True); return p


def encode(cfg):
    root = Path(cfg['output']); p = prepare(cfg)
    if (root/'prepared.json').exists(): return validate(cfg)
    for path, sha in p['input_artifacts'].items(): assert digest(root/path) == sha
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2); device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(cfg['encoder_snapshot'], local_files_only=True)
    model = AutoModel.from_pretrained(cfg['encoder_snapshot'], local_files_only=True).to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    texts = read(root/'data/texts.json'); progress_path = root/'encoding_progress.json'
    if progress_path.exists():
        progress = read(progress_path); assert progress['protocol_sha256'] == digest(root/'protocol.json')
        vectors = np.lib.format.open_memmap(root/'data/vectors.npy', mode='r+')
        lengths = np.lib.format.open_memmap(root/'data/token_counts.npy', mode='r+')
    else:
        vectors = np.lib.format.open_memmap(root/'data/vectors.npy', mode='w+', dtype=np.float32, shape=(len(texts), 384)); vectors[0] = 0
        lengths = np.lib.format.open_memmap(root/'data/token_counts.npy', mode='w+', dtype=np.int32, shape=(len(texts),)); lengths[0] = 0
        progress = dict(next_text=1, seconds=0., protocol_sha256=digest(root/'protocol.json'), device=device)
    start = time.monotonic(); prior_seconds = progress['seconds']
    for begin in range(progress['next_text'], len(texts), cfg['encoding']['block_size']):
        end = min(begin+cfg['encoding']['block_size'], len(texts))
        v, n = encoding.encode_block(texts[begin:end], tokenizer, model, device, cfg)
        vectors[begin:end] = v; lengths[begin:end] = n; vectors.flush(); lengths.flush()
        progress.update(next_text=end, seconds=prior_seconds+time.monotonic()-start, updated_utc=now(), total_texts=len(texts))
        json_write(progress_path, progress)
        print({'encoding': end-1, 'total': len(texts)-1, 'seconds': progress['seconds']}, flush=True)
    np.testing.assert_allclose(np.linalg.norm(vectors[1:], axis=1), 1, atol=2e-6)
    common = dict(np.load(core.state.config()['common'])); ids = np.load(root/'data/message_ids.npy')
    variance = {}
    with threadpool_limits(limits=2):
        for stage, questions in [('inner', p['inner']['fit']), ('full', p['fold']['train_questions'])]:
            folder = root/'features'/stage; folder.mkdir(parents=True, exist_ok=True)
            fit = np.isin(common['episode_qids'], questions)
            reduced, fitted = compact.fit_message_pca(vectors, ids[fit], 32)
            np.save(folder/'vectors32.npy', reduced); np.savez(folder/'pca32.npz', **fitted)
            variance[stage] = float(fitted['explained_variance_fraction'])
    json_write(root/'encoding.json', dict(completed_utc=now(), seconds=progress['seconds'], device=device, frozen_encoder_parameters=sum(v.numel() for v in model.parameters()),
        tokens=int(lengths.sum()), max_tokens=int(lengths.max()), multiwindow_texts=int((lengths > 510).sum()), tokens_dropped=0, pca32_variance=variance))
    files = [f for folder in ('data', 'features') for f in (root/folder).rglob('*') if f.is_file()]
    json_write(root/'prepared.json', dict(protocol_sha256=digest(root/'protocol.json'), encoding_sha256=digest(root/'encoding.json'), artifacts={str(f.relative_to(root)): digest(f) for f in files}))
    print({'encoding_complete': True, 'seconds': progress['seconds'], 'variance': variance}, flush=True)


def validate(cfg):
    p = base_validate(cfg); root = Path(cfg['output'])
    for path, sha in read(root/'prepared.json')['artifacts'].items(): assert digest(root/path) == sha, path
    for path, sha in p['encoder_files'].items(): assert digest(path) == sha, path
    return p


class Data(BaseData):
    def __init__(self, stage):
        super().__init__(stage); root = Path(config()['output']); e, t = self.common['pairs'].T
        ids = np.load(root/'data/message_ids.npy')[e, t]; choice_ids = np.load(root/'data/choice_ids.npy')[e]
        chosen = np.take_along_axis(choice_ids[:, None, :], (self.s == 1)[..., None].astype(np.int64), -1)[..., 0]
        arg_lexical = self.values['text_arguments40'][..., -8:].numpy().copy()
        choice_lexical = self.values['text_conclusion40'][..., -8:].numpy().copy()
        for dim, path in [(40, root/'features'/stage/'vectors32.npy'), (392, root/'data/vectors.npy')]:
            vectors = np.load(path, mmap_mode='r'); scale = np.sqrt(40/dim)
            self.values['text_arguments'+str(dim)] = torch.as_tensor(np.concatenate([vectors[ids], arg_lexical*scale], -1), dtype=torch.float32)
            self.values['text_conclusion'+str(dim)] = torch.as_tensor(np.concatenate([vectors[chosen], choice_lexical*scale], -1), dtype=torch.float32)


def phase(cfg, label):
    root = Path(cfg['output']); pending = list(cfg['models']); active, done = {}, {}
    while pending or active:
        while pending and len(active) < cfg['training']['max_concurrent_models']:
            name = pending.pop(0); stream = (root/f'{label}_{name}.log').open('a', buffering=1)
            active[name] = subprocess.Popen([sys.executable, __file__, 'worker', '--model', name, '--phase', label], stdout=stream, stderr=subprocess.STDOUT), stream
        for name, (process, stream) in list(active.items()):
            if process.poll() is not None: stream.close(); done[name] = process.returncode; del active[name]
        json_write(root/'progress.json', dict(phase=label, updated_utc=now(), active={k: p.pid for k, (p, s) in active.items()}, pending=pending, completed=done))
        if any(done.values()):
            for process, stream in active.values(): process.terminate(); process.wait(); stream.close()
            raise RuntimeError(str(done))
        if active: time.sleep(2)


def run(cfg):
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    with (root/'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); encode(cfg); validate(cfg)
        if (root/'models_frozen.json').exists(): return
        phase(cfg, 'tune'); records = [(p.parent, read(p)) for p in (root/'tune').glob('*/manifest.json')]
        assert len(records) == 8 and all(r['status'] == 'complete' for p, r in records)
        choices = []
        for name in cfg['models']:
            path, best = min([(p, r) for p, r in records if r['width'] == name], key=lambda v: (v[1]['validation']['error'], v[1]['validation']['nll'], v[1]['selected_step'], v[1]['candidate']))
            choices.append(dict(width=name, condition='mode', candidate=best['candidate'], refit_steps=best['selected_step'], tuning_directory=str(path)))
        selection = dict(protocol_sha256=digest(root/'protocol.json'), choices=choices)
        if (root/'selection.json').exists(): assert read(root/'selection.json') == selection
        else: json_write(root/'selection.json', selection)
        phase(cfg, 'refit'); refits = [(p.parent, read(p)) for p in (root/'refit').glob('*/manifest.json')]
        assert len(refits) == 12 and all(r['status'] == 'complete' for p, r in refits)
        files = [root/n for n in ('protocol.json', 'prepared.json', 'encoding.json', 'selection.json')]
        files += [p for folder in ('tune', 'refit') for p in (root/folder).rglob('*') if p.is_file()]
        json_write(root/'models_frozen.json', dict(frozen_utc=now(), artifacts={str(p): digest(p) for p in files}, refits=[str(p) for p, r in refits], assessment_loaded=False))
        json_write(root/'progress.json', dict(phase='frozen_ready_to_evaluate', updated_utc=now()))


if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'encode', 'run', 'worker']); parser.add_argument('--model'); parser.add_argument('--phase', choices=['tune', 'refit'])
    args = parser.parse_args(); cfg = config()
    if args.action == 'prepare': prepare(cfg)
    elif args.action == 'encode': encode(cfg)
    elif args.action == 'run': run(cfg)
    else:
        core.Data = Data; core.validate = validate
        core.worker(cfg, args.model, 'mode', args.phase)
