"""Frozen delivered-message inputs for the separate 320/80 argument study."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config as load_config, code_digest, digest, environment, json_write

CONFIG = Path('configs/breadth_arguments_320.yaml')
CONCLUSIONS = ['', 'I favor option B.', 'I favor option A.']

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def config(): return load_config(CONFIG)

def snapshot_messages(episode, t):
    """Only state t and messages delivered for t -> t+1 enter this interface."""
    graph = np.asarray(episode['J'])
    state = np.asarray(episode['spins_history'][t])
    messages = episode['messages_history'][t]
    communicated = episode['communicated_stances_history'][t]
    n = len(state)
    assert graph.shape == (n, n) and np.isin(state, [-1, 1]).all()
    assert np.array_equal(graph, graph.T) and not np.diag(graph).any()
    expected = {f'{i},{j}' for i, j in zip(*np.nonzero(graph))}
    assert set(messages) == expected, 'Missing or extra delivered edge'
    result = [''] * n
    for sender in range(n):
        targets = np.flatnonzero(graph[sender])
        if not len(targets): continue
        assert communicated[str(sender)] == state[sender], 'Stance/state misalignment'
        texts = {messages[f'{sender},{int(receiver)}'] for receiver in targets}
        assert len(texts) == 1, 'Source is not broadcast; need edge-specific representation'
        text = texts.pop()
        assert isinstance(text, str) and text.strip() and not text.startswith('[error:')
        result[sender] = text
    return result

def verify_sources(cfg):
    source, ref = Path(cfg['source']), Path(cfg['reference'])
    index = read(source/'episode_index.json')
    assert Counter(r['split'] for r in index) == {s: cfg['counts'][s] for s in ('train', 'validation')}
    assert len({r['qid'] for r in index}) == len(index)
    old = read(ref/'predictors/protocol.json')
    assert code_digest() == old['core_sha256']
    frozen_paths = [ref/'models_frozen.json', Path(cfg['size_reference'])/'models_frozen.json']
    prior = {}
    for frozen_path in frozen_paths:
        f = read(frozen_path)
        for name, sha in f['artifacts'].items():
            assert digest(name) == sha, name
            prior[name] = sha
        prior[str(frozen_path)] = digest(frozen_path)
    for path in [ref/'development_results/summary.json', Path(cfg['size_reference'])/'results/summary.json',
                 Path('runs/breadth_320_learning_curve_v1/models_frozen.json'),
                 Path('runs/breadth_320_learning_curve_v1/results/summary.json')]:
        prior[str(path)] = digest(path)
    for r in index:
        path = Path(r['path'])
        assert path.parent.name in ('train', 'validation')
        assert digest(path) == r['sha256'], str(path)
    for name, sha in old['encoder_files'].items(): assert digest(name) == sha, name
    return index, old, prior

def prepare(cfg):
    root = Path(cfg['output'])
    if (root/'prepared.json').exists():
        validate(cfg)
        print(json.dumps({'prepared': True, 'reused': True}), flush=True)
        return
    index, old, prior = verify_sources(cfg)
    root.mkdir(parents=True, exist_ok=True)
    data_root = root/'data'; data_root.mkdir(exist_ok=True)
    ref = Path(cfg['reference'])/'predictors'
    copies = {'development.npz': ref/'development.npz', 'features.npz': ref/'folds/main/features.npz',
              'ising.json': ref/'folds/main/ising.json'}
    for name, path in copies.items(): shutil.copy2(path, data_root/name)
    common = dict(np.load(data_root/'development.npz', allow_pickle=False))
    qids = common['episode_qids'].tolist()
    rows = {r['qid']: r for r in index}
    assert set(rows) == set(qids)
    texts, lookup = list(CONCLUSIONS), {s:i for i,s in enumerate(CONCLUSIONS)}
    text_ids = np.zeros((len(qids), 8, 32), np.int32)
    stats = Counter()
    for e, qid in enumerate(qids):
        episode = read(rows[qid]['path'])
        assert episode['meta']['qid'] == qid
        np.testing.assert_array_equal(episode['J'], common['J'][e])
        np.testing.assert_array_equal(episode['spins_history'], common['spins'][e])
        assert len(episode['messages_history']) == len(episode['communicated_stances_history']) == 8
        for t in range(8):
            values = snapshot_messages(episode, t)
            for sender, text in enumerate(values):
                if not text: continue
                if text not in lookup:
                    lookup[text] = len(texts); texts.append(text)
                text_ids[e, t, sender] = lookup[text]
                stats['sender_messages'] += 1
                stats['delivered_edges'] += int(np.count_nonzero(common['J'][e, sender]))
                expected = 'A' if common['spins'][e,t,sender] == 1 else 'B'
                claims = re.findall(r'\bI favor option\s+([AB])\b',text,flags=re.I)
                matched = bool(claims) and {x.upper() for x in claims} == {expected}
                stats['explicit_matching_stance' if matched else 'explicit_stance_mismatch' if claims else 'no_standard_stance_phrase'] += 1
    np.save(data_root/'message_ids.npy', text_ids, allow_pickle=False)
    json_write(data_root/'texts.json', texts)
    json_write(data_root/'episode_metadata.json', [rows[q] for q in qids])
    fold = old['fold']
    assert not set(fold['train_questions']) & set(fold['validation_questions'])
    assert not set(qids) & set(fold['assessment_questions'])
    files = [CONFIG, Path(__file__), Path('scripts/breadth_argument_study.py')]
    protocol = {
        'created_utc': now(), 'config': cfg, 'config_sha256': digest(CONFIG),
        'code_files': {str(p): digest(p) for p in files}, 'core_sha256': code_digest(),
        'source_artifacts': {r['path']: r['sha256'] for r in index},
        'source_index': {str(Path(cfg['source'])/'episode_index.json'): digest(Path(cfg['source'])/'episode_index.json')},
        'prior_artifacts': prior, 'encoder_files': old['encoder_files'], 'fold': fold,
        'forecast_boundary': 'After the eight-round source broadcast for the current update has been delivered; before any target vote. Original 818 and Ising are earlier-boundary reference forecasts.',
        'feature_policy': 'Reuse exact 320-training-question PCA and fixed known-persona context. No history, raw votes, source probabilities, correct answers, solutions, question IDs or future states in inputs. Actual delivered sender messages are frozen 384-dimensional MiniLM vectors, no message PCA. Queries use receiver opinion/context only; keys/values additionally use the sender message.',
        'control_policy': 'Conclusion-only branch receives the fixed text I favor option A. or I favor option B., determined solely by the verified current sender opinion. Same architecture, initialization, optimizer and search as arguments; different input rank is intentional, so equal nominal parameters do not imply equal effective capacity.',
        'model_policy': 'Extend original width-8 and width-128 GraphUpdate with a bias-free 384-to-width projection added to normalized sender state for keys/values. Scale unit-norm embeddings by sqrt(384) to give projection activations order-one initialization. Receiver local state and queries remain unchanged. Four heads, one signed attention block. No encoder training.',
        'selection_policy': 'Nine LR/output-penalty choices per width/condition at seed42, validation-NLL stopping; freeze selected settings and duration, then refit seeds142/143/144 on the same 320 questions. Average individual scores, not probabilities.',
        'evaluation_policy': 'Primary contrasts arguments minus conclusion at each width. Compare frozen 818, full opinion-only and Ising references. Paired 5000-draw question bootstrap after seed averaging; exploratory, pointwise and conditional on fitted models; validation is reused for selection. Report train/validation NLL,error,Brier and stay/switch errors. Replacing messages at evaluation is a predictor sensitivity diagnostic, not a new source intervention.',
        'assessment_policy': cfg['evaluation']['assessment_policy'], 'source_llm_calls': 0,
        'assessment_loaded': False, 'autonomous_rollout_eligible': False,
        'message_audit': {**stats, 'unique_texts_including_controls': len(texts)-1,
                          'all_broadcasts_verified': True, 'all_current_stances_verified': True},
    }
    json_write(root/'protocol.json', protocol)
    json_write(root/'prepared.json', {'protocol_sha256': digest(root/'protocol.json'),
        'artifacts': {str(p.relative_to(root)): digest(p) for p in data_root.iterdir() if p.is_file()}})
    print(json.dumps({'stage': 'prepared', 'audit': protocol['message_audit'], 'assessment_loaded': False}), flush=True)

def validate(cfg, encoded=False):
    root = Path(cfg['output']); protocol = read(root/'protocol.json')
    assert protocol['config'] == cfg and protocol['config_sha256'] == digest(CONFIG)
    assert code_digest() == protocol['core_sha256']
    for key in ('code_files', 'source_index', 'encoder_files'):
        for p, sha in protocol[key].items(): assert digest(p) == sha, p
    prepared = read(root/'prepared.json')
    assert prepared['protocol_sha256'] == digest(root/'protocol.json')
    for p, sha in prepared['artifacts'].items(): assert digest(root/p) == sha, p
    if encoded:
        f = read(root/'encoding.json')
        assert f['protocol_sha256'] == digest(root/'protocol.json')
        for p, sha in f['artifacts'].items(): assert digest(root/p) == sha, p
    return protocol

def encode_block(texts, tokenizer, model, device, cfg):
    tokens = tokenizer(texts, add_special_tokens=False, truncation=False, verbose=False)['input_ids']
    if not all(tokens): raise ValueError('Empty nonpadding text')
    width = cfg['encoding']['window_tokens']
    chunks = [(i, t[j:j+width]) for i,t in enumerate(tokens) for j in range(0,len(t),width)]
    chunks.sort(key=lambda item: len(item[1]))
    sums = np.zeros((len(texts),384), np.float64)
    observed = np.zeros(len(texts), np.int64)
    bs = cfg['encoding']['batch_size']
    for start in range(0,len(chunks),bs):
        pack = chunks[start:start+bs]; length = max(len(t) for _,t in pack)+2
        ids = torch.full((len(pack),length),tokenizer.pad_token_id,dtype=torch.long,device=device)
        attention = torch.zeros_like(ids); pool = torch.zeros_like(ids)
        for k,(_,t) in enumerate(pack):
            ids[k,:len(t)+2] = torch.tensor([tokenizer.cls_token_id]+t+[tokenizer.sep_token_id],device=device)
            attention[k,:len(t)+2] = 1; pool[k,1:len(t)+1] = 1
        with torch.inference_mode():
            hidden = model(input_ids=ids,attention_mask=attention).last_hidden_state
            vectors = (hidden*pool[...,None]).sum(1).cpu().numpy()
        for k,(i,t) in enumerate(pack): sums[i] += vectors[k]; observed[i] += len(t)
    np.testing.assert_array_equal(observed, [len(t) for t in tokens])
    return (sums/np.linalg.norm(sums,axis=1,keepdims=True)).astype(np.float32), observed

def encode(cfg, device):
    validate(cfg)
    root = Path(cfg['output'])
    if (root/'encoding.json').exists(): validate(cfg,True); return
    texts = read(root/'data/texts.json')
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer, AutoModel
    snapshot = str(Path(cfg['encoder_snapshot']).resolve())
    tokenizer = AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    model = AutoModel.from_pretrained(snapshot,local_files_only=True).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    progress_path = root/'encoding_progress.json'
    identity = {'protocol_sha256':digest(root/'protocol.json'),'device':str(device),'texts_sha256':digest(root/'data/texts.json')}
    if progress_path.exists():
        progress = read(progress_path)
        assert all(progress[k] == v for k,v in identity.items())
        matrix = np.lib.format.open_memmap(root/'data/vectors.npy',mode='r+')
        lengths = np.lib.format.open_memmap(root/'data/token_counts.npy',mode='r+')
    else:
        matrix = np.lib.format.open_memmap(root/'data/vectors.npy',mode='w+',dtype=np.float32,shape=(len(texts),384))
        lengths = np.lib.format.open_memmap(root/'data/token_counts.npy',mode='w+',dtype=np.int32,shape=(len(texts),))
        matrix[0] = 0; lengths[0] = 0; matrix.flush(); lengths.flush()
        progress = {**identity,'next_text':1,'seconds':0.,'started_utc':now()}
        json_write(progress_path,progress)
    initial = progress['seconds']; started = time.monotonic()
    for begin in range(progress['next_text'],len(texts),cfg['encoding']['block_size']):
        end = min(begin+cfg['encoding']['block_size'],len(texts))
        values,count = encode_block(texts[begin:end],tokenizer,model,device,cfg)
        assert np.isfinite(values).all()
        matrix[begin:end] = values; lengths[begin:end] = count
        matrix.flush(); lengths.flush()
        progress.update(next_text=end,seconds=initial+time.monotonic()-started,updated_utc=now(),total_texts=len(texts))
        json_write(progress_path,progress)
        print(json.dumps({'stage':'encoding','completed':end-1,'total':len(texts)-1,'seconds':progress['seconds']}),flush=True)
    np.testing.assert_allclose(np.linalg.norm(matrix[1:],axis=1),1,atol=2e-6)
    json_write(root/'encoding.json',{**identity,'completed_utc':now(),'seconds':progress['seconds'],
        'encoder_parameters':sum(p.numel() for p in model.parameters()),'unique_texts':len(texts)-1,
        'tokens':int(lengths.sum()),'maximum_tokens':int(lengths.max()),'multiwindow_texts':int((lengths>510).sum()),
        'tokens_dropped':0,'environment':environment(),
        'artifacts':{f'data/{n}':digest(root/'data'/n) for n in ('vectors.npy','token_counts.npy')}})
    print(json.dumps({'stage':'encoding_complete','seconds':progress['seconds']}),flush=True)

if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action',choices=['prepare','encode']);parser.add_argument('--device',default='cpu')
    args = parser.parse_args(); cfg = config(); torch.set_num_threads(cfg['training']['torch_threads'])
    if args.action == 'prepare': prepare(cfg)
    else: encode(cfg,args.device)
