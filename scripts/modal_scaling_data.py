"""Audited 320-to-512 augmentation using frozen 320-question representations."""
from argparse import ArgumentParser
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import modal_neural_study as core
from scripts import modal_causal_history as history
from scripts import modal_question_arguments as paired
from scripts import breadth_argument_features as encoding
from scripts import breadth_train
from scripts.local_conversation_pilot import deliver, formatter, opinion_prompt, aggregate

ROOT = Path('runs/modal_scaling_512_v1')
DATA = ROOT/'data'
MANIFEST = Path('runs/hpc3_scale_3200_v1/status_checks/20261115T074918Z_first192.json')
read, now = core.read, core.now


def audit_export():
    complete = read(ROOT/'export/complete.json'); manifest = read(MANIFEST)
    assert complete['manifest_sha256'] == digest(MANIFEST)
    assert complete['export_code_sha256'] == digest('scripts/export_modal_scaling.py')
    old = read(core.state.config()['index']); fold = read(Path(core.config()['output'])/'protocol.json')['fold']
    assert not ({r['qid'] for r in old}|set(fold['assessment_questions'])) & {r['qid'] for r in manifest['selected_units']}
    graphs = {g['id']: np.asarray(g['J']) for g in read('data/raw/upstream/data/J_matrices.json')['graphs']}
    personas = read(old[0]['path'])['personas']; fmt = formatter()
    probabilities = np.empty((192, 9, 32), dtype=np.float64); masses = np.empty_like(probabilities)
    quality = Counter(); index = []; max_spread = 0.; mode_disagreements = 0; execution_types = Counter()
    for i, item in enumerate(manifest['selected_units']):
        relative = 'questions/'+item['qid']+'.json.gz'; path = ROOT/'export'/relative
        assert complete['files'][relative] == digest(path)
        with gzip.open(path, 'rt') as stream: value = json.load(stream)
        assert value['identity']['manifest_sha256'] == digest(MANIFEST)
        assert value['episode_sha256'] == item['sha256'] and value['source_path'] == item['path']
        ep = value['episode']; assert ep['meta']['qid'] == item['qid'] and ep['meta']['split'] == 'train'
        assert ep['meta']['protocol_sha256'] == manifest['source_protocol_sha256']
        assert ep['personas'] == personas
        np.testing.assert_array_equal(ep['J'], graphs[item['graph_id']])
        assert np.asarray(ep['spins_history']).shape == (9, 32) and np.isin(ep['spins_history'], [-1, 1]).all()
        rows = {(r['round'], r['agent']): r for r in value['rows']}; assert len(rows) == 288
        for t in range(9):
            agree, disagree = [[] for _ in range(32)], [[] for _ in range(32)]
            if t:
                senders = {}
                for edge, text in ep['messages_history'][t-1].items():
                    sender, receiver = map(int, edge.split(','))
                    assert ep['J'][sender][receiver] != 0 and (sender not in senders or senders[sender] == text)
                    senders[sender] = text
                agree, disagree, edges = deliver(ep['J'], senders, ep['meta']['name'], t)
                assert edges == ep['messages_history'][t-1]
            for a in range(32):
                row = rows[t, a]; assert row['qid'] == item['qid'] and row['split'] == 'train'
                assert row['raw_votes'] == ep['spins_raw_history'][t][a]
                assert aggregate(row['raw_votes']) == ep['spins_history'][t][a]
                prompt = opinion_prompt(fmt, ep['personas'][a], {'question': ep['statement'], 'choices': ep['choices']}, agree[a], disagree[a])
                assert hashlib.sha256(prompt.encode()).hexdigest() == row['prompt_sha256']
                aa, bb = [np.asarray(row[k], dtype=np.float64) for k in ('first_literal_A_probabilities', 'first_literal_B_probabilities')]
                assert aa.shape == bb.shape == (5,) and (aa+bb > 0).all()
                pp = aa/(aa+bb); assert ((pp > 0)&(pp < 1)).all()
                probabilities[i, t, a] = pp[0]; masses[i, t, a] = max(0., 1-aa[0]-bb[0])
                max_spread = max(max_spread, float(np.ptp(pp)))
                mode_disagreements += int(((pp > .5) != (pp[0] > .5)).any())
        for t in range(8): encoding.snapshot_messages(ep, t)
        quality.update(value['quality'])
        devices = {v['runtime']['gpu_device']['device_name'] for v in value['executions'].values()}
        assert len(devices) == 1; execution_types.update(devices)
        local = DATA/'episodes/train'/(ep['meta']['name']+'.json'); json_write(local, ep)
        index.append(dict(qid=item['qid'], split='train', origin='expansion_3200', graph_id=item['graph_id'], path=str(local),
            sha256=digest(local), source_path=item['path'], source_sha256=item['sha256'], metadata=value['question_metadata']))
        if (i+1) % 32 == 0: print({'audit_questions': i+1, 'prompts': (i+1)*288}, flush=True)
    np.savez_compressed(DATA/'new_probabilities.npz', single=probabilities, other_mass=masses)
    json_write(DATA/'new_index.json', index)
    result = dict(passed=True, completed_utc=now(), questions=192, prompt_reconstructions=192*288, raw_votes=192*288*5,
        manifest_sha256=digest(MANIFEST), export_sha256=digest(ROOT/'export/complete.json'),
        probability_sha256=digest(DATA/'new_probabilities.npz'), index_sha256=digest(DATA/'new_index.json'),
        probability_policy='First original response per prompt, normalized over literal A/B, identical to old400. No confidence filtering.',
        max_within_prompt_normalized_probability_spread=max_spread, prompts_with_disagreeing_modes=mode_disagreements,
        other_mass_mean=float(masses.mean()), other_mass_max=float(masses.max()), quality=dict(quality), execution_types=dict(execution_types),
        new_validation_loaded=False, assessment_loaded=False, source_calls=0)
    json_write(ROOT/'source_audit.json', result); print(result, flush=True)


def prepare_texts():
    if (DATA/'text_plan.json').exists(): return read(DATA/'text_plan.json')
    audit_export()
    old_common = dict(np.load(core.state.config()['common'])); index = read(DATA/'new_index.json')
    eps = [read(r['path']) for r in index]; qids = np.concatenate([old_common['episode_qids'], [r['qid'] for r in index]])
    common = {k: np.concatenate([old_common[k], np.asarray([ep[source] for ep in eps])])
        for k, source in [('spins', 'spins_history'), ('J', 'J'), ('raw_votes', 'spins_raw_history')]}
    common.update(episode_qids=qids, pairs=np.asarray([(e,t) for e in range(592) for t in range(8)], dtype=np.int64))
    np.testing.assert_array_equal(common['pairs'][:3200], old_common['pairs'])
    np.savez_compressed(DATA/'common.npz', **common)
    p = np.load(Path(core.dist.config()['probability_reference'])/'probabilities.npz')['single']
    pnew = np.load(DATA/'new_probabilities.npz')['single']; np.save(DATA/'single.npy', np.concatenate([p, pnew]))
    old_index = {r['qid']: r for r in read(core.state.config()['index'])}
    json_write(DATA/'index.json', [old_index[q] for q in old_common['episode_qids']]+index)
    names = [Path(r['path']).stem for r in read(DATA/'index.json')]
    neighbors, signs = core.dist.ordered.pair.neighbor_slots(common['J'])
    order = core.state.ordered_neighbors(common['J'], names, common['pairs'])
    positions = core.dist.ordered.positions_for_slots(common, order, neighbors)
    original = np.load(Path(core.dist.config()['ordered_reference'])/'data/positions.npz')
    for key, a in [('neighbors', neighbors[:400]), ('signs', signs[:400]), ('actual', positions[:3200])]: np.testing.assert_array_equal(a, original[key])
    np.savez_compressed(DATA/'positions.npz', neighbors=neighbors, signs=signs, actual=positions, order=order)
    old_plain = read('runs/breadth_arguments_320_v1/data/texts.json')
    old_paired = read(Path(paired.config()['output'])/'data/texts.json')
    plain_lookup, paired_lookup = {s:i for i,s in enumerate(old_plain)}, {s:i for i,s in enumerate(old_paired)}
    texts, global_lookup, new_plain, new_paired, qtext = [], {}, [], [], []
    def add(text):
        if text not in global_lookup: global_lookup[text] = len(texts); texts.append(text)
        return global_lookup[text]
    def add_group(text, lookup, old_count, new_ids):
        if text not in lookup:
            lookup[text] = old_count+len(new_ids); new_ids.append(add(text))
        return lookup[text]
    ids = np.concatenate([np.load(core.state.config()['message_ids']), np.zeros((192,8,32), dtype=np.int32)])
    pair_ids = np.concatenate([np.load(Path(paired.config()['output'])/'data/message_ids.npy'), np.zeros((192,8,32), dtype=np.int32)])
    pair_choice = np.concatenate([np.load(Path(paired.config()['output'])/'data/choice_ids.npy'), np.zeros((192,2), dtype=np.int64)])
    lexical = np.concatenate([np.load(core.lexical.ROOT/'sender_lexical.npy'), np.zeros((192,8,32,8), dtype=np.float32)])
    for offset, ep in enumerate(eps, start=400):
        qtext.append(add(breadth_train.question_text({'question': ep['statement'], 'choices': ep['choices']})))
        for sign, label in enumerate(('B','A')):
            pair_choice[offset, sign] = add_group(paired.pair_text(ep['statement'], ep['choices'], label), paired_lookup, len(old_paired), new_paired)
        for t in range(8):
            for a, text in enumerate(encoding.snapshot_messages(ep, t)):
                if not text: continue
                ids[offset,t,a] = add_group(text, plain_lookup, len(old_plain), new_plain)
                pair_ids[offset,t,a] = add_group(paired.pair_text(ep['statement'], ep['choices'], text), paired_lookup, len(old_paired), new_paired)
                lexical[offset,t,a] = core.lexical.text_features(text)
    for name, array in [('message_ids', ids), ('pair_message_ids', pair_ids), ('pair_choice_ids', pair_choice), ('sender_lexical', lexical)]: np.save(DATA/(name+'.npy'), array)
    json_write(DATA/'new_texts.json', texts)
    plan = dict(created_utc=now(), plain=new_plain, paired=new_paired, questions=qtext,
        old_plain=len(old_plain), old_paired=len(old_paired), texts=len(texts), texts_sha256=digest(DATA/'new_texts.json'))
    json_write(DATA/'text_plan.json', plan); print({k:v for k,v in plan.items() if k not in ('plain','paired','questions')}, flush=True)
    return plan


def encode_bounded(texts, tokenizer, model, device):
    """Complete-token pooling with bounded MPS graph shapes and frozen weights."""
    tokens = tokenizer(texts, add_special_tokens=False, truncation=False, verbose=False)['input_ids']
    assert all(tokens)
    windows = [(i, t[j:j+510]) for i,t in enumerate(tokens) for j in range(0,len(t),510)]
    sums = np.zeros((len(texts),384), np.float64); counts = np.zeros(len(texts), np.int64)
    for low, high in [(0,128),(128,256),(256,384),(384,512)]:
        pieces = [(i,t) for i,t in windows if low < len(t)+2 <= high]
        for begin in range(0,len(pieces),16):
            pack = pieces[begin:begin+16]
            ids = torch.full((16,high),tokenizer.pad_token_id,dtype=torch.long,device=device)
            attention = torch.zeros_like(ids); pool = torch.zeros_like(ids)
            ids[:,0] = tokenizer.cls_token_id; ids[:,1] = tokenizer.sep_token_id; attention[:,:2] = 1
            for k,(_,t) in enumerate(pack):
                ids[k,:len(t)+2] = torch.tensor([tokenizer.cls_token_id]+t+[tokenizer.sep_token_id],device=device)
                attention[k,:len(t)+2] = 1; pool[k,1:len(t)+1] = 1
            with torch.inference_mode(): hidden = model(input_ids=ids,attention_mask=attention).last_hidden_state
            values = (hidden*pool[...,None]).sum(1).cpu().numpy()
            for k,(i,t) in enumerate(pack): sums[i] += values[k]; counts[i] += len(t)
    np.testing.assert_array_equal(counts, [len(t) for t in tokens])
    return (sums/np.linalg.norm(sums,axis=-1,keepdims=True)).astype(np.float32), counts


def encode():
    plan = prepare_texts()
    if (ROOT/'encoding.json').exists(): return
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2); device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(paired.config()['encoder_snapshot'], local_files_only=True)
    model = AutoModel.from_pretrained(paired.config()['encoder_snapshot'], local_files_only=True).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    texts = read(DATA/'new_texts.json'); progress_path = ROOT/'encoding_progress.json'
    if progress_path.exists():
        progress = read(progress_path); assert progress['text_plan_sha256'] == digest(DATA/'text_plan.json')
        vectors = np.lib.format.open_memmap(DATA/'new_vectors.npy', mode='r+'); lengths = np.lib.format.open_memmap(DATA/'new_lengths.npy', mode='r+')
    else:
        progress = dict(next_text=0, seconds=0., text_plan_sha256=digest(DATA/'text_plan.json'), device=device)
        vectors = np.lib.format.open_memmap(DATA/'new_vectors.npy', mode='w+', dtype=np.float32, shape=(len(texts),384))
        lengths = np.lib.format.open_memmap(DATA/'new_lengths.npy', mode='w+', dtype=np.int32, shape=(len(texts),))
    started = time.monotonic(); previous = progress['seconds']
    for begin in range(progress['next_text'],len(texts),1024):
        end = min(begin+1024,len(texts)); v,n = encode_bounded(texts[begin:end],tokenizer,model,device)
        vectors[begin:end] = v; lengths[begin:end] = n; vectors.flush(); lengths.flush()
        progress.update(next_text=end, total_texts=len(texts), seconds=previous+time.monotonic()-started, updated_utc=now())
        json_write(progress_path, progress)
        print({'encoded':end,'total':len(texts),'seconds':progress['seconds']},flush=True)
    np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6)
    # Independently reproduce a fixed training-only subset on CPU using the old pooling implementation.
    ids = np.unique(np.concatenate([np.arange(3),np.argsort(lengths)[-3:]]))
    model.to('cpu'); old_cfg = paired.config(); old_cfg['encoding']['batch_size'] = 8
    check, count = encoding.encode_block([texts[i] for i in ids],tokenizer,model,'cpu',old_cfg)
    error = float(np.max(np.abs(vectors[ids]-check))); assert error < 2e-5
    np.testing.assert_array_equal(lengths[ids],count)
    json_write(ROOT/'encoding.json',dict(completed_utc=now(),device=device,seconds=progress['seconds'],texts=len(texts),
        tokens=int(lengths.sum()),multiwindow_texts=int((lengths>510).sum()),tokens_dropped=0,
        frozen_encoder_parameters=sum(p.numel() for p in model.parameters()),cpu_replay_texts=len(ids),cpu_max_vector_difference=error,
        vector_sha256=digest(DATA/'new_vectors.npy'),lengths_sha256=digest(DATA/'new_lengths.npy'),text_plan_sha256=digest(DATA/'text_plan.json')))


def context_from_frozen(qvectors, pvectors, fitted):
    q = (qvectors.astype(float)-fitted['question_mean'])@fitted['question_components'].T
    p = (pvectors.astype(float)-fitted['persona_mean'])@fitted['persona_components'].T
    e,n = len(q),len(p)
    return np.concatenate([np.ones((e,n,1)),np.broadcast_to(p,(e,n,3)),np.broadcast_to(q[:,None],(e,n,3)),
        (p[None,:,:,None]*q[:,None,None,:]).reshape(e,n,9)],-1).astype(np.float32)


def features():
    encode()
    if (ROOT/'features_complete.json').exists(): return
    plan = read(DATA/'text_plan.json'); new = np.load(DATA/'new_vectors.npy',mmap_mode='r')
    fitted = np.load(Path(core.state.config()['output'])/'features/full/transforms.npz')
    old_context = np.load(core.state.config()['context_vectors'])
    qvectors = np.concatenate([old_context['question_vectors'],new[plan['questions']]])
    context = context_from_frozen(qvectors,old_context['persona_vectors'],fitted)
    np.testing.assert_allclose(context[:400],fitted['context'],rtol=0,atol=1e-7)
    context[:400] = fitted['context']
    np.save(DATA/'context.npy',context)
    q32 = ((qvectors-fitted['question32_mean'])@fitted['question32_axes'].T).astype(np.float32); np.save(DATA/'question32.npy',q32)
    old_reduced = np.load(Path(core.dist.config()['ordered_reference'])/'data/vectors_full.npy')
    with threadpool_limits(limits=2): reduced = ((new[plan['plain']].astype(float)-fitted['message_mean'])@fitted['message_axes'].T/(float(fitted['message_scale'])*math.sqrt(8))).astype(np.float32)
    np.save(DATA/'vectors8.npy',np.concatenate([old_reduced,reduced]))
    old_pairs = np.load(Path(paired.config()['output'])/'data/vectors.npy')
    np.save(DATA/'paired_vectors.npy',np.concatenate([old_pairs,new[plan['paired']]]))
    # Construct only new tree rows; old rows remain byte-identical.
    common = dict(np.load(DATA/'common.npz')); new_common = {k:common[k][400:] for k in ('spins','J','raw_votes','episode_qids')}
    new_common['pairs'] = common['pairs'][3200:].copy(); new_common['pairs'][:,0] -= 400
    pos = np.load(DATA/'positions.npz'); order = pos['order'][3200:]; e,t = new_common['pairs'].T
    b, slots, history_features, _, _ = core.state.state_features(new_common,context[400:],order,order)
    persona = np.broadcast_to(np.eye(32,dtype=np.float32)[None],(len(e),32,32)); question = np.broadcast_to(q32[400:][e,None],(len(e),32,32))
    current_votes = (new_common['raw_votes'][e,t] == 1).mean(-1); initial_votes = (new_common['raw_votes'][e,0] == 1).mean(-1)
    globals_ = np.broadcast_to(np.stack([current_votes.mean(-1),initial_votes.mean(-1)],-1)[:,None],(len(e),32,2))
    rich = np.concatenate([b,slots[0],history_features,question,persona,globals_],-1).astype(np.float32)
    probabilities = np.load(DATA/'single.npy')[400:]
    current,initial,hist = core.screen.probability_features(probabilities,new_common['pairs'],new_common['J'][e],order)
    lex = np.load(DATA/'sender_lexical.npy')[400:][e,t]; ids = np.load(DATA/'message_ids.npy')[400:][e,t]
    vec = np.load(DATA/'vectors8.npy')
    tree = np.concatenate([rich,current,initial,hist,core.screen.slots(lex,order),core.screen.slots(vec[ids],order)],-1)
    tree = np.ascontiguousarray(tree.reshape(-1,tree.shape[-1]),dtype=np.float32)
    old_tree = np.load(core.screen.ROOT/'features/full/semantic8.npy',mmap_mode='r')
    target = np.lib.format.open_memmap(DATA/'tree_semantic8.npy',mode='w+',dtype=np.float32,shape=(len(old_tree)+len(tree),old_tree.shape[-1]))
    target[:len(old_tree)] = old_tree; target[len(old_tree):] = tree; target.flush()
    paths = [f for f in DATA.iterdir() if f.is_file()]
    json_write(ROOT/'features_complete.json',dict(completed_utc=now(),artifacts={str(f):digest(f) for f in paths},
        frozen_transforms_sha256=digest(Path(core.state.config()['output'])/'features/full/transforms.npz'),
        old_features_preserved=True,feature_transform_fit_questions=320,new_transform_fits=0,assessment_loaded=False))
    print({'features_complete':True,'questions':592,'training':512,'validation':80},flush=True)


if __name__ == '__main__':
    p = ArgumentParser(); p.add_argument('action',choices=['audit','prepare','encode','features']); a=p.parse_args()
    DATA.mkdir(parents=True,exist_ok=True)
    {'audit':audit_export,'prepare':prepare_texts,'encode':encode,'features':features}[a.action]()
