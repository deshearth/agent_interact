"""Training-only feasibility screen of jointly encoded receiver prompts.

Every prompt token is retained, but MiniLM attends within 510-token windows.
The pilot covers two predeclared rounds on internal256/64 training questions;
it is not an all-case outer-validation result or a claim of 95% success.
"""
from argparse import ArgumentParser
from collections import Counter
import gzip
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_scaling_data as encoding
from scripts import modal_neural_study as core
from scripts.local_conversation_pilot import formatter,deliver,opinion_prompt

ROOT=Path('runs/modal_inbox_screen_320_v1')
ROUNDS=[0,4]
VARIANTS=['state','state_prompt32','state_prompt384','state_truth_oracle']
read,now=core.read,core.now


def prepare():
    if (ROOT/'protocol.json').exists(): return read(ROOT/'protocol.json')
    ROOT.mkdir(parents=True,exist_ok=True); folder=ROOT/'data'; folder.mkdir(exist_ok=True)
    parent=read(Path(core.config()['output'])/'protocol.json'); train=parent['fold']['train_questions']
    common=dict(np.load(core.state.config()['common'])); e,t=common['pairs'].T
    take=np.flatnonzero(np.isin(common['episode_qids'][e],train)&np.isin(t,ROUNDS))
    assert len(take)==640
    index={q['qid']:q for q in read(core.state.config()['index'])}; fmt=formatter(); texts=[]; source_hashes={}
    with gzip.open('runs/generalization_investigation_320_v1/source_probabilities.json.gz','rt') as stream: source=__import__('json').load(stream)
    hashes={(r['qid'],r['round'],r['agent']):r['prompt_sha256'] for r in source['rows'] if r['qid'] in set(train) and r['round'] in (1,5)}
    for episode,round_number in common['pairs'][take]:
        qid=str(common['episode_qids'][episode]); item=index[qid]; assert item['split']=='train' and digest(item['path'])==item['sha256']
        ep=read(item['path']); source_hashes[item['path']]=item['sha256']; senders={}
        for edge,text in ep['messages_history'][round_number].items():
            sender=int(edge.split(',')[0]); assert sender not in senders or senders[sender]==text; senders[sender]=text
        agree,disagree,edges=deliver(ep['J'],senders,ep['meta']['name'],int(round_number)+1)
        assert edges==ep['messages_history'][round_number]
        for agent in range(32):
            text=opinion_prompt(fmt,ep['personas'][agent],{'question':ep['statement'],'choices':ep['choices']},agree[agent],disagree[agent])
            assert __import__('hashlib').sha256(text.encode()).hexdigest()==hashes[qid,int(round_number)+1,agent]
            texts.append(text)
    json_write(folder/'texts.json',texts)
    single=np.load(Path(core.dist.config()['probability_reference'])/'probabilities.npz')['single']
    row_qids=np.repeat(common['episode_qids'][e[take]],32); y=(single[e[take],t[take]+1]>.5).astype(np.int64).ravel()
    source_index=read('runs/question_breadth_320_v1/selection.json')['questions']
    selected_questions=[q for q in source_index if q['qid'] in set(train)]
    assert all(q['answer'] in ('A','B') and q['choices'][q['answer']]==q['correct_answer'] for q in selected_questions)
    truth={q['qid']:1 if q['answer']=='A' else -1 for q in selected_questions}
    np.savez_compressed(folder/'labels.npz',target=y,qids=row_qids,pairs=common['pairs'][take],pair_indices=take,truth=np.asarray([truth[q] for q in row_qids]))
    refs=[Path(core.config()['output'])/'protocol.json',core.screen.ROOT/'features/inner/semantic8.npy',
        Path('runs/generalization_investigation_320_v1/source_probabilities.json.gz'),Path('runs/modal_scaling_512_v1/evaluation/completion_audit.json')]
    p=dict(created_utc=now(),hypothesis='Jointly contextualizing nearby incoming messages with the receiver persona preserves information lost by encoding each argument separately.',
        inner=parent['inner'],original_training=train,rounds=ROUNDS,pilot_prompts=len(texts),variants=VARIANTS,
        input_policy='Exact source opinion prompt, including receiver persona, question/options and ordered friendly/unfriendly incoming messages. No response appended. '
            'Frozen MiniLM retains all prompt tokens through independent510-token windows; context does not cross window boundaries. '
            'Append pooled full384 or training-only PCA32 to the same semantic8 state features. State_truth_oracle adds the known correct choice only as a privileged diagnostic, ineligible for promotion.',
        protocol_policy='Training-only pilot: fixed transitions t=0 and4, all32 agents, internal256/64 questions. All cases at those rounds retained. '
            'No outer80 or new validation used. Not an all-round95% result. Same booster settings for all variants; choose internal accuracy/NLL checkpoint.',
        promotion_policy='Only legitimate prompt32/prompt384 candidates may proceed to all-round training if internal accuracy improves at least0.5 percentage points over the matched state-only pilot. '
            'A promoted candidate must still be independently trained/refitted/frozen and tested on every outer update. Oracle truth can never satisfy the goal.',
        promotion_gain=.005,settings=core.screen.SETTINGS,
        source_artifacts=source_hashes,references={str(f):digest(f) for f in refs},
        code_files={str(f):digest(f) for f in [Path(__file__),Path(encoding.__file__),Path(core.screen.__file__)]},
        data_artifacts={str(f):digest(f) for f in folder.iterdir()},source_calls=0,outer_validation_scored=False,new_validation_loaded=False,assessment_loaded=False)
    json_write(ROOT/'protocol.json',p); print({'prepared':True,'prompts':len(texts)},flush=True); return p


def validate():
    p=prepare()
    for group in ('source_artifacts','references','code_files','data_artifacts'):
        for path,sha in p[group].items(): assert digest(path)==sha,path
    return p


def encode():
    p=validate()
    if (ROOT/'encoding.json').exists(): return
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(2); device='mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer=AutoTokenizer.from_pretrained(encoding.paired.config()['encoder_snapshot'],local_files_only=True)
    model=AutoModel.from_pretrained(encoding.paired.config()['encoder_snapshot'],local_files_only=True).to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    texts=read(ROOT/'data/texts.json'); progress_path=ROOT/'encoding_progress.json'
    if progress_path.exists():
        progress=read(progress_path); assert progress['protocol_sha256']==digest(ROOT/'protocol.json')
        vectors=np.lib.format.open_memmap(ROOT/'data/vectors.npy',mode='r+'); lengths=np.lib.format.open_memmap(ROOT/'data/lengths.npy',mode='r+')
    else:
        progress=dict(next_text=0,seconds=0.,protocol_sha256=digest(ROOT/'protocol.json'),device=device)
        vectors=np.lib.format.open_memmap(ROOT/'data/vectors.npy',mode='w+',dtype=np.float32,shape=(len(texts),384))
        lengths=np.lib.format.open_memmap(ROOT/'data/lengths.npy',mode='w+',dtype=np.int32,shape=(len(texts),))
    start=time.monotonic(); previous=progress['seconds']
    for begin in range(progress['next_text'],len(texts),512):
        end=min(begin+512,len(texts)); v,n=encoding.encode_bounded(texts[begin:end],tokenizer,model,device)
        vectors[begin:end]=v; lengths[begin:end]=n; vectors.flush(); lengths.flush()
        progress.update(next_text=end,total_texts=len(texts),seconds=previous+time.monotonic()-start,updated_utc=now())
        json_write(progress_path,progress); print({'encoded':end,'total':len(texts),'seconds':progress['seconds']},flush=True)
    np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6)
    labels=np.load(ROOT/'data/labels.npz'); train=np.isin(labels['qids'],p['inner']['fit']); a=np.asarray(vectors,dtype=float)
    with threadpool_limits(limits=2):
        mean=a[train].mean(0); cov=(a[train]-mean).T@(a[train]-mean); values,axes=np.linalg.eigh(cov); axes=axes[:,-32:][:,::-1].T
        reduced=((a-mean)@axes.T).astype(np.float32)
    np.save(ROOT/'data/vectors32.npy',reduced); np.savez(ROOT/'data/pca32.npz',mean=mean,axes=axes,training_indices=np.flatnonzero(train))
    json_write(ROOT/'encoding.json',dict(completed_utc=now(),seconds=progress['seconds'],device=device,texts=len(texts),tokens=int(lengths.sum()),
        max_tokens=int(lengths.max()),multiwindow_prompts=int((lengths>510).sum()),tokens_dropped=0,frozen_encoder_parameters=sum(x.numel() for x in model.parameters()),
        pca_fit_questions=len(set(labels['qids'][train])),pca32_variance=float(values[-32:].sum()/values.sum()),
        artifacts={str(f):digest(f) for f in (ROOT/'data').iterdir()}))


def screen():
    p=validate(); enc=read(ROOT/'encoding.json')
    for path,sha in enc['artifacts'].items(): assert digest(path)==sha,path
    labels=np.load(ROOT/'data/labels.npz'); fit=np.isin(labels['qids'],p['inner']['fit']); hold=np.isin(labels['qids'],p['inner']['validation'])
    assert fit.sum()==256*64 and hold.sum()==64*64 and (fit^hold).all()
    rows=(labels['pair_indices'][:,None]*32+np.arange(32)).ravel()
    baseline=np.load(core.screen.ROOT/'features/inner/semantic8.npy',mmap_mode='r')[rows]; y=labels['target']; results=[]
    for name in VARIANTS:
        folder=ROOT/'fits'/name; folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists(): results.append(read(folder/'complete.json')); continue
        if name=='state': x=baseline
        elif name=='state_truth_oracle': x=np.concatenate([baseline,labels['truth'][:,None]],-1).astype(np.float32)
        else: x=np.concatenate([baseline,np.load(ROOT/'data'/('vectors32.npy' if name.endswith('32') else 'vectors.npy'))],-1).astype(np.float32)
        model=HistGradientBoostingClassifier(**p['settings'],early_stopping=True); start=time.monotonic(); history=[]; best=None; chosen=None
        with threadpool_limits(limits=4):
            model.fit(x[fit],y[fit],X_val=x[hold],y_val=y[hold])
            for step,probability in enumerate(model.staged_predict_proba(x[hold]),1):
                pp=np.clip(probability[:,1],1e-10,1-1e-10); error=float(((pp>.5)!=y[hold]).mean()); nll=float((-y[hold]*np.log(pp)-(1-y[hold])*np.log1p(-pp)).mean())
                history.append(dict(step=step,error=error,nll=nll)); key=error,nll,step
                if best is None or key<best: best=key; chosen=pp.copy()
            train=model.predict_proba(x[fit])[:,1]
        np.save(folder/'selected_probability.npy',chosen); np.save(folder/'terminal_train_probability.npy',train)
        json_write(folder/'history.json',history)
        with (folder/'terminal_model.pkl').open('wb') as stream: pickle.dump(model,stream)
        record=dict(variant=name,selected_step=best[2],inner_accuracy=1-best[0],inner_nll=best[1],
            terminal_train_accuracy=float(((train>.5)==y[fit]).mean()),fitted_iterations=model.n_iter_,features=x.shape[-1],
            seconds=time.monotonic()-start,completed_utc=now(),eligible_for_promotion=name in ('state_prompt32','state_prompt384'),
            artifacts={str(f):digest(f) for f in folder.iterdir()})
        json_write(folder/'complete.json',record); results.append(record); print(record,flush=True)
    baseline_result=next(r for r in results if r['variant']=='state'); best=max([r for r in results if r['eligible_for_promotion']],key=lambda r:(r['inner_accuracy'],-r['inner_nll']))
    gate=best['inner_accuracy']>=baseline_result['inner_accuracy']+p['promotion_gain']
    frozen_files=[f for folder in ('data','fits') for f in (ROOT/folder).rglob('*') if f.is_file()]
    json_write(ROOT/'results.json',dict(completed_utc=now(),results=results,promotion_gate_passed=gate,best_legitimate=best['variant'],
        best_gain=best['inner_accuracy']-baseline_result['inner_accuracy'],protocol_sha256=digest(ROOT/'protocol.json'),
        artifacts={str(f):digest(f) for f in frozen_files},outer_validation_scored=False,new_validation_loaded=False,assessment_loaded=False,source_calls=0))
    print({'promotion_gate_passed':gate,'best_gain':best['inner_accuracy']-baseline_result['inner_accuracy']},flush=True)


if __name__=='__main__':
    parser=ArgumentParser(); parser.add_argument('action',choices=['prepare','encode','screen']); args=parser.parse_args()
    {'prepare':prepare,'encode':encode,'screen':screen}[args.action]()
