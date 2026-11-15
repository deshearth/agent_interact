"""Full-context frozen encoder probe on the existing training-only prompt pilot."""
from argparse import ArgumentParser
from pathlib import Path
import os
import pickle
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_inbox_screen as source
from scripts import benchmark_modal_long_encoder as bench

ROOT=bench.ROOT
read,now=source.read,source.now
BUCKETS=[512,1024,1536,2048,3072,4096,6144,8192]


def prepare():
    if (ROOT/'protocol.json').exists(): return read(ROOT/'protocol.json')
    parent=source.validate(); info=read(ROOT/'encoder_manifest.json'); benchmark=read(ROOT/'benchmark.json')
    assert benchmark['cpu_max_vector_difference']<1e-4 and benchmark['device']=='mps'
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
    texts=read(source.ROOT/'data/texts.json'); lengths=np.asarray([len(x) for x in tokenizer(texts,truncation=False)['input_ids']],dtype=np.int32)
    assert lengths.max()<=8192 and len(texts)==20480
    buckets=np.asarray([next(b for b in BUCKETS if n<=b) for n in lengths])
    order=np.argsort(buckets,kind='stable'); folder=ROOT/'data'; folder.mkdir(exist_ok=True)
    np.savez(folder/'plan.npz',lengths=lengths,buckets=buckets,order=order)
    refs=[source.ROOT/'protocol.json',source.ROOT/'results.json',source.ROOT/'completion_audit.json',source.ROOT/'data/texts.json',source.ROOT/'data/labels.npz',
        source.ROOT/'fits/state/terminal_model.pkl',source.ROOT/'fits/state/complete.json',ROOT/'encoder_manifest.json',ROOT/'benchmark.json']
    p=dict(created_utc=now(),inner=parent['inner'],rounds=parent['rounds'],rows=20480,encoder=info,
        source_protocol_sha256=digest(source.ROOT/'protocol.json'),frozen_encoder_parameters=benchmark['frozen_encoder_parameters'],
        representation='ModernBERT-base last-layer nonspecial-token mean and first-token vector, separately unit normalized then concatenated/normalized. '
            'Full untruncated prompt with native alternating local/global attention. PCA32/128 and standardization fit only internal256 training questions.',
        inference=dict(device='mps',dtype='float32',attention='sdpa',buckets=BUCKETS,batch_size_up_to2048=2,batch_size_longer=1),
        fitting='Two fixed state+prompt PCA32/128 tree recipes; same old booster settings and accuracy/NLL checkpoint selection. '
            'Also test logistic corrections to the frozen state-tree logit, with PCA32/128/full1536 and L2=0.001,0.01,0.1,1. '
            'A state-logit calibration-only control separates calibration from text information. All conditions remain internal256/64, on two fixed rounds.',
        promotion_gain=.005,promotion_policy='A legitimate tree must beat the matched state pilot by0.5pp. A logistic text correction must beat both the uncorrected state and calibration-only control by0.5pp. '
            'Passing only triggers a subsequent all-round study; no internal or filtered result counts toward95% outer accuracy.',
        tree_dimensions=[32,128],linear_dimensions=[32,128,1536],regularization=[.001,.01,.1,1.],settings=source.core.screen.SETTINGS,
        references={str(f):digest(f) for f in refs},code_files={str(f):digest(f) for f in [Path(__file__),Path(bench.__file__)]},
        plan_sha256=digest(folder/'plan.npz'),tokens=int(lengths.sum()),maximum_tokens=int(lengths.max()),source_calls=0,outer_validation_scored=False,new_validation_loaded=False,assessment_loaded=False)
    json_write(ROOT/'protocol.json',p); print({'prepared':True,'tokens':p['tokens'],'max_tokens':p['maximum_tokens']},flush=True); return p


def validate():
    p=prepare()
    for group in ('references','code_files'):
        for path,sha in p[group].items(): assert digest(path)==sha,path
    for path,sha in p['encoder']['files'].items(): assert digest(path)==sha,path
    assert p['plan_sha256']==digest(ROOT/'data/plan.npz')
    return p


def encode():
    p=validate()
    if (ROOT/'encoding.json').exists(): return
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(2); assert torch.backends.mps.is_available()
    tokenizer=AutoTokenizer.from_pretrained(p['encoder']['snapshot'],local_files_only=True)
    model=AutoModel.from_pretrained(p['encoder']['snapshot'],local_files_only=True,attn_implementation='sdpa',dtype=torch.float32).to('mps').eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    plan=np.load(ROOT/'data/plan.npz'); texts=read(source.ROOT/'data/texts.json'); order=plan['order']
    progress_path=ROOT/'encoding_progress.json'
    if progress_path.exists():
        progress=read(progress_path); assert progress['protocol_sha256']==digest(ROOT/'protocol.json')
        vectors=np.lib.format.open_memmap(ROOT/'data/vectors.npy',mode='r+')
    else:
        progress=dict(completed=0,seconds=0.,protocol_sha256=digest(ROOT/'protocol.json'))
        vectors=np.lib.format.open_memmap(ROOT/'data/vectors.npy',mode='w+',dtype=np.float32,shape=(20480,1536))
    start=time.monotonic(); previous=progress['seconds']; position=progress['completed']; last_saved=position
    while position<len(order):
        bucket=int(plan['buckets'][order[position]]); batch_size=2 if bucket<=2048 else 1
        selected=order[position:position+batch_size]; selected=selected[plan['buckets'][selected]==bucket]
        batch=[texts[i] for i in selected]
        if len(batch)<batch_size: batch += [batch[-1]]*(batch_size-len(batch))
        v,lengths=bench.pooled(model,tokenizer,batch,'mps',bucket)
        np.testing.assert_array_equal(lengths[:len(selected)],plan['lengths'][selected]); assert np.isfinite(v).all()
        vectors[selected]=v[:len(selected)]; position+=len(selected)
        if position-last_saved>=128 or position==len(order):
            vectors.flush(); progress.update(completed=position,total=20480,seconds=previous+time.monotonic()-start,updated_utc=now(),bucket=bucket)
            json_write(progress_path,progress); print(progress,flush=True); last_saved=position
    np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6)
    labels=np.load(source.ROOT/'data/labels.npz'); fit=np.isin(labels['qids'],p['inner']['fit'])
    with threadpool_limits(limits=2):
        a=np.asarray(vectors,dtype=float); mean=a[fit].mean(0); centered=a[fit]-mean
        values,axes=np.linalg.eigh(centered.T@centered); axes=axes[:,-128:][:,::-1].T
        reduced=((a-mean)@axes.T).astype(np.float32)
    np.save(ROOT/'data/vectors128.npy',reduced); np.savez(ROOT/'data/pca128.npz',mean=mean,axes=axes,training_indices=np.flatnonzero(fit))
    json_write(ROOT/'encoding.json',dict(completed_utc=now(),seconds=progress['seconds'],texts=20480,tokens=p['tokens'],tokens_dropped=0,independent_windows=False,
        frozen_encoder_parameters=p['frozen_encoder_parameters'],pca128_variance=float(values[-128:].sum()/values.sum()),
        artifacts={str(f):digest(f) for f in (ROOT/'data').iterdir()}))


def data():
    p=validate(); labels=np.load(source.ROOT/'data/labels.npz'); fit=np.isin(labels['qids'],p['inner']['fit']); hold=np.isin(labels['qids'],p['inner']['validation'])
    rows=(labels['pair_indices'][:,None]*32+np.arange(32)).ravel()
    state=np.load(source.core.screen.ROOT/'features/inner/semantic8.npy',mmap_mode='r')[rows]
    return p,labels,fit,hold,state


def linear_fit(x,y,ridge):
    initial=np.zeros(x.shape[1]); initial[0]=1
    penalty=np.full(x.shape[1],ridge,dtype=float); penalty[:2]=.0001
    center=np.zeros(x.shape[1]); center[0]=1
    def objective(w):
        z=x@w; r=w-center
        return float(np.mean(np.logaddexp(0,z)-y*z)+.5*np.sum(penalty*r*r)),x.T@(expit(z)-y)/len(y)+penalty*r
    fit=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(0,5),(None,None)]+[(None,None)]*(x.shape[1]-2),
        options=dict(maxiter=1000,ftol=1e-10,gtol=1e-7))
    assert np.isfinite(fit.x).all() and fit.success,(fit.status,fit.message)
    return fit.x,dict(iterations=fit.nit,objective=float(fit.fun),converged=bool(fit.success))


def score(probability,y):
    pp=np.clip(probability,1e-10,1-1e-10)
    return dict(accuracy=float(((pp>.5)==y).mean()),nll=float((-y*np.log(pp)-(1-y)*np.log1p(-pp)).mean()))


def screen():
    p,labels,fit,hold,state=data(); enc=read(ROOT/'encoding.json')
    for path,sha in enc['artifacts'].items(): assert digest(path)==sha,path
    full=np.load(ROOT/'data/vectors.npy'); reduced=np.load(ROOT/'data/vectors128.npy'); y=labels['target']; results=[]
    for dim in p['tree_dimensions']:
        name='tree_prompt'+str(dim); folder=ROOT/'fits'/name; folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists(): results.append(read(folder/'complete.json')); continue
        x=np.concatenate([state,reduced[:,:dim]],-1); model=HistGradientBoostingClassifier(**p['settings'],early_stopping=True); start=time.monotonic(); best=None; history=[]
        with threadpool_limits(limits=4):
            model.fit(x[fit],y[fit],X_val=x[hold],y_val=y[hold])
            for step,prob in enumerate(model.staged_predict_proba(x[hold]),1):
                met=score(prob[:,1],y[hold]); key=(-met['accuracy'],met['nll'],step); history.append(dict(step=step,**met))
                if best is None or key<best: best=key; chosen=prob[:,1].copy()
        np.save(folder/'internal_probability.npy',chosen);json_write(folder/'history.json',history)
        with (folder/'terminal_model.pkl').open('wb') as stream: pickle.dump(model,stream)
        record=dict(model=name,family='tree',dimension=dim,internal=score(chosen,y[hold]),selected_step=best[2],seconds=time.monotonic()-start)
        json_write(folder/'complete.json',record);results.append(record)
    # The old selected state tree is a common fixed offset in every linear arm.
    with (source.ROOT/'fits/state/terminal_model.pkl').open('rb') as stream: baseline=pickle.load(stream)
    step=read(source.ROOT/'fits/state/complete.json')['selected_step']
    with threadpool_limits(limits=2):
        for i,prob in enumerate(baseline.staged_predict_proba(state),1):
            if i==step: offset=logit(np.clip(prob[:,1],1e-7,1-1e-7)); break
    np.save(ROOT/'data/offset.npy',offset)
    for dim in [0,*p['linear_dimensions']]:
        features=np.empty((len(y),0)) if dim==0 else (full if dim==1536 else reduced[:,:dim]).astype(float)
        mean=features[fit].mean(0); scale=features[fit].std(0); scale=np.maximum(scale,1e-6)
        x=np.column_stack([offset,np.ones(len(y)),(features-mean)/scale])
        for ridge in ([.001] if dim==0 else p['regularization']):
            name=f'linear_prompt{dim}_l2{ridge:g}'; folder=ROOT/'fits'/name; folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists(): results.append(read(folder/'complete.json'));continue
            start=time.monotonic()
            with threadpool_limits(limits=4): weights,optimization=linear_fit(x[fit],y[fit],ridge); pp=expit(x@weights)
            np.savez(folder/'model.npz',weights=weights,mean=mean,scale=scale); np.save(folder/'probability.npy',pp)
            record=dict(model=name,family='linear',dimension=dim,ridge=ridge,parameters=len(weights),optimization=optimization,
                train=score(pp[fit],y[fit]),internal=score(pp[hold],y[hold]),seconds=time.monotonic()-start)
            json_write(folder/'complete.json',record);results.append(record);print(record,flush=True)
    baseline_accuracy=read(source.ROOT/'fits/state/complete.json')['inner_accuracy']
    calibration=next(r['internal']['accuracy'] for r in results if r['family']=='linear' and r['dimension']==0)
    for row in results:
        row['reference_accuracy']=baseline_accuracy if row['family']=='tree' else max(baseline_accuracy,calibration)
        row['promotion_gate_passed']=row['dimension']>0 and row['internal']['accuracy']>=row['reference_accuracy']+p['promotion_gain']
    json_write(ROOT/'results.json',dict(completed_utc=now(),results=results,promotion_gate_passed=any(r['promotion_gate_passed'] for r in results),
        baseline_accuracy=baseline_accuracy,calibration_accuracy=calibration,protocol_sha256=digest(ROOT/'protocol.json'),
        artifacts={str(f):digest(f) for f in (ROOT/'fits').rglob('*') if f.is_file()},source_calls=0,outer_validation_scored=False,new_validation_loaded=False,assessment_loaded=False))


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('action',choices=['prepare','encode','screen']);args=parser.parse_args()
    {'prepare':prepare,'encode':encode,'screen':screen}[args.action]()
