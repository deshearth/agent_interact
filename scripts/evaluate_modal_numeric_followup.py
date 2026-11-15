"""Score only after all numeric follow-up models have been frozen."""
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_numeric_followup as study

ROOT=study.ROOT


def predict(model,data,ids,batch_size=128):
    model.eval();values=[]
    with torch.inference_mode():
        for start in range(0,len(ids),batch_size):
            z=study.core.score_model(model,data.batch(ids[start:start+batch_size]))
            values.append(torch.stack([-F.softplus(z),-F.softplus(-z)],-1).numpy())
    return np.concatenate(values)


def analyze(rows):
    keys=sorted({r['name'] for r in rows});summary={}
    for name in keys:
        summary[name]={}
        for split in ('train','validation'):
            group=[r for r in rows if r['name']==name and r['split']==split]
            summary[name][split]={m:float(np.mean([r[m] for r in group])) for m in ('error','nll','brier_two_class')}
            summary[name][split]['accuracy']=1-summary[name][split]['error']
    def per(name,metric):
        group=[r for r in rows if r['name']==name and r['split']=='validation']
        return np.array([np.mean([r[metric] for r in group if r['qid']==q]) for q in sorted({r['qid'] for r in group})])
    contrasts=[('numeric320','matched_duration320'),('numeric512','matched_duration512'),
        ('numeric320','reference_tree320'),('numeric512','reference_tree512'),('numeric512','numeric320'),
        ('neural_numeric','neural_blank'),('neural_numeric','reference_history_arguments320'),('neural_numeric','reference_history_choice320')]
    draws=np.random.default_rng(92891).integers(80,size=(5000,80));comparisons=[]
    for new,old in contrasts:
        for metric in ('error','nll','brier_two_class'):
            delta=per(old,metric)-per(new,metric)
            comparisons.append(dict(new=new,reference=old,metric=metric,reduction=float(delta.mean()),ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
    return summary,comparisons


def main():
    p=study.validate();frozen=study.read(ROOT/'models_frozen.json')
    for path,sha in frozen['artifacts'].items():assert digest(path)==sha,path
    out=ROOT/'evaluation';out.mkdir(exist_ok=True)
    if (out/'results.json').exists():return
    torch.set_num_threads(2);rows=[];files=[]
    def record(name,seed,split,lp,target,probability,qids,pairs,checkpoint,reference=False):
        met,per=study.core.dist.base.metrics(lp,target,qids)
        rows.extend(dict(name=name,seed=seed,split=split,**r) for r in per)
        path=out/'predictions'/f'{name}_s{seed}_{split}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,logp=lp,target=target,probability=probability,qids=qids,pairs=pairs)
        files.append(dict(name=name,seed=seed,split=split,metrics=met,path=str(path),sha256=digest(path),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),reference=reference))
    data=study.Data('full');cfg=study.config();selected=study.read(ROOT/'neural/selection.json')
    for choice in selected['choices']:
        name=choice['width']
        for seed in cfg['training']['refit_seeds']:
            folder=ROOT/'neural/refit'/f"{name}-mode-{choice['candidate']}-s{seed}";checkpoint=folder/'best.pt'
            model=study.make_model(cfg,name,seed);model.load_state_dict(torch.load(checkpoint,weights_only=False)['model'])
            for split,questions in [('train',p['fold']['train_questions']),('validation',p['fold']['validation_questions'])]:
                ids=data.ids(questions);lp=predict(model,data,ids)
                if split=='train':np.testing.assert_array_equal(lp,np.load(folder/'train_logp.npy'))
                record('neural_'+name,seed,split,lp,data.y[ids],data.source[ids],data.qids[ids],data.common['pairs'][ids],checkpoint)
    large=study.scaling.protocol();common=np.load(study.scaling.DATA/'common.npz');e,t=common['pairs'].T
    probability=np.load(study.scaling.DATA/'single.npy')[e,t+1];target=(probability>.5).astype(int);qids=common['episode_qids'][e]
    base=np.load(study.scaling.DATA/'tree_semantic8.npy',mmap_mode='r');extra=np.load(ROOT/'ordered_numeric.npy')
    for name in p['tree_conditions']:
        count=320 if name.endswith('320') else 512;checkpoint=ROOT/'trees'/name/'model.pkl'
        with checkpoint.open('rb') as stream:model=pickle.load(stream)
        for split,questions in [('train',large['train'+str(count)]),('validation',large['validation'])]:
            ids=np.flatnonzero(np.isin(qids,questions));r=(ids[:,None]*32+np.arange(32)).ravel()
            x=np.concatenate([base[r],extra[r]],-1) if name.startswith('numeric') else base[r]
            with threadpool_limits(limits=2):pp=np.clip(model.predict_proba(x)[:,1],1e-10,1-1e-10).reshape(-1,32)
            lp=np.stack([np.log1p(-pp),np.log(pp)],-1)
            record(name,0,split,lp,target[ids],probability[ids],qids[ids],common['pairs'][ids],checkpoint)
    reference=study.read(study.scaling.ROOT/'evaluation/results.json')
    for f in reference['prediction_files']:
        keep=(f['model']=='tree_semantic8') or (f['cohort']==320 and f['model'] in ('history_arguments2306','history_choice2306'))
        if not keep:continue
        assert digest(f['path'])==f['sha256'];d=np.load(f['path'])
        name='reference_tree'+str(f['cohort']) if f['model']=='tree_semantic8' else 'reference_'+f['model'].replace('2306','320')
        record(name,f['seed'],f['split'],d['logp'],d['target'],d['probability'],d['qids'],d['pairs'],Path(f['checkpoint']),True)
    summary,contrasts=analyze(rows);json_write(out/'by_question.json',rows)
    json_write(out/'results.json',dict(completed_utc=study.now(),protocol_sha256=digest(ROOT/'protocol.json'),
        models_frozen_sha256=digest(ROOT/'models_frozen.json'),summary=summary,contrasts=contrasts,prediction_files=files,
        evaluation_code_sha256=digest(__file__),source_calls=0,new_validation_loaded=False,assessment_loaded=False))
    print(summary,flush=True)


if __name__=='__main__':main()
