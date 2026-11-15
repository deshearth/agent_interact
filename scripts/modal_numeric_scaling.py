"""Fixed-recipe 512-question replication of numeric history attention."""
from argparse import ArgumentParser
from pathlib import Path
import math
import subprocess
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_numeric_followup as prior
from scripts import modal_scaling_study as engine

ROOT=Path('runs/modal_numeric_scaling_512_v1')
OLD_SCALING=engine.ROOT
BaseData=engine.Data
read,now=prior.read,prior.now


def protocol():
    path=ROOT/'protocol.json'
    if path.exists():return read(path)
    p=prior.validate();assert read(prior.ROOT/'completion_audit.json')['passed']
    old=engine.validate();cfg=prior.config();selection=read(prior.ROOT/'neural/selection.json');models={}
    for chosen in selection['choices']:
        name=chosen['width'];refs={str(seed):str(prior.ROOT/'neural/refit'/f"{name}-mode-{chosen['candidate']}-s{seed}") for seed in cfg['training']['refit_seeds']}
        manifest=read(Path(refs['142'])/'manifest.json')
        models[name]=dict(spec=cfg['models'][name],parameters=manifest['parameters'],training=cfg['training'],optimizer=manifest['candidate_settings'],
            refits=refs,steps=dict(same_updates=chosen['refit_steps'],same_passes=math.ceil(chosen['refit_steps']*512/320)))
    references=[prior.ROOT/'protocol.json',prior.ROOT/'models_frozen.json',prior.ROOT/'completion_audit.json',prior.ROOT/'evaluation/results.json',
        prior.ROOT/'features.json',OLD_SCALING/'features_complete.json',OLD_SCALING/'protocol.json']
    for value in models.values():
        references.extend(Path(folder)/file for folder in value['refits'].values() for file in ['best.pt','manifest.json'])
    files=[Path(__file__),Path(engine.__file__),Path(prior.__file__),Path('tests/test_modal_numeric_scaling.py')]
    p=dict(created_utc=now(),models=models,seeds=cfg['training']['refit_seeds'],train320=old['train320'],train512=old['train512'],validation=old['validation'],
        references={str(f):digest(f) for f in references},code_files={str(f):digest(f) for f in files},
        hypothesis='Broader questions also improve the newly positive numeric-argument transformer.',
        timing='Exploratory follow-up motivated after seeing the audited320 numeric result. All512 recipes are frozen before fitting; no new tuning or new validation is used.',
        policy='Replay original320 models. Each512 fit uses the same initialization, rate, regularization, batch size and old320 feature transforms. '
            'Save equal-update and ceil(oldsteps*512/320) equal-data-pass checkpoints. Report both without choosing by outer80. '
            'Compare numeric fields against the same-size zero-field control; both use real argumentPCA8 and observed history.',
        statistics='5000 paired whole-question bootstrap draws, seed92931. Mean individual-seed accuracy, no ensemble. Intervals descriptive after validation reuse.',
        source_calls=0,new_validation_loaded=False,assessment_loaded=False)
    ROOT.mkdir(parents=True,exist_ok=True);json_write(path,p)
    references=[prior.ROOT/'sender_numeric.npy',prior.ROOT/'features.json',OLD_SCALING/'features_complete.json']
    json_write(ROOT/'features_complete.json',dict(references={str(f):digest(f) for f in references},old_features_preserved=True))
    return p


def validate():
    p=protocol()
    for group in ('references','code_files'):
        for path,sha in p[group].items():assert digest(path)==sha,path
    for path,sha in read(ROOT/'features_complete.json')['references'].items():assert digest(path)==sha,path
    for path,sha in read(OLD_SCALING/'features_complete.json')['artifacts'].items():assert digest(path)==sha,path
    return p


class Data(BaseData):
    def __init__(self,name):
        super().__init__('history_arguments2306')
        e,t=self.common['pairs'].T;raw=torch.as_tensor(np.load(prior.ROOT/'sender_numeric.npy')[e,t]);semantic=self.values['text_arguments8']
        self.values['text_numeric16']=torch.cat([semantic,raw/4],-1)/np.sqrt(2)
        self.values['text_blank16']=torch.cat([semantic,torch.zeros_like(raw)],-1)/np.sqrt(2)
        self.name=name


def make_model(name,seed):return prior.make_model(prior.config(),name,seed)


def configure_engine():
    engine.ROOT=ROOT;engine.validate=validate;engine.protocol=protocol;engine.Data=Data;engine.make_model=make_model


def run():
    p=validate()
    if (ROOT/'models_frozen.json').exists():return
    pending=[(n,s) for n in p['models'] for s in p['seeds']];active={};finished={}
    while pending or active:
        while pending and len(active)<4:
            name,seed=pending.pop(0);key=f'{name}_s{seed}';stream=(ROOT/(key+'.log')).open('a',buffering=1)
            active[key]=(subprocess.Popen([sys.executable,__file__,'worker','--model',name,'--seed',str(seed)],stdout=stream,stderr=subprocess.STDOUT),stream)
        for key,(proc,stream) in list(active.items()):
            if proc.poll() is not None:finished[key]=proc.returncode;stream.close();del active[key]
        json_write(ROOT/'progress.json',dict(phase='fitting',updated_utc=now(),active={k:pr.pid for k,(pr,st) in active.items()},pending=pending,finished=finished))
        if any(finished.values()):
            for proc,stream in active.values():proc.terminate();proc.wait();stream.close()
            raise RuntimeError(str(finished))
        if active:time.sleep(2)
    files=[ROOT/'protocol.json',ROOT/'features_complete.json']+[f for f in (ROOT/'fits').rglob('*') if f.is_file()]
    json_write(ROOT/'models_frozen.json',dict(frozen_utc=now(),artifacts={str(f):digest(f) for f in files},new_fits=6,new_endpoints=12,validation_used=False))
    json_write(ROOT/'progress.json',dict(phase='all_frozen_ready_for_outer',updated_utc=now()))


def summaries(rows):
    keys=sorted({(r['model'],r['cohort'],r['budget']) for r in rows});result={}
    for name,cohort,budget in keys:
        key=f'{name}_{cohort}_{budget}';result[key]={}
        for split in ('train','validation'):
            group=[r for r in rows if (r['model'],r['cohort'],r['budget'],r['split'])==(name,cohort,budget,split)]
            result[key][split]={m:float(np.mean([r[m] for r in group])) for m in ('error','nll','brier_two_class')}
            result[key][split]['accuracy']=1-result[key][split]['error']
    def perq(key,metric):
        name,cohort,budget=key;group=[r for r in rows if (r['model'],r['cohort'],r['budget'],r['split'])==(name,cohort,budget,'validation')]
        return np.array([np.mean([r[metric] for r in group if r['qid']==q]) for q in sorted({r['qid'] for r in group})])
    comparisons=[((n,512,b),(n,320,'original')) for n in ['numeric','blank'] for b in ['same_updates','same_passes']]
    comparisons += [(('numeric',c,b),('blank',c,b)) for c,b in [(320,'original'),(512,'same_updates'),(512,'same_passes')]]
    draws=np.random.default_rng(92931).integers(80,size=(5000,80));contrasts=[]
    for new,old in comparisons:
        for metric in ('error','nll','brier_two_class'):
            delta=perq(old,metric)-perq(new,metric)
            contrasts.append(dict(new=list(new),reference=list(old),metric=metric,reduction=float(delta.mean()),ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
    return result,contrasts


def evaluate():
    p=validate();frozen=read(ROOT/'models_frozen.json')
    for path,sha in frozen['artifacts'].items():assert digest(path)==sha,path
    out=ROOT/'evaluation';out.mkdir(exist_ok=True)
    if (out/'results.json').exists():return
    torch.set_num_threads(2);rows=[];files=[]
    for name in p['models']:
        data=Data(name)
        for cohort,budget in [(320,'original'),(512,'same_updates'),(512,'same_passes')]:
            for seed in p['seeds']:
                checkpoint=Path(p['models'][name]['refits'][str(seed)])/'best.pt' if cohort==320 else ROOT/'fits'/name/f's{seed}'/(budget+'.pt')
                model=make_model(name,seed);model.load_state_dict(torch.load(checkpoint,weights_only=False)['model'])
                for split,questions in [('train',p['train'+str(cohort)]),('validation',p['validation'])]:
                    ids=data.ids(questions);lp=engine.predict(model,data,ids);met,per=prior.core.dist.base.metrics(lp,data.y[ids],data.qids[ids])
                    rows.extend(dict(model=name,cohort=cohort,budget=budget,seed=seed,split=split,**r) for r in per)
                    path=out/'predictions'/f'{name}_{cohort}_{budget}_s{seed}_{split}.npz';path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(path,logp=lp,target=data.y[ids],probability=data.source[ids],qids=data.qids[ids],pairs=data.common['pairs'][ids])
                    files.append(dict(model=name,cohort=cohort,budget=budget,seed=seed,split=split,metrics=met,path=str(path),sha256=digest(path),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint)))
    summary,contrasts=summaries(rows);json_write(out/'by_question.json',rows)
    json_write(out/'results.json',dict(completed_utc=now(),protocol_sha256=digest(ROOT/'protocol.json'),models_frozen_sha256=digest(ROOT/'models_frozen.json'),
        summary=summary,contrasts=contrasts,prediction_files=files,source_calls=0,new_validation_loaded=False,assessment_loaded=False));print(summary,flush=True)


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('action',choices=['prepare','run','worker','evaluate']);parser.add_argument('--model');parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.action=='prepare':protocol()
    elif args.action=='run':run()
    elif args.action=='evaluate':evaluate()
    else:configure_engine();engine.worker(args.model,args.seed)
