"""Frozen full-data follow-up to the positive numeric-option internal screen."""
from argparse import ArgumentParser
from pathlib import Path
import copy
import pickle
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest,digest,json_write
from scripts import modal_option_values as pilot
from scripts import modal_causal_history as history
from scripts import modal_scaling_study as scaling

core=pilot.core
ROOT=Path('runs/modal_numeric_followup_320_v1')
read,now=core.read,core.now


def config():
    cfg=copy.deepcopy(history.config());cfg['output']=str(ROOT/'neural')
    spec=cfg['models']['history_arguments_observed']
    cfg['models']={name:dict(spec,text=name,text_dim=16) for name in ('numeric','blank')}
    return cfg


def prepare_features():
    if (ROOT/'features.json').exists():return
    pilot.validate();assert read(pilot.ROOT/'completion_audit.json')['passed']
    assert read(pilot.ROOT/'results.json')['promotion_gate_passed']
    scaling.validate();ROOT.mkdir(parents=True,exist_ok=True)
    common=np.load(scaling.DATA/'common.npz');index=read(scaling.DATA/'index.json')
    assert len(index)==592
    raw=np.zeros((592,8,32,8),dtype=np.float32);sources={}
    for e,item in enumerate(index):
        assert digest(item['path'])==item['sha256']
        episode=read(item['path']);assert episode['meta']['qid']==common['episode_qids'][e]
        sources[item['path']]=item['sha256']
        for t in range(8):
            for sender,text in enumerate(pilot.messages.snapshot_messages(episode,t)):
                if text:raw[e,t,sender]=pilot.numeric_features(text,episode['choices'])
    np.save(ROOT/'sender_numeric.npy',raw)
    e,t=common['pairs'].T;order=np.load(scaling.DATA/'positions.npz')['order']
    ordered=core.screen.slots(raw[e,t],order).reshape(-1,144)
    np.save(ROOT/'ordered_numeric.npy',ordered)
    internal=np.load(pilot.ROOT/'data.npz')
    np.testing.assert_array_equal(raw[e[internal['pair_indices']],t[internal['pair_indices']]],internal['sender_values'])
    np.testing.assert_array_equal(ordered[internal['rows']],internal['ordered'])
    json_write(ROOT/'features.json',dict(created_utc=now(),sources=sources,old_internal_features_identical=True,
        artifacts={str(ROOT/n):digest(ROOT/n) for n in ['sender_numeric.npy','ordered_numeric.npy']},
        common_sha256=digest(scaling.DATA/'common.npz'),positions_sha256=digest(scaling.DATA/'positions.npz'),
        source_calls=0,new_validation_loaded=False,assessment_loaded=False))


class Data(history.Data):
    def __init__(self,stage):
        super().__init__(stage)
        e,t=self.common['pairs'].T
        raw=torch.as_tensor(np.load(ROOT/'sender_numeric.npy')[:400][e,t])
        semantic=self.values['text_arguments8']
        self.values['text_numeric16']=torch.cat([semantic,raw/4],-1)/np.sqrt(2)
        self.values['text_blank16']=torch.cat([semantic,torch.zeros_like(raw)],-1)/np.sqrt(2)


def make_model(cfg,name,seed=None):return history.make_model(cfg,name,seed)


def prepare():
    prepare_features();cfg=config();path=ROOT/'protocol.json'
    if path.exists():return read(path)
    parent=read(Path(core.config()['output'])/'protocol.json');large=scaling.protocol()
    chosen=read(pilot.ROOT/'ordered/complete.json');assert chosen['internal_accuracy']>=read(pilot.ROOT/'protocol.json')['promotion_reference']+.005
    files=[Path(__file__),Path('tests/test_modal_numeric_followup.py'),Path(pilot.__file__),Path(history.__file__),Path(core.__file__),Path(core.trainer.__file__)]
    refs=[pilot.ROOT/'protocol.json',pilot.ROOT/'results.json',pilot.ROOT/'completion_audit.json',ROOT/'features.json',scaling.ROOT/'protocol.json',scaling.ROOT/'features_complete.json',scaling.ROOT/'evaluation/results.json']
    p=dict(created_utc=now(),hypothesis='Exact rational matches link argument content to supplied options and can improve transfer beyond generic embeddings.',
        fold=parent['fold'],inner=parent['inner'],train512=large['train512'],tree_iterations=chosen['selected_step'],tree_settings=core.screen.SETTINGS,
        tree_conditions=['numeric320','numeric512','matched_duration320','matched_duration512'],
        tree_policy='Append only the promoted144 ordered numeric fields. Fit320 and512 at the internal-selected155 iterations. '
            'Also refit the original state tree at the identical155 iterations to separate numeric features from duration. Old105-step selected baselines remain references.',
        neural_policy='Two history-attention models have identical2370 parameters and initial weights. Both retain original argumentPCA8; '
            'the augmented arm adds the eight exact numeric fields, the control adds zeros. Both channels scaled by1/sqrt2 to preserve old semantic scaling; numeric fields additionally divide by4. '
            'Two learning rates on internal256/64, then three fresh seeds on all320. Same old history training recipe; every final checkpoint freezes before outer80 scoring.',
        target='All-case source-mode A iff next single-response p(A)>0.5 on unchanged stochastic histories. No current or future answer keys.',
        core_sha256=code_digest(),code_files={str(f):digest(f) for f in files},references={str(f):digest(f) for f in refs},
        parameters={n:sum(v.numel() for v in make_model(cfg,n).parameters()) for n in cfg['models']},
        statistics='5000 whole-question paired bootstrap draws. Neural results average individual seeds, not ensembles. Reused80 validation remains exploratory.',
        source_calls=0,new_validation_loaded=False,assessment_loaded=False)
    json_write(path,p)
    neural=dict(p,config=cfg)
    folder=ROOT/'neural';folder.mkdir(exist_ok=True);json_write(folder/'protocol.json',neural)
    json_write(folder/'encoding.json',dict(features_sha256=digest(ROOT/'features.json'),frozen_encoder_parameters=22713216))
    json_write(folder/'prepared.json',dict(protocol_sha256=digest(folder/'protocol.json'),encoding_sha256=digest(folder/'encoding.json')))
    return p


def validate(cfg=None):
    p=read(ROOT/'protocol.json')
    for group in ('code_files','references'):
        for path,sha in p[group].items():assert digest(path)==sha,path
    f=read(ROOT/'features.json')
    for path,sha in f['artifacts'].items():assert digest(path)==sha,path
    assert p['core_sha256']==code_digest()
    if cfg is not None:return core.validate(cfg)
    return p


def trees():
    p=validate();large=scaling.protocol();common=np.load(scaling.DATA/'common.npz');e,t=common['pairs'].T
    target=(np.load(scaling.DATA/'single.npy')[e,t+1]>.5).ravel();qids=np.repeat(common['episode_qids'][e],32)
    base=np.load(scaling.DATA/'tree_semantic8.npy',mmap_mode='r');extra=np.load(ROOT/'ordered_numeric.npy')
    for name in p['tree_conditions']:
        folder=ROOT/'trees'/name;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():continue
        count=320 if name.endswith('320') else 512;fit=np.isin(qids,large['train'+str(count)])
        x=np.concatenate([base[fit],extra[fit]],-1) if name.startswith('numeric') else base[fit]
        settings=dict(p['tree_settings'],max_iter=p['tree_iterations'],early_stopping=False)
        start=time.monotonic();model=HistGradientBoostingClassifier(**settings)
        with threadpool_limits(limits=2):model.fit(x,target[fit])
        with (folder/'model.pkl').open('wb') as stream:pickle.dump(model,stream)
        json_write(folder/'complete.json',dict(model=name,completed_utc=now(),seconds=time.monotonic()-start,
            model_sha256=digest(folder/'model.pkl'),tree_nodes=sum(len(z[0].nodes) for z in model._predictors),features=x.shape[1],training_questions=count))


def phase(label):
    active={};pending=list(config()['models']);done={}
    while pending or active:
        while pending and len(active)<2:
            name=pending.pop(0);stream=(ROOT/f'{label}_{name}.log').open('a',buffering=1)
            process=subprocess.Popen([sys.executable,__file__,'worker','--model',name,'--phase',label],stdout=stream,stderr=subprocess.STDOUT)
            active[name]=(process,stream)
        for name,(proc,stream) in list(active.items()):
            if proc.poll() is not None:done[name]=proc.returncode;stream.close();del active[name]
        json_write(ROOT/'progress.json',dict(phase=label,updated_utc=now(),active={n:pr.pid for n,(pr,st) in active.items()},pending=pending,completed=done))
        if any(done.values()):
            for proc,stream in active.values():proc.terminate();proc.wait();stream.close()
            raise RuntimeError(str(done))
        if active:time.sleep(2)


def run():
    prepare();validate()
    if (ROOT/'models_frozen.json').exists():return
    trees();phase('tune');folder=ROOT/'neural';records=[(f.parent,read(f)) for f in (folder/'tune').glob('*/manifest.json')]
    assert len(records)==4 and all(r['status']=='complete' for f,r in records)
    choices=[]
    for name in config()['models']:
        f,r=min([(f,r) for f,r in records if r['width']==name],key=lambda z:(z[1]['validation']['error'],z[1]['validation']['nll'],z[1]['selected_step'],z[1]['candidate']))
        choices.append(dict(width=name,condition='mode',candidate=r['candidate'],refit_steps=r['selected_step'],tuning_directory=str(f)))
    selection=dict(protocol_sha256=digest(folder/'protocol.json'),choices=choices)
    if (folder/'selection.json').exists():assert read(folder/'selection.json')==selection
    else:json_write(folder/'selection.json',selection)
    phase('refit');records=[read(f) for f in (folder/'refit').glob('*/manifest.json')]
    assert len(records)==6 and all(r['status']=='complete' for r in records)
    files=[ROOT/'protocol.json',ROOT/'features.json']+[f for directory in [ROOT/'trees',folder] for f in directory.rglob('*') if f.is_file()]
    json_write(ROOT/'models_frozen.json',dict(frozen_utc=now(),artifacts={str(f):digest(f) for f in files}))
    json_write(ROOT/'progress.json',dict(phase='all_frozen_ready_for_outer',updated_utc=now()))


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('action',choices=['features','prepare','run','worker']);parser.add_argument('--model');parser.add_argument('--phase',choices=['tune','refit']);args=parser.parse_args()
    if args.action=='features':prepare_features()
    elif args.action=='prepare':prepare()
    elif args.action=='run':run()
    else:
        core.Data,core.make_model=Data,make_model
        core.worker(config(),args.model,'mode',args.phase)
