"""Training-only test of a shared potential difference for belief updates.

z_next = alpha * logit(p_current) + f(current inbox) - f(previous inbox).
No future outcome enters the model. The previous inbox is empty at t=0.
"""
from argparse import ArgumentParser
import copy
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write,seed_all
from scripts import modal_neural_study as core

ROOT=Path('runs/modal_change_rule_320_v1')
read,now=core.read,core.now


def config():
    cfg=copy.deepcopy(core.config()); cfg['output']=str(ROOT); cfg['conditions']=['mode']
    cfg['models']={f'{text}_{gain}':dict(text='conclusion' if text=='choice' else 'arguments',learn_gain=gain=='learned')
        for text in ('choice','arguments') for gain in ('fixed','learned')}
    cfg['training'].update(max_steps=32768,patience=6)
    return cfg


class Data(core.dist.Data):
    def __init__(self,stage):
        super().__init__(core.dist.config(),stage)
        self.source=self.y.copy(); self.y=(self.source>.5).astype(np.int64); self.values['target']=torch.as_tensor(self.y,dtype=torch.float32)
        pairs=self.common['pairs']; e,t=pairs.T
        previous=np.arange(len(pairs))-1; previous[t==0]=np.arange(len(pairs))[t==0]
        assert np.array_equal(pairs[previous][t>0],np.stack([e[t>0],t[t>0]-1],-1))
        for key in ('s','context','J','text_arguments','text_conclusion','neighbors','signs','positions'):
            value=self.values[key][previous].clone()
            # No incoming message exists in the pre-discussion opinion prompt.
            if key in ('J','signs','positions','text_arguments','text_conclusion'): value[t==0]=0
            self.values['previous_'+key]=value
        # Current source probability is observed at t, never t+1.
        self.values['current_logit']=torch.as_tensor(np.log(self.current)-np.log1p(-self.current),dtype=torch.float32)
        self.condition='mode'

    def batch(self,ids,condition=None):
        assert condition in (None,'mode')
        return {k:v[ids] for k,v in self.values.items()}


class ChangeRule(nn.Module):
    def __init__(self,spec,dropout):
        super().__init__(); self.spec=spec
        self.potential=core.dist.ordered.OrderedPairAttention(d_model=8,heads=4,d_k=1,d_v=2,ffn_dim=16,text_dim=8,dropout=dropout)
        # A common output bias cancels exactly between the two potentials.
        old=self.potential.classifier[-1]; self.potential.classifier[-1]=nn.Linear(old.in_features,old.out_features,bias=False)
        nn.init.zeros_(self.potential.classifier[-1].weight)
        if spec['learn_gain']: self.gain=nn.Parameter(torch.tensor(1.))
        else: self.register_buffer('gain',torch.tensor(1.))

    def score(self,batch):
        text='text_'+self.spec['text']; arguments={}
        for name in ('s','context','J',text,'neighbors','signs','positions'):
            arguments[name]=torch.cat([batch[name],batch['previous_'+name]],0)
        logits=self.potential(arguments['s'],arguments['context'],arguments['J'],arguments[text],
            neighbors=arguments['neighbors'],signs=arguments['signs'],positions=arguments['positions'])
        values=logits[...,1]-logits[...,0]; new,old=values.chunk(2,dim=0)
        return self.gain*batch['current_logit']+(new-old)


def make_model(cfg,name,seed=None):
    if seed is not None: seed_all(seed)
    return ChangeRule(cfg['models'][name],cfg['training']['dropout'])


def score_model(model,batch,opinion=False):
    assert not opinion
    return model.score(batch)


def prepare():
    if (ROOT/'protocol.json').exists(): return read(ROOT/'protocol.json')
    ROOT.mkdir(parents=True,exist_ok=True); parent=read(Path(core.config()['output'])/'protocol.json'); cfg=config()
    history_selection=read('runs/modal_accuracy_investigation_320_v1/causal_history_v1/selection.json')
    selected=next(r for r in history_selection['choices'] if r['width']=='history_choice_observed')
    reference=read(Path(selected['tuning_directory'])/'manifest.json'); reference_accuracy=1-reference['validation']['error']
    files=[Path(__file__),Path('tests/test_modal_change_rule.py'),Path(core.trainer.__file__),Path(core.dist.ordered.__file__)]
    p=dict(created_utc=now(),config=cfg,fold=parent['fold'],inner=parent['inner'],parameters={n:sum(v.numel() for v in make_model(cfg,n).parameters()) for n in cfg['models']},
        hypothesis='A stable question/persona contribution can cancel when learning the change in inbox response, with the observed current probability as an offset.',
        formula='logit(p_next)=alpha*logit(p_current)+f(inbox_t_to_t+1)-f(inbox_t-1_to_t). Shared compact ordered attention; alpha fixed1 or learned.',
        feature_policy='Current/previous sender opinions, actual delivered messages or matched choice tokens, relation signs, exact inbox positions and training-only context. '
            'At t=0 the previous inbox has zero edges/text. No source probability enters either potential; observed current probability enters only the logit offset. '
            'No target-round probabilities, answer keys, question IDs or solutions become inputs.',
        selection_policy='Internal256/64 whole-question tuning only, two original learning rates per condition. All8 rounds and32 agents included. '
            'Accuracy then mode NLL checkpoint selection. No outer80 unless a candidate passes the gate; fixed1 conserves probability under identical inbox inputs, learned gain is a relaxation.',
        promotion_reference_accuracy=reference_accuracy,promotion_gain=.005,promotion_threshold=reference_accuracy+.005,
        promotion_policy='At least0.5pp above the existing best history-choice internal reference before any outer refit. No95% claim from internal selection.',
        references={str(Path(selected['tuning_directory'])/'manifest.json'):digest(Path(selected['tuning_directory'])/'manifest.json'),
            str(Path(core.dist.config()['output'])/'protocol.json'):digest(Path(core.dist.config()['output'])/'protocol.json')},
        code_files={str(f):digest(f) for f in files},core_sha256=core.code_digest(),source_calls=0,outer_validation_scored=False,assessment_loaded=False)
    json_write(ROOT/'protocol.json',p); json_write(ROOT/'encoding.json',dict(parent_probability_encoding_sha256=digest(Path(core.dist.config()['output'])/'encoding.json')))
    return p


def validate():
    p=prepare(); assert p['config']==config() and p['core_sha256']==core.code_digest()
    for group in ('references','code_files'):
        for path,sha in p[group].items(): assert digest(path)==sha,path
    return p


def worker(name):
    p=validate(); cfg=config(); torch.set_num_threads(2); data=Data('inner'); runtime=copy.deepcopy(p)
    runtime['fold']=dict(train_questions=p['inner']['fit'],validation_questions=p['inner']['validation'])
    trainer=core.trainer; shared=core.dist.trainer
    trainer.make_model=make_model; trainer.score_model=score_model; shared.score_model=score_model
    original=trainer.json_write
    def write(path,value):
        if Path(path).name=='progress.json': path=ROOT/'progress'/f'{name}.json'
        return original(path,value)
    trainer.json_write=write
    for candidate in core.dist.base.candidates(cfg):
        item=dict(phase='tune',width=name,condition='mode',candidate=candidate['id'],seed=42)
        _,result=trainer.train_one(cfg,runtime,data,item,time.time()+7200); assert result['status']=='complete'


def run():
    p=validate(); pending=list(config()['models']); active={}; done={}
    while pending or active:
        while pending and len(active)<4:
            name=pending.pop(0); stream=(ROOT/(name+'.log')).open('a',buffering=1)
            active[name]=(subprocess.Popen([sys.executable,__file__,'worker','--model',name],stdout=stream,stderr=subprocess.STDOUT),stream)
        for name,(process,stream) in list(active.items()):
            if process.poll() is not None: done[name]=process.returncode; stream.close(); del active[name]
        json_write(ROOT/'progress.json',dict(updated_utc=now(),active={n:p.pid for n,(p,s) in active.items()},pending=pending,completed=done))
        if any(done.values()):
            for process,stream in active.values(): process.terminate(); process.wait(); stream.close()
            raise RuntimeError(str(done))
        if active: time.sleep(2)
    records=[read(f) for f in (ROOT/'tune').glob('*/manifest.json')]; assert len(records)==8
    selected={name:min([r for r in records if r['width']==name],key=lambda r:(r['validation']['error'],r['validation']['nll'],r['selected_step'])) for name in config()['models']}
    best=min(selected.values(),key=lambda r:(r['validation']['error'],r['validation']['nll'])); accuracy=1-best['validation']['error']
    json_write(ROOT/'results.json',dict(completed_utc=now(),selected=selected,best=best['width'],best_internal_accuracy=accuracy,
        promotion_gate_passed=accuracy>=p['promotion_threshold'],promotion_threshold=p['promotion_threshold'],protocol_sha256=digest(ROOT/'protocol.json'),
        artifacts={str(f):digest(f) for f in (ROOT/'tune').rglob('*') if f.is_file()},outer_validation_scored=False,assessment_loaded=False,source_calls=0))
    print({'best_internal_accuracy':accuracy,'promotion_gate_passed':accuracy>=p['promotion_threshold']},flush=True)


if __name__=='__main__':
    parser=ArgumentParser(); parser.add_argument('action',choices=['prepare','run','worker']); parser.add_argument('--model'); args=parser.parse_args()
    if args.action=='prepare': print(prepare())
    elif args.action=='run': run()
    else: worker(args.model)
