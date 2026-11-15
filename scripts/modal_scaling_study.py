"""Fixed-recipe 320-versus-512 source-mode learning comparison.

The 320 endpoint is replayed from immutable models. Each 512 fit supplies two
predeclared checkpoints: equal updates and approximately equal data passes.
No outer validation is used to select a checkpoint or training recipe.
"""
from argparse import ArgumentParser
from collections import Counter
from contextlib import contextmanager
import copy
import math
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write, seed_all
from mac_agents.models.baselines import author_module
from scripts import modal_scaling_data as inputs
from scripts import modal_neural_study as core
from scripts import modal_causal_history as history
from scripts import modal_question_arguments as paired

ROOT, DATA = inputs.ROOT, inputs.DATA
read, now = core.read, core.now
SEEDS = [142,143,144]
MODEL_SOURCES = {
    'choice978': (core, 'nano_current_choice'),
    'history_choice2306': (history, 'history_choice_observed'),
    'history_arguments2306': (history, 'history_arguments_observed'),
    'joint_arguments26210': (paired, 'paired_full_arguments'),
}


def specifications():
    result = {}
    for name,(module,key) in MODEL_SOURCES.items():
        cfg = module.config(); parent = Path(cfg['output'])
        selected = next(c for c in read(parent/'selection.json')['choices'] if c['width'] == key and c['condition'] == 'mode')
        refits = {str(seed): str(parent/'refit'/f"{key}-mode-{selected['candidate']}-s{seed}") for seed in SEEDS}
        m = read(Path(refits['142'])/'manifest.json')
        result[name] = dict(spec=cfg['models'][key], training=cfg['training'], parameters=m['parameters'],
            optimizer=m['candidate_settings'], steps={'same_updates': selected['refit_steps'],
                'same_passes': math.ceil(selected['refit_steps']*512/320)}, refits=refits,
            parent_selection=str(parent/'selection.json'), parent_selection_sha256=digest(parent/'selection.json'))
    return result


def protocol():
    path = ROOT/'protocol.json'
    if path.exists(): return read(path)
    old = read(Path(core.config()['output'])/'protocol.json'); selected = read(inputs.MANIFEST)['selected_units']
    specs = specifications(); refs = {}
    for value in specs.values():
        for folder in value['refits'].values():
            for name in ('best.pt','manifest.json'): refs[str(Path(folder)/name)] = digest(Path(folder)/name)
    for name,file in [('ising_current_mode','model.json'),('tree_semantic8','model.pkl')]:
        p = Path(core.config()['output'])/'baselines'/name/file; refs[str(p)] = digest(p)
    files = [Path(__file__),Path(inputs.__file__),Path(core.__file__),Path(history.__file__),Path(paired.__file__),
        Path(core.trainer.__file__),Path('scripts/export_modal_scaling.py'),Path('tests/test_modal_scaling.py')]
    p = dict(created_utc=now(), hypothesis='Broader training questions reduce the generalization gap under fixed model representations and recipes.',
        target='All-case next single-response mode A iff p(A)>0.5, on existing stochastic histories, after messages are delivered.',
        train320=old['fold']['train_questions'],train512=old['fold']['train_questions']+[x['qid'] for x in selected],
        validation=old['fold']['validation_questions'],assessment=old['fold']['assessment_questions'],models=specs,seeds=SEEDS,
        selection_manifest=str(inputs.MANIFEST),selection_manifest_sha256=digest(inputs.MANIFEST),references=refs,
        code_files={str(f):digest(f) for f in files},
        selection_policy='First192 new training units in preexisting frozen collection order; all complete. No new validation conversations. '
            'Old80 remains the exploratory common anchor. New bank has revised topic proportions and selection fractions; not an identical-population count-only intervention.',
        representation_policy='Reuse old400 features byte-for-byte. Encode only added training texts using the same frozen MiniLM and complete token pooling. '
            'Question/persona and message PCA are frozen at their old320 fits in both conditions. No refitted PCA, answer keys, solutions, screening scores or next-state observations in inputs.',
        fitting_policy='Replay three original320 seeds. Train512 from the same initial seeds with pinned320 learning rates, regularization and batch size. '
            'Save a checkpoint at the original optimizer-step count and one at ceil(original steps*512/320). No tuning or selection using outer80. '
            'These are two declared budget contrasts, not two chances to select a winner. Neural scores average individual seeds, not an ensemble.',
        baselines='Refit Ising22 with the unchanged author optimizer on hard modal labels. Refit semantic8 tree with its old selected105 iterations and identical settings.',
        statistics='5000 paired whole-question bootstrap draws with seed92813, averaging matched individual-seed scores per question. Descriptive intervals on reused validation.',
        source_calls=0,new_validation_loaded=False,assessment_loaded=False)
    assert len(p['train320']) == 320 and len(p['train512']) == len(set(p['train512'])) == 512 and len(p['validation']) == 80
    assert not (set(p['train512'])&set(p['validation']) or set(p['train512'])&set(p['assessment']))
    json_write(path,p); return p


def validate():
    p = protocol()
    for group in ('references','code_files'):
        for path,sha in p[group].items(): assert digest(path) == sha,path
    assert digest(inputs.MANIFEST) == p['selection_manifest_sha256']
    f = read(ROOT/'features_complete.json')
    for path,sha in f['artifacts'].items(): assert digest(path) == sha,path
    assert read(ROOT/'source_audit.json')['passed']
    return p


class Data:
    def __init__(self,name):
        self.device='cpu'; self.name=name; self.common=dict(np.load(DATA/'common.npz')); e,t=self.common['pairs'].T
        self.qids=self.common['episode_qids'][e]; single=np.load(DATA/'single.npy')
        self.source=single[e,t+1]; self.y=(self.source>.5).astype(np.int64)
        self.current=single[e,t]; self.initial=single[e,0]
        self.s=self.common['spins'][e,t].astype(np.float32); self.context=np.load(DATA/'context.npy')[e]
        pos=np.load(DATA/'positions.npz')
        values=dict(s=self.s,context=self.context,J=self.common['J'][e].astype(np.float32),
            current_probability=self.current.astype(np.float32),target=self.y.astype(np.float32),
            neighbors=pos['neighbors'][e],signs=pos['signs'][e],positions=pos['actual'])
        if name.startswith('history'):
            values['observed_history']=history.causal_states(self.common['spins'],single,self.common['J'],self.common['pairs'])
            values['current_round']=t.copy()
        if name == 'joint_arguments26210':
            values['initial_probability']=self.initial.astype(np.float32)
            values['initial_population']=np.broadcast_to(self.initial.astype(np.float32)[:,None],(*self.initial.shape,32)).copy()
            values['persona32']=np.broadcast_to(np.eye(32,dtype=np.float32)[None],(len(e),32,32)).copy()
            values['question32']=np.broadcast_to(np.load(DATA/'question32.npy')[e,None],(len(e),32,32)).copy()
            ids=np.load(DATA/'pair_message_ids.npy')[e,t]; vectors=np.load(DATA/'paired_vectors.npy',mmap_mode='r')
            lex=np.load(DATA/'sender_lexical.npy')[e,t]/4.*np.sqrt(40/392)
            values['text_arguments392']=np.concatenate([vectors[ids],lex],-1).astype(np.float32)
        else:
            vectors=np.load(DATA/'vectors8.npy'); ids=np.load(DATA/'message_ids.npy')[e,t]
            if name=='history_arguments2306': values['text_arguments8']=vectors[ids]
            elif name not in ('ising22','tree_semantic8'): values['text_conclusion8']=vectors[np.where(self.s==1,2,1)]
        self.values={k:torch.as_tensor(v) for k,v in values.items()}

    def ids(self,questions): return np.flatnonzero(np.isin(self.qids,questions))
    def batch(self,ids): return {k:v[ids] for k,v in self.values.items()}


def make_model(name,seed):
    spec=protocol()['models'][name]; seed_all(seed)
    model=(history.HistoryAttention if name.startswith('history') else core.ModeAttention)(spec['spec'],spec['training']['dropout'])
    torch.nn.init.zeros_(model.classifier[-1].weight); torch.nn.init.zeros_(model.classifier[-1].bias)
    assert sum(p.numel() for p in model.parameters())==spec['parameters']
    return model


def predict(model,data,ids):
    model.eval(); values=[]
    with torch.inference_mode():
        for begin in range(0,len(ids),128):
            z=core.score_model(model,data.batch(ids[begin:begin+128]))
            values.append(torch.stack([-F.softplus(z),-F.softplus(-z)],-1).numpy())
    return np.concatenate(values)


def worker(name,seed):
    p=validate(); torch.set_num_threads(2); data=Data(name); spec=p['models'][name]; tc=spec['training']
    folder=ROOT/'fits'/name/f's{seed}'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists(): return
    identity=dict(model=name,seed=seed,protocol_sha256=digest(ROOT/'protocol.json'),features_sha256=digest(ROOT/'features_complete.json'))
    model=make_model(name,seed); initial=core.dist.base.model_weights_sha(model)
    old=read(Path(spec['refits'][str(seed)])/'manifest.json'); assert initial==old['initial_weights_sha256']
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['optimizer']['learning_rate'],weight_decay=tc['weight_decay'])
    train=data.ids(p['train512']); assert len(train)==4096
    state=dict(step=0,epoch=0,position=0,seconds=0.)
    if (folder/'resume.pt').exists():
        ck=torch.load(folder/'resume.pt',weights_only=False); assert ck['identity']==identity
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); state=ck['state']; core.trainer.restore_rng(ck['rng'])
    start=time.monotonic(); previous=state['seconds']
    def checkpoint():
        state['seconds']=previous+time.monotonic()-start
        core.trainer.save_checkpoint(folder/'resume.pt',dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),state=state,rng=core.trainer.rng_state('cpu')))
        json_write(folder/'progress.json',dict(**identity,**state,planned=spec['steps']['same_passes'],updated_utc=now()))
    while state['step']<spec['steps']['same_passes']:
        order=np.random.default_rng(seed+state['epoch']).permutation(train)
        ids=order[state['position']:state['position']+tc['batch_size']]
        batch=data.batch(ids); model.train(); optimizer.zero_grad(set_to_none=True)
        z=core.score_model(model,batch); loss=core.dist.base.loss_for(z,z,batch['target'],spec['optimizer']['penalty'])
        assert torch.isfinite(loss); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),tc['gradient_clip'],error_if_nonfinite=True); optimizer.step()
        state['step']+=1; state['position']+=len(ids)
        if state['position']==len(train): state['epoch']+=1; state['position']=0
        for budget,steps in spec['steps'].items():
            if state['step']==steps:
                core.trainer.save_checkpoint(folder/(budget+'.pt'),dict(identity=identity,model=model.state_dict(),step=steps,budget=budget))
        if state['step']%512==0 or state['step'] in spec['steps'].values(): checkpoint()
    json_write(folder/'complete.json',dict(**identity,completed_utc=now(),initial_weights_sha256=initial,parameters=spec['parameters'],
        steps=spec['steps'],seconds=state['seconds'],checkpoint_hashes={b:digest(folder/(b+'.pt')) for b in spec['steps']},
        validation_used=False,optimization_snapshots=len(train)))
    print(dict(model=name,seed=seed,complete=True,seconds=state['seconds']),flush=True)


def fit_baselines():
    p=validate(); data=Data('ising22'); ids=data.ids(p['train512']); folder=ROOT/'baselines'; folder.mkdir(exist_ok=True)
    if not (folder/'ising22.json').exists():
        design=core.ising_design(data,ids,False).reshape(-1,22)
        author=author_module({'data':{'root':'data'}})
        with threadpool_limits(limits=2): weights=author.two_stage_fit(design[:,:16],design[:,16:-1],design[:,-1],np.ones(len(design),dtype=bool),data.y[ids].ravel())
        json_write(folder/'ising22.json',dict(weights=weights.tolist(),completed_utc=now(),parameters=len(weights)))
    if not (folder/'tree_complete.json').exists():
        selected=read(Path(core.config()['output'])/'baselines/tree_semantic8/complete.json')
        settings=dict(core.screen.SETTINGS,max_iter=selected['selected_iterations'],early_stopping=False)
        model=HistGradientBoostingClassifier(**settings); x=np.load(DATA/'tree_semantic8.npy',mmap_mode='r'); rows=(ids[:,None]*32+np.arange(32)).ravel()
        with threadpool_limits(limits=2): model.fit(x[rows],data.y[ids].ravel())
        with (folder/'tree_semantic8.pkl').open('wb') as stream: pickle.dump(model,stream)
        json_write(folder/'tree_complete.json',dict(completed_utc=now(),settings=settings,tree_nodes=sum(len(t[0].nodes) for t in model._predictors),sha256=digest(folder/'tree_semantic8.pkl')))


def run():
    p=validate()
    if (ROOT/'models_frozen.json').exists(): return
    fit_baselines(); pending=[(name,seed) for name in MODEL_SOURCES for seed in SEEDS]; active={}; finished={}
    while pending or active:
        while pending and len(active)<4:
            name,seed=pending.pop(0); key=f'{name}_s{seed}'; stream=(ROOT/(key+'.log')).open('a',buffering=1)
            active[key]=(subprocess.Popen([sys.executable,__file__,'worker','--model',name,'--seed',str(seed)],stdout=stream,stderr=subprocess.STDOUT),stream)
        for key,(process,stream) in list(active.items()):
            if process.poll() is not None: finished[key]=process.returncode; stream.close(); del active[key]
        json_write(ROOT/'progress.json',dict(phase='fitting',updated_utc=now(),active={k:p.pid for k,(p,s) in active.items()},pending=pending,finished=finished))
        if any(finished.values()):
            for process,stream in active.values(): process.terminate(); process.wait(); stream.close()
            raise RuntimeError(str(finished))
        if active: time.sleep(2)
    files=[ROOT/n for n in ('protocol.json','source_audit.json','encoding.json','features_complete.json')]
    files += [f for d in ('fits','baselines') for f in (ROOT/d).rglob('*') if f.is_file()]
    json_write(ROOT/'models_frozen.json',dict(frozen_utc=now(),artifacts={str(f):digest(f) for f in files},new_neural_fits=12,new_neural_endpoints=24,
        new_baselines=2,validation_used=False,assessment_loaded=False))
    json_write(ROOT/'progress.json',dict(phase='frozen_ready_to_evaluate',updated_utc=now()))


def evaluate():
    p=validate(); frozen=read(ROOT/'models_frozen.json')
    for path,sha in frozen['artifacts'].items(): assert digest(path)==sha,path
    out=ROOT/'evaluation'; out.mkdir(exist_ok=True)
    if (out/'results.json').exists(): return
    torch.set_num_threads(2); rows=[]; files=[]; mean={}
    def record(name,cohort,budget,seed,split,lp,data,ids,checkpoint):
        metric,per=core.dist.base.metrics(lp,data.y[ids],data.qids[ids]); key=f'{name}_{cohort}_{budget}'
        for r in per: rows.append(dict(model=name,cohort=cohort,budget=budget,seed=seed,split=split,**r))
        f=out/'predictions'/f'{key}_s{seed}_{split}.npz'; f.parent.mkdir(exist_ok=True)
        np.savez_compressed(f,logp=lp,target=data.y[ids],probability=data.source[ids],qids=data.qids[ids],pairs=data.common['pairs'][ids])
        files.append(dict(model=name,cohort=cohort,budget=budget,seed=seed,split=split,path=str(f),sha256=digest(f),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),metrics=metric))
    for name in MODEL_SOURCES:
        data=Data(name)
        for cohort,budget in [(320,'original'),(512,'same_updates'),(512,'same_passes')]:
            for seed in SEEDS:
                checkpoint=Path(p['models'][name]['refits'][str(seed)])/'best.pt' if cohort==320 else ROOT/'fits'/name/f's{seed}'/(budget+'.pt')
                model=make_model(name,seed); model.load_state_dict(torch.load(checkpoint,weights_only=False)['model'])
                for split,questions in [('train',p[f'train{cohort}']),('validation',p['validation'])]:
                    ids=data.ids(questions); lp=predict(model,data,ids); record(name,cohort,budget,seed,split,lp,data,ids,checkpoint)
    data=Data('ising22')
    for name in ('ising22','tree_semantic8'):
        for cohort in (320,512):
            if name=='ising22':
                checkpoint=Path(core.config()['output'])/'baselines/ising_current_mode/model.json' if cohort==320 else ROOT/'baselines/ising22.json'
                weights=np.asarray(read(checkpoint)['weights'])
            else:
                checkpoint=Path(core.config()['output'])/'baselines/tree_semantic8/model.pkl' if cohort==320 else ROOT/'baselines/tree_semantic8.pkl'
                with checkpoint.open('rb') as stream: tree=pickle.load(stream)
                x=np.load(DATA/'tree_semantic8.npy',mmap_mode='r')
            for split,questions in [('train',p[f'train{cohort}']),('validation',p['validation'])]:
                ids=data.ids(questions)
                if name=='ising22': z=core.ising_design(data,ids,False)@weights; lp=np.stack([-np.logaddexp(0,z),-np.logaddexp(0,-z)],-1)
                else:
                    index=(ids[:,None]*32+np.arange(32)).ravel()
                    with threadpool_limits(limits=2): pp=np.clip(tree.predict_proba(x[index])[:,1].reshape(-1,32),1e-10,1-1e-10)
                    lp=np.stack([np.log1p(-pp),np.log(pp)],-1)
                record(name,cohort,'original' if cohort==320 else 'fixed',0,split,lp,data,ids,checkpoint)
    conditions=sorted({(r['model'],r['cohort'],r['budget']) for r in rows})
    for name,cohort,budget in conditions:
        mean[f'{name}_{cohort}_{budget}']={}
        for split in ('train','validation'):
            group=[r for r in rows if (r['model'],r['cohort'],r['budget'],r['split'])==(name,cohort,budget,split)]
            mean[f'{name}_{cohort}_{budget}'][split]={m:float(np.mean([r[m] for r in group])) for m in ('error','nll','brier_two_class')}
            mean[f'{name}_{cohort}_{budget}'][split]['accuracy']=1-mean[f'{name}_{cohort}_{budget}'][split]['error']
    draws=np.random.default_rng(92813).integers(80,size=(5000,80)); contrasts=[]
    def perq(name,cohort,budget,metric):
        group=[r for r in rows if (r['model'],r['cohort'],r['budget'],r['split'])==(name,cohort,budget,'validation')]
        return np.asarray([np.mean([r[metric] for r in group if r['qid']==q]) for q in sorted(p['validation'])])
    for name,cohort,budget in conditions:
        if cohort!=512: continue
        for metric in ('error','nll','brier_two_class'):
            change=perq(name,320,'original',metric)-perq(name,512,budget,metric)
            contrasts.append(dict(model=name,budget=budget,metric=metric,reduction=float(change.mean()),ci95=np.quantile(change[draws].mean(1),[.025,.975]).tolist()))
    json_write(out/'by_question.json',rows)
    json_write(out/'results.json',dict(completed_utc=now(),protocol_sha256=digest(ROOT/'protocol.json'),models_frozen_sha256=digest(ROOT/'models_frozen.json'),
        summary=mean,contrasts=contrasts,prediction_files=files,target_accuracy=.95,source_calls=0,new_validation_loaded=False,assessment_loaded=False))
    print(mean,flush=True)


if __name__=='__main__':
    parser=ArgumentParser(); parser.add_argument('action',choices=['prepare','run','worker','evaluate']); parser.add_argument('--model'); parser.add_argument('--seed',type=int); args=parser.parse_args()
    if args.action=='prepare': print(protocol())
    elif args.action=='run': run()
    elif args.action=='worker': worker(args.model,args.seed)
    else: evaluate()
