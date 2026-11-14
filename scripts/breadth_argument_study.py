"""Matched argument/conclusion graph predictors on the fixed 320/80 corpus."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write, seed_all, environment
from mac_agents.models.graph import GraphUpdate
from mac_agents.models.baselines import baseline_logp
from mac_agents.train import rng_state, restore_rng, save_checkpoint
from scripts import transformer_generalization as base
from scripts import breadth_argument_features as features

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())

class ArgumentGraph(GraphUpdate):
    """Only sender keys/values see text; receiver queries keep the original state."""
    def __init__(self, **kwargs):
        super().__init__(kind='A2',context_dim=16,extra_dim=0,**kwargs)
        self.text_projection = nn.Linear(384,self.encoder[0].out_features,bias=False)

    def forward(self,s,context,J,text,return_social=False):
        b,n = s.shape
        state = torch.stack([s == -1,s == 1],-1).to(context.dtype)
        structure = torch.stack([(J>0).sum(-1),(J<0).sum(-1),J.abs().sum(-1)],-1).to(context.dtype).log1p()
        h = self.encoder(torch.cat([state,context,structure],-1)); hn = self.norm(h)
        sender = hn + self.text_projection(text*math.sqrt(384))
        q = self.q(hn).view(b,n,self.heads,self.d_k).transpose(1,2)
        k = self.k(sender).view(b,n,self.heads,self.d_k).transpose(1,2)
        v = self.v(sender).view(b,n,self.heads,self.d_v).transpose(1,2)
        mask = J.ne(0)[:,None]; isolated = ~mask.any(-1,keepdim=True)
        scores = q@k.transpose(-1,-2)/math.sqrt(self.d_k)
        positive,negative = (J>0)[:,None],(J<0)[:,None]
        scores = scores+positive*self.edge_bias[None,:,None,None,0]+negative*self.edge_bias[None,:,None,None,1]
        scores = torch.where(isolated,torch.zeros_like(scores),scores.masked_fill(~mask,-torch.inf))
        alpha = scores.softmax(-1)*mask
        signed = positive*self.gamma[None,:,None,None,0]+negative*self.gamma[None,:,None,None,1]
        rho = F.softplus(self.strength(structure)).transpose(1,2)[...,None]
        aggregate = rho*((alpha*signed)@v)
        social = self.out(aggregate.transpose(1,2).reshape(b,n,-1))
        h = h+self.dropout(social); h = h+self.dropout(self.ffn(h))
        logits = self.classifier(h)
        return (logits,social,alpha) if return_social else logits

def make_model(cfg,width,seed=None):
    if seed is not None: seed_all(seed)
    model = ArgumentGraph(dropout=cfg['training']['dropout'],**cfg['models'][width])
    nn.init.zeros_(model.classifier[-1].weight); nn.init.zeros_(model.classifier[-1].bias)
    return model

class Data:
    def __init__(self,cfg,device):
        root = Path(cfg['output'])/'data'
        self.common = dict(np.load(root/'development.npz',allow_pickle=False))
        self.fitted = dict(np.load(root/'features.npz',allow_pickle=False))
        self.vectors = np.load(root/'vectors.npy',mmap_mode='r',allow_pickle=False)
        self.message_ids = np.load(root/'message_ids.npy',allow_pickle=False)
        e,t = self.common['pairs'].T
        self.e,self.t = e,t
        self.s = self.common['spins'][e,t].astype(np.float32)
        self.y = (self.common['spins'][e,t+1] == 1).astype(np.int64)
        self.qids = self.common['episode_qids'][e]
        self.context = self.fitted['context'][e]
        self.graph = self.common['J'][e].astype(np.float32)
        self.ids_text = self.message_ids[e,t]
        self.condition,self.device = 'arguments',str(device)

    def ids(self,questions): return np.flatnonzero(np.isin(self.qids,questions))

    def batch(self,ids,condition=None):
        condition = self.condition if condition is None else condition
        s = self.s[ids]
        if condition == 'arguments': text_ids = self.ids_text[ids]
        elif condition == 'conclusion': text_ids = np.where(s==1,2,1)
        elif condition == 'zero': text_ids = np.zeros_like(s,dtype=np.int64)
        else: raise ValueError(condition)
        values = {'s':s,'context':self.context[ids],'J':self.graph[ids],
                  'text':np.asarray(self.vectors[text_ids]),'target':self.y[ids].astype(np.float32)}
        return {k:torch.as_tensor(v,device=self.device) for k,v in values.items()}

def score_model(model,batch,opinion=False):
    extra = torch.empty((*batch['s'].shape,0),device=batch['s'].device) if opinion else batch['text']
    z = model(batch['s'],batch['context'],batch['J'],extra)
    return z[...,1]-z[...,0]

def predictions(model,data,ids,batch_size=128,condition=None,opinion=False):
    model.eval(); values=[]
    with torch.inference_mode():
        for start in range(0,len(ids),batch_size):
            z = score_model(model,data.batch(ids[start:start+batch_size],condition),opinion)
            values.append(torch.stack([-F.softplus(z),-F.softplus(-z)],-1).cpu().numpy())
    result = np.concatenate(values)
    assert np.isfinite(result).all()
    return result

def candidates(cfg): return base.candidates(cfg)

def train_one(cfg,protocol,data,item,deadline,fixed_steps=None):
    root,tc = Path(cfg['output']),cfg['training']
    phase,width,condition,seed = (item[k] for k in ('phase','width','condition','seed'))
    candidate = next(c for c in candidates(cfg) if c['id']==item['candidate'])
    out = root/phase/f'{width}-{condition}-{candidate["id"]}-s{seed}'
    out.mkdir(parents=True,exist_ok=True)
    identity = {**item,'device':data.device,'protocol_sha256':digest(root/'protocol.json'),
                'encoding_sha256':digest(root/'encoding.json'),'fixed_steps':fixed_steps}
    if phase=='refit': identity['selection_sha256'] = digest(root/'selection.json')
    if (out/'manifest.json').exists():
        record = read(out/'manifest.json')
        assert all(record[k]==v for k,v in identity.items())
        if record['status']=='complete':
            for name,sha in record['artifacts'].items(): assert digest(out/name)==sha,name
            return out,record
    train = data.ids(protocol['fold']['train_questions']); val = data.ids(protocol['fold']['validation_questions'])
    assert not set(data.qids[train])&set(data.qids[val])
    data.condition = condition
    model = make_model(cfg,width,seed).to(data.device)
    initial = base.model_weights_sha(model)
    opt = torch.optim.AdamW(model.parameters(),lr=candidate['learning_rate'],weight_decay=tc['weight_decay'])
    state = {'step':0,'epoch':0,'position':0,'best':None,'bad':0,'history':[],'seconds':0.,'created_utc':now()}
    if (out/'resume.pt').exists():
        ck = torch.load(out/'resume.pt',map_location=data.device,weights_only=False)
        assert ck['identity']==identity
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']);state=ck['state'];restore_rng(ck['rng'])
    started,previous_seconds = time.monotonic(),state['seconds']
    common = {**identity,'initial_weights_sha256':initial,'parameters':sum(p.numel() for p in model.parameters()),
              'candidate_settings':candidate,'created_utc':state['created_utc'],'zero_initialized_output':True}
    json_write(out/'manifest.json',{**common,'status':'running'})

    def checkpoint():
        state['seconds'] = previous_seconds+time.monotonic()-started
        save_checkpoint(out/'resume.pt',{'identity':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),'state':state,'rng':rng_state(data.device)})
        json_write(out/'history.json',state['history'])

    def assess():
        lp = predictions(model,data,val,tc['evaluation_batch_size'])
        metric,_ = base.metrics(lp,data.y[val],data.qids[val])
        history = {'step':state['step'],'epoch':state['epoch'],'validation':metric}
        state['history'].append(history)
        if state['best'] is None or metric['nll']<state['best']-tc['min_delta']:
            state['best'],state['bad']=metric['nll'],0
            save_checkpoint(out/'best.pt',{'identity':identity,'model':model.state_dict(),'step':state['step'],'validation':metric})
        else: state['bad']+=1
        json_write(root/'progress.json',{**item,**history,'updated_utc':now()})

    if phase=='tune' and not state['history']: assess();checkpoint()
    maximum = tc['max_steps'] if phase=='tune' else fixed_steps
    reason = 'step_cap' if phase=='tune' else 'validation_selected_fixed_duration'
    while state['step']<maximum:
        if phase=='tune' and state['bad']>=tc['patience']:reason='early_stopping';break
        if time.time()>=deadline:
            checkpoint();json_write(out/'manifest.json',{**common,'status':'bounded','step':state['step']});return out,{'status':'bounded'}
        order = np.random.default_rng(seed+state['epoch']).permutation(train)
        ids = order[state['position']:state['position']+tc['batch_size']]
        batch = data.batch(ids);model.train();opt.zero_grad(set_to_none=True)
        score = score_model(model,batch)
        loss = base.loss_for(score,score,batch['target'],candidate['penalty'])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),tc['gradient_clip'],error_if_nonfinite=True);opt.step()
        state['step']+=1;state['position']+=len(ids)
        if state['position']==len(train):state['epoch']+=1;state['position']=0
        if state['step']%tc['validation_every_steps']==0 or state['step']==maximum:
            if phase=='tune':assess()
            else:json_write(root/'progress.json',{**item,'step':state['step'],'fixed_steps':fixed_steps,'updated_utc':now()})
            checkpoint()
    checkpoint(); terminal = base.model_weights_sha(model)
    if phase=='tune':
        chosen = torch.load(out/'best.pt',map_location=data.device,weights_only=False);model.load_state_dict(chosen['model']);selected_step=chosen['step']
        lp = predictions(model,data,val,tc['evaluation_batch_size'])
        vm,vr = base.metrics(lp,data.y[val],data.qids[val]);np.save(out/'validation_logp.npy',lp,allow_pickle=False)
        np.testing.assert_allclose(vm['nll'],chosen['validation']['nll'],atol=1e-7,rtol=1e-6)
    else:
        selected_step=state['step'];vm,vr=None,[]
        save_checkpoint(out/'best.pt',{'identity':identity,'model':model.state_dict(),'step':selected_step})
    lp = predictions(model,data,train,tc['evaluation_batch_size'])
    tm,tr = base.metrics(lp,data.y[train],data.qids[train]);np.save(out/'train_logp.npy',lp,allow_pickle=False)
    json_write(out/'by_question.json',{'train':tr,'validation':vr})
    result = {**common,'status':'complete','completed_utc':now(),'steps':state['step'],'selected_step':selected_step,
              'selected_weights_sha256':base.model_weights_sha(model),'terminal_weights_sha256':terminal,
              'train':tm,'validation':vm,'seconds':previous_seconds+time.monotonic()-started,'stopping_reason':reason,
              'artifacts':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}}
    json_write(out/'manifest.json',result)
    print(json.dumps({'stage':'fit_complete','run':out.name,'selected_step':selected_step,'validation':vm,'seconds':result['seconds']}),flush=True)
    return out,result

def choose(cfg,records):
    choices=[]
    for width in cfg['models']:
        for condition in cfg['conditions']:
            values=[(p,r) for p,r in records if r['width']==width and r['condition']==condition]
            assert len(values)==len(candidates(cfg))
            assert len({r['initial_weights_sha256'] for p,r in values})==1
            p,r=min(values,key=lambda item:(item[1]['validation']['nll'],item[1]['candidate']))
            choices.append({'width':width,'condition':condition,'candidate':r['candidate'],'refit_steps':r['selected_step'],
                            'validation_nll':r['validation']['nll'],'tuning_directory':str(p)})
    return choices

def fit(cfg,device):
    protocol=features.validate(cfg,True);root=Path(cfg['output'])
    if (root/'models_frozen.json').exists():validate_frozen(cfg);return
    data=Data(cfg,device);deadline=time.time()+60*cfg['training']['aggregate_minutes'];records=[]
    json_write(root/'training_environment.json',environment())
    for width in cfg['models']:
        for condition in cfg['conditions']:
            for c in candidates(cfg):
                item={'phase':'tune','width':width,'condition':condition,'candidate':c['id'],'seed':cfg['training']['tuning_seed']}
                p,r=train_one(cfg,protocol,data,item,deadline)
                if r['status']!='complete':raise SystemExit(75)
                records.append((p,r))
    selected={'protocol_sha256':digest(root/'protocol.json'),'choices':choose(cfg,records),
              'tuning_manifests':{str(p/'manifest.json'):digest(p/'manifest.json') for p,r in records}}
    if (root/'selection.json').exists(): assert read(root/'selection.json')==selected
    else:json_write(root/'selection.json',selected)
    refits=[]
    for choice in selected['choices']:
        for seed in cfg['training']['refit_seeds']:
            item={k:choice[k] for k in ('width','condition','candidate')};item.update(phase='refit',seed=seed)
            p,r=train_one(cfg,protocol,data,item,deadline,choice['refit_steps'])
            if r['status']!='complete':raise SystemExit(75)
            refits.append((p,r))
    artifacts={str(p):digest(p) for p in [root/'protocol.json',root/'prepared.json',root/'encoding.json',root/'selection.json']}
    for p,r in records+refits:
        artifacts[str(p/'manifest.json')]=digest(p/'manifest.json')
        artifacts.update({str(p/name):sha for name,sha in r['artifacts'].items()})
    json_write(root/'models_frozen.json',{'frozen_utc':now(),'artifacts':artifacts,'refits':[str(p) for p,r in refits],
        'parameters':{w:sum(p.numel() for p in make_model(cfg,w).parameters()) for w in cfg['models']},
        'tuning_fits':len(records),'refits_completed':len(refits),'fit_seconds':sum(r['seconds'] for p,r in records+refits),
        'assessment_loaded':False,'source_llm_calls':0})
    print(json.dumps({'stage':'models_frozen','tuning_fits':len(records),'refits':len(refits)}),flush=True)

def validate_frozen(cfg):
    features.validate(cfg,True);f=read(Path(cfg['output'])/'models_frozen.json')
    for p,sha in f['artifacts'].items():assert digest(p)==sha,p
    return f

def additional_metrics(lp,data,ids):
    error=lp.argmax(-1)!=data.y[ids]
    switched=data.y[ids]!=(data.s[ids]==1)
    result={}
    for name,mask in [('switch',switched),('stay',~switched)]:
        result[name+'_error']=float(error[mask].mean()) if mask.any() else None
        result[name+'_count']=int(mask.sum())
    return result

def contrasts(by_question,cfg):
    rng=np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    ids=sorted({r['qid'] for r in by_question if r['split']=='validation'})
    draws=rng.integers(0,len(ids),(cfg['evaluation']['bootstrap_replicates'],len(ids)))
    models=sorted({r['model'] for r in by_question})
    values={}
    for m in models:
        values[m]={}
        for metric in ('nll','error','brier_two_class'):
            values[m][metric]=np.array([np.mean([r[metric] for r in by_question if r['model']==m and r['split']=='validation' and r['qid']==q]) for q in ids])
    pairs=[(f'{w}_arguments',f'{w}_conclusion') for w in cfg['models']]
    pairs += [(f'{w}_arguments','opinion_nano') for w in cfg['models']]
    pairs += [(f'{w}_arguments','I0') for w in cfg['models']]
    pairs += [('full_arguments','nano_arguments')]
    result=[]
    for left,right in pairs:
        for metric in ('nll','error','brier_two_class'):
            difference=values[left][metric]-values[right][metric]
            result.append({'left':left,'right':right,'metric':metric,'mean_difference':float(difference.mean()),
                           'ci95':np.quantile(difference[draws].mean(1),[.025,.975]).tolist(),'questions':len(ids)})
    return result

def evaluate(cfg,device):
    frozen=validate_frozen(cfg);root=Path(cfg['output']);out=root/'results'
    if (out/'summary.json').exists():return
    protocol=read(root/'protocol.json');data=Data(cfg,device);out.mkdir(exist_ok=True)
    seed_rows=[];question_rows=[];prediction_index=[];cpu_checks=[];sensitivity=[]

    def record(name,seed,split,lp,ids):
        metrics,rows=base.metrics(lp,data.y[ids],data.qids[ids]);metrics.update(additional_metrics(lp,data,ids))
        seed_rows.append({'model':name,'seed':seed,'split':split,**metrics})
        question_rows.extend({'model':name,'seed':seed,'split':split,**r} for r in rows)
        file=out/f'{name}-s{seed}-{split}.npz'
        np.savez_compressed(file,logp=lp,target=data.y[ids],current=data.s[ids],qids=data.qids[ids],pairs=data.common['pairs'][ids])
        prediction_index.append({'path':str(file),'sha256':digest(file),'model':name,'seed':seed,'split':split})

    for directory in frozen['refits']:
        p=Path(directory);m=read(p/'manifest.json');width,condition,seed=m['width'],m['condition'],m['seed']
        model=make_model(cfg,width).to(device);ck=torch.load(p/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model'])
        data.condition=condition;name=f'{width}_{condition}'
        for split in ('train','validation'):
            ids=data.ids(protocol['fold'][split+'_questions']);lp=predictions(model,data,ids,cfg['training']['evaluation_batch_size'])
            record(name,seed,split,lp,ids)
            if split=='train':np.testing.assert_allclose(lp,np.load(p/'train_logp.npy'),atol=1e-6,rtol=1e-5)
            if split=='validation':
                picked=np.random.default_rng(193).choice(len(ids),size=min(32,len(ids)),replace=False)
                cpu=make_model(cfg,width);cpu.load_state_dict({k:v.cpu() for k,v in ck['model'].items()})
                old_device=data.device;data.device='cpu'
                cp=predictions(cpu,data,ids[picked],32);data.device=old_device
                delta=float(np.max(np.abs(np.exp(cp)-np.exp(lp[picked]))))
                assert delta<5e-5
                cpu_checks.append({'model':name,'seed':seed,'maximum_probability_difference':delta,
                    'class_disagreements':int((cp.argmax(-1)!=lp[picked].argmax(-1)).sum())})
                if condition=='arguments':
                    replaced=predictions(model,data,ids,cfg['training']['evaluation_batch_size'],'conclusion')
                    rm,_=base.metrics(replaced,data.y[ids],data.qids[ids])
                    sensitivity.append({'model':name,'seed':seed,'original_vs_conclusion_probability_mae':float(np.abs(np.exp(lp[...,1])-np.exp(replaced[...,1])).mean()),
                        'class_disagreement':float((lp.argmax(-1)!=replaced.argmax(-1)).mean()),'replacement_validation':rm})
                    np.savez_compressed(out/f'{name}-s{seed}-conclusion_diagnostic.npz',logp=replaced,target=data.y[ids],qids=data.qids[ids])
        print(json.dumps({'stage':'scored','model':name,'seed':seed}),flush=True)
    for width,ref in [('nano',Path(cfg['reference'])),('full',Path(cfg['size_reference']))]:
        for path in sorted((ref/'predictors/refit/main').glob(f'{width}-*/manifest.json')):
            m=read(path);seed=m['seed'];model=base.make_model(cfg,width).to(device)
            ck=torch.load(path.parent/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model'])
            for split in ('train','validation'):
                ids=data.ids(protocol['fold'][split+'_questions'])
                record('opinion_'+width,seed,split,predictions(model,data,ids,cfg['training']['evaluation_batch_size'],'zero',True),ids)
    ising=read(root/'data/ising.json')
    for split in ('train','validation'):
        ids=data.ids(protocol['fold'][split+'_questions'])
        lp=baseline_logp(ising,data.s[ids],data.context[ids],data.graph[ids]);record('I0',-1,split,lp,ids)
    summary={}
    for model in sorted({r['model'] for r in seed_rows}):
        summary[model]={}
        for split in ('train','validation'):
            rows=[r for r in seed_rows if r['model']==model and r['split']==split]
            summary[model][split]={k:float(np.mean([r[k] for r in rows])) for k in ('nll','error','brier_two_class','switch_error','stay_error')}
    for model,key in [('opinion_nano','nano'),('opinion_full','full')]:
        prior=read(Path(cfg['size_reference'])/'results/summary.json')
        expected=prior['summary'][key]
        for split in ('train','validation'):
            for metric in ('nll','error','brier_two_class'):
                np.testing.assert_allclose(summary[model][split][metric],expected[split][metric],atol=3e-6)
    json_write(out/'by_question.json',question_rows);json_write(out/'predictions.json',prediction_index)
    json_write(out/'summary.json',{'completed_utc':now(),'summary':summary,'by_seed':seed_rows,
        'paired_differences':contrasts(question_rows,cfg),'cpu_reproduction':cpu_checks,'message_replacement_diagnostic':sensitivity,
        'parameters':{'I0':19,'opinion_nano':818,'opinion_full':135458,**{f'{w}_{c}':frozen['parameters'][w] for w in cfg['models'] for c in cfg['conditions']}},
        'models_frozen_sha256':digest(root/'models_frozen.json'),'encoding':read(root/'encoding.json'),
        'fit_seconds':frozen['fit_seconds'],'scope':'Reused 320/80 development corpus; delivered-message response prediction, not autonomous forecasting.',
        'assessment_loaded':False,'source_llm_calls':0,'reference_scores_reproduced':True})
    print(json.dumps({'stage':'evaluation_complete','summary':summary}),flush=True)

if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('action',choices=['fit','evaluate']);parser.add_argument('--device',default='cpu');args=parser.parse_args()
    cfg=features.config();torch.set_num_threads(cfg['training']['torch_threads'])
    if args.action=='fit':fit(cfg,args.device)
    else:evaluate(cfg,args.device)
