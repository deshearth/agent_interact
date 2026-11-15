"""Frozen training fork: checkpoint selection uses modal accuracy, then NLL.

Derived from the existing resumable trainer; optimizer, batches, checkpoint
resumption and seed behavior are unchanged. Probability labels may remain soft
in the matched control, but evaluation labels are always the source mode.
"""
from pathlib import Path
import json
import time
import numpy as np
import torch
from mac_agents.common import digest, json_write
from scripts.breadth_argument_study import (base, candidates, read, now, predictions,
    make_model, score_model, rng_state, restore_rng, save_checkpoint)

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
        key = (metric['error'], metric['nll'])
        if state['best'] is None or key < tuple(state['best']):
            state['best'],state['bad']=key,0
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
