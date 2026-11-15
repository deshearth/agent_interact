"""Training-only test of exact numeric answer-option mentions in arguments."""
from argparse import ArgumentParser
from pathlib import Path
import pickle
import re
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_neural_study as core
from scripts import breadth_argument_features as messages
from scripts.question_breadth import rational_answer

ROOT=Path('runs/modal_option_values_320_v1')
read,now=core.read,core.now
NUMBERS=re.compile(r'(?<![A-Za-z0-9_.])[+-]?(?:\\(?:d|t)?frac\s*\{\s*[+-]?\d+\s*\}\s*\{\s*[+-]?\d+\s*\}|\d+\s*/\s*[+-]?\d+|(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?![A-Za-z0-9_]|\.\d)')


def canonical(text): return re.sub(r'\\+',lambda m:'\\',text)


def numeric_features(text,choices):
    """Only provided options and incoming text; no correct answer is accepted."""
    option=[rational_answer(canonical(choices[label])) for label in ('A','B')]
    assert all(x is not None for x in option) and option[0]!=option[1]
    text=canonical(text); numbers=[]
    for match in NUMBERS.finditer(text):
        value=rational_answer(match.group());
        if value is not None: numbers.append((match.start(),value))
    def sign(value): return 1 if value==option[0] else -1 if value==option[1] else 0
    hits=[(position,sign(value)) for position,value in numbers]
    matches=[(position,s) for position,s in hits if s]
    tail=[s for position,s in hits if position>=2*len(text)/3]
    boxed=0
    for m in re.finditer(r'\\boxed\s*\{',text):
        depth=1; start=m.end()
        for end in range(start,len(text)):
            if text[end]=='{': depth+=1
            elif text[end]=='}': depth-=1
            if depth==0:
                value=rational_answer(text[start:end]); boxed=sign(value) if value is not None else 0; break
    return np.array([np.log1p(sum(s==1 for _,s in hits)),np.log1p(sum(s==-1 for _,s in hits)),
        matches[-1][1] if matches else 0,hits[-1][1] if hits else 0,
        np.log1p(tail.count(1)),np.log1p(tail.count(-1)),boxed,np.log1p(len(numbers))],dtype=np.float32)


def prepare():
    if (ROOT/'protocol.json').exists(): return read(ROOT/'protocol.json')
    ROOT.mkdir(parents=True,exist_ok=True); parent=read(Path(core.config()['output'])/'protocol.json'); common=dict(np.load(core.state.config()['common']))
    e,t=common['pairs'].T; take=np.flatnonzero(np.isin(common['episode_qids'][e],parent['fold']['train_questions']))
    index={r['qid']:r for r in read(core.state.config()['index'])}; all_features=np.zeros((400,8,32,8),dtype=np.float32); source_hashes={}
    for episode in np.unique(e[take]):
        qid=str(common['episode_qids'][episode]); item=index[qid]; assert item['split']=='train' and digest(item['path'])==item['sha256']
        ep=read(item['path']); source_hashes[item['path']]=item['sha256']
        for round_number in range(8):
            for sender,text in enumerate(messages.snapshot_messages(ep,round_number)):
                if text: all_features[episode,round_number,sender]=numeric_features(text,ep['choices'])
    names=[Path(index[q]['path']).stem for q in common['episode_qids']]
    order=core.state.ordered_neighbors(common['J'],names,common['pairs'][take]); values=all_features[e[take],t[take]]
    graph=common['J'][e[take]]; spins=common['spins'][e[take],t[take]]; parts=[]
    for sign in (1,-1):
        for stance in (1,-1): parts.append(core.state.aggregate(((graph==sign)&(spins[:,None,:]==stance)).astype(np.float32),values))
        group=order[:,:,0 if sign==1 else 1]; count=(group>=0).sum(-1)
        for last in (False,True):
            sender=np.take_along_axis(group,np.maximum(count-1,0)[...,None],-1)[...,0] if last else group[...,0]
            parts.append(values[np.arange(len(values))[:,None],np.maximum(sender,0)]*(count>0)[...,None])
    raw_labels=np.load(core.screen.ROOT/'labels.npz'); rows=(take[:,None]*32+np.arange(32)).ravel()
    np.savez_compressed(ROOT/'data.npz',summary=np.concatenate(parts,-1).reshape(-1,64),ordered=core.screen.slots(values,order).reshape(-1,144),
        target=raw_labels['target'][rows],qids=raw_labels['qids'][rows],rows=rows,pair_indices=take,sender_values=values)
    references=[core.screen.ROOT/'features/inner/semantic8.npy',core.screen.ROOT/'screen/semantic8/complete.json',Path(core.config()['output'])/'protocol.json']
    p=dict(created_utc=now(),inner=parent['inner'],training_questions=parent['fold']['train_questions'],variants=['summary','ordered'],settings=core.screen.SETTINGS,
        hypothesis='Numerical values and equivalent fractions in arguments identify answer choices in a way that compressed language vectors may miss.',
        features='Eight fields per actual sender message: A/B value counts, last matched option, last numeric literal match, A/B tail counts, final boxed-value match, number count. '
            'Parse literal integers, decimals and fractions only; equivalent rational values match exactly. No equation solving or correctness labels. '
            'Append either64 grouped/first/last fields or144 exact-inbox-slot fields to the existing semantic8 state baseline.',
        selection='All rounds of internal256/64 only. Same booster settings and accuracy/NLL iteration selection as the reference. No outer evaluation unless a0.5pp gain is observed.',
        promotion_gain=.005,promotion_reference=read(core.screen.ROOT/'screen/semantic8/complete.json')['inner_accuracy'],
        references={str(f):digest(f) for f in references},sources=source_hashes,code_files={str(f):digest(f) for f in [Path(__file__),Path('tests/test_modal_option_values.py')]},
        data_sha256=digest(ROOT/'data.npz'),outer_validation_scored=False,new_validation_loaded=False,assessment_loaded=False,source_calls=0)
    json_write(ROOT/'protocol.json',p); return p


def validate():
    p=prepare()
    for group in ('references','sources','code_files'):
        for path,sha in p[group].items(): assert digest(path)==sha,path
    assert digest(ROOT/'data.npz')==p['data_sha256']; return p


def run():
    p=validate(); d=np.load(ROOT/'data.npz'); fit=np.isin(d['qids'],p['inner']['fit']); hold=np.isin(d['qids'],p['inner']['validation']); y=d['target']
    base=np.load(core.screen.ROOT/'features/inner/semantic8.npy',mmap_mode='r')[d['rows']]; results=[]
    for name in p['variants']:
        folder=ROOT/name;folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():results.append(read(folder/'complete.json'));continue
        x=np.concatenate([base,d[name]],-1).astype(np.float32); model=HistGradientBoostingClassifier(**p['settings'],early_stopping=True); start=time.monotonic(); best=None; history=[]
        with threadpool_limits(limits=2):
            model.fit(x[fit],y[fit],X_val=x[hold],y_val=y[hold])
            for step,prob in enumerate(model.staged_predict_proba(x[hold]),1):
                pp=np.clip(prob[:,1],1e-10,1-1e-10); error=float(((pp>.5)!=y[hold]).mean()); nll=float((-y[hold]*np.log(pp)-(1-y[hold])*np.log1p(-pp)).mean())
                key=error,nll,step;history.append(dict(step=step,error=error,nll=nll))
                if best is None or key<best:best=key;chosen=pp.copy()
        np.save(folder/'internal_probability.npy',chosen);json_write(folder/'history.json',history)
        with (folder/'terminal_model.pkl').open('wb') as stream:pickle.dump(model,stream)
        record=dict(variant=name,internal_accuracy=1-best[0],internal_nll=best[1],selected_step=best[2],features=x.shape[1],seconds=time.monotonic()-start,
            artifacts={str(f):digest(f) for f in folder.iterdir()})
        json_write(folder/'complete.json',record);results.append(record);print(record,flush=True)
    json_write(ROOT/'results.json',dict(completed_utc=now(),results=results,promotion_reference=p['promotion_reference'],
        promotion_gate_passed=max(r['internal_accuracy'] for r in results)>=p['promotion_reference']+p['promotion_gain'],
        protocol_sha256=digest(ROOT/'protocol.json'),source_calls=0,outer_validation_scored=False,new_validation_loaded=False,assessment_loaded=False))


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args();{'prepare':prepare,'run':run}[args.action]()
