"""Internal diagnostic: option-specific meaning versus numeric mention counts."""
from pathlib import Path
import pickle
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_option_values as study

ROOT=Path('runs/modal_numeric_identity_control_320_v1')


def hide_identity(values):
    result=values.copy()
    for a,b in [(0,1),(4,5)]:
        total=np.expm1(values[...,a])+np.expm1(values[...,b])
        result[...,a]=result[...,b]=np.log1p(total/2)
    result[...,[2,3,6]]=0
    return result


def main():
    p=study.validate();assert study.read(study.ROOT/'completion_audit.json')['passed']
    ROOT.mkdir(exist_ok=True);chosen=study.read(study.ROOT/'ordered/complete.json')
    protocol=dict(created_utc=study.now(),references={str(study.ROOT/n):digest(study.ROOT/n) for n in ['protocol.json','results.json','completion_audit.json','data.npz']},
        code_sha256=digest(__file__),steps=chosen['selected_step'],inner=p['inner'],settings=study.core.screen.SETTINGS,
        intervention='Retain total option-value mention counts, tail counts and number counts in the same144 slots, but symmetrize A/B counts and zero the last-option/last-literal/boxed signs. '
            'This removes which option the numbers match. It does not alter delivered messages or source outcomes.',
        fitting='One fixed155-iteration refit on internal256, no new selection. Compare to the same155-step identity-aware pilot. '
            'Also predict with identity removed/swapped under the already trained identity-aware model as input-ablation diagnostics.',
        source_calls=0,outer_validation_scored=False,new_validation_loaded=False,assessment_loaded=False)
    if (ROOT/'protocol.json').exists():
        old=study.read(ROOT/'protocol.json');protocol['created_utc']=old['created_utc'];assert old==protocol
    else:json_write(ROOT/'protocol.json',protocol)
    if (ROOT/'results.json').exists():return
    d=np.load(study.ROOT/'data.npz');y=d['target'];fit=np.isin(d['qids'],p['inner']['fit']);hold=np.isin(d['qids'],p['inner']['validation'])
    slots=d['ordered'].reshape(-1,18,8);hidden=hide_identity(slots);swapped=slots.copy()
    swapped[...,[0,1]]=slots[...,[1,0]];swapped[...,[4,5]]=slots[...,[5,4]];swapped[...,[2,3,6]]=-slots[...,[2,3,6]]
    base=np.load(study.core.screen.ROOT/'features/inner/semantic8.npy',mmap_mode='r')[d['rows']]
    def x(value,mask):return np.concatenate([base[mask],value[mask].reshape(mask.sum(),144)],-1)
    with (study.ROOT/'ordered/terminal_model.pkl').open('rb') as stream:original=pickle.load(stream)
    model=HistGradientBoostingClassifier(**dict(p['settings'],max_iter=chosen['selected_step'],early_stopping=False));preds={};records=[]
    with threadpool_limits(limits=2):
        model.fit(x(hidden,fit),y[fit])
        with (ROOT/'count_only_model.pkl').open('wb') as stream:pickle.dump(model,stream)
        for name,value,estimator in [('original',slots,original),('counts_only_refit',hidden,model),('identity_removed_input',hidden,original),('identity_swapped_input',swapped,original)]:
            for step,prob in enumerate(estimator.staged_predict_proba(x(value,hold)),1):
                if step==chosen['selected_step']:pp=np.clip(prob[:,1],1e-10,1-1e-10);break
            preds[name]=pp;np.save(ROOT/(name+'.npy'),pp)
            records.append(dict(condition=name,accuracy=float(((pp>.5)==y[hold]).mean()),nll=float((-y[hold]*np.log(pp)-(1-y[hold])*np.log1p(-pp)).mean())))
        # Reopen the serialized model and verify predictions, independently of fitting.
        with (ROOT/'count_only_model.pkl').open('rb') as stream:replay=pickle.load(stream)
        np.testing.assert_array_equal(replay.predict_proba(x(hidden,hold))[:,1],preds['counts_only_refit'])
    np.testing.assert_array_equal(preds['original'],np.load(study.ROOT/'ordered/internal_probability.npy'))
    questions=sorted(set(d['qids'][hold]));qids=d['qids'][hold]
    per={name:np.array([((pp>.5)==y[hold])[qids==q].mean() for q in questions]) for name,pp in preds.items()}
    draws=np.random.default_rng(92893).integers(64,size=(5000,64));contrasts=[]
    for name in list(preds)[1:]:
        delta=per['original']-per[name];contrasts.append(dict(control=name,original_minus_control=float(delta.mean()),ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
    json_write(ROOT/'results.json',dict(completed_utc=study.now(),protocol_sha256=digest(ROOT/'protocol.json'),results=records,contrasts=contrasts,
        model_replay_exact=True,original_replay_exact=True,artifacts={str(f):digest(f) for f in ROOT.iterdir() if f.suffix in ['.npy','.pkl']},
        source_calls=0,outer_validation_scored=False,new_validation_loaded=False,assessment_loaded=False));print(records,flush=True)


if __name__=='__main__':main()
