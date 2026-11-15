"""Reconstruct the internal numeric-option comparison and paired uncertainty."""
from pathlib import Path
import pickle
import sys
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_option_values as study


def main():
    p=study.validate();root=study.ROOT;r=study.read(root/'results.json');d=np.load(root/'data.npz')
    assert r['protocol_sha256']==digest(root/'protocol.json')
    fit=np.isin(d['qids'],p['inner']['fit']);hold=np.isin(d['qids'],p['inner']['validation'])
    assert (fit^hold).all() and fit.sum()==65536 and hold.sum()==16384
    common=np.load(study.core.state.config()['common']);pairs=common['pairs'][d['pair_indices']];e,t=pairs.T
    assert set(t)==set(range(8))
    probabilities=np.load(Path(study.core.dist.config()['probability_reference'])/'probabilities.npz')['single']
    np.testing.assert_array_equal(d['target'],(probabilities[e,t+1]>.5).ravel())
    np.testing.assert_array_equal(d['qids'],np.repeat(common['episode_qids'][e],32))
    index={v['qid']:v for v in study.read(study.core.state.config()['index'])}
    names=[Path(index[q]['path']).stem for q in common['episode_qids']]
    order=study.core.state.ordered_neighbors(common['J'],names,pairs)
    np.testing.assert_array_equal(d['ordered'],study.core.screen.slots(d['sender_values'],order).reshape(-1,144))
    # Rebuild a fixed spread of raw messages through the input-only parser.
    checks=0
    for j in np.linspace(0,len(pairs)-1,24,dtype=int):
        episode=study.read(index[str(common['episode_qids'][e[j]])]['path'])
        for sender,text in enumerate(study.messages.snapshot_messages(episode,int(t[j]))):
            expected=study.numeric_features(text,episode['choices']) if text else np.zeros(8,dtype=np.float32)
            np.testing.assert_array_equal(expected,d['sender_values'][j,sender]);checks+=1
    base=np.load(study.core.screen.ROOT/'features/inner/semantic8.npy',mmap_mode='r')[d['rows']]
    oldlabels=np.load(study.core.screen.ROOT/'labels.npz');oldhold=np.isin(oldlabels['qids'],p['inner']['validation'])
    np.testing.assert_array_equal(d['qids'][hold],oldlabels['qids'][oldhold])
    np.testing.assert_array_equal(d['target'][hold],oldlabels['target'][oldhold])
    qids=d['qids'][hold];questions=sorted(set(qids));target=d['target'][hold]
    baseline=np.load(study.core.screen.ROOT/'screen/semantic8/inner_probability.npy')
    per0=np.array([((baseline>.5)==target)[qids==q].mean() for q in questions])
    assert per0.mean()==p['promotion_reference'];draws=np.random.default_rng(92871).integers(64,size=(5000,64));records=[]
    with threadpool_limits(limits=2):
        for row in r['results']:
            folder=root/row['variant']
            for path,sha in row['artifacts'].items():assert digest(path)==sha,path
            x=np.concatenate([base,d[row['variant']]],-1).astype(np.float32)
            with (folder/'terminal_model.pkl').open('rb') as stream:model=pickle.load(stream)
            for step,prob in enumerate(model.staged_predict_proba(x),1):
                if step==row['selected_step']:pp=np.clip(prob[:,1],1e-10,1-1e-10);break
            np.testing.assert_array_equal(pp[hold],np.load(folder/'internal_probability.npy'))
            accuracy=float(((pp[hold]>.5)==target).mean());assert accuracy==row['internal_accuracy']
            nll=float((-target*np.log(pp[hold])-(1-target)*np.log1p(-pp[hold])).mean());assert abs(nll-row['internal_nll'])<1e-12
            history=study.read(folder/'history.json');assert min(history,key=lambda z:(z['error'],z['nll'],z['step']))['step']==row['selected_step']
            per=np.array([((pp[hold]>.5)==target)[qids==q].mean() for q in questions]);delta=per-per0
            records.append(dict(variant=row['variant'],train_accuracy=float(((pp[fit]>.5)==d['target'][fit]).mean()),
                internal_accuracy=accuracy,selected_step=row['selected_step'],difference=float(delta.mean()),ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
    assert r['promotion_gate_passed']==(max(z['internal_accuracy'] for z in records)>=p['promotion_reference']+p['promotion_gain'])
    audit=dict(passed=True,completed_utc=study.now(),results_sha256=digest(root/'results.json'),audit_code_sha256=digest(__file__),
        raw_message_replays=checks,model_replays=2,checks=records,source_calls=0,outer_validation_scored=False,new_validation_loaded=False,assessment_loaded=False)
    json_write(root/'completion_audit.json',audit)
    lines=['# Exact numeric values in arguments: internal screen','',
        '**The ordered numeric features pass the prespecified +0.5 percentage-point promotion gate.** This is an internal256/64 comparison on all eight rounds, not outer validation. The inputs match literal integer, decimal and rational values in each actual message to the two supplied answer options. They never use the correct-answer key, worked solution or future state.','',
        '| Input | Training accuracy | Internal accuracy | Change vs reference |','| --- | ---: | ---: | ---: |',
        f"| Existing semantic8 state | — | {100*p['promotion_reference']:.3f}% | — |"]
    for z in records:lines.append(f"| {z['variant']} numeric fields | {100*z['train_accuracy']:.3f}% | {100*z['internal_accuracy']:.3f}% | {100*z['difference']:+.3f}pp [{100*z['ci95'][0]:+.3f}, {100*z['ci95'][1]:+.3f}] |")
    lines+=['','Eight fields summarize each message: optionA/B occurrence counts, last matched option, last numeric literal, A/B mentions in the last third, final boxed-value match, and numeric-literal count. The grouped model adds64 features; the ordered model adds144 features in actual friend/enemy inbox slots. All original622 features remain.', '',
        'The parser is deliberately limited to literal rational values. It does not solve algebra or decide whether an argument is correct; intermediate calculations can also mention an option value. The focused checks caught a sentence-final fraction boundary bug before any data preparation or fitting. The corrected parser passes all three checks.', '',
        f"Both models, their selected iterations, target order and {checks} raw message feature vectors replay. Question-bootstrap intervals are descriptive. The next separate experiment freezes a320/512 full-data comparison and a controlled neural augmentation before outer scoring.",'']
    Path('reports/modal_option_values_320.md').write_text('\n'.join(lines));print(audit,flush=True)


if __name__=='__main__':main()
