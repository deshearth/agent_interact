"""Independent verification and reporting of the frozen numeric follow-up."""
from pathlib import Path
import pickle
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_numeric_followup as study
from scripts import evaluate_modal_numeric_followup as evaluation


def independent_metrics(lp,y):
    lp=np.asarray(lp,dtype=float);yy=np.asarray(y,dtype=float);p=np.exp(lp[...,1])
    return dict(error=float(((lp[...,1]>lp[...,0])!=yy).mean()),
        nll=float((-(1-yy)*lp[...,0]-yy*lp[...,1]).mean()),brier_two_class=float((2*(p-yy)**2).mean()))


def main():
    root=study.ROOT;p=study.validate();frozen=study.read(root/'models_frozen.json');r=study.read(root/'evaluation/results.json')
    assert r['models_frozen_sha256']==digest(root/'models_frozen.json') and r['protocol_sha256']==digest(root/'protocol.json')
    assert frozen['frozen_utc']<r['completed_utc']
    for path,sha in frozen['artifacts'].items():assert digest(path)==sha,path
    for path,sha in study.read(root/'features.json')['sources'].items():assert digest(path)==sha,path
    tests=ET.parse('reports/modal_numeric_followup_checks.xml').getroot();suites=list(tests.iter('testsuite'))
    assert sum(int(s.get('tests','0')) for s in suites)==2 and all(int(s.get(k,'0'))==0 for s in suites for k in ('errors','failures','skipped'))
    data=study.Data('full');common=np.load(study.scaling.DATA/'common.npz');e,t=common['pairs'].T
    probability=np.load(study.scaling.DATA/'single.npy')[e,t+1];target=(probability>.5).astype(int);qids=common['episode_qids'][e]
    base=np.load(study.scaling.DATA/'tree_semantic8.npy',mmap_mode='r');extra=np.load(root/'ordered_numeric.npy')
    raw=np.load(root/'sender_numeric.npy');order=np.load(study.scaling.DATA/'positions.npz')['order']
    np.testing.assert_array_equal(extra,study.core.screen.slots(raw[e,t],order).reshape(-1,144))
    internal=np.load(study.pilot.ROOT/'data.npz');np.testing.assert_array_equal(extra[internal['rows']],internal['ordered'])
    cfg=study.config();torch.set_num_threads(2);rows=[];max_logp=0.;max_metric=0.;replays=0;class_disagreements=0
    reference_map={f['path']:f for f in study.read(study.scaling.ROOT/'evaluation/results.json')['prediction_files']}
    with threadpool_limits(limits=2):
        for f in r['prediction_files']:
            assert digest(f['path'])==f['sha256'] and digest(f['checkpoint'])==f['checkpoint_sha256']
            d=np.load(f['path']);split=f['split'];name=f['name'];cohort=512 if name.endswith('512') else 320
            questions=p['fold']['validation_questions'] if split=='validation' else p['train512'] if cohort==512 else p['fold']['train_questions']
            ids=np.flatnonzero(np.isin(qids,questions))
            np.testing.assert_array_equal(d['qids'],qids[ids]);np.testing.assert_array_equal(d['pairs'],common['pairs'][ids])
            np.testing.assert_array_equal(d['probability'],probability[ids]);np.testing.assert_array_equal(d['target'],target[ids])
            lp=d['logp'];met=independent_metrics(lp,d['target'])
            for k,v in met.items():max_metric=max(max_metric,abs(v-f['metrics'][k]))
            for q in sorted(questions):
                at=d['qids']==q;rows.append(dict(name=name,seed=f['seed'],split=split,qid=str(q),**independent_metrics(lp[at],d['target'][at])))
            if f['reference']:
                matched=[z for z in reference_map.values() if z['checkpoint']==f['checkpoint'] and z['split']==split]
                assert len(matched)==1
                old=matched[0];assert digest(old['path'])==old['sha256']
                np.testing.assert_array_equal(lp,np.load(old['path'])['logp']);continue
            if name.startswith('neural_'):
                key=name.removeprefix('neural_');model=study.make_model(cfg,key,f['seed'])
                model.load_state_dict(torch.load(f['checkpoint'],weights_only=False)['model']);model.eval();pred=[]
                small_ids=data.ids(questions);np.testing.assert_array_equal(small_ids,ids)
                with torch.inference_mode():
                    for start in range(0,len(ids),64):
                        z=model(data.batch(ids[start:start+64]));score=z[...,1]-z[...,0]
                        pred.append(torch.stack([-F.softplus(score),-F.softplus(-score)],-1).numpy())
                actual=np.concatenate(pred)
            else:
                with Path(f['checkpoint']).open('rb') as stream:model=pickle.load(stream)
                rr=(ids[:,None]*32+np.arange(32)).ravel();x=np.concatenate([base[rr],extra[rr]],-1) if name.startswith('numeric') else base[rr]
                pp=np.clip(model.predict_proba(x)[:,1],1e-10,1-1e-10).reshape(-1,32);actual=np.stack([np.log1p(-pp),np.log(pp)],-1)
            max_logp=max(max_logp,float(np.abs(actual-lp).max()));class_disagreements+=int(((actual[...,1]>actual[...,0])!=(lp[...,1]>lp[...,0])).sum())
            np.testing.assert_allclose(actual,lp,rtol=0,atol=2e-5);replays+=1
    assert max_metric<1e-12 and class_disagreements==0 and replays==20
    tuning=study.Data('inner');tuning_replays=0
    for manifest in sorted((root/'neural/tune').glob('*/manifest.json')):
        item=study.read(manifest);folder=manifest.parent
        assert item['status']=='complete' and item['protocol_sha256']==digest(root/'neural/protocol.json')
        model=study.make_model(cfg,item['width'],42)
        model.load_state_dict(torch.load(folder/'best.pt',weights_only=False)['model'])
        for split,key in [('train','fit'),('validation','validation')]:
            ids=tuning.ids(p['inner'][key]);actual=evaluation.predict(model,tuning,ids)
            saved=np.load(folder/(split+'_logp.npy'));np.testing.assert_array_equal(actual,saved)
            measured=independent_metrics(actual,tuning.y[ids])
            for k,v in measured.items():assert abs(v-item[split][k])<1e-12
            tuning_replays+=1
    assert tuning_replays==8
    selection=study.read(root/'neural/selection.json')
    for choice in selection['choices']:
        records=[(f.parent,study.read(f)) for f in (root/'neural/tune').glob('*/manifest.json') if study.read(f)['width']==choice['width']]
        best=min(records,key=lambda z:(z[1]['validation']['error'],z[1]['validation']['nll'],z[1]['selected_step'],z[1]['candidate']))
        assert str(best[0])==choice['tuning_directory'] and best[1]['selected_step']==choice['refit_steps']
    for seed in cfg['training']['refit_seeds']:
        identities=[study.read(next((root/'neural/refit').glob(f'{name}-mode-*-s{seed}/manifest.json')))['initial_weights_sha256'] for name in ['numeric','blank']]
        assert identities[0]==identities[1]
    summary,contrasts=evaluation.analyze(rows)
    for name,splits in summary.items():
        for split,values in splits.items():
            for k,v in values.items():assert abs(v-r['summary'][name][split][k])<1e-12
    for a,b in zip(contrasts,r['contrasts']):
        assert (a['new'],a['reference'],a['metric'])==(b['new'],b['reference'],b['metric'])
        assert abs(a['reduction']-b['reduction'])<1e-12;np.testing.assert_allclose(a['ci95'],b['ci95'],rtol=0,atol=1e-12)
    # These error subsets explain observed results only; no candidate or threshold is selected here.
    def stored(name):return np.load(next(f['path'] for f in r['prediction_files'] if f['name']==name and f['split']=='validation'))
    enhanced,control=stored('numeric512'),stored('matched_duration512');y=enhanced['target'];pa=enhanced['logp'].argmax(-1);pb=control['logp'].argmax(-1)
    ids=np.flatnonzero(np.isin(qids,p['fold']['validation_questions']));mentions=extra[(ids[:,None]*32+np.arange(32)).ravel()].reshape(-1,32,18,8)[...,:2].sum((-1,-2))>0
    error_groups=[]
    for label,mask in [('all',np.ones(y.shape,dtype=bool)),('an option value appears',mentions),('no option value appears',~mentions)]:
        error_groups.append(dict(group=label,updates=int(mask.sum()),numeric_accuracy=float((pa==y)[mask].mean()),control_accuracy=float((pb==y)[mask].mean()),
            corrected=float(((pa==y)&(pb!=y))[mask].mean()),introduced=float(((pa!=y)&(pb==y))[mask].mean())))
    audit=dict(passed=True,completed_utc=study.now(),results_sha256=digest(root/'evaluation/results.json'),audit_code_sha256=digest(__file__),
        prediction_arrays=len(r['prediction_files']),new_prediction_replays=replays,new_models=10,tuning_prediction_replays=tuning_replays,max_logp_difference=max_logp,max_metric_difference=max_metric,
        class_disagreements=class_disagreements,focused_checks=2,per_question_rows=len(rows),contrasts=len(contrasts),error_groups=error_groups,
        source_calls=0,new_validation_loaded=False,assessment_loaded=False)
    json_write(root/'completion_audit.json',audit)
    labels={'reference_tree320':'Original tree /320','reference_tree512':'Original tree /512','matched_duration320':'Duration-matched tree /320',
        'matched_duration512':'Duration-matched tree /512','numeric320':'Numeric argument tree /320','numeric512':'Numeric argument tree /512',
        'neural_blank':'History transformer, blank extra fields /320','neural_numeric':'History transformer, numeric arguments /320',
        'reference_history_arguments320':'Previous history/argument transformer /320','reference_history_choice320':'Previous history/choice transformer /320'}
    best=max(v['validation']['accuracy'] for v in summary.values())
    lines=['# Numeric arguments: full-data transfer comparison','',f'**Best validation accuracy in this comparison: {100*best:.3f}%.** '+('The95% target remains unmet.' if best<.95 else 'The95% point-estimate target is reached on the reused validation anchor; fresh assessment is still required.'),'',
        'The experiment followed a positive internal-only screen: exact rational matching between argument numbers and supplied answer options improved internal accuracy by0.531percentage points. All four tree refits and six neural refits froze before these80-question validation scores were calculated. All eight rounds and32agents are retained.','',
        '| Model / training questions | Training accuracy | Validation accuracy |','| --- | ---: | ---: |']
    for name in labels:
        z=summary[name];lines.append(f"| {labels[name]} | {100*z['train']['accuracy']:.3f}% | {100*z['validation']['accuracy']:.3f}% |")
    lines+=['','| Contrast | Accuracy change | Descriptive95% interval |','| --- | ---: | ---: |']
    for z in contrasts:
        if z['metric']=='error':lines.append(f"| {labels[z['new']]} vs {labels[z['reference']]} | {100*z['reduction']:+.3f}pp | [{100*z['ci95'][0]:+.3f}, {100*z['ci95'][1]:+.3f}] |")
    lines+=['','The numeric fields use only the two supplied option values and actual sender message text: counts of exact rational matches, their order/tail locations and a final boxed value. Equivalent fractions/decimals are recognized; algebra is not solved. The correct-answer key and future probabilities never enter predictor inputs.', '',
        'Both neural arms have2370 learned parameters, identical initial weights at each seed and the same history architecture. Both retain PCA8 argument vectors; only the eight numeric fields differ from zeros. Each arm selects its own learning rate and duration on the internal256/64 split, with the same search budget, then refits three fresh seeds. Reported neural scores average those individual models, not an ensemble. The frozen MiniLM encoder remains additional22,713,216parameters.', '',
        'Tree conditions append144 ordered numeric features to the original622 state/argument features. Both numeric and duration-matched controls use155 iterations, selected solely in the internal numeric pilot; original trees use their independently selected105 iterations. The512 corpus adds the same frozen first192new training questions used in the earlier scaling study. Its revised topic proportions limit a pure sample-count interpretation.', '',
        '| Validation subset, diagnostic only | Updates | Numeric512 accuracy | Duration control accuracy | Errors corrected | Errors introduced |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for z in error_groups:lines.append(f"| {z['group']} | {z['updates']:,} | {100*z['numeric_accuracy']:.3f}% | {100*z['control_accuracy']:.3f}% | {100*z['corrected']:.3f}% | {100*z['introduced']:.3f}% |")
    lines+=['',f"Verification checks all{len(r['prediction_files'])} prediction files, independently replays20 new training/validation outputs from10 models and all8 tuning outputs, confirms{class_disagreements} class disagreements, and reproduces all24 paired contrasts. Matched seed initializations and internal checkpoint selection also reproduce. Maximum log-probability discrepancy is{max_logp:.2g}. Two model-input checks and the earlier three numeric-parser checks pass. Validation remains exploratory after repeated reuse; new validation and reserved assessment conversations remain unused.",'']
    Path('reports/modal_numeric_followup_320.md').write_text('\n'.join(lines));print(audit,flush=True)


if __name__=='__main__':main()
