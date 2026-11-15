"""Replay the 320/512 numeric-transformer comparison and write its report."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_numeric_scaling as study
from scripts.audit_modal_numeric_followup import independent_metrics


def main():
    root=study.ROOT;p=study.validate();frozen=study.read(root/'models_frozen.json');r=study.read(root/'evaluation/results.json')
    assert r['protocol_sha256']==digest(root/'protocol.json') and r['models_frozen_sha256']==digest(root/'models_frozen.json')
    assert frozen['frozen_utc']<r['completed_utc']
    for path,sha in frozen['artifacts'].items():assert digest(path)==sha,path
    suites=list(ET.parse('reports/modal_numeric_scaling_checks.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests','0')) for s in suites)==2
    assert all(int(s.get(k,'0'))==0 for s in suites for k in ('failures','errors','skipped'))
    old=study.read(study.prior.ROOT/'evaluation/results.json');old_files={(f['name'].removeprefix('neural_'),f['seed'],f['split']):f for f in old['prediction_files'] if f['name'].startswith('neural_')}
    data={name:study.Data(name) for name in p['models']};torch.set_num_threads(2);maximum=0.;metric_max=0.;classes=0;rows=[]
    for f in r['prediction_files']:
        assert digest(f['path'])==f['sha256'] and digest(f['checkpoint'])==f['checkpoint_sha256']
        d=np.load(f['path']);source=data[f['model']];questions=p['validation'] if f['split']=='validation' else p['train'+str(f['cohort'])]
        ids=source.ids(questions)
        for key,value in [('target',source.y[ids]),('probability',source.source[ids]),('qids',source.qids[ids]),('pairs',source.common['pairs'][ids])]:
            np.testing.assert_array_equal(d[key],value)
        model=study.make_model(f['model'],f['seed']);model.load_state_dict(torch.load(f['checkpoint'],weights_only=False)['model'])
        actual=study.engine.predict(model,source,ids);saved=d['logp']
        maximum=max(maximum,float(np.abs(actual-saved).max()));classes+=int((actual.argmax(-1)!=saved.argmax(-1)).sum())
        np.testing.assert_array_equal(actual,saved)
        m=independent_metrics(saved,d['target'])
        for key,value in m.items():metric_max=max(metric_max,abs(value-f['metrics'][key]))
        for q in sorted(questions):
            at=d['qids']==q;rows.append(dict(model=f['model'],cohort=f['cohort'],budget=f['budget'],seed=f['seed'],split=f['split'],qid=str(q),**independent_metrics(saved[at],d['target'][at])))
        if f['cohort']==320:
            prior=old_files[f['model'],f['seed'],f['split']];assert digest(prior['path'])==prior['sha256']
            np.testing.assert_array_equal(saved,np.load(prior['path'])['logp'])
        else:
            record=study.read(Path(f['checkpoint']).parent/'complete.json');parent=study.read(Path(p['models'][f['model']]['refits'][str(f['seed'])])/'manifest.json')
            assert record['initial_weights_sha256']==parent['initial_weights_sha256'] and not record['validation_used']
            assert record['checkpoint_hashes'][f['budget']]==f['checkpoint_sha256']
    summary,contrasts=study.summaries(rows)
    for name,splits in summary.items():
        for split,metrics in splits.items():
            for key,value in metrics.items():assert abs(value-r['summary'][name][split][key])<1e-12
    for a,b in zip(contrasts,r['contrasts']):
        assert (a['new'],a['reference'],a['metric'])==(b['new'],b['reference'],b['metric'])
        assert abs(a['reduction']-b['reduction'])<1e-12;np.testing.assert_allclose(a['ci95'],b['ci95'],rtol=0,atol=1e-12)
    assert len(r['prediction_files'])==36 and len(contrasts)==21 and maximum==0 and classes==0 and metric_max<1e-12
    audit=dict(passed=True,completed_utc=study.now(),results_sha256=digest(root/'evaluation/results.json'),audit_code_sha256=digest(__file__),
        prediction_arrays=36,new_endpoints=12,old_prediction_arrays_preserved=12,checkpoint_prediction_replays=36,
        per_question_rows=len(rows),paired_contrasts=21,max_logp_difference=maximum,max_metric_difference=metric_max,class_disagreements=classes,
        focused_checks=2,source_calls=0,new_validation_loaded=False,assessment_loaded=False)
    json_write(root/'completion_audit.json',audit)
    best=max((z['validation']['accuracy'] for k,z in summary.items() if k.startswith('numeric_512')))
    lines=['# Scaling the numeric-argument transformer to 512 questions','',
        f'**Numeric transformer validation accuracy reaches {100*best:.3f}% among the two predeclared 512-question budgets.** Both budgets are reported below; this maximum is not used to select another checkpoint. '+('The 95% target remains unmet.' if best<.95 else 'The 95% point-estimate target is reached on the reused anchor, pending fresh assessment.'),'',
        '| Model | 320 questions | 512, same updates | 512, same data passes |','| --- | ---: | ---: | ---: |']
    for name in ['blank','numeric']:
        values=[100*summary[f'{name}_{c}_{b}']['validation']['accuracy'] for c,b in [(320,'original'),(512,'same_updates'),(512,'same_passes')]]
        label='Argument/history + zero-field control' if name=='blank' else 'Argument/history + numeric features'
        lines.append(f'| {label} | {values[0]:.3f}% | {values[1]:.3f}% | {values[2]:.3f}% |')
    lines+=['','Both models have 2,370 learned parameters and use the same frozen MiniLM/PCA argument vectors, observed history, actual inbox order, current probabilities and signed graph. The numeric arm additionally matches literal rational values in messages to the two supplied answer choices. The control receives zeros in those eight fields. No correct-answer key or future observation enters the inputs. The frozen 22,713,216-parameter encoder is additional to the update-rule count.', '',
        'The 512 cohort adds the same first192 new training questions in the collection’s frozen ordering. All original features and PCA transforms remain unchanged. Each seed starts from exactly its original320 initialization, with the same optimizer and regularization. One checkpoint holds optimizer steps fixed; another multiplies steps by512/320. There is no new tuning. All12 checkpoints froze before outer scoring. Values average three individual seeds rather than an ensemble.', '',
        '| Accuracy contrast | Change | Descriptive 95% question interval |','| --- | ---: | ---: |']
    for c in contrasts:
        if c['metric']=='error':
            a=' / '.join(map(str,c['new']));b=' / '.join(map(str,c['reference']))
            lines.append(f"| {a} vs {b} | {100*c['reduction']:+.3f} pp | [{100*c['ci95'][0]:+.3f}, {100*c['ci95'][1]:+.3f}] |")
    lines+=['','| Model / cohort / budget | Training accuracy | Validation accuracy |','| --- | ---: | ---: |']
    for name,z in summary.items():lines.append(f"| {name} | {100*z['train']['accuracy']:.3f}% | {100*z['validation']['accuracy']:.3f}% |")
    lines+=['','This follow-up was motivated after seeing the audited320 result and is exploratory. The same80 validation questions have been repeatedly reused; new validation and the40 reserved assessment questions remain unused. Revised topic sampling in the added questions limits a pure sample-count interpretation. These are all-case next-source-mode predictions on existing stochastic histories, with known graph families and personas.', '',
        f"All36 prediction files and checkpoint outputs replay exactly; all12 original320 prediction arrays remain unchanged. Target/question ordering, seed initialization, feature identity and all21 paired contrasts pass verification. Two focused checks pass. Maximum metric discrepancy is {metric_max:.2g}. Six new fits yielded12 fixed endpoints. No source calls were made.",'']
    Path('reports/modal_numeric_scaling_512.md').write_text('\n'.join(lines));print(audit,flush=True)


if __name__=='__main__':main()
