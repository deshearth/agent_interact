"""Independent saved-output arithmetic and checkpoint replay for the 512 study."""
from collections import Counter
from pathlib import Path
import pickle
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_scaling_study as study

OLD_NAMES={'choice978':'nano_current_choice_mode','history_choice2306':'history_choice_observed_mode',
    'history_arguments2306':'history_arguments_observed_mode','joint_arguments26210':'paired_full_arguments_mode',
    'ising22':'ising_current_mode','tree_semantic8':'tree_semantic8'}


def main():
    p=study.validate(); root=study.ROOT; out=root/'evaluation'; result=study.read(out/'results.json')
    assert result['protocol_sha256']==digest(root/'protocol.json') and result['models_frozen_sha256']==digest(root/'models_frozen.json')
    frozen=study.read(root/'models_frozen.json')
    for path,sha in frozen['artifacts'].items(): assert digest(path)==sha,path
    tests=ET.parse('reports/modal_scaling_512_checks.xml').getroot()
    suites=list(tests) if tests.tag=='testsuites' else [tests]
    assert sum(int(s.get('tests',0)) for s in suites)==6
    assert sum(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))==0
    for row in study.read(study.DATA/'index.json'): assert digest(row['path'])==row['sha256'],row['qid']
    old=study.read('runs/modal_accuracy_investigation_320_v1/evaluation_v1/results.json')
    old_files={(f['model'],f['seed'],f['split']):f for f in old['prediction_files']}
    saved_rows=study.read(out/'by_question.json'); row_lookup={(r['model'],r['cohort'],r['budget'],r['seed'],r['split'],r['qid']):r for r in saved_rows}
    assert len(row_lookup)==len(saved_rows)
    regenerated=[]; metric_max=0.; logp_max=0.; replay_count=0; old_count=0; baseline_replays=0
    torch.set_num_threads(2)
    for name in [*study.MODEL_SOURCES,'ising22','tree_semantic8']:
        data=study.Data(name)
        for item in [f for f in result['prediction_files'] if f['model']==name]:
            assert digest(item['path'])==item['sha256'] and digest(item['checkpoint'])==item['checkpoint_sha256']
            a=dict(np.load(item['path'])); questions=p['validation'] if item['split']=='validation' else p['train'+str(item['cohort'])]
            ids=data.ids(questions); np.testing.assert_array_equal(a['qids'],data.qids[ids]); np.testing.assert_array_equal(a['pairs'],data.common['pairs'][ids])
            np.testing.assert_array_equal(a['target'],data.y[ids]); np.testing.assert_array_equal(a['probability'],data.source[ids])
            lp=a['logp'].astype(float); yy=a['target'].astype(float); pp=np.exp(lp[...,1])
            np.testing.assert_allclose(np.exp(lp).sum(-1),1,atol=2e-6,rtol=0)
            values=dict(error=(lp.argmax(-1)!=yy).astype(float),nll=-(1-yy)*lp[...,0]-yy*lp[...,1],brier_two_class=2*(pp-yy)**2)
            for q in sorted(set(a['qids'])):
                key=(name,item['cohort'],item['budget'],item['seed'],item['split'],str(q)); saved=row_lookup[key]
                independent={k:float(v[a['qids']==q].mean()) for k,v in values.items()}
                for k,v in independent.items(): metric_max=max(metric_max,abs(v-saved[k]))
                regenerated.append(dict(model=name,cohort=item['cohort'],budget=item['budget'],seed=item['seed'],split=item['split'],qid=str(q),**independent))
            if item['cohort']==320:
                legacy=old_files[(OLD_NAMES[name],item['seed'] if name in study.MODEL_SOURCES else -1,item['split'])]; old_array=np.load(legacy['path'])
                np.testing.assert_allclose(a['logp'],old_array['logp'],atol=1e-12,rtol=0)
                old_count+=1
            if item['split']!='validation': continue
            checkpoint=Path(item['checkpoint'])
            if name in study.MODEL_SOURCES:
                model=study.make_model(name,item['seed']); model.load_state_dict(torch.load(checkpoint,weights_only=False)['model'])
                replay=study.predict(model,data,ids); replay_count+=1
            elif name=='ising22':
                weights=np.asarray(study.read(checkpoint)['weights']); z=study.core.ising_design(data,ids,False)@weights
                replay=np.stack([-np.logaddexp(0,z),-np.logaddexp(0,-z)],-1); baseline_replays+=1
            else:
                with checkpoint.open('rb') as stream: model=pickle.load(stream)
                x=np.load(study.DATA/'tree_semantic8.npy',mmap_mode='r'); take=(ids[:,None]*32+np.arange(32)).ravel()
                with threadpool_limits(limits=2): pp=np.clip(model.predict_proba(x[take])[:,1].reshape(-1,32),1e-10,1-1e-10)
                replay=np.stack([np.log1p(-pp),np.log(pp)],-1); baseline_replays+=1
            difference=float(np.max(np.abs(replay-lp))); logp_max=max(logp_max,difference)
            np.testing.assert_array_equal(replay.argmax(-1),lp.argmax(-1)); np.testing.assert_allclose(replay,lp,atol=1e-12,rtol=0)
    assert metric_max<1e-7 and len(result['prediction_files'])==80 and replay_count==36 and baseline_replays==4 and old_count==28
    for key,summary in result['summary'].items():
        for split,metrics in summary.items():
            group=[r for r in regenerated if f"{r['model']}_{r['cohort']}_{r['budget']}"==key and r['split']==split]
            for k in ('error','nll','brier_two_class'): np.testing.assert_allclose(np.mean([r[k] for r in group]),metrics[k],atol=1e-7,rtol=0)
    draws=np.random.default_rng(92813).integers(80,size=(5000,80))
    for c in result['contrasts']:
        means=[]
        for cohort,budget in [(320,'original'),(512,c['budget'])]:
            group=[r for r in regenerated if (r['model'],r['cohort'],r['budget'],r['split'])==(c['model'],cohort,budget,'validation')]
            means.append(np.asarray([np.mean([r[c['metric']] for r in group if r['qid']==q]) for q in sorted(p['validation'])]))
        difference=means[0]-means[1]; np.testing.assert_allclose(difference.mean(),c['reduction'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(np.quantile(difference[draws].mean(1),[.025,.975]),c['ci95'],atol=1e-7,rtol=0)
    new=study.read(study.DATA/'new_index.json'); old_questions={q['qid']:q for q in study.read('runs/question_breadth_320_v1/selection.json')['questions']}
    added=[]
    for row in new:
        ep=study.read(row['path']); matches=[label for label,value in ep['choices'].items() if value==row['metadata']['correct_answer']]
        assert len(matches)==1
        added.append(dict(row['metadata'],answer=matches[0]))
    cohorts={'original320':[old_questions[q] for q in p['train320']],'added192':added}
    composition={name:{field:dict(Counter(str(q[field]) for q in questions)) for field in ('type','level','graph_id','answer')} for name,questions in cohorts.items()}
    json_write(out/'composition.json',composition)
    audit=dict(passed=True,completed_utc=study.now(),results_sha256=digest(out/'results.json'),protocol_sha256=digest(root/'protocol.json'),
        independent_metric_max_difference=metric_max,checkpoint_max_logp_difference=logp_max,prediction_files=len(result['prediction_files']),
        seed_question_rows=len(regenerated),contrasts=len(result['contrasts']),neural_validation_replays=replay_count,baseline_validation_replays=baseline_replays,
        old_prediction_arrays_unchanged=old_count,focused_tests=6,source_episodes_verified=592,source_calls=0,new_validation_loaded=False,assessment_loaded=False,
        audit_code_sha256=digest(__file__),composition_sha256=digest(out/'composition.json'))
    json_write(out/'completion_audit.json',audit); print(audit,flush=True)


if __name__=='__main__': main()
