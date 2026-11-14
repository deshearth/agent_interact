"""Recompute saved results and verify source/checkpoint integrity without fitting."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import breadth_argument_features as features
from scripts import breadth_argument_study as study

def read(path):return json.loads(Path(path).read_text())

def recompute(logp,target,qids):
    assert np.isfinite(logp).all()
    np.testing.assert_allclose(np.exp(logp).sum(-1),1,atol=2e-6)
    arrays={'nll':-np.take_along_axis(logp.astype(np.float64),target[...,None],-1)[...,0],
            'error':(logp.argmax(-1)!=target).astype(float),
            'brier_two_class':2*(np.exp(logp.astype(np.float64)[...,1])-target)**2}
    grouped={metric:{q:float(v[qids==q].mean()) for q in sorted(set(qids))} for metric,v in arrays.items()}
    return {m:float(np.mean(list(v.values()))) for m,v in grouped.items()},grouped

def audit():
    cfg=features.config();root=Path(cfg['output']);protocol=features.validate(cfg,True);frozen=study.validate_frozen(cfg)
    for group in ('source_artifacts','prior_artifacts'):
        for path,sha in protocol[group].items():assert digest(path)==sha,path
    results=read(root/'results/summary.json');index=read(root/'results/predictions.json')
    encoding_check=read(root/'encoding_audit.json');geometry=read(root/'representation_audit.json')
    vector_sha=digest(root/'data/vectors.npy')
    assert encoding_check['status']=='passed' and encoding_check['vectors_sha256']==vector_sha
    assert geometry['cache_sha256']==vector_sha
    assert results['models_frozen_sha256']==digest(root/'models_frozen.json')
    assert not results['assessment_loaded'] and results['source_llm_calls']==0
    common=dict(np.load(root/'data/development.npz',allow_pickle=False));grouped={};largest=0.
    for record in index:
        path=Path(record['path']);assert digest(path)==record['sha256']
        with np.load(path,allow_pickle=False) as data:
            lp,y,qids,pairs=[data[k] for k in ('logp','target','qids','pairs')]
            e,t=pairs.T
            np.testing.assert_array_equal(common['episode_qids'][e],qids)
            np.testing.assert_array_equal((common['spins'][e,t+1]==1).astype(int),y)
            np.testing.assert_array_equal(common['spins'][e,t],data['current'])
            expected=set(protocol['fold'][record['split']+'_questions'])
            assert set(qids)==expected and not expected & set(protocol['fold']['assessment_questions'])
            actual,questions=recompute(lp,y,qids)
            stored=next(r for r in results['by_seed'] if all(r[k]==record[k] for k in ('model','seed','split')))
            for metric,value in actual.items():
                discrepancy=abs(value-stored[metric]);largest=max(largest,discrepancy);assert discrepancy<1e-12
            grouped[(record['model'],record['seed'],record['split'])]=questions
            changed=(data['current']>0)!=y;wrong=lp.argmax(-1)!=y
            for label,mask in [('switch',changed),('stay',~changed)]:
                np.testing.assert_allclose(stored[label+'_error'],wrong[mask].mean(),atol=1e-12)
    for model,split_scores in results['summary'].items():
        for split,stored in split_scores.items():
            for metric in ('nll','error','brier_two_class'):
                values=[np.mean(list(q[metric].values())) for (m,s,role),q in grouped.items() if m==model and role==split]
                np.testing.assert_allclose(np.mean(values),stored[metric],atol=1e-12)
    for diagnostic in results['message_replacement_diagnostic']:
        stem=f"{diagnostic['model']}-s{diagnostic['seed']}"
        with np.load(root/'results'/f'{stem}-validation.npz') as original, np.load(root/'results'/f'{stem}-conclusion_diagnostic.npz') as replaced:
            np.testing.assert_array_equal(original['target'],replaced['target'])
            np.testing.assert_array_equal(original['qids'],replaced['qids'])
            scores,_=recompute(replaced['logp'],replaced['target'],replaced['qids'])
            for metric,value in scores.items():
                np.testing.assert_allclose(value,diagnostic['replacement_validation'][metric],atol=1e-12)
            difference=np.abs(np.exp(original['logp'][...,1])-np.exp(replaced['logp'][...,1])).mean()
            disagreement=(original['logp'].argmax(-1)!=replaced['logp'].argmax(-1)).mean()
            np.testing.assert_allclose(difference,diagnostic['original_vs_conclusion_probability_mae'],atol=1e-12)
            np.testing.assert_allclose(disagreement,diagnostic['class_disagreement'],atol=1e-12)
    questions=sorted(protocol['fold']['validation_questions'])
    rng=np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    draws=rng.integers(0,len(questions),(cfg['evaluation']['bootstrap_replicates'],len(questions)))
    def per_question(model,metric):
        arrays=[np.array([values[metric][q] for q in questions]) for (m,s,split),values in grouped.items() if m==model and split=='validation']
        return np.mean(arrays,axis=0)
    for row in results['paired_differences']:
        delta=per_question(row['left'],row['metric'])-per_question(row['right'],row['metric'])
        np.testing.assert_allclose(row['mean_difference'],delta.mean(),atol=1e-12)
        np.testing.assert_allclose(row['ci95'],np.quantile(delta[draws].mean(1),[.025,.975]),atol=1e-12)
    selections=read(root/'selection.json')['choices'];gradient_checks=[];backend_checks=[]
    data=study.Data(cfg,'cpu');torch.set_num_threads(4)
    for directory in frozen['refits']:
        path=Path(directory);manifest=read(path/'manifest.json')
        selection=next(x for x in selections if all(x[k]==manifest[k] for k in ('width','condition')))
        assert manifest['candidate']==selection['candidate'] and manifest['selected_step']==selection['refit_steps']
        assert manifest['seed'] in cfg['training']['refit_seeds'] and manifest['seed']!=cfg['training']['tuning_seed']
        model=study.make_model(cfg,manifest['width']).eval()
        ck=torch.load(path/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
        assert study.base.model_weights_sha(model)==manifest['selected_weights_sha256']
        data.condition=manifest['condition'];train=data.ids(protocol['fold']['train_questions'])[:16]
        batch=data.batch(train);score=study.score_model(model,batch)
        torch.nn.functional.binary_cross_entropy_with_logits(score,batch['target']).backward()
        gradient=float(model.text_projection.weight.grad.norm())
        assert np.isfinite(gradient) and gradient>0
        initial=study.make_model(cfg,manifest['width'],manifest['seed'])
        change=float((model.text_projection.weight.detach()-initial.text_projection.weight.detach()).norm())
        assert change>0
        gradient_checks.append({'model':manifest['width']+'_'+manifest['condition'],'seed':manifest['seed'],
                                'text_projection_gradient_norm':gradient,'text_projection_weight_change':change})
        ids=data.ids(protocol['fold']['validation_questions'])[::31][:16]
        actual=study.predictions(model,data,ids,16)
        saved=np.load(root/'results'/f'{manifest["width"]}_{manifest["condition"]}-s{manifest["seed"]}-validation.npz')
        val=data.ids(protocol['fold']['validation_questions']);positions=np.flatnonzero(np.isin(val,ids))
        expected=saved['logp'][positions];maxdiff=float(np.abs(np.exp(actual)-np.exp(expected)).max())
        assert maxdiff<5e-5
        backend_checks.append({'model':manifest['width']+'_'+manifest['condition'],'seed':manifest['seed'],
                               'maximum_probability_difference':maxdiff,'class_disagreements':int((actual.argmax(-1)!=expected.argmax(-1)).sum())})
    for width in cfg['models']:
        for seed in [cfg['training']['tuning_seed'],*cfg['training']['refit_seeds']]:
            phase='tune' if seed==cfg['training']['tuning_seed'] else 'refit'
            initials={read(p)['initial_weights_sha256'] for p in (root/phase).glob(f'{width}-*-s{seed}/manifest.json')}
            assert len(initials)==1
    artifacts={str(p):digest(p) for p in (root/'results').iterdir() if p.is_file()}
    result={'status':'passed','completed_utc':study.now(),'prediction_arrays_recomputed':len(index),
            'replacement_diagnostics_recomputed':len(results['message_replacement_diagnostic']),
            'paired_contrasts_recomputed':len(results['paired_differences']),'maximum_metric_difference':largest,
            'frozen_artifacts_verified':len(frozen['artifacts']),'prior_artifacts_verified':len(protocol['prior_artifacts']),
            'source_episodes_verified':len(protocol['source_artifacts']),'text_branch_checks':gradient_checks,
            'encoding_audit_sha256':digest(root/'encoding_audit.json'),
            'representation_audit_sha256':digest(root/'representation_audit.json'),
            'independent_cpu_replay':backend_checks,'assessment_loaded':False,'source_llm_calls':0,
            'results_artifacts':artifacts,'audit_code_sha256':digest(__file__)}
    json_write(root/'completion_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(dict,list))},indent=2))

if __name__=='__main__':audit()
