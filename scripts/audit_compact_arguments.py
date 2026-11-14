"""Independent statistics, input-isolation and checkpoint checks; no fitting."""
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import compact_argument_study as study
from scripts import breadth_argument_study as shared
from scripts import breadth_argument_features as features

def recompute(lp,y,qids):
    lp=lp.astype(np.float64)
    np.testing.assert_allclose(np.exp(lp).sum(-1),1,atol=2e-6)
    values={'nll':-np.take_along_axis(lp,y[...,None],-1)[...,0],
            'error':(lp.argmax(-1)!=y).astype(float),'brier_two_class':2*(np.exp(lp[...,1])-y)**2}
    q={m:{k:float(a[qids==k].mean()) for k in sorted(set(qids))} for m,a in values.items()}
    return {m:float(np.mean(list(a.values()))) for m,a in q.items()},q

def audit():
    cfg=study.config();root=Path(cfg['output']);p=study.validate(cfg);frozen=study.validate_frozen(cfg)
    for group in ('source_artifacts','prior_artifacts'):
        for file,sha in p[group].items():assert digest(file)==sha,file
    companion=shared.validate_frozen(features.config())
    encoding=study.read(root/'encoding_replay.json')
    assert encoding['status']=='passed' and encoding['samples']==14
    assert encoding['reference_encoding_sha256']==digest(Path(cfg['message_reference'])/'encoding.json')
    assert encoding['maximum_cpu_vs_gpu_component_difference']<5e-5
    results=study.read(root/'results/summary.json');index=study.read(root/'results/predictions.json')
    assert results['models_frozen_sha256']==digest(root/'models_frozen.json')
    assert results['companion_models_frozen_sha256']==digest(Path(cfg['message_reference'])/'models_frozen.json')
    common=dict(np.load(root/'data/development.npz',allow_pickle=False));allq={};largest=0.
    for record in index:
        assert digest(record['path'])==record['sha256']
        with np.load(record['path'],allow_pickle=False) as a:
            lp,y,qids,pairs=[a[k] for k in ('logp','target','qids','pairs')];e,t=pairs.T
            np.testing.assert_array_equal(common['spins'][e,t+1]==1,y)
            np.testing.assert_array_equal(common['spins'][e,t],a['current'])
            np.testing.assert_array_equal(common['episode_qids'][e],qids)
            assert set(qids)==set(p['fold'][record['split']+'_questions'])
            assert not set(qids)&set(p['fold']['assessment_questions'])
            actual,rows=recompute(lp,y,qids)
            expected=next(r for r in results['by_seed'] if all(r[k]==record[k] for k in ('model','seed','split')))
            for metric,value in actual.items():
                error=abs(expected[metric]-value);largest=max(largest,error);assert error<1e-12
            wrong=lp.argmax(-1)!=y;changed=(a['current']>0)!=y
            for label,mask in [('switch',changed),('stay',~changed)]:
                np.testing.assert_allclose(expected[label+'_error'],wrong[mask].mean(),atol=1e-12)
            allq[record['model'],record['seed'],record['split']]=rows
    for model,splits in results['summary'].items():
        for split,metrics in splits.items():
            for metric in ('nll','error','brier_two_class'):
                v=[np.mean(list(q[metric].values())) for (m,seed,s),q in allq.items() if m==model and s==split]
                np.testing.assert_allclose(metrics[metric],np.mean(v),atol=1e-12)
    qs=sorted(p['fold']['validation_questions']);rng=np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    draws=rng.integers(0,len(qs),(cfg['evaluation']['bootstrap_replicates'],len(qs)))
    def scores(model,metric):
        return np.mean([np.array([q[metric][i] for i in qs]) for (m,s,role),q in allq.items() if m==model and role=='validation'],axis=0)
    for row in results['paired_differences']:
        delta=scores(row['left'],row['metric'])-scores(row['right'],row['metric'])
        np.testing.assert_allclose(row['mean_difference'],delta.mean(),atol=1e-12)
        np.testing.assert_allclose(row['ci95'],np.quantile(delta[draws].mean(1),[.025,.975]),atol=1e-12)
    ids=np.load(root/'data/message_ids.npy',allow_pickle=False)
    fit=np.load(root/'data/message_pca.npz',allow_pickle=False)
    train_e=np.flatnonzero(np.isin(common['episode_qids'],p['fold']['train_questions']))
    expected=np.unique(ids[train_e]);expected=expected[expected>2]
    np.testing.assert_array_equal(fit['training_ids'],expected)
    vectors=np.load(Path(cfg['message_reference'])/'data/vectors.npy',allow_pickle=False)
    reduced=np.load(root/'data/vectors_pca8.npy',allow_pickle=False)
    centered=vectors[expected].astype(np.float64)-fit['mean']
    np.testing.assert_allclose(fit['mean'],vectors[expected].astype(np.float64).mean(0),atol=1e-14)
    np.testing.assert_allclose(fit['components']@fit['components'].T,np.eye(8),atol=1e-12)
    covariance=centered.T@centered/len(centered);values=np.linalg.eigvalsh(covariance)[-8:][::-1]
    np.testing.assert_allclose(np.diag(fit['components']@covariance@fit['components'].T),values,atol=1e-12)
    recalculated=((vectors.astype(np.float64)-fit['mean'])@fit['components'].T/(fit['scale']*np.sqrt(8))).astype(np.float32)
    recalculated[0]=0;np.testing.assert_array_equal(recalculated,reduced)
    replacement=np.load(root/'data/matched_substitution.npz',allow_pickle=False)
    for e in range(len(ids)):
        for sender in range(32):
            donors=replacement['donor_round'][e,:,sender]
            np.testing.assert_array_equal(replacement['message_ids'][e,:,sender],ids[e,donors,sender])
            np.testing.assert_array_equal(common['spins'][e,:8,sender],common['spins'][e,donors,sender])
            np.testing.assert_array_equal(np.sort(replacement['message_ids'][e,:,sender]),np.sort(ids[e,:,sender]))
    checks=[];torch.set_num_threads(4);data=study.Data(cfg,'cpu')
    selection=study.read(root/'selection.json')['choices']
    for directory in frozen['refits']:
        path=Path(directory);m=study.read(path/'manifest.json');width=m['width'];data.set_width(width);data.condition=m['condition']
        choice=next(r for r in selection if r['width']==width and r['condition']==m['condition'])
        assert m['candidate']==choice['candidate'] and m['selected_step']==choice['refit_steps']
        assert m['seed'] in cfg['training']['refit_seeds']
        model=study.make_model(cfg,width).eval();ck=torch.load(path/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
        assert shared.base.model_weights_sha(model)==m['selected_weights_sha256']
        train=data.ids(p['fold']['train_questions'])[:16];batch=data.batch(train)
        score=shared.score_model(model,batch);torch.nn.functional.binary_cross_entropy_with_logits(score,batch['target']).backward()
        gradient=float(model.text_projection.weight.grad.norm());assert gradient>0 and np.isfinite(gradient)
        initial=study.make_model(cfg,width,m['seed'])
        moved=float((model.text_projection.weight.detach()-initial.text_projection.weight.detach()).norm());assert moved>0
        val=data.ids(p['fold']['validation_questions']);positions=np.arange(0,len(val),31);chosen=val[positions]
        lp=shared.predictions(model,data,chosen,32)
        saved=np.load(root/'results'/f'{width}_{m["condition"]}-s{m["seed"]}-validation.npz',allow_pickle=False)['logp'][positions]
        delta=float(np.abs(np.exp(lp)-np.exp(saved)).max());assert delta<5e-5
        checks.append({'width':width,'condition':m['condition'],'seed':m['seed'],
                       'text_gradient_norm':gradient,'text_weight_change':moved,'cpu_probability_difference':delta,
                       'class_disagreements':int((lp.argmax(-1)!=saved.argmax(-1)).sum())})
    for width in cfg['models']:
        for seed in [cfg['training']['tuning_seed'],*cfg['training']['refit_seeds']]:
            phase='tune' if seed==cfg['training']['tuning_seed'] else 'refit'
            initials={study.read(path)['initial_weights_sha256'] for path in (root/phase).glob(f'{width}-*-s{seed}/manifest.json')}
            assert len(initials)==1
    audit={'status':'passed','completed_utc':shared.now(),'code_sha256':digest(__file__),
        'prediction_arrays_recomputed':len(index),'paired_contrasts_recomputed':len(results['paired_differences']),
        'maximum_metric_difference':largest,'source_episodes_verified':len(p['source_artifacts']),
        'prior_artifacts_verified':len(p['prior_artifacts']),'frozen_artifacts_verified':len(frozen['artifacts']),
        'companion_artifacts_verified':len(companion['artifacts']),'pca_train_only_verified':True,
        'substitution_donors_verified':True,'trained_message_branch_checks':checks,
        'independent_encoder_replay':encoding,'independent_encoder_replay_sha256':digest(root/'encoding_replay.json'),
        'results_sha256':{str(path):digest(path) for path in (root/'results').iterdir() if path.is_file()},
        'assessment_loaded':False,'source_llm_calls':0}
    json_write(root/'completion_audit.json',audit)
    print({k:v for k,v in audit.items() if not isinstance(v,(list,dict))})

if __name__=='__main__':audit()
