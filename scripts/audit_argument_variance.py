"""Independently check the fixed-recipe experiment and its saved predictions."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import argument_variance_diagnosis as study
from scripts import compact_argument_study as compact

def audit():
    cfg=study.config();root=Path(cfg['output']);protocol=study.validate(cfg,True)
    summary=study.read(root/'results/summary.json');frozen=study.read(root/'models_frozen.json')
    old=Path(cfg['reference']);data=compact.Data(cfg,'cpu');torch.set_num_threads(4)
    index=study.read(root/'results/predictions.json')+study.read(old/'results/predictions.json')
    differences=[];loaded={};checked=0
    for row in summary['models']:
        ids=data.ids(protocol['fold'][row['split']+'_questions']);records=sorted([r for r in index
            if r['model']==row['model'] and r['split']==row['split']],key=lambda r:r['seed'])
        lps=[]
        for record in records:
            assert digest(record['path'])==record['sha256'];z=np.load(record['path'],allow_pickle=False)
            np.testing.assert_array_equal(z['target'],data.y[ids]);np.testing.assert_array_equal(z['pairs'],data.common['pairs'][ids])
            lp=z['logp'].astype(np.float64);np.testing.assert_allclose(np.exp(lp).sum(-1),1,atol=2e-7)
            lps.append(lp);checked+=1
        lp=np.stack(lps);y=data.y[ids];pred=(lp[...,1]>lp[...,0]).astype(int)
        independent={'error':float(np.mean(pred!=y)),
            'nll':float(np.mean(-np.where(y,lp[...,1],lp[...,0]))),
            'brier_two_class':float(np.mean(2*(np.exp(lp[...,1])-y)**2))}
        for key,value in independent.items():
            difference=abs(value-row[key]);assert difference<1e-7,(row['model'],key,difference);differences.append(difference)
        e,t=data.common['pairs'][ids].T;votes=data.common['raw_votes'][e,t+1]
        counts=(votes==1).sum(-1);agreement=np.maximum(counts,5-counts);errors=pred!=y
        for k in (3,4,5):
            mask=agreement==k
            np.testing.assert_allclose([mask.mean(),errors[:,mask].mean()],
                [row['vote_agreement'][str(k)]['fraction'],row['vote_agreement'][str(k)]['error']],atol=1e-12)
        assert abs(errors[:,agreement<5].sum()/errors.sum()-row['error_fraction_on_split_votes'])<1e-12
        loaded[row['split'],row['model']]=(lp,y,data.qids[ids])
    contrasts=0
    for row in summary['comparisons']:
        a,y,qids=loaded[row['split'],row['left']];b,yb,qb=loaded[row['split'],row['right']]
        np.testing.assert_array_equal(y,yb);np.testing.assert_array_equal(qids,qb)
        ea,eb=np.broadcast_arrays(a.argmax(-1)!=y,b.argmax(-1)!=y)
        corrections=(eb&~ea).mean();introduced=(~eb&ea).mean();common=(ea&eb).mean()
        for key,value in [('corrected',corrections),('introduced',introduced),('shared_errors',common),('disagreement',(ea!=eb).mean())]:
            assert abs(value-row[key])<1e-12,key
        assert abs((eb.mean()-ea.mean())-(corrections-introduced))<1e-12
        if row['split']=='validation':
            qs=sorted(set(qids));rng=np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
            draws=rng.integers(0,len(qs),(cfg['evaluation']['bootstrap_replicates'],len(qs)))
            for metric in ('error','nll','brier_two_class'):
                def loss(lp):
                    if metric=='error':return (lp.argmax(-1)!=y).astype(float)
                    if metric=='nll':return -np.where(y,lp[...,1],lp[...,0])
                    return 2*(np.exp(lp[...,1])-y)**2
                la,lb=loss(a),loss(b)
                d=np.array([la[:,qids==q].mean()-lb[:,qids==q].mean() for q in qs])
                record=row['paired'][metric]
                np.testing.assert_allclose([d.mean(),*np.quantile(d[draws].mean(1),[.025,.975])],
                    [record['mean_difference'],*record['ci95']],atol=1e-7)
                contrasts+=1
    fit=np.load(root/'data/message_pca.npz',allow_pickle=False)
    train=np.flatnonzero(np.isin(data.common['episode_qids'],protocol['fold']['train_questions']))
    groups=[np.unique(ids[ids>2]) for ids in data.message_ids[train]]
    all_ids=np.unique(np.concatenate(groups));np.testing.assert_array_equal(fit['training_ids'],all_ids)
    membership=np.bincount(np.concatenate(groups),minlength=len(data.full_vectors))
    direct=np.zeros((384,384))
    for group in groups:
        x=np.asarray(data.full_vectors[group],dtype=np.float64);w=1/membership[group]
        mu=np.average(x,axis=0,weights=w);residual=x-mu
        direct+=residual.T@(w[:,None]*residual)
    direct/=len(all_ids)
    np.testing.assert_allclose(direct,fit['within_covariance'],atol=2e-15)
    eigenvalues=np.linalg.eigvalsh(direct);axes=fit['components']
    np.testing.assert_allclose(axes@axes.T,np.eye(8),atol=3e-14)
    np.testing.assert_allclose(np.diag(axes@direct@axes.T),eigenvalues[-8:][::-1],atol=2e-15)
    expected=((np.asarray(data.full_vectors,dtype=np.float64)-fit['mean'])@axes.T/(float(fit['scale'])*np.sqrt(8))).astype(np.float32)
    expected[0]=0;np.testing.assert_array_equal(data.pca_vectors,expected)
    np.testing.assert_allclose((expected[all_ids]**2).mean(),1/8,atol=1e-7)
    replay=[]
    for path in frozen['refits']:
        path=Path(path);m=study.read(path/'manifest.json');prior=study.read(old/'refit'/path.name/'manifest.json')
        assert m['initial_weights_sha256']==prior['initial_weights_sha256']
        assert m['candidate_settings']==prior['candidate_settings']
        assert m['selected_step']==prior['selected_step']==m['fixed_steps']
        assert m['parameters']==882
        model=compact.make_model(cfg,'nano_pca8',m['seed']);initial=model.text_projection.weight.detach().clone()
        model.load_state_dict(torch.load(path/'best.pt',map_location='cpu',weights_only=False)['model']);model.eval()
        assert (model.text_projection.weight-initial).abs().max()>0
        ids=data.ids(protocol['fold']['validation_questions']);batch=data.batch(ids[:5],m['condition'])
        score=model(batch['s'],batch['context'],batch['J'],batch['text']);p=torch.softmax(score,-1).detach().numpy()
        stored=np.load(root/'results'/f'within_pca8_{m["condition"]}-s{m["seed"]}-validation.npz')['logp'][:5]
        diff=float(np.max(np.abs(p-np.exp(stored))))
        assert diff<2e-6
        torch.nn.functional.cross_entropy(score.reshape(-1,2),batch['target'].long().flatten()).backward()
        assert model.text_projection.weight.grad.norm()>0
        replay.append({'condition':m['condition'],'seed':m['seed'],'max_probability_difference':diff,
            'class_disagreements':int((p.argmax(-1)!=stored.argmax(-1)).sum()),
            'text_projection_gradient_norm':float(model.text_projection.weight.grad.norm())})
    # Establish that the saved probability anchor describes these exact source episodes.
    source_index=study.read('runs/question_breadth_320_v1/episode_index.json');source={r['qid']:r for r in source_index}
    original_questions=study.read('runs/question_breadth_v1/selection.json')['questions']
    anchor_qids={q['qid'] for q in original_questions if q['split'] in ('train','validation')}
    matched=0
    for q in anchor_qids:
        r=source[q];original=Path('runs/question_breadth_v1/episodes')/r['split']/f'{q}__{r["graph_id"]}__r00.json'
        a=study.read(original);b=study.read(r['path'])
        for key in ('J','choices','messages_history','spins_history','spins_raw_history','statement','personas'):
            assert a[key]==b[key],(q,key)
        matched+=1
    assert matched==100
    p,quality=study.anchor_probabilities(data.common)
    for split in ('train','validation'):
        ids=data.ids(protocol['fold'][split+'_questions']);e,t=data.common['pairs'][ids].T;v=p[e,t+1];take=np.isfinite(v).all(1)
        v=v[take];q=10*v**3*(1-v)**2+5*v**4*(1-v)+v**5
        np.testing.assert_allclose(np.minimum(q,1-q).mean(),summary['anchor'][split]['sampling_reference_error'],atol=2e-15)
        for name,scores in summary['anchor'][split]['models'].items():
            lp,y,qids=loaded[split,name];lp=lp[:,take];y=y[take];pred=lp.argmax(-1)
            checks={'expected_error':np.where(pred,1-q,q).mean(),
                'expected_nll':(-(1-q)*lp[...,0]-q*lp[...,1]).mean(),
                'probability_mse':((np.exp(lp[...,1])-q)**2).mean()}
            for key,value in checks.items():np.testing.assert_allclose(value,scores[key],atol=1e-7)
    study.validate(cfg,True)
    result={'status':'passed','completed_utc':study.shared.now(),'prediction_arrays_recomputed':checked,
        'paired_contrasts_recomputed':contrasts,'maximum_metric_difference':max(differences),
        'new_frozen_artifacts_verified':len(frozen['artifacts']),'prior_inputs_verified':len(protocol['inputs']),
        'training_unique_texts':len(all_ids),'pca_covariance_directly_verified':True,'source_anchor_episodes_verified':matched,
        'model_replay':replay,'assessment_loaded':False,'source_llm_calls':0,
        'results_sha256':{str(p):digest(p) for p in (root/'results').iterdir() if p.is_file()},
        'audit_code_sha256':digest(__file__)}
    json_write(root/'completion_audit.json',result)
    print({'status':'passed','arrays':checked,'contrasts':contrasts,'max_metric_difference':max(differences)})

if __name__=='__main__':audit()
