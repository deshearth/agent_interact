"""Interpret frozen 818-parameter predictors with small explicit update rules.

This is an exploratory surrogate study on the old twenty-question GPT pool.
It never opens the new Llama study's conversations or fits a source LLM.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, xlogy
from threadpoolctl import threadpool_limits
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, config, digest, json_write
from scripts import transformer_generalization as base

CONFIG=Path('configs/transformer_rule_distillation.yaml')
REPORT=Path('reports/transformer_rule_distillation.md')
LABELS={'ising_own':'Ising score + own opinion','pooled':'Pooled linear rule',
        'contextual':'Context-dependent influence','quadratic':'Context-dependent + quadratic influence',
        'teacher':'Transformer 818','I0':'Original Ising','I0_current':'Ising + observed-label own-opinion correction'}


def now(): return datetime.now(timezone.utc).isoformat()


def log_prob(score):
    score=np.asarray(score,dtype=np.float64)
    return np.stack([-np.logaddexp(0,score),-np.logaddexp(0,-score)],-1)


def features(s,phi,graph,prior,family):
    """Only current opinions and supplied context/graph are accepted here."""
    s=np.asarray(s,dtype=np.float64); phi=np.asarray(phi,dtype=np.float64)
    pos=(graph>0).astype(np.float64); neg=(graph<0).astype(np.float64)
    friendly=np.einsum('bij,bj->bi',pos,s)
    unfriendly=np.einsum('bij,bj->bi',neg,s)
    # Unfriendly is the raw opinion sum on negative edges; its coefficient
    # carries its sign. This differs from the negative-signed Ising channel.
    if family=='ising_own':
        return np.stack([np.ones_like(s),prior,s],-1),['intercept','ising_score','own_opinion']
    names=['intercept']+[f'persona_{i}' for i in range(3)]+[f'question_{i}' for i in range(3)]
    names += [f'persona_{i}*question_{j}' for i in range(3) for j in range(3)]
    columns=[phi,s[...,None],friendly[...,None],unfriendly[...,None],pos.sum(-1)[...,None],neg.sum(-1)[...,None]]
    names += ['own_opinion','friendly_sum','unfriendly_sum','friendly_degree','unfriendly_degree']
    if family in ['contextual','quadratic']:
        context=phi[...,1:7]
        for label,signal in [('own_opinion',s),('friendly_sum',friendly),('unfriendly_sum',unfriendly)]:
            columns.append(signal[...,None]*context)
            names += [f'{label}*{name}' for name in names[1:7]]
    if family=='quadratic':
        columns += [friendly[...,None]**2,(friendly*unfriendly)[...,None],unfriendly[...,None]**2]
        names += ['friendly_sum^2','friendly_sum*unfriendly_sum','unfriendly_sum^2']
    if family not in ['pooled','contextual','quadratic']: raise ValueError(family)
    x=np.concatenate(columns,-1)
    assert x.shape[-1]==len(names) and np.all(x[...,0]==1)
    return x,names


def equal_question_weights(qids,agents):
    names,indices,counts=np.unique(qids,return_inverse=True,return_counts=True)
    return np.repeat(1/(len(names)*counts[indices]*agents),agents)


def standardize(x,weights):
    mean=weights@x
    scale=np.sqrt(weights@((x-mean)**2))
    mean[0]=0.; scale[0]=1.
    scale[scale<1e-10]=1.
    return (x-mean)/scale,mean,scale


def objective(beta,x,p,weights,ridge):
    z=x@beta
    value=weights@(np.logaddexp(0,z)-p*z)+.5*ridge*np.sum(beta[1:]**2)
    gradient=x.T@(weights*(expit(z)-p))
    gradient[1:]+=ridge*beta[1:]
    return float(value),gradient


def fit_rule(x,teacher_logp,qids,cfg):
    shape=x.shape; weights=equal_question_weights(qids,shape[1])
    x=np.asarray(x,dtype=np.float64).reshape(-1,shape[-1])
    target=np.exp(np.asarray(teacher_logp,dtype=np.float64)[...,1]).reshape(-1).clip(1e-12,1-1e-12)
    assert len(target)==len(x) and np.isclose(weights.sum(),1.)
    standardized,mean,scale=standardize(x,weights)
    initial=np.zeros(x.shape[1]); p=float(weights@target); initial[0]=np.log(p/(1-p))
    started=time.monotonic()
    result=minimize(objective,initial,args=(standardized,target,weights,cfg['ridge']),jac=True,
                    method='L-BFGS-B',options={'maxiter':cfg['max_iterations'],
                    'gtol':cfg['gradient_tolerance'],'ftol':cfg['function_tolerance'],'maxls':40})
    maximum=float(np.abs(result.jac).max())
    if not result.success or maximum>cfg['maximum_final_gradient']:
        raise RuntimeError(f'Convex fit did not converge: {result.message}; gradient={maximum}')
    raw=result.x/scale; raw[0]-=float(mean@raw)
    np.testing.assert_allclose(standardized@result.x,x@raw,atol=1e-10,rtol=1e-10)
    return {'weights':result.x.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),
            'raw_weights':raw.tolist(),'coefficients':len(raw),'objective':float(result.fun),
            'iterations':int(result.nit),'gradient_maximum':maximum,'optimizer_success':bool(result.success),
            'seconds':time.monotonic()-started}


def predict(rule,x):
    score=((x-np.asarray(rule['mean']))/np.asarray(rule['scale']))@np.asarray(rule['weights'])
    raw=x@np.asarray(rule['raw_weights'])
    np.testing.assert_allclose(score,raw,atol=1e-10,rtol=1e-10)
    return log_prob(score)


def fidelity(lp,teacher_lp,qids):
    p=np.exp(np.asarray(teacher_lp,dtype=np.float64)[...,1]).clip(1e-12,1-1e-12)
    q=np.exp(lp[...,1])
    kl=xlogy(p,p)+xlogy(1-p,1-p)-p*lp[...,1]-(1-p)*lp[...,0]
    assert kl.min()>-1e-10
    rows=[]
    for qid in sorted(set(qids)):
        mask=qids==qid; delta=q[mask]-p[mask]
        rows.append({'qid':str(qid),'probability_mae':float(np.abs(delta).mean()),
            'probability_rmse':float(np.sqrt(np.mean(delta**2))),
            'probability_error_p95':float(np.quantile(np.abs(delta),.95)),
            'teacher_kl':float(kl[mask].mean()),
            'teacher_label_disagreement':float(((q[mask]>.5)!=(p[mask]>.5)).mean())})
    return rows


def source_runs(cfg):
    root=Path(cfg['source']); result=[]
    for p in sorted((root/'refit').glob('*/*/manifest.json')):
        row=json.loads(p.read_text())
        if row.get('model')==cfg['teacher']:
            assert row['status']=='complete' and row['parameters']==cfg['teacher_parameters']
            result.append((p.parent,row))
    assert len(result)==12
    return result


def prepare(cfg):
    root,source=Path(cfg['output']),Path(cfg['source']); root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists(): validate(cfg); return
    original=json.loads((source/'protocol.json').read_text())
    assert original['core_sha256']==code_digest()==base.CORE_SHA
    assert original['config_sha256']==digest(cfg['source_config'])
    index=json.loads((source/'episode_index.json').read_text())
    assert all(x['official_split']=='train' and x['graph_role']=='fit_graph' for x in index)
    assert len({x['qid'] for x in index})==20
    paths=[Path(__file__),CONFIG,Path(base.__file__),Path(cfg['source_config'])]
    paths += [source/p for p in ['protocol.json','development.npz','episode_index.json','outer_by_question.csv','refit_complete.json','selection.json']]
    teachers=[]
    for directory,record in source_runs(cfg):
        for name,sha in record['artifacts'].items(): assert digest(directory/name)==sha
        assessed=source/'assessment'/f"{record['fold']}__{directory.name}"
        completed=json.loads((assessed/'complete.json').read_text())
        assert completed['run_manifest_sha256']==digest(directory/'manifest.json')
        assert digest(assessed/'assessment_logp.npy')==completed['artifacts']['assessment_logp.npy']
        paths += [directory/p for p in ['manifest.json','best.pt','train_logp.npy']]
        paths += [assessed/'complete.json',assessed/'assessment_logp.npy']
        teachers.append({'directory':str(directory),'assessment_directory':str(assessed),
                         'outer':record['outer'],'fold':record['fold'],'seed':record['seed']})
    folds=[f for f in original['folds'] if f['phase']=='refit']
    assert len(folds)==4
    for f in folds:
        paths += [source/'folds'/f['id']/p for p in ['features.npz','ising.json','manifest.json']]
        assert len(f['train_questions'])==15 and len(f['assessment_questions'])==5
        assert not set(f['train_questions'])&set(f['assessment_questions'])
    active=Path('runs/question_breadth_v1/source_protocol.json')
    frozen=json.loads(active.read_text())
    active_files={**frozen['source_files'],str(active):digest(active)}
    for p,sha in active_files.items(): assert digest(p)==sha,p
    # The map contains only old immutable artifacts, not new conversation data.
    prior=json.loads(Path('runs/question_breadth_v1/prior_artifacts.json').read_text())
    for p,sha in prior.items(): assert digest(p)==sha,p
    json_write(root/'prior_artifacts.json',prior)
    json_write(root/'protocol.json',{'created_utc':now(),'config':cfg,'core_sha256':code_digest(),
        'source_files':{str(p):digest(p) for p in paths},'active_study_frozen_files':active_files,
        'prior_artifacts_sha256':digest(root/'prior_artifacts.json'),'teachers':teachers,'folds':folds,
        'families':{'ising_own':3,'pooled':21,'contextual':39,'quadratic':42},
        'ising_own_count_note':'Three new fitted coefficients retain the original 19-coefficient Ising score; the other families are standalone rules over shared context.',
        'target_policy':'Imitate each frozen transformer probability on its 15 fitting questions. No observed target label enters the optimizer.',
        'input_policy':'Receiver PCA3 persona/question context and products, current own opinion, positive/negative-edge raw opinion sums and degree. Contextual adds six context components times own/friendly/unfriendly signals; quadratic adds three neighbor-field squares/products.',
        'selection_policy':'Fixed four formula families and one regularization strength; report all. Freeze all 48 fits before evaluating on the corresponding five old outer questions.',
        'assessment_policy':'Previously studied 20-question GPT development pool only. Equal question means, then seed-score averages. Paired intervals are exploratory, not fresh confirmation.',
        'interpretation_limit':'Surrogate agreement describes these predictor functions on observed states, not causal LLM influence or a universal dynamical law.',
        'source_llm_calls':0,'new_assessment_data_read':False,'device':'cpu','cpu_threads':1})
    print(json.dumps({'prepared':True,'teachers':12,'formula_fits':48,'new_assessment_data_read':False}),flush=True)


def validate(cfg):
    root=Path(cfg['output']); p=json.loads((root/'protocol.json').read_text())
    assert p['config']==cfg and p['core_sha256']==code_digest()==base.CORE_SHA
    for path,sha in {**p['source_files'],**p['active_study_frozen_files']}.items(): assert digest(path)==sha,path
    assert p['prior_artifacts_sha256']==digest(root/'prior_artifacts.json')
    return p


def fit(cfg):
    protocol=validate(cfg); root,source=Path(cfg['output']),Path(cfg['source'])
    common=dict(np.load(source/'development.npz',allow_pickle=False)); e,t=common['pairs'].T
    qids=common['episode_qids'][e]
    deadline=time.monotonic()+60*cfg['fitting']['session_minutes']; manifests={}
    for fold in protocol['folds']:
        fitted=dict(np.load(source/'folds'/fold['id']/'features.npz',allow_pickle=False))
        ids=np.flatnonzero(np.isin(qids,fold['train_questions']))
        assert not set(qids[ids])&set(fold['assessment_questions'])
        for family in cfg['families']:
            x,names=features(common['spins'][e[ids],t[ids]],fitted['context'][e[ids]],common['J'][e[ids]],fitted['prior_logit'][ids],family)
            assert len(names)==protocol['families'][family]
            for teacher in [r for r in protocol['teachers'] if r['fold']==fold['id']]:
                out=root/'fits'/fold['id']/f'{family}_s{teacher["seed"]}.json'
                if out.exists():
                    saved=json.loads(out.read_text())
                    assert saved['protocol_sha256']==digest(root/'protocol.json')
                    manifests[str(out)]=digest(out); continue
                if time.monotonic()>=deadline: raise RuntimeError('Side-study time guard reached; completed fits saved')
                lp=np.load(Path(teacher['directory'])/'train_logp.npy',allow_pickle=False)
                assert lp.shape==(*x.shape[:2],2)
                record=fit_rule(x,lp,qids[ids],cfg['fitting'])
                row={**record,'protocol_sha256':digest(root/'protocol.json'),'completed_utc':now(),
                    'family':family,'outer':fold['outer'],'fold':fold['id'],'seed':teacher['seed'],
                    'feature_names':names,'training_questions':fold['train_questions'],
                    'training_teacher_predictions_sha256':digest(Path(teacher['directory'])/'train_logp.npy'),
                    'fitting_uses_observed_labels':False,'fitting_uses_outer_inputs':False}
                json_write(out,row); manifests[str(out)]=digest(out)
                print(json.dumps({'fit':str(out),'seconds':record['seconds'],'iterations':record['iterations']}),flush=True)
    assert len(manifests)==48
    payload={'protocol_sha256':digest(root/'protocol.json'),'fits':manifests,'all_fits':48,
             'all_coefficients_frozen_before_outer_evaluation':True}
    path=root/'fits_complete.json'
    if path.exists():
        old=json.loads(path.read_text()); assert all(old[k]==v for k,v in payload.items())
    else: json_write(path,{**payload,'frozen_utc':now()})


def evaluate(cfg):
    protocol=validate(cfg); root,source=Path(cfg['output']),Path(cfg['source'])
    frozen=json.loads((root/'fits_complete.json').read_text()); assert frozen['all_fits']==48
    for path,sha in frozen['fits'].items(): assert digest(path)==sha,path
    common=dict(np.load(source/'development.npz',allow_pickle=False)); e,t=common['pairs'].T
    qids=common['episode_qids'][e]; y=(common['spins'][e,t+1]==1).astype(np.int64)
    output=root/'evaluation'; output.mkdir(parents=True,exist_ok=True)
    rows=[]; fidelity_rows=[]; coefficients=[]; reproductions=[]; subgroup=[]
    old_scores=pd.read_csv(source/'outer_by_question.csv')
    for fold in protocol['folds']:
        fitted=dict(np.load(source/'folds'/fold['id']/'features.npz',allow_pickle=False))
        td=base.FoldData(common,fitted,'cpu')
        train=td.ids(fold['train_questions']); outer=td.ids(fold['assessment_questions'])
        baseline_lp=log_prob(fitted['prior_logit'][outer])
        _,byq=base.metrics(baseline_lp,y[outer],qids[outer])
        rows.extend({'model':'I0','outer':fold['outer'],'seed':-1,'split':'outer',**r} for r in byq)
        for teacher in [r for r in protocol['teachers'] if r['fold']==fold['id']]:
            directory=Path(teacher['directory'])
            teacher_train=np.load(directory/'train_logp.npy',allow_pickle=False)
            teacher_outer=np.load(Path(teacher['assessment_directory'])/'assessment_logp.npy',allow_pickle=False)
            neural=base.make_model(config(cfg['source_config']),cfg['teacher'])
            ck=torch.load(directory/'best.pt',map_location='cpu',weights_only=False); neural.load_state_dict(ck['model'])
            manifest=json.loads((directory/'manifest.json').read_text())
            assert base.model_weights_sha(neural)==manifest['selected_weights_sha256']
            maximum=0.
            for split,ids,teacher_lp in [('train',train,teacher_train),('outer',outer,teacher_outer)]:
                positions=np.linspace(0,len(ids)-1,32,dtype=int)
                actual=np.exp(base.predictions(neural,td,ids[positions],False,32)[...,1])
                expected=np.exp(teacher_lp[positions,...,1])
                maximum=max(maximum,float(np.abs(actual-expected).max()))
                np.testing.assert_allclose(actual,expected,atol=cfg['evaluation']['probability_reproduction_tolerance'],rtol=cfg['evaluation']['probability_reproduction_tolerance'])
                _,byq=base.metrics(teacher_lp,y[ids],qids[ids])
                rows.extend({'model':'teacher','outer':fold['outer'],'seed':teacher['seed'],'split':split,**r} for r in byq)
                expected_rows=old_scores[(old_scores.model==cfg['teacher'])&(old_scores.outer==fold['outer'])&(old_scores.seed==teacher['seed'])&(old_scores.split==('assessment' if split=='outer' else 'train'))].set_index('qid')
                for r in byq:
                    for metric in ['error','nll','brier_two_class']:
                        np.testing.assert_allclose(r[metric],expected_rows.loc[r['qid'],metric],atol=1e-10,rtol=1e-7)
                for family in cfg['families']:
                    rule=json.loads((root/'fits'/fold['id']/f'{family}_s{teacher["seed"]}.json').read_text())
                    assert rule['completed_utc']<=frozen['frozen_utc']
                    x,names=features(common['spins'][e[ids],t[ids]],fitted['context'][e[ids]],common['J'][e[ids]],fitted['prior_logit'][ids],family)
                    assert names==rule['feature_names']
                    lp=predict(rule,x)
                    _,byq=base.metrics(lp,y[ids],qids[ids])
                    fields={'model':family,'outer':fold['outer'],'seed':teacher['seed'],'split':split}
                    rows.extend({**fields,**r} for r in byq)
                    fidelity_rows.extend({**fields,**r} for r in fidelity(lp,teacher_lp,qids[ids]))
                    if split=='outer':
                        np.save(output/f'o{fold["outer"]}_{family}_s{teacher["seed"]}_logp.npy',lp,allow_pickle=False)
                        coefficients += [{'model':family,'outer':fold['outer'],'seed':teacher['seed'],'feature':name,'raw_coefficient':coef}
                                         for name,coef in zip(names,rule['raw_weights'])]
                        p=np.exp(teacher_lp[...,1]); uncertainty=np.maximum(p,1-p)<.8
                        stay=common['spins'][e[ids],t[ids]]==common['spins'][e[ids],t[ids]+1]
                        for group,mask in [('teacher_confidence_lt_0.8',uncertainty),('stay',stay),('switch',~stay)]:
                            for qid in sorted(set(qids[ids])):
                                take=(qids[ids]==qid)[:,None]&mask
                                if not take.any(): continue
                                yy=y[ids][take]; sublp=lp[take]; teacher_sub=teacher_lp[take]
                                subgroup.append({**fields,'qid':str(qid),'group':group,'transitions':int(take.sum()),
                                    'probability_mae':float(np.abs(np.exp(sublp[:,1])-np.exp(teacher_sub[:,1])).mean()),
                                    'teacher_label_disagreement':float((sublp.argmax(-1)!=teacher_sub.argmax(-1)).mean()),
                                    'error':float((sublp.argmax(-1)!=yy).mean()),'teacher_error':float((teacher_sub.argmax(-1)!=yy).mean()),
                                    'nll':float(-sublp[np.arange(len(yy)),yy].mean()),'teacher_nll':float(-teacher_sub[np.arange(len(yy)),yy].mean())})
            reproductions.append({'outer':fold['outer'],'seed':teacher['seed'],'maximum_probability_difference':maximum})
        old=old_scores[(old_scores.model=='I0_current')&(old_scores.outer==fold['outer'])&(old_scores.split=='assessment')]
        rows.extend({**r,'split':'outer'} for r in old[['model','outer','seed','qid','error','nll','brier_two_class','transitions']].to_dict('records'))
    frame=pd.DataFrame(rows); fid=pd.DataFrame(fidelity_rows)
    frame.to_csv(output/'by_question.csv',index=False); fid.to_csv(output/'fidelity_by_question.csv',index=False)
    pd.DataFrame(coefficients).to_csv(output/'coefficients.csv',index=False)
    pd.DataFrame(reproductions).to_csv(output/'teacher_reproduction.csv',index=False)
    pd.DataFrame(subgroup).to_csv(output/'subgroups_by_question.csv',index=False)
    question=frame.groupby(['split','model','qid'])[['error','nll','brier_two_class']].mean().reset_index()
    question.groupby(['split','model'])[['error','nll','brier_two_class']].mean().to_csv(output/'performance_summary.csv')
    fq=fid.groupby(['split','model','qid'])[['probability_mae','probability_rmse','probability_error_p95','teacher_kl','teacher_label_disagreement']].mean().reset_index()
    fq.groupby(['split','model'])[['probability_mae','probability_rmse','probability_error_p95','teacher_kl','teacher_label_disagreement']].mean().to_csv(output/'fidelity_summary.csv')
    rng=np.random.default_rng(cfg['evaluation']['bootstrap_seed']); qnames=sorted(question[question.split=='outer'].qid.unique())
    assert len(qnames)==20
    draws=rng.integers(0,20,size=(cfg['evaluation']['bootstrap_replicates'],20)); contrasts=[]
    for family in cfg['families']:
        a=question[(question.split=='outer')&(question.model==family)].set_index('qid').loc[qnames]
        for reference in ['teacher','I0']:
            b=question[(question.split=='outer')&(question.model==reference)].set_index('qid').loc[qnames]
            for metric in ['error','nll','brier_two_class']:
                diff=(a[metric]-b[metric]).to_numpy(); lower,upper=np.quantile(diff[draws].mean(1),[.025,.975])
                contrasts.append({'model':family,'reference':reference,'metric':metric,'difference':float(diff.mean()),'lower':float(lower),'upper':float(upper)})
    pd.DataFrame(contrasts).to_csv(output/'paired_differences.csv',index=False)
    prior=json.loads((root/'prior_artifacts.json').read_text())
    for p,sha in {**prior,**protocol['active_study_frozen_files']}.items(): assert digest(p)==sha,p
    json_write(root/'evaluation_complete.json',{'completed_utc':now(),'fits_complete_sha256':digest(root/'fits_complete.json'),
        'prior_artifacts_unchanged':len(prior),'active_study_frozen_files_unchanged':len(protocol['active_study_frozen_files']),
        'teacher_checkpoints_reproduced':12,'maximum_probability_reproduction_difference':max(r['maximum_probability_difference'] for r in reproductions),
        'new_assessment_data_read':False,'new_source_calls':0,
        'artifacts':{str(p):digest(p) for p in output.rglob('*') if p.is_file()}})
    print(question.groupby(['split','model'])[['error','nll']].mean().to_string(),flush=True)
    print(fq.groupby(['split','model'])[['probability_mae','teacher_kl','teacher_label_disagreement']].mean().to_string(),flush=True)


def main():
    parser=ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['prepare','fit','evaluate'])
    args=parser.parse_args(); cfg=config(CONFIG); torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        {'prepare':prepare,'fit':fit,'evaluate':evaluate}[args.stage](cfg)


if __name__=='__main__': main()
