"""Frozen-model diagnosis and a controlled within-question PCA experiment.

The only new intervention is the unsupervised message projection. Reuse each
condition's original hyperparameters, duration, data and initialization seeds.
No new validation search; source probabilities are retrospective diagnostics.
"""
from argparse import ArgumentParser
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, config as load_config, digest, environment, json_write
from scripts import compact_argument_study as compact
from scripts import breadth_argument_study as shared
from scripts import transformer_generalization as base

CONFIG = Path('configs/argument_variance_320.yaml')
ANCHOR = Path('runs/question_breadth_v1/error_diagnosis_v1/development_source_probabilities_20261110.json.gz')
ANCHOR_SHA = 'b7c6fd634be77e6fd2a86eb80783066bce3f57f35970e1f9216493dc34a1c0f5'

def read(path): return json.loads(Path(path).read_text())
def config(): return load_config(CONFIG)

def fit_within_question_pca(vectors, training_question_ids, dimensions=8):
    """Each unique training text has weight one, even if shared by questions.

    A text shared by k training questions contributes 1/k within each question.
    This preserves the original total-covariance weighting exactly. Per-question
    means are used only to fit the covariance, never to encode held-out inputs.
    """
    groups = [np.unique(row[row > 2]) for row in training_question_ids]
    assert all(len(g) for g in groups)
    training_ids = np.unique(np.concatenate(groups))
    membership = np.bincount(np.concatenate(groups), minlength=len(vectors))
    x = np.asarray(vectors[training_ids], dtype=np.float64)
    mean = x.mean(0)
    xc = x - mean
    total = xc.T @ xc / len(x)
    masses, means = [], []
    for group in groups:
        weights = 1. / membership[group]
        mass = weights.sum()
        means.append((np.asarray(vectors[group], dtype=np.float64) * weights[:, None]).sum(0) / mass)
        masses.append(mass)
    masses, means = np.asarray(masses), np.asarray(means)
    np.testing.assert_allclose(masses.sum(), len(x), rtol=1e-12)
    between = (means - mean).T @ ((means - mean) * masses[:, None]) / len(x)
    within = total - between
    within = (within + within.T) / 2
    values, axes = np.linalg.eigh(within)
    assert values.min() > -1e-10
    components = axes[:, -dimensions:][:, ::-1].T.copy()
    components *= np.sign(components[np.arange(dimensions), np.abs(components).argmax(1)])[:, None]
    rms = float(np.sqrt(np.mean((xc @ components.T) ** 2)))
    assert rms > 0
    result = ((np.asarray(vectors, dtype=np.float64) - mean) @ components.T / (rms * math.sqrt(dimensions))).astype(np.float32)
    result[0] = 0
    return result, dict(mean=mean, components=components, scale=np.asarray(rms), training_ids=training_ids,
        total_covariance=total, within_covariance=within, between_covariance=between,
        question_masses=masses, within_eigenvalues=values,
        within_variance_retained=np.asarray(np.trace(components @ within @ components.T) / np.trace(within)),
        total_variance_retained=np.asarray(np.trace(components @ total @ components.T) / np.trace(total)))

def validate(cfg, frozen=False):
    root = Path(cfg['output']); protocol = read(root/'protocol.json')
    assert protocol['config'] == cfg and protocol['config_sha256'] == digest(CONFIG)
    assert code_digest() == protocol['core_sha256']
    for group in ('code_files', 'inputs'):
        for path, sha in protocol[group].items(): assert digest(path) == sha, path
    prepared = read(root/'prepared.json')
    assert prepared['protocol_sha256'] == digest(root/'protocol.json')
    for path, sha in prepared['artifacts'].items(): assert digest(path) == sha, path
    if frozen:
        for path, sha in read(root/'models_frozen.json')['artifacts'].items(): assert digest(path) == sha, path
    return protocol

def prepare(cfg):
    root, reference = Path(cfg['output']), Path(cfg['reference'])
    if (root/'prepared.json').exists(): validate(cfg); return
    compact.validate_frozen(compact.config())
    old = read(reference/'protocol.json')
    for key, value in cfg['training'].items(): assert value == old['config']['training'][key], key
    assert cfg['models']['nano_pca8'] == old['config']['models']['nano_pca8']
    folder = root/'data'; folder.mkdir(parents=True, exist_ok=True)
    for name in ('development.npz','features.npz','ising.json','message_ids.npy','matched_substitution.npz'):
        shutil.copyfile(reference/'data'/name, folder/name)
    data = np.load(folder/'development.npz', allow_pickle=False)
    ids = np.load(folder/'message_ids.npy', allow_pickle=False)
    vectors = np.load(Path(cfg['message_reference'])/'data/vectors.npy', mmap_mode='r', allow_pickle=False)
    train = np.flatnonzero(np.isin(data['episode_qids'],old['fold']['train_questions']))
    result, transform = fit_within_question_pca(vectors, ids[train])
    original_transform = np.load(reference/'data/message_pca.npz', allow_pickle=False)
    np.testing.assert_array_equal(transform['training_ids'], original_transform['training_ids'])
    np.testing.assert_allclose(transform['mean'], original_transform['mean'], atol=0, rtol=0)
    np.save(folder/'vectors_pca8.npy', result, allow_pickle=False)
    np.savez(folder/'message_pca.npz', **transform)
    choices = [r for r in read(reference/'selection.json')['choices'] if r['width']=='nano_pca8']
    assert len(choices)==2
    parameters = sum(p.numel() for p in compact.make_model(cfg,'nano_pca8',42).parameters())
    assert parameters==882
    inputs = {str(reference/n):digest(reference/n) for n in ('protocol.json','prepared.json','models_frozen.json',
        'selection.json','results/summary.json','results/predictions.json','data/message_pca.npz')}
    inputs.update(read(reference/'models_frozen.json')['artifacts'])
    inputs.update({r['path']:r['sha256'] for r in read(reference/'results/predictions.json')})
    inputs[str(Path(cfg['message_reference'])/'data/vectors.npy')] = digest(Path(cfg['message_reference'])/'data/vectors.npy')
    assert digest(ANCHOR)==ANCHOR_SHA
    inputs[str(ANCHOR)] = ANCHOR_SHA
    code = [CONFIG,Path(__file__),Path(compact.__file__),Path(shared.__file__),Path(base.__file__)]
    protocol = dict(created_utc=shared.now(),config=cfg,config_sha256=digest(CONFIG),core_sha256=code_digest(),
        code_files={str(p):digest(p) for p in code},inputs=inputs,fold=old['fold'],parameters={'nano_pca8':parameters},
        hypothesis='Ordinary PCA emphasizes between-question variation; emphasizing within-question message variation may transfer better.',
        design='Change only the eight-dimensional frozen projection and its training RMS scaling. Equal unique-training-text weights. Use per-question means only for fitting covariance; encode every message by a single global training mean and fixed axes. No per-validation-question centering.',
        training_policy='No new tuning. Reuse the original nano_pca8 condition-specific candidate and duration, same optimizer, seeds and initialization. Six new refits. A fixed-recipe representation comparison, not a search for an optimal representation.',
        evaluation_policy='All three seed scores averaged. Primary new contrast is within-question PCA versus original PCA for arguments; report conclusion control and difference-in-differences too. Question-paired 5000-draw pointwise bootstrap on reused 80 validation questions. Exploratory.',
        anchor_policy='Logged source probabilities cover only the retained 80 train and 20 validation questions. First A/B token approximation to majority-of-five probabilities. Diagnostic only, never predictor inputs or labels for these fits.',
        diagnostic_policy='Error overlap, corrections/regressions, vote agreement, threshold margins, within/between message variation and source-probability error on the retained anchor. No causal or universal noise-ceiling claim.',
        assessment_loaded=False,source_llm_calls=0,new_tuning_fits=0,new_refits=6,choices=choices)
    json_write(root/'protocol.json',protocol)
    json_write(root/'encoding.json',{'source_encoding_sha256':digest(Path(cfg['message_reference'])/'encoding.json'),
        'message_pca_sha256':digest(folder/'message_pca.npz'),'encoder_parameters':22713216})
    json_write(root/'selection.json',{'protocol_sha256':digest(root/'protocol.json'),'choices':choices,
        'selection_source_sha256':digest(reference/'selection.json'),'new_validation_search':False})
    json_write(root/'prepared.json',{'protocol_sha256':digest(root/'protocol.json'),
        'artifacts':{str(p):digest(p) for p in list(folder.iterdir())+[root/'encoding.json',root/'selection.json']}})
    print(json.dumps({'prepared':True,'parameters':parameters,'new_fits':6,
        'within_variance_retained':float(transform['within_variance_retained']),
        'total_variance_retained':float(transform['total_variance_retained'])}),flush=True)

def fit(cfg, device='cpu'):
    protocol = validate(cfg); root = Path(cfg['output'])
    if (root/'models_frozen.json').exists(): validate(cfg,True); return
    torch.set_num_threads(cfg['training']['torch_threads'])
    data = compact.Data(cfg,device); deadline=time.time()+60*cfg['training']['aggregate_minutes']; refits=[]
    json_write(root/'training_environment.json',environment())
    with compact.adapted_trainer():
        for choice in protocol['choices']:
            for seed in cfg['training']['refit_seeds']:
                item={k:choice[k] for k in ('width','condition','candidate')};item.update(phase='refit',seed=seed)
                path,manifest=shared.train_one(cfg,protocol,data,item,deadline,choice['refit_steps'])
                if manifest['status']!='complete':raise SystemExit(75)
                old=Path(cfg['reference'])/'refit'/path.name
                assert manifest['initial_weights_sha256']==read(old/'manifest.json')['initial_weights_sha256']
                refits.append((path,manifest))
    artifacts={str(root/n):digest(root/n) for n in ('protocol.json','prepared.json','selection.json','encoding.json')}
    for path,m in refits:
        artifacts[str(path/'manifest.json')]=digest(path/'manifest.json')
        artifacts.update({str(path/n):sha for n,sha in m['artifacts'].items()})
    json_write(root/'models_frozen.json',dict(frozen_utc=shared.now(),artifacts=artifacts,
        refits=[str(p) for p,m in refits],fit_seconds=sum(m['seconds'] for p,m in refits),assessment_loaded=False))
    print(json.dumps({'models_frozen':True,'refits':len(refits)}),flush=True)

def paired(a,b,draws):
    """Arrays are seed x question x within-question observation."""
    d=np.asarray(a).mean(0).mean(-1)-np.asarray(b).mean(0).mean(-1)
    return {'mean_difference':float(d.mean()),'ci95':np.quantile(d[draws].mean(1),[.025,.975]).tolist()}

def metrics(logp,y):
    return {'error':(logp.argmax(-1)!=y).astype(float),
        'nll':-np.take_along_axis(logp,np.broadcast_to(y,logp.shape[:-1])[...,None],axis=-1)[...,0],
        'brier_two_class':2*(np.exp(logp[...,1])-y)**2}

def error_pair(a,b,y,mask=None):
    ea=a.argmax(-1)!=y; eb=b.argmax(-1)!=y
    ea,eb=np.broadcast_arrays(ea,eb)
    if mask is None:mask=np.ones(y.shape,bool)
    ea,eb=ea[:,mask],eb[:,mask]
    disagree=ea!=eb; corrections=eb&~ea; regressions=~eb&ea; common=ea&eb
    return {'fraction':float(mask.mean()),'observations':int(mask.sum()),'left_error':float(ea.mean()),
        'right_error':float(eb.mean()),'disagreement':float(disagree.mean()),'corrected':float(corrections.mean()),
        'introduced':float(regressions.mean()),'shared_errors':float(common.mean()),
        'shared_fraction_of_left_errors':float(common.sum()/ea.sum()) if ea.sum() else None,
        'left_accuracy_when_disagree':float(corrections.sum()/disagree.sum()) if disagree.sum() else None,
        'probability_mae':float(np.abs(np.exp(a[...,1])-np.exp(b[...,1]))[:,mask].mean())}

def representation_geometry(vectors,ids,states):
    active=ids>0;x=np.asarray(vectors[ids],dtype=np.float64);mu=x[active].mean(0)
    total=float(((x[active]-mu)**2).sum());within_q=0.;within_qss=0.
    for e in range(len(ids)):
        z=x[e][active[e]];within_q+=float(((z-z.mean(0))**2).sum())
        for s in range(ids.shape[-1]):
            for stance in (-1,1):
                take=active[e,:,s]&(states[e,:,s]==stance);z=x[e,take,s]
                if len(z):within_qss+=float(((z-z.mean(0))**2).sum())
    return {'messages':int(active.sum()),'between_question_fraction':1-within_q/total,
        'within_question_fraction':within_q/total,'within_question_sender_stance_fraction':within_qss/total}

def anchor_probabilities(common):
    assert digest(ANCHOR)==ANCHOR_SHA
    with gzip.open(ANCHOR,'rt') as f:source=json.load(f)
    lookup={q:i for i,q in enumerate(common['episode_qids'])};p=np.full(common['spins'].shape,np.nan)
    raw=np.zeros_like(common['raw_votes']);observed=np.zeros(common['spins'].shape,bool)
    for row in source['rows']:
        key=(lookup[row['qid']],row['round'],row['agent'])
        assert not observed[key];observed[key]=True;raw[key]=row['raw_votes']
        a,b=np.asarray(row['first_literal_A_probabilities']),np.asarray(row['first_literal_B_probabilities'])
        assert np.ptp(a)==np.ptp(b)==0
        p[key]=a[0]/(a[0]+b[0])
    np.testing.assert_array_equal(raw[observed],common['raw_votes'][observed])
    assert len(source['rows'])==28800
    return p,{'retained_questions':int(observed.any(axis=(1,2)).sum()),'verified_vote_groups':int(observed.sum()),
        'strict_responses':source['strict_response_count'],'non_single_token_responses':source['not_single_bare_token_response_count'],
        'within_prompt_probability_spread':source['maximum_within_prompt_probability_spread']}

def evaluate(cfg,device='cpu'):
    protocol=validate(cfg,True);root=Path(cfg['output']);out=root/'results';out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():return
    torch.set_num_threads(cfg['training']['torch_threads']);data=compact.Data(cfg,device)
    index=[];replay=[]
    for path in read(root/'models_frozen.json')['refits']:
        path=Path(path);m=read(path/'manifest.json');model=compact.make_model(cfg,m['width']).to(device)
        model.load_state_dict(torch.load(path/'best.pt',map_location=device,weights_only=False)['model'])
        assert base.model_weights_sha(model)==m['selected_weights_sha256']
        for split in ('train','validation'):
            ids=data.ids(protocol['fold'][split+'_questions'])
            lp=shared.predictions(model,data,ids,cfg['training']['evaluation_batch_size'],m['condition'])
            name='within_pca8_'+m['condition'];file=out/f'{name}-s{m["seed"]}-{split}.npz'
            np.savez_compressed(file,logp=lp,target=data.y[ids],current=data.s[ids],qids=data.qids[ids],pairs=data.common['pairs'][ids])
            index.append({'path':str(file),'sha256':digest(file),'model':name,'seed':m['seed'],'split':split,'origin':str(path)})
            if split=='train':
                old=np.load(path/'train_logp.npy',allow_pickle=False)
                replay.append({'model':name,'seed':m['seed'],'max_probability_difference':float(np.abs(np.exp(lp)-np.exp(old)).max()),
                    'class_disagreements':int((lp.argmax(-1)!=old.argmax(-1)).sum())})
    json_write(out/'predictions.json',index)
    index+=read(Path(cfg['reference'])/'results/predictions.json')
    common=data.common;p,quality=anchor_probabilities(common);rows=[];comparisons=[];geometry={};diagnostics={};anchor={}
    selected_names=['I0','opinion_nano','pico_pca8_conclusion','pico_pca8_arguments','nano_pca8_conclusion','nano_pca8_arguments',
        'pico_full_conclusion','pico_full_arguments','nano_full_conclusion','nano_full_arguments',
        'within_pca8_conclusion','within_pca8_arguments']
    for split in ('train','validation'):
        loaded={};ids=data.ids(protocol['fold'][split+'_questions']);e,t=common['pairs'][ids].T
        y=data.y[ids];current=data.s[ids];qids=data.qids[ids];questions=np.unique(qids)
        assert len(ids)==8*len(questions)
        order=np.concatenate([np.flatnonzero(qids==q) for q in questions])
        vote_agreement=np.maximum((common['raw_votes'][e,t+1]==1).sum(-1),(common['raw_votes'][e,t+1]==-1).sum(-1))
        source_p=p[e,t+1];take_anchor=np.isfinite(source_p).all(1);source_p=source_p[take_anchor]
        source_q=np.clip(source_p**3*(10-15*source_p+6*source_p**2),0,1)
        anchor[split]={'questions':int(take_anchor.sum()/8),'sampling_reference_error':float(np.minimum(source_q,1-source_q).mean()),
            'models':{},'source_mode_observed_error':float(((source_q>.5)!=y[take_anchor]).mean())}
        rng=np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
        draws=rng.integers(0,len(questions),(cfg['evaluation']['bootstrap_replicates'],len(questions)))
        for name in selected_names:
            records=sorted([r for r in index if r['model']==name and r['split']==split],key=lambda r:r['seed'])
            arrays=[]
            for r in records:
                assert digest(r['path'])==r['sha256'];z=np.load(r['path'],allow_pickle=False)
                for key,expected in [('target',y),('current',current),('qids',qids),('pairs',common['pairs'][ids])]:
                    np.testing.assert_array_equal(z[key],expected)
                arrays.append(z['logp'])
            lp=np.stack(arrays);loaded[name]=lp
            mm=metrics(lp,y);error=mm['error']
            row=dict(model=name,split=split,seeds=[r['seed'] for r in records],**{k:float(v.mean()) for k,v in mm.items()})
            row['per_seed']=[{k:float(v[i].mean()) for k,v in mm.items()} for i in range(len(lp))]
            row['vote_agreement']={str(k):{'fraction':float((vote_agreement==k).mean()),
                'error':float(error[:,vote_agreement==k].mean())} for k in (3,4,5)}
            row['error_fraction_on_split_votes']=float(error[:,vote_agreement<5].sum()/error.sum())
            for category,mask in [('stay',(current==1)==y),('switch',(current==1)!=y)]:
                row[category]={'fraction':float(mask.mean()),'error':float(error[:,mask].mean())}
            rows.append(row)
            a=lp[:,take_anchor];pred=a.argmax(-1);expected_error=np.where(pred,1-source_q,source_q)
            anchor[split]['models'][name]={'observed_error':float((pred!=y[take_anchor]).mean()),
                'expected_error':float(expected_error.mean()),'excess_expected_error':float((expected_error-np.minimum(source_q,1-source_q)).mean()),
                'expected_nll':float((-(1-source_q)*a[...,0]-source_q*a[...,1]).mean()),
                'probability_mse':float(((np.exp(a[...,1])-source_q)**2).mean())}
        pairs=[('opinion_nano','I0')]+[(n+'_arguments',n+'_conclusion') for n in
            ('pico_pca8','nano_pca8','pico_full','nano_full','within_pca8')]
        pairs += [('within_pca8_arguments','nano_pca8_arguments'),('within_pca8_conclusion','nano_pca8_conclusion'),
            ('within_pca8_arguments','opinion_nano')]
        for left,right in pairs:
            a,b=loaded[left],loaded[right];record={'split':split,'left':left,'right':right,**error_pair(a,b,y)}
            if split=='validation':
                am,bm=metrics(a,y),metrics(b,y)
                record['paired']={k:paired(am[k][:,order].reshape(len(a),len(questions),-1),
                    bm[k][:,order].reshape(len(b),len(questions),-1),draws) for k in am}
                margins=np.abs(np.exp(b[...,1])-.5)
                record['margin_bins']=[]
                for lo,hi in [(0.,.1),(.1,.4),(.4,.500001)]:
                    values=[]
                    for seed in range(len(a)):
                        mask=(margins[seed%len(b)]>=lo)&(margins[seed%len(b)]<hi)
                        values.append(error_pair(a[seed:seed+1],b[seed%len(b):seed%len(b)+1],y,mask))
                    record['margin_bins'].append({'lower':lo,'upper':hi,'mean_seed_metrics':{
                        k:float(np.mean([v[k] for v in values])) for k in values[0] if all(v[k] is not None for v in values)}})
            comparisons.append(record)
        if split=='validation':
            diagnostic={}
            for metric in ('error','nll','brier_two_class'):
                def loss(name):return metrics(loaded[name],y)[metric][:,order].reshape(3,len(questions),-1)
                diagnostic[metric]=paired(loss('within_pca8_arguments')-loss('within_pca8_conclusion'),
                    loss('nano_pca8_arguments')-loss('nano_pca8_conclusion'),draws)
            diagnostics['difference_in_differences']=diagnostic
        episodes=np.flatnonzero(np.isin(common['episode_qids'],protocol['fold'][split+'_questions']))
        mid=data.message_ids[episodes];states=common['spins'][episodes,:-1]
        geometry[split]={name:representation_geometry(v,mid,states) for name,v in [
            ('full384',data.full_vectors),('ordinary_pca8',np.load(Path(cfg['reference'])/'data/vectors_pca8.npy')),
            ('within_pca8',data.pca_vectors)]}
    transform=np.load(root/'data/message_pca.npz',allow_pickle=False)
    original=np.load(Path(cfg['reference'])/'data/message_pca.npz',allow_pickle=False)
    subspaces={}
    for name,axes in [('ordinary_pca8',original['components']),('within_pca8',transform['components'])]:
        total=float(np.trace(axes@transform['total_covariance']@axes.T));within=float(np.trace(axes@transform['within_covariance']@axes.T))
        subspaces[name]={'total_variance_retained':total/float(np.trace(transform['total_covariance'])),
            'within_variance_retained':within/float(np.trace(transform['within_covariance'])),
            'between_fraction_in_projected_variance':1-within/total}
    result=dict(completed_utc=shared.now(),models=rows,comparisons=comparisons,diagnostics=diagnostics,
        representation_geometry=geometry,training_unique_text_subspaces=subspaces,anchor=anchor,anchor_quality=quality,
        training_replay=replay,fit_seconds=read(root/'models_frozen.json')['fit_seconds'],parameters=882,
        models_frozen_sha256=digest(root/'models_frozen.json'),protocol_sha256=digest(root/'protocol.json'),
        source_llm_calls=0,assessment_loaded=False,new_tuning_fits=0,new_refits=6)
    json_write(out/'summary.json',result)
    print(json.dumps({'evaluation':'complete','new_models':[r for r in rows if r['model'].startswith('within_')],
        'subspaces':subspaces,'fit_seconds':result['fit_seconds']}),flush=True)

if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('action',choices=['prepare','fit','evaluate','all']);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();cfg=config()
    if args.action in ('prepare','all'):prepare(cfg)
    if args.action in ('fit','all'):fit(cfg,args.device)
    if args.action in ('evaluate','all'):evaluate(cfg,args.device)
