"""Frozen 818-parameter transformer versus original Ising on fresh questions."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config as load_config, code_digest, digest, json_write
from mac_agents.models.baselines import fit_baseline, baseline_logp
from scripts import transformer_generalization as base
from scripts.question_breadth import CONFIG, ROOT, UPSTREAM, config
from scripts.breadth_collect import authorize_split, verify_protocol

ENCODER=Path('.cache/huggingface/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41')
REPORT=Path('reports/question_breadth.md')


def now(): return datetime.now(timezone.utc).isoformat()


def question_text(q):
    # Correct-answer labels, worked solutions, entropy probes and future outcomes
    # are deliberately outside the predictor input interface.
    return f"{q['question']}\nA: {q['choices']['A']}\nB: {q['choices']['B']}"


def encode_texts(texts,device):
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from transformers import AutoTokenizer, AutoModel
    tokenizer=AutoTokenizer.from_pretrained(str(ENCODER.resolve()),local_files_only=True)
    model=AutoModel.from_pretrained(str(ENCODER.resolve()),local_files_only=True).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    tokens=tokenizer(texts,add_special_tokens=False,truncation=False,verbose=False)['input_ids']
    if not all(tokens): raise ValueError('Empty context text')
    windows=[(i,t[j:j+510]) for i,t in enumerate(tokens) for j in range(0,len(t),510)]
    windows.sort(key=lambda x:len(x[1]))
    accum=np.zeros((len(texts),384),np.float64); counts=np.zeros(len(texts),np.int64)
    for offset in range(0,len(windows),32):
        part=windows[offset:offset+32]; width=max(len(t) for _,t in part)+2
        ids=torch.full((len(part),width),tokenizer.pad_token_id,dtype=torch.long,device=device)
        attention=torch.zeros_like(ids); pool=torch.zeros_like(ids)
        for k,(_,t) in enumerate(part):
            seq=[tokenizer.cls_token_id]+t+[tokenizer.sep_token_id]
            ids[k,:len(seq)]=torch.tensor(seq,device=device); attention[k,:len(seq)]=1
            pool[k,1:len(seq)-1]=1
        with torch.inference_mode():
            hidden=model(input_ids=ids,attention_mask=attention).last_hidden_state
            sums=(hidden*pool[...,None]).sum(1).cpu().numpy()
        for k,(i,t) in enumerate(part): accum[i]+=sums[k]; counts[i]+=len(t)
    assert np.array_equal(counts,[len(t) for t in tokens])
    vectors=(accum/np.linalg.norm(accum,axis=-1,keepdims=True).clip(1e-30)).astype(np.float32)
    assert np.isfinite(vectors).all()
    return vectors,{'encoder_parameters':sum(p.numel() for p in model.parameters()),
                    'text_count':len(texts),'windows':len(windows),'tokens_dropped':0,
                    'token_counts':counts.tolist()}


def load_episodes(questions,root=None):
    root=ROOT if root is None else Path(root)
    spins=[]; graphs=[]; votes=[]; qids=[]; source={}
    for q in questions:
        path=Path(root)/'episodes'/q['split']/f"{q['qid']}__{q['graph_id']}__r00.json"
        r=json.loads(path.read_text())
        assert r['meta']['qid']==q['qid'] and r['meta']['split']==q['split']
        assert r['meta']['protocol_sha256']==digest(Path(root)/'source_protocol.json')
        s=np.asarray(r['spins_history'],dtype=np.int8)
        raw=np.asarray(r['spins_raw_history'],dtype=np.int8)
        j=np.asarray(r['J'],dtype=np.int8)
        assert s.shape==(9,32) and raw.shape==(9,32,5) and j.shape==(32,32)
        assert np.isin(s,[-1,0,1]).all() and np.isin(raw,[-1,0,1]).all()
        assert np.isin(j,[-1,0,1]).all() and np.array_equal(j,j.T) and not np.diag(j).any()
        summed=raw.sum(-1)
        derived=np.where(summed>0,1,np.where(summed<0,-1,raw[...,0]))
        np.testing.assert_array_equal(s,derived)
        spins.append(s); graphs.append(j); votes.append(raw); qids.append(q['qid']); source[str(path)]=digest(path)
    spins=np.array(spins); graphs=np.array(graphs); raw=np.array(votes)
    valid=((spins[:,:-1]!=0)&(spins[:,1:]!=0)).all(-1)
    pairs=np.array([(e,t) for e in range(len(qids)) for t in range(8) if valid[e,t]],dtype=np.int64)
    if not all(valid.any(axis=1)): raise ValueError('A question has no valid transitions')
    return {'spins':spins,'J':graphs,'valid':valid,'pairs':pairs,
            'episode_qids':np.array(qids),'raw_votes':raw},source


def apply_context(question_vectors,persona_vectors,fitted):
    q=(np.asarray(question_vectors)-fitted['question_mean'])@fitted['question_components'].T
    p=(np.asarray(persona_vectors)-fitted['persona_mean'])@fitted['persona_components'].T
    e,n=len(q),len(p)
    return np.concatenate([np.ones((e,n,1)),np.broadcast_to(p,(e,n,3)),
        np.broadcast_to(q[:,None],(e,n,3)),(p[None,:,:,None]*q[:,None,None,:]).reshape(e,n,9)],-1).astype(np.float32)


def predictor_config():
    cfg=config(); cfg['output']=str(ROOT/'predictors'); return cfg


def validate_prepared():
    root=ROOT/'predictors'; protocol=json.loads((root/'protocol.json').read_text())
    assert protocol['code_sha256']==digest(__file__)
    assert protocol['base_code_sha256']==digest(base.__file__)
    assert protocol['config_sha256']==digest(CONFIG)
    assert protocol['core_sha256']==code_digest()==base.CORE_SHA
    assert protocol['question_selection_sha256']==digest(ROOT/'selection.json')
    for path,sha in protocol['source_artifacts'].items(): assert digest(path)==sha,path
    for path,sha in protocol['encoder_files'].items(): assert digest(path)==sha,path
    prepared=json.loads((root/'prepared.json').read_text())
    assert prepared['protocol_sha256']==digest(root/'protocol.json')
    for path,sha in prepared['artifacts'].items(): assert digest(root/path)==sha,path
    return protocol


def prepare(device):
    verify_protocol()
    cfg=predictor_config(); root=Path(cfg['output']); root.mkdir(parents=True,exist_ok=True)
    if (root/'prepared.json').exists(): validate_prepared(); return
    assert not list((ROOT/'episodes/assessment').glob('*.json'))
    selection=json.loads((ROOT/'selection.json').read_text())
    questions=[q for q in selection['questions'] if q['split'] in ['train','validation']]
    train=[q['qid'] for q in questions if q['split']=='train']; val=[q['qid'] for q in questions if q['split']=='validation']
    assess=[q['qid'] for q in selection['questions'] if q['split']=='assessment']
    assert [len(train),len(val),len(assess)]==[80,20,40]
    common,sources=load_episodes(questions)
    personas=[p['persona'] for p in json.loads((UPSTREAM/'obj/personas.json').read_text())]
    texts=[question_text(q) for q in questions]+personas
    vectors,encoding=encode_texts(texts,device)
    qvec={q['qid']:vectors[i] for i,q in enumerate(questions)}; pvec=vectors[len(questions):]
    fitted=base.fit_context(qvec,pvec,train,common['episode_qids'])
    np.testing.assert_allclose(fitted['context'],apply_context(vectors[:len(questions)],pvec,fitted),atol=1e-7)
    ds=base.fold_dataset(common,fitted['context'],train)
    original=load_config('configs/paper_binary_inductive.yaml')
    with threadpool_limits(limits=cfg['training']['torch_threads']):
        baseline=fit_baseline('I0',ds,original)
    assert baseline['parameters']==19
    e,t=common['pairs'].T
    lp=baseline_logp(baseline,common['spins'][e,t],fitted['context'][e],common['J'][e])
    fitted['prior_logit']=lp[...,1]-lp[...,0]
    fold={'id':'main','outer':0,'phase':'development','train_questions':train,
          'validation_questions':val,'assessment_questions':assess}
    sources.update({str(ROOT/'source_protocol.json'):digest(ROOT/'source_protocol.json'),
                    str(ROOT/'selection.json'):digest(ROOT/'selection.json')})
    protocol={'created_utc':now(),'code_sha256':digest(__file__),'base_code_sha256':digest(base.__file__),
              'config_sha256':digest(CONFIG),'core_sha256':code_digest(),'config':cfg,
              'source_artifacts':sources,'question_selection_sha256':digest(ROOT/'selection.json'),
              'encoder_files':{str(p):digest(p) for p in ENCODER.iterdir() if p.is_file()},
              'fold':fold,'candidates':base.candidates(cfg),
              'parameters':{'nano':sum(p.numel() for p in base.make_model(cfg,'nano').parameters()),'I0':19},
              'selection_policy':'Nine LR/penalty candidates at seed 42; choose validation-question NLL and that checkpoint step. Freeze settings and duration before three fresh-seed refits on the same 80 training questions. Do not refit on validation questions.',
              'assessment_policy':'Collect 40 independent assessment conversations only after all final weights freeze; one final comparison. Average seed scores within question, then questions equally.',
              'encoder_policy':'Shared frozen MiniLM; complete 510-token windows, token-weighted hidden means, L2 normalization. PCA3 is fitted on 80 training questions and the 32 known personas only.',
              'entropy_reporting_bins_bits':[0.,.25,.5,.75,1.000000001],
              'source_llm_calls_in_predictor_stage':0}
    assert protocol['parameters']['nano']==818 and protocol['core_sha256']==base.CORE_SHA
    json_write(root/'protocol.json',protocol)
    np.savez_compressed(root/'development.npz',**common)
    np.savez_compressed(root/'context_vectors.npz',question_vectors=vectors[:len(questions)],persona_vectors=pvec)
    json_write(root/'encoding.json',encoding)
    out=root/'folds/main'; out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'features.npz',**fitted)
    json_write(out/'ising.json',baseline)
    json_write(out/'manifest.json',{**fold,'features_sha256':digest(out/'features.npz'),
              'ising_sha256':digest(out/'ising.json'),'protocol_sha256':digest(root/'protocol.json')})
    names=['development.npz','context_vectors.npz','encoding.json','folds/main/features.npz','folds/main/ising.json','folds/main/manifest.json']
    json_write(root/'prepared.json',{'protocol_sha256':digest(root/'protocol.json'),'created_utc':now(),
                                   'artifacts':{name:digest(root/name) for name in names}})
    print(json.dumps({'predictors_prepared':True,'train_questions':80,'validation_questions':20,
                      'assessment_loaded':False,'neural_parameters':818}),flush=True)


def fit(device,deadline):
    protocol=validate_prepared(); cfg=predictor_config(); root=Path(cfg['output'])
    if (ROOT/'models_frozen.json').exists(): authorize_split(ROOT,'assessment'); return
    assert not list((ROOT/'episodes/assessment').glob('*.json'))
    common=dict(np.load(root/'development.npz',allow_pickle=False))
    features=dict(np.load(root/'folds/main/features.npz',allow_pickle=False))
    data=base.FoldData(common,features,device); fold=protocol['fold']; tuned=[]
    deadline=min(deadline,time.time()+60*cfg['training']['aggregate_minutes'])
    for candidate in protocol['candidates']:
        item={'phase':'tune','fold':'main','outer':0,'model':'nano','candidate':candidate['id'],
              'seed':cfg['training']['tuning_seed']}
        directory,record=base.train_one(cfg,protocol,data,fold,item,device,deadline)
        if record['status']!='complete': raise SystemExit(75)
        tuned.append((directory,record))
    chosen=min(tuned,key=lambda x:(x[1]['validation']['nll'],x[1]['candidate']))[1]
    selection={'protocol_sha256':digest(root/'protocol.json'),'candidate':chosen['candidate'],
               'refit_steps':chosen['selected_step'],'validation_nll':chosen['validation']['nll'],
               'tuning_manifests':{str(p):digest(p/'manifest.json') for p,_ in tuned}}
    if (root/'selection.json').exists():
        previous=json.loads((root/'selection.json').read_text())
        assert all(previous[k]==v for k,v in selection.items())
    else: json_write(root/'selection.json',{**selection,'frozen_utc':now()})
    refs=[]
    for seed in cfg['training']['refit_seeds']:
        item={'phase':'refit','fold':'main','outer':0,'model':'nano','candidate':chosen['candidate'],'seed':seed}
        directory,record=base.train_one(cfg,protocol,data,fold,item,device,deadline,selection['refit_steps'])
        if record['status']!='complete': raise SystemExit(75)
        refs.append({'directory':str(directory),'seed':seed,'manifest_sha256':digest(directory/'manifest.json')})
    artifacts={str(p):digest(p) for p in root.rglob('*') if p.is_file()}
    artifacts.update({str(CONFIG):digest(CONFIG),str(Path(__file__)):digest(__file__),
                      str(ROOT/'selection.json'):digest(ROOT/'selection.json'),
                      str(ROOT/'source_protocol.json'):digest(ROOT/'source_protocol.json')})
    json_write(ROOT/'models_frozen.json',{'frozen_utc':now(),'training_questions':80,'validation_questions':20,
               'assessment_outcomes_observed':False,'refits':refs,'neural_parameters':818,'ising_coefficients':19,
               'artifacts':artifacts})
    print(json.dumps({'models_frozen':True,'neural_refits':3,'assessment_unlocked_for_collection':True}),flush=True)


def paired_difference(frame,cfg):
    values=frame.groupby(['model','qid'])[['error','nll','brier_two_class']].mean()
    a=values.loc['nano'].sort_index(); b=values.loc['I0'].sort_index()
    assert list(a.index)==list(b.index) and len(a)==40
    rng=np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    draws=rng.integers(0,40,size=(cfg['evaluation']['bootstrap_replicates'],40))
    rows=[]
    for metric in ['nll','error','brier_two_class']:
        difference=(a[metric]-b[metric]).to_numpy()
        lower,upper=np.quantile(difference[draws].mean(axis=1),[.025,.975])
        rows.append({'metric':metric,'difference':float(difference.mean()),'lower':float(lower),'upper':float(upper),
                     'questions_better':int((difference<0).sum())})
    return pd.DataFrame(rows)


def population_summary(questions,common):
    rows=[]
    for i,q in enumerate(questions):
        votes=common['raw_votes'][i]
        good=common['valid'][i]
        rows.append({'qid':q['qid'],'split':q['split'],'type':q['type'],'level':q['level'],
            'correct_option':q['answer'],'graph_id':q['graph_id'],
            'distractor_format_valid':q['distractor_construction']['format_valid'],
            'valid_snapshots':int(good.sum()),'excluded_snapshots':int((~good).sum()),
            'unparsed_votes':int((votes==0).sum()),
            'unanimous_vote_fraction':float((np.abs(votes.sum(-1))==5).mean()),
            'observed_switch_fraction':float((common['spins'][i,1:][good]!=common['spins'][i,:-1][good]).mean()),
            **{k:q['screen'][k] for k in ['pooled_entropy_bits','within_agent_entropy_bits','between_agent_entropy_bits']}})
    return pd.DataFrame(rows)


def assess(device):
    authorize_split(ROOT,'assessment'); protocol=validate_prepared(); cfg=predictor_config(); root=Path(cfg['output'])
    frozen=json.loads((ROOT/'models_frozen.json').read_text())
    if (ROOT/'comparison_complete.json').exists():
        old=json.loads((ROOT/'comparison_complete.json').read_text())
        assert old['models_frozen_sha256']==digest(ROOT/'models_frozen.json')
        for path,sha in {**old['assessment_sources'],**old['artifacts']}.items():
            assert digest(path)==sha,path
        print(json.dumps({'comparison_already_complete':True,'assessment_questions':40}),flush=True)
        return
    selected=json.loads((ROOT/'selection.json').read_text())['questions']
    questions=[q for q in selected if q['split']=='assessment']; assert len(questions)==40
    common,sources=load_episodes(questions)
    for path in sources:
        episode=json.loads(Path(path).read_text())
        assert frozen['frozen_utc']<=episode['completed_utc']
    out=root/'assessment'; out.mkdir(parents=True,exist_ok=True)
    features=dict(np.load(root/'folds/main/features.npz',allow_pickle=False))
    vectors,encoding=encode_texts([question_text(q) for q in questions],device)
    pvec=np.load(root/'context_vectors.npz',allow_pickle=False)['persona_vectors']
    context=apply_context(vectors,pvec,features)
    np.savez_compressed(out/'context.npz',question_vectors=vectors,context=context)
    json_write(out/'encoding.json',encoding)
    baseline=json.loads((root/'folds/main/ising.json').read_text())
    e,t=common['pairs'].T
    baseline_lp=baseline_logp(baseline,common['spins'][e,t],context[e],common['J'][e])
    assessment_features={'context':context,'prior_logit':baseline_lp[...,1]-baseline_lp[...,0]}
    data=base.FoldData(common,assessment_features,device); cpu_data=base.FoldData(common,assessment_features,'cpu')
    ids=np.arange(len(e)); records=[]; qrows=[]; groups=[]; repro=[]
    predictions=[('I0',-1,baseline_lp)]
    for run in frozen['refits']:
        directory=Path(run['directory']); manifest=json.loads((directory/'manifest.json').read_text())
        assert digest(directory/'manifest.json')==run['manifest_sha256']
        model=base.make_model(cfg,'nano').to(device)
        checkpoint=torch.load(directory/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model'])
        assert base.model_weights_sha(model)==manifest['selected_weights_sha256']
        lp=base.predictions(model,data,ids,False,256)
        cpu=base.make_model(cfg,'nano')
        cpu.load_state_dict(torch.load(directory/'best.pt',map_location='cpu',weights_only=False)['model'])
        positions=np.linspace(0,len(e)-1,min(64,len(e)),dtype=int)
        actual=np.exp(base.predictions(cpu,cpu_data,positions,False,64)[...,1])
        expected=np.exp(lp[positions,...,1]); np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=5e-5)
        repro.append({'seed':run['seed'],'maximum_probability_difference':float(np.max(np.abs(actual-expected)))})
        predictions.append(('nano',run['seed'],lp))
    qlookup={q['qid']:q for q in questions}
    for kind,seed,lp in predictions:
        np.save(out/f'{kind}_{seed}_logp.npy',lp,allow_pickle=False)
        result,by_question=base.metrics(lp,data.y,data.qids)
        records.append({'model':kind,'seed':seed,**result})
        for row in by_question:
            q=qlookup[row['qid']]
            qrows.append({'model':kind,'seed':seed,**row,'type':q['type'],'level':q['level'],
                          'distractor_format_valid':q['distractor_construction']['format_valid'],
                          **{key:q['screen'][key] for key in ['pooled_entropy_bits','within_agent_entropy_bits','between_agent_entropy_bits']}})
        nll=-np.take_along_axis(lp,data.y[...,None],-1)[...,0]
        error=lp.argmax(-1)!=data.y
        stay=(common['spins'][e,t]==common['spins'][e,t+1])
        for qid in sorted(qlookup):
            qmask=data.qids==qid
            for label,mask in [('stay',stay),('switch',~stay)]:
                take=qmask[:,None]&mask
                if take.any(): groups.append({'model':kind,'seed':seed,'qid':qid,'group':label,
                    'transitions':int(take.sum()),'error':float(error[take].mean()),'nll':float(nll[take].mean())})
            for round_id in range(8):
                take=qmask&(t==round_id)
                if take.any(): groups.append({'model':kind,'seed':seed,'qid':qid,'group':f'round_{round_id+1}',
                    'transitions':int(take.sum()*32),'error':float(error[take].mean()),'nll':float(nll[take].mean())})
    frame=pd.DataFrame(qrows); frame.to_csv(out/'by_question.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'by_question_subgroups.csv',index=False)
    summary=frame.groupby(['model','qid'])[['error','nll','brier_two_class']].mean().groupby('model').mean()
    summary.to_csv(out/'summary.csv'); contrasts=paired_difference(frame,cfg); contrasts.to_csv(out/'paired.csv',index=False)
    pd.DataFrame(repro).to_csv(out/'cpu_reproduction.csv',index=False)
    pd.DataFrame(records).to_csv(out/'by_seed.csv',index=False)
    strata=frame.groupby(['model','qid','type','level'])[['error','nll','brier_two_class']].mean().reset_index()
    strata.groupby(['model','type'])[['error','nll']].agg(['mean','count']).to_csv(out/'by_topic.csv')
    strata.groupby(['model','level'])[['error','nll']].agg(['mean','count']).to_csv(out/'by_level.csv')
    # Fixed bins are specified before assessment. No threshold is chosen from
    # the eventual predictor scores, and subgroup intervals are not confirmatory.
    for key in ['pooled_entropy_bits','within_agent_entropy_bits','between_agent_entropy_bits']:
        sub=frame.groupby(['model','qid'])[[key,'error','nll']].mean().reset_index()
        sub['entropy_bin']=pd.cut(sub[key],protocol['entropy_reporting_bins_bits'],
            labels=['[0,0.25)','[0.25,0.50)','[0.50,0.75)','[0.75,1]'],right=False)
        sub.groupby(['model','entropy_bin'],observed=True)[['error','nll']].agg(['mean','count']).to_csv(out/f'by_{key}.csv')
    frame.groupby(['model','qid','distractor_format_valid'])[['error','nll']].mean().reset_index().groupby(
        ['model','distractor_format_valid'])[['error','nll']].agg(['mean','count']).to_csv(out/'by_distractor_format.csv')
    pd.DataFrame(groups).groupby(['model','qid','group'])[['transitions','error','nll']].mean().reset_index().groupby(
        ['model','group'])[['error','nll']].agg(['mean','count']).to_csv(out/'by_round_and_stay_switch.csv')
    # Earlier files, including the previous completed size sweep, remain frozen.
    prior=json.loads((ROOT/'prior_artifacts.json').read_text())
    for path,sha in prior.items(): assert digest(path)==sha,path
    training_common=dict(np.load(root/'development.npz',allow_pickle=False))
    training_features=dict(np.load(root/'folds/main/features.npz',allow_pickle=False))
    td=base.FoldData(training_common,training_features,'cpu'); train_ids=td.ids(protocol['fold']['train_questions'])
    development=[q for q in selected if q['split'] in ['train','validation']]
    population=pd.concat([population_summary(development,training_common),population_summary(questions,common)])
    population.to_csv(out/'population_by_question.csv',index=False)
    training_repro=[]
    for run in frozen['refits']:
        directory=Path(run['directory'])
        model=base.make_model(cfg,'nano')
        model.load_state_dict(torch.load(directory/'best.pt',map_location='cpu',weights_only=False)['model'])
        positions=np.linspace(0,len(train_ids)-1,min(64,len(train_ids)),dtype=int)
        expected=np.exp(np.load(directory/'train_logp.npy')[positions,...,1])
        actual=np.exp(base.predictions(model,td,train_ids[positions],False,64)[...,1])
        np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=5e-5)
        training_repro.append({'seed':run['seed'],'maximum_probability_difference':float(np.abs(actual-expected).max())})
    pd.DataFrame(training_repro).to_csv(out/'cpu_training_reproduction.csv',index=False)
    ee,tt=training_common['pairs'][train_ids].T
    train_baseline,_=base.metrics(baseline_logp(baseline,training_common['spins'][ee,tt],training_features['context'][ee],training_common['J'][ee]),td.y[train_ids],td.qids[train_ids])
    neural_train=[json.loads((Path(r['directory'])/'manifest.json').read_text())['train'] for r in frozen['refits']]
    lines=['# Fresh-question breadth comparison','',
           'Completed the predeclared comparison of the 818-parameter transformer and original 19-coefficient Ising model on 80 training, 20 validation and 40 independent assessment questions. Both use the same locally generated Llama data and the same 16-dimensional context. The source model differs from the earlier GPT study.','',
           '| Model | Training error | Assessment error | Assessment NLL | Assessment Brier |',
           '| --- | ---: | ---: | ---: | ---: |']
    for kind,label,tr in [('I0','Ising (19 coefficients)',train_baseline['error']),('nano','Transformer (818 parameters)',np.mean([x['error'] for x in neural_train]))]:
        row=summary.loc[kind]
        lines.append(f'| {label} | {100*tr:.2f}% | {100*row.error:.2f}% | {row.nll:.5f} | {row.brier_two_class:.5f} |')
    lines+=['','Differences below are transformer minus Ising; negative favors the transformer. Scores average three neural seeds within question, then questions equally; these are not ensemble predictions. Intervals use 5,000 paired question resamples. They describe this selected population and fixed training procedure, not uncertainty over new training datasets.','',
            '| Metric | Difference | 95% question interval |','| --- | ---: | ---: |']
    for r in contrasts.itertuples():
        scale=100 if r.metric=='error' else 1
        unit=' pp' if r.metric=='error' else ''
        lines.append(f'| {r.metric} | {scale*r.difference:+.5f}{unit} | [{scale*r.lower:+.5f}, {scale*r.upper:+.5f}]{unit} |')
    lines+=['','## Population and controls','',
        'The source bank is pinned MATH data. This first cohort accepts single rational numeric answers and excludes diagrams, unsupported answer forms, old study questions, persona worked examples and approximate near-duplicate templates. Each of seven math topics contributes 20 selected questions. A local Llama is asked to choose a distractor among four mathematically unequal options; failed strict-format responses use the first checked option, as declared before generation. Numerical inequality does not establish distractor plausibility. This construction differs from the paper\'s distractor generator.','',
        f'Of the 140 selected questions, {int((~population["distractor_format_valid"]).sum())} used the fallback distractor. Invalid vote count: {int(population["unparsed_votes"].sum())}; excluded agent-state snapshots: {int(population["excluded_snapshots"].sum())}. Population counts, unanimity, and observed switching are recorded per question.','',
        'Selection ranks pooled pre-interaction response entropy within topic and correct-option label after 8 samples from each of 32 agents. Within-agent variability and between-agent disagreement are recorded separately. Eight samples give noisy descriptive estimates; these are not irreducible error bounds. Screening data never enter predictor features. Split allocation balances topic, difficulty, correct labels and screening properties without looking at interaction outcomes or predictor scores.','',
        'Each question has one conversation, with one balanced assignment among eight familiar signed graphs, 32 fixed personas, eight synchronous updates and fresh five-vote majority targets. One sender message is broadcast to all its neighbors. This reduces replication per question relative to the released study. It does not test new graph families or autonomous rollouts.','',
        'The source is pinned 4-bit Llama-3.1-8B, with floating parameters/activations promoted to float32 for stable batching. Sampling uses temperature 0.7 and independent per-request/per-token RNG keys. Full prompts, token IDs, responses and parse retries are retained. Single-request and batched first-token probabilities were checked before collection.','',
        'The shared MiniLM text encoder is pretrained and frozen. Full text is windowed without dropping tokens. Question PCA3 fits only 80 training questions; persona PCA3 fits the 32 personas present in training. The 16 context features are a constant, three persona components, three question components and nine products. These preprocessing quantities and encoder weights are outside the 818 graph-predictor parameter count. Arguments are generated as part of the source process but are not inputs to either predictor.','',
        'The transformer chooses among nine learning-rate/logit-penalty candidates using only 20 validation questions at seed 42. Three fresh-seed models are then fitted on the same 80 questions for the frozen selected duration. Ising uses the pinned original two-stage fitting procedure. Assessment conversations are generated only after final weights freeze; assessment is never used for early stopping or model selection.','',
        'Per-question, topic, difficulty, fixed entropy-bin, distractor-format, round and stay/switch scores are saved under the assessment artifact directory. Subgroups are descriptive and may have few questions. Main accuracy means predicting the agent\'s next majority opinion, not solving the mathematics problem. Higher or lower scores than the old GPT study cannot be attributed solely to having more questions because the source, question population and replication budget differ.','',
        f'All three selected checkpoints reproduce independently on CPU on both training and assessment inputs; maximum probability difference {max(x["maximum_probability_difference"] for x in repro+training_repro):.2e}. All {len(prior):,} tracked prior artifacts remain unchanged. No paid source API calls were used.','',
        'Artifacts: `runs/question_breadth_v1/`; resume command: `.venv/bin/python scripts/breadth_run.py --hours <authorized-hours>`.']
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('\n'.join(lines)+'\n')
    artifacts={str(p):digest(p) for p in out.rglob('*') if p.is_file()}
    artifacts[str(REPORT)]=digest(REPORT)
    json_write(ROOT/'comparison_complete.json',{'status':'complete','completed_utc':now(),
               'models_frozen_sha256':digest(ROOT/'models_frozen.json'),'assessment_sources':sources,
               'assessment_questions':40,'train_questions':80,'validation_questions':20,
               'cpu_reproduced_models':3,'prior_artifacts_unchanged':len(prior),'artifacts':artifacts,
               'source_paid_api_calls':0,'scores':summary.reset_index().to_dict('records'),
               'paired':contrasts.to_dict('records')})
    print(summary.to_string()); print(contrasts.to_string(index=False))


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','fit','assess'])
    parser.add_argument('--device',choices=['cpu','mps'],default='mps')
    parser.add_argument('--deadline',type=float,default=None)
    args=parser.parse_args(); torch.set_num_threads(config()['training']['torch_threads'])
    if args.stage=='prepare': prepare(args.device)
    elif args.stage=='fit': fit(args.device,args.deadline or time.time()+5400)
    else: assess(args.device)


if __name__=='__main__': main()
