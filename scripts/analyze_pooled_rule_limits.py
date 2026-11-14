"""Empirical limits of predicting the frozen teacher from pooled features.

Group observed states with exactly equal 21-dimensional pooled inputs. The
resulting oracle floors are descriptive constraints, not fitted generalization
scores and not estimates of irreducible LLM sampling error.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import xlogy
from threadpoolctl import threadpool_limits
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config,digest,json_write
from scripts import transformer_generalization as base
from scripts.distill_transformer_rule import CONFIG,equal_question_weights,features,validate


def group_inputs(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1,x.shape[-1])
    assert np.isfinite(x).all()
    _,inverse,counts=np.unique(x,axis=0,return_inverse=True,return_counts=True)
    return inverse,counts


def grouped_floors(p,inverse,counts,weights):
    p=np.asarray(p,dtype=np.float64).reshape(-1).clip(1e-12,1-1e-12)
    assert len(p)==len(inverse)==len(weights) and np.isclose(weights.sum(),1.)
    mass=np.bincount(inverse,weights=weights,minlength=len(counts))
    mean=np.bincount(inverse,weights=weights*p,minlength=len(counts))/mass
    order=np.lexsort((p,inverse)); ordered_groups=inverse[order]
    starts=np.r_[0,np.flatnonzero(np.diff(ordered_groups))+1]
    cumulative=np.cumsum(weights[order])
    preceding=np.r_[0.,cumulative[starts[1:]-1]]
    within=cumulative-np.repeat(preceding,counts)
    candidates=np.where(within>=.5*mass[ordered_groups],np.arange(len(p)),len(p))
    med_positions=np.minimum.reduceat(candidates,starts)
    assert (med_positions<len(p)).all()
    median=p[order[med_positions]]
    positive_mass=np.bincount(inverse,weights=weights*(p>.5),minlength=len(counts))
    low=np.full(len(counts),np.inf); high=np.full(len(counts),-np.inf)
    np.minimum.at(low,inverse,p); np.maximum.at(high,inverse,p)
    span=high-low; largest=int(span.argmax())
    members=np.flatnonzero(inverse==largest)
    low_row=int(members[np.argmin(p[members])]); high_row=int(members[np.argmax(p[members])])
    q=mean[inverse]
    kl=xlogy(p,p)+xlogy(1-p,1-p)-xlogy(p,q)-xlogy(1-p,1-q)
    return {'states':len(p),'distinct_pooled_groups':len(counts),
            'repeated_group_state_fraction':float(weights[counts[inverse]>1].sum()),
            'groups_with_range_over_1pp':int((span>.01).sum()),
            'state_fraction_in_groups_with_range_over_1pp':float(weights[span[inverse]>.01].sum()),
            'minimum_probability_mae':float(weights@np.abs(p-median[inverse])),
            'minimum_probability_rmse':float(np.sqrt(weights@((p-q)**2))),
            'minimum_teacher_kl':float(weights@kl),
            'minimum_teacher_label_disagreement':float(np.minimum(positive_mass,mass-positive_mass).sum()),
            'largest_group_probability_range':float(span[largest]),
            'example_low_row':low_row,'example_high_row':high_row,
            'example_low_probability':float(p[low_row]),'example_high_probability':float(p[high_row]),
            'example_group_count':int(counts[largest])}


def main():
    cfg=config(CONFIG); protocol=validate(cfg); root=Path(cfg['output']); source=Path(cfg['source'])
    out=root/'pooled_state_limits'; out.mkdir(parents=True,exist_ok=True)
    identity={'code_sha256':digest(__file__),'parent_protocol_sha256':digest(root/'protocol.json'),
              'distillation_evaluation_sha256':digest(root/'evaluation_complete.json'),
              'purpose':'Follow-up after the four surrogate families left about 5pp probability MAE; distinguish information lost by pooling from the tested parametric restrictions.',
              'grouping':'Exactly equal 21 pooled input features; excludes arbitrary episode, question, graph and agent IDs and all outcomes. Folds and splits kept separate.',
              'interpretation':'Finite-observation oracle floors computed using teacher outputs within each group. These are not usable heldout predictors or source-model noise estimates. The MAE, RMSE/KL and classification minima may require different predictions.',
              'new_assessment_data_read':False,'source_calls':0,'cpu_threads':1}
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        old=json.loads(protocol_path.read_text()); assert all(old[k]==v for k,v in identity.items())
    else: json_write(protocol_path,{**identity,'created_utc':datetime.now(timezone.utc).isoformat()})
    common=dict(np.load(source/'development.npz',allow_pickle=False)); e,t=common['pairs'].T
    qids=common['episode_qids'][e]
    index=json.loads((source/'episode_index.json').read_text())
    rows=[]; examples=[]
    for fold in protocol['folds']:
        fitted=dict(np.load(source/'folds'/fold['id']/'features.npz',allow_pickle=False))
        data=base.FoldData(common,fitted,'cpu')
        for split,questions in [('train',fold['train_questions']),('outer',fold['assessment_questions'])]:
            ids=np.flatnonzero(np.isin(qids,questions))
            x,names=features(common['spins'][e[ids],t[ids]],fitted['context'][e[ids]],common['J'][e[ids]],fitted['prior_logit'][ids],'pooled')
            inverse,counts=group_inputs(x); weights=equal_question_weights(qids[ids],32)
            for teacher in [r for r in protocol['teachers'] if r['fold']==fold['id']]:
                path=Path(teacher['directory'])/'train_logp.npy' if split=='train' else Path(teacher['assessment_directory'])/'assessment_logp.npy'
                lp=np.load(path,allow_pickle=False); p=np.exp(lp[...,1])
                # Use float64 exponentiation consistently with the surrogate study.
                p=np.exp(lp[...,1].astype(np.float64))
                floors=grouped_floors(p,inverse,counts,weights)
                row={'outer':fold['outer'],'seed':teacher['seed'],'split':split,**floors}; rows.append(row)
                if split!='outer': continue
                a,b=floors['example_low_row'],floors['example_high_row']
                flat=x.reshape(-1,21)
                np.testing.assert_array_equal(flat[a],flat[b])
                cases=[]
                for label,flat_id in [('low',a),('high',b)]:
                    snap,node=divmod(flat_id,32); pair=ids[snap]; episode,round_=common['pairs'][pair]
                    graph=common['J'][episode]; current=common['spins'][episode,round_]
                    cases.append({'case':label,'qid':str(qids[pair]),'episode':int(episode),'round':int(round_),'receiver':node,
                        'graph_id':index[episode]['graph_id'],'replica':index[episode]['replica'],
                        'teacher_probability_A':float(p[snap,node]),'common_pooled_features':dict(zip(names,flat[flat_id].tolist())),
                        'friendly_neighbors':[{'agent':int(j),'opinion':int(current[j])} for j in np.flatnonzero(graph[node]>0)],
                        'unfriendly_neighbors':[{'agent':int(j),'opinion':int(current[j])} for j in np.flatnonzero(graph[node]<0)]})
                neural=base.make_model(config(cfg['source_config']),cfg['teacher'])
                neural.load_state_dict(torch.load(Path(teacher['directory'])/'best.pt',map_location='cpu',weights_only=False)['model'])
                positions=np.array([ids[a//32],ids[b//32]])
                cpu=np.exp(base.predictions(neural,data,positions,False,2)[...,1])
                cpu_pair=np.array([cpu[0,a%32],cpu[1,b%32]])
                saved_pair=np.array([floors['example_low_probability'],floors['example_high_probability']])
                np.testing.assert_allclose(cpu_pair,saved_pair,atol=5e-5,rtol=5e-5)
                examples.append({'outer':fold['outer'],'seed':teacher['seed'],'cases':cases,
                    'cpu_probabilities':cpu_pair.tolist(),'maximum_cpu_difference':float(np.abs(cpu_pair-saved_pair).max()),
                    'same_recorded_pooled_inputs':True})
    frame=pd.DataFrame(rows); frame.to_csv(out/'by_teacher_and_split.csv',index=False)
    numeric=[c for c in frame.columns if c.startswith('minimum_') or c in ['distinct_pooled_groups','largest_group_probability_range','state_fraction_in_groups_with_range_over_1pp']]
    frame.groupby('split')[numeric].mean().to_csv(out/'summary.csv')
    json_write(out/'natural_state_examples.json',examples)
    validate(cfg)
    json_write(out/'complete.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
        'teachers':12,'examples_reproduced_on_cpu':12,'protocol_sha256':digest(protocol_path),
        'artifacts':{str(p):digest(p) for p in [out/'by_teacher_and_split.csv',out/'summary.csv',out/'natural_state_examples.json']}})
    print(frame.groupby('split')[numeric].mean().to_string())


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): main()
