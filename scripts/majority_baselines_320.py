"""Unfitted voting baselines on the fixed 320/80 next-opinion task."""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import breadth_argument_study as shared

ROOT=Path('runs/majority_baselines_320_v1')
REFERENCE=Path('runs/compact_arguments_320_v1')
NAMES={
    'persistence':'Keep own current opinion',
    'neighbor_majority':'Majority of all communicating neighbors',
    'signed_neighbor_majority':'Signed majority of communicating neighbors',
    'friendly_majority':'Majority of friendly neighbors only',
    'global_majority':'Majority of all 32 agents',
    'training_majority':'Always predict the training majority class',
}

def read(path):return json.loads(Path(path).read_text())

def choose(score,current):
    """Zero votes, including empty inboxes, preserve the receiver's opinion."""
    return np.where(score>0,1,np.where(score<0,-1,current)).astype(np.int8)

def predict(current,graph,training_majority):
    # Cast before accumulation so signed int8 cannot overflow if reused at scale.
    state=current.astype(np.int64);g=graph.astype(np.int64)
    scores={
        'neighbor_majority':np.einsum('bij,bj->bi',np.abs(g),state),
        'signed_neighbor_majority':np.einsum('bij,bj->bi',g,state),
        'friendly_majority':np.einsum('bij,bj->bi',g>0,state),
        'global_majority':np.broadcast_to(state.sum(-1,keepdims=True),state.shape),
    }
    predictions={name:choose(value,state) for name,value in scores.items()}
    predictions['persistence']=current.copy()
    predictions['training_majority']=np.full_like(current,training_majority)
    return predictions,scores

def independently_check(current,graph,predictions,training_majority):
    """Direct integer loops verify sign, adjacency, exclusion and tie handling."""
    for b in range(len(current)):
        for i in range(current.shape[1]):
            unsigned=signed=friendly=0
            for j in range(current.shape[1]):
                edge=int(graph[b,i,j]);value=int(current[b,j])
                if edge:unsigned+=value;signed+=edge*value
                if edge>0:friendly+=value
            for name,value in [('neighbor_majority',unsigned),('signed_neighbor_majority',signed),
                               ('friendly_majority',friendly),('global_majority',sum(map(int,current[b])))]:
                expected=1 if value>0 else -1 if value<0 else current[b,i]
                assert predictions[name][b,i]==expected,(name,b,i)
            assert predictions['persistence'][b,i]==current[b,i]
            assert predictions['training_majority'][b,i]==training_majority

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    source_files=[REFERENCE/'protocol.json',REFERENCE/'prepared.json',REFERENCE/'data/development.npz',
        REFERENCE/'results/summary.json',REFERENCE/'results/predictions.json',Path(__file__)]
    old=read(REFERENCE/'protocol.json');fold=old['fold'];prepared=read(REFERENCE/'prepared.json')
    assert digest(REFERENCE/'data/development.npz')==prepared['artifacts']['data/development.npz']
    common=dict(np.load(REFERENCE/'data/development.npz',allow_pickle=False))
    e,t=common['pairs'].T;current=common['spins'][e,t];target=common['spins'][e,t+1];qids=common['episode_qids'][e]
    graph=common['J'][e];train=np.isin(qids,fold['train_questions']);val=np.isin(qids,fold['validation_questions'])
    assert not (train&val).any() and (train|val).all()
    assert not set(qids)&set(fold['assessment_questions'])
    assert common['valid'][e,t].all() and np.isin(current,[-1,1]).all() and np.isin(target,[-1,1]).all()
    assert np.isin(graph,[-1,0,1]).all() and not np.diagonal(graph,axis1=1,axis2=2).any()
    assert len(set(qids[train]))==320 and len(set(qids[val]))==80
    training_majority=1 if (target[train]==1).mean()>.5 else -1
    protocol={
        'created_utc':shared.now(),'inputs':{str(p):digest(p) for p in source_files},
        'fold':fold,'strategies':NAMES,'tie_policy':'Keep the receiving agent current opinion for an exactly tied vote or empty neighbor set.',
        'signed_rule':'Predict sign(sum_j J_ij * s_j), with J=+1 friendly, -1 unfriendly and 0 absent. This prescribes unit agreement/opposition weights.',
        'unsigned_rule':'Predict sign(sum_j abs(J_ij) * s_j); exclude self and nonneighbors.',
        'friendly_rule':'Only J_ij=+1 senders vote. No positive neighbors falls back to own current opinion.',
        'global_rule':'All 32 current agent opinions vote, including self and nonneighbors; extra information compared with the local communication rules.',
        'constant_rule':'The most frequent next-opinion label on the 320 training questions, chosen before validation scoring.',
        'training_majority_spin':training_majority,'target':'Each agent next observed majority-of-five A/B opinion, not mathematical correctness.',
        'evaluation':'All 80 reused validation questions, 8 updates x 32 agents each. Deterministic one-step predictions on observed states, no rollout or probability fitting.',
        'bootstrap_seed':94711,'bootstrap_draws':5000,'uncertainty':'Pointwise paired question bootstrap, conditional on frozen neural models. Exploratory reused validation.',
        'assessment_loaded':False,'source_llm_calls':0,'new_neural_fits':0,
    }
    if (ROOT/'protocol.json').exists():
        saved=read(ROOT/'protocol.json');protocol['created_utc']=saved['created_utc'];assert saved==protocol
    else:json_write(ROOT/'protocol.json',protocol)
    predictions,scores=predict(current,graph,training_majority)
    independently_check(current,graph,predictions,training_majority)
    # Synthetic isolated, negative-edge and exact-tie cases are explicitly covered.
    example=np.array([[1,-1,1,-1]],np.int8)
    edges=np.array([[[0,1,1,0],[1,0,-1,0],[1,-1,0,0],[0,0,0,0]]],np.int8)
    trial,_=predict(example,edges,-1);independently_check(example,edges,trial,-1)
    assert trial['neighbor_majority'][0,0]==1 and trial['signed_neighbor_majority'][0,1]==-1
    assert all(trial[n][0,3]==-1 for n in ['neighbor_majority','signed_neighbor_majority','friendly_majority'])
    rows=[];per_question={};reference_index=read(REFERENCE/'results/predictions.json');reference_arrays=0
    for split,take in [('train',train),('validation',val)]:
        questions=sorted(set(qids[take]));ys=target[take];states=current[take];qs=qids[take]
        for name in NAMES:
            pred=predictions[name][take];correct=pred==ys
            by_question=np.array([correct[qs==q].mean() for q in questions]);per_question[split,name]=by_question
            row={'split':split,'model':name,'accuracy':float(by_question.mean()),'questions':len(questions),
                'transitions':int(correct.size),'correct_count':int(correct.sum()),
                'stay_accuracy':float(correct[ys==states].mean()),'switch_accuracy':float(correct[ys!=states].mean())}
            if name in scores:row['tie_fraction']=float((scores[name][take]==0).mean())
            rows.append(row)
        for name in ('I0','opinion_nano'):
            question_scores=[]
            for record in reference_index:
                if record['model']!=name or record['split']!=split:continue
                assert digest(record['path'])==record['sha256']
                z=np.load(record['path'],allow_pickle=False)
                np.testing.assert_array_equal(z['pairs'],common['pairs'][take]);np.testing.assert_array_equal(z['qids'],qs)
                np.testing.assert_array_equal(z['target'],ys==1)
                correct=z['logp'].argmax(-1)==z['target']
                question_scores.append([correct[qs==q].mean() for q in questions]);reference_arrays+=1
            by_question=np.mean(question_scores,axis=0);per_question[split,name]=by_question
            expected=1-read(REFERENCE/'results/summary.json')['summary'][name][split]['error']
            assert abs(by_question.mean()-expected)<1e-12
            rows.append({'split':split,'model':name,'accuracy':float(by_question.mean()),'questions':len(questions),
                'transitions':int(ys.size),'seeds':len(question_scores)})
    questions=sorted(set(qids[val]));rng=np.random.default_rng(protocol['bootstrap_seed'])
    draws=rng.integers(0,len(questions),(protocol['bootstrap_draws'],len(questions)))
    contrasts=[]
    for name in NAMES:
        for reference in ('I0','opinion_nano'):
            delta=per_question['validation',reference]-per_question['validation',name]
            contrasts.append({'model':reference,'baseline':name,'accuracy_gain':float(delta.mean()),
                'ci95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist()})
    np.savez_compressed(ROOT/'predictions.npz',**predictions,current=current,target=target,qids=qids,pairs=common['pairs'])
    np.savez_compressed(ROOT/'by_question.npz',**{f'{s}__{n}':v for (s,n),v in per_question.items()})
    result={'completed_utc':shared.now(),'protocol_sha256':digest(ROOT/'protocol.json'),'rows':rows,'contrasts':contrasts,
        'validation_majority_A_fraction':float((target[val]==1).mean()),'training_majority_A_fraction':float((target[train]==1).mean()),
        'validation_isolated_fraction':float(((graph[val]!=0).sum(-1)==0).mean()),
        'validation_questions':80,'training_questions':320,'assessment_loaded':False,
        'tie_policy':protocol['tie_policy'],'likelihood_policy':'No NLL is assigned to deterministic rules; no confidence calibration was fitted.'}
    json_write(ROOT/'results.json',result)
    # Independent confusion-count score replay from the compressed output file.
    saved=np.load(ROOT/'predictions.npz',allow_pickle=False)
    for row in rows:
        if row['model'] not in NAMES:continue
        take=train if row['split']=='train' else val;pred=saved[row['model']][take];truth=saved['target'][take]
        tp=np.count_nonzero((pred==1)&(truth==1));tn=np.count_nonzero((pred==-1)&(truth==-1))
        assert abs((tp+tn)/truth.size-row['accuracy'])<1e-12
    for p,sha in protocol['inputs'].items():assert digest(p)==sha
    json_write(ROOT/'completion_audit.json',{'status':'passed','completed_utc':shared.now(),
        'all_rule_predictions_verified_by_direct_loop':True,'agent_updates_checked':int(target.size),
        'isolated_negative_edge_and_tie_checks':True,'split_scores_recomputed':12,'reference_prediction_arrays_verified':reference_arrays,
        'artifacts':{str(ROOT/n):digest(ROOT/n) for n in ['protocol.json','results.json','predictions.npz','by_question.npz']},
        'assessment_loaded':False,'source_llm_calls':0,'new_neural_fits':0})
    print(json.dumps({'training_majority':'A' if training_majority==1 else 'B','validation':[r for r in rows if r['split']=='validation'],
        'signed_majority_contrasts':[r for r in contrasts if r['baseline']=='signed_neighbor_majority']},indent=2))

if __name__=='__main__':main()
