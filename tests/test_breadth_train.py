"""Synthetic integration checks, never empirical scores for the new study."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mac_agents.common import digest, json_write
from scripts import breadth_train as study
from scripts import transformer_generalization as base
from scripts.question_breadth import config


def test_predictor_text_excludes_answer_solution_and_screening():
    question={'question':'What is 4+5?', 'choices':{'A':'9','B':'8'},
              'answer':'A','solution':'Private solution', 'screen':{'pooled_entropy_bits':.9}}
    original=study.question_text(question)
    changed={**question,'answer':'B','solution':'Changed private solution',
             'screen':{'pooled_entropy_bits':0},'future_outcome':[-1]*32}
    assert study.question_text(changed)==original=='What is 4+5?\nA: 9\nB: 8'


def test_training_pca_applies_to_new_questions_without_refitting():
    rng=np.random.default_rng(51)
    q={f'q{i}':rng.normal(size=16) for i in range(12)}
    p=rng.normal(size=(32,16)); train=list(q)[:8]; ids=np.array(list(q))
    fitted=base.fit_context(q,p,train,ids)
    np.testing.assert_allclose(study.apply_context(np.array(list(q.values())),p,fitted),fitted['context'],atol=1e-7)
    changed={k:v.copy() for k,v in q.items()}
    for k in list(q)[8:]: changed[k]=rng.normal(size=16)*1000
    refit=base.fit_context(changed,p,train,ids)
    for key in ['question_mean','question_components','persona_mean','persona_components']:
        np.testing.assert_array_equal(fitted[key],refit[key])
    np.testing.assert_array_equal(fitted['context'][:8],refit['context'][:8])


def test_paired_comparison_averages_seeds_and_preserves_question_pairs():
    rows=[]
    for i in range(40):
        original=.2+i/1000
        rows.append({'model':'I0','qid':f'q{i:02d}','seed':-1,'error':original,'nll':original,'brier_two_class':original})
        for seed,offset in enumerate([-.03,-.02,-.01]):
            rows.append({'model':'nano','qid':f'q{i:02d}','seed':seed,
                         'error':original+offset,'nll':original+offset,'brier_two_class':original+offset})
    frame=pd.DataFrame(rows)
    result=study.paired_difference(frame,config())
    np.testing.assert_allclose(result[['difference','lower','upper']],-.02,atol=1e-15)
    pd.testing.assert_frame_equal(result,study.paired_difference(frame.sample(frac=1,random_state=7),config()))
    assert (result.questions_better==40).all()


def fixture_episode(root,q,protocol_sha):
    rng=np.random.default_rng(int(q['qid'][1:]))
    # All snapshots have valid majority labels, with enough label diversity for
    # the real author Ising fitter. This is artificial data, not LLM output.
    spins=np.where(rng.random((9,32))<.85,1,-1).astype(np.int8)
    graph=np.ones((32,32),np.int8)-np.eye(32,dtype=np.int8)
    graph[0,1]=graph[1,0]=-1
    path=root/'episodes'/q['split']/f"{q['qid']}__{q['graph_id']}__r00.json"
    json_write(path,{'meta':{'qid':q['qid'],'split':q['split'],'protocol_sha256':protocol_sha},
                    'spins_history':spins.tolist(),'spins_raw_history':np.repeat(spins[...,None],5,-1).tolist(),
                    'J':graph.tolist(),'completed_utc':study.now()})
    return path


def test_episode_reader_rejects_inconsistent_majority_targets(tmp_path):
    json_write(tmp_path/'source_protocol.json',{'synthetic_fixture':True})
    q={'qid':'q0','split':'train','graph_id':'J0'}
    path=fixture_episode(tmp_path,q,digest(tmp_path/'source_protocol.json'))
    common,_=study.load_episodes([q],tmp_path)
    assert common['pairs'].shape==(8,2)
    row=json.loads(path.read_text()); row['spins_history'][1][0]*=-1; json_write(path,row)
    with pytest.raises(AssertionError): study.load_episodes([q],tmp_path)


def test_full_pipeline_fits_freezes_then_assesses_without_touching_real_study(tmp_path,monkeypatch):
    root=tmp_path/'synthetic_study'; root.mkdir()
    cfg=copy.deepcopy(config())
    cfg['output']=str(root)
    cfg['training'].update(max_steps=4,validation_every_steps=2,patience=99,min_delta=0.,
        evaluation_batch_size=128,torch_threads=2,learning_rates=[.001,.003,.01])
    path=tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(study,'ROOT',root)
    monkeypatch.setattr(study,'CONFIG',path)
    monkeypatch.setattr(study,'REPORT',tmp_path/'synthetic_report.md')
    monkeypatch.setattr(study,'config',lambda:copy.deepcopy(cfg))
    monkeypatch.setattr(study,'verify_protocol',lambda:None)
    encoder=tmp_path/'fixture_encoder'; encoder.mkdir(); (encoder/'fixture.txt').write_text('synthetic embeddings only')
    monkeypatch.setattr(study,'ENCODER',encoder)
    calls=[]

    def encode(texts,device):
        calls.append(list(texts))
        vec=np.stack([np.random.default_rng(int.from_bytes(hashlib.sha256(t.encode()).digest()[:8],'little')).normal(size=384) for t in texts])
        return vec.astype(np.float32),{'synthetic_fixture':True,'text_count':len(texts),'tokens_dropped':0}

    monkeypatch.setattr(study,'encode_texts',encode)
    json_write(root/'source_protocol.json',{'synthetic_fixture':True})
    json_write(root/'prior_artifacts.json',{})
    questions=[]
    for i in range(140):
        split='train' if i<80 else 'validation' if i<100 else 'assessment'
        questions.append({'qid':f'q{i}','split':split,'question':f'Fixture question {i}',
            'choices':{'A':'1','B':'2'},'answer':'A' if i%2==0 else 'B',
            'graph_id':f'J{i%8}','type':str(i%7),'level':str(i%5),
            'screen':{'pooled_entropy_bits':.9,'within_agent_entropy_bits':.4,'between_agent_entropy_bits':.5},
            'distractor_construction':{'format_valid':bool(i%2)}})
    json_write(root/'selection.json',{'questions':questions})
    for q in questions[:100]: fixture_episode(root,q,digest(root/'source_protocol.json'))
    old_threads=torch.get_num_threads(); torch.set_num_threads(2)
    try:
        with pytest.raises(RuntimeError): study.authorize_split(root,'assessment')
        study.prepare('cpu')
        assert len(calls)==1 and len(calls[0])==132
        assert not any('Fixture question 100\n' in text for text in calls[0])
        before={str(p):digest(p) for p in (root/'predictors').rglob('*') if p.is_file()}
        study.prepare('cpu')
        assert len(calls)==1 and all(digest(p)==sha for p,sha in before.items())
        study.fit('cpu',time.time()+90)
        frozen=json.loads((root/'models_frozen.json').read_text())
        assert frozen['neural_parameters']==818 and frozen['ising_coefficients']==19
        assert len(frozen['refits'])==3 and not frozen['assessment_outcomes_observed']
        selected=json.loads((root/'predictors/selection.json').read_text())
        assert selected['refit_steps']>0
        for refit in frozen['refits']:
            manifest=json.loads((Path(refit['directory'])/'manifest.json').read_text())
            assert manifest['initial_weights_sha256']!=manifest['selected_weights_sha256']
            assert len(manifest['training_questions'])==80
            assert manifest['validation'] is None
            assert manifest['steps']==selected['refit_steps']
        study.authorize_split(root,'assessment')
        freeze_hash=digest(root/'models_frozen.json'); study.fit('cpu',time.time()+90)
        assert digest(root/'models_frozen.json')==freeze_hash
        # Assessment outcomes only become available after fitting and freezing.
        for q in questions[100:]: fixture_episode(root,q,digest(root/'source_protocol.json'))
        study.assess('cpu')
        assert len(calls)==2 and len(calls[1])==40
        result=json.loads((root/'comparison_complete.json').read_text())
        assert result['status']=='complete' and result['assessment_questions']==40
        assert all(digest(p)==sha for p,sha in frozen['artifacts'].items())
        assert pd.read_csv(root/'predictors/assessment/by_question.csv').shape[0]==160
        assert pd.read_csv(root/'predictors/assessment/population_by_question.csv').shape[0]==140
        assert pd.read_csv(root/'predictors/assessment/cpu_training_reproduction.csv').maximum_probability_difference.max()<1e-6
        completion_hash=digest(root/'comparison_complete.json'); study.assess('cpu')
        assert len(calls)==2 and digest(root/'comparison_complete.json')==completion_hash
    finally:
        torch.set_num_threads(old_threads)
