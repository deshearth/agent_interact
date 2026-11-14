import json
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.question_breadth import (
    config, distractors, last_boxed, normalize, rational_answer,
    screening_statistics, split_selected,
)
from scripts.breadth_collect import Recorder, authorize_split
from scripts.breadth_source import write_json
from scripts.transformer_generalization import make_model


def test_numeric_options_have_verified_distinct_values():
    for text in ['0','1','-3','0.125',r'\frac{2}{3}',r'-\frac{7}{4}','1,024']:
        value=rational_answer(text)
        wrong=distractors(value,'What is the answer?')
        assert len(wrong)==4
        parsed=[rational_answer(x) for x in wrong]
        assert None not in parsed and value not in parsed and len(set(parsed))==4
    assert rational_answer('0.125')==Fraction(1,8)
    assert rational_answer(r'\frac{1}{0}') is None
    assert rational_answer('1,2') is None
    assert rational_answer('x+1') is None
    assert rational_answer('12%') is None
    assert last_boxed(r'First \boxed{2}, finally \boxed{\frac{3}{4}}.')==r'\frac{3}{4}'


def test_screen_separates_disagreement_from_response_variability():
    polarized=np.array([[1]*8]*16+[[-1]*8]*16)
    variable=np.tile([1,-1]*4,(32,1))
    a,b=screening_statistics(polarized),screening_statistics(variable)
    assert a['pooled_entropy_bits']==b['pooled_entropy_bits']==1
    assert a['within_agent_entropy_bits']==0 and a['between_agent_entropy_bits']==1
    assert b['within_agent_entropy_bits']==1 and b['between_agent_entropy_bits']==0
    constant=screening_statistics(np.ones((32,8)))
    assert constant['pooled_entropy_bits']==0
    assert constant['within_agent_entropy_bits']==0
    with pytest.raises(ValueError): screening_statistics(np.zeros((32,8)))


def test_problem_family_normalization_detects_number_variants():
    assert normalize('Compute 12 + 45.',True)==normalize('Compute 76 + 81.',True)
    assert normalize('Compute 12 + 45.')!=normalize('Compute 76 + 81.')


def test_split_is_balanced_disjoint_and_independent_of_future_labels():
    rows=[]
    for topic in range(7):
        for label in ['A','B']:
            for j in range(10):
                qid=f'q{topic}_{label}_{j}'
                rows.append({'qid':qid,'family_key':qid,'type':str(topic),'level':str(j%5),
                    'answer':label,'screen':{'pooled_entropy_bits':.98+j*.001,'within_agent_entropy_bits':.2+j*.05},
                    'future_outcome':0})
    cfg=config(); assignment,diag=split_selected(rows,cfg)
    changed=[{**r,'future_outcome':1} for r in rows]
    assert split_selected(changed,cfg)==(assignment,diag)
    for name,total in [('train',80),('validation',20),('assessment',40)]:
        own=[r for r in rows if assignment[r['qid']]['split']==name]
        assert len(own)==total and sum(r['answer']=='A' for r in own)==total//2
        assert len({r['type'] for r in own})==7
    assert len(assignment)==140


class FakeSource:
    def __init__(self): self.calls=0
    def generate_many(self,prompts,identities,**kwargs):
        self.calls+=1
        return {'responses':[{'sample_id':sid,'raw_response':'A','parsed_sign':1} for sid in identities]}


def test_source_group_resume_reuses_samples_and_rejects_changed_prompt(tmp_path):
    write_json(tmp_path/'source_protocol.json',{'fixture':True})
    source=FakeSource(); recorder=Recorder(source,tmp_path,float('inf'))
    requests=[{'sample_id':'x','prompt':'Which option?','metadata':{'kind':'opinion'}}]
    # The progress timestamp needs a finite deadline; no wall-clock sleep.
    recorder.deadline=4102444800
    first=recorder.request('group',requests)
    assert recorder.request('group',requests)==first and source.calls==1
    with pytest.raises(AssertionError):
        recorder.request('group',[{**requests[0],'prompt':'A different question'}])


def test_assessment_generation_is_locked_until_verified_freeze(tmp_path):
    authorize_split(tmp_path,'train')
    with pytest.raises(RuntimeError): authorize_split(tmp_path,'assessment')
    model=tmp_path/'model'; model.write_text('frozen')
    from scripts.breadth_source import sha
    write_json(tmp_path/'models_frozen.json',{'artifacts':{str(model):sha(model)}})
    authorize_split(tmp_path,'assessment')
    model.write_text('changed')
    with pytest.raises(AssertionError): authorize_split(tmp_path,'assessment')


def test_predictor_remains_exactly_818_parameters():
    model=make_model(config(),'nano')
    assert sum(p.numel() for p in model.parameters())==818
