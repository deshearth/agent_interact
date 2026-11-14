"""Question construction and frozen selection for the new 80/20/40 study."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.breadth_source import sha, write_json

CONFIG=Path('configs/question_breadth.yaml')
ROOT=Path('runs/question_breadth_v1')
UPSTREAM=Path('data/raw/upstream/data')


def config():
    return yaml.safe_load(CONFIG.read_text())


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def normalize(text, mask_numbers=False):
    text=text.lower().replace('\\n',' ').replace('\\\\','\\')
    text=re.sub(r'\\(?:left|right|displaystyle|quad|qquad|,|!|;)', '', text)
    text=re.sub(r'\s+',' ',text).strip()
    if mask_numbers:
        text=re.sub(r'\d+(?:\.\d+)?','#',text)
    return text


def last_boxed(solution):
    matches=list(re.finditer(r'\\(?:boxed|fbox)\s*\{',solution))
    if not matches:
        return None
    start=matches[-1].end(); depth=1
    for i in range(start,len(solution)):
        depth += (solution[i]=='{')-(solution[i]=='}')
        if depth==0:
            return solution[start:i].strip()
    return None


def rational_answer(text):
    if text is None:
        return None
    text=text.replace('\\dfrac','\\frac').replace('\\tfrac','\\frac')
    text=re.sub(r'\s+','',text).strip('$')
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+',text):
        text=text.replace(',','')
    match=re.fullmatch(r'([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}',text)
    if match:
        sign=-1 if match[1]=='-' else 1
        if int(match[3])==0:
            return None
        return sign*Fraction(int(match[2]),int(match[3]))
    if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)',text):
        return Fraction(text)
    if re.fullmatch(r'[+-]?\d+/[+-]?\d+',text):
        a,b=text.split('/')
        return Fraction(int(a),int(b)) if int(b) else None
    return None


def display_rational(value):
    value=Fraction(value)
    if value.denominator==1:
        return str(value.numerator)
    sign='-' if value<0 else ''
    return f'{sign}\\frac{{{abs(value.numerator)}}}{{{value.denominator}}}'


def distractors(value, question):
    value=Fraction(value)
    if 0<=value<=1 and ('probability' in question.lower() or value.denominator>1):
        proposed=[Fraction(value.numerator+1,value.denominator),
                  Fraction(value.numerator-1,value.denominator),
                  Fraction(value.numerator,value.denominator+1),
                  1-value,Fraction(1,2),Fraction(1,3),Fraction(2,3),Fraction(1,4),Fraction(3,4),Fraction(0),Fraction(1)]
        proposed=[p for p in proposed if 0<=p<=1]
    else:
        unit=Fraction(1) if value.denominator==1 else Fraction(1,value.denominator)
        proposed=[value-unit,value+unit,value-2*unit,value+2*unit,
                  value-3*unit,value+3*unit,value-4*unit,value+4*unit,2*value,value/2,-value]
        if value>=0:
            proposed=[p for p in proposed if p>=0]
    result=[]
    for p in proposed:
        if p!=value and p not in result:
            result.append(p)
    if len(result)<4:
        return None
    return [display_rational(x) for x in result[:4]]


def entropy_bits(p):
    p=np.asarray(p,dtype=float)
    result=np.zeros_like(p)
    mask=(p>0)&(p<1)
    result[mask]=-p[mask]*np.log2(p[mask])-(1-p[mask])*np.log2(1-p[mask])
    return result


def screening_statistics(votes):
    votes=np.asarray(votes)
    if votes.shape!=(32,8) or not np.isin(votes,[-1,1]).all():
        raise ValueError('Screen requires all 32 by 8 valid pre-interaction votes')
    probabilities=(votes==1).mean(axis=1)
    pooled=float(entropy_bits(probabilities.mean()))
    within=float(entropy_bits(probabilities).mean())
    return {'pA':float(probabilities.mean()),'pooled_entropy_bits':pooled,
            'within_agent_entropy_bits':within,
            'between_agent_entropy_bits':max(0.,pooled-within),
            'per_agent_pA':probabilities.tolist(),
            'agents_majority_A':int((probabilities>.5).sum()),
            'agents_majority_B':int((probabilities<.5).sum()),
            'agents_tied':int((probabilities==.5).sum())}


def prepare():
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    cfg=config(); root=Path(cfg['output']); root.mkdir(parents=True,exist_ok=True)
    if (root/'candidate_bank.json').exists():
        meta=json.loads((root/'candidate_preparation.json').read_text())
        assert meta['config_sha256']==sha(CONFIG)
        assert meta['code_sha256']==sha(__file__)
        assert meta['candidate_bank_sha256']==sha(root/'candidate_bank.json')
        print(json.dumps({'prepared':True,'reused':True,'candidates':meta['candidate_count']})); return
    path=root/'source/math/competition_math.parquet'
    provenance=json.loads((root/'source/math/provenance.json').read_text())
    assert provenance['sha256']==sha(path)
    assert provenance['revision']==cfg['questions']['source_revision']
    frame=pd.read_parquet(path)
    old=read_jsonl(UPSTREAM/'obj/train.jsonl')+read_jsonl(UPSTREAM/'obj/test.jsonl')
    personas=json.loads((UPSTREAM/'obj/personas.json').read_text())
    excluded=[r['question'] for r in old]+[p['problem'] for p in personas]
    exact_excluded={normalize(s) for s in excluded}
    template_excluded={normalize(s,True) for s in excluded}
    eligible=[]; reasons=Counter()
    for idx,row in frame.iterrows():
        question=str(row['problem']); solution=str(row['solution'])
        boxed=last_boxed(solution); value=rational_answer(boxed)
        if value is None or abs(value)>10**12:
            reasons['not_supported_single_rational_answer']+=1; continue
        if len(question)>6000 or '[asy]' in question.lower():
            reasons['diagram_or_long_question']+=1; continue
        if normalize(question) in exact_excluded or normalize(question,True) in template_excluded:
            reasons['previous_question_or_persona_family']+=1; continue
        wrong=distractors(value,question)
        if not wrong:
            reasons['insufficient_checked_distractors']+=1; continue
        eligible.append({'qid':f'breadth_math_{idx:05d}', 'source_row':int(idx),
                         'question':question,'solution':solution,'correct_answer':display_rational(value),
                         'original_boxed_answer':boxed,'distractor_candidates':wrong,
                         'type':str(row['type']),'level':str(row['level']),
                         'family_key':hashlib.sha256(normalize(question,True).encode()).hexdigest()})
    # Approximate family exclusion uses text only, before any source outcomes.
    corpus=[normalize(x,True) for x in excluded]+[normalize(r['question'],True) for r in eligible]
    matrix=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1,max_features=60000).fit_transform(corpus)
    for i,row in enumerate(eligible): row['_vector_index']=len(excluded)+i
    topics=sorted({r['type'] for r in eligible}); assert len(topics)==7
    rng=random.Random(cfg['questions']['bank_seed'])
    pools={topic:[r for r in eligible if r['type']==topic] for topic in topics}
    for rows in pools.values(): rng.shuffle(rows)
    selected=[]; chosen_indices=list(range(len(excluded))); families=set(); counts=Counter()
    target=cfg['questions']['candidate_questions_per_topic']
    while any(counts[t]<target for t in topics):
        for topic in topics:
            if counts[topic]>=target: continue
            while pools[topic]:
                row=pools[topic].pop()
                if row['family_key'] in families:
                    reasons['duplicate_candidate_family']+=1; continue
                similarities=(matrix[row['_vector_index']]@matrix[chosen_indices].T).toarray().ravel()
                if similarities.max(initial=0)>=cfg['questions']['near_duplicate_cosine']:
                    reasons['near_duplicate_question_or_persona']+=1; continue
                # Correct-label assignment precedes screening and is balanced within topic.
                row['answer']='A' if counts[topic]%2==0 else 'B'
                counts[topic]+=1; families.add(row['family_key'])
                chosen_indices.append(row.pop('_vector_index')); selected.append(row); break
            else:
                raise RuntimeError(f'Insufficient independent candidates for {topic}: {counts[topic]}')
    assert len(selected)==560 and len(families)==560
    write_json(root/'candidate_bank.json',selected)
    write_json(root/'candidate_preparation.json',{'config_sha256':sha(CONFIG),'code_sha256':sha(__file__),
        'candidate_bank_sha256':sha(root/'candidate_bank.json'),'source_sha256':sha(path),
        'source_rows':len(frame),'eligible_rational_questions':len(eligible),'candidate_count':len(selected),
        'topic_counts':dict(counts),'level_counts':dict(Counter(r['level'] for r in selected)),
        'filter_counts':dict(reasons),'excluded_previous_questions':len(old),'excluded_persona_examples':len(personas),
        'family_policy':'Exact number-masked template and character-ngram cosine >=.90 exclusions across candidates, prior questions and persona examples. This is an approximate duplicate guard, not a guarantee of semantic independence.',
        'scope_limit':'Single rational numeric answers; diagrams and unsupported symbolic/multicomponent answers excluded. This differs from the paper\'s unrestricted answer forms.',
        'selection_policy':'Paper-inspired pooled response entropy, ranked separately within topic and correct-option label; original paper does not fully specify entropy aggregation or ties. Topic quotas are our extension.',
        'screening_started':False,'new_source_responses':0})
    print(json.dumps({'candidates':len(selected),'topic_counts':dict(counts),'eligible':len(eligible),'filtered':dict(reasons)}),flush=True)


def split_selected(rows,cfg):
    assert len(rows)==140 and len({r['family_key'] for r in rows})==140
    by_label={a:[r for r in rows if r['answer']==a] for a in ['A','B']}
    assert all(len(x)==70 for x in by_label.values())
    sizes=cfg['questions']['splits']; assert sizes=={'train':80,'validation':20,'assessment':40}
    properties=['type','level']
    totals={key:Counter(r[key] for r in rows) for key in properties}
    best=None
    for attempt in range(2000):
        rng=random.Random(cfg['questions']['split_seed']+attempt)
        splits={name:[] for name in sizes}
        for label in ['A','B']:
            shuffled=by_label[label].copy(); rng.shuffle(shuffled); pos=0
            for name,size in sizes.items():
                splits[name]+=shuffled[pos:pos+size//2]; pos+=size//2
        if any(len({r['type'] for r in values})!=7 for values in splits.values()): continue
        score=0.
        for name,part in splits.items():
            for prop in properties:
                actual=Counter(r[prop] for r in part)
                for category,total in totals[prop].items():
                    expected=total*len(part)/len(rows)
                    score+=(actual[category]-expected)**2/max(expected,1.)
            for field in ['pooled_entropy_bits','within_agent_entropy_bits']:
                all_values=np.array([r['screen'][field] for r in rows])
                own=np.array([r['screen'][field] for r in part])
                scale=max(float(all_values.std()),.01)
                score+=((float(own.mean()-all_values.mean()))/scale)**2
        candidate=(score,attempt,splits)
        if best is None or candidate[:2]<best[:2]: best=candidate
    if best is None: raise RuntimeError('Cannot construct all-topic split')
    result={}
    for name,part in best[2].items():
        ordered=sorted(part,key=lambda r:r['qid'])
        graph_ids=cfg['source']['graph_ids']
        schedule=[graph_ids[i%len(graph_ids)] for i in range(len(part))]
        random.Random(cfg['questions']['graph_seed']+sum(map(ord,name))).shuffle(schedule)
        for row,graph in zip(ordered,schedule):
            result[row['qid']]={'split':name,'graph_id':graph}
    return result,{'search_attempt':best[1],'imbalance_score':best[0]}


def select():
    cfg=config(); root=Path(cfg['output'])
    bank=json.loads((root/'candidate_bank.json').read_text())
    records=[]; invalid=[]
    for row in bank:
        q=json.loads((root/'constructed'/f"{row['qid']}.json").read_text())
        screen=json.loads((root/'screen'/f"{row['qid']}.json").read_text())
        assert screen['qid']==q['qid'] and screen['question_sha256']==sha(root/'constructed'/f"{row['qid']}.json")
        if screen['status']!='complete':
            invalid.append(q['qid']); continue
        records.append({**q,'screen':screening_statistics(screen['votes'])})
    picked=[]
    for topic in sorted({r['type'] for r in records}):
        for label in ['A','B']:
            part=[r for r in records if r['type']==topic and r['answer']==label]
            assert len(part)>=10,(topic,label,len(part))
            picked+=sorted(part,key=lambda r:(-r['screen']['pooled_entropy_bits'],r['qid']))[:10]
    assert len(picked)==140
    assignment,diagnostic=split_selected(picked,cfg)
    payload={'config_sha256':sha(CONFIG),'candidate_bank_sha256':sha(root/'candidate_bank.json'),
             'questions':[{**r,**assignment[r['qid']]} for r in sorted(picked,key=lambda r:r['qid'])],
             'split_diagnostic':diagnostic,'selection_uses_interaction_outcomes':False,
             'selection_uses_predictor_scores':False,
             'excluded_incomplete_screens':sorted(invalid),
             'assessment_interaction_outcomes_collected':False}
    if (root/'selection.json').exists():
        assert json.loads((root/'selection.json').read_text())==payload
    else:
        write_json(root/'selection.json',payload)
    print(json.dumps({'selected':140,'splits':dict(Counter(x['split'] for x in payload['questions'])),
                      'pooled_entropy_range':[min(r['screen']['pooled_entropy_bits'] for r in picked),max(r['screen']['pooled_entropy_bits'] for r in picked)]}),flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','select'])
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else select)()
