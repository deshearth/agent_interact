"""Resumable construction, screening and conversations for question breadth."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.breadth_source import BreadthSource, sha, text_sha, write_json
from scripts.question_breadth import CONFIG, ROOT, UPSTREAM, config, screening_statistics, rational_answer
from scripts.local_conversation_pilot import formatter, opinion_prompt, message_prompt, deliver, aggregate


def now():
    return datetime.now(timezone.utc).isoformat()


class TimeLimit(Exception):
    pass


def verify_protocol(root=ROOT):
    root=Path(root)
    protocol=json.loads((root/'source_protocol.json').read_text())
    assert protocol['config_sha256']==sha(CONFIG)
    for path,digest in protocol['source_files'].items():
        assert sha(path)==digest,path
    return protocol


def freeze():
    cfg=config(); root=Path(cfg['output'])
    bench=json.loads((root/'source_benchmark.json').read_text())
    assert bench['status']=='complete' and bench['source_code_sha256']==sha('scripts/breadth_source.py')
    assert bench['max_serial_batch_first_token_probability_difference']<1e-3
    pa=[r['first_literal_A_probability'] for r in bench['eight_independent_samples']['responses']]
    assert max(pa)-min(pa)<1e-5
    prep=json.loads((root/'candidate_preparation.json').read_text())
    assert prep['code_sha256']==sha('scripts/question_breadth.py')
    assert prep['config_sha256']==sha(CONFIG) and prep['candidate_bank_sha256']==sha(root/'candidate_bank.json')
    model=json.loads(Path(cfg['source']['model_provenance']).read_text())
    for file,digest in model['files_sha256'].items():
        assert sha(Path(model['snapshot_path'])/file)==digest,file
    previous=Path('runs/transformer_micro_sweep_v1')
    old_audit=json.loads((previous/'completion_audit.json').read_text())
    prior=json.loads((previous/'prior_artifacts.json').read_text())
    prior.update({p:s for p,s in old_audit['artifacts'].items()
                  if p not in {'README.md','STATUS.md','mac_experiment_work_mode.md'}})
    prior[str(previous/'completion_audit.json')]=sha(previous/'completion_audit.json')
    for path,digest in prior.items(): assert sha(path)==digest,path
    write_json(root/'prior_artifacts.json',prior)
    source_files=[CONFIG,Path(__file__),Path('scripts/breadth_source.py'),Path('scripts/question_breadth.py'),
                  Path('scripts/local_argument_source.py'),Path('scripts/local_conversation_pilot.py'),
                  Path('scripts/argument_intervention_pilot.py'),UPSTREAM/'obj/personas.json',UPSTREAM/'J_matrices.json',
                  Path('data/raw/upstream/lib/datagen/samplers.py'),Path(cfg['source']['model_provenance']),
                  root/'candidate_bank.json',root/'candidate_preparation.json',root/'source/math/provenance.json',
                  root/'prior_artifacts.json']
    payload={'config':cfg,'config_sha256':sha(CONFIG),'source_files':{str(p):sha(p) for p in source_files},
             'source_benchmark_sha256':sha(root/'source_benchmark.json'),
             'model_repository':model['repository'],'model_revision':model['revision'],
             'precision':'4-bit weights; floating parameters/activations promoted to float32',
             'sampling':'Temperature .7, full vocabulary, per-request/per-token explicit RNG keys; EOS terminated full responses',
             'question_policy':'Single rational-answer MATH problems; locally chosen exactly unequal distractors; 80 candidates per topic. Rank pooled 32x8 pre-interaction entropy within topic/correct-label cells. Select10 per label/topic. No predictor errors or interaction outcomes select questions.',
             'split_policy':'80/20/40, exact A/B ground-truth balance, all seven topics in each split. Text-only approximate family exclusions across all candidates and the 40 old questions plus32 persona examples. Screening summaries balance split composition but never enter predictor inputs.',
             'untouched_assessment_meaning':'Questions and initial screening are used only to define the fixed target population and split; assessment interaction outcomes are not generated until model freeze, and never guide fitting or selection.',
             'collection_policy':'32 agents,8 synchronous updates,5 fresh votes per state,1 conversation per question,1 balanced familiar graph assignment per question. Broadcast one message per connected sender, preserving prescribed relationship signs.',
             'changes_from_released_gpt':'Different local Llama source and numerical precision, new rational-answer question bank, topic-stratified selection, local distractor construction, frozen MiniLM context encoder, one graph/replica per question. Scores must not be attributed to question count by comparing with old GPT scores.',
             'main_comparison':'818-parameter opinion/context graph transformer versus original 19-coefficient two-stage Ising. Same source data and16-dimensional context. NLL primary; error/Brier secondary. No argument inputs.',
             'paid_api_calls':0}
    if (root/'source_protocol.json').exists():
        old=json.loads((root/'source_protocol.json').read_text())
        assert all(old[k]==v for k,v in payload.items())
    else:
        write_json(root/'source_protocol.json',{**payload,'frozen_utc':now()})
    print(json.dumps({'source_protocol_frozen':True,'questions':[80,20,40],'candidates':560}),flush=True)


class Recorder:
    def __init__(self,source,root,deadline):
        self.source,self.root,self.deadline=source,Path(root),deadline
        self.protocol_sha256=sha(self.root/'source_protocol.json')
        self.new_samples=0

    def request(self,group_id,requests,max_tokens=256):
        path=self.root/'chunks'/f'{text_sha(group_id)}.json'
        identity={'group_id':group_id,'protocol_sha256':self.protocol_sha256,
                  'request_sha256':text_sha(json.dumps(requests,sort_keys=True,ensure_ascii=False)),
                  'max_tokens':max_tokens}
        if path.exists():
            old=json.loads(path.read_text())
            assert all(old[k]==v for k,v in identity.items())
            return old['responses']
        if time.time()>=self.deadline: raise TimeLimit()
        result=self.source.generate_many([r['prompt'] for r in requests],[r['sample_id'] for r in requests],
                                         batch_size=4,max_tokens=max_tokens)
        for request,row in zip(requests,result['responses']):
            assert request['sample_id']==row['sample_id']
            row.update(request['metadata'])
            row['prompt_sha256']=text_sha(request['prompt'])
        write_json(path,{**identity,**result})
        self.new_samples+=len(requests)
        write_json(self.root/'source_progress.json',{'updated_utc':now(),'group_id':group_id,
                   'new_samples_this_invocation':self.new_samples,'deadline_utc':datetime.fromtimestamp(self.deadline,timezone.utc).isoformat()})
        return result['responses']


def request_opinions(recorder,group_id,prompts,agents,vote_count,metadata):
    assert len(prompts)==len(agents)<=4
    requests=[]
    for vote in range(vote_count):
        for agent,prompt in zip(agents,prompts):
            requests.append({'sample_id':f'{group_id}|agent{agent}|vote{vote}|attempt0','prompt':prompt,
                             'metadata':{**metadata,'kind':'opinion','agent':agent,'vote_position':vote,'parse_attempt':0}})
    rows=recorder.request(group_id,requests)
    output={(r['agent'],r['vote_position']):r['parsed_sign'] for r in rows}
    # A replacement is used only for an unparseable response; all attempts remain saved.
    for attempt in [1,2]:
        pending=[]
        for (agent,vote),value in output.items():
            if value not in [-1,1]:
                pending.append({'sample_id':f'{group_id}|agent{agent}|vote{vote}|attempt{attempt}',
                                'prompt':prompts[agents.index(agent)],
                                'metadata':{**metadata,'kind':'opinion','agent':agent,'vote_position':vote,'parse_attempt':attempt}})
        if not pending: break
        replacements=recorder.request(f'{group_id}|retry{attempt}',pending)
        for row in replacements: output[row['agent'],row['vote_position']]=row['parsed_sign']
    return [[output[a,v] if output[a,v] in [-1,1] else 0 for v in range(vote_count)] for a in agents]


def construct(recorder,bank):
    for begin in range(0,len(bank),4):
        part=bank[begin:begin+4]
        if all((recorder.root/'constructed'/f"{q['qid']}.json").exists() for q in part): continue
        requests=[]
        for q in part:
            choices='\n'.join(f'{i+1}: {x}' for i,x in enumerate(q['distractor_candidates']))
            prompt=(f"Select the most plausible INCORRECT answer for a binary-choice version of this mathematics problem. "
                    f"The correct answer is supplied. Choose a distractor that could arise from a common calculation or method error. "
                    f"All four listed candidates have been checked to differ from the correct answer.\n\n"
                    f"Problem: {q['question']}\nCorrect answer: {q['correct_answer']}\nCandidates:\n{choices}\n\n"
                    "Return only the candidate index: 1, 2, 3, or 4.")
            requests.append({'sample_id':f"construct|{q['qid']}",'prompt':prompt,'metadata':{'kind':'distractor_selection','qid':q['qid']}})
        rows=recorder.request(f'construct|group{begin}',requests,max_tokens=64)
        for q,row in zip(part,rows):
            match=re.fullmatch(r'\s*([1-4])[.)]?\s*',row['raw_response']) if row['finish_reason']=='stop' else None
            selected=int(match[1])-1 if match else 0
            wrong=q['distractor_candidates'][selected]
            assert rational_answer(wrong)!=rational_answer(q['correct_answer'])
            choices={q['answer']:q['correct_answer'],('B' if q['answer']=='A' else 'A'):wrong}
            payload={**q,'choices':dict(sorted(choices.items())),'distractor':wrong,
                     'distractor_construction':{'sample_id':row['sample_id'],'selected_index':selected+1,
                                               'format_valid':bool(match),'fallback_if_invalid':'first exactly unequal candidate'}}
            write_json(recorder.root/'constructed'/f"{q['qid']}.json",payload)
        print(json.dumps({'stage':'construct','completed':begin+len(part),'planned':len(bank)}),flush=True)


def screen(recorder,bank,personas,fmt):
    for position,q0 in enumerate(bank):
        path=recorder.root/'screen'/f"{q0['qid']}.json"
        if path.exists(): continue
        qpath=recorder.root/'constructed'/f"{q0['qid']}.json"
        q=json.loads(qpath.read_text()); votes=[]
        for start in range(0,32,4):
            agents=list(range(start,start+4))
            prompts=[opinion_prompt(fmt,personas[a],q,[],[]) for a in agents]
            votes+=request_opinions(recorder,f"screen|{q['qid']}|group{start}",prompts,agents,8,
                                    {'qid':q['qid'],'phase':'screen','round':0})
        valid=all(x in [-1,1] for row in votes for x in row)
        write_json(path,{'qid':q['qid'],'question_sha256':sha(qpath),'votes':votes,
                        'status':'complete' if valid else 'invalid',
                        'statistics':screening_statistics(votes) if valid else None,
                        'protocol_sha256':recorder.protocol_sha256,'completed_utc':now()})
        print(json.dumps({'stage':'screen','completed':position+1,'planned':len(bank),'qid':q['qid'],
                          'status':'complete' if valid else 'invalid'}),flush=True)


def authorize_split(root,split):
    if split=='assessment':
        path=Path(root)/'models_frozen.json'
        if not path.exists():
            raise RuntimeError('Assessment conversation generation is locked until models are frozen')
        freeze=json.loads(path.read_text())
        for file,digest in freeze['artifacts'].items():
            assert sha(file)==digest,file


def run_episode(recorder,q,personas,graph,fmt,steps=8):
    name=f"{q['qid']}__{q['graph_id']}__r00"
    path=recorder.root/'episodes'/q['split']/f'{name}.json'
    if path.exists():
        old=json.loads(path.read_text())
        assert old['meta']['protocol_sha256']==recorder.protocol_sha256
        return old
    agree,disagree=[[] for _ in personas],[[] for _ in personas]
    spins,raw,messages,stances=[],[],[],[]
    for step in range(steps+1):
        if step:
            sender_messages={}; current_stances={}
            active=[a for a in range(32) if any(graph[a])]
            for start in range(0,len(active),4):
                agents=active[start:start+4]; requests=[]
                for agent in agents:
                    sid=f'{name}|round{step}|message|agent{agent}'
                    prompt,stance=message_prompt(fmt,personas[agent],q,agree[agent],disagree[agent],spins[-1][agent],sid)
                    current_stances[str(agent)]=stance
                    requests.append({'sample_id':sid,'prompt':prompt,'metadata':{'kind':'message','qid':q['qid'],
                                    'episode':name,'round':step,'agent':agent,'communicated_stance':stance}})
                rows=recorder.request(f'{name}|round{step}|messages{start}',requests)
                for agent,row in zip(agents,rows): sender_messages[agent]=row['raw_response'].strip()
            agree,disagree,edges=deliver(graph,sender_messages,name,step)
            messages.append(edges); stances.append(current_stances)
        votes=[]
        for start in range(0,32,4):
            agents=list(range(start,start+4))
            prompts=[opinion_prompt(fmt,personas[a],q,agree[a],disagree[a]) for a in agents]
            votes+=request_opinions(recorder,f'{name}|round{step}|opinions{start}',prompts,agents,5,
                                    {'qid':q['qid'],'episode':name,'phase':'interaction','round':step})
        raw.append(votes); spins.append([aggregate(v) for v in votes])
        write_json(recorder.root/'phase_checkpoints'/f'{name}.json',{'name':name,'completed_round':step,
                    'spins_history':spins,'spins_raw_history':raw,'messages_history':messages})
        print(json.dumps({'stage':'collect','split':q['split'],'episode':name,'completed_round':step}),flush=True)
    payload={'meta':{'name':name,'qid':q['qid'],'graph_id':q['graph_id'],'split':q['split'],
                     'n':32,'k':5,'steps':steps,'replica':0,'protocol_sha256':recorder.protocol_sha256},
             'mode':'objective','statement':q['question'],'choices':q['choices'],'personas':personas,'J':graph,
             'spins_history':spins,'spins_raw_history':raw,'messages_history':messages,
             'communicated_stances_history':stances,'completed_utc':now()}
    write_json(path,payload)
    return payload


def collect(recorder,split,personas,fmt,cfg):
    authorize_split(recorder.root,split)
    selection=json.loads((recorder.root/'selection.json').read_text())
    questions=[q for q in selection['questions'] if q['split']==split]
    assert len(questions)==cfg['questions']['splits'][split]
    graphs={g['id']:g['J'] for g in json.loads((UPSTREAM/'J_matrices.json').read_text())['graphs']}
    for q in questions:
        run_episode(recorder,q,personas,graphs[q['graph_id']],fmt,cfg['source']['update_rounds'])


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','construct','screen','collect'])
    parser.add_argument('--split',choices=['train','validation','assessment'])
    parser.add_argument('--deadline',type=float,default=None)
    args=parser.parse_args()
    if args.stage=='freeze': freeze(); return
    cfg=config(); root=Path(cfg['output']); verify_protocol(root)
    if args.stage=='collect':
        if args.split is None: raise ValueError('collect requires --split')
        authorize_split(root,args.split)
    deadline=args.deadline if args.deadline is not None else time.time()+7200
    with (root/'.source.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=BreadthSource(precision=cfg['source']['precision'])
        recorder=Recorder(source,root,deadline)
        bank=json.loads((root/'candidate_bank.json').read_text())
        personas=[p['persona'] for p in json.loads((UPSTREAM/'obj/personas.json').read_text())]
        started=now()
        try:
            if args.stage=='construct': construct(recorder,bank)
            elif args.stage=='screen': screen(recorder,bank,personas,formatter())
            else: collect(recorder,args.split,personas,formatter(),cfg)
        except TimeLimit:
            write_json(root/'source_session_result.json',{'status':'time_limit','stage':args.stage,'split':args.split,
                        'started_utc':started,'ended_utc':now(),'new_samples':recorder.new_samples})
            print('Collection time limit reached; complete request groups are saved.',flush=True)
            sys.exit(75)
        write_json(root/'stage_completion'/f"{args.stage}_{args.split or 'all'}.json",
                   {'status':'complete','protocol_sha256':recorder.protocol_sha256,'started_utc':started,
                    'completed_utc':now(),'new_samples_this_invocation':recorder.new_samples,'paid_api_calls':0})


if __name__=='__main__': main()
