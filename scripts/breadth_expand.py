"""Fresh question screening and immutable extension from 80/20 to 320/80.

Preparation uses local pandas/sklearn. Selection and final collection checks use
the existing cluster runtime. No source responses or predictor errors select the
candidate bank; only declared pre-interaction screening selects conversations.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.hpc3_prepare import read,sha,write
from scripts import question_breadth as original

ROOT=Path("runs/question_breadth_320_v1")
PARENT=Path("runs/question_breadth_v1")
CONFIG=Path("configs/question_breadth_320.yaml")
CODE=[Path("scripts/breadth_expand.py"),Path("scripts/hpc3_breadth_expand.py"),
      Path("hpc3/breadth_expand.sbatch"),Path("hpc3/breadth_expand_finish.sbatch")]


def now():return datetime.now(timezone.utc).isoformat()


def cfg():return yaml.safe_load(CONFIG.read_text())


def make_quotas(questions,settings):
    development=[q for q in questions if q["split"] in ["train","validation"]]
    topics=sorted({q["type"] for q in development})
    assert len(topics)==7 and len(development)==100
    order=topics.copy();random.Random(settings["selection"]["quota_seed"]).shuffle(order)
    def topic_counts(total):
        return {t:total//7+int(order.index(t)<total%7) for t in topics}
    def cells(total):
        counts=topic_counts(total)
        odd=[t for t in order if counts[t]%2]
        assert len(odd)%2==0
        extra_a=set(odd[:len(odd)//2])
        return {t:{a:counts[t]//2+int(counts[t]%2 and ((a=="A")== (t in extra_a))) for a in ["A","B"]} for t in topics}
    target_dev=cells(400);target_val=cells(80)
    existing=Counter((q["type"],q["answer"]) for q in development)
    old_val=Counter((q["type"],q["answer"]) for q in development if q["split"]=="validation")
    new={t:{a:target_dev[t][a]-existing[t,a] for a in ["A","B"]} for t in topics}
    val={t:{a:target_val[t][a]-old_val[t,a] for a in ["A","B"]} for t in topics}
    candidates={t:{a:n*settings["selection"]["candidates_per_selected_question"] for a,n in labels.items()} for t,labels in new.items()}
    assert sum(n for labels in new.values() for n in labels.values())==300
    assert sum(n for labels in val.values() for n in labels.values())==60
    assert all(0<=val[t][a]<new[t][a] for t in topics for a in ["A","B"])
    assert all(sum(new[t][a] for t in topics)==150 for a in ["A","B"])
    assert all(sum(val[t][a] for t in topics)==30 for a in ["A","B"])
    return {"development":target_dev,"validation":target_val,"new_selected":new,"new_validation":val,"candidates":candidates}


def build_candidates(settings,quotas):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    source=PARENT/"source/math/competition_math.parquet"
    provenance=read(PARENT/"source/math/provenance.json")
    assert sha(source)==provenance["sha256"]
    previous_bank=read(PARENT/"candidate_bank.json")
    old=original.read_jsonl(original.UPSTREAM/"obj/train.jsonl")+original.read_jsonl(original.UPSTREAM/"obj/test.jsonl")
    personas=read(original.UPSTREAM/"obj/personas.json")
    excluded=[q["question"] for q in previous_bank]+[q["question"] for q in old]+[p["problem"] for p in personas]
    excluded_ids={q["qid"] for q in previous_bank}
    excluded_families={original.normalize(s,True) for s in excluded}
    frame=pd.read_parquet(source)
    eligible=[];reasons=Counter()
    for idx,row in frame.iterrows():
        qid=f"breadth_math_{idx:05d}"
        if qid in excluded_ids:reasons["previous_candidate"]+=1;continue
        question,solution=str(row["problem"]),str(row["solution"])
        boxed=original.last_boxed(solution);value=original.rational_answer(boxed)
        if value is None or abs(value)>10**12:reasons["unsupported_rational_answer"]+=1;continue
        if len(question)>6000 or "[asy]" in question.lower():reasons["diagram_or_long_question"]+=1;continue
        if original.normalize(question,True) in excluded_families:reasons["previous_family"]+=1;continue
        wrong=original.distractors(value,question)
        if not wrong:reasons["insufficient_distractors"]+=1;continue
        eligible.append({"qid":qid,"source_row":int(idx),"question":question,"solution":solution,
            "correct_answer":original.display_rational(value),"original_boxed_answer":boxed,
            "distractor_candidates":wrong,"type":str(row["type"]),"level":str(row["level"]),
            "family_key":hashlib.sha256(original.normalize(question,True).encode()).hexdigest()})
    corpus=[original.normalize(x,True) for x in excluded]+[original.normalize(q["question"],True) for q in eligible]
    matrix=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=1,max_features=60000).fit_transform(corpus)
    for i,q in enumerate(eligible):q["_vector_index"]=len(excluded)+i
    rng=random.Random(settings["selection"]["bank_seed"])
    topics=sorted(quotas["candidates"])
    pools={t:[q for q in eligible if q["type"]==t] for t in topics}
    for pool in pools.values():rng.shuffle(pool)
    chosen=list(range(len(excluded)));families=set();counts=Counter();bank=[];maximum_similarity=[]
    while len(bank)<1200:
        for topic in topics:
            for label in ["A","B"]:
                if counts[topic,label]>=quotas["candidates"][topic][label]:continue
                while pools[topic]:
                    q=pools[topic].pop()
                    if q["family_key"] in families:reasons["duplicate_new_family"]+=1;continue
                    similarity=float((matrix[q["_vector_index"]]@matrix[chosen].T).max())
                    if similarity>=settings["selection"]["near_duplicate_cosine"]:reasons["near_duplicate"]+=1;continue
                    chosen.append(q.pop("_vector_index"));families.add(q["family_key"])
                    q["answer"]=label;counts[topic,label]+=1;bank.append(q);maximum_similarity.append(similarity)
                    break
                else:raise RuntimeError(f"Insufficient independent candidates for {topic}/{label}")
    assert len({q["qid"] for q in bank})==len(families)==1200
    assert not {q["qid"] for q in bank}&excluded_ids
    return bank,{"source_sha256":sha(source),"source_revision":provenance["revision"],"eligible_fresh_questions":len(eligible),
        "excluded_previous_candidates":len(previous_bank),"excluded_old_questions":len(old),"excluded_persona_examples":len(personas),
        "filter_counts":dict(reasons),"maximum_retained_character_ngram_cosine":max(maximum_similarity),
        "level_counts":dict(Counter(q["level"] for q in bank)),"candidate_count":len(bank)}


def verify(root=ROOT):
    root=Path(root);protocol=read(root/"protocol.json")
    assert sha(root/"source_protocol.json")==sha(root/"protocol.json")
    assert sha(CONFIG)==protocol["config_sha256"]
    for path,digest in protocol["code_sha256"].items():assert sha(path)==digest,path
    for path,digest in protocol["source_files_sha256"].items():assert sha(path)==digest,path
    for name,digest in protocol["assets"].items():assert sha(root/name)==digest,name
    assert sha(PARENT/"selection.json")==protocol["parent_selection_sha256"]
    assert sha(PARENT/"source_protocol.json")==protocol["parent_source_protocol_sha256"]
    assert not list((PARENT/"episodes/assessment").glob("*.json"))
    assert not list((root/"episodes/assessment").glob("*.json"))
    return protocol


def prepare():
    from scripts import breadth_probability_learning_curve as prior
    prior.validate()
    settings=cfg()
    if (ROOT/"protocol.json").exists():verify();return
    parents=read(PARENT/"selection.json")["questions"]
    quotas=make_quotas(parents,settings)
    bank,metadata=build_candidates(settings,quotas)
    ROOT.mkdir(parents=True,exist_ok=True)
    write(ROOT/"candidate_bank.json",bank)
    write(ROOT/"candidate_preparation.json",metadata)
    write(ROOT/"quotas.json",quotas)
    shutil.copyfile(PARENT/"selection.json",ROOT/"parent_selection.json")
    retained=[]
    for q in parents:
        if q["split"]=="assessment":continue
        path=PARENT/"episodes"/q["split"]/f"{q['qid']}__{q['graph_id']}__r00.json"
        retained.append({"qid":q["qid"],"split":q["split"],"graph_id":q["graph_id"],
            "source_path":str(path),"sha256":sha(path),"source_protocol_sha256":sha(PARENT/"source_protocol.json")})
    write(ROOT/"retained_episodes.json",retained)
    assets=["candidate_bank.json","candidate_preparation.json","quotas.json","parent_selection.json","retained_episodes.json"]
    sources=["scripts/breadth_collect.py","scripts/breadth_source.py","scripts/question_breadth.py",
        "scripts/local_conversation_pilot.py","scripts/hpc3_prepare.py","scripts/hpc3_breadth.py",
        "scripts/hpc3_parallel_collect.py","hpc3/container.sh","data/raw/upstream/data/J_matrices.json",
        "data/raw/upstream/data/obj/personas.json"]
    protocol={"created_utc":now(),"config":settings,"config_sha256":sha(CONFIG),
        "code_sha256":{str(p):sha(p) for p in CODE},"source_files_sha256":{p:sha(p) for p in sources},
        "assets":{p:sha(ROOT/p) for p in assets},"parent_selection_sha256":sha(PARENT/"selection.json"),
        "parent_source_protocol_sha256":sha(PARENT/"source_protocol.json"),
        "parent_models_frozen_sha256":sha(PARENT/"models_frozen.json"),
        "new_questions":{"train":240,"validation":60},"target_questions":{"train":320,"validation":80},
        "candidate_count":1200,"reserved_assessment_qids":[q["qid"] for q in parents if q["split"]=="assessment"],
        "bank_policy":"Fresh MATH rows; original rational-answer/distractor rules. Exclude all 560 previous candidates, 40 released questions and 32 persona examples; number-masked families and character-ngram cosine >=.90 excluded across old/new bank. Correct A/B placement fixed before source screening.",
        "selection_policy":"Exactly four fresh candidates per added selected question in every topic/answer-label cell. Original pooled 32-persona by 8-vote entropy ranks within each cell; deterministic qid ties. No interaction outcomes or predictor errors enter selection. Do not use the 420 rejected old candidates to inflate sample size.",
        "split_policy":"Retain every original train/validation/assessment assignment. Add 240/60 questions with exact overall A/B balance and nearly equal topic coverage in final320/80. Among 2000 seeded within-cell splits choose level/screen-entropy balance, as declared before collection. Graphs J0-J7 have40 training and10 validation questions each in the final dataset.",
        "source_policy":"Original pinned Llama/MLX CUDA float32 source and five-group numerical replay gate. Screening votes are separate from fresh conversation votes. Each selected question has one assigned familiar graph, 32agents,8updates,5votes/state.",
        "interpretation":"A larger training corpus plus proportional validation. Existing20 validation questions remain a common comparison anchor; report them separately from60 new validation questions and all80. Same selection fraction reduces an easier-question confound but does not guarantee identical observed entropy distributions.",
        "quality_gate":"Before conversation generation, newly selected pooled entropy mean may be at most .02 bits below parent development, and within-agent entropy mean at most .10 bits below. Report all selection distributions; a failed gate stops collection for diagnosis.",
        "source_llm_paid_api_calls":0,"assessment_generated_or_scored":False,"automatic_retraining":False,
        "pipeline":"8 screening GPU jobs -> free CPU selection ->8 collection GPU jobs ->free CPU data integrity audit; dependencies enforce stage barriers. Frozen parent experiments are never rewritten."}
    write(ROOT/"protocol.json",protocol);shutil.copyfile(ROOT/"protocol.json",ROOT/"source_protocol.json")
    verify()
    print(json.dumps({"prepared":True,"candidate_count":1200,"new_conversations":300,"target":protocol["target_questions"],"candidate_audit":metadata},indent=2))


def choose_split(selected,parents,quotas,settings):
    retained=[{**q,"origin":"retained"} for q in parents if q["split"] in ["train","validation"]]
    combined=retained+selected
    levels=Counter(q["level"] for q in combined)
    fields=["pooled_entropy_bits","within_agent_entropy_bits"]
    means={f:float(np.mean([q["screen"][f] for q in combined])) for f in fields}
    scales={f:max(float(np.std([q["screen"][f] for q in combined])),.01) for f in fields}
    cells={(t,a):[q for q in selected if q["type"]==t and q["answer"]==a] for t in sorted(quotas["new_selected"]) for a in ["A","B"]}
    best=None
    for attempt in range(settings["selection"]["split_attempts"]):
        rng=random.Random(settings["selection"]["split_seed"]+attempt)
        parts={"train":[],"validation":[]}
        for (topic,label),values in cells.items():
            order=values.copy();rng.shuffle(order);n=quotas["new_validation"][topic][label]
            parts["validation"]+=order[:n];parts["train"]+=order[n:]
        loss=0.
        for split in parts:
            full=parts[split]+[q for q in retained if q["split"]==split]
            counts=Counter(q["level"] for q in full)
            for level,total in levels.items():
                expected=total*len(full)/400
                loss+=(counts[level]-expected)**2/max(expected,1.)
            for field in fields:loss+=((np.mean([q["screen"][field] for q in full])-means[field])/scales[field])**2
        if best is None or (loss,attempt)<best[:2]:best=(float(loss),attempt,parts)
    added=[]
    for split,part in best[2].items():
        existing_counts=Counter(q["graph_id"] for q in retained if q["split"]==split)
        total=settings["target_splits"][split]
        graph_schedule=[g for g in settings["source"]["graph_ids"] for _ in range(total//8-existing_counts[g])]
        assert len(graph_schedule)==len(part)
        random.Random(settings["selection"]["graph_seed"]+sum(map(ord,split))).shuffle(graph_schedule)
        added += [{**q,"split":split,"graph_id":g,"origin":"new"} for q,g in zip(sorted(part,key=lambda q:q["qid"]),graph_schedule)]
    return added,{"attempt":best[1],"imbalance_score":best[0]}


def select():
    protocol=verify();settings=protocol["config"]
    if (ROOT/"selection_ready.json").exists():return verify_selection()
    bank=read(ROOT/"candidate_bank.json");quotas=read(ROOT/"quotas.json")
    records=[];invalid=[]
    for row in bank:
        qpath=ROOT/"constructed"/(row["qid"]+".json")
        spath=ROOT/"screen"/(row["qid"]+".json")
        q,screen=read(qpath),read(spath)
        assert q["qid"]==screen["qid"]==row["qid"] and screen["question_sha256"]==sha(qpath)
        assert screen["protocol_sha256"]==sha(ROOT/"source_protocol.json")
        if screen["status"]!="complete":invalid.append(q["qid"]);continue
        stats=original.screening_statistics(screen["votes"])
        assert stats.keys()==screen["statistics"].keys()
        for key,value in stats.items():
            np.testing.assert_allclose(value,screen["statistics"][key],atol=1e-12,rtol=0,err_msg=key)
        records.append({**q,"screen":stats})
    selected=[]
    for topic,labels in quotas["new_selected"].items():
        for label,n in labels.items():
            part=sorted((q for q in records if q["type"]==topic and q["answer"]==label),key=lambda q:(-q["screen"]["pooled_entropy_bits"],q["qid"]))
            assert len(part)>=n,(topic,label,len(part),n)
            selected+=part[:n]
    assert len(selected)==300
    parents=read(ROOT/"parent_selection.json")["questions"]
    added,split_details=choose_split(selected,parents,quotas,settings)
    prior=[q for q in parents if q["split"] in ["train","validation"]]
    distributions={}
    for label,rows in [("retained",prior),("new",added),("combined",prior+added)]:
        distributions[label]={field:{"mean":float(np.mean([q["screen"][field] for q in rows])),
            "quantiles":np.quantile([q["screen"][field] for q in rows],[0,.1,.25,.5,.75,.9,1]).tolist()}
            for field in ["pooled_entropy_bits","within_agent_entropy_bits"]}
    passed=all(distributions["new"][f]["mean"]>=distributions["retained"][f]["mean"]+limit for f,limit in
        [("pooled_entropy_bits",settings["selection"]["minimum_new_pooled_entropy_mean_relative_to_parent"]),
         ("within_agent_entropy_bits",settings["selection"]["minimum_new_within_agent_entropy_mean_relative_to_parent"])])
    write(ROOT/"selection_diagnostic.json",{"created_utc":now(),"quality_gate_passed":passed,
        "entropy":distributions,"split":split_details,"invalid_screens":invalid,"candidate_screens":len(bank)})
    if not passed:raise RuntimeError("Expanded selection failed declared entropy-quality gate; conversations remain blocked")
    questions=sorted([{**q,"origin":"retained"} for q in parents]+added,key=lambda q:q["qid"])
    assert Counter(q["split"] for q in questions)==settings["target_splits"]
    assert len({q["qid"] for q in questions})==len({q["family_key"] for q in questions})==440
    for split,n in [("train",320),("validation",80)]:
        part=[q for q in questions if q["split"]==split]
        assert Counter(q["answer"] for q in part)=={"A":n//2,"B":n//2}
        assert Counter(q["graph_id"] for q in part)=={g:n//8 for g in settings["source"]["graph_ids"]}
    write(ROOT/"selection.json",{"protocol_sha256":sha(ROOT/"protocol.json"),"questions":questions,
        "new_questions":added,"uses_predictor_scores":False,"uses_interaction_outcomes":False,"assessment_generated":False})
    write(ROOT/"selection_ready.json",{"completed_utc":now(),"protocol_sha256":sha(ROOT/"protocol.json"),
        "selection_sha256":sha(ROOT/"selection.json"),"diagnostic_sha256":sha(ROOT/"selection_diagnostic.json"),
        "selected_new":300,"quality_gate_passed":True})
    print(json.dumps({"selection_ready":True,"new_counts":dict(Counter(q['split'] for q in added)),"entropy":distributions},indent=2))


def verify_selection():
    verify();ready=read(ROOT/"selection_ready.json")
    assert ready["protocol_sha256"]==sha(ROOT/"protocol.json") and ready["quality_gate_passed"]
    assert ready["selection_sha256"]==sha(ROOT/"selection.json")
    assert ready["diagnostic_sha256"]==sha(ROOT/"selection_diagnostic.json")
    return read(ROOT/"selection.json")


def episode_summary(q,path,graphs):
    assert q["split"] in ["train","validation"] and q["origin"] in ["retained","new"]
    episode=read(path)
    expected_sha=sha((PARENT if q["origin"]=="retained" else ROOT)/"source_protocol.json")
    expected={**{k:q[k] for k in ["qid","graph_id","split"]},"protocol_sha256":expected_sha,
              "name":f"{q['qid']}__{q['graph_id']}__r00","n":32,"k":5,"steps":8,"replica":0}
    assert all(episode["meta"].get(key)==value for key,value in expected.items())
    s=np.asarray(episode["spins_history"]);raw=np.asarray(episode["spins_raw_history"])
    assert s.shape==(9,32) and raw.shape==(9,32,5)
    assert np.isin(s,[-1,0,1]).all() and np.isin(raw,[-1,0,1]).all()
    summed=raw.sum(-1);np.testing.assert_array_equal(s,np.where(summed>0,1,np.where(summed<0,-1,raw[...,0])))
    np.testing.assert_array_equal(episode["J"],graphs[q["graph_id"]])
    assert len(episode["messages_history"])==len(episode["communicated_stances_history"])==8
    return {"qid":q["qid"],"split":q["split"],"graph_id":q["graph_id"],"origin":q["origin"],
        "path":str(path),"sha256":sha(path),"source_protocol_sha256":expected_sha,
        "valid_snapshots":int(((s[:-1]!=0)&(s[1:]!=0)).all(-1).sum()),"invalid_votes":int((raw==0).sum())}


def finalize():
    selection=verify_selection();graphs={g["id"]:g["J"] for g in read(original.UPSTREAM/"J_matrices.json")["graphs"]}
    rows=[]
    for q in selection["questions"]:
        if q["split"]=="assessment":continue
        path=ROOT/"episodes"/q["split"]/f"{q['qid']}__{q['graph_id']}__r00.json"
        rows.append(episode_summary(q,path,graphs))
    assert Counter(r["split"] for r in rows)=={"train":320,"validation":80}
    write(ROOT/"episode_index.json",rows)
    write(ROOT/"collection_complete.json",{"completed_utc":now(),"protocol_sha256":sha(ROOT/"protocol.json"),
        "selection_sha256":sha(ROOT/"selection.json"),"index_sha256":sha(ROOT/"episode_index.json"),
        "counts":dict(Counter(r["split"] for r in rows)),"new_conversations":300,"retained_conversations":100,
        "invalid_votes":sum(r["invalid_votes"] for r in rows),"assessment_generated_or_scored":False,
        "predictors_trained":False,"next_stage":"Compare818-parameter transformer and Ising with whole-question training/validation separation."})
    print(json.dumps(read(ROOT/"collection_complete.json")),flush=True)


if __name__=="__main__":
    parser=ArgumentParser();parser.add_argument("stage",choices=["prepare","verify","select","finalize"])
    stage=parser.parse_args().stage
    {"prepare":prepare,"verify":verify,"select":select,"finalize":finalize}[stage]()
