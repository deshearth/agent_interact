"""Four isolated free-GPU workers for the paired unseen-graph experiment.

The pinned source generator, original collection and existing weights are never
modified. Initial votes are reused with explicit provenance. Fresh chunks and
episodes are published only inside runs/unseen_graph_v1. The last successful
worker scores the complete cohort automatically with frozen NumPy predictors.
"""
from argparse import ArgumentParser
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.hpc3_prepare import read,sha,write
from scripts import unseen_graph_experiment as experiment
from scripts import hpc3_breadth as cuda
from scripts import hpc3_parallel_collect as parallel

ROOT=experiment.ROOT
SOURCE=experiment.SOURCE


def chunk_path(root,group):
    return Path(root)/"chunks"/(hashlib.sha256(group.encode()).hexdigest()+".json")


def prepare_cluster():
    protocol=experiment.validate()
    assert sha(SOURCE/"source_protocol.json")==protocol["original_source_protocol_sha256"]
    assert sha(SOURCE/"selection.json")==protocol["original_selection_sha256"]
    assert not list((SOURCE/"episodes/assessment").glob("*.json"))
    if (ROOT/"deployment.json").exists():
        verify_deployment()
        return read(ROOT/"deployment.json")
    report_path=protocol["config"]["collection"]["benchmark_report"]
    manifest=cuda.verify_package()
    report,fixtures=read(report_path),read("hpc3/fixtures.json")
    cuda.check_collection_gate(manifest,report,report["runtime"],fixtures)
    parallel.check_replay_files(report)
    initial_files={}
    for q in protocol["questions"]:
        original=f"{q['qid']}__{q['seen_graph_id']}__r00"
        episode=SOURCE/"episodes/validation"/(original+".json")
        assert sha(episode)==protocol["paired_episode_sha256"][q["qid"]]
        for start in range(0,32,4):
            group=f"{original}|round0|opinions{start}"
            path=chunk_path(SOURCE,group)
            chunk=read(path)
            assert chunk["group_id"]==group and chunk["protocol_sha256"]==protocol["original_source_protocol_sha256"]
            assert len(chunk["responses"])==20
            assert all(r["parsed_sign"] in [-1,1] and r["parse_attempt"]==0 for r in chunk["responses"])
            initial_files[str(path)]=sha(path)
    workers=protocol["config"]["collection"]["workers"]
    owned=[protocol["questions"][i::workers] for i in range(workers)]
    deployment={"prepared_utc":experiment.now(),"protocol_sha256":sha(ROOT/"protocol.json"),
        "benchmark_report":report_path,"benchmark_report_sha256":sha(report_path),
        "initial_chunk_sha256":initial_files,
        "assignments":[[q["qid"] for q in part] for part in owned],
        "workers":workers,"assessment_included":False,"initial_votes_reused":20*32*5}
    for i in range(workers):
        directory=ROOT/"workers"/f"worker_{i:02d}"
        parallel.publish_immutable(ROOT/"source_protocol.json",directory/"source_protocol.json")
    write(ROOT/"deployment.json",deployment)
    verify_deployment()
    return deployment


def verify_deployment():
    protocol=experiment.validate()
    deployment=read(ROOT/"deployment.json")
    assert deployment["protocol_sha256"]==sha(ROOT/"protocol.json")
    assert deployment["benchmark_report_sha256"]==sha(deployment["benchmark_report"])
    assert not deployment["assessment_included"]
    assert sha(SOURCE/"source_protocol.json")==protocol["original_source_protocol_sha256"]
    assert sha(SOURCE/"selection.json")==protocol["original_selection_sha256"]
    for path,digest in deployment["initial_chunk_sha256"].items():assert sha(path)==digest,path
    assigned=sum(deployment["assignments"],[])
    assert len(assigned)==len(set(assigned))==20
    assert set(assigned)==set(protocol["validation_question_ids"])
    return protocol,deployment


def initial_responses(group,requests,question,protocol,initial_hashes,source_root=None):
    """Remap identifiers, not sampled content; retain original RNG provenance."""
    source_root=SOURCE if source_root is None else Path(source_root)
    name,phase,suffix=group.split("|")
    assert name==experiment.episode_name(question) and phase=="round0" and suffix.startswith("opinions")
    original=f"{question['qid']}__{question['seen_graph_id']}__r00|{phase}|{suffix}"
    path=chunk_path(source_root,original)
    assert sha(path)==initial_hashes[str(path)]
    chunk=read(path)
    assert chunk["group_id"]==original and chunk["protocol_sha256"]==protocol["original_source_protocol_sha256"]
    lookup={(r["agent"],r["vote_position"],r["parse_attempt"]):r for r in chunk["responses"]}
    assert len(lookup)==len(requests)==20
    output=[]
    for request in requests:
        meta=request["metadata"]
        row=copy.deepcopy(lookup[(meta["agent"],meta["vote_position"],meta["parse_attempt"])])
        assert row["prompt_sha256"]==hashlib.sha256(request["prompt"].encode()).hexdigest()
        assert row["parsed_sign"] in [-1,1]
        row["reused_source_sample_id"]=row["sample_id"]
        row["reused_source_sample_seed"]=row.pop("sample_seed",None)
        row["reused_source_chunk"]=str(path)
        row["reused_source_chunk_sha256"]=initial_hashes[str(path)]
        row["sample_id"]=request["sample_id"]
        row.update(meta)
        output.append(row)
    return output,{"reused_source_group":original,"reused_source_chunk_sha256":sha(path)}


def make_recorder(source,root,deadline,questions,protocol,deployment,execution):
    from scripts.breadth_collect import Recorder
    allowed={experiment.episode_name(q):q for q in questions}
    class PairedRecorder(Recorder):
        def request(self,group_id,requests,max_tokens=256):
            name=group_id.split("|",1)[0]
            if name not in allowed:raise ValueError("Request is not owned by this worker")
            path=chunk_path(self.root,group_id)
            if not path.exists() and "|round0|" in group_id:
                rows,origin=initial_responses(group_id,requests,allowed[name],protocol,deployment["initial_chunk_sha256"])
                identity={"group_id":group_id,"protocol_sha256":self.protocol_sha256,
                    "request_sha256":hashlib.sha256(json.dumps(requests,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                    "max_tokens":max_tokens}
                write(path,{**identity,"responses":rows,"initial_state_reuse":True,
                    "new_source_calls":0,"completed_utc":experiment.now(),**origin})
            if path.exists():
                cached=read(path)
                if not cached.get("initial_state_reuse"):
                    assert cached.get("hpc3_execution")==execution,"Changed runtime on resume"
            rows=super().request(group_id,requests,max_tokens)
            parallel.publish_immutable(path,chunk_path(ROOT,group_id))
            return rows
    return PairedRecorder(source,root,deadline)


def maybe_score():
    protocol=experiment.validate()
    paths=[ROOT/"episodes/validation"/(experiment.episode_name(q)+".json") for q in protocol["questions"]]
    if not all(p.exists() for p in paths):return False
    with (ROOT/".scoring.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        experiment.evaluate()
    return True


def worker(index,hours):
    if not 0<hours<=47.5:raise ValueError("Invalid time budget")
    protocol,deployment=verify_deployment()
    if not 0<=index<deployment["workers"]:raise ValueError("Invalid worker index")
    job=os.environ.get("SLURM_JOB_ID","")
    if not job.isdigit():raise RuntimeError("Generation requires a Slurm GPU allocation")
    root=ROOT/"workers"/f"worker_{index:02d}"
    with (root/".worker.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert sha(root/"source_protocol.json")==sha(ROOT/"protocol.json")
        state={"worker":index,"job_id":job,"started_utc":experiment.now(),"status":"verifying",
               "protocol_sha256":sha(ROOT/"protocol.json"),"completed_episodes":[],"assessment_included":False}
        def save(**values):
            state.update(values,updated_utc=experiment.now())
            write(root/"state.json",state)
            write(root/"attempts"/(job+".json"),state)
        def interrupted(signum,frame):raise InterruptedError(f"Signal {signum}; cached groups retained")
        signal.signal(signal.SIGTERM,interrupted)
        deadline=time.time()+hours*3600
        save()
        try:
            os.environ["MLX_PTX_CACHE_DIR"]=str((root/"cache/mlx").resolve())
            (root/"cache/mlx").mkdir(parents=True,exist_ok=True)
            manifest=cuda.verify_package()
            report=read(deployment["benchmark_report"])
            fixtures=read("hpc3/fixtures.json")
            identity=cuda.runtime_identity()
            batch=cuda.check_collection_gate(manifest,report,identity,fixtures)
            parallel.check_replay_files(report)
            execution=parallel.execution_record(deployment["benchmark_report"],report)
            save(status="loading_model")
            source=cuda.load_source(batch,execution)
            save(status="checking_replay")
            parallel.replay_worker(source,fixtures,root/"replay"/job)
            from scripts.breadth_collect import run_episode,TimeLimit
            from scripts.local_conversation_pilot import formatter
            personas=[p["persona"] for p in read("data/raw/upstream/data/obj/personas.json")]
            graphs=read(ROOT/"graphs.json")
            questions={q["qid"]:q for q in protocol["questions"]}
            owned=[questions[qid] for qid in deployment["assignments"][index]]
            recorder=make_recorder(source,root,deadline,owned,protocol,deployment,execution)
            fmt=formatter()
            paired=dict(__import__("numpy").load(ROOT/"evaluation_inputs.npz",allow_pickle=False))
            lookup={str(q):i for i,q in enumerate(paired["qids"])}
            try:
                for q in owned:
                    save(status="generating",active_episode=experiment.episode_name(q))
                    episode=run_episode(recorder,q,personas,graphs[q["graph_id"]],fmt,8)
                    experiment.episode_arrays(episode,q,sha(ROOT/"source_protocol.json"),graphs[q["graph_id"]],paired["seen_votes"][lookup[q["qid"]],0])
                    relative=Path("episodes/validation")/(experiment.episode_name(q)+".json")
                    parallel.publish_immutable(root/relative,ROOT/relative)
                    state["completed_episodes"].append(experiment.episode_name(q))
                    save(new_samples_this_attempt=recorder.new_samples)
            except TimeLimit:
                save(status="time_limit",new_samples_this_attempt=recorder.new_samples)
                return 75
            save(status="collection_complete",active_episode=None)
            scored=maybe_score()
            save(status="complete",scoring_complete=scored,completed_utc=experiment.now())
            return 0
        except InterruptedError as exc:
            save(status="interrupted",error=str(exc));return 75
        except Exception as exc:
            save(status="failed",error=f"{type(exc).__name__}: {exc}");raise


def status():
    protocol,deployment=verify_deployment()
    workers=[]
    for i,qids in enumerate(deployment["assignments"]):
        root=ROOT/"workers"/f"worker_{i:02d}"
        workers.append({"worker":i,"assigned":len(qids),
            "state":read(root/"state.json") if (root/"state.json").exists() else None,
            "progress":read(root/"source_progress.json") if (root/"source_progress.json").exists() else None})
    return {"checked_utc":experiment.now(),"workers":workers,
        "completed_episodes":len(list((ROOT/"episodes/validation").glob("*.json"))),
        "planned_episodes":20,"scoring_complete":(ROOT/"results.json").exists(),"assessment_included":False}


def main():
    parser=ArgumentParser()
    parser.add_argument("stage",choices=["prepare","worker","status"])
    parser.add_argument("--worker",type=int)
    parser.add_argument("--hours",type=float,default=47.5)
    args=parser.parse_args()
    if args.stage=="prepare":print(json.dumps(prepare_cluster(),indent=2))
    elif args.stage=="status":print(json.dumps(status(),indent=2))
    else:raise SystemExit(worker(args.worker,args.hours))


if __name__=="__main__":main()
