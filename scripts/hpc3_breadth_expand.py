"""Resumable free-GPU screening/collection workers for the 320/80 corpus."""
from argparse import ArgumentParser
from collections import Counter
import fcntl
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.hpc3_prepare import read,sha,write
from scripts import breadth_expand as study
from scripts import hpc3_breadth as cuda
from scripts import hpc3_parallel_collect as parallel

ROOT=study.ROOT


def assignments(values,workers,phase):
    output=[[] for _ in range(workers)]
    assert len({q["qid"] for q in values})==len(values)
    if phase=="screen":
        for i,q in enumerate(values):output[i%workers].append(q)
    elif phase=="collect":
        assert Counter(q["split"] for q in values)=={"train":240,"validation":60}
        for split in ["train","validation"]:
            for i,q in enumerate(sorted((q for q in values if q["split"]==split),key=lambda q:q["qid"])):
                output[i%workers].append(q)
    else:raise ValueError("Unknown phase")
    return output


def prepare_cluster():
    protocol=study.verify()
    report_path=protocol["config"]["collection"]["benchmark_report"]
    manifest=cuda.verify_package();report=read(report_path);fixtures=read("hpc3/fixtures.json")
    cuda.check_collection_gate(manifest,report,report["runtime"],fixtures)
    parallel.check_replay_files(report)
    workers=protocol["config"]["collection"]["workers"]
    shards=assignments(read(ROOT/"candidate_bank.json"),workers,"screen")
    record={"protocol_sha256":sha(ROOT/"protocol.json"),"workers":workers,
        "benchmark_report":report_path,"benchmark_report_sha256":sha(report_path),
        "screening_assignments":[[q["qid"] for q in shard] for shard in shards],
        "retained_episodes_sha256":sha(ROOT/"retained_episodes.json"),"assessment_included":False}
    if (ROOT/"deployment.json").exists():
        old=read(ROOT/"deployment.json")
        assert all(old[k]==v for k,v in record.items())
        verify_deployment();return old
    for phase in ["screen","collect"]:
        for i in range(workers):
            root=ROOT/"workers"/f"{phase}_{i:02d}"
            parallel.publish_immutable(ROOT/"source_protocol.json",root/"source_protocol.json")
    for item in read(ROOT/"retained_episodes.json"):
        path=Path(item["source_path"])
        assert sha(path)==item["sha256"]
        parallel.publish_immutable(path,ROOT/"episodes"/item["split"]/path.name)
    write(ROOT/"deployment.json",{**record,"prepared_utc":study.now()})
    verify_deployment()
    return read(ROOT/"deployment.json")


def verify_deployment():
    protocol=study.verify();deployment=read(ROOT/"deployment.json")
    assert deployment["protocol_sha256"]==sha(ROOT/"protocol.json")
    assert deployment["benchmark_report_sha256"]==sha(deployment["benchmark_report"])
    assert deployment["retained_episodes_sha256"]==sha(ROOT/"retained_episodes.json")
    assert not deployment["assessment_included"]
    for item in read(ROOT/"retained_episodes.json"):
        assert sha(item["source_path"])==item["sha256"]
        assert sha(ROOT/"episodes"/item["split"]/Path(item["source_path"]).name)==item["sha256"]
    return protocol,deployment


def make_recorder(source,root,deadline,owned,phase,execution):
    from scripts.breadth_collect import Recorder
    allowed={q["qid"] for q in owned}
    class ExpansionRecorder(Recorder):
        def request(self,group_id,requests,max_tokens=256):
            if not requests or any(r["metadata"].get("qid") not in allowed for r in requests):
                raise ValueError("Request belongs to another worker or a reserved question")
            if phase=="screen" and not group_id.startswith(("construct|","screen|")):
                raise ValueError("Interaction entered a screening worker")
            if group_id.startswith("construct|"):
                if phase!="screen":raise ValueError("Unexpected construction during collection")
                # The original helper numbers groups from zero on each call.
                # A qid-scoped cache name avoids collisions between workers and batches;
                # request/sample IDs and therefore sampling RNG are unchanged.
                group_id="expand_construct|"+requests[0]["metadata"]["qid"]
            if phase=="collect" and any(r["metadata"].get("phase")=="screen" for r in requests):
                raise ValueError("Screening entered a collection worker")
            path=parallel.chunk_path(self.root,group_id)
            if path.exists():assert read(path).get("hpc3_execution")==execution,"Changed execution on resume"
            rows=super().request(group_id,requests,max_tokens)
            parallel.publish_immutable(path,parallel.chunk_path(ROOT,group_id))
            return rows
    return ExpansionRecorder(source,root,deadline)


def worker(phase,index,hours):
    if phase not in ["screen","collect"] or not 0<hours<=47.5:raise ValueError("Invalid phase/budget")
    protocol,deployment=verify_deployment()
    if not 0<=index<deployment["workers"]:raise ValueError("Invalid worker")
    if phase=="screen":
        bank=read(ROOT/"candidate_bank.json")
        owned=assignments(bank,deployment["workers"],phase)[index]
        assert [q["qid"] for q in owned]==deployment["screening_assignments"][index]
    else:
        selection=study.verify_selection()
        owned=assignments(selection["new_questions"],deployment["workers"],phase)[index]
        assert not {q["qid"] for q in owned}&set(protocol["reserved_assessment_qids"])
    job=os.environ.get("SLURM_JOB_ID","")
    if not job.isdigit():raise RuntimeError("Use a Slurm GPU allocation")
    root=ROOT/"workers"/f"{phase}_{index:02d}"
    with (root/".worker.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert sha(root/"source_protocol.json")==sha(ROOT/"protocol.json")
        state={"phase":phase,"worker":index,"job_id":job,"started_utc":study.now(),"assigned":len(owned),
            "protocol_sha256":sha(ROOT/"protocol.json"),"completed_questions":[],"status":"verifying","assessment_included":False}
        def save(**values):
            state.update(values,updated_utc=study.now());write(root/"state.json",state);write(root/"attempts"/(job+".json"),state)
        def interrupted(signum,frame):raise InterruptedError(f"Signal {signum}; saved request groups are resumable")
        signal.signal(signal.SIGTERM,interrupted)
        deadline=time.time()+hours*3600;save()
        try:
            os.environ["MLX_PTX_CACHE_DIR"]=str((root/"cache/mlx").resolve());(root/"cache/mlx").mkdir(parents=True,exist_ok=True)
            manifest=cuda.verify_package();report=read(deployment["benchmark_report"]);fixtures=read("hpc3/fixtures.json")
            identity=cuda.runtime_identity();batch=cuda.check_collection_gate(manifest,report,identity,fixtures)
            parallel.check_replay_files(report);execution=parallel.execution_record(deployment["benchmark_report"],report)
            save(status="loading_model");source=cuda.load_source(batch,execution)
            save(status="checking_replay");parallel.replay_worker(source,fixtures,root/"replay"/job)
            from scripts.breadth_collect import construct,screen,run_episode,TimeLimit
            from scripts.local_conversation_pilot import formatter
            personas=[p["persona"] for p in read("data/raw/upstream/data/obj/personas.json")]
            graphs={g["id"]:g["J"] for g in read("data/raw/upstream/data/J_matrices.json")["graphs"]}
            recorder=make_recorder(source,root,deadline,owned,phase,execution);fmt=formatter()
            try:
                if phase=="screen":
                    for begin in range(0,len(owned),4):
                        part=owned[begin:begin+4];save(status="constructing",active_qid=part[0]["qid"])
                        construct(recorder,part)
                        for q in part:
                            relative=Path("constructed")/(q["qid"]+".json")
                            parallel.publish_immutable(root/relative,ROOT/relative)
                        for q in part:
                            save(status="screening",active_qid=q["qid"])
                            screen(recorder,[q],personas,fmt)
                            relative=Path("screen")/(q["qid"]+".json")
                            parallel.publish_immutable(root/relative,ROOT/relative)
                            state["completed_questions"].append(q["qid"]);save(new_samples_this_attempt=recorder.new_samples)
                else:
                    for q in owned:
                        save(status="generating",active_qid=q["qid"],active_split=q["split"])
                        run_episode(recorder,q,personas,graphs[q["graph_id"]],fmt,8)
                        relative=Path("episodes")/q["split"]/(parallel.episode_name(q)+".json")
                        minimal={k:q[k] for k in ["qid","graph_id","split"]}
                        old_cfg={"source":{"n_agents":32,"update_votes_per_agent":5,"update_rounds":8}}
                        parallel.validate_episode(root/relative,minimal,sha(ROOT/"protocol.json"),old_cfg)
                        parallel.publish_immutable(root/relative,ROOT/relative)
                        state["completed_questions"].append(q["qid"]);save(new_samples_this_attempt=recorder.new_samples)
            except TimeLimit:
                save(status="time_limit",new_samples_this_attempt=recorder.new_samples);return 75
            save(status="complete",active_qid=None,completed_utc=study.now());return 0
        except InterruptedError as exc:
            save(status="interrupted",error=str(exc));return 75
        except Exception as exc:
            save(status="failed",error=f"{type(exc).__name__}: {exc}");raise


def status():
    protocol,deployment=verify_deployment()
    states=[]
    for phase in ["screen","collect"]:
        for worker in range(deployment["workers"]):
            root=ROOT/"workers"/f"{phase}_{worker:02d}"
            states.append({"phase":phase,"worker":worker,
                "state":read(root/"state.json") if (root/"state.json").exists() else None,
                "progress":read(root/"source_progress.json") if (root/"source_progress.json").exists() else None})
    return {"checked_utc":study.now(),"planned_candidates":1200,
        "constructed":len(list((ROOT/"constructed").glob("*.json"))),"screened":len(list((ROOT/"screen").glob("*.json"))),
        "selection_ready":(ROOT/"selection_ready.json").exists(),
        "episodes":{split:len(list((ROOT/"episodes"/split).glob("*.json"))) for split in ["train","validation","assessment"]},
        "target":{"train":320,"validation":80},"collection_complete":(ROOT/"collection_complete.json").exists(),"workers":states}


if __name__=="__main__":
    import json
    parser=ArgumentParser();parser.add_argument("stage",choices=["prepare","worker","status"])
    parser.add_argument("--phase",choices=["screen","collect"]);parser.add_argument("--worker",type=int);parser.add_argument("--hours",type=float,default=47.5)
    args=parser.parse_args()
    if args.stage=="prepare":print(json.dumps(prepare_cluster(),indent=2))
    elif args.stage=="status":print(json.dumps(status(),indent=2))
    else:raise SystemExit(worker(args.phase,args.worker,args.hours))
