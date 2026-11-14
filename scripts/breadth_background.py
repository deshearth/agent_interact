"""Continue the user-authorized breadth experiment in a detached local process.

The current bounded collection session finishes first. Later bounded sessions
resume automatically until the comparison completes or a stage fails. This
controller never changes scientific settings or opens assessment early.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.breadth_source import sha, write_json
from scripts.question_breadth import ROOT

REPOSITORY=Path(__file__).resolve().parents[1]
LOG=REPOSITORY/'reports/question_breadth_background.log'


def now(): return datetime.now(timezone.utc).isoformat()


def pipeline_busy():
    with (ROOT/'.pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock,fcntl.LOCK_UN)
        return False


def completed_and_verified():
    path=ROOT/'comparison_complete.json'
    if not path.exists(): return False
    result=json.loads(path.read_text())
    assert result['status']=='complete'
    assert result['models_frozen_sha256']==sha(ROOT/'models_frozen.json')
    for p,digest in {**result['assessment_sources'],**result['artifacts']}.items():
        assert sha(p)==digest,p
    frozen=json.loads((ROOT/'models_frozen.json').read_text())
    for p,digest in frozen['artifacts'].items(): assert sha(p)==digest,p
    return True


def supervise():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'.background.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'pid':os.getpid(),'started_utc':now(),'status':'running',
               'authorization':'User requested run to completion in the background',
               'automatic_resume_sessions':0,'controller_sha256':sha(__file__)}
        keep_awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])

        def record(**updates):
            state.update(updates)
            write_json(ROOT/'background_session.json',{**state,'updated_utc':now()})

        try:
            while True:
                if completed_and_verified():
                    record(status='complete',completed_utc=now())
                    print('Verified completed comparison: reports/question_breadth.md',flush=True)
                    return
                if pipeline_busy():
                    record(stage='waiting_for_current_session',active_pipeline=json.loads((ROOT/'pipeline_session.json').read_text()))
                    time.sleep(10)
                    continue
                previous_path=ROOT/'pipeline_session.json'
                if previous_path.exists():
                    previous=json.loads(previous_path.read_text())
                    archive=ROOT/'background_sessions'/f"session_{previous['started_utc'].replace(':','-')}.json"
                    write_json(archive,previous)
                    if previous['status']=='failed':
                        raise RuntimeError(f"Pipeline failed at {previous.get('stage')}; inspect the source log before resuming")
                count=state['automatic_resume_sessions']+1
                # The per-invocation seven-day process guard remains; a bounded
                # return is resumed, so the authorized study has no two-hour cap.
                command=[sys.executable,'scripts/breadth_run.py','--hours','168']
                worker=subprocess.Popen(command,cwd=REPOSITORY)
                record(stage='pipeline',worker_pid=worker.pid,automatic_resume_sessions=count)
                print(json.dumps({'started_utc':now(),'automatic_resume_session':count,'worker_pid':worker.pid}),flush=True)
                while worker.poll() is None:
                    if previous_path.exists():
                        record(active_pipeline=json.loads(previous_path.read_text()))
                    time.sleep(10)
                if worker.returncode:
                    raise RuntimeError(f'Pipeline returned exit status {worker.returncode}')
                # A short pause prevents a tight retry loop around a process
                # handoff; scientific stages enforce their own resume checks.
                time.sleep(1)
        except BaseException as exc:
            record(status='failed',ended_utc=now(),error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            keep_awake.terminate()
            keep_awake.wait(timeout=10)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args()
    os.chdir(REPOSITORY)
    if args.detach:
        LOG.parent.mkdir(parents=True,exist_ok=True)
        with LOG.open('a') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                cwd=REPOSITORY,start_new_session=True,close_fds=True)
        print(json.dumps({'background_pid':child.pid,'log':str(LOG),'state':str((ROOT/'background_session.json').resolve())}))
        return
    supervise()


if __name__=='__main__': main()
