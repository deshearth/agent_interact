"""Run or resume the declared breadth study without opening assessment early."""
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
from scripts.breadth_source import write_json
from scripts.question_breadth import ROOT


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--hours',type=float,default=2.)
    args=parser.parse_args()
    if not 0<args.hours<=168:
        raise ValueError('Invocation must have a positive guard of at most seven days')
    deadline=time.time()+3600*args.hours
    source=['.venv-source/bin/python','scripts/breadth_collect.py']
    neural=['.venv/bin/python','scripts/breadth_train.py']
    stages=[
        ('construct',source+['construct','--deadline',str(deadline)]),
        ('screen',source+['screen','--deadline',str(deadline)]),
        ('select',['.venv/bin/python','scripts/question_breadth.py','select']),
        ('train_conversations',source+['collect','--split','train','--deadline',str(deadline)]),
        ('validation_conversations',source+['collect','--split','validation','--deadline',str(deadline)]),
        ('prepare_predictors',neural+['prepare','--device','mps']),
        ('fit_predictors',neural+['fit','--device','mps','--deadline',str(deadline)]),
        ('assessment_conversations',source+['collect','--split','assessment','--deadline',str(deadline)]),
        ('assess',neural+['assess','--device','mps']),
    ]
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'.pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        session={'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
                 'deadline_utc':datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
                 'hours':args.hours,'status':'running'}
        write_json(ROOT/'pipeline_session.json',session)
        for name,command in stages:
            if time.time()>=deadline:
                write_json(ROOT/'pipeline_session.json',{**session,'status':'time_limit','next_stage':name})
                return
            write_json(ROOT/'pipeline_session.json',{**session,'stage':name})
            print(json.dumps({'stage':name,'status':'starting'}),flush=True)
            try:
                result=subprocess.run(command,timeout=max(1.,deadline-time.time()))
            except subprocess.TimeoutExpired:
                write_json(ROOT/'pipeline_session.json',{**session,'status':'time_limit','stage':name,
                           'terminated_at_guard':True,'ended_utc':datetime.now(timezone.utc).isoformat()})
                return
            if result.returncode:
                status='time_limit' if result.returncode==75 else 'failed'
                write_json(ROOT/'pipeline_session.json',{**session,'status':status,'stage':name,'exit_code':result.returncode,
                           'ended_utc':datetime.now(timezone.utc).isoformat()})
                if result.returncode!=75:
                    raise SystemExit(result.returncode)
                return
        write_json(ROOT/'pipeline_session.json',{**session,'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__': main()
