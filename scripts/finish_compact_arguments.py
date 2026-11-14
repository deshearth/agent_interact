"""One-shot local postprocessing after both independently owned studies freeze."""
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'runs/compact_arguments_320_v1/postprocess_status.json'

def status(stage,**extra):
    record={'stage':stage,'updated_utc':datetime.now(timezone.utc).isoformat(),**extra}
    temporary=STATUS.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(STATUS)
    print(json.dumps(record),flush=True)

def main():
    required=[ROOT/'runs/compact_arguments_320_v1/models_frozen.json',ROOT/'runs/breadth_arguments_320_v1/models_frozen.json']
    status('waiting_for_frozen_models',required=[str(p.relative_to(ROOT)) for p in required])
    while not all(p.exists() for p in required):time.sleep(20)
    for script in ['report_compact_arguments.py','audit_compact_arguments.py','plot_compact_arguments.py']:
        status('running',script=script)
        result=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT)
        if result.returncode:
            status('failed',script=script,exit_code=result.returncode);raise SystemExit(result.returncode)
    status('complete',results='runs/compact_arguments_320_v1/results/summary.json',
           audit='runs/compact_arguments_320_v1/completion_audit.json',report='reports/compact_arguments_320.md')

if __name__=='__main__':main()
