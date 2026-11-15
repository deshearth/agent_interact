"""Finish the frozen modal experiment, then await human-visible figure inspection."""
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import json_write, digest
from scripts import evaluate_modal_accuracy as evaluation


def run():
    root = evaluation.ROOT.parent
    while not (evaluation.focus.ROOT/'models_frozen.json').exists():
        path = evaluation.focus.ROOT/'progress.json'
        if path.exists():
            progress = evaluation.read(path)
            assert not any(progress.get('completed', {}).values()), progress
        json_write(root/'completion_progress.json', dict(phase='waiting_for_final_refits', updated_utc=evaluation.now()))
        time.sleep(5)
    for phase, command in [('evaluate', ['scripts/evaluate_modal_accuracy.py', 'run']),
        ('audit', ['scripts/audit_modal_accuracy.py']), ('report', ['scripts/report_modal_accuracy.py'])]:
        json_write(root/'completion_progress.json', dict(phase=phase, updated_utc=evaluation.now()))
        subprocess.run([sys.executable, *command], check=True)
    json_write(root/'completion_progress.json', dict(phase='awaiting_visual_verification', updated_utc=evaluation.now(), controller_sha256=digest(__file__)))


if __name__ == '__main__': run()
