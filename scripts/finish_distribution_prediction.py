"""Bounded four-worker distribution comparison with a common scoring barrier."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write, environment
from scripts import distribution_prediction_study as study

def phase(cfg, label):
    root = Path(cfg['output']); pending = list(cfg['models']); active, done = {}, {}
    while pending or active:
        while pending and len(active) < cfg['training']['max_concurrent_models']:
            name = pending.pop(0); stream = (root/f'{label}_{name}.log').open('a', buffering=1)
            process = subprocess.Popen([sys.executable, 'scripts/distribution_prediction_study.py', 'worker', '--model', name, '--phase', label], stdout=stream, stderr=subprocess.STDOUT)
            active[name] = process, stream
        for name, (process, stream) in list(active.items()):
            if process.poll() is not None: stream.close(); done[name] = process.returncode; del active[name]
        json_write(root/'controller_progress.json', {'phase': label, 'updated_utc': study.now(), 'pending': pending,
            'active': {name: process.pid for name, (process, stream) in active.items()}, 'completed': done})
        if any(done.values()):
            for process, stream in active.values(): process.wait(); stream.close()
            raise RuntimeError(str(done))
        if active: time.sleep(2)

def main():
    cfg = study.config(); root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    with (root/'controller.lock').open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); study.prepare(cfg)
        if not (root/'models_frozen.json').exists():
            json_write(root/'environment.json', environment()); study.fit_baselines(cfg); phase(cfg, 'tune')
            records = [(p.parent, study.read(p)) for p in (root/'tune').glob('*/manifest.json')]
            assert len(records) == 54 and all(r['status'] == 'complete' for _, r in records)
            selection = {'protocol_sha256': digest(root/'protocol.json'), 'choices': study.trainer.choose(cfg, records),
                'tuning_manifests': {str(p/'manifest.json'): digest(p/'manifest.json') for p, r in records}}
            if (root/'selection.json').exists(): assert study.read(root/'selection.json') == selection
            else: json_write(root/'selection.json', selection)
            phase(cfg, 'refit'); refits = [(p.parent, study.read(p)) for p in (root/'refit').glob('*/manifest.json')]
            assert len(refits) == 18 and all(r['status'] == 'complete' for _, r in refits)
            artifacts = {str(root/n): digest(root/n) for n in ('protocol.json', 'prepared.json', 'encoding.json', 'selection.json')}
            for path, record in records+refits:
                artifacts[str(path/'manifest.json')] = digest(path/'manifest.json')
                artifacts.update({str(path/name): sha for name, sha in record['artifacts'].items()})
            artifacts.update({str(p): digest(p) for p in (root/'baselines').rglob('*') if p.is_file()})
            json_write(root/'models_frozen.json', {'frozen_utc': study.now(), 'artifacts': artifacts, 'refits': [str(p) for p, r in refits],
                'new_tuning_fits': 54, 'new_neural_refits': 18, 'new_ising_fits': 2, 'assessment_loaded': False})
        study.evaluate(cfg); json_write(root/'controller_progress.json', {'phase': 'complete', 'updated_utc': study.now()})

if __name__ == '__main__': main()
