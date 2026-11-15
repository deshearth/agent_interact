"""Finite two-worker driver; all choices freeze before outer validation scoring."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write, environment
from scripts import generalization_state_models as study

def phase(cfg, label):
    root = Path(cfg['output']); pending = list(cfg['variants']); active = {}; completed = {}
    while pending or active:
        while pending and len(active) < cfg['training']['max_concurrent_models']:
            name = pending.pop(0); stream = (root/f'{label}_{name}.log').open('a', buffering=1)
            process = subprocess.Popen([sys.executable, 'scripts/generalization_state_models.py', 'worker',
                                        '--variant', name, '--phase', label], stdout=stream, stderr=subprocess.STDOUT)
            active[name] = process, stream
        for name, (process, stream) in list(active.items()):
            code = process.poll()
            if code is not None:
                stream.close(); completed[name] = code; del active[name]
        json_write(root/'progress.json', {'updated_utc': study.now(), 'phase': label, 'pending': pending,
                   'active': {name: p.pid for name, (p, f) in active.items()}, 'completed': completed})
        if any(completed.values()):
            for process, stream in active.values(): process.wait(); stream.close()
            raise RuntimeError(f'{label} failed: {completed}')
        if active: time.sleep(2)

def main():
    cfg = study.config(); root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    with (root/'controller.lock').open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.prepare(cfg)
        if not (root/'models_frozen.json').exists():
            json_write(root/'environment.json', environment())
            phase(cfg, 'tune'); choices = []
            for name in cfg['variants']:
                candidates = [study.read(p) for p in (root/'tune'/name).glob('*.json')]
                assert len(candidates) == 6
                choices.append(min(candidates, key=lambda r: (r['inner_nll'], r['inner_error'], r['selected_iterations'], r['candidate']['id'])))
            selection = {'protocol_sha256': digest(root/'protocol.json'), 'choices': choices}
            if (root/'selection.json').exists(): assert study.read(root/'selection.json') == selection
            else: json_write(root/'selection.json', selection)
            phase(cfg, 'refit')
            files = [root/'protocol.json', root/'prepared.json', root/'selection.json']
            files += [p for folder in ('tune', 'refit') for p in (root/folder).rglob('*') if p.is_file()]
            json_write(root/'models_frozen.json', {'frozen_utc': study.now(), 'artifacts': {str(p): digest(p) for p in files},
                       'new_tuning_fits': 36, 'new_refits': 6, 'assessment_loaded': False})
        study.evaluate(cfg)
        json_write(root/'progress.json', {'phase': 'complete', 'updated_utc': study.now()})
        print('State-model cycle complete.', flush=True)

if __name__ == '__main__': main()
