"""Finite controller with a mandatory freeze before outer scoring."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write, environment
from scripts import generalization_distillation as study

def phase(cfg, label):
    root = Path(cfg['output']); pending = [v[0] for v in study.variants(cfg)]; active, done = {}, {}
    while pending or active:
        while pending and len(active) < cfg['training']['max_concurrent_models']:
            name = pending.pop(0); stream = (root/f'{label}_{name}.log').open('a', buffering=1)
            process = subprocess.Popen([sys.executable, 'scripts/generalization_distillation.py', 'worker', '--variant', name, '--phase', label],
                stdout=stream, stderr=subprocess.STDOUT)
            active[name] = process, stream
        for name, (process, stream) in list(active.items()):
            if process.poll() is not None: stream.close(); done[name] = process.returncode; del active[name]
        json_write(root/'progress.json', {'phase': label, 'updated_utc': study.now(), 'pending': pending,
            'active': {name: process.pid for name, (process, stream) in active.items()}, 'completed': done})
        if any(done.values()):
            for process, stream in active.values(): process.wait(); stream.close()
            raise RuntimeError(str(done))
        if active: time.sleep(2)

def main():
    cfg = study.config(); root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    with (root/'controller.lock').open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.prepare(cfg)
        if not (root/'models_frozen.json').exists():
            json_write(root/'environment.json', environment()); phase(cfg, 'tune'); choices = []
            for variant, _, _ in study.variants(cfg):
                records = [study.read(path) for path in (root/'tune'/variant).glob('*.json')]; assert len(records) == 6
                choices.append(min(records, key=lambda r: (r['inner_nll'], r['inner_error'], r['selected_iterations'], r['candidate']['id'])))
            selection = {'protocol_sha256': digest(root/'protocol.json'), 'choices': choices}
            if (root/'selection.json').exists(): assert study.read(root/'selection.json') == selection
            else: json_write(root/'selection.json', selection)
            phase(cfg, 'refit')
            paths = [root/p for p in ('protocol.json', 'prepared.json', 'training_targets.npz', 'selection.json')]
            paths += [p for folder in ('tune', 'refit') for p in (root/folder).rglob('*') if p.is_file()]
            json_write(root/'models_frozen.json', {'frozen_utc': study.now(), 'artifacts': {str(p): digest(p) for p in paths},
                'new_tuning_fits': 36, 'new_refits': 6, 'assessment_loaded': False})
        study.evaluate(cfg); json_write(root/'progress.json', {'phase': 'complete', 'updated_utc': study.now()})

if __name__ == '__main__': main()
