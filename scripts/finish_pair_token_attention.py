"""Independent model sizes run concurrently; scientific protocol is unchanged."""
from argparse import ArgumentParser
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from mac_agents.common import digest, json_write, environment
from scripts import pair_token_attention_study as study

def worker(cfg, width, phase):
    run = Path(cfg['output']); protocol = study.validate(cfg)
    torch.set_num_threads(cfg['training']['torch_threads'])
    data = study.Data(cfg, 'cpu', width)
    original_write = study.shared.json_write
    def separate_progress(path, value):
        if Path(path) == run/'progress.json': path = run/'worker_progress'/f'{width}.json'
        original_write(path, value)
    study.shared.json_write = separate_progress
    if phase == 'tune':
        tasks = [(dict(phase=phase, width=width, condition=condition, candidate=c['id'], seed=cfg['training']['tuning_seed']), None)
                 for condition in cfg['conditions'] for c in study.base.candidates(cfg)]
    else:
        tasks = []
        for choice in study.read(run/'selection.json')['choices']:
            if choice['width'] != width: continue
            for seed in cfg['training']['refit_seeds']:
                item = {k: choice[k] for k in ('width', 'condition', 'candidate')}; item.update(phase=phase, seed=seed)
                tasks.append((item, choice['refit_steps']))
    deadline = time.time()+cfg['training']['aggregate_minutes']*60
    with study.adapted_trainer():
        for item, duration in tasks:
            while True:
                path, record = study.shared.train_one(cfg, protocol, data, item, deadline, duration)
                if record['status'] == 'complete': break
                assert record['status'] == 'bounded'
                deadline = time.time()+cfg['training']['aggregate_minutes']*60
    return 0

def run_phase(cfg, phase):
    run = Path(cfg['output']); processes, streams = {}, {}
    for width in cfg['models']:
        path = run/f'worker_{width}_{phase}.log'; streams[width] = path.open('a', buffering=1)
        processes[width] = subprocess.Popen([sys.executable, __file__, '--worker', width, '--phase', phase],
            cwd=root, stdout=streams[width], stderr=subprocess.STDOUT)
    json_write(run/'controller_state.json', {'phase': phase, 'started_utc': study.shared.now(),
        'workers': {w: {'pid': p.pid, 'log': str(run/f'worker_{w}_{phase}.log')} for w, p in processes.items()}})
    status = {width: process.wait() for width, process in processes.items()}
    for stream in streams.values(): stream.close()
    json_write(run/f'{phase}_worker_completion.json', {'completed_utc': study.shared.now(), 'return_codes': status})
    if any(status.values()): raise RuntimeError(f'{phase} workers did not all complete: {status}')

def main():
    parser = ArgumentParser(); parser.add_argument('--worker'); parser.add_argument('--phase', choices=['tune', 'refit'])
    args = parser.parse_args(); cfg = study.config(); run = Path(cfg['output']); run.mkdir(parents=True, exist_ok=True)
    if args.worker:
        assert args.worker in cfg['models'] and args.phase
        return worker(cfg, args.worker, args.phase)
    with (run/'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.validate(cfg)
        if not (run/'models_frozen.json').exists():
            json_write(run/'parallel_execution.json', {'started_utc': study.shared.now(), 'controller_sha256': digest(__file__),
                'workers': 4, 'partition': 'one model width/text combination per process',
                'device': 'cpu', 'threads_per_worker': cfg['training']['torch_threads'],
                'scientific_protocol_sha256': digest(run/'protocol.json'),
                'resumed_serial_checkpoint': True,
                'policy': 'Same frozen trainer, settings, seeds, candidate order within model and model identities. '
                  'Only progress JSON is redirected to prevent concurrent write conflicts. Freeze all tuning choices before any refit.'})
            if not (run/'training_environment.json').exists(): json_write(run/'training_environment.json', environment())
            run_phase(cfg, 'tune')
            records = [(p.parent, study.read(p)) for p in sorted((run/'tune').glob('*/manifest.json'))]
            assert len(records) == 72 and all(r['status'] == 'complete' for p, r in records)
            selection = {'protocol_sha256': digest(run/'protocol.json'), 'choices': study.shared.choose(cfg, records),
                         'tuning_manifests': {str(p/'manifest.json'): digest(p/'manifest.json') for p, r in records}}
            if (run/'selection.json').exists(): assert study.read(run/'selection.json') == selection
            else: json_write(run/'selection.json', selection)
            run_phase(cfg, 'refit')
            assert len(list((run/'refit').glob('*/manifest.json'))) == 24
            # The original driver checks completed fits and writes its exact freeze manifest.
            study.fit(cfg, 'cpu')
        log = root/'reports/pair_token_attention_320_training.log'
        with log.open('a', buffering=1) as stream:
            for script in ('report_pair_token_attention.py', 'audit_pair_token_attention.py', 'plot_pair_token_attention.py'):
                result = subprocess.run([sys.executable, 'scripts/'+script], cwd=root, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode: raise SystemExit(result.returncode)
        json_write(run/'controller_state.json', {'phase': 'complete', 'completed_utc': study.shared.now()})
        print('Pair-token attention study, evaluation, independent audit and report complete.', flush=True)

if __name__ == '__main__': main()
