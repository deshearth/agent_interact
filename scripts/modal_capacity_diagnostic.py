"""Training-only memorization and prolonged-fit diagnosis for modal labels."""
from argparse import ArgumentParser
from pathlib import Path
import copy
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write, seed_all
from scripts import modal_question_arguments as paired
from scripts import evaluate_modal_accuracy as evaluation

ROOT = Path('runs/modal_accuracy_investigation_320_v1/capacity_diagnostic_v1')
SETTINGS = dict(width=128, seed=42, learning_rate=.001, weight_decay=0., dropout=0., batch_size=32,
                max_steps=8192, evaluation_every=512, near_perfect_training=.999, promotion_accuracy=.89)


def prepare():
    ROOT.mkdir(exist_ok=True); parent = paired.validate(paired.config()); data = paired.Data('inner')
    seen = set(); selected = []
    candidates = np.random.default_rng(92742).permutation(parent['inner']['fit'])
    for qid in candidates:
        e = int(np.flatnonzero(data.common['episode_qids'] == qid)[0]); key = data.common['J'][e].tobytes()
        if key not in seen: selected.append(str(qid)); seen.add(key)
    assert len(selected) == 8
    protocol = dict(settings=SETTINGS, subset8=selected, inner=parent['inner'], fold=parent['fold'],
        code_sha256=digest(__file__), parent_models_sha256=digest(Path(paired.config()['output'])/'models_frozen.json'),
        policy='Four training-only diagnostics:8 or256 fitting questions crossed with matched full-vector choice/arguments. '
            'Width128, no dropout/weight decay/output penalty, one fixed learning rate. Same seed. '
            'The8-question runs use existing inner256 preprocessing, test memorization only, and do not assess any holdout. '
            'The256 runs record internal64 accuracy but never score outer80. Stop at8192 or two consecutive near-perfect training evaluations. '
            'Promote a256 candidate only if internal mode accuracy reaches89%; otherwise this is a diagnostic, not a selected final model.',
        assessment_loaded=False, source_calls=0, outer_validation_scored=False)
    if (ROOT/'protocol.json').exists(): assert paired.read(ROOT/'protocol.json') == protocol
    else: json_write(ROOT/'protocol.json', protocol)
    return protocol


def worker(kind, count):
    p = paired.read(ROOT/'protocol.json'); assert p['code_sha256'] == digest(__file__)
    folder = ROOT/f'{kind}_{count}'; folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists(): return
    torch.set_num_threads(2); seed_all(42); cfg = copy.deepcopy(paired.config()); cfg['training']['dropout'] = 0.
    name = 'paired_full_'+kind; cfg['models'][name]['width'] = SETTINGS['width']
    model = paired.core.make_model(cfg, name); data = paired.Data('inner')
    fit = data.ids(p['subset8'] if count == 8 else p['inner']['fit']); val = data.ids(p['inner']['validation'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['learning_rate'], weight_decay=0.)
    rng = np.random.default_rng(42); history = []; best = None; perfect = 0; started = time.monotonic()
    for step in range(SETTINGS['max_steps']+1):
        if step:
            model.train(); optimizer.zero_grad(set_to_none=True)
            batch = data.batch(rng.choice(fit, size=min(len(fit), SETTINGS['batch_size']), replace=False), 'mode')
            z = paired.core.score_model(model, batch); loss = F.binary_cross_entropy_with_logits(z, batch['target'])
            assert torch.isfinite(loss); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        if step % SETTINGS['evaluation_every'] == 0:
            lp = evaluation.predict(model, data, fit)
            train, rows = evaluation.metrics(lp, data.source[fit], data.qids[fit])
            validation = None
            if count == 256:
                lp = evaluation.predict(model, data, val)
                validation, rows = evaluation.metrics(lp, data.source[val], data.qids[val])
                key = -validation['accuracy'], validation['modal_nll']
                if best is None or key < best:
                    best = key; torch.save(dict(model=model.state_dict(), step=step), folder/'best_internal.pt')
            history.append(dict(step=step, train=train, validation=validation, seconds=time.monotonic()-started))
            json_write(folder/'history.json', history)
            print(dict(kind=kind, fitting_questions=count, **history[-1]), flush=True)
            perfect = perfect+1 if train['accuracy'] >= SETTINGS['near_perfect_training'] else 0
            if perfect == 2: break
    torch.save(dict(model=model.state_dict(), step=step), folder/'terminal.pt')
    selected = min([h for h in history if h['validation'] is not None], key=lambda h: (-h['validation']['accuracy'], h['validation']['modal_nll'])) if count == 256 else None
    json_write(folder/'complete.json', dict(kind=kind, fitting_questions=count, parameters=sum(v.numel() for v in model.parameters()),
        terminal=history[-1], selected=selected, seconds=time.monotonic()-started, promotion_gate_passed=count == 256 and selected['validation']['accuracy'] >= SETTINGS['promotion_accuracy'],
        artifacts={f.name: digest(f) for f in folder.iterdir() if f.is_file()}, completed_utc=paired.now(), outer_validation_scored=False))


def run():
    prepare(); active = []
    for kind in ('choice', 'arguments'):
        for count in (8, 256):
            stream = (ROOT/f'{kind}_{count}.log').open('a', buffering=1)
            active.append((subprocess.Popen([sys.executable, __file__, 'worker', '--kind', kind, '--count', str(count)], stdout=stream, stderr=subprocess.STDOUT), stream))
    while any(p.poll() is None for p, s in active):
        if any(p.poll() not in (None, 0) for p, s in active):
            for p, s in active:
                if p.poll() is None: p.terminate(); p.wait()
            raise RuntimeError('Capacity diagnostic failed.')
        time.sleep(3)
    assert all(p.returncode == 0 for p, s in active)
    for p, s in active: s.close()
    results = [paired.read(p) for p in sorted(ROOT.glob('*/complete.json'))]; assert len(results) == 4
    json_write(ROOT/'results.json', dict(results=results, promotion_gate_passed=any(r['promotion_gate_passed'] for r in results),
        completed_utc=paired.now(), outer_validation_scored=False, assessment_loaded=False))


if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['run', 'worker']); parser.add_argument('--kind'); parser.add_argument('--count', type=int); args = parser.parse_args()
    if args.action == 'run': run()
    else: worker(args.kind, args.count)
