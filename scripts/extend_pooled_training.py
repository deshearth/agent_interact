"""A fixed doubled-duration check, specified after inner cap diagnostics only."""
import copy
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from scripts import transformer_generalization as base
from scripts import pooled_state_comparison as study


def run():
    cfg = config(study.CONFIG); parent = Path(cfg['output']); p = study.validate(cfg)
    assert (parent/'all_fits_frozen.json').exists()
    assert not (parent/'evaluation_complete.json').exists(), 'Specify this check before outer scoring'
    root = parent/'doubled_duration'; root.mkdir(exist_ok=True)
    original = [(d, r) for d, r in base.completed_runs(parent, 'refit') if r['model'] == 'pooled_small']
    assert len(original) == 12
    selected = json.loads((parent/'selection.json').read_text())
    choices = [{**r, 'refit_steps': 2*r['refit_steps']} for r in selected['choices'] if r['model'] == 'pooled_small']
    cfg = {**cfg, 'output': str(root)}
    specification = {
        'implementation_sha256': digest(__file__), 'parent_protocol_sha256': digest(parent/'protocol.json'),
        'parent_all_fits_frozen_sha256': digest(parent/'all_fits_frozen.json'),
        'trigger': 'Eight of twelve inner selected checkpoints for the 809-parameter model reached the 3072-update cap. No outer assessment of the new models was examined.',
        'intervention': 'Same inner-selected rate/penalty, initialization seed, dropout, normalization and minibatch schedule. Refit from scratch for exactly twice each selected duration. No new hyperparameter or checkpoint selection. Report every result even if worse.',
        'candidates': p['candidates'], 'folds': p['folds'], 'choices': choices,
        'original_models': {str(d/'manifest.json'): digest(d/'manifest.json') for d, _ in original},
        'outer_outcomes_scored': False, 'planned_fits': 12,
    }
    if (root/'protocol.json').exists():
        old = json.loads((root/'protocol.json').read_text())
        assert all(old[k] == v for k, v in specification.items())
    else:
        json_write(root/'protocol.json', {**specification, 'created_utc': base.now()})
        json_write(root/'selection.json', {'choices': choices, 'parent_selection_sha256': digest(parent/'selection.json'),
                                         'duration_multiplier': 2, 'frozen_utc': base.now()})
        for f in p['folds']:
            if f['phase'] != 'refit': continue
            dest = root/'folds'/f['id']; dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent/'folds'/f['id']/'manifest.json', dest/'manifest.json')
    common = dict(np.load(parent/'development.npz', allow_pickle=False))
    manifests = {}; prefixes = []; real_save = base.save_checkpoint; deadline = time.time()+30*60
    with patch.object(base, 'make_model', study.make_model):
        for choice in choices:
            fold = next(f for f in p['folds'] if f['id'] == choice['fold'])
            data = study.PooledData(common, dict(np.load(parent/'folds'/fold['id']/'features.npz', allow_pickle=False)), 'cpu')
            norms = json.loads((parent/'folds'/fold['id']/'normalization.json').read_text())
            local = {**cfg, '_normalization': {k: norms[k] for k in ['mean', 'scale']}}
            for seed in cfg['training']['refit_seeds']:
                prior = next(r for _, r in original if r['outer'] == fold['outer'] and r['seed'] == seed)
                item = {**choice, 'phase': 'refit', 'seed': seed}
                midpoint = []

                def traced_save(path, payload):
                    if Path(path).name == 'resume.pt' and payload['state']['step'] == prior['selected_step']:
                        # Avoid constructing a fresh randomly initialized model here: it
                        # would consume RNG state and change subsequent training dropout.
                        import hashlib
                        h = hashlib.sha256()
                        for name, value in sorted(payload['model'].items()):
                            h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
                        actual = h.hexdigest()
                        assert actual == prior['selected_weights_sha256']
                        midpoint.append(actual)
                    real_save(path, payload)

                with patch.object(base, 'save_checkpoint', traced_save):
                    out, record = base.train_one(local, specification, data, fold, item, 'cpu', deadline, item['refit_steps'])
                assert record['status'] == 'complete'
                if not midpoint:
                    saved = json.loads((root/'prefix_checks.json').read_text()) if (root/'prefix_checks.json').exists() else []
                    midpoint = [r['weights_sha256'] for r in saved if r['outer'] == fold['outer'] and r['seed'] == seed]
                assert midpoint
                prefixes.append({'outer': fold['outer'], 'seed': seed, 'step': prior['selected_step'],
                                 'weights_sha256': midpoint[0], 'exact_original_prefix': True})
                json_write(root/'prefix_checks.json', prefixes)
                manifests[str(out.relative_to(root))] = digest(out/'manifest.json')
    json_write(root/'refit_complete.json', {'completed_utc': base.now(), 'fits': 12,
               'protocol_sha256': digest(root/'protocol.json'), 'manifests': manifests})
    json_write(root/'frozen.json', {'frozen_utc': base.now(), 'outer_outcomes_scored': False,
        'artifacts': {n: digest(root/n) for n in ['protocol.json', 'selection.json', 'refit_complete.json', 'prefix_checks.json']}})


if __name__ == '__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): run()
