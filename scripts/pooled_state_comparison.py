"""Test nonlinear aggregate state against frozen individual-neighbor attention.

Only the old twenty GPT development questions are used. Source generation and
the new Llama assessment cohort are never opened. The existing training loop is
reused with a process-local model factory override, without editing that loop.
"""
from argparse import ArgumentParser
import copy
import json
import math
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
from torch import nn
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import code_digest, config, digest, environment, json_write
from scripts import transformer_generalization as base
from scripts import transformer_micro_sweep as reference_driver
from scripts import distill_transformer_rule as rules

CONFIG = Path('configs/pooled_state_comparison.yaml')
now = base.now


def pooled_inputs(s, context, graph):
    positive = (graph > 0).to(context.dtype)
    negative = (graph < 0).to(context.dtype)
    friendly = (positive @ s[..., None]).squeeze(-1)
    unfriendly = (negative @ s[..., None]).squeeze(-1)
    return torch.cat([context[..., 1:], s[..., None], friendly[..., None],
                      unfriendly[..., None], positive.sum(-1)[..., None],
                      negative.sum(-1)[..., None]], -1)


class PooledUpdate(nn.Module):
    def __init__(self, hidden, dropout, mean=None, scale=None):
        super().__init__()
        self.register_buffer('mean', torch.tensor(np.zeros(20) if mean is None else mean, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(np.ones(20) if scale is None else scale, dtype=torch.float32))
        widths = [20, *hidden]
        layers = []
        for a, b in zip(widths, widths[1:]):
            layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(widths[-1], 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, context, graph, extra):
        x = extra if extra.shape[-1] == 20 else pooled_inputs(s, context, graph)
        score = self.net((x-self.mean)/self.scale).squeeze(-1)
        return torch.stack([-score/2, score/2], -1)


def make_model(cfg, kind):
    normalization = cfg.get('_normalization', {})
    return PooledUpdate(**cfg['models'][kind], dropout=cfg['training']['dropout'], **normalization)


class PooledData(base.FoldData):
    def __init__(self, common, features, device):
        super().__init__(common, features, device)
        self.pooled = pooled_inputs(self.values['s'], self.values['context'], self.values['J'])

    def batch(self, ids):
        batch = super().batch(ids)
        batch['extra'] = self.pooled[ids]
        return batch


def normalization(x, qids):
    weights = rules.equal_question_weights(qids, x.shape[1])
    values = np.asarray(x, dtype=np.float64).reshape(-1, 20)
    mean = weights @ values
    scale = np.sqrt(weights @ ((values-mean)**2))
    scale[scale < 1e-10] = 1.
    return {'mean': mean.tolist(), 'scale': scale.tolist()}


def prepare(cfg):
    root, source = Path(cfg['output']), Path(cfg['reference'])
    if (root/'prepared.json').exists():
        validate(cfg)
        return
    reference = reference_driver.validate(config(cfg['reference_config']))
    for key, value in cfg['training'].items():
        if key not in ['torch_threads', 'aggregate_minutes']:
            assert value == reference['config']['training'][key], key
    root.mkdir(parents=True, exist_ok=True)
    common = dict(np.load(source/'development.npz', allow_pickle=False))
    index = json.loads((source/'episode_index.json').read_text())
    assert all(x['official_split'] == 'train' and x['graph_role'] == 'fit_graph' for x in index)
    assert len(set(common['episode_qids'])) == 20
    copied = []
    for name in ['development.npz', 'episode_index.json']:
        shutil.copyfile(source/name, root/name)
        copied.append(name)
    for fold in reference['folds']:
        destination = root/'folds'/fold['id']
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['features.npz', 'ising.json', 'manifest.json']:
            relative = Path('folds')/fold['id']/name
            shutil.copyfile(source/relative, root/relative)
            copied.append(str(relative))
        data = PooledData(common, dict(np.load(destination/'features.npz', allow_pickle=False)), 'cpu')
        ids = data.ids(fold['train_questions'])
        assert not set(data.qids[ids]) & set(fold['assessment_questions'])
        record = normalization(data.pooled[ids].numpy(), data.qids[ids])
        json_write(destination/'normalization.json', {**record, 'fitted_questions': fold['train_questions']})
        copied.append(str(Path('folds')/fold['id']/'normalization.json'))
    previous = json.loads(Path('runs/transformer_rule_distillation_v1/protocol.json').read_text())
    prior = json.loads(Path('runs/transformer_rule_distillation_v1/prior_artifacts.json').read_text())
    # Include the completed interpretation experiment as immutable evidence too.
    for p in Path('runs/transformer_rule_distillation_v1').rglob('*'):
        if p.is_file():
            prior[str(p)] = digest(p)
    for p, sha in {**prior, **previous['active_study_frozen_files']}.items():
        assert digest(p) == sha, p
    cohort = [{'phase': 'tune', 'fold': f['id'], 'outer': f['outer'], 'model': kind,
               'candidate': c['id'], 'seed': cfg['training']['tuning_seed']}
              for f in reference['folds'] if f['phase'] == 'tune'
              for kind in cfg['models'] for c in base.candidates(cfg)]
    json_write(root/'prior_artifacts.json', prior)
    json_write(root/'protocol.json', {
        'created_utc': now(), 'config': cfg, 'folds': reference['folds'],
        'candidates': base.candidates(cfg), 'tuning_cohort': cohort,
        'parameters': {k: sum(p.numel() for p in make_model(cfg, k).parameters()) for k in cfg['models']},
        'linear_parameters': 21, 'new_neural_tuning_fits': len(cohort), 'new_neural_refits': 24,
        'new_linear_tuning_fits': 36, 'new_linear_refits': 4,
        'implementation_sha256': digest(__file__), 'config_sha256': digest(CONFIG),
        'training_driver_sha256': digest(base.__file__), 'rule_driver_sha256': digest(rules.__file__),
        'core_sha256': code_digest(), 'reference_protocol_sha256': digest(source/'protocol.json'),
        'active_study_frozen_files': previous['active_study_frozen_files'],
        'prior_artifacts_sha256': digest(root/'prior_artifacts.json'),
        'input_policy': 'Twenty nonconstant features: 15 receiver persona/question PCA context components and products, own current spin, friendly/unfriendly raw spin sums, friendly/unfriendly degrees. No sender identities, sender personas, individual neighbor opinions, text, future information or question IDs. Input means/scales use fitting questions only.',
        'neural_policy': 'Two GELU MLPs, 809 and 3457 weights. Same six candidates, dropout, AdamW, batch of 32 graph snapshots, clipping, stopping and four outer/three inner question folds as the frozen 818-weight graph reference. Factory override changes only the model; extra supplies precomputed equivalent pools. New fits use CPU. Architecture and normalization change together, so this is an alternative predictor comparison, not a surgical attention ablation.',
        'linear_policy': 'Jointly fit the same 20 features plus intercept directly to observed labels, with training-only standardization and convex equal-question BCE plus ridge on non-intercept coefficients. Three inner-selected penalties. This differs from the original two-stage Ising estimator and is a functional-form control.',
        'selection_policy': 'All neural/linear choices and final models freeze before outer evaluation. Equal-question NLL selects each family within outer training questions. Neural seeds 142/143/144 are scored separately and averaged, not ensembled. No model family is selected from outer scores.',
        'interpretation': 'Exploratory reused GPT development data; no equivalence or causal claim. Failure of a pooled architecture does not prove pooling is insufficient. Report fitting gaps, cap status and question-specific effects.',
        'new_source_calls': 0, 'new_assessment_data_read': False,
    })
    json_write(root/'prepared.json', {'protocol_sha256': digest(root/'protocol.json'),
                                    'artifacts': {f: digest(root/f) for f in copied}})
    print(json.dumps({'prepared': True, 'neural_fits': 168, 'linear_fits': 40,
                      'parameters': {k: sum(p.numel() for p in make_model(cfg, k).parameters()) for k in cfg['models']}}), flush=True)


def validate(cfg):
    root = Path(cfg['output'])
    p = json.loads((root/'protocol.json').read_text())
    assert p['config'] == cfg and p['config_sha256'] == digest(CONFIG)
    assert p['implementation_sha256'] == digest(__file__)
    assert p['training_driver_sha256'] == digest(base.__file__)
    assert p['rule_driver_sha256'] == digest(rules.__file__)
    assert p['core_sha256'] == code_digest() == base.CORE_SHA
    assert p['reference_protocol_sha256'] == digest(Path(cfg['reference'])/'protocol.json')
    prepared = json.loads((root/'prepared.json').read_text())
    assert prepared['protocol_sha256'] == digest(root/'protocol.json')
    for f, sha in prepared['artifacts'].items():
        assert digest(root/f) == sha, f
    for f, sha in p['active_study_frozen_files'].items():
        assert digest(f) == sha, f
    assert digest(root/'prior_artifacts.json') == p['prior_artifacts_sha256']
    return p


def run_neural(cfg, phase):
    root = Path(cfg['output']); protocol = validate(cfg)
    common = dict(np.load(root/'development.npz', allow_pickle=False))
    if phase == 'tune':
        cohort = protocol['tuning_cohort']
    else:
        selected = json.loads((root/'selection.json').read_text())
        assert selected['tuning_complete_sha256'] == digest(root/'tune_complete.json')
        assert selected['protocol_sha256'] == digest(root/'protocol.json')
        cohort = [{**r, 'phase': 'refit', 'seed': seed} for r in selected['choices']
                  for seed in cfg['training']['refit_seeds']]
    timer = base.session(root, cfg); manifests = {}; key = None
    with patch.object(base, 'make_model', make_model):
        for item in cohort:
            fold = next(f for f in protocol['folds'] if f['id'] == item['fold'])
            if key != fold['id']:
                features = dict(np.load(root/'folds'/fold['id']/'features.npz', allow_pickle=False))
                data = PooledData(common, features, 'cpu')
                norms = json.loads((root/'folds'/fold['id']/'normalization.json').read_text())
                local_cfg = {**cfg, '_normalization': {k: norms[k] for k in ['mean', 'scale']}}
                key = fold['id']
            out, record = base.train_one(local_cfg, protocol, data, fold, item, 'cpu',
                                         timer['deadline'], item.get('refit_steps'))
            if record['status'] != 'complete':
                raise RuntimeError('Time guard reached; completed fits and checkpoint retained')
            manifests[str(out.relative_to(root))] = digest(out/'manifest.json')
    assert len(manifests) == (144 if phase == 'tune' else 24)
    json_write(root/f'{phase}_complete.json', {'completed_utc': now(), 'fits': len(manifests),
        'protocol_sha256': digest(root/'protocol.json'), 'manifests': manifests, 'environment': environment()})


def select_neural(cfg):
    root = Path(cfg['output']); p = validate(cfg)
    runs = base.completed_runs(root, 'tune'); choices = []; details = []
    for outer in range(4):
        for kind in cfg['models']:
            options = []
            for c in p['candidates']:
                rows = [r for _, r in runs if r['outer'] == outer and r['model'] == kind and r['candidate'] == c['id']]
                assert len(rows) == 3
                options.append({'outer': outer, 'model': kind, 'candidate': c['id'],
                    'nll': float(np.mean([r['validation']['nll'] for r in rows])),
                    'steps': [r['selected_step'] for r in rows]})
            best = min(options, key=lambda x: (x['nll'], x['candidate']))
            choices.append({'fold': f'o{outer}_refit', 'outer': outer, 'model': kind,
                'candidate': best['candidate'], 'refit_steps': int(math.ceil(np.median(best['steps'])*1.5))})
            details.extend(options)
    json_write(root/'selection.json', {'frozen_utc': now(), 'choices': choices, 'all_candidates': details,
        'protocol_sha256': digest(root/'protocol.json'), 'tuning_complete_sha256': digest(root/'tune_complete.json'),
        'selection_uses_outer_outcomes': False})


def fit_linear(x, target, qids, penalty):
    weights = rules.equal_question_weights(qids, x.shape[1])
    flat = np.asarray(x, dtype=np.float64).reshape(-1, 20)
    flat = np.column_stack([np.ones(len(flat)), flat])
    standardized, mean, scale = rules.standardize(flat, weights)
    result = minimize(rules.objective, np.zeros(21), args=(standardized, target.ravel(), weights, penalty),
                      jac=True, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-8})
    assert result.success and np.abs(result.jac).max() < 5e-6
    raw = result.x/scale; raw[0] -= mean@raw
    return {'weights': result.x.tolist(), 'mean': mean.tolist(), 'scale': scale.tolist(),
            'raw_weights': raw.tolist(), 'gradient_maximum': float(np.abs(result.jac).max()),
            'iterations': int(result.nit), 'penalty': penalty}


def linear_predict(record, x):
    xx = np.concatenate([np.ones((*x.shape[:-1], 1)), x], -1)
    return rules.predict(record, xx)


def run_linear(cfg):
    root = Path(cfg['output']); p = validate(cfg)
    common = dict(np.load(root/'development.npz', allow_pickle=False)); rows = []
    for fold in p['folds']:
        if fold['phase'] != 'tune':
            continue
        data = PooledData(common, dict(np.load(root/'folds'/fold['id']/'features.npz', allow_pickle=False)), 'cpu')
        train, val = data.ids(fold['train_questions']), data.ids(fold['validation_questions'])
        x = data.pooled.numpy()
        for penalty in cfg['linear']['penalties']:
            fit = fit_linear(x[train], data.y[train], data.qids[train], penalty)
            metric, _ = base.metrics(linear_predict(fit, x[val]), data.y[val], data.qids[val])
            rows.append({'outer': fold['outer'], 'fold': fold['id'], 'penalty': penalty, **metric})
    tuning = pd.DataFrame(rows); tuning.to_csv(root/'linear_inner_scores.csv', index=False)
    candidates = tuning.groupby(['outer', 'penalty']).nll.mean().reset_index()
    selected = candidates.sort_values(['nll', 'penalty'], ascending=[True, False]).groupby('outer').head(1)
    json_write(root/'linear_selection.json', {'frozen_utc': now(), 'choices': selected.to_dict('records'),
                                             'outer_outcomes_scored': False})
    fitted = []
    for fold in p['folds']:
        if fold['phase'] != 'refit':
            continue
        data = PooledData(common, dict(np.load(root/'folds'/fold['id']/'features.npz', allow_pickle=False)), 'cpu')
        ids = data.ids(fold['train_questions'])
        penalty = float(selected[selected.outer == fold['outer']].iloc[0].penalty)
        record = fit_linear(data.pooled[ids].numpy(), data.y[ids], data.qids[ids], penalty)
        fitted.append({**record, 'outer': fold['outer'], 'fold': fold['id'], 'training_questions': fold['train_questions']})
    json_write(root/'linear_refits.json', {'frozen_utc': now(), 'fits': fitted, 'outer_outcomes_scored': False})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'tune', 'select', 'refit', 'linear', 'fit'])
    args = parser.parse_args(); cfg = config(CONFIG)
    torch.set_num_threads(cfg['training']['torch_threads'])
    with threadpool_limits(limits=1):
        if args.stage == 'prepare': prepare(cfg)
        elif args.stage == 'select': select_neural(cfg)
        elif args.stage == 'linear': run_linear(cfg)
        elif args.stage == 'fit':
            run_neural(cfg, 'tune'); select_neural(cfg); run_neural(cfg, 'refit'); run_linear(cfg)
            root = Path(cfg['output'])
            json_write(root/'all_fits_frozen.json', {'frozen_utc': now(), 'outer_outcomes_scored': False,
                'artifacts': {n: digest(root/n) for n in ['refit_complete.json', 'selection.json', 'linear_refits.json', 'linear_selection.json']}})
        else: run_neural(cfg, args.stage)


if __name__ == '__main__': main()
