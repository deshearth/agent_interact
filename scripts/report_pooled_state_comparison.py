"""Assess the frozen pooled-state comparison on the old GPT development folds."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from scripts import transformer_generalization as base
from scripts import distill_transformer_rule as rules
from scripts import pooled_state_comparison as study

LABELS = {'I0': 'Original Ising', 'I0_current': 'Ising + own-opinion correction',
          'pooled_linear': 'Pooled linear rule', 'pooled_small': 'Pooled MLP, 809',
          'pooled_large': 'Pooled MLP, 3,457', 'nano': 'Graph transformer, 818',
          'pooled_small_doubled': 'Pooled MLP, 809, doubled training'}
METRICS = ['error', 'nll', 'brier_two_class']


def compare_by_question(frame, comparisons, seed=420916, draws=3000):
    question = frame[frame.split == 'assessment'].groupby(['model', 'qid'])[METRICS].mean().reset_index()
    qids = sorted(question.qid.unique()); assert len(qids) == 20
    resamples = np.random.default_rng(seed).integers(0, len(qids), (draws, len(qids)))
    rows = []
    for model, reference in comparisons:
        a = question[question.model == model].set_index('qid').loc[qids]
        b = question[question.model == reference].set_index('qid').loc[qids]
        for metric in METRICS:
            difference = (a[metric]-b[metric]).to_numpy()
            lo, hi = np.quantile(difference[resamples].mean(1), [.025, .975])
            rows.append({'model': model, 'reference': reference, 'metric': metric,
                         'difference': float(difference.mean()), 'lower': float(lo), 'upper': float(hi),
                         'questions_better': int((difference < 0).sum()), 'questions': len(qids)})
    return pd.DataFrame(rows)


def evaluate():
    cfg = config(study.CONFIG); root = Path(cfg['output']); source = Path(cfg['reference'])
    p = study.validate(cfg); freeze_path = root/'all_fits_frozen.json'
    frozen = json.loads(freeze_path.read_text())
    assert not frozen['outer_outcomes_scored']
    for path, sha in frozen['artifacts'].items(): assert digest(root/path) == sha, path
    refits = base.completed_runs(root, 'refit')
    tuning = base.completed_runs(root, 'tune')
    assert len(refits) == 24 and len(tuning) == 144
    assert all(r['completed_utc'] <= frozen['frozen_utc'] for _, r in refits)
    extended = root/'doubled_duration'
    extension_freeze = json.loads((extended/'frozen.json').read_text())
    assert not extension_freeze['outer_outcomes_scored']
    for path, sha in extension_freeze['artifacts'].items(): assert digest(extended/path) == sha, path
    added = base.completed_runs(extended, 'refit'); assert len(added) == 12
    assert all(r['completed_utc'] <= extension_freeze['frozen_utc'] for _, r in added)
    prefixes = json.loads((extended/'prefix_checks.json').read_text())
    assert len(prefixes) == 12 and all(r['exact_original_prefix'] for r in prefixes)
    refits += [(d, {**r, 'architecture': r['model'], 'model': 'pooled_small_doubled'}) for d, r in added]
    linear = json.loads((root/'linear_refits.json').read_text())
    assert linear['frozen_utc'] <= frozen['frozen_utc']
    common = dict(np.load(root/'development.npz', allow_pickle=False))
    episode, round_id = common['pairs'].T
    all_qids = common['episode_qids'][episode]
    target = (common['spins'][episode, round_id+1] == 1).astype(np.int64)
    current = common['spins'][episode, round_id] > 0
    output = root/'evaluation'; output.mkdir(exist_ok=True)
    rows = []; checks = []; subgroup = []; overlap = []
    expected = pd.read_csv(source/'outer_by_question.csv')
    baseline = np.load('runs/ising_initial_state_probe_v1/predictions.npz', allow_pickle=False)
    np.testing.assert_array_equal(baseline['target'], target)
    for fold in p['folds']:
        if fold['phase'] != 'refit': continue
        feature = dict(np.load(root/'folds'/fold['id']/'features.npz', allow_pickle=False))
        data = study.PooledData(common, feature, 'cpu')
        original_data = base.FoldData(common, feature, 'cpu')
        ids_by_split = {'train': data.ids(fold['train_questions']), 'assessment': data.ids(fold['assessment_questions'])}
        new = [(directory, r) for directory, r in refits if r['fold'] == fold['id']]
        # Reconstruct all new checkpoints and independently reproduce saved fitting predictions.
        loaded = []
        for directory, record in new:
            m = study.make_model(cfg, record.get('architecture', record['model']))
            ck = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
            m.load_state_dict(ck['model']); m.eval()
            assert base.model_weights_sha(m) == record['selected_weights_sha256']
            train_lp = base.predictions(m, data, ids_by_split['train'], False)
            saved = np.load(directory/'train_logp.npy', allow_pickle=False)
            difference = float(np.abs(np.exp(train_lp[..., 1])-np.exp(saved[..., 1])).max())
            assert difference < 1e-6
            checks.append({'model': record['model'], 'outer': fold['outer'], 'seed': record['seed'],
                           'maximum_probability_difference': difference})
            loaded.append((m, record, train_lp))
        rule = next(r for r in linear['fits'] if r['fold'] == fold['id'])
        graph_runs = []
        for path in (source/'refit'/fold['id']).glob('nano-*/manifest.json'):
            record = json.loads(path.read_text()); assert record['status'] == 'complete'
            for filename in ['best.pt', 'train_logp.npy']:
                assert digest(path.parent/filename) == record['artifacts'][filename]
            graph_runs.append((path.parent, record))
        assert len(graph_runs) == 3
        for split, ids in ids_by_split.items():
            yy, qq = data.y[ids], data.qids[ids]
            predictions = [('I0', -1, rules.log_prob(feature['prior_logit'][ids])),
                           ('pooled_linear', -1, study.linear_predict(rule, data.pooled[ids].numpy()))]
            if split == 'assessment':
                predictions.append(('I0_current', -1, rules.log_prob(baseline['current_state'][ids])))
            for model, record, train_lp in loaded:
                lp = train_lp if split == 'train' else base.predictions(model, data, ids, False)
                predictions.append((record['model'], record['seed'], lp))
                if split == 'assessment':
                    np.save(output/f"{fold['id']}__{record['model']}__s{record['seed']}_logp.npy", lp, allow_pickle=False)
            for directory, record in graph_runs:
                if split == 'train': lp = np.load(directory/'train_logp.npy', allow_pickle=False)
                else:
                    assessed = source/'assessment'/f"{fold['id']}__{directory.name}"
                    audit = json.loads((assessed/'complete.json').read_text())
                    assert audit['run_manifest_sha256'] == digest(directory/'manifest.json')
                    assert digest(assessed/'assessment_logp.npy') == audit['artifacts']['assessment_logp.npy']
                    lp = np.load(assessed/'assessment_logp.npy', allow_pickle=False)
                predictions.append(('nano', record['seed'], lp))
                # A small independent forward reconstruction checks reference row alignment.
                model = base.make_model(config(cfg['reference_config']), 'nano')
                model.load_state_dict(torch.load(directory/'best.pt', map_location='cpu', weights_only=False)['model'])
                positions = np.linspace(0, len(ids)-1, 32, dtype=int)
                direct = base.predictions(model, original_data, ids[positions], False)
                difference = float(np.abs(np.exp(direct[..., 1])-np.exp(lp[positions, ..., 1])).max())
                assert difference < 5e-5
                checks.append({'model': 'nano', 'outer': fold['outer'], 'seed': record['seed'],
                               'split': split, 'maximum_probability_difference': difference})
            reference_lp = predictions[0][2]
            for kind, seed, lp in predictions:
                _, byq = base.metrics(lp, yy, qq)
                rows.extend({'model': kind, 'outer': fold['outer'], 'seed': seed, 'split': split, **r} for r in byq)
                if kind in ['I0', 'nano', 'I0_current']:
                    saved_rows = expected[(expected.model == kind) & (expected.outer == fold['outer']) &
                                          (expected.seed == seed) & (expected.split == split)].set_index('qid')
                    for row in byq:
                        for metric in METRICS:
                            np.testing.assert_allclose(row[metric], saved_rows.loc[row['qid'], metric], atol=1e-10, rtol=1e-7)
                if split != 'assessment': continue
                p_ising = np.exp(reference_lp[..., 1]); confidence = np.maximum(p_ising, 1-p_ising)
                masks = {'ising_confidence_lt_0.8': confidence < .8,
                         'ising_confidence_ge_0.8': confidence >= .8,
                         'stay': yy == current[ids], 'switch': yy != current[ids]}
                wrong_ref = reference_lp.argmax(-1) != yy; wrong = lp.argmax(-1) != yy
                for qid in sorted(set(qq)):
                    take_q = qq == qid
                    overlap.append({'model': kind, 'seed': seed, 'qid': str(qid),
                                    'corrected': float((wrong_ref[take_q] & ~wrong[take_q]).mean()),
                                    'introduced': float((~wrong_ref[take_q] & wrong[take_q]).mean()),
                                    'shared': float((wrong_ref[take_q] & wrong[take_q]).mean())})
                    for group, mask in masks.items():
                        take = take_q[:, None] & mask
                        if not take.any(): continue
                        suby, sublp = yy[take], lp[take]
                        subgroup.append({'model': kind, 'seed': seed, 'qid': str(qid), 'group': group,
                                         'transitions': int(take.sum()), 'errors': int((sublp.argmax(-1) != suby).sum()),
                                         'nll_sum': float(-sublp[np.arange(len(suby)), suby].sum())})
    frame = pd.DataFrame(rows); frame.to_csv(output/'by_question.csv', index=False)
    question = frame.groupby(['split', 'model', 'qid'])[METRICS].mean().reset_index()
    summary = question.groupby(['split', 'model'])[METRICS].mean().reset_index()
    summary.to_csv(output/'summary.csv', index=False)
    comparisons = [(k, r) for k in ['pooled_linear', *cfg['models']] for r in ['I0', 'nano']]
    comparisons += [('pooled_small', 'pooled_linear'), ('pooled_large', 'pooled_small')]
    comparisons += [('pooled_small_doubled', r) for r in ['pooled_small', 'nano', 'I0']]
    compare_by_question(frame, comparisons, **{'seed': cfg['evaluation']['bootstrap_seed'],
                        'draws': cfg['evaluation']['bootstrap_replicates']}).to_csv(output/'paired_differences.csv', index=False)
    pd.DataFrame(checks).to_csv(output/'checkpoint_reproduction.csv', index=False)
    pd.DataFrame(subgroup).to_csv(output/'subgroups.csv', index=False)
    pd.DataFrame(overlap).to_csv(output/'error_overlap.csv', index=False)
    cap = [{'model': r['model'], 'outer': r['outer'], 'fold': r['fold'], 'candidate': r['candidate'],
            'steps': r['steps'], 'selected_step': r['selected_step'],
            'selected_at_cap': r['selected_step'] == cfg['training']['max_steps'],
            'seconds': r['seconds']} for _, r in tuning]
    pd.DataFrame(cap).to_csv(output/'tuning_diagnostics.csv', index=False)
    prior = json.loads((root/'prior_artifacts.json').read_text())
    for path, sha in {**prior, **p['active_study_frozen_files']}.items(): assert digest(path) == sha, path
    json_write(root/'evaluation_complete.json', {'completed_utc': base.now(), 'all_fits_frozen_sha256': digest(freeze_path),
        'doubled_duration_frozen_sha256': digest(extended/'frozen.json'),
        'report_driver_sha256': digest(__file__), 'new_neural_fits': 180, 'new_linear_fits': 40,
        'extended_training_prefixes_reproduced_exactly': 12,
        'prior_artifacts_unchanged': len(prior), 'active_study_frozen_files_unchanged': len(p['active_study_frozen_files']),
        'maximum_checkpoint_probability_difference': max(r['maximum_probability_difference'] for r in checks),
        'new_source_calls': 0, 'new_assessment_data_read': False,
        'artifacts': {str(path): digest(path) for path in output.iterdir() if path.is_file()}})
    print(summary.to_string(index=False), flush=True)


def plot():
    root = Path(config(study.CONFIG)['output'])
    frame = pd.read_csv(root/'evaluation/by_question.csv')
    summary = pd.read_csv(root/'evaluation/summary.csv').set_index(['split', 'model'])
    comparisons = pd.read_csv(root/'evaluation/paired_differences.csv')
    order = ['I0', 'pooled_linear', 'pooled_small', 'pooled_small_doubled', 'pooled_large', 'nano']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    colors = ['#64748b', '#64748b', '#059669', '#84a72a', '#0f766e', '#2563eb']
    x = np.arange(len(order))
    for split, marker, label in [('train', 'o', 'Fitting questions'), ('assessment', 'D', 'Older held-out questions')]:
        for i, kind in enumerate(order):
            value = summary.loc[(split, kind), 'error']*100
            axes[0].scatter(i+(-.10 if split == 'train' else .10), value, marker=marker,
                            facecolors='white' if split == 'train' else colors[i], edgecolors=colors[i],
                            s=52, label=label if i == 0 else None, zorder=3)
    axes[0].set_xticks(x, ['Ising\n19', 'Linear\n21', 'Pooled\n809', '809\n2× time', 'Pooled\n3,457', 'Graph\n818'])
    axes[0].set_ylabel('Next-opinion error (%)'); axes[0].set_title('Fit and transfer')
    axes[0].legend(frameon=False, fontsize=8); axes[0].grid(axis='y', alpha=.15)
    for i, kind in enumerate(['pooled_linear', 'pooled_small', 'pooled_large']):
        row = comparisons[(comparisons.model == kind) & (comparisons.reference == 'nano') & (comparisons.metric == 'nll')].iloc[0]
        axes[1].errorbar(row.difference, i, xerr=[[row.difference-row.lower], [row.upper-row.difference]],
                         color=colors[order.index(kind)], marker='o', capsize=4, lw=1.8)
    axes[1].axvline(0, color='#2563eb', lw=1, ls='--')
    axes[1].set_yticks(range(3), ['Linear, 21', 'Pooled, 809', 'Pooled, 3,457'])
    axes[1].invert_yaxis(); axes[1].set_xlabel('NLL difference from graph transformer')
    axes[1].set_title('Question-bootstrap intervals')
    q = frame[frame.split == 'assessment'].groupby(['model', 'qid']).nll.mean().unstack('model')
    axes[2].scatter(q.nano, q.pooled_small, s=35, color='#059669', alpha=.8)
    limit = max(q.nano.max(), q.pooled_small.max())*1.06
    axes[2].plot([0, limit], [0, limit], '--', color='#94a3b8', lw=1)
    axes[2].set(xlim=(0, limit), ylim=(0, limit), xlabel='Graph transformer NLL', ylabel='Pooled 809 NLL',
                title='Twenty development questions')
    fig.suptitle('Do individual-neighbor features add predictive value?', fontsize=14, y=1.02)
    fig.text(.5, -.02, 'Previously studied GPT data • scores averaged over three seeds • intervals are exploratory',
             ha='center', color='#475569', fontsize=9)
    fig.tight_layout(w_pad=2.1)
    fig.savefig('reports/pooled_state_comparison.png', bbox_inches='tight')
    fig.savefig('reports/pooled_state_comparison.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        evaluate()
        plot()
