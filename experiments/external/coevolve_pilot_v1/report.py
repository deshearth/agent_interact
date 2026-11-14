"""Reproduce stored development predictions, then render the pilot report."""
from __future__ import annotations
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
import sklearn
import scipy

from ..common import ROOT, now, sha256, write_json
from .models import Inputs, make_model
from .train import metrics, predict


def main():
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    run, data_root, report = [ROOT/cfg[k] for k in ['run_root', 'data_root', 'report_root']]
    protocol = json.loads((run/'protocol.json').read_text())
    for name, expected in protocol['code_sha256'].items():
        assert sha256(ROOT/name) == expected, name
    prepared = json.loads((run/'prepared.json').read_text())
    assert sha256(data_root/'derived/pilot_graphs.npz') == prepared['arrays_sha256']
    arrays = dict(np.load(data_root/'derived/pilot_graphs.npz', allow_pickle=False))
    dev = arrays['partition'] != 'assessment'
    arrays = {k: v[dev] for k, v in arrays.items()}; arrays.pop('probability_before')
    data = Inputs(arrays); torch.set_num_threads(1)
    checked = []; maximum_difference = 0.
    for folder in sorted((run/'fits').glob('*/*/*')):
        record = json.loads((folder/'complete.json').read_text())
        assert sha256(folder/'checkpoint.pt') == record['checkpoint_sha256']
        checkpoint = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
        model = make_model(checkpoint['model'], checkpoint['normalization'], checkpoint['dropout'])
        model.load_state_dict(checkpoint['state_dict'])
        assert sum(x.numel() for x in model.parameters()) == 795
        for role in ['train', 'validation']:
            saved = dict(np.load(folder/(role+'_predictions.npz'), allow_pickle=False))
            ids = saved['index']; assert (data.partitions[ids] == role).all()
            fresh = predict(model, data, ids)
            difference = float(np.max(np.abs(fresh-saved['probability'])))
            assert difference <= 1e-7, (folder, role, difference)
            maximum_difference = max(maximum_difference, difference)
            m = metrics(fresh, data.target[ids].numpy(), data.qids[ids])
            for key in ['nll', 'error']:
                assert abs(m[key]-record[role][key]) < 1e-8
        checked.append(str(folder.relative_to(run)))
    assert len(checked) == 12
    baseline_difference = 0.
    for kind in ['markov', 'local_logistic', 'identity_logistic', 'aggregate_logistic']:
        folder = run/'baselines'/kind; model = json.loads((folder/'model.json').read_text())
        record = json.loads((folder/'complete.json').read_text())
        for role in ['train', 'validation']:
            saved = dict(np.load(folder/(role+'_predictions.npz'), allow_pickle=False)); ids = saved['index']
            assert (data.partitions[ids] == role).all()
            if kind == 'markov':
                fresh = np.array(model['transition_probabilities'])[data.own[ids]]
            else:
                x = {'local_logistic': data.local, 'identity_logistic': data.local_identity,
                     'aggregate_logistic': data.pooled}[kind].numpy()[ids]
                # Match StandardScaler.transform's recorded float32 arithmetic:
                # it casts each fitted statistic before subtracting/dividing.
                z = x.copy()
                z -= np.asarray(model['mean'], dtype=x.dtype)
                z /= np.asarray(model['scale'], dtype=x.dtype)
                fresh = softmax(z @ np.asarray(model['coef'], dtype=x.dtype).T+
                                np.asarray(model['intercept'], dtype=x.dtype), axis=-1)
            difference = float(np.max(np.abs(fresh-saved['probability'])))
            assert difference <= 1e-7, (kind, role, difference)
            baseline_difference = max(baseline_difference, difference)
            m = metrics(fresh, data.target[ids].numpy(), data.qids[ids])
            for key in ['nll', 'error']:
                assert abs(m[key]-record[role][key]) < 1e-8
    protection = json.loads((ROOT/'runs/external/isolation_manifest.json').read_text())
    for name, expected in protection['files'].items():
        assert sha256(ROOT/name) == expected, name
    verification = {'verified_utc': now(), 'neural_checkpoints': len(checked),
                    'max_reloaded_probability_difference': maximum_difference,
                    'reconstructed_baselines': 4, 'max_baseline_probability_difference': baseline_difference,
                    'protected_files_unchanged': len(protection['files']),
                    'assessment_scored': False, 'checked_checkpoints': checked,
                    'source_sha256': sha256(data_root/'derived/pilot_graphs.npz'),
                    'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                    'numpy': np.__version__, 'scikit_learn': sklearn.__version__,
                                    'scipy': scipy.__version__}}
    write_json(run/'verification.json', verification)
    results = json.loads((run/'development_results.json').read_text())['results']
    names = {'markov': 'Empirical Markov', 'local_logistic': 'Local-opinion logistic',
             'identity_logistic': 'Local + receiver identity', 'aggregate_logistic': 'Full aggregate logistic',
             'pooled_mlp': 'Pooled MLP', 'tiny_attention': 'Tiny attention'}
    rows = [{'model': names[r['model']], 'parameters': r['parameters'],
             'train_error': r['train']['error'], 'validation_error': r['validation']['error'],
             'train_nll': r['train']['nll'], 'validation_nll': r['validation']['nll']} for r in results]
    report.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report/'metrics.csv', index=False)
    byq = []
    for r in results:
        for q, scores in r['validation']['by_question'].items():
            byq.append({'model': names[r['model']], 'question': q, **scores})
    pd.DataFrame(byq).to_csv(report/'validation_by_question.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    labels = [names[r['model']] for r in results]
    colors = ['#a8b2bf']*4+['#178584', '#b34f2f']
    x = np.arange(len(rows))
    ax[0].barh(x, [100*r['validation_error'] for r in rows], color=colors)
    ax[0].set_yticks(x, labels); ax[0].invert_yaxis(); ax[0].set_xlabel('Validation error (%) — lower is better')
    ax[0].axvline(100*prepared['development']['validation']['persistence_error'], color='#555', linestyle='--', label='Persistence')
    ax[0].set_title('Dashed line: persistence (3.28%)', fontsize=10, loc='left', pad=8)
    ax[1].barh(x, [r['validation_nll'] for r in rows], color=colors)
    ax[1].set_yticks(x, labels); ax[1].invert_yaxis(); ax[1].set_xlabel('Validation NLL — lower is better')
    for axis, key, scale, fmt in [(ax[0], 'validation_error', 100, '.2f'), (ax[1], 'validation_nll', 1, '.3f')]:
        for i, row in enumerate(rows):
            axis.text(row[key]*scale+(.035 if scale == 100 else .002), i, format(row[key]*scale, fmt),
                      va='center', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .4})
        axis.set_xlim(0, max(r[key] for r in rows)*scale*1.18)
    fig.suptitle('Coevolution pilot: four validation statements\nBoth neural models: 795 weights; assessment remains unscored', fontsize=13)
    fig.savefig(report/'validation_comparison.png', dpi=170)
    fig.savefig(report/'validation_comparison.pdf'); plt.close(fig)
    lines = ['# Coevolution: first external prediction pilot', '',
             'The compact attention model improves validation probability scores over the logistic baselines, but its advantage over an equally sized pooled neural network is small and metric-dependent. This is a validation-selected pilot on four statements, not an independent assessment result.', '',
             '| Predictor | Parameters / stored coefficients | Training error | Validation error | Training NLL | Validation NLL |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['model']} | {r['parameters']} | {r['train_error']:.2%} | {r['validation_error']:.2%} | {r['train_nll']:.5f} | {r['validation_nll']:.5f} |")
    lines += ['', 'Neural rows average the three seed scores, not ensemble predictions. Logistic counts are stored softmax coefficients, not identifiable degrees of freedom; Markov reports six free transition probabilities.', '',
              '![Validation comparison](validation_comparison.png)', '',
              '## Data and input definition', '',
              'The source is [CoevolveSim](https://github.com/carlomarxdk/coevolve-sim), pinned at `b5ca480ccf292d0fa6a531475abbf47433f3a342`. This bounded pilot uses 40 runs: the 20 medical-indication statements under matched specialist roles, on `erdos-renyi-01` and `watts-strogatz-01`. Each run has 48 agents and ten updates. This is not a replication of all 32 original graph realizations.', '',
              'The split has 12 fitting, 4 validation, and 4 reserved assessment statement groups, balanced by source true/false labels. All graphs and agents for a statement stay together. Ground truth and question identifiers are excluded from predictor features. We predict the native next belief category (true/false/neither), not answer correctness.', '',
              'The recorded neighbor summaries provide role identities and counts by current belief. The models use that information, the receiver’s current belief/identity, round, and degree. Source probability distributions are excluded. The full aggregate baseline and MLP receive exact role-by-belief fractions, so they retain the same visible multiset information as attention. The graph carries no friendship signs.', '',
              'The attention architecture is adapted from the original width-eight/four-head predictor to these inputs and three outputs; it has 795 weights, not 818. The pooled model is `68 → 8 → 20 → 3`, also 795 weights. Both are trained from scratch; no text encoder or pretrained neural weights are used in this pilot.', '',
              '## Fitting and selection', '',
              'Four logistic penalties are compared on validation NLL. Each neural family uses two learning rates and three seeds, with up to 1,536 updates, checks every 128 steps, and validation early stopping. The mean seed NLL selects the learning rate. Training-only standardization is used for logistic models and the pooled MLP. The selected rate is 0.003 for both neural models.', '',
              'A preliminary run with an accidentally unmatched 895-weight MLP versus 795-weight attention was archived after the parameter-count check failed. Exact original source and outputs are retained under `runs/external/coevolve_pilot_v1/superseded/unmatched_budget_895_795/`. The reported table uses the corrected matched-budget refits. No assessment scores were involved in the correction.', '',
              '## Interpretation and limits', '',
              '- Persistence error is 11.51% on fitting statements and only 3.28% on validation statements. Lower validation than training error reflects a substantially easier question group; it is not evidence of generalization beyond fitting performance.',
              '- The pooled MLP has slightly lower validation classification error; attention has slightly lower validation NLL. Four statements do not support a strong architecture ranking, and the validation set selects the models.',
              '- The native source interaction compresses arguments to belief/role summaries. This pilot cannot answer whether full argument text would add predictive information.',
              '- These rates are not directly comparable to the original Ising/818-weight results: questions, source models, labels, graph protocol, and sampling differ. The appropriate simple controls here are categorical transition and logistic models.',
              '- This is a fixed-budget pilot. Several fits reach the training cap; their selected training errors are not architecture capacity bounds.', '',
              '## Verification', '',
              f"All {len(checked)} checkpoints were reloaded independently and reproduce saved fitting and validation probabilities with maximum difference {maximum_difference:.3g}. All four baselines were reconstructed from saved coefficients with maximum difference {baseline_difference:.3g}. The causal-input, masking, permutation, parameter-count, and question-overlap checks pass. All {len(protection['files'])} protected original implementation/protocol files remain unchanged.", '',
              'Assessment predictions were neither generated nor scored. The trainer removes reserved examples before fitting and does not use the recorded source confidence vectors.', '',
              'See `audit.md`, `metrics.csv`, and `validation_by_question.csv` in this folder for data coverage and per-statement scores.', '']
    (report/'results.md').write_text('\n'.join(lines))
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
