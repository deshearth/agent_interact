"""Independent final statistics, graph packing, learned branches and artifact audit."""
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import pair_token_attention_study as study
from scripts import breadth_argument_study as shared

def recompute(lp, target, qids):
    lp = lp.astype(np.float64)
    np.testing.assert_allclose(np.exp(lp).sum(-1), 1, atol=2e-6)
    values = {'nll': -np.take_along_axis(lp, target[..., None], -1)[..., 0],
              'error': (lp.argmax(-1) != target).astype(float),
              'brier_two_class': 2*(np.exp(lp[..., 1])-target)**2}
    rows = {metric: {q: float(v[qids == q].mean()) for q in sorted(set(qids))} for metric, v in values.items()}
    return {m: float(np.mean(list(q.values()))) for m, q in rows.items()}, rows

def audit():
    cfg = study.config(); root = Path(cfg['output']); p = study.validate(cfg); frozen = study.validate_frozen(cfg)
    assert frozen['tuning_fits'] == 72 and frozen['refits_completed'] == 24
    for group in ('source_artifacts', 'prior_artifacts'):
        for file, sha in p[group].items(): assert digest(file) == sha, file
    oldroot = Path(cfg['compact_reference'])
    for path in (root/'data').iterdir():
        assert digest(path) == digest(oldroot/'data'/path.name), path
    encoding = study.read(oldroot/'encoding_replay.json'); assert encoding['status'] == 'passed'
    assert encoding['maximum_cpu_vs_gpu_component_difference'] < 5e-5
    results = study.read(root/'results/summary.json'); index = study.read(root/'results/predictions.json')
    assert len(index) == 134 and sum(r['reused'] for r in index) == 74
    assert results['models_frozen_sha256'] == digest(root/'models_frozen.json')
    assert results['previous_results_sha256'] == digest(oldroot/'results/summary.json')
    common = dict(np.load(root/'data/development.npz', allow_pickle=False)); allq = {}; largest = 0.
    for record in index:
        assert digest(record['path']) == record['sha256']
        with np.load(record['path'], allow_pickle=False) as a:
            lp, y, qids, pairs = [a[k] for k in ('logp', 'target', 'qids', 'pairs')]; e, t = pairs.T
            np.testing.assert_array_equal(common['spins'][e, t+1] == 1, y)
            np.testing.assert_array_equal(common['spins'][e, t], a['current'])
            np.testing.assert_array_equal(common['episode_qids'][e], qids)
            assert set(qids) == set(p['fold'][record['split']+'_questions'])
            assert not set(qids) & set(p['fold']['assessment_questions'])
            actual, rows = recompute(lp, y, qids)
            expected = next(r for r in results['by_seed'] if all(r[k] == record[k] for k in ('model', 'seed', 'split')))
            for metric, value in actual.items():
                difference = abs(expected[metric]-value); largest = max(largest, difference); assert difference < 1e-12
            wrong = lp.argmax(-1) != y; switch = (a['current'] > 0) != y
            for label, mask in [('switch', switch), ('stay', ~switch)]:
                np.testing.assert_allclose(expected[label+'_error'], wrong[mask].mean(), atol=1e-12)
            allq[record['model'], record['seed'], record['split']] = rows
    for model, splits in results['summary'].items():
        for split, metrics in splits.items():
            for metric in ('nll', 'error', 'brier_two_class'):
                values = [np.mean(list(q[metric].values())) for (m, seed, s), q in allq.items() if m == model and s == split]
                np.testing.assert_allclose(metrics[metric], np.mean(values), atol=1e-12)
    questions = sorted(p['fold']['validation_questions']); rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    draws = rng.integers(0, len(questions), (cfg['evaluation']['bootstrap_replicates'], len(questions)))
    def scores(model, metric):
        return np.mean([np.array([q[metric][i] for i in questions]) for (m, seed, role), q in allq.items()
                        if m == model and role == 'validation'], axis=0)
    for row in results['paired_differences']:
        delta = scores(row['left'], row['metric'])-scores(row['right'], row['metric'])
        np.testing.assert_allclose(row['mean_difference'], delta.mean(), atol=1e-12)
        np.testing.assert_allclose(row['ci95'], np.quantile(delta[draws].mean(1), [.025, .975]), atol=1e-12)
    assert len(results['paired_differences']) == 84
    torch.set_num_threads(cfg['training']['torch_threads']); data = study.Data(cfg, 'cpu')
    reconstructed = np.zeros_like(common['J'])
    for e in range(len(common['J'])):
        for receiver in range(common['J'].shape[1]):
            live = data.signs[e, receiver] != 0
            senders = data.neighbors[e, receiver, live]
            assert len(senders) == len(set(senders))
            reconstructed[e, receiver, senders] = data.signs[e, receiver, live]
    np.testing.assert_array_equal(reconstructed, common['J'])
    selection = study.read(root/'selection.json'); checks = []
    execution = study.read(root/'parallel_execution.json')
    assert execution['scientific_protocol_sha256'] == digest(root/'protocol.json')
    assert execution['controller_sha256'] == digest('scripts/finish_pair_token_attention.py')
    tuning_completed = [datetime.fromisoformat(study.read(path)['completed_utc'])
                        for path in (root/'tune').glob('*/manifest.json')]
    refits_started = [datetime.fromisoformat(study.read(Path(path)/'manifest.json')['created_utc'])
                     for path in frozen['refits']]
    assert max(tuning_completed) <= min(refits_started)
    # Recompute candidate selection without invoking the original chooser.
    for choice in selection['choices']:
        manifests = [study.read(path) for path in (root/'tune').glob(f'{choice["width"]}-{choice["condition"]}-*/manifest.json')]
        assert len(manifests) == 9
        chosen = sorted(manifests, key=lambda r: (r['validation']['nll'], r['candidate']))[0]
        assert choice['candidate'] == chosen['candidate'] and choice['refit_steps'] == chosen['selected_step']
    with study.adapted_trainer():
        for directory in frozen['refits']:
            path = Path(directory); m = study.read(path/'manifest.json'); width = m['width']
            data.set_width(width); data.condition = m['condition']
            choice = next(r for r in selection['choices'] if r['width'] == width and r['condition'] == m['condition'])
            assert m['candidate'] == choice['candidate'] and m['selected_step'] == choice['refit_steps']
            assert m['selection_sha256'] == digest(root/'selection.json')
            assert m['seed'] in cfg['training']['refit_seeds']
            model = study.make_model(cfg, width).eval()
            ck = torch.load(path/'best.pt', map_location='cpu', weights_only=False); model.load_state_dict(ck['model'])
            assert shared.base.model_weights_sha(model) == m['selected_weights_sha256']
            assert not any(hasattr(model, key) for key in ('edge_bias', 'gamma', 'strength'))
            assert sum(v.numel() for v in model.parameters()) == cfg['expected_parameters'][width]
            train = data.ids(p['fold']['train_questions'])[:16]; batch = data.batch(train)
            score = study.score_model(model, batch)
            torch.nn.functional.binary_cross_entropy_with_logits(score, batch['target']).backward()
            text_gradient = float(model.text_projection.weight.grad.norm())
            sign_gradient = float(model.pair_encoder[0].weight.grad[:, -1].norm())
            assert text_gradient > 0 and sign_gradient > 0 and np.isfinite([text_gradient, sign_gradient]).all()
            initial = study.make_model(cfg, width, m['seed'])
            text_change = float((model.text_projection.weight.detach()-initial.text_projection.weight.detach()).norm())
            pair_change = float((model.pair_encoder[0].weight.detach()-initial.pair_encoder[0].weight.detach()).norm())
            assert text_change > 0 and pair_change > 0
            # Dense masks and packed incoming edges must agree after learning too.
            with torch.no_grad():
                dense, social, _ = model(batch['s'], batch['context'], batch['J'], batch['text'], True)
                packed = score
                packed_difference = float((dense[..., 1]-dense[..., 0]-packed).abs().max())
                assert packed_difference < 5e-5
                empty = ~batch['J'].ne(0).any(-1)
                if empty.any(): assert social[empty].abs().max() == 0
            val = data.ids(p['fold']['validation_questions']); positions = np.arange(0, len(val), 31)
            lp = shared.predictions(model, data, val[positions], 32)
            saved = np.load(root/'results'/f'pair_{width}_{m["condition"]}-s{m["seed"]}-validation.npz', allow_pickle=False)['logp'][positions]
            probability_difference = float(np.abs(np.exp(lp)-np.exp(saved)).max()); assert probability_difference < 5e-5
            checks.append({'width': width, 'condition': m['condition'], 'seed': m['seed'],
                'text_gradient_norm': text_gradient, 'relationship_gradient_norm': sign_gradient,
                'text_weight_change': text_change, 'pair_weight_change': pair_change,
                'packed_vs_dense_score_difference': packed_difference,
                'replayed_probability_difference': probability_difference,
                'class_disagreements': int((lp.argmax(-1) != saved.argmax(-1)).sum())})
    for width in cfg['models']:
        for seed in [cfg['training']['tuning_seed'], *cfg['training']['refit_seeds']]:
            phase = 'tune' if seed == cfg['training']['tuning_seed'] else 'refit'
            initials = {study.read(path)['initial_weights_sha256'] for path in (root/phase).glob(f'{width}-*-s{seed}/manifest.json')}
            assert len(initials) == 1
    audit = {'status': 'passed', 'completed_utc': shared.now(), 'code_sha256': digest(__file__),
        'prediction_arrays_recomputed': len(index), 'paired_contrasts_recomputed': len(results['paired_differences']),
        'maximum_metric_difference': largest, 'source_episodes_verified': len(p['source_artifacts']),
        'prior_artifacts_verified': len(p['prior_artifacts']), 'frozen_artifacts_verified': len(frozen['artifacts']),
        'all_inputs_identical_to_prior': True, 'all_graph_edges_reconstructed': True, 'maximum_neighbors': data.neighbors.shape[-1],
        'all_tuning_finished_before_any_refit': True, 'parallel_controller_hash_verified': True,
        'learned_branch_checks': checks, 'reference_encoder_replay_sha256': digest(oldroot/'encoding_replay.json'),
        'results_sha256': {str(path): digest(path) for path in (root/'results').iterdir() if path.is_file()},
        'assessment_loaded': False, 'source_llm_calls': 0}
    json_write(root/'completion_audit.json', audit)
    print({k: v for k, v in audit.items() if not isinstance(v, (list, dict))}, flush=True)

if __name__ == '__main__': audit()
