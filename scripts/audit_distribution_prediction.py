"""Independent soft-target scoring, split checks, and saved-model replay."""
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, rel_entr
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import distribution_prediction_study as study
from scripts import distribution_prediction_references as references

METRICS = ('nll', 'kl', 'mae', 'mse', 'within_05', 'mode_error')


def independent_scores(lp, target):
    lp = lp.astype(np.float64)
    truth = np.stack([1-target, target], -1)
    probability = np.exp(lp)
    np.testing.assert_allclose(probability.sum(-1), 1, rtol=0, atol=2e-6)
    absolute = np.abs(probability[..., 1]-target)
    divergence = rel_entr(truth, probability).sum(-1)
    assert divergence.min() > -2e-6
    return dict(nll=-(truth*lp).sum(-1), kl=np.maximum(divergence, 0),
                mae=absolute, mse=absolute**2, within_05=(absolute <= .05).astype(float),
                mode_error=(lp.argmax(-1) != (target > .5)).astype(float))


def independent_design(data, ids, enriched):
    graph = data.common['J'][data.common['pairs'][ids, 0]].astype(np.float64)
    current = data.s[ids].astype(np.float64)
    positive, negative = np.maximum(graph, 0), np.minimum(graph, 0)
    pos = (positive*current[:, None, :]).sum(-1)
    neg = (negative*current[:, None, :]).sum(-1)
    columns = [data.context[ids].astype(np.float64), pos[..., None], neg[..., None]]
    if enriched:
        mean_spin = 2*data.current[ids]-1
        columns.extend([mean_spin[..., None], (positive*mean_spin[:, None, :]).sum(-1)[..., None],
                        (negative*mean_spin[:, None, :]).sum(-1)[..., None]])
    columns.append((pos-neg)[..., None])
    return np.concatenate(columns, -1)


def main():
    cfg = study.config(); root = Path(cfg['output']); protocol = study.validate(cfg)
    frozen = study.read(root/'models_frozen.json'); results = study.read(root/'results.json')
    for path, sha in frozen['artifacts'].items():
        assert digest(path) == sha, path
    assert results['models_frozen_sha256'] == digest(root/'models_frozen.json')
    assert frozen['frozen_utc'] < results['completed_utc']
    assert (frozen['new_tuning_fits'], frozen['new_neural_refits'], frozen['new_ising_fits']) == (54, 18, 2)
    assert not results['assessment_loaded'] and not frozen['assessment_loaded']
    fold = protocol['fold']; train, val, assessment = [set(fold[s+'_questions']) for s in ('train', 'validation', 'assessment')]
    inner_fit, inner_val = set(protocol['inner']['fit']), set(protocol['inner']['validation'])
    assert (len(train), len(val), len(assessment), len(inner_fit), len(inner_val)) == (320, 80, 40, 256, 64)
    assert not (train & val or train & assessment or val & assessment or inner_fit & inner_val)
    assert inner_fit | inner_val == train
    torch.set_num_threads(2); data = study.Data(cfg, 'full'); inner = study.Data(cfg, 'inner')
    assert set(data.qids) == train | val and not (set(data.qids) & assessment)
    raw = dict(np.load(Path(cfg['probability_reference'])/'probabilities.npz', allow_pickle=False))
    e, t = data.common['pairs'].T
    np.testing.assert_array_equal(data.current, raw['single'][e, t])
    np.testing.assert_array_equal(data.y, raw['single'][e, t+1])
    assert data.y.dtype.kind == 'f' and np.any((data.y > 0)&(data.y < 1))
    assert data.y.shape == (3200, 32)
    choices = study.read(root/'selection.json')['choices']; assert len(choices) == 6
    maximum = 0.; tuning_entropy_differences = []
    tuning = [study.read(path) for path in (root/'tune').glob('*/manifest.json')]
    assert len(tuning) == 54
    for choice in choices:
        group = [r for r in tuning if r['width'] == choice['width']]
        assert len(group) == 9 and len({r['initial_weights_sha256'] for r in group}) == 1
        winner = min(group, key=lambda r: (r['validation']['nll'], r['candidate']))
        assert choice['candidate'] == winner['candidate'] and choice['refit_steps'] == winner['selected_step']
        for record in group:
            assert record['train']['questions'] == 256 and record['validation']['questions'] == 64
            assert record['status'] == 'complete' and record['condition'] == 'distribution' and record['seed'] == 42
            path = root/'tune'/f"{record['width']}-distribution-{record['candidate']}-s42"/'validation_logp.npy'
            ids = inner.ids(protocol['inner']['validation']); lp = np.load(path, allow_pickle=False)
            arrays = independent_scores(lp, inner.y[ids])
            for metric in METRICS:
                maximum = max(maximum, abs(arrays[metric].mean()-record['validation'][metric]))
            # Fixed target entropy makes cross-entropy and KL choose the same recipe.
            tuning_entropy_differences.append(record['validation']['nll']-record['validation']['kl'])
        assert min(group, key=lambda r: (r['validation']['kl'], r['candidate']))['candidate'] == choice['candidate']
    assert np.ptp(tuning_entropy_differences) < 2e-9
    for suffix in ('opinion', 'probability'):
        a = {r['initial_weights_sha256'] for r in tuning if r['width'] == 'ordered_choice_'+suffix}
        b = {r['initial_weights_sha256'] for r in tuning if r['width'] == 'ordered_arguments_'+suffix}
        assert a == b
    for folder in frozen['refits']:
        record = study.read(Path(folder)/'manifest.json')
        choice = next(c for c in choices if c['width'] == record['width'])
        assert record['selected_step'] == choice['refit_steps'] and record['completed_utc'] < frozen['frozen_utc']
        assert record['parameters'] == cfg['models'][record['width']]['parameters']
        assert record['validation'] is None and record['train']['questions'] == 320
        assert record['seed'] in cfg['training']['refit_seeds'] and record['status'] == 'complete'
    original = {(r['model'], r['seed'], r['split'], r['qid']): r for r in study.read(root/'seed_by_question.json')}
    original_secondary = {(r['model'], r['seed'], r['split'], r['qid']): r for r in study.read(root/'majority_secondary.json')}
    recomputed, replays, sensitivities, optimizer_checks = {}, [], [], []
    assert len(results['prediction_files']) == 40
    for item in results['prediction_files']:
        assert digest(item['path']) == item['sha256']
        saved = dict(np.load(item['path'], allow_pickle=False))
        name, seed, split = item['model'], item['seed'], item['split']; ids = data.ids(fold[split+'_questions'])
        assert len(ids) == (2560 if split == 'train' else 640)
        for label, expected in [('pairs', data.common['pairs'][ids]), ('target_probability', data.y[ids]),
                                ('current_probability', data.current[ids]), ('qids', data.qids[ids]), ('hard_target', data.hard[ids])]:
            np.testing.assert_array_equal(saved[label], expected)
        np.testing.assert_array_equal(saved['probability'], np.exp(saved['logp'].astype(float)[..., 1]))
        arrays = independent_scores(saved['logp'], data.y[ids])
        majority = lambda p: np.clip(10*p**3*(1-p)**2+5*p**4*(1-p)+p**5, 0, 1)
        secondary = dict(majority_probability_mae=np.abs(majority(saved['probability'])-majority(data.y[ids])),
                         sampled_majority_accuracy=((saved['probability'] > .5) == data.hard[ids]).astype(float))
        for qid in sorted(set(data.qids[ids])):
            take = data.qids[ids] == qid; key = (name, seed, split, qid)
            values = {metric: float(value[take].mean()) for metric, value in arrays.items()}
            for metric in METRICS:
                maximum = max(maximum, abs(values[metric]-original[key][metric]))
            for metric, value in secondary.items():
                maximum = max(maximum, abs(float(value[take].mean())-original_secondary[key][metric]))
            recomputed[key] = values
        if split != 'validation':
            continue
        if name.startswith('ising'):
            fitted = study.read(item['checkpoint']); weights = np.array(fitted['weights'])
            assert len(weights) == (22 if fitted['probability_input'] else 19)
            scores = independent_design(data, ids, fitted['probability_input'])@weights
            replay = np.stack([-np.logaddexp(0, scores), -np.logaddexp(0, -scores)], -1)
            train_ids = data.ids(fold['train_questions'])
            x = independent_design(data, train_ids, fitted['probability_input']).reshape(-1, len(weights))
            target = data.y[train_ids].ravel(); mu, sd = x.mean(0), x.std(0)
            mu[0] = 0; sd[0] = 1; sd = np.where(sd < 1e-8, 1, sd)
            z = (x-mu)/sd; gradient = z.T@(expit(x@weights)-target)/len(target)
            gradient[1:16] += .001*(weights*sd)[1:16]
            optimizer_checks.append(dict(model=name, stage_two_max_gradient=float(np.abs(gradient[:-1]).max()),
                                         frozen_unsigned_coefficient_gradient=float(gradient[-1])))
            assert np.abs(gradient[:-1]).max() < 1e-6
        else:
            fitted = study.make_model(cfg, name)
            checkpoint = torch.load(item['checkpoint'], map_location='cpu', weights_only=False)
            fitted.load_state_dict(checkpoint['model']); fitted.eval()
            with study.adapted_trainer():
                replay = study.trainer.predictions(fitted, data, ids, 128)
            batch = data.batch(data.ids(fold['train_questions'])[:8])
            for feature in ('current_probability', 'text_arguments', 'text_conclusion'):
                batch[feature] = batch[feature].detach().clone().requires_grad_(True)
            score = study.score_model(fitted, batch)
            gradients = torch.autograd.grad(score.sum(), [batch[k] for k in ('current_probability', 'text_arguments', 'text_conclusion')], allow_unused=True)
            gradient_norms = {k: (0. if g is None else float(g.detach().abs().sum())) for k, g in zip(('current_probability', 'text_arguments', 'text_conclusion'), gradients)}
            assert (gradient_norms['current_probability'] > 0) == cfg['models'][name]['probability_input']
            assert (gradient_norms['text_arguments'] > 0) == name.startswith('ordered_arguments')
            assert (gradient_norms['text_conclusion'] > 0) == name.startswith('ordered_choice')
            changed = dict(batch); changed['target'] = 1-batch['target']
            with torch.no_grad():
                np.testing.assert_array_equal(study.score_model(fitted, changed).numpy(), score.detach().numpy())
            sensitivities.append(dict(model=name, seed=seed, input_gradient_absolute_sums=gradient_norms,
                added_probability_weight_absolute_sum=float(fitted.encoder[0].weight[:, 18].detach().abs().sum()) if cfg['models'][name]['probability_input'] else None,
                subset='first eight training snapshots', target_replacement_changes_output=False))
        difference = float(np.abs(replay-saved['logp']).max())
        assert difference < 1e-12
        replays.append(dict(model=name, seed=seed, max_log_probability_difference=difference,
                            modal_choice_differences=int((replay.argmax(-1) != saved['logp'].argmax(-1)).sum())))
        print(f'Verified {name}, seed {seed}', flush=True)
    assert len(recomputed) == len(original) == 8000
    averaged = {}
    for name, splits in results['summary'].items():
        seeds = [-1] if name.startswith('ising') else cfg['training']['refit_seeds']
        for split in ('train', 'validation'):
            for qid in fold[split+'_questions']:
                averaged[name, split, qid] = {metric: float(np.mean([recomputed[name, seed, split, qid][metric] for seed in seeds])) for metric in METRICS}
            for metric in METRICS:
                value = np.mean([averaged[name, split, qid][metric] for qid in fold[split+'_questions']])
                maximum = max(maximum, abs(value-splits[split][metric]))
    for row in study.read(root/'by_question.json'):
        for metric in METRICS:
            maximum = max(maximum, abs(row[metric]-averaged[row['model'], row['split'], row['qid']][metric]))
    questions = sorted(val); rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    draws = rng.integers(len(questions), size=(cfg['evaluation']['bootstrap_replicates'], len(questions)))
    assert len(results['paired_differences']) == 60
    for contrast in results['paired_differences']:
        delta = np.array([averaged[contrast['left'], 'validation', q][contrast['metric']]-averaged[contrast['right'], 'validation', q][contrast['metric']] for q in questions])
        maximum = max(maximum, abs(delta.mean()-contrast['difference']), float(np.max(np.abs(np.quantile(delta[draws].mean(1), [.025, .975])-contrast['ci95']))))
    reference = references.evaluate()
    train_ids = data.ids(fold['train_questions'])
    assert reference['training_mean'] == float(data.y[train_ids].mean())
    for name in ('persistence', 'training_mean'):
        predicted = data.current if name == 'persistence' else np.full_like(data.current, reference['training_mean'])
        arrays = independent_scores(np.stack([np.log1p(-predicted), np.log(predicted)], -1), data.y)
        for split in ('train', 'validation'):
            for q in fold[split+'_questions']:
                take = data.qids == q
                averaged[name, split, q] = {metric: float(arrays[metric][take].mean()) for metric in ('kl', 'mae')}
            for metric in ('kl', 'mae'):
                value = np.mean([averaged[name, split, q][metric] for q in fold[split+'_questions']])
                maximum = max(maximum, abs(value-reference['summary'][name][split][metric]))
    for row in reference['by_question']:
        for metric in ('kl', 'mae'):
            maximum = max(maximum, abs(row[metric]-averaged[row['model'], row['split'], row['qid']][metric]))
    for contrast in reference['paired_differences']:
        delta = np.array([averaged[contrast['left'], 'validation', q][contrast['metric']]-averaged[contrast['right'], 'validation', q][contrast['metric']] for q in questions])
        maximum = max(maximum, abs(delta.mean()-contrast['difference']), float(np.max(np.abs(np.quantile(delta[draws].mean(1), [.025, .975])-contrast['ci95']))))
    assert maximum < 1e-12
    assert all(r['modal_choice_differences'] == 0 for r in replays)
    result = dict(passed=True, completed_utc=study.now(), audit_code_sha256=digest(__file__),
        results_sha256=digest(root/'results.json'), models_frozen_sha256=digest(root/'models_frozen.json'),
        verified_prediction_files=40, verified_seed_question_rows=8000, verified_contrasts=60,
        verified_tuning_fits=54, verified_neural_refits=18, verified_ising_fits=2,
        max_metric_difference=maximum, checkpoint_replays=replays, trained_input_sensitivity=sensitivities,
        ising_optimizer_checks=optimizer_checks, inner_selection_entropy_spread=float(np.ptp(tuning_entropy_differences)),
        reference_results_sha256=digest(root/'reference_results.json'), verified_reference_contrasts=6,
        assessment_loaded=False, source_calls=0)
    json_write(root/'completion_audit.json', result)
    print({k: v for k, v in result.items() if not isinstance(v, list)}, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
