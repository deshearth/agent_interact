"""Frozen scoring plan: new ordinary attention versus the exact prior predictions."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import pair_token_attention_study as study
from scripts import breadth_argument_study as shared
from scripts import transformer_generalization as base

def previous_name(name):
    return 'signed_' + name if name.startswith(('pico_', 'nano_')) else name

def paired_contrasts(rows, cfg):
    questions = sorted({r['qid'] for r in rows if r['split'] == 'validation'})
    grouped = {}
    for r in rows:
        if r['split'] != 'validation': continue
        for metric in ('nll', 'error', 'brier_two_class'):
            grouped.setdefault((r['model'], metric, r['qid']), []).append(r[metric])
    rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    draws = rng.integers(0, len(questions), (cfg['evaluation']['bootstrap_replicates'], len(questions)))
    pairs = []
    for width in cfg['models']:
        name = 'pair_' + width
        pairs.extend([(name+'_arguments', name+'_conclusion', 'arguments_vs_choice'),
            (name+'_arguments', 'signed_'+width+'_arguments', 'new_vs_previous_arguments'),
            (name+'_conclusion', 'signed_'+width+'_conclusion', 'new_vs_previous_choice'),
            (name+'_arguments', 'opinion_'+width.split('_')[0], 'arguments_vs_original_opinion'),
            (name+'_arguments', 'I0', 'arguments_vs_Ising'),
            (name+'_arguments', name+'_arguments_matched', 'postfit_matched_text_sensitivity')])
    pairs.extend([('pair_pico_full_arguments', 'pair_pico_pca8_arguments', 'compression'),
                  ('pair_nano_full_arguments', 'pair_nano_pca8_arguments', 'compression'),
                  ('pair_nano_pca8_arguments', 'pair_pico_pca8_arguments', 'width'),
                  ('pair_nano_full_arguments', 'pair_pico_full_arguments', 'width')])
    result = []
    for left, right, category in pairs:
        for metric in ('nll', 'error', 'brier_two_class'):
            delta = np.array([np.mean(grouped[left, metric, q])-np.mean(grouped[right, metric, q]) for q in questions])
            result.append({'left': left, 'right': right, 'category': category, 'metric': metric,
                'mean_difference': float(delta.mean()), 'ci95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
                'questions': len(questions)})
    return result

def evaluate(cfg, device):
    frozen = study.validate_frozen(cfg); root = Path(cfg['output']); out = root/'results'
    if (out/'summary.json').exists(): return
    out.mkdir(exist_ok=True)
    protocol = study.read(root/'protocol.json'); data = study.Data(cfg, device)
    seed_rows, question_rows, index, replay, sensitivity, references = [], [], [], [], [], []
    oldroot = Path(cfg['compact_reference'])
    old = study.read(oldroot/'results/summary.json')
    parameters = {previous_name(k): v for k, v in old['parameters'].items()}

    def record(name, seed, split, lp, ids, path, origin, reused=False):
        metric, rows = base.metrics(lp, data.y[ids], data.qids[ids])
        metric.update(shared.additional_metrics(lp, data, ids))
        seed_rows.append({'model': name, 'seed': seed, 'split': split, **metric})
        question_rows.extend({'model': name, 'seed': seed, 'split': split, **r} for r in rows)
        if not reused:
            np.savez_compressed(path, logp=lp, target=data.y[ids], current=data.s[ids],
                                qids=data.qids[ids], pairs=data.common['pairs'][ids])
        index.append({'path': str(path), 'sha256': digest(path), 'model': name, 'seed': seed,
                      'split': split, 'origin': origin, 'reused': reused})

    for item in study.read(oldroot/'results/predictions.json'):
        assert digest(item['path']) == item['sha256']
        ids = data.ids(protocol['fold'][item['split']+'_questions'])
        saved = np.load(item['path'], allow_pickle=False)
        for key, expected in [('qids', data.qids[ids]), ('target', data.y[ids]),
                              ('pairs', data.common['pairs'][ids]), ('current', data.s[ids])]:
            np.testing.assert_array_equal(saved[key], expected)
        record(previous_name(item['model']), item['seed'], item['split'], saved['logp'], ids,
               Path(item['path']), item['origin'], True)

    with study.adapted_trainer():
        for directory in frozen['refits']:
            path = Path(directory); m = study.read(path/'manifest.json')
            width, condition, seed = (m[k] for k in ('width', 'condition', 'seed'))
            data.set_width(width); data.condition = condition
            model = study.make_model(cfg, width).to(device)
            ck = torch.load(path/'best.pt', map_location=device, weights_only=False)
            model.load_state_dict(ck['model'])
            assert base.model_weights_sha(model) == m['selected_weights_sha256']
            name = 'pair_'+width+'_'+condition; parameters[name] = m['parameters']
            references.append({'path': str(path), 'manifest_sha256': digest(path/'manifest.json'),
                               'width': width, 'condition': condition, 'seed': seed})
            for split in ('train', 'validation'):
                ids = data.ids(protocol['fold'][split+'_questions'])
                lp = shared.predictions(model, data, ids, cfg['training']['evaluation_batch_size'])
                record(name, seed, split, lp, ids, out/f'{name}-s{seed}-{split}.npz', str(path))
                if split == 'train':
                    expected = np.load(path/'train_logp.npy', allow_pickle=False)
                    delta = float(np.abs(np.exp(lp)-np.exp(expected)).max()); assert delta < 5e-5
                    replay.append({'model': name, 'seed': seed, 'maximum_probability_difference': delta,
                                   'class_disagreements': int((lp.argmax(-1) != expected.argmax(-1)).sum())})
                if split == 'validation' and condition == 'arguments':
                    swapped = shared.predictions(model, data, ids, cfg['training']['evaluation_batch_size'], 'matched')
                    record(name+'_matched', seed, split, swapped, ids, out/f'{name}_matched-s{seed}-{split}.npz', str(path))
                    sensitivity.append({'model': name, 'seed': seed,
                        'probability_mae': float(np.abs(np.exp(lp[..., 1])-np.exp(swapped[..., 1])).mean()),
                        'class_disagreement': float((lp.argmax(-1) != swapped.argmax(-1)).mean())})
            print(json.dumps({'scored': name, 'seed': seed}), flush=True)
    summary = {}
    for name in sorted({r['model'] for r in seed_rows}):
        summary[name] = {}
        for split in ('train', 'validation'):
            rows = [r for r in seed_rows if r['model'] == name and r['split'] == split]
            if rows:
                summary[name][split] = {k: float(np.mean([r[k] for r in rows]))
                    for k in ('nll', 'error', 'brier_two_class', 'switch_error', 'stay_error')}
    for name, splits in old['summary'].items():
        for split, metrics in splits.items():
            for metric, value in metrics.items():
                np.testing.assert_allclose(summary[previous_name(name)][split][metric], value, atol=1e-12, rtol=0)
    json_write(out/'by_question.json', question_rows); json_write(out/'predictions.json', index)
    json_write(out/'summary.json', {'completed_utc': shared.now(), 'summary': summary, 'by_seed': seed_rows,
        'paired_differences': paired_contrasts(question_rows, cfg), 'text_sensitivity': sensitivity,
        'parameters': parameters, 'refit_sources': references, 'reproduction': replay,
        'models_frozen_sha256': digest(root/'models_frozen.json'), 'previous_results_sha256': digest(oldroot/'results/summary.json'),
        'new_fit_seconds': frozen['fit_seconds'], 'new_tuning_fits': frozen['tuning_fits'], 'new_refits': frozen['refits_completed'],
        'reused_prediction_arrays': sum(r['reused'] for r in index), 'encoder_parameters': 22713216,
        'assessment_loaded': False, 'source_llm_calls': 0, 'evaluation_device': str(device),
        'scope': protocol['evaluation_policy']})
    print(json.dumps({'evaluation': 'complete', 'new_models': {k: v for k, v in summary.items() if k.startswith('pair_')}}), flush=True)

if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('--device', default='cpu'); args = parser.parse_args()
    cfg = study.config(); torch.set_num_threads(cfg['training']['torch_threads']); evaluate(cfg, args.device)
