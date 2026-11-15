"""Independent checkpoint, preprocessing and metric checks for the internal pilot."""
from pathlib import Path
import os
import pickle
import sys

import numpy as np
import torch
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import modal_long_context_screen as study


def metrics(p, y):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return dict(accuracy=float(((p > .5) == y).mean()),
                nll=float((-y * np.log(p) - (1 - y) * np.log1p(-p)).mean()))


def selected(model, x, step):
    for i, probability in enumerate(model.staged_predict_proba(x), 1):
        if i == step:
            return probability[:, 1]
    raise AssertionError('Selected iteration missing')


def main():
    p, labels, fit, hold, state = study.data()
    root = study.ROOT
    result = study.read(root / 'results.json')
    enc = study.read(root / 'encoding.json')
    assert result['protocol_sha256'] == digest(root / 'protocol.json')
    for mapping in (result['artifacts'], enc['artifacts']):
        for path, sha in mapping.items():
            assert digest(path) == sha, path
    assert (fit ^ hold).all() and fit.sum() == 16384 and hold.sum() == 4096
    assert set(labels['qids']) == set(p['inner']['fit']) | set(p['inner']['validation'])
    assert set(labels['pairs'][:, 1]) == {0, 4}
    source_probability = np.load(Path(study.source.core.dist.config()['probability_reference']) / 'probabilities.npz')['single']
    e, t = labels['pairs'].T
    np.testing.assert_array_equal(labels['target'], (source_probability[e, t + 1] > .5).ravel())

    vectors = np.load(root / 'data/vectors.npy')
    reduced = np.load(root / 'data/vectors128.npy')
    transform = np.load(root / 'data/pca128.npz')
    np.testing.assert_array_equal(transform['training_indices'], np.flatnonzero(fit))
    np.testing.assert_allclose(transform['mean'], vectors[fit].astype(float).mean(0), atol=1e-14, rtol=0)
    with threadpool_limits(limits=2):
        np.testing.assert_allclose(transform['axes'] @ transform['axes'].T, np.eye(128), atol=2e-12, rtol=0)
        np.testing.assert_allclose((vectors - transform['mean']) @ transform['axes'].T, reduced, atol=6e-8, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-6)
    plan = np.load(root / 'data/plan.npz')
    np.testing.assert_array_equal(np.sort(plan['order']), np.arange(20480))
    assert (plan['lengths'] <= plan['buckets']).all() and plan['lengths'].max() < 8192
    assert enc['tokens_dropped'] == 0 and not enc['independent_windows']

    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(p['encoder']['snapshot'], local_files_only=True)
    model = AutoModel.from_pretrained(p['encoder']['snapshot'], local_files_only=True,
                                     attn_implementation='sdpa', dtype=torch.float32).eval()
    assert sum(x.numel() for x in model.parameters()) == p['frozen_encoder_parameters']
    texts = study.read(study.source.ROOT / 'data/texts.json')
    ids = np.unique(np.concatenate([np.arange(3), np.argsort(plan['lengths'])[-3:]]))
    vector_error = 0.
    for i in ids:
        replay, lengths = study.bench.pooled(model, tokenizer, [texts[i]], 'cpu', int(plan['buckets'][i]))
        np.testing.assert_array_equal(lengths, plan['lengths'][[i]])
        vector_error = max(vector_error, float(np.abs(replay[0] - vectors[i]).max()))
    assert vector_error < 2e-5
    del model

    with (study.source.ROOT / 'fits/state/terminal_model.pkl').open('rb') as stream:
        baseline = pickle.load(stream)
    step = study.read(study.source.ROOT / 'fits/state/complete.json')['selected_step']
    with threadpool_limits(limits=2):
        baseline_p = selected(baseline, state, step)
    offset = logit(np.clip(baseline_p, 1e-7, 1 - 1e-7))
    np.testing.assert_array_equal(offset, np.load(root / 'data/offset.npy'))
    base_metrics = metrics(baseline_p[hold], labels['target'][hold])
    assert base_metrics['accuracy'] == result['baseline_accuracy']
    qids = labels['qids'][hold]
    questions = sorted(set(qids))
    per = {'state': np.array([((baseline_p[hold] > .5) == labels['target'][hold])[qids == q].mean() for q in questions])}
    checks = []
    with threadpool_limits(limits=2):
        for row in result['results']:
            name = row['model']
            folder = root / 'fits' / name
            dim = row['dimension']
            if row['family'] == 'tree':
                x = np.concatenate([state, reduced[:, :dim]], -1)
                with (folder / 'terminal_model.pkl').open('rb') as stream:
                    model = pickle.load(stream)
                pp = selected(model, x, row['selected_step'])
                np.testing.assert_array_equal(pp[hold], np.load(folder / 'internal_probability.npy'))
                history = study.read(folder / 'history.json')
                best = min(history, key=lambda z: (-z['accuracy'], z['nll'], z['step']))
                assert best['step'] == row['selected_step']
                learned = sum(tree[0].nodes.shape[0] for tree in model._predictors[:row['selected_step']])
            else:
                f = np.empty((len(fit), 0)) if dim == 0 else (vectors if dim == 1536 else reduced[:, :dim]).astype(float)
                model = np.load(folder / 'model.npz')
                np.testing.assert_allclose(model['mean'], f[fit].mean(0), atol=1e-14, rtol=0)
                np.testing.assert_allclose(model['scale'], np.maximum(f[fit].std(0), 1e-6), atol=1e-14, rtol=0)
                x = np.column_stack([offset, np.ones(len(fit)), (f - model['mean']) / model['scale']])
                pp = expit(x @ model['weights'])
                np.testing.assert_allclose(pp, np.load(folder / 'probability.npy'), atol=1e-13, rtol=0)
                learned = len(model['weights'])
            train = metrics(pp[fit], labels['target'][fit])
            internal = metrics(pp[hold], labels['target'][hold])
            for key in internal:
                assert abs(internal[key] - row['internal'][key]) < 1e-12
            if 'train' in row:
                for key in train:
                    assert abs(train[key] - row['train'][key]) < 1e-12
            per[name] = np.array([((pp[hold] > .5) == labels['target'][hold])[qids == q].mean() for q in questions])
            checks.append(dict(model=name,train=train,internal=internal,head_nodes_or_parameters=learned,
                               dimension=dim,family=row['family'],promotion_gate_passed=row['promotion_gate_passed']))

    calibration_name = next(r['model'] for r in result['results'] if r['family'] == 'linear' and r['dimension'] == 0)
    calibration_accuracy = float(per[calibration_name].mean())
    assert calibration_accuracy == result['calibration_accuracy']
    draws = np.random.default_rng(92771).integers(64, size=(5000, 64))
    comparisons = []
    for row in result['results']:
        reference = 'state' if row['family'] == 'tree' or result['baseline_accuracy'] >= calibration_accuracy else calibration_name
        delta = per[row['model']] - per[reference]
        assert row['reference_accuracy'] == float(per[reference].mean())
        gate = row['dimension'] > 0 and row['internal']['accuracy'] >= row['reference_accuracy'] + p['promotion_gain']
        assert row['promotion_gate_passed'] == gate
        comparisons.append(dict(model=row['model'],reference=reference,difference=float(delta.mean()),
                                ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
    assert result['promotion_gate_passed'] == any(r['promotion_gate_passed'] for r in result['results'])
    audit = dict(passed=True,completed_utc=study.now(),results_sha256=digest(root/'results.json'),
                 audit_code_sha256=digest(__file__),model_replays=len(checks),checks=checks,comparisons=comparisons,
                 cpu_encoder_replay_texts=len(ids),cpu_max_vector_difference=vector_error,
                 offset_sha256=digest(root/'data/offset.npy'),source_calls=0,outer_validation_scored=False,
                 new_validation_loaded=False,assessment_loaded=False)
    json_write(root/'completion_audit.json',audit)
    gate_text = 'A candidate passed the internal promotion gate; an all-round follow-up is required.' if result['promotion_gate_passed'] else 'No candidate passed the internal promotion gate.'
    lines = ['# Full-context ModernBERT: training-only pilot','',f'**{gate_text}** These scores use two fixed rounds of the internal 256/64 question split. They do not measure all-round accuracy on the 80-question validation anchor and cannot establish 95% performance.','',
             '| Model | Training accuracy | Internal accuracy | Learned head size |','| --- | ---: | ---: | ---: |',
             f"| Frozen state-tree reference | — | {100*result['baseline_accuracy']:.3f}% | 622 input features |"]
    for row in checks:
        lines.append(f"| {row['model']} | {100*row['train']['accuracy']:.3f}% | {100*row['internal']['accuracy']:.3f}% | {row['head_nodes_or_parameters']:,} {'tree nodes' if row['family']=='tree' else 'coefficients'} |")
    lines += ['',f"The [ModernBERT-base checkpoint](https://huggingface.co/answerdotai/ModernBERT-base) has {p['frozen_encoder_parameters']:,} frozen encoder parameters in addition to the fitted prediction heads. This is a test of frozen representations, not end-to-end fine-tuning. Its native full-context attention sees every prompt token together; {p['tokens']:,} tokens were retained, with maximum length {p['maximum_tokens']:,}. No independent windows or truncation were used.", '',
              'The inputs are exactly the source receiver prompts: persona, question, supplied answer options, and actual ordered friendly/unfriendly messages. No correct-answer key, solution, future probability or future message enters these models. Mean and first-token vectors are concatenated; PCA and feature standardization use only the 256 fitting questions.', '',
              'Tree heads append PCA32 or PCA128 to the existing state features. Linear heads correct the frozen state-tree logit using PCA32, PCA128 or all1536 encoder features at four fixed regularization strengths. A calibration-only control prevents mistaking improved logit calibration for useful argument information. Promotion requires at least0.5 percentage points above the matched reference; linear text corrections must also beat the calibration control.', '',
              f"The audit reproduces all{len(checks)} fitted heads, training-only transforms and exact source-target alignment, and re-encodes {len(ids)} prompts on CPU including the three longest. Maximum CPU/GPU vector difference is {vector_error:.2g}. Question-bootstrap intervals are descriptive after internal model selection. No new validation or reserved assessment data was used.", '',
              'All-case outer-validation best results remain88.514% for the neural history/argument model and88.677% for the tree on512 training questions, until a separately frozen all-round follow-up is completed.','']
    Path('reports/modal_long_context_screen_320.md').write_text('\n'.join(lines))
    print(dict(passed=True,model_replays=len(checks),cpu_max_vector_difference=vector_error,promotion_gate_passed=result['promotion_gate_passed']),flush=True)


if __name__ == '__main__':
    main()
