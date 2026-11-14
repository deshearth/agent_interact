"""Train an isolated, validation-only external pilot; no assessment scoring."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

from ..common import ROOT, now, sha256, write_json
from .models import Inputs, make_model


def metrics(probability, target, questions):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.)
    target = np.asarray(target)
    error = probability.argmax(-1) != target
    loss = -np.log(probability[np.arange(len(target)), target])
    by_question = {str(q): {'nll': float(loss[questions == q].mean()),
                            'error': float(error[questions == q].mean()),
                            'n': int((questions == q).sum())} for q in sorted(set(questions))}
    return {'nll': float(np.mean([v['nll'] for v in by_question.values()])),
            'error': float(np.mean([v['error'] for v in by_question.values()])),
            'by_question': by_question}


@torch.no_grad()
def predict(model, data, ids):
    model.eval(); result = []
    for pos in range(0, len(ids), 1024):
        result.append(model(*data.batch(ids[pos:pos+1024])).softmax(-1).numpy())
    return np.concatenate(result)


def neural_fit(kind, seed, lr, cfg, norm, data, train_ids, val_ids, out):
    if (out/'complete.json').exists():
        return json.loads((out/'complete.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); rng = np.random.default_rng(seed+9101)
    model = make_model(kind, norm, cfg['dropout'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    best = float('inf'); best_step = 0; stale = 0; history = []; start = time.monotonic()
    for step in range(1, cfg['max_steps']+1):
        model.train()
        ids = rng.choice(train_ids, size=cfg['batch_size'], replace=True)
        logits = model(*data.batch(ids))
        loss = F.cross_entropy(logits, data.target[ids])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        optimizer.step()
        if step % cfg['evaluate_every'] == 0:
            p = predict(model, data, val_ids)
            m = metrics(p, data.target[val_ids].numpy(), data.qids[val_ids])
            history.append({'step': step, 'validation': m, 'minibatch_nll': float(loss.detach())})
            if m['nll'] < best-cfg['min_delta']:
                best = m['nll']; best_step = step; stale = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= cfg['patience']:
                break
    model.load_state_dict(best_state)
    torch.save({'state_dict': best_state, 'model': kind, 'normalization': norm,
                'dropout': cfg['dropout'], 'seed': seed, 'learning_rate': lr}, out/'checkpoint.pt')
    result = {'model': kind, 'seed': seed, 'learning_rate': lr, 'selected_step': best_step,
              'terminal_step': step, 'parameters': sum(p.numel() for p in model.parameters()),
              'wall_seconds': time.monotonic()-start, 'checkpoint_sha256': sha256(out/'checkpoint.pt')}
    for role, ids in [('train', train_ids), ('validation', val_ids)]:
        p = predict(model, data, ids)
        np.savez_compressed(out/(role+'_predictions.npz'), probability=p, index=ids)
        result[role] = metrics(p, data.target[ids].numpy(), data.qids[ids])
    write_json(out/'history.json', history); write_json(out/'complete.json', result)
    print(json.dumps({'fit': kind, 'seed': seed, 'lr': lr, 'step': best_step,
                      'train_error': result['train']['error'], 'validation_error': result['validation']['error'],
                      'validation_nll': result['validation']['nll']}), flush=True)
    return result


def main():
    folder = Path(__file__).parent
    cfg = json.loads((folder/'config.json').read_text()); tc = json.loads((folder/'training.json').read_text())
    data_root, run = ROOT/cfg['data_root'], ROOT/cfg['run_root']
    prepared = json.loads((run/'prepared.json').read_text())
    source = data_root/'derived/pilot_graphs.npz'
    assert sha256(source) == prepared['arrays_sha256']
    assert sha256(run/'split.json') == prepared['split_sha256']
    arrays = dict(np.load(source, allow_pickle=False))
    # Discard reserved outcomes before constructing the trainer's data object.
    dev = arrays['partition'] != 'assessment'
    arrays = {k: v[dev] for k, v in arrays.items()}
    arrays.pop('probability_before')  # Privileged source-model confidence excluded.
    data = Inputs(arrays); train_ids, val_ids = data.ids('train'), data.ids('validation')
    assert not set(data.qids[train_ids]) & set(data.qids[val_ids])
    assert len(data.ids('assessment')) == 0
    torch.set_num_threads(tc['torch_threads'])
    x = data.pooled[train_ids].numpy()
    mean = x.mean(0); scale = x.std(0); scale[scale < 1e-6] = 1.
    norm = {'mean': mean.tolist(), 'scale': scale.tolist()}
    code_files = [folder/'models.py', folder/'train.py', folder/'training.json', folder/'config.json']
    signatures = {str(p.relative_to(ROOT)): sha256(p) for p in code_files}
    protocol = {'created_utc': now(), 'training': tc, 'code_sha256': signatures,
                'prepared_sha256': sha256(run/'prepared.json'), 'split_sha256': sha256(run/'split.json'),
                'normalization': norm, 'assessment_examples_loaded_into_trainer': 0,
                'parameter_counts': {k: sum(p.numel() for p in make_model(k, norm, tc['dropout']).parameters()) for k in tc['neural_models']}}
    if (run/'protocol.json').exists():
        previous = json.loads((run/'protocol.json').read_text())
        assert previous['code_sha256'] == signatures and previous['prepared_sha256'] == protocol['prepared_sha256']
    else:
        write_json(run/'protocol.json', protocol)
    results = []
    with threadpool_limits(limits=1):
        for kind in ['markov', 'local_logistic', 'identity_logistic', 'aggregate_logistic']:
            out = run/'baselines'/kind; out.mkdir(parents=True, exist_ok=True)
            if kind == 'markov':
                counts = np.ones((3, 3), dtype=float)
                np.add.at(counts, (data.own[train_ids], data.target[train_ids].numpy()), 1.)
                prob = counts/counts.sum(-1, keepdims=True)
                probs = {role: prob[data.own[ids]] for role, ids in [('train', train_ids), ('validation', val_ids)]}
                write_json(out/'model.json', {'transition_probabilities': prob.tolist(), 'pseudocount': 1.})
                parameters = 6
            else:
                values = {'local_logistic': data.local, 'identity_logistic': data.local_identity,
                          'aggregate_logistic': data.pooled}[kind].numpy()
                scaler = StandardScaler().fit(values[train_ids]); transformed = scaler.transform(values)
                options = []
                for c in tc['logistic_C']:
                    fit = LogisticRegression(C=c, max_iter=2000, tol=1e-7, solver='lbfgs').fit(transformed[train_ids], data.target[train_ids].numpy())
                    m = metrics(fit.predict_proba(transformed[val_ids]), data.target[val_ids].numpy(), data.qids[val_ids])
                    options.append((m['nll'], c, fit, m))
                _, c, fit, _ = min(options, key=lambda x: (x[0], x[1]))
                assert fit.classes_.tolist() == [0, 1, 2]
                probs = {role: fit.predict_proba(transformed[ids]) for role, ids in [('train', train_ids), ('validation', val_ids)]}
                parameters = fit.coef_.size+fit.intercept_.size
                write_json(out/'model.json', {'C': c, 'coef': fit.coef_.tolist(), 'intercept': fit.intercept_.tolist(),
                           'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                           'candidates': [{'C': x[1], 'validation': x[3]} for x in options]})
            row = {'model': kind, 'parameters': parameters}
            for role, ids in [('train', train_ids), ('validation', val_ids)]:
                row[role] = metrics(probs[role], data.target[ids].numpy(), data.qids[ids])
                np.savez_compressed(out/(role+'_predictions.npz'), probability=probs[role], index=ids)
            write_json(out/'complete.json', row); results.append(row)
            print(json.dumps({'baseline': kind, 'train_error': row['train']['error'],
                              'validation_error': row['validation']['error'], 'validation_nll': row['validation']['nll']}), flush=True)
        neural = []
        for kind in tc['neural_models']:
            for lr in tc['learning_rates']:
                for seed in tc['seeds']:
                    out = run/'fits'/kind/f'lr_{lr:g}'/f'seed_{seed}'
                    neural.append(neural_fit(kind, seed, lr, tc, norm, data, train_ids, val_ids, out))
        choices = []
        for kind in tc['neural_models']:
            options = []
            for lr in tc['learning_rates']:
                fits = [x for x in neural if x['model'] == kind and x['learning_rate'] == lr]
                options.append({'learning_rate': lr, 'mean_validation_nll': float(np.mean([x['validation']['nll'] for x in fits]))})
            selected = min(options, key=lambda x: (x['mean_validation_nll'], x['learning_rate']))
            fits = [x for x in neural if x['model'] == kind and x['learning_rate'] == selected['learning_rate']]
            choices.append({'model': kind, **selected, 'candidates': options})
            row = {'model': kind, 'parameters': fits[0]['parameters'], 'selected_rate': selected['learning_rate'], 'seeds': fits}
            for role in ['train', 'validation']:
                row[role] = {metric: float(np.mean([x[role][metric] for x in fits])) for metric in ['nll', 'error']}
                row[role]['by_question'] = {q: {metric: float(np.mean([x[role]['by_question'][q][metric] for x in fits])) for metric in ['nll', 'error']} for q in fits[0][role]['by_question']}
            results.append(row)
        write_json(run/'selection.json', {'frozen_utc': now(), 'choices': choices, 'assessment_scored': False,
                                         'protocol_sha256': sha256(run/'protocol.json')})
        write_json(run/'development_results.json', {'completed_utc': now(), 'results': results, 'neural_fits': len(neural),
                                                   'assessment_scored': False, 'selection_sha256': sha256(run/'selection.json')})
    print(json.dumps({'complete': cfg['study'], 'neural_fits': len(neural), 'assessment_scored': False}), flush=True)


if __name__ == '__main__':
    main()
