"""Frozen all-case mode evaluation; no model selection on the outer80."""
from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import modal_neural_study as core
from scripts import modal_question_arguments as paired
from scripts import modal_causal_history as history
from scripts import modal_token_pooling as tokens
from scripts import modal_token_focus as focus
from scripts import modal_population_screen as population

ROOT = Path('runs/modal_accuracy_investigation_320_v1/evaluation_v1')
FT = ROOT.parent/'encoder_finetune_screen_v1'
CAPACITY = ROOT.parent/'capacity_diagnostic_v1'
METRICS = ('accuracy', 'modal_nll', 'brier_two_class')
read, now = core.read, core.now


def metrics(logp, source, qids):
    lp = np.asarray(logp, dtype=float); source = np.asarray(source, dtype=float)
    assert lp.shape == (*source.shape, 2) and np.isfinite(lp).all()
    np.testing.assert_allclose(np.exp(lp).sum(-1), 1., atol=2e-7)
    target = source > .5; pred = lp[..., 1] > lp[..., 0]
    values = dict(accuracy=(pred == target).astype(float),
        modal_nll=-np.where(target, lp[..., 1], lp[..., 0]),
        brier_two_class=2*(np.exp(lp[..., 1])-target)**2)
    rows = [dict(qid=str(q), updates=int((qids == q).sum()*source.shape[1]),
                 **{k: float(v[qids == q].mean()) for k, v in values.items()}) for q in sorted(set(qids))]
    return {k: float(np.mean([r[k] for r in rows])) for k in METRICS}, rows


def comparisons():
    result = []
    for family in core.config()['models']:
        result.append((family+'_mode', family+'_soft'))
    for objective in ('soft', 'mode'):
        result += [('nano_initial_choice_'+objective, 'nano_current_choice_'+objective),
            ('rich_choice_'+objective, 'nano_initial_choice_'+objective),
            ('rich_arguments_'+objective, 'rich_choice_'+objective),
            ('ising_initial_'+objective, 'ising_current_'+objective)]
    result += [('ising_'+state+'_mode', 'ising_'+state+'_soft') for state in ('current', 'initial')]
    result += [('tree_'+variant, 'tree_current') for variant in core.screen.VARIANTS if variant != 'current']
    result += [('tree_semantic8', 'tree_lexical'), ('tree_semantic32', 'tree_lexical'), ('tree_semantic32', 'tree_semantic8')]
    for representation in ('pca', 'full'):
        result.append(('paired_'+representation+'_arguments_mode', 'paired_'+representation+'_choice_mode'))
    for text in ('choice', 'arguments'):
        result += [('paired_full_'+text+'_mode', 'paired_pca_'+text+'_mode'),
            ('paired_pca_'+text+'_mode', 'rich_'+text+'_mode'),
            ('paired_full_'+text+'_mode', 'rich_'+text+'_mode')]
    names = [n+'_'+c for n in core.config()['models'] for c in core.config()['conditions']]
    names += [n+'_mode' for n in paired.config()['models']] + ['tree_'+v for v in core.screen.VARIANTS]
    for text in ('choice', 'arguments'):
        result.append((f'history_{text}_observed_mode', f'history_{text}_round_mode'))
    for state in ('round', 'observed'):
        result += [(f'history_arguments_{state}_mode', f'history_choice_{state}_mode'),
                   (f'history_choice_{state}_mode', 'nano_current_choice_mode')]
    names += [n+'_mode' for n in history.config()['models']]
    for pool in ('mean', 'attention'):
        result.append((f'tokens_arguments_{pool}_mode', f'tokens_choice_{pool}_mode'))
    for text in ('choice', 'arguments'):
        result += [(f'tokens_{text}_attention_mode', f'tokens_{text}_mean_mode'),
                   (f'tokens_{text}_attention_mode', f'paired_pca_{text}_mode')]
    names += [n+'_mode' for n in tokens.config()['models']]
    for text in ('choice', 'arguments'):
        result += [(f'focus_{text}_all_mode', f'tokens_{text}_attention_mode'),
                   (f'focus_{text}_message_mode', f'focus_{text}_all_mode')]
    for region in ('all', 'message'):
        result.append((f'focus_arguments_{region}_mode', f'focus_choice_{region}_mode'))
    names += [n+'_mode' for n in focus.config()['models']]
    result += [(n, 'reference_ordered_choice_probability') for n in names]
    return result


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    p = core.validate(core.config())
    protocol = dict(fold=p['fold'], inner=p['inner'], target='A iff source next single-response p(A)>0.5; all updates retained.',
        code_files={str(f): digest(f) for f in [Path(__file__), Path('tests/test_evaluate_modal_accuracy.py')]},
        comparisons=comparisons(), bootstrap_seed=92741, bootstrap_replicates=5000,
        policy='Mean individual-seed equal-question metrics. No ensembling or best-seed selection. Outer80 are reused exploratory validation. '
            'All parent model freezes and encoder/capacity pilot decisions must precede scoring. Promoted candidates require their own frozen refits before inclusion.',
        assessment_loaded=False, source_calls=0)
    # JSON converts tuples to lists.
    protocol['comparisons'] = [list(p) for p in protocol['comparisons']]
    if (ROOT/'protocol.json').exists(): assert read(ROOT/'protocol.json') == protocol
    else: json_write(ROOT/'protocol.json', protocol)
    return protocol


def freeze():
    p = prepare(); files = []
    core.dist.validate(core.dist.config()); core.screen.validate()
    for module in (core, paired, history, tokens, focus):
        cfg = module.config(); module.validate(cfg); root = Path(cfg['output'])
        frozen = read(root/'models_frozen.json')
        for path, sha in frozen['artifacts'].items(): assert digest(path) == sha, path
        files.append(root/'models_frozen.json')
    ft = read(FT/'results.json')
    assert not ft['promotion_gate_passed'], 'Promoted encoder must be refitted and integrated before outer scoring.'
    assert not ft['outer_validation_scored']
    capacity = read(CAPACITY/'results.json')
    assert not capacity['promotion_gate_passed'] and not capacity['outer_validation_scored']
    population_result = read(population.ROOT/'results.json')
    assert not population_result['promotion_gate_passed'] and not population_result['outer_validation_scored']
    files += [FT/'results.json', FT/'protocol.json', CAPACITY/'results.json', CAPACITY/'protocol.json', Path(core.dist.config()['output'])/'results.json']
    files += [population.ROOT/'results.json', population.ROOT/'protocol.json']
    diagnostic_audit = ROOT.parent/'diagnostics_v1/completion_audit.json'
    assert read(diagnostic_audit)['passed']; files.append(diagnostic_audit)
    if (ROOT/'model_snapshot.json').exists():
        snapshot = read(ROOT/'model_snapshot.json')
        assert snapshot['artifacts'] == {str(f): digest(f) for f in files}
    else:
        snapshot = dict(frozen_utc=now(), artifacts={str(f): digest(f) for f in files}, protocol_sha256=digest(ROOT/'protocol.json'))
        json_write(ROOT/'model_snapshot.json', snapshot)
    return p


def predict(model, data, ids, batch_size=128):
    model.eval(); arrays = []
    with torch.inference_mode():
        for begin in range(0, len(ids), batch_size):
            z = core.score_model(model, data.batch(ids[begin:begin+batch_size], 'mode'))
            arrays.append(torch.stack([-F.softplus(z), -F.softplus(-z)], -1).cpu().numpy())
    return np.concatenate(arrays)


def aggregate(records, fold, pairs, replicates, seed):
    models = sorted(set(r['model'] for r in records)); summary, by_question = {}, []
    for model in models:
        summary[model] = {}
        for split in ('train', 'validation'):
            group = [r for r in records if r['model'] == model and r['split'] == split]
            seeds = sorted(set(r['seed'] for r in group))
            assert len(group) == len(seeds)*len(fold[split+'_questions'])
            lookup = {(r['seed'], r['qid']): r for r in group}
            rows = [dict(model=model, split=split, qid=q, **{k: float(np.mean([lookup[s, q][k] for s in seeds])) for k in METRICS})
                    for q in fold[split+'_questions']]
            by_question += rows
            summary[model][split] = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
            summary[model][split]['seeds'] = {str(s): {k: float(np.mean([r[k] for r in group if r['seed'] == s])) for k in METRICS} for s in seeds}
    qs = sorted(fold['validation_questions']); lookup = {(r['model'], r['qid']): r for r in by_question if r['split'] == 'validation'}
    draws = np.random.default_rng(seed).integers(len(qs), size=(replicates, len(qs)))
    contrasts = []
    for left, right in pairs:
        for metric in METRICS:
            d = np.array([lookup[left, q][metric]-lookup[right, q][metric] for q in qs])
            contrasts.append(dict(left=left, right=right, metric=metric, difference=float(d.mean()),
                                 ci95=np.quantile(d[draws].mean(1), [.025, .975]).tolist()))
    return summary, by_question, contrasts


def run():
    p = freeze(); torch.set_num_threads(2); records, files, metadata = [], [], {}
    if (ROOT/'results.json').exists(): return
    (ROOT/'predictions').mkdir(exist_ok=True)
    def save(name, seed, split, data, ids, lp, checkpoint, family):
        _, rows = metrics(lp, data.source[ids], data.qids[ids])
        records.extend(dict(model=name, seed=seed, split=split, **r) for r in rows)
        path = ROOT/'predictions'/f'{name}-s{seed}-{split}.npz'
        np.savez_compressed(path, logp=lp, target_probability=data.source[ids], qids=data.qids[ids], pairs=data.common['pairs'][ids])
        files.append(dict(model=name, seed=seed, split=split, path=str(path), sha256=digest(path),
                          checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), family=family))
    for module, family in [(core, 'decision'), (paired, 'paired'), (history, 'history'), (tokens, 'tokens'), (focus, 'focus')]:
        cfg = module.config(); root = Path(cfg['output']); data = module.Data('full')
        frozen = read(root/'models_frozen.json'); protocol = read(root/'protocol.json')
        for path in frozen['refits']:
            folder = Path(path); record = read(folder/'manifest.json'); width, condition, seed = record['width'], record['condition'], record['seed']
            name = width+'_'+condition; factory = getattr(module, 'make_model', core.make_model)
            model = factory(cfg, width)
            if family in ('tokens', 'focus'): data.text_kind = cfg['models'][width]['text']
            model.load_state_dict(torch.load(folder/'best.pt', map_location='cpu', weights_only=False)['model'])
            metadata[name] = dict(parameters=protocol['parameters'][width], trainable_encoder_parameters=0, family=family, training_target=condition)
            for split in ('train', 'validation'):
                ids = data.ids(p['fold'][split+'_questions'])
                lp = predict(model, data, ids, cfg['training']['evaluation_batch_size'])
                save(name, seed, split, data, ids, lp, folder/'best.pt', family)
            print({'evaluated': name, 'seed': seed}, flush=True)
        if family == 'decision':
            for folder in sorted((root/'baselines').iterdir()):
                name = folder.name
                if name.startswith('ising'):
                    fitted = read(folder/'model.json'); checkpoint = folder/'model.json'
                    metadata[name] = dict(parameters=len(fitted['weights']), family='ising', training_target=fitted['condition'])
                else:
                    checkpoint = folder/'model.pkl'
                    with checkpoint.open('rb') as stream: model = pickle.load(stream)
                    features = np.load(core.screen.ROOT/'features/full'/f'{name[5:]}.npy', mmap_mode='r')
                    metadata[name] = dict(parameters=None, family='tree', tree_nodes=read(folder/'complete.json')['tree_nodes'], training_target='mode')
                for split in ('train', 'validation'):
                    ids = data.ids(p['fold'][split+'_questions'])
                    if name.startswith('ising'):
                        z = core.ising_design(data, ids, fitted['initial'])@np.asarray(fitted['weights'])
                        lp = np.stack([-np.logaddexp(0, z), -np.logaddexp(0, -z)], -1)
                    else:
                        row_ids = (ids[:, None]*32+np.arange(32)).ravel()
                        with threadpool_limits(limits=2): pr = model.predict_proba(features[row_ids]).reshape(len(ids), 32, 2)
                        lp = np.log(np.clip(pr, 1e-15, 1))
                    save(name, -1, split, data, ids, lp, checkpoint, metadata[name]['family'])
        del data
    previous = read(Path(core.dist.config()['output'])/'results.json')
    old_protocol = read(Path(core.dist.config()['output'])/'protocol.json')
    for item in previous['prediction_files']:
        assert digest(item['path']) == item['sha256']; d = dict(np.load(item['path']))
        name = 'reference_'+item['model']; _, rows = metrics(d['logp'], d['target_probability'], d['qids'])
        records.extend(dict(model=name, seed=item['seed'], split=item['split'], **r) for r in rows)
        files.append(dict(item, model=name, checkpoint_sha256=digest(item['checkpoint']), family='reference'))
        count = dict(ising_opinion=19, ising_probability=22, **old_protocol['parameters'])[item['model']]
        metadata[name] = dict(parameters=count, family='reference', training_target='soft')
    assert len(files) == 204 and len(records) == 40800
    summary, by_question, contrasts = aggregate(records, p['fold'], p['comparisons'], p['bootstrap_replicates'], p['bootstrap_seed'])
    json_write(ROOT/'seed_by_question.json', records); json_write(ROOT/'by_question.json', by_question)
    json_write(ROOT/'results.json', dict(completed_utc=now(), summary=summary, models=metadata, paired_differences=contrasts, prediction_files=files,
        model_snapshot_sha256=digest(ROOT/'model_snapshot.json'), assessment_loaded=False, source_calls=0,
        target_reached=any(r['validation']['accuracy'] >= .95 for r in summary.values()), metric='All-case source-mode accuracy, mean individual seeds.'))
    print({n: round(100*r['validation']['accuracy'], 4) for n, r in summary.items()}, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'run']); args = parser.parse_args()
    if args.action == 'prepare': prepare()
    else: run()
