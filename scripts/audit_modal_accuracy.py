"""Independent arithmetic, provenance, split, selection and checkpoint replay audit."""
from datetime import datetime
from pathlib import Path
import pickle
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import evaluate_modal_accuracy as evaluation
from scripts import modal_neural_study as core
from scripts import modal_question_arguments as paired
from scripts import modal_causal_history as history
from scripts import modal_token_pooling as tokens
from scripts import modal_token_focus as focus
from scripts import modal_population_screen as population


def main():
    root = evaluation.ROOT; p = evaluation.prepare(); result = evaluation.read(root/'results.json')
    core.dist.validate(core.dist.config()); core.screen.validate()
    assert result['model_snapshot_sha256'] == digest(root/'model_snapshot.json')
    snapshot = evaluation.read(root/'model_snapshot.json'); frozen_at = datetime.fromisoformat(snapshot['frozen_utc'])
    for path, sha in snapshot['artifacts'].items(): assert digest(path) == sha, path
    train, val = set(p['fold']['train_questions']), set(p['fold']['validation_questions'])
    assert len(train) == 320 and len(val) == 80 and not train & val
    assert set(p['inner']['fit']) | set(p['inner']['validation']) == train
    assert not set(p['inner']['fit']) & set(p['inner']['validation'])
    hashes, tunes, refits = 0, 0, 0
    for module in (core, paired, history, tokens, focus):
        cfg = module.config(); module.validate(cfg); r = Path(cfg['output']); frozen = evaluation.read(r/'models_frozen.json')
        assert datetime.fromisoformat(frozen['frozen_utc']) <= frozen_at
        for path, sha in frozen['artifacts'].items(): assert digest(path) == sha, path; hashes += 1
        records = [(f.parent, evaluation.read(f)) for f in (r/'tune').glob('*/manifest.json')]
        for choice in evaluation.read(r/'selection.json')['choices']:
            group = [(path, m) for path, m in records if m['width'] == choice['width'] and m['condition'] == choice['condition']]
            assert len(group) == 2
            path, best = min(group, key=lambda v: (v[1]['validation']['error'], v[1]['validation']['nll'], v[1]['selected_step'], v[1]['candidate']))
            assert str(path) == choice['tuning_directory'] and best['selected_step'] == choice['refit_steps']
            assert all(m['train']['questions'] == 256 and m['validation']['questions'] == 64 for f, m in group)
            tunes += 2
        for path in frozen['refits']:
            m = evaluation.read(Path(path)/'manifest.json')
            assert m['train']['questions'] == 320 and m['validation'] is None and m['status'] == 'complete'
            assert m['selected_step'] == m['fixed_steps'] == m['steps']
            refits += 1
    assert (tunes, refits) == (48, 72)
    ft = evaluation.read(evaluation.FT/'results.json')
    assert datetime.fromisoformat(ft['completed_utc']) <= frozen_at and not ft['promotion_gate_passed']
    assert not ft['outer_validation_scored']
    for arm in ft['results']:
        folder = evaluation.FT/f"arm{arm['arm']}"
        for name, sha in arm['artifacts'].items(): assert digest(folder/name) == sha, name
        arm_history = evaluation.read(folder/'history.json')
        best = min(arm_history, key=lambda h: (h['validation']['error'], h['validation']['nll']))
        assert best['epoch'] == arm['selected_epoch']
    capacity = evaluation.read(evaluation.CAPACITY/'results.json')
    assert not capacity['promotion_gate_passed'] and not capacity['outer_validation_scored']
    assert datetime.fromisoformat(capacity['completed_utc']) <= frozen_at
    for record in capacity['results']:
        folder = evaluation.CAPACITY/f"{record['kind']}_{record['fitting_questions']}"
        for name, sha in record['artifacts'].items(): assert digest(folder/name) == sha, name
    saved_rows = evaluation.read(root/'seed_by_question.json')
    lookup = {(r['model'], r['seed'], r['split'], r['qid']): r for r in saved_rows}
    assert len(lookup) == len(saved_rows) == 40800
    recomputed, max_diff = {}, 0.
    common = dict(np.load(core.state.config()['common']))
    message_ids = np.load(Path(paired.config()['output'])/'data/message_ids.npy')
    for stage, questions in [('inner', p['inner']['fit']), ('full', p['fold']['train_questions'])]:
        transform = dict(np.load(tokens.ROOT/'features'/stage/'pca.npz'))
        allowed = set(np.unique(message_ids[np.isin(common['episode_qids'], questions)]))
        assert set(transform['training_text_ids']) <= allowed and 0 not in transform['training_text_ids']
        np.testing.assert_allclose(transform['axes']@transform['axes'].T, np.eye(32), atol=1e-6)
    np.testing.assert_array_equal(np.diff(np.load(tokens.ROOT/'data/offsets.npy')), np.load(Path(paired.config()['output'])/'data/token_counts.npy'))
    probabilities = np.load(Path(core.dist.config()['probability_reference'])/'probabilities.npz')['single']
    population_result = evaluation.read(population.ROOT/'results.json'); population_protocol = evaluation.read(population.ROOT/'protocol.json')
    assert population_protocol['code_sha256'] == digest(population.__file__)
    assert not population_result['promotion_gate_passed'] and not population_result['outer_validation_scored']
    assert datetime.fromisoformat(population_result['completed_utc']) <= frozen_at
    qids = common['episode_qids'][common['pairs'][:, 0]]
    inner_mask = np.isin(qids, p['inner']['validation'])
    inner_truth = (probabilities[common['pairs'][inner_mask, 0], common['pairs'][inner_mask, 1]+1] > .5).astype(float).ravel()
    for record in population_result['results']:
        folder = population.ROOT/record['variant']
        for name, sha in record['artifacts'].items(): assert digest(folder/name) == sha, name
        pr = np.load(folder/'inner_probability.npy')
        assert float(((pr > .5) == inner_truth).mean()) == record['inner_accuracy']
        nll = float((-inner_truth*np.log(pr)-(1-inner_truth)*np.log1p(-pr)).mean())
        assert abs(nll-record['inner_nll']) < 1e-12
    for item in result['prediction_files']:
        assert digest(item['path']) == item['sha256'] and digest(item['checkpoint']) == item['checkpoint_sha256']
        d = dict(np.load(item['path'])); source = probabilities[d['pairs'][:, 0], d['pairs'][:, 1]+1]
        np.testing.assert_array_equal(d['target_probability'], source)
        np.testing.assert_array_equal(d['qids'], common['episode_qids'][d['pairs'][:, 0]])
        assert set(d['qids']) == (train if item['split'] == 'train' else val)
        assert len(d['pairs']) == len(set(d['qids']))*8
        z = d['logp'][..., 1].astype(float)-d['logp'][..., 0].astype(float); target = source > .5
        # Independent stable logistic arithmetic, using logits rather than saved class probabilities.
        arrays = dict(accuracy=((z > 0) == target).astype(float), modal_nll=np.logaddexp(0, z)-target*z,
                      brier_two_class=2*(1/(1+np.exp(-z))-target)**2)
        for q in sorted(set(d['qids'])):
            values = {k: float(v[d['qids'] == q].mean()) for k, v in arrays.items()}
            key = item['model'], item['seed'], item['split'], str(q); recomputed[key] = values
            for metric in evaluation.METRICS:
                delta = abs(values[metric]-lookup[key][metric]); max_diff = max(max_diff, delta)
                assert delta < 3e-8, (key, metric, delta)
    by_question = {}
    for model, summary in result['summary'].items():
        for split in ('train', 'validation'):
            seeds = [int(s) for s in summary[split]['seeds']]
            for q in p['fold'][split+'_questions']:
                by_question[model, split, q] = {m: float(np.mean([recomputed[model, s, split, q][m] for s in seeds])) for m in evaluation.METRICS}
            for metric in evaluation.METRICS:
                value = np.mean([by_question[model, split, q][metric] for q in p['fold'][split+'_questions']])
                assert abs(value-summary[split][metric]) < 3e-8
    qs = sorted(val); draws = np.random.default_rng(p['bootstrap_seed']).integers(len(qs), size=(p['bootstrap_replicates'], len(qs)))
    for c in result['paired_differences']:
        values = np.array([by_question[c['left'], 'validation', q][c['metric']]-by_question[c['right'], 'validation', q][c['metric']] for q in qs])
        assert abs(values.mean()-c['difference']) < 6e-8
        np.testing.assert_allclose(np.quantile(values[draws].mean(1), [.025, .975]), c['ci95'], atol=6e-8, rtol=0)
    # Replay all newly fitted final models on every validation update.
    torch.set_num_threads(2); replays, replay_difference, disagreements = 0, 0., 0
    for module, family in [(core, 'decision'), (paired, 'paired'), (history, 'history'), (tokens, 'tokens'), (focus, 'focus')]:
        cfg = module.config(); data = module.Data('full'); ids = data.ids(sorted(val))
        assert set(data.qids[ids]) == val
        for item in result['prediction_files']:
            if item['split'] != 'validation' or item['family'] == 'reference': continue
            if family != 'decision' and item['family'] != family: continue
            if family == 'decision' and item['family'] not in ('decision', 'tree', 'ising'): continue
            saved = dict(np.load(item['path'])); np.testing.assert_array_equal(saved['qids'], data.qids[ids])
            if item['family'] in ('decision', 'paired', 'history', 'tokens', 'focus'):
                record = evaluation.read(Path(item['checkpoint']).parent/'manifest.json')
                factory = getattr(module, 'make_model', core.make_model)
                model = factory(cfg, record['width'])
                if family in ('tokens', 'focus'): data.text_kind = cfg['models'][record['width']]['text']
                model.load_state_dict(torch.load(item['checkpoint'], map_location='cpu', weights_only=False)['model'])
                lp = evaluation.predict(model, data, ids, cfg['training']['evaluation_batch_size'])
                # Neither target representation may affect inference.
                batch = data.batch(ids[:2], 'mode')
                with torch.inference_mode():
                    before = core.score_model(model, batch)
                    for key in ('target', 'target_soft', 'target_mode'):
                        if key in batch: batch[key] = 1-batch[key]
                    torch.testing.assert_close(before, core.score_model(model, batch), atol=0, rtol=0)
            elif item['family'] == 'ising':
                fit = evaluation.read(item['checkpoint']); z = core.ising_design(data, ids, fit['initial'])@np.asarray(fit['weights'])
                lp = np.stack([-np.logaddexp(0, z), -np.logaddexp(0, -z)], -1)
            else:
                with Path(item['checkpoint']).open('rb') as stream: model = pickle.load(stream)
                features = np.load(core.screen.ROOT/'features/full'/f"{item['model'][5:]}.npy", mmap_mode='r')
                rows = (ids[:, None]*32+np.arange(32)).ravel()
                with threadpool_limits(limits=2): probability = model.predict_proba(features[rows]).reshape(len(ids), 32, 2)
                lp = np.log(np.clip(probability, 1e-15, 1))
            delta = float(np.max(np.abs(lp-saved['logp']))); replay_difference = max(delta, replay_difference)
            disagreements += int((lp.argmax(-1) != saved['logp'].argmax(-1)).sum())
            assert delta < 1e-6
            replays += 1
        del data
    assert replays == 82 and disagreements == 0
    assert result['target_reached'] == any(v['validation']['accuracy'] >= .95 for v in result['summary'].values())
    json_write(root/'completion_audit.json', dict(passed=True, completed_utc=evaluation.now(), audit_code_sha256=digest(__file__),
        results_sha256=digest(root/'results.json'), prediction_files=len(result['prediction_files']), seed_question_rows=len(lookup),
        contrasts=len(result['paired_differences']), new_checkpoint_replays=replays, replay_logp_max_difference=replay_difference,
        replay_class_disagreements=disagreements, independent_metric_max_difference=max_diff, frozen_artifacts_verified=hashes,
        tune_fits=tunes, neural_refits=refits, assessment_loaded=False, source_calls=0))
    print(evaluation.read(root/'completion_audit.json'))


if __name__ == '__main__': main()
