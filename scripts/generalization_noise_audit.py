"""Read-only, full-cohort check of saved prompts and source sampling uncertainty."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts.local_conversation_pilot import deliver, formatter, opinion_prompt, aggregate

ROOT = Path('runs/generalization_investigation_320_v1')
ARCHIVE = ROOT/'source_probabilities.json.gz'
ARCHIVE_SHA = '0120101b0441572756508f73f392bae5744bdaf3544d8dad0edf234c1a696892'
OUT = ROOT/'noise_audit_v1'

def majority(p):
    return p**3*(10-15*p+6*p*p)

def interval(values, draws):
    return np.quantile(np.asarray(values)[draws].mean(1), [.025, .975]).tolist()

def main():
    assert digest(ARCHIVE) == ARCHIVE_SHA
    raw = json.load(gzip.open(ARCHIVE, 'rt'))
    assert raw['export_code_sha256'] == digest('scripts/export_generalization_probabilities.py')
    index_path = Path('runs/question_breadth_320_v1/episode_index.json')
    assert raw['index_sha256'] == digest(index_path)
    index = json.loads(index_path.read_text()); assert Counter(x['split'] for x in index) == {'train': 320, 'validation': 80}
    common_path = Path('runs/question_breadth_320_train_v1/predictors/development.npz')
    common = dict(np.load(common_path, allow_pickle=False))
    episode_position = {str(q): e for e, q in enumerate(common['episode_qids'])}
    lookup = {(r['qid'], r['round'], r['agent']): r for r in raw['rows']}
    assert len(lookup) == len(raw['rows']) == 400*9*32
    shape = common['spins'].shape
    p, mass = np.full(shape, np.nan), np.full(shape, np.nan)
    bare = np.zeros(shape, dtype=np.int16); retries = np.zeros(shape, dtype=np.int16)
    fmt = formatter(); prompt_count = 0
    for number, item in enumerate(index):
        path = Path(item['path']); assert digest(path) == item['sha256'] == raw['episode_artifacts'][str(path)]
        ep = json.loads(path.read_text()); e = episode_position[item['qid']]
        np.testing.assert_array_equal(ep['J'], common['J'][e])
        np.testing.assert_array_equal(ep['spins_history'], common['spins'][e])
        np.testing.assert_array_equal(ep['spins_raw_history'], common['raw_votes'][e])
        assert ep['meta']['name'] == path.stem
        for t in range(9):
            agree, disagree = [[] for _ in range(32)], [[] for _ in range(32)]
            if t:
                senders = {}
                for edge, text in ep['messages_history'][t-1].items():
                    sender, receiver = map(int, edge.split(','))
                    assert ep['J'][sender][receiver] != 0
                    assert sender not in senders or senders[sender] == text
                    senders[sender] = text
                agree, disagree, edges = deliver(ep['J'], senders, ep['meta']['name'], t)
                assert edges == ep['messages_history'][t-1]
            for a in range(32):
                row = lookup[item['qid'], t, a]
                assert row['split'] == item['split'] and row['origin'] == item['origin']
                np.testing.assert_array_equal(row['raw_votes'], common['raw_votes'][e, t, a])
                assert aggregate(row['raw_votes']) == common['spins'][e, t, a]
                prompt = opinion_prompt(fmt, ep['personas'][a], {'question': ep['statement'], 'choices': ep['choices']}, agree[a], disagree[a])
                assert hashlib.sha256(prompt.encode()).hexdigest() == row['prompt_sha256'], (item['qid'], t, a)
                prompt_count += 1
                aa, bb = (np.array(row[k], dtype=np.float64) for k in ('first_literal_A_probabilities', 'first_literal_B_probabilities'))
                assert aa.shape == bb.shape == (5,) and np.ptp(aa) == np.ptp(bb) == 0
                assert aa[0]+bb[0] > 0
                p[e, t, a] = aa[0]/(aa[0]+bb[0]); mass[e, t, a] = max(0, 1-aa[0]-bb[0])
                bare[e, t, a] = sum(row['initial_bare']); retries[e, t, a] = sum(x > 0 for x in row['parse_attempts'])
        if (number+1) % 100 == 0: print(json.dumps({'verified_episodes': number+1, 'prompt_hashes': prompt_count}), flush=True)
    assert np.isfinite(p).all() and np.isin(common['spins'], [-1, 1]).all()
    e, t = common['pairs'].T
    q = majority(p[e, t+1]); yy = (common['spins'][e, t+1] == 1).astype(float)
    pair_qids = common['episode_qids'][e]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/'probabilities.npz', single=p, majority=q, other_mass=mass,
                        initial_bare_count=bare, retry_count=retries, pairs=common['pairs'], qids=pair_qids)
    per_question = []
    for item in index:
        keep = pair_qids == item['qid']; qq, y = q[keep], yy[keep]
        eps = mass[e[keep], t[keep]+1]
        per_question.append({'qid': item['qid'], 'split': item['split'], 'origin': item['origin'],
            'reference_expected_accuracy': float(np.maximum(qq, 1-qq).mean()),
            'reference_observed_accuracy': float(((qq > .5) == y).mean()),
            'other_mass_mean': float(eps.mean()), 'other_mass_max': float(eps.max()),
            'nonbare_initial_responses': int((5-bare[e[keep], t[keep]+1]).sum()),
            'raw_failed_votes': int((common['raw_votes'][e[keep], t[keep]+1] == 0).sum()),
            'vote_A_bias': float(((common['raw_votes'][e[keep], t[keep]+1] == 1).mean(-1)-p[e[keep], t[keep]+1]).mean()),
            'majority_A_bias': float((y-qq).mean()),
            # TV perturbation for five independent first-token draws only;
            # this does not prove equivalence of first tokens and parsed text.
            'first_token_coupling_bound': float((1-(1-eps)**5).mean())})
    summaries = {}; rng = np.random.default_rng(720184)
    for split, origin in [('train', None), ('validation', None), ('validation', 'retained'), ('validation', 'new')]:
        rows = [r for r in per_question if r['split'] == split and (origin is None or r['origin'] == origin)]
        draws = rng.integers(len(rows), size=(10000, len(rows)))
        result = {'questions': len(rows), 'agent_updates': len(rows)*256}
        for key in ('reference_expected_accuracy', 'reference_observed_accuracy', 'other_mass_mean', 'vote_A_bias', 'majority_A_bias', 'first_token_coupling_bound'):
            values = np.array([r[key] for r in rows]); result[key] = float(values.mean()); result[key+'_ci95'] = interval(values, draws)
        result['other_mass_max'] = max(r['other_mass_max'] for r in rows)
        result['initial_nonbare_responses'] = sum(r['nonbare_initial_responses'] for r in rows)
        result['raw_failed_votes'] = sum(r['raw_failed_votes'] for r in rows)
        summaries[split+('_'+origin if origin else '')] = result
    calibration = []
    for split in ('train', 'validation'):
        ids = [x['qid'] for x in index if x['split'] == split]; take = np.isin(pair_qids, ids)
        pp = p[e[take], t[take]+1].ravel(); qq = q[take].ravel()
        votes = (common['raw_votes'][e[take], t[take]+1] == 1).sum(-1).ravel(); y = yy[take].ravel()
        for target, prob, actual in [('single', pp, votes/5), ('majority', qq, y)]:
            for low in np.arange(0, 1, .1):
                k = (prob >= low)&(prob < low+.10000001)
                if k.any(): calibration.append({'split': split, 'target': target, 'lower': float(low), 'n': int(k.sum()),
                    'predicted': float(prob[k].mean()), 'observed': float(actual[k].mean())})
    model_rows = []
    manifests = json.loads(Path('runs/compact_arguments_320_v1/results/predictions.json').read_text())
    model_files = [(x['model'], x['seed'], x['split'], x['path']) for x in manifests if x['model'] in ('I0', 'opinion_nano')]
    state = json.loads((ROOT/'state_models_v1/results.json').read_text())
    model_files += [(x['model'], -1, x['split'], x['path']) for x in state['prediction_files']]
    for name, seed, split, path in model_files:
        d = np.load(path, allow_pickle=False)
        take = np.isin(pair_qids, [i['qid'] for i in index if i['split'] == split])
        target_qids = np.repeat(pair_qids[take], 32); target_q = q[take].ravel(); target_y = yy[take].ravel()
        if 'probability' in d:
            probabilities = d['probability']; qids = d['qids']; target = d['target']
        else:
            probabilities = np.exp(d['logp'][..., 1]).ravel(); qids = np.repeat(d['qids'], 32); target = d['target'].ravel()
        np.testing.assert_array_equal(qids, target_qids); np.testing.assert_array_equal(target, target_y)
        expected = np.where(probabilities > .5, target_q, 1-target_q)
        for question in sorted(set(qids)):
            keep = qids == question
            model_rows.append({'model': name, 'seed': seed, 'split': split, 'qid': str(question),
                'expected_accuracy': float(expected[keep].mean()),
                'observed_accuracy': float(((probabilities[keep] > .5) == target_y[keep]).mean()),
                'reference_expected_accuracy': float(np.maximum(target_q[keep], 1-target_q[keep]).mean())})
    model_summary = {}
    for model in sorted({r['model'] for r in model_rows}):
        model_summary[model] = {}
        for split in ('train', 'validation'):
            records = [r for r in model_rows if r['model'] == model and r['split'] == split]
            model_summary[model][split] = {m: float(np.mean([r[m] for r in records])) for m in
                ('expected_accuracy', 'observed_accuracy', 'reference_expected_accuracy')}
    json_write(OUT/'by_question.json', per_question); json_write(OUT/'model_by_question.json', model_rows)
    json_write(OUT/'calibration.json', calibration)
    json_write(OUT/'audit.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': digest(__file__),
        'archive_sha256': ARCHIVE_SHA, 'index_sha256': digest(index_path), 'common_sha256': digest(common_path),
        'probabilities_sha256': digest(OUT/'probabilities.npz'), 'verified_episodes': 400, 'verified_prompts': prompt_count,
        'verified_raw_votes': 400*9*32*5, 'source_quality': raw['quality'], 'summary': summaries, 'models': model_summary,
        'assumptions': 'p=literal A/(literal A+literal B) at actual source temperature .7; q=P(at least 3 A in 5 independent Bernoulli(p) votes). '
            'This approximates parsed complete responses. Non-A/B first mass and observed formatting failures are reported; '
            'first-token coupling bounds do not bound later-token parsing anomalies. Expected oracle accuracy is an approximate '
            'Bayes reference for this sampled-majority target conditional on the full prompt, not a limit for deterministic modal choices or all tasks. '
            'Confidence intervals resample whole questions, conditional on current prompts/source probabilities.',
        'actual_prompt_order_verified': True, 'assessment_loaded': False, 'source_calls': 0})
    print(json.dumps({'noise_audit_complete': True, 'summary': summaries, 'models': model_summary}), flush=True)

if __name__ == '__main__': main()
