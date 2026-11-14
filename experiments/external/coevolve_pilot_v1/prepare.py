"""Prepare synchronous graph states with native three-category labels."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from ..common import ROOT, freeze_split, now, sha256, write_json


def read_run(path, model_vocabulary):
    agents = json.loads((path/'agents_data.json').read_text())
    manifest = json.loads((path/'agent_manifest.json').read_text())
    network = json.loads((path/'network_edges.json').read_text())
    n = network['n']; assert n == 48 and len(agents) == n
    adjacency = np.zeros((n, n), dtype=bool)
    for a, b in network['edges']:
        assert a != b
        adjacency[a, b] = adjacency[b, a] = True
    probabilities = np.zeros((11, n, 3), dtype=np.float64)
    models = []
    label_map = {1.: 0, 0.: 1, -1.: 2}  # true, false, neither
    for node in range(n):
        a = agents[str(node)]; meta = manifest[str(node)]
        assert a['node_id'] == node and meta['node_id'] == node
        assert a['model'] == meta['model'] and a['role'] == meta['role']
        assert set(np.flatnonzero(adjacency[node])) == set(meta['network']['adjacent_agents'])
        assert set(a['complete_scores']) == {str(t) for t in range(11)}
        for t in range(11):
            prob = np.asarray(a['complete_scores'][str(t)], dtype=float)
            assert np.all(np.isfinite(prob)) and np.min(prob) >= -1e-6
            assert abs(prob.sum()-1) < 2e-5
            label = label_map[a['beliefs'][str(t)]]
            assert label == int(prob.argmax()), 'Native label/probability order mismatch'
            probabilities[t, node] = prob
        models.append(model_vocabulary.index(a['model']))
    return probabilities.argmax(-1), probabilities, np.array(models), adjacency


def main():
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    data, run, report = [ROOT/cfg[k] for k in ['data_root', 'run_root', 'report_root']]
    paths = sorted((data/'raw/trajectories'/cfg['scenario']).glob('*/*'))
    paths = [p for p in paths if (p/'agents_data.json').exists()]
    assert len(paths) == 40
    qids = sorted({p.name for p in paths})
    split = freeze_split(run/'split.json', qids, cfg['split'], forced_train=cfg['split']['forced_train'],
                         strata={q: q.split('_')[0] for q in qids})
    parts = {q: k for k, ids in split['groups'].items() for q in ids}
    # Vocabulary uses public agent identity metadata, not outcomes or fitted statistics.
    vocabulary = sorted({a['model'] for p in paths for a in json.loads((p/'agent_manifest.json').read_text()).values()})
    roles = {a['model']: a['role'] for p in paths for a in json.loads((p/'agent_manifest.json').read_text()).values()}
    x, y, probability, identities, graphs, steps, questions, graph_ids, partitions = ([] for _ in range(9))
    for path in paths:
        states, probs, model_ids, graph = read_run(path, vocabulary)
        for t in range(10):
            x.append(states[t]); y.append(states[t+1]); probability.append(probs[t])
            identities.append(model_ids); graphs.append(graph); steps.append(t)
            questions.append(path.name); graph_ids.append(path.parent.name); partitions.append(parts[path.name])
    arrays = {'state': np.array(x, dtype=np.int64), 'target': np.array(y, dtype=np.int64),
              'probability_before': np.array(probability, dtype=np.float32),
              'identity': np.array(identities, dtype=np.int64), 'adjacency': np.array(graphs),
              'step': np.array(steps), 'question': np.array(questions), 'graph_id': np.array(graph_ids),
              'partition': np.array(partitions)}
    derived = data/'derived'; derived.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(derived/'pilot_graphs.npz', **arrays)
    write_json(derived/'identity_vocabulary.json', {'models': vocabulary, 'roles': roles,
               'labels': ['true', 'false', 'neither'], 'confidence_policy': 'Saved for provenance; excluded from the opinion-only predictor pilot.'})
    stats = {}
    for role in ['train', 'validation']:
        ids = arrays['partition'] == role
        own = arrays['state'][ids]; target = arrays['target'][ids]
        flip = own != target; qs = arrays['question'][ids]
        graph = arrays['adjacency'][ids]
        counts = graph @ np.eye(3)[own]
        majority = counts.argmax(-1)
        majority = np.where(graph.sum(-1) > 0, majority, own)
        stats[role] = {'questions': int(len(set(qs))), 'graph_snapshots': int(ids.sum()),
                       'agent_transitions': int(target.size), 'changes': int(flip.sum()),
                       'questions_with_change': int(len(set(qs[flip.any(-1)]))),
                       'persistence_error': float(flip.mean()),
                       'neighbor_majority_error': float((majority != target).mean()),
                       'target_counts': np.bincount(target.ravel(), minlength=3).tolist()}
    report.mkdir(parents=True, exist_ok=True)
    audit = {'created_utc': now(), 'scenario': cfg['scenario'], 'graphs': cfg['graph_folders'],
             'raw_runs': len(paths), 'distinct_statements': len(qids), 'labels': ['true', 'false', 'neither'],
             'split_counts': {k: len(v) for k, v in split['groups'].items()}, 'development': stats,
             'assessment_outcome_diagnostics': False, 'split_sha256': sha256(run/'split.json'),
             'arrays_sha256': sha256(derived/'pilot_graphs.npz'), 'vocabulary_sha256': sha256(derived/'identity_vocabulary.json'),
             'input_policy': 'Current categorical belief, model/role identity and neighbor belief/role composition. No ground truth, question ID, future states or confidence probabilities enter predictor features.',
             'scope': '40 runs on two prespecified graph realizations. Does not reproduce the full-paper graph cohort. Whole-statement split generalizes to new statements within this source population.'}
    write_json(run/'prepared.json', audit); write_json(report/'audit.json', audit)
    lines = ['# Coevolution pilot: data audit', '',
             'Matched specialist roles; one prespecified graph from each of the ER and WS families.',
             '20 statement groups are split into 12 fitting, 4 validation, and 4 reserved assessment statements.',
             'The inspected schema example `false_0` is assigned to fitting. Splits are balanced by the source true/false statement labels, which are not predictor inputs.', '',
             '| Partition | Agent updates | Opinion changes | Persistence error | Neighbor-majority error |',
             '| --- | ---: | ---: | ---: | ---: |']
    for role, s in stats.items():
        lines.append(f"| {role} | {s['agent_transitions']} | {s['changes']} | {s['persistence_error']:.2%} | {s['neighbor_majority_error']:.2%} |")
    lines += ['', 'All native labels were checked against the argmax of recorded probability distributions. Graph edges were checked against every agent’s recorded inbox neighbors.',
              'Assessment outcomes were not summarized or scored. Only development partitions contribute the table above.', '',
              'Source: https://github.com/carlomarxdk/coevolve-sim', '']
    (report/'audit.md').write_text('\n'.join(lines))
    print(json.dumps({'study': cfg['study'], 'development': stats}), flush=True)


if __name__ == '__main__':
    main()
