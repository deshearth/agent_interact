"""Fixed persistence and training-mean references, declared before outer scoring."""
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
from scipy.special import rel_entr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import distribution_prediction_study as study


def prepare():
    cfg = study.config(); root = Path(cfg['output']); path = root/'reference_protocol.json'
    if path.exists():
        p = study.read(path); assert p['code_sha256'] == digest(__file__)
        return
    assert not (root/'results.json').exists()
    json_write(path, dict(created_utc=study.now(), code_sha256=digest(__file__),
        main_protocol_sha256=digest(root/'protocol.json'),
        references={'persistence': 'Predict p(t+1)=p(t), no fitting or clipping.',
                    'training_mean': 'Predict the mean p(t+1) across training320 only, one constant shared by all states.'},
        contrasts=[['graph_probability', 'persistence'], ['ordered_arguments_probability', 'persistence'],
                   ['ising_probability', 'persistence']],
        metrics=['kl', 'mae'], selection='Fixed descriptive controls; never used to select predictors.',
        source_calls=0, assessment_loaded=False))


def evaluate():
    cfg = study.config(); root = Path(cfg['output']); p = study.read(root/'reference_protocol.json')
    assert p['code_sha256'] == digest(__file__) and p['main_protocol_sha256'] == digest(root/'protocol.json')
    results = study.read(root/'results.json'); frozen = study.read(root/'models_frozen.json')
    assert p['created_utc'] < frozen['frozen_utc'] < results['completed_utc']
    protocol = study.validate(cfg); d = dict(np.load(root/'data/probabilities.npz', allow_pickle=False))
    train = np.isin(d['qids'], protocol['fold']['train_questions']); mean = float(d['target'][train].mean())
    rows = []; summary = {}
    for name, predicted in [('persistence', d['current']), ('training_mean', np.full_like(d['target'], mean))]:
        truth = np.stack([1-d['target'], d['target']], -1); probability = np.stack([1-predicted, predicted], -1)
        arrays = dict(kl=rel_entr(truth, probability).sum(-1), mae=np.abs(d['target']-predicted))
        assert all(np.isfinite(a).all() for a in arrays.values())
        summary[name] = {}
        for split in ('train', 'validation'):
            for question in protocol['fold'][split+'_questions']:
                take = d['qids'] == question
                rows.append(dict(model=name, split=split, qid=question, **{k: float(v[take].mean()) for k, v in arrays.items()}))
            summary[name][split] = {m: float(np.mean([r[m] for r in rows if r['model'] == name and r['split'] == split])) for m in arrays}
    primary_rows = study.read(root/'by_question.json')
    lookup = {(r['model'], r['qid']): r for r in primary_rows+rows if r['split'] == 'validation'}
    questions = sorted(protocol['fold']['validation_questions']); rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    draws = rng.integers(len(questions), size=(cfg['evaluation']['bootstrap_replicates'], len(questions)))
    contrasts = []
    for left, right in p['contrasts']:
        for metric in p['metrics']:
            difference = np.array([lookup[left, q][metric]-lookup[right, q][metric] for q in questions])
            contrasts.append(dict(left=left, right=right, metric=metric, difference=float(difference.mean()),
                                  ci95=np.quantile(difference[draws].mean(1), [.025, .975]).tolist()))
    outcome = dict(completed_utc=study.now(), protocol_sha256=digest(root/'reference_protocol.json'),
        main_results_sha256=digest(root/'results.json'), training_mean=mean, summary=summary,
        by_question=rows, paired_differences=contrasts, assessment_loaded=False)
    json_write(root/'reference_results.json', outcome)
    return outcome


if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('action', choices=['prepare', 'evaluate'])
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    else: evaluate()
