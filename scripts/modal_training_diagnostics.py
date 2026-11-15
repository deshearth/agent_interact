"""Training-only source-margin, round, and causal residual-memory diagnostics."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import distribution_prediction_study as study


def run():
    source = Path(study.config()['output']); results = study.read(source/'results.json')
    protocol = study.read(source/'protocol.json')
    selected = [r for r in results['prediction_files'] if r['model'] == 'ordered_choice_probability' and r['split'] == 'train']
    assert len(selected) == 3
    records, rounds = [], []
    bins = [0, .025, .05, .1, .2, .3, .4, .5000001]
    for item in selected:
        assert digest(item['path']) == item['sha256']; d = dict(np.load(item['path']))
        assert set(d['qids']) == set(protocol['fold']['train_questions'])
        y = d['target_probability'] > .5; correct = (d['logp'].argmax(-1) == y)
        margin = np.abs(d['target_probability']-.5)
        for lower, upper in zip(bins, bins[1:]):
            mask = (margin >= lower)&(margin < upper)
            records.append(dict(seed=item['seed'], lower=lower, upper=min(upper, .5), updates=int(mask.sum()),
                share=float(mask.mean()), accuracy=float(correct[mask].mean()), fraction_of_errors=float((~correct & mask).sum()/(~correct).sum())))
        for t in range(8):
            mask = d['pairs'][:, 1] == t
            rounds.append(dict(seed=item['seed'], round=t+1, accuracy=float(correct[mask].mean()), updates=int(correct[mask].size)))
    item = next(r for r in selected if r['seed'] == 142); d = dict(np.load(item['path']))
    y = d['target_probability'] > .5; z = d['logp'][..., 1].astype(float)-d['logp'][..., 0]
    p = np.clip(d['target_probability'], 1e-9, 1-1e-9)
    residual = np.clip(np.log(p)-np.log1p(-p)-z, -4, 4)
    own, pooled = np.zeros_like(residual), np.zeros_like(residual); lag0, lag1 = [], []
    for q in sorted(set(d['qids'])):
        ids = np.flatnonzero(d['qids'] == q); assert len(ids) == 8
        ids = ids[np.argsort(d['pairs'][ids, 1])]
        np.testing.assert_array_equal(d['pairs'][ids, 1], np.arange(8))
        for t in range(1, 8):
            own[ids[t]] = residual[ids[:t]].mean(0)
            pooled[ids[t]] = residual[ids[:t]].mean()
        lag0.extend(residual[ids[:-1]].ravel()); lag1.extend(residual[ids[1:]].ravel())
    memory = [dict(pooled_weight=a, own_weight=b, training_accuracy=float(((z+a*pooled+b*own > 0) == y).mean()))
              for a, b in [(0, 0), (.5, 0), (0, .5), (.5, .5), (0, 1)]]
    root = Path('runs/modal_accuracy_investigation_320_v1/diagnostics_v1'); root.mkdir(exist_ok=True)
    json_write(root/'training.json', dict(created_utc=study.now(), code_sha256=digest(__file__), source_files=selected,
        scope='Training320 only. Mean individual-seed accuracy for fixed source-margin and round groups. No outer or assessment data inspected.',
        margin_bins=records, rounds=rounds, residual_memory_seed=142, residual_lag1_correlation=float(np.corrcoef(lag0, lag1)[0, 1]),
        residual_memory=memory, memory_policy='Only past observed source probabilities and past forecasts enter the correction; never future residuals. '
            'Fixed exploratory training-only weights, not tuned on outer80. No successful model is inferred from these training diagnostics.'))
    print(memory)


if __name__ == '__main__': run()
