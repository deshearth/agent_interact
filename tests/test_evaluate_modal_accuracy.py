import numpy as np
from scripts import evaluate_modal_accuracy as study


def test_mode_labels_are_probability_thresholds_and_ties_are_B():
    qids = np.array(['a', 'a', 'b'])
    source = np.array([[.9, .49], [.5, .2], [.51, .99]])
    prob = np.array([[.51, .1], [.5, .8], [.49, .8]])
    lp = np.log(np.stack([1-prob, prob], -1))
    result, rows = study.metrics(lp, source, qids)
    assert [r['accuracy'] for r in rows] == [.75, .5]
    assert result['accuracy'] == .625  # Equal question, not unequal transition weighting.
    assert sum(r['updates'] for r in rows) == 6


def test_seed_aggregation_is_not_an_ensemble_and_bootstrap_is_paired():
    records = []
    for model, values in [('left', [.8, .6]), ('right', [.7, .5])]:
        for split in ('train', 'validation'):
            for seed in (142, 143):
                for q, value in zip(('a', 'b'), values):
                    records.append(dict(model=model, seed=seed, split=split, qid=q,
                        accuracy=value, modal_nll=1-value, brier_two_class=2*(1-value)))
    result, rows, contrasts = study.aggregate(records, dict(train_questions=['a', 'b'], validation_questions=['a', 'b']), [('left', 'right')], 100, 42)
    assert np.isclose(result['left']['validation']['accuracy'], .7)
    assert len(rows) == 8
    np.testing.assert_allclose(contrasts[0]['ci95'], [.1, .1])
