import numpy as np
import pandas as pd

from scripts.report_pooled_state_comparison import compare_by_question


def test_question_bootstrap_averages_seed_scores_and_does_not_treat_seeds_as_questions():
    rows = []
    for q in range(20):
        for model, offset in [('reference', 0.), ('candidate', .02*q/20)]:
            for seed, shift in [(1, -.003), (2, .003)]:
                rows.append({'split': 'assessment', 'qid': f'q{q:02}', 'model': model, 'seed': seed,
                             'error': .1+offset+shift, 'nll': .3+offset+shift,
                             'brier_two_class': .12+offset+shift})
    frame = pd.DataFrame(rows)
    ordinary = compare_by_question(frame, [('candidate', 'reference')])
    duplicated = compare_by_question(pd.concat([frame, frame]), [('candidate', 'reference')])
    shuffled = compare_by_question(frame.sample(frac=1, random_state=3), [('candidate', 'reference')])
    pd.testing.assert_frame_equal(ordinary, duplicated)
    pd.testing.assert_frame_equal(ordinary, shuffled)
    np.testing.assert_allclose(ordinary.difference, np.mean(.02*np.arange(20)/20))
    assert (ordinary.questions == 20).all()
    assert (ordinary.lower < ordinary.difference).all() and (ordinary.upper > ordinary.difference).all()
