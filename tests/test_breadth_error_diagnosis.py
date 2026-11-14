"""Check the probability interpretation used in the retrospective diagnosis."""
from itertools import product

import numpy as np

from scripts.diagnose_breadth_probabilities import majority_probability, probability_metrics
from scripts.diagnose_breadth_errors import stated_option


def test_majority_probability_matches_all_independent_vote_sequences():
    for p in [0., 1e-12, .02, .2, .5, .8, .98, 1-1e-8, 1-1e-12, 1.]:
        exact = sum(p**sum(votes)*(1-p)**(5-sum(votes))
                    for votes in product([0, 1], repeat=5) if sum(votes) >= 3)
        np.testing.assert_allclose(majority_probability(p), exact, atol=1e-14)
        assert 0 <= majority_probability(p) <= 1


def test_perfect_source_distribution_has_zero_excess_expected_loss():
    q = np.array([.01, .3, .5, .8, .99])
    lp = np.log(np.stack([1-q, q], -1))
    result = probability_metrics(lp, np.array([0, 1, 0, 0, 1]), q)
    assert abs(result['excess_expected_error']) < 1e-14
    assert abs(result['excess_expected_nll']) < 1e-14
    assert result['probability_mean_squared_error'] < 1e-28
    assert result['observed_error'] != result['expected_error']


def test_stance_parser_requires_explicit_unambiguous_self_endorsement():
    assert stated_option('I favor option A because option B is incorrect.') == 1
    assert stated_option('I strongly support option B.') == -1
    assert stated_option('Option A is mentioned, but I cannot decide.') == 0
    assert stated_option('I favor a solution with two parts.') == 0
    assert stated_option('I favor option A. I prefer option B.') == 0
