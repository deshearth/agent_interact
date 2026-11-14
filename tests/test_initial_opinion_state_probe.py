import numpy as np
from scipy.optimize import check_grad

from scripts.probe_initial_opinion_state import fit, initial_features, objective, standardize


def test_initial_features_ignore_future_votes():
    rng = np.random.default_rng(420)
    votes = rng.choice([-1, 1], size=(4, 9, 32, 5))
    original = initial_features(votes)
    votes[:, 1:] *= -1
    np.testing.assert_array_equal(initial_features(votes), original)
    votes[0, 0] = 1
    changed = initial_features(votes)
    assert not np.array_equal(changed[0], original[0])
    np.testing.assert_array_equal(changed[1:], original[1:])


def test_standardization_uses_only_fitting_features():
    rng = np.random.default_rng(421)
    x, other = rng.normal(size=(30, 5)), rng.normal(size=(20, 5))
    x[:, 0], other[:, 0] = 1, 1
    fit1, _, mean1, scale1 = standardize(x, other)
    fit2, _, mean2, scale2 = standardize(x, other+100)
    for a, b in [(fit1, fit2), (mean1, mean2), (scale1, scale2)]:
        np.testing.assert_array_equal(a, b)
    assert (fit1[:, 0] == 1).all()


def test_penalized_residual_gradient_and_solution():
    rng = np.random.default_rng(422)
    x = rng.normal(size=(100, 5))
    z = rng.normal(size=100)
    y = rng.binomial(1, .6, size=100)
    w = rng.normal(size=5)
    error = check_grad(lambda w: objective(w, x, z, y, .01)[0],
                       lambda w: objective(w, x, z, y, .01)[1], w)
    assert error < 1e-6
    fitted, info = fit(x, z, y, .01)
    assert info["maximum_gradient"] < 1e-6
    assert objective(fitted, x, z, y, .01)[0] <= objective(np.zeros(5), x, z, y, .01)[0]
