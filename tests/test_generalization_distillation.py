import numpy as np
from scripts.generalization_distillation import expand, teacher_targets

def test_fractional_bernoulli_equivalence():
    target = np.array([0., .1, .5, .9, 1.]); probability = np.array([.3, .8, .1, .5, .7])
    x = np.arange(10).reshape(5, 2); xx, yy, weights = expand(x, target)
    pp = np.tile(probability, 2)
    loss = -yy*np.log(pp)-(1-yy)*np.log1p(-pp)
    np.testing.assert_allclose((loss*weights).sum(), (-target*np.log(probability)-(1-target)*np.log1p(-probability)).sum())
    np.testing.assert_allclose(((pp-yy)*weights).reshape(2, 5).sum(0), probability-target)
    np.testing.assert_allclose((pp*(1-pp)*weights).reshape(2, 5).sum(0), probability*(1-probability))
    np.testing.assert_array_equal(xx[:5], xx[5:]); assert weights.sum() == 5

def test_unsafe_teacher_labels_fall_back_without_dropping_rows():
    q = np.array([.1, .2, .3, .4]); hard = np.array([1., 0., 1., 0.])
    target, reliable = teacher_targets(q, hard, np.array([0., .1, 0., 0.]), np.array([5, 5, 4, 5]), np.array([0, 0, 0, 1]), .001)
    np.testing.assert_array_equal(reliable, [True, False, False, False])
    np.testing.assert_allclose(target, [.1, 0., 1., 0.])

def test_polynomial_roundoff_is_clamped_without_accepting_invalid_probabilities():
    import pytest
    target, _ = teacher_targets(np.array([1+2e-15, -1e-16]), np.array([0, 1]), np.zeros(2), np.full(2, 5), np.zeros(2), .001)
    np.testing.assert_array_equal(target, [1, 0])
    with pytest.raises(AssertionError):
        teacher_targets(np.array([1.01]), np.array([0]), np.zeros(1), np.full(1, 5), np.zeros(1), .001)
