import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from scripts.diagnose_ising_gap import crossfit_offsets, fit_offsets, overlap, score_metrics


def test_offsets_match_independent_scalar_optimizer():
    rng = np.random.default_rng(941)
    z = rng.normal(size=200)
    y = rng.binomial(1, expit(z+1.3))
    group = np.arange(200) % 4
    fitted, _, gradient = fit_offsets(z, y, group, 4)
    assert gradient < 1e-8
    for g in range(4):
        m = group == g
        objective = lambda b: np.sum(np.logaddexp(0, z[m]+b)-y[m]*(z[m]+b)) + b*b/2
        scalar = minimize_scalar(objective, method="bounded", bounds=(-10, 10))
        np.testing.assert_allclose(fitted[g], scalar.x, atol=1e-5)


def test_assessment_labels_do_not_enter_own_adaptation():
    rng = np.random.default_rng(942)
    z = rng.normal(size=(80, 2))
    y = rng.binomial(1, .7, size=z.shape)
    group = np.broadcast_to(np.arange(2), z.shape)
    replica = np.broadcast_to((np.arange(80) % 4)[:, None], z.shape)
    original, _ = crossfit_offsets(z, y, group, replica)
    changed = y.copy()
    changed[replica >= 2] = 1-changed[replica >= 2]
    perturbed, _ = crossfit_offsets(z, changed, group, replica)
    np.testing.assert_array_equal(original[replica >= 2], perturbed[replica >= 2])
    assert not np.array_equal(original[replica < 2], perturbed[replica < 2])


def test_all_one_class_groups_are_finite_and_shrunk():
    z = np.zeros(100)
    y = np.r_[np.zeros(50), np.ones(50)]
    group = np.repeat([0, 1], 50)
    offsets, _, _ = fit_offsets(z, y, group, 3)
    assert offsets[0] < 0 < offsets[1]
    assert offsets[2] == 0
    np.testing.assert_allclose(offsets[0], -offsets[1])
    assert np.isfinite(offsets).all()


def test_error_overlap_explains_net_difference():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    result = overlap(a, b)
    assert sum(result.values()) == 1
    assert result["shared_error"] + result["fixed_ising_error"] == a.mean()
    assert result["shared_error"] + result["introduced_error"] == b.mean()
    assert result["introduced_error"] - result["fixed_ising_error"] == b.mean()-a.mean()


def test_logit_metrics_match_binary_log_probabilities():
    z = np.array([-1000., -1., 0., 1., 1000.])
    y = np.array([0, 1, 0, 1, 1])
    metrics = score_metrics(z, y)
    lp = np.stack([-np.logaddexp(0, z), -np.logaddexp(0, -z)], axis=-1)
    np.testing.assert_allclose(metrics["nll"], -lp[np.arange(5), y].mean())
    assert metrics["error"] == (lp.argmax(-1) != y).mean()
