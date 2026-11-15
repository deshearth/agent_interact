import numpy as np
from scripts.modal_population_screen import population_features


def test_population_features_never_look_after_current_round():
    rng = np.random.default_rng(42); s = rng.choice([-1, 1], size=(2, 9, 32)); p = rng.random(s.shape); pairs = np.array([[0, 0], [1, 3]])
    before = population_features(s, p, pairs)
    p[0, 1:] = 1-p[0, 1:]; p[1, 4:] = 1-p[1, 4:]
    s[0, 1:] *= -1; s[1, 4:] *= -1
    np.testing.assert_array_equal(population_features(s, p, pairs), before)
    np.testing.assert_allclose(before[0, 32:64], p[0, 0], atol=3e-8)
