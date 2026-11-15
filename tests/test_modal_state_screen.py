import numpy as np
from scripts import modal_state_screen as study


def test_probability_history_never_reads_next_state():
    rng = np.random.default_rng(3); p = rng.uniform(.01, .99, (2, 9, 3))
    pairs = np.array([[0, 2], [1, 4]])
    graph = np.array([[[0, 1, -1], [1, 0, 0], [-1, 0, 0]]]*2)
    order = np.array([[[[1, -1], [2, -1]], [[0, -1], [-1, -1]], [[-1, -1], [0, -1]]]]*2)
    before = study.probability_features(p, pairs, graph, order)
    altered = p.copy(); altered[0, 3:] = .99; altered[1, 5:] = .01
    after = study.probability_features(altered, pairs, graph, order)
    for a, b in zip(before, after): np.testing.assert_array_equal(a, b)
    altered = p.copy(); altered[:, 0] = 1-altered[:, 0]
    changed = study.probability_features(altered, pairs, graph, order)
    np.testing.assert_array_equal(before[0], changed[0])
    assert not np.array_equal(before[1], changed[1])


def test_slots_keep_sender_zero_and_zero_only_padding():
    values = np.array([[[1., 2.], [3., 4.], [5., 6.]]])
    order = np.array([[[[0, -1], [2, -1]], [[1, -1], [-1, -1]], [[-1, -1], [0, -1]]]])
    x = study.slots(values, order)
    np.testing.assert_array_equal(x[0, 0], [1, 2, 0, 0, 5, 6, 0, 0])
    np.testing.assert_array_equal(x[0, 2], [0, 0, 0, 0, 1, 2, 0, 0])
