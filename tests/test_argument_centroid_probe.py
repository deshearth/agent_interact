"""Check that centroid interventions cannot fit held-out text or outcomes."""
from copy import deepcopy

import numpy as np
import torch

from scripts.argument_mechanisms import SplitSnapshots
from scripts.probe_argument_centroids import CentroidSnapshots, fit_centroids
from test_argument_response import fixture_cache


def fixture():
    arrays, index, vectors, donors = fixture_cache()
    index = [{**r, "diagnostic_role": "train" if e < 2 else "familiar"} for e, r in enumerate(index)]
    return arrays, index, vectors, donors


def test_centroids_ignore_held_out_messages_and_all_response_targets():
    a, index, vectors, _ = fixture()
    qids, q, qs = fit_centroids(a, index, vectors)
    changed = deepcopy(a)
    changed["text_ids"][2:] = 1
    changed["spins"] *= -1
    changed["context"][:] = 1e9
    ids2, q2, qs2 = fit_centroids(changed, index, vectors)
    assert qids == ids2
    np.testing.assert_array_equal(q, q2)
    np.testing.assert_array_equal(qs, qs2)


def test_centroid_probe_replaces_every_real_vector_and_preserves_other_inputs():
    a, index, vectors, donors = fixture()
    qids, q, qs = fit_centroids(a, index, vectors)
    base = SplitSnapshots(a, index, vectors, "familiar", "full_arguments", donors)
    for variant in ("question_mean", "question_stance_mean"):
        ds = CentroidSnapshots(a, index, vectors, donors, qids, q, qs, variant)
        assert ds.pairs == base.pairs
        for item in range(len(ds)):
            first, second = base[item], ds[item]
            for key, value in first.items():
                if key == "text":
                    continue
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, second[key], rtol=0, atol=0)
                else:
                    assert value == second[key]
            real = second["signs"] != 0
            assert ((first["text"]-second["text"]).abs().sum(-1)[real] > 0).all()
            torch.testing.assert_close(second["text"][real].norm(dim=-1), torch.ones(real.sum()), rtol=0, atol=2e-7)
            assert not second["text"][~real].any()
