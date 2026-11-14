import numpy as np

from mac_agents.common import config
from scripts import argument_residual_diagnostic as study
from scripts.argument_features import load_cache


def test_centroids_are_training_only_and_residuals_reconstruct_original_vectors():
    cfg = config(study.CONFIG)
    caches = study.prepare(cfg)
    arrays, index, raw, _ = load_cache(cfg, verify=True)
    with np.load(study.ROOT/"centroids.npz", allow_pickle=False) as c:
        qids, means, signed = c["qids"].tolist(), c["question"], c["question_stance"]
        scale = float(c["residual_rms_scale"])
    transformed = caches["individual_residual"][2]
    count, square = 0, 0.
    for qi, qid in enumerate(qids):
        es = [e for e,r in enumerate(index) if r["qid"] == qid and r["replica"] in (0,1)]
        ids = arrays["text_ids"][es].reshape(-1)
        stance = arrays["stance"][es].reshape(-1)[ids != 0]
        ids = ids[ids != 0]
        x = np.asarray(raw[ids], dtype=np.float64)
        np.testing.assert_allclose(means[qi], x.mean(0), rtol=0, atol=1e-12)
        for sign in (-1, 1):
            keep = stance == sign
            center = signed[qi, int(sign == 1)]
            np.testing.assert_allclose(center, x[keep].mean(0), rtol=0, atol=1e-12)
            r = np.asarray(transformed[ids[keep]], dtype=np.float64)
            np.testing.assert_allclose(r.mean(0), 0., rtol=0, atol=1e-7)
            np.testing.assert_allclose(r*scale+center, x[keep], rtol=0, atol=1e-7)
            square += np.square(r).sum(); count += len(r)
    assert abs(square/count-1)<1e-6


def test_centroids_remove_individual_variation_and_no_unseen_centroids_exist():
    cfg = config(study.CONFIG)
    caches = study.prepare(cfg)
    for kind in ("question_mean", "question_stance_mean"):
        a, index, vec, _ = caches[kind]
        np.testing.assert_allclose(np.linalg.norm(vec[1:], axis=-1), 1., rtol=0, atol=2e-6)
        assert not vec[0].any()
        seen = {}
        for e,r in enumerate(index):
            ids = a["text_ids"][e]
            if r["diagnostic_role"] == "unseen":
                assert not ids.any()
                continue
            for sign in (-1, 1):
                selected = ids[a["stance"][e] == sign]
                if len(selected):
                    assert len(np.unique(selected)) == 1
                    key = (r["qid"], sign if kind == "question_stance_mean" else 0)
                    if key in seen:
                        assert seen[key] == selected[0]
                    seen[key] = selected[0]


def test_runtime_adapter_restores_frozen_driver():
    before = study.driver.ROOT
    with study.adapted_driver():
        assert study.driver.ROOT == study.ROOT
        assert len(study.driver.cohort(config(study.CONFIG))) == 9
    assert study.driver.ROOT == before


def test_original_context_variant_preserves_the_original_model_and_inputs():
    import torch
    from mac_agents.common import seed_all
    from scripts import argument_mechanisms as previous
    from scripts import argument_residual_original_context as original_context
    cfg = config(original_context.CONFIG)
    caches = original_context.prepare(cfg)
    raw, _, _, _ = load_cache(cfg, verify=True)
    for a, _, _, _ in caches.values():
        np.testing.assert_array_equal(a["context"], raw["context"])
    seed_all(42)
    base = previous.make_model(cfg, "A2")
    seed_all(42)
    current = original_context.make_model(cfg, "pca3_question_mean")
    assert sum(p.numel() for p in current.parameters()) == 185890
    for k,v in base.state_dict().items():
        torch.testing.assert_close(v, current.state_dict()[k], rtol=0, atol=0)
