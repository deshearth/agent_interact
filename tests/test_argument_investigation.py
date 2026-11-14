"""Scientific input invariants for the bounded argument investigation."""
import numpy as np
import pytest
import torch

from mac_agents.common import config, seed_all
from scripts.joint_argument_encoding_windowed import windows
from scripts import joint_argument_context as study
from scripts import question_context


@pytest.mark.parametrize("question_size,message_size", [(0, 25), (30, 80), (538, 288), (1024, 3)])
def test_windows_retain_every_token_and_full_message(question_size, message_size):
    question = list(range(question_size))
    message = list(range(10000, 10000+message_size))
    chunks = windows(question, message)
    assert [token for q, _, _ in chunks for token in q] == question
    assert all(m == message and len(q)+len(m)+3 <= 512 for q, m, _ in chunks)
    assert abs(sum(w for _, _, w in chunks)-1) < 1e-12


def test_overlong_message_is_rejected_without_truncation():
    with pytest.raises(ValueError, match="silently truncate"):
        windows([1], list(range(510)))


def test_same_predictor_and_initialization_for_both_encodings():
    cfg = config(study.CONFIG)
    seed_all(42)
    original = question_context.make_model(cfg, "full_embedding")
    for kind in study.VARIANTS:
        seed_all(42)
        current = study.make_model(cfg, kind)
        assert sum(p.numel() for p in current.parameters()) == 382498
        for k, v in current.state_dict().items():
            torch.testing.assert_close(v, original.state_dict()[k], rtol=0, atol=0)


def test_question_conditioning_and_mapping_complete():
    cfg = config(study.CONFIG)
    caches = study.prepare(cfg)
    arrays, index, message, _ = caches["message_only"]
    conditioned = caches["question_conditioned"][2]
    assert message.shape == conditioned.shape
    assert not message[0].any() and not conditioned[0].any()
    sample = np.random.default_rng(451).choice(np.arange(1, len(message)), 1000, replace=False)
    np.testing.assert_allclose(np.linalg.norm(message[sample], axis=-1), 1., rtol=0, atol=3e-6)
    np.testing.assert_allclose(np.linalg.norm(conditioned[sample], axis=-1), 1., rtol=0, atol=3e-6)
    assert (np.linalg.norm(message[sample]-conditioned[sample], axis=-1) > .001).all()
    assert {r["diagnostic_role"] for r in index} == {"train", "validation", "familiar", "unseen"}
    assert arrays["context"].shape[-1] == 16+1536
