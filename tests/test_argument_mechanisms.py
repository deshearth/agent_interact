"""Scientific checks for separated text fusion and conversation-level holdout."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mac_agents.common import config, seed_all
from scripts.argument_features import matched_donors
from scripts.argument_mechanisms import (CONFIG, SeparateTextUpdate, SplitSnapshots,
    audit_donors, make_model, split_groups, split_index)
from scripts.argument_model import ResponseUpdate
from test_argument_response import fixture_cache


def test_roles_hold_out_whole_episodes_and_keep_unseen_questions_out():
    cfg = config(CONFIG)
    idx = [{"qid": q, "replica": r, "question_split": split, "episode_id": f"{q}_{r}"}
           for q, split in (("fitting_q", "train"), ("unseen_q", "validation")) for r in range(4)]
    got = split_index(idx, cfg)
    assert [r["diagnostic_role"] for r in got] == ["train", "train", "validation", "familiar"]+["unseen"]*4
    assert {r["qid"] for r in got if r["diagnostic_role"] == "train"} == {"fitting_q"}
    broken = deepcopy(cfg)
    broken["diagnostic"]["selection_replicas"] = [1, 2]
    with pytest.raises(ValueError, match="overlap"):
        split_index(idx, broken)


def test_new_shuffle_cannot_import_held_out_conversations():
    a, idx, vectors, _ = fixture_cache()
    idx = [{**r, "diagnostic_role": role} for r, role in zip(idx, ["train", "validation", "familiar", "unseen"])]
    donor = matched_donors(split_groups(a, idx), a["text_ids"].shape, 42)
    audit_donors(a, idx, donor)
    # One episode per role means there is no eligible other-episode donor.
    np.testing.assert_array_equal(donor.ravel(), np.arange(donor.size))
    for role in ("train", "validation", "familiar", "unseen"):
        ds = SplitSnapshots(a, idx, vectors, role, "full_arguments", donor)
        assert len(ds) == 2
        assert all(idx[e]["diagnostic_role"] == role for e, _ in ds.pairs)
    with pytest.raises(ValueError, match="development"):
        SplitSnapshots(a, idx, vectors, "test", "full_arguments", donor)


def batch_fixture():
    a, idx, vectors, donor = fixture_cache()
    idx = [{**r, "diagnostic_role": "train"} for r in idx]
    ds = SplitSnapshots(a, idx, vectors, "train", "full_arguments", donor)
    return next(iter(DataLoader(ds, batch_size=2)))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_zero_text_preserves_shared_function_gradients_and_optimizer(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("GPU requires host access")
    b = {k: v.to(device) for k, v in batch_fixture().items()}
    b["text"].zero_()
    seed_all(42)
    base = ResponseUpdate(text_dim=8).to(device)
    after_base = torch.get_rng_state()
    seed_all(42)
    separate = SeparateTextUpdate(text_dim=8).to(device)
    assert torch.equal(after_base, torch.get_rng_state())
    opts = [torch.optim.AdamW(m.parameters(), lr=.0003, weight_decay=.001) for m in (base, separate)]
    for step in range(3):
        ys = []
        for m, opt in zip((base, separate), opts):
            seed_all(500+step)
            m.train()
            opt.zero_grad(set_to_none=True)
            y = m(b)
            ys.append(y.detach())
            torch.nn.functional.cross_entropy(y.reshape(-1, 2), b["target"].reshape(-1)).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step()
        torch.testing.assert_close(*ys, rtol=0, atol=0)
        for name, p in base.named_parameters():
            torch.testing.assert_close(p, dict(separate.named_parameters())[name], rtol=0, atol=0)


def test_separate_text_branch_is_active_but_padding_and_isolated_nodes_are_safe():
    seed_all(81)
    model = SeparateTextUpdate(text_dim=8).eval()
    b = batch_fixture()
    first = model(b)
    zero = {**b, "text": torch.zeros_like(b["text"])}
    assert (first-model(zero)).abs().max() > 1e-4
    changed = deepcopy(b)
    changed["text"][b["signs"] == 0] = 1e5
    torch.testing.assert_close(first, model(changed), rtol=0, atol=0)
    torch.testing.assert_close(first[:, 3], model(zero)[:, 3], rtol=0, atol=0)
    loss = torch.nn.functional.cross_entropy(first.reshape(-1, 2), b["target"].reshape(-1))
    loss.backward()
    for name in ("text_projection", "text_k", "text_v"):
        grad = getattr(model, name).weight.grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
