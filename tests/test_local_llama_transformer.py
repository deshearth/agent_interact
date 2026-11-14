"""Scientific invariants for the local-source predictor experiment."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.argument_model import ResponseUpdate
from scripts.local_llama_transformer import LocalResponseUpdate, build_inputs, question_folds


def test_question_partitions_are_disjoint_and_cover_assessment_once():
    qids = [f"q{i}" for i in range(8)]
    folds = question_folds(qids, 20261108)
    assert sorted(q for f in folds for q in f["heldout"]) == qids
    for f in folds:
        assert [len(f[x]) for x in ("train", "selection", "heldout")] == [4, 2, 2]
        assert set(f["train"]).isdisjoint(f["selection"]+f["heldout"])
        assert set(f["selection"]).isdisjoint(f["heldout"])


def test_features_do_not_read_outcomes_and_history_is_preupdate():
    root = Path("runs/local_llama_intervention_v1/teacher_pairs")
    rows = [json.loads(s) for s in (root/"examples.jsonl").read_text().splitlines()]
    static = json.loads((root/"static_context.json").read_text())
    mapping = {"": 0}
    def lookup(s):
        return mapping.setdefault(s, len(mapping))
    original = build_inputs(rows, static, lookup)
    modified = copy.deepcopy(rows)
    for r in modified:
        r["outcomes"] = {"unusable": None}
        r["audit"] = None
    again = build_inputs(modified, static, lookup)
    for k in original:
        np.testing.assert_array_equal(original[k], again[k])
    for i, row in enumerate(rows):
        x = row["inputs"]
        if x["previous_spins"] is None:
            assert not original["history"][i, :, :3].any()
        else:
            prev = x["previous_spins"][x["receiver"]]
            np.testing.assert_array_equal(original["history"][i, 0, :3], [prev == -1, prev == 1, 1])


def equivalent_models_and_batches():
    torch.manual_seed(4)
    cfg = dict(d_model=32, heads=2, d_k=4, d_v=8, ffn_dim=64, dropout=0.)
    original = ResponseUpdate(**cfg)
    original.encoder[0] = nn.Linear(779, 32)
    packed = LocalResponseUpdate(**cfg)
    packed.load_state_dict(original.state_dict())
    b, n, width = 2, 4, 3
    signs = torch.tensor([[[1., -1., 0.], [-1., 0., 0.], [1., 0., 0.], [0., 0., 0.]]]).repeat(b, 1, 1)
    # Receiver zero's sender order matches the packed node order [1, 2, 3].
    neighbors = torch.tensor([[[1, 2, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]).repeat(b, 1, 1)
    batch = {"s": torch.randint(0, 2, (b, n))*2-1, "context": torch.randn(b, n, 768),
             "history": torch.randn(b, n, 6), "signs": signs, "neighbors": neighbors,
             "stance": torch.randint(0, 2, (b, n, width))*2-1, "text": torch.randn(b, n, width, 384)}
    local = {k: batch[k].clone() for k in ("s", "context", "history")}
    local.update({k: batch[k][:, 0].clone() for k in ("signs", "stance", "text")})
    local["structure"] = torch.stack([(signs > 0).sum(-1), (signs < 0).sum(-1), signs.abs().sum(-1)], -1).float().log1p()
    return original.eval(), packed.eval(), batch, local


def test_packed_attention_matches_prior_full_graph_implementation():
    original, packed, full, local = equivalent_models_and_batches()
    torch.testing.assert_close(original(full)[:, 0], packed(local), rtol=1e-5, atol=1e-6)


def test_padding_and_empty_inboxes_cannot_transmit_messages():
    _, model, _, local = equivalent_models_and_batches()
    baseline = model(local).detach()
    changed = {k: v.clone() for k, v in local.items()}
    changed["text"][:, 2] = 500
    changed["context"][:, 3] = -500
    changed["stance"][:, 2] = 1
    torch.testing.assert_close(baseline, model(changed))
    local["signs"].zero_()
    baseline = model(local).detach()
    local["text"] = torch.randn_like(local["text"])*100
    torch.testing.assert_close(baseline, model(local))
    assert torch.isfinite(model(local)).all()


def test_individual_argument_vectors_receive_gradient():
    _, model, _, local = equivalent_models_and_batches()
    local["text"].requires_grad_()
    model(local)[:, 1].sum().backward()
    assert local["text"].grad[:, :2].abs().sum() > 0
    assert local["text"].grad[:, 2].abs().sum() == 0
    assert model.text_projection.weight.grad.abs().sum() > 0


def test_signed_graph_cues_affect_prediction():
    _, model, _, local = equivalent_models_and_batches()
    baseline = model(local).detach()
    local["signs"] *= -1
    assert not torch.allclose(baseline, model(local))
