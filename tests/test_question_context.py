"""Leakage, information-control and model invariants for question context."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from mac_agents.common import seed_all
from scripts.argument_model import ResponseUpdate
from scripts.question_context import QuestionResponseUpdate, augmented_arrays, fit_features
from test_argument_mechanisms import batch_fixture


def test_held_out_embeddings_cannot_change_fitting_transforms():
    x = np.random.default_rng(92).normal(size=(10, 40))
    fit = list(range(7))
    before = fit_features(x, fit)
    changed = x.copy()
    changed[7:] *= 1000
    after = fit_features(changed, fit)
    for key in ("mean", "components", "full_scale", "pca3_scale", "training_singular_values"):
        np.testing.assert_array_equal(before[key], after[key])
    for key in ("pca3_matched", "full_embedding"):
        np.testing.assert_array_equal(before[key][:7], after[key][:7])
        np.testing.assert_allclose((before[key][:7]**2).sum(-1).mean(), 1., atol=1e-7)
    assert not np.array_equal(before["full_embedding"][7:], after["full_embedding"][7:])


def test_matched_control_contains_only_three_fitted_components():
    x = np.random.default_rng(19).normal(size=(10, 40))
    got = fit_features(x, range(7))
    v = got["components"]
    expected = ((x-got["mean"])@v.T)@v/got["pca3_scale"]
    np.testing.assert_allclose(got["pca3_matched"], expected, rtol=1e-6, atol=1e-7)
    # A held-out perturbation orthogonal to the three PCs is absent in the
    # matched control and retained by the full representation.
    noise = np.random.default_rng(15).normal(size=40)
    noise -= (noise@v.T)@v
    changed = x.copy()
    changed[-1] += noise
    other = fit_features(changed, range(7))
    np.testing.assert_allclose(got["pca3_matched"], other["pca3_matched"], rtol=0, atol=1e-7)
    assert np.linalg.norm(got["full_embedding"][-1]-other["full_embedding"][-1]) > .1


def test_augmentation_preserves_old_context_and_all_dynamic_fields(tmp_path):
    q = np.arange(24, dtype=np.float32).reshape(3, 8)
    path = tmp_path/"features.npz"
    np.savez(path, qids=np.array(["a", "b", "c"]), pca3_matched=q, full_embedding=q*2)
    arrays = {"context": np.random.default_rng(3).normal(size=(2, 4, 16)).astype(np.float32),
              "spins": np.ones((2, 3, 4)), "other": np.array([123])}
    index = [{"qid": "c"}, {"qid": "a"}]
    result = augmented_arrays(arrays, index, path, "full_embedding")
    np.testing.assert_array_equal(result["context"][..., :16], arrays["context"])
    for key in ("spins", "other"):
        assert result[key] is arrays[key]
    for e, source in enumerate((q[2]*2, q[0]*2)):
        np.testing.assert_array_equal(result["context"][e, :, 16:], np.tile(source, (4, 1)))


def augmented_batch(device, zeros=False):
    b = {k: v.to(device) for k, v in batch_fixture().items()}
    q = torch.zeros((b["s"].shape[0], 8), device=device) if zeros else torch.tensor(
        np.random.default_rng(5).normal(size=(b["s"].shape[0], 8)), device=device, dtype=torch.float32)
    return b, {**b, "context": torch.cat([b["context"], q[:, None].expand(-1, b["s"].shape[1], -1)], -1)}


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_shared_initial_function_rng_and_active_question_gradients(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("GPU requires host access")
    base_batch, rich_batch = augmented_batch(device)
    seed_all(42)
    base = ResponseUpdate(text_dim=8).to(device)
    rng = torch.get_rng_state()
    seed_all(42)
    rich = QuestionResponseUpdate(question_dim=8, text_dim=8).to(device)
    assert torch.equal(rng, torch.get_rng_state())
    seed_all(901)
    a = base(base_batch)
    seed_all(901)
    b = rich(rich_batch)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    loss = torch.nn.functional.cross_entropy(b.reshape(-1, 2), rich_batch["target"].reshape(-1))
    loss.backward()
    grad = rich.encoder.question_projection.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    old = rich.encoder.question_projection.weight.detach().clone()
    torch.optim.AdamW(rich.parameters(), lr=.0003).step()
    assert not torch.equal(old, rich.encoder.question_projection.weight)
    rich.eval()
    perturbed = deepcopy(rich_batch)
    perturbed["context"][..., 16:] *= -1
    assert (rich(perturbed)-rich(rich_batch)).abs().max() > 1e-5


def test_question_context_keeps_padding_safe_and_node_permutation_equivariant():
    _, b = augmented_batch("cpu")
    seed_all(34)
    model = QuestionResponseUpdate(question_dim=8, text_dim=8).eval()
    with torch.no_grad():
        model.encoder.question_projection.weight.normal_(0, .1)
    first = model(b)
    padded = deepcopy(b)
    padded["text"][b["signs"] == 0] = 1e5
    torch.testing.assert_close(first, model(padded), rtol=0, atol=0)
    order = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(order)
    permuted = {k: (v[:, order] if k in ("s", "context", "history", "signs", "stance", "text", "target") else v)
                for k, v in b.items()}
    permuted["neighbors"] = inverse[b["neighbors"][:, order]]
    torch.testing.assert_close(first[:, order], model(permuted), rtol=1e-6, atol=1e-6)
