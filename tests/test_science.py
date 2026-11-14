from argparse import Namespace
import copy
import json

import numpy as np
import pytest
import torch

from mac_agents.common import config, json_write
from mac_agents.data.features import Snapshots, history_features
from mac_agents.data.messages import pool_existing_messages
from mac_agents.data.prepare import question_splits, pca_fit_transform
from mac_agents.evaluate import check_selection, freeze, rollout
from mac_agents.metrics import paired_bootstrap
from mac_agents.models.graph import GraphUpdate
from mac_agents.train import train, rng_state


def inputs(n=5):
    torch.manual_seed(173)
    s = torch.tensor([[-1., 1., -1., 1., 1.]])[:, :n]
    c = torch.randn(1, n, 16)
    j = torch.zeros(1, n, n)
    for i in range(n-1):
        j[:, i, i+1] = (-1.) ** i
        j[:, i+1, i] = (-1.) ** i
    return s, c, j, torch.zeros(1, n, 0)


def small_model(kind="A2"):
    torch.manual_seed(91)
    return GraphUpdate(kind=kind, d_model=16, heads=2, d_k=2, d_v=4, ffn_dim=32, dropout=0).eval()


@pytest.mark.parametrize("kind", ["A0", "A1", "A2"])
def test_graph_mask_and_one_hop(kind):
    model = small_model(kind)
    s, c, j, extra = inputs()
    initial = model(s, c, j, extra)
    changed = s.clone()
    changed[:, 2] *= -1
    assert torch.equal(initial[:, 0], model(changed, c, j, extra)[:, 0])
    # A connected sender can affect the receiver (positive control).
    changed = s.clone()
    changed[:, 1] *= -1
    assert not torch.allclose(initial[:, 0], model(changed, c, j, extra)[:, 0], atol=1e-7)


@pytest.mark.parametrize("kind", ["A0", "A1", "A2"])
def test_permutation_and_isolation(kind):
    model = small_model(kind)
    s, c, j, extra = inputs()
    perm = torch.tensor([3, 1, 4, 0, 2])
    expected = model(s, c, j, extra)[:, perm]
    actual = model(s[:, perm], c[:, perm], j[:, perm][:, :, perm], extra[:, perm])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    logits, social, weights = model(s, c, torch.zeros_like(j), extra, return_social=True)
    assert torch.isfinite(logits).all()
    assert torch.count_nonzero(social) == 0
    assert torch.count_nonzero(weights) == 0
    torch.testing.assert_close(logits.softmax(-1).sum(-1), torch.ones_like(s))
    logits.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_context_routing_and_projection_dimensions():
    s, c, j, extra = inputs()
    j = torch.ones_like(j) - torch.eye(5)[None]
    changed = c * 3 + torch.randn_like(c)
    a1, a2 = small_model("A1"), small_model("A2")
    assert torch.equal(a1(s, c, j, extra, True)[2], a1(s, changed, j, extra, True)[2])
    assert not torch.allclose(a2(s, c, j, extra, True)[2], a2(s, changed, j, extra, True)[2])
    default = GraphUpdate()
    assert default.q.out_features == default.k.out_features == 4*16
    assert default.v.out_features == 4*32


def test_question_and_graph_roles():
    bank = {f"q{i}": {"official_split": "train" if i < 20 else "test"} for i in range(40)}
    splits = question_splits(bank)
    assert list(map(len, splits.values())) == [16, 4, 20]
    assert not set(splits["train"]) & set(splits["validation"])
    assert not (set(splits["train"]) | set(splits["validation"])) & set(splits["test"])
    assert question_splits(dict(reversed(list(bank.items())))) == splits
    from mac_agents.data.features import partition_indices
    rows = [{"question_split": "train", "graph_role": "fit_graph"},
            {"question_split": "train", "graph_role": "heldout_family"},
            {"question_split": "test", "graph_role": "fit_graph"}]
    assert partition_indices(rows, "train") == [0]
    assert partition_indices(rows, "test") == [1, 2]


def test_history_and_message_forecast_cutoff():
    spins = np.array([[-1, 1], [1, 1], [-1, -1]])
    first = history_features(spins, 0, 2)
    assert np.all(first[:, :3] == 0)
    changed = spins.copy()
    changed[1:] *= -1
    np.testing.assert_array_equal(first, history_features(changed, 0, 2))
    raw = {"personas": ["a", "b"], "spins_history": np.zeros((4, 2)).tolist(),
           "J": [[0, 1], [-1, 0]],
           "messages_history": [{"0,1": "old", "1,0": "old"},
                                {"0,1": "new", "1,0": "new"},
                                {"0,1": "future", "1,0": "future"}]}
    vectors = {"old": np.array([1., 0.]), "new": np.array([0., 1.]), "future": np.array([9., 9.])}
    features = pool_existing_messages(raw, vectors, 2)
    assert np.all(features[0] == 0)
    np.testing.assert_array_equal(features[1, 1, :2], [1, 0])
    np.testing.assert_array_equal(features[1, 0, 2:4], [1, 0])
    raw["messages_history"][1] = {"0,1": "future", "1,0": "future"}
    changed = pool_existing_messages(raw, vectors, 2)
    np.testing.assert_array_equal(features[1], changed[1])
    assert not np.array_equal(features[2], changed[2])


def test_rollout_closure_and_future_independence():
    s, c, j, extra = inputs()
    cfg = {"state": {"condition": "opinion", "history_window": 2}}
    model = small_model()
    class HistoryRule(torch.nn.Module):
        def forward(self, current, context, graph, memory):
            # Reuse the previous generated state; when unavailable, flip the initial state.
            previous = torch.where(memory[..., 1] > 0, 1., -1.)
            target = torch.where(memory[..., 2] > 0, previous, -current)
            return torch.stack([-10*target, 10*target], -1)
    cfg["state"]["condition"] = "history"
    simulated = rollout(HistoryRule(), s.numpy(), c.numpy(), j.numpy(), cfg, 3, torch.device("cpu"))
    np.testing.assert_array_equal(simulated, np.stack([-s.numpy(), s.numpy(), -s.numpy()], 1))
    cfg["state"]["condition"] = "existing_messages"
    with pytest.raises(ValueError, match="autonomous"):
        rollout(model, s.numpy(), c.numpy(), j.numpy(), cfg, 3, torch.device("cpu"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable in this execution context")
def test_cpu_mps_agreement():
    model = small_model()
    values = inputs()
    cpu = model(*values)
    metal = copy.deepcopy(model).to("mps")(*[v.to("mps") for v in values]).cpu()
    torch.testing.assert_close(cpu, metal, atol=2e-5, rtol=2e-4)
    y = torch.zeros(5, dtype=torch.long)
    torch.testing.assert_close(torch.nn.functional.cross_entropy(cpu.reshape(-1, 2), y),
                               torch.nn.functional.cross_entropy(metal.reshape(-1, 2), y), atol=2e-5, rtol=2e-4)


@pytest.fixture
def synthetic_project(tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "data/processed/author_pca"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(5)
    spins = rng.choice([-1, 1], size=(4, 4, 4)).astype(np.int8)
    graph = np.ones((4, 4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)[None]
    np.savez_compressed(directory / "episodes.npz", spins=spins, votes=np.repeat(spins[..., None], 5, -1),
                        context=rng.normal(size=(4, 4, 16)).astype(np.float32), J=graph,
                        valid=np.ones((4, 3), dtype=bool), message_ok=np.ones((4, 3), dtype=bool))
    index = [{"episode_id": f"synthetic-{i}", "qid": f"synthetic-q{i}", "question_split": "train" if i < 3 else "validation",
              "graph_role": "fit_graph", "graph_hash": "synthetic-graph", "graph_family": "synthetic"} for i in range(4)]
    json_write(directory / "index.json", index)
    json_write(directory / "splits.json", {"synthetic": True})
    cfg["data"]["root"] = "data"
    cfg["model"].update(d_model=16, heads=2, d_k=2, d_v=4, ffn_dim=32, dropout=.2)
    cfg["training"].update(batch_size=2, epochs=4, patience=10)
    import yaml
    path = tmp_path / "fixture.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_exact_cpu_training_resume(synthetic_project):
    def args(steps, resume=None):
        return Namespace(config=str(synthetic_project), model="A2", state="opinion", seed=42,
                         device="cpu", max_steps=steps, wall_minutes=2, limit_episodes=None,
                         match_state=None, resume=resume, smoke=True)
    whole = train(args(6))
    stopped = train(args(3))
    resumed = train(args(6, stopped))
    a = torch.load(whole / "checkpoint.pt", weights_only=False)
    b = torch.load(resumed / "checkpoint.pt", weights_only=False)
    assert a["step"] == b["step"] == 6
    assert a["position"] == b["position"]
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="Preparation"):
        freeze([whole], whole.parent / "selection.json", "synthetic test")
    with pytest.raises(ValueError, match="requires"):
        check_selection(whole, None)


def test_paired_question_bootstrap():
    import pandas as pd
    result = paired_bootstrap(pd.Series([1., 2.], index=["q1", "q2"]),
                              pd.Series([1., 1.], index=["q1", "q2"]))
    assert result["paired_questions"] == 2
    assert result["difference"] == .5


def test_training_only_feature_fit():
    values = np.random.default_rng(11).normal(size=(20, 8))
    transformed, mean, components = pca_fit_transform(values, np.arange(16))
    changed = values.copy()
    changed[16:] *= 1000
    other, other_mean, other_components = pca_fit_transform(changed, np.arange(16))
    np.testing.assert_array_equal(mean, other_mean)
    np.testing.assert_array_equal(components, other_components)
    np.testing.assert_array_equal(transformed[:16], other[:16])


def test_author_adapter_constant_feature_stability():
    from pathlib import Path
    from types import SimpleNamespace
    from mac_agents.models.baselines import fit_baseline
    if not Path("data/raw/upstream/res/utils.py").exists():
        pytest.skip("Pinned author sources not downloaded")
    context = np.full((2, 32, 16), .123456, dtype=np.float32)
    context[..., 0] = 1
    rng = np.random.default_rng(10)
    arrays = {"context": context, "J": np.zeros((2, 32, 32), dtype=np.float32),
              "spins": rng.choice([-1, 1], (2, 2, 32)).astype(np.int8)}
    dataset = SimpleNamespace(arrays=arrays, pairs=[(0, 0), (1, 0)])
    fitted = fit_baseline("I0", dataset, config())
    np.testing.assert_allclose(fitted["weights"][1:16], 0, atol=1e-10)


def test_cpu_checkpoint_does_not_require_gpu_rng():
    assert rng_state("cpu")["mps"] is None
