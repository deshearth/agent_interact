"""Controls for nested sampling and privileged labels in the Llama diagnostic."""
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mac_agents.common import config, json_write
from scripts import breadth_probability_learning_curve as curve
from scripts import transformer_generalization as base


def synthetic_questions():
    return [{"qid": f"q{i:03d}", "split": "train" if i < 80 else "validation",
             "type": f"topic{i % 7}", "graph_id": f"J{i % 8}",
             "answer": "A" if i % 2 else "B", "level": str(i % 5)} for i in range(100)]


def synthetic_arrays():
    rng = np.random.default_rng(6901)
    common = {"spins": rng.choice([-1, 1], (5, 9, 32)).astype(np.int8),
              "J": np.tile(np.ones((32, 32)) - np.eye(32), (5, 1, 1)).astype(np.int8),
              "valid": np.ones((5, 8), bool), "episode_qids": np.array([f"q{i}" for i in range(5)]),
              "pairs": np.array([(e, t) for e in range(5) for t in range(8)])}
    common["raw_votes"] = np.repeat(common["spins"][..., None], 5, -1)
    features = {"context": rng.normal(size=(5, 32, 16)).astype(np.float32),
                "prior_logit": np.zeros((40, 32))}
    return common, features


def test_nested_subsets_balance_metadata_and_ignore_holdouts_and_outcomes():
    questions = synthetic_questions()
    subsets = curve.balanced_subsets(questions, [10, 20, 40, 80], [67031, 67032, 67033])
    assert len(subsets) == 10 and subsets[0]["n_questions"] == 80
    lookup = {q["qid"]: q for q in questions[:80]}
    for path in range(3):
        previous = set()
        for n in [10, 20, 40]:
            part = next(s for s in subsets if s["path"] == path and s["n_questions"] == n)
            chosen = set(part["questions"])
            assert previous <= chosen <= set(lookup) and len(chosen) == n
            assert Counter(lookup[q]["answer"] for q in chosen) == {"A": n // 2, "B": n // 2}
            for key in ["type", "graph_id"]:
                total = Counter(q[key] for q in lookup.values())
                actual = Counter(lookup[q][key] for q in chosen)
                for value, count in total.items():
                    assert np.floor(count*n/80) <= actual[value] <= np.ceil(count*n/80)
            previous = chosen
    changed = copy.deepcopy(questions)
    for q in changed:
        q["future_opinions"] = [123] * 32
        if q["split"] != "train":
            q.update(type="irrelevant", answer="X", graph_id="unused")
    assert curve.balanced_subsets(changed, [10, 20, 40, 80], [67031, 67032, 67033]) == subsets


def test_soft_targets_are_next_round_probabilities_and_never_features():
    common, features = synthetic_arrays()
    rows = []
    for e, qid in enumerate(common["episode_qids"]):
        for t in range(9):
            for agent in range(32):
                p = .1 + .08 * t
                rows.append({"qid": qid, "split": "train" if e < 4 else "validation",
                    "round": t, "agent": agent, "raw_votes": common["raw_votes"][e,t,agent].tolist(),
                    "first_literal_A_probabilities": [p]*5, "first_literal_B_probabilities": [1-p]*5})
    targets = curve.source_targets(common, rows)
    _, t = common["pairs"].T
    expected = np.broadcast_to(curve.majority_probability(.1 + .08*(t+1))[:,None], (40,32))
    np.testing.assert_allclose(targets["soft"], expected, atol=1e-14)
    assert not np.allclose(targets["soft"][0], curve.majority_probability(.1))
    hard = curve.TargetData(common, features, "cpu", targets["hard"])
    soft = curve.TargetData(common, features, "cpu", targets["soft"])
    for key in ("s", "context", "J", "prior", "extra"):
        torch.testing.assert_close(hard.batch([0,1])[key], soft.batch([0,1])[key], rtol=0, atol=0)
    np.testing.assert_array_equal(hard.y, soft.y)
    assert not torch.equal(hard.values["target"], soft.values["target"])
    with pytest.raises(ValueError, match="outside"):
        curve.source_targets(common, [{**rows[0], "split": "assessment"}, *rows[1:]])
    with pytest.raises(AssertionError, match="Duplicate"):
        curve.source_targets(common, rows + [rows[0]])


def test_fixed_step_training_uses_soft_labels_but_scores_hard_outcomes(tmp_path):
    common, features = synthetic_arrays()
    e, t = common["pairs"].T
    hard = (common["spins"][e,t+1] == 1).astype(float)
    cfg = config(curve.CONFIG)
    cfg["output"] = str(tmp_path)
    cfg["training"]["validation_every_steps"] = 4
    protocol = {"candidates": [{"id": "c7", "learning_rate": .001, "penalty": .001}]}
    json_write(tmp_path / "protocol.json", protocol)
    json_write(tmp_path / "selection.json", {"synthetic_fixed_steps": 8})
    models, manifests = [], []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for name, targets in [("hard", hard), ("binary_soft", hard.copy()), ("fractional", .2 + .5*hard)]:
            fold = {"id": name, "train_questions": ["q0", "q1", "q2", "q3"],
                    "validation_questions": ["q4"], "assessment_questions": ["q5"]}
            json_write(tmp_path / "folds" / name / "manifest.json", fold)
            data = curve.TargetData(common, features, "cpu", targets)
            item = {"phase": "refit", "fold": name, "outer": 0, "model": "nano", "candidate": "c7", "seed": 142}
            out, manifest = base.train_one(cfg, protocol, data, fold, item, "cpu", float("inf"), fixed_steps=8)
            models.append(torch.load(out / "best.pt", weights_only=False)["model"])
            manifests.append(manifest)
            assert manifest["steps"] == 8 and manifest["parameters"] == 818 and manifest["validation"] is None
            aggregate, _ = base.metrics(np.load(out / "train_logp.npy"), data.y[:32], data.qids[:32])
            assert aggregate == manifest["train"]
        for key in models[0]:
            torch.testing.assert_close(models[0][key], models[1][key], rtol=0, atol=0)
        assert any(not torch.equal(models[0][key], models[2][key]) for key in models[0])
        assert len({m["initial_weights_sha256"] for m in manifests}) == 1
    finally:
        torch.set_num_threads(old_threads)
