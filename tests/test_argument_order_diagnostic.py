import ast
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from mac_agents.common import config, seed_all
from scripts import argument_mechanisms as previous
from scripts import argument_order_diagnostic as study
from scripts import question_context


def test_positions_match_released_function_and_keep_padding_zero():
    source = ast.parse(Path("data/raw/upstream/lib/datagen/dynamics.py").read_text())
    func = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "_build_inboxes")
    module = ast.Module(body=[func], type_ignores=[])
    scope = {"random": random, "Replica": object, "Dict": Dict, "List": List, "Tuple": Tuple}
    exec(compile(module, "released_dynamics", "exec"), scope)
    neighbors = np.array([[1, 2, 3, 0], [0, 2, 3, 0], [0, 1, 3, 0], [0, 1, 2, 0]])
    signs = np.array([[1, -1, 1, 0], [-1, -1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 0]])
    graph, messages = np.zeros((4, 4), dtype=np.int8), {}
    for r in range(4):
        for k in np.flatnonzero(signs[r]):
            s = int(neighbors[r, k])
            graph[s, r] = signs[r, k]
            messages[s, r] = str(s)
    rep = SimpleNamespace(name="toy-replica", n=4, J=graph)
    for t in (0, 1, 7):
        pos = study.positions(rep.name, t, neighbors, signs, "released_order")
        agree, disagree = scope["_build_inboxes"](rep, messages, t+1)
        for r in range(4):
            for sign, expected in ((1, agree[r]), (-1, disagree[r])):
                slots = np.flatnonzero(signs[r] == sign)
                slots = sorted(slots, key=lambda k: pos[r, k, 0])
                assert [str(neighbors[r, k]) for k in slots] == expected
        assert not pos[signs == 0].any()


def test_model_initial_equivalence_and_position_gradient():
    torch.set_num_threads(2)
    cfg = config(study.CONFIG)
    cache = study.prepare(cfg)["released_order"]
    arrays, index, vectors, _ = cache
    donors = np.load(previous.ROOT/"donors_s42.npy", allow_pickle=False)
    ds = study.OrderSnapshots(arrays, index, vectors, "train", "full_arguments", donors)
    batch = default_collate([ds[0], ds[1]])
    seed_all(42)
    base = question_context.make_model(cfg, "full_embedding").eval()
    seed_all(42)
    model = study.make_model(cfg, "released_order").eval()
    assert sum(p.numel() for p in model.parameters()) == 382882
    for k, v in base.state_dict().items():
        torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
    expected, actual = base(batch), model(batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.nn.functional.cross_entropy(actual.flatten(0, 1), batch["target"].flatten()).backward()
    assert model.order_projection.weight.grad.norm() > 0
    flipped = {**batch, "order": -batch["order"]}
    with torch.no_grad():
        model.order_projection.weight.add_(.1*torch.randn_like(model.order_projection.weight))
        assert (model(batch)-model(flipped)).abs().max() > 1e-4


def test_independent_order_control_preserves_each_sign_group_multiset():
    cfg = config(study.CONFIG)
    caches = study.prepare(cfg)
    a = caches["released_order"][0]["order"]
    b = caches["random_order"][0]["order"]
    signs = caches["released_order"][0]["signs"]
    assert (a != b).any()
    for e in range(0, len(a), 41):
        for r in range(32):
            for sign in (-1, 1):
                mask = signs[e, r] == sign
                for t in range(8):
                    np.testing.assert_array_equal(np.sort(a[e, t, r, mask], axis=0), np.sort(b[e, t, r, mask], axis=0))
