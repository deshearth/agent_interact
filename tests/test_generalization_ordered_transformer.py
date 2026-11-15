import numpy as np
import torch
from scripts import generalization_ordered_transformer as study
from scripts.generalization_state_models import ordered_neighbors
from scripts.local_conversation_pilot import deliver

def fixture():
    graph = np.zeros((1, 5, 5), dtype=np.int8)
    for a, b, s in [(0, 1, 1), (0, 2, 1), (0, 3, -1), (1, 2, -1)]: graph[:, a, b] = graph[:, b, a] = s
    common = {'J': graph, 'pairs': np.array([[0, 0], [0, 1]])}
    neighbors, signs = study.pair.neighbor_slots(graph)
    order = ordered_neighbors(graph, ['example'], common['pairs'])
    positions = study.positions_for_slots(common, order, neighbors)
    return common, neighbors, signs, positions

def test_positions_match_source_inboxes_and_padding():
    common, neighbors, signs, positions = fixture()
    for b in range(2):
        agree, disagree, _ = deliver(common['J'][0].tolist(), {i: str(i) for i in range(4)}, 'example', b+1)
        for receiver in range(5):
            for sequence in (agree[receiver], disagree[receiver]):
                for rank, name in enumerate(sequence):
                    slot = next(j for j, sender in enumerate(neighbors[0, receiver]) if sender == int(name) and signs[0, receiver, j] != 0)
                    np.testing.assert_allclose(positions[b, receiver, slot], [rank/max(len(sequence)-1, 1), rank == 0, rank == len(sequence)-1])
    assert not positions[:, 4].any()
    assert not positions[np.broadcast_to(signs == 0, positions.shape[:-1])].any()

def test_positions_are_effective_and_packed_permutation_preserves_predictions():
    torch.manual_seed(13); common, neighbors, signs, positions = fixture()
    model = study.OrderedPairAttention(dropout=0., **study.config()['models']['nano_pca8']).eval()
    assert sum(p.numel() for p in model.parameters()) == 970
    s = torch.tensor([[1, -1, 1, -1, 1]], dtype=torch.float32); context = torch.randn(1, 5, 16); text = torch.randn(1, 5, 8)
    graph = torch.tensor(common['J'], dtype=torch.float32); n = torch.tensor(neighbors); sign = torch.tensor(signs); pos = torch.tensor(positions[:1])
    a = model(s, context, graph, text, neighbors=n, signs=sign, positions=pos)
    perm = torch.arange(n.shape[-1]-1, -1, -1)
    b = model(s, context, graph, text, neighbors=n[..., perm], signs=sign[..., perm], positions=pos[..., perm, :])
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    c = model(s, context, graph, text, neighbors=n, signs=sign, positions=torch.zeros_like(pos))
    assert float((a-c).detach().abs().max()) > 1e-5
    torch.testing.assert_close(a[:, 4], c[:, 4]); assert torch.isfinite(a).all()
    a.sum().backward(); assert model.pair_encoder[0].weight.grad[:, -3:].abs().sum() > 0
