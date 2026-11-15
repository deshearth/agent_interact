from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from scripts.modal_encoder_finetune import encode_ids


class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(12, 4); self.layer = nn.Linear(4, 4)
        self.embedding.weight.requires_grad_(False)
    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.layer(self.embedding(input_ids)))


def test_complete_window_pooling_preserves_padding_and_gradients():
    torch.manual_seed(2); model = Encoder(); token_spec = SimpleNamespace(cls=1, sep=2, pad=0)
    tokens = [[], [3, 4]*300, [5, 6, 7]]; ids = np.array([[0, 1, 2, 1]])
    actual = encode_ids(model, ids, tokens, token_spec, 'cpu', 2)
    torch.testing.assert_close(actual[0, 0], torch.zeros(4), atol=0, rtol=0)
    torch.testing.assert_close(actual[0, 1], actual[0, 3], atol=0, rtol=0)
    expected = torch.nn.functional.normalize(model.layer(model.embedding(torch.tensor(tokens[1]))).sum(0), dim=0)
    torch.testing.assert_close(actual[0, 1], expected, atol=1e-6, rtol=1e-6)
    actual.sum().backward(); assert model.layer.weight.grad.abs().sum() > 0
    assert model.embedding.weight.grad is None
