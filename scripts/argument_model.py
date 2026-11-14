"""One-hop response predictor with individual message keys and values."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ResponseUpdate(nn.Module):
    def __init__(self, kind="A2", text_dim=384, d_model=128, heads=4, d_k=16,
                 d_v=32, ffn_dim=256, dropout=0.1):
        super().__init__()
        if kind not in ("A0", "A2"):
            raise ValueError(kind)
        self.kind, self.heads, self.d_k, self.d_v = kind, heads, d_k, d_v
        self.encoder = nn.Sequential(nn.Linear(2+16+6+3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.text_projection = nn.Linear(text_dim, d_model, bias=False)
        self.stance_projection = nn.Linear(2, d_model, bias=False)
        self.edge_norm = nn.LayerNorm(d_model)
        if kind == "A2":
            self.q = nn.Linear(d_model, heads*d_k, bias=False)
            self.k = nn.Linear(d_model, heads*d_k, bias=False)
            self.edge_bias = nn.Parameter(torch.zeros(heads, 2))
        self.v = nn.Linear(d_model, heads*d_v, bias=False)
        self.gamma = nn.Parameter(torch.tensor([[1., -1.]]).repeat(heads, 1))
        self.strength = nn.Linear(3, heads)
        self.out = nn.Linear(heads*d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def forward(self, batch):
        s, context, signs = batch["s"], batch["context"], batch["signs"]
        b, n = s.shape
        width = signs.shape[-1]
        state = torch.stack([s == -1, s == 1], -1).to(context.dtype)
        structure = torch.stack([(signs > 0).sum(-1), (signs < 0).sum(-1),
                                 signs.abs().sum(-1)], -1).to(context.dtype).log1p()
        h = self.encoder(torch.cat([state, context, batch["history"], structure], -1))
        hn = self.norm(h)
        sender = hn[torch.arange(b, device=s.device)[:, None, None], batch["neighbors"]]
        stance = torch.stack([batch["stance"] == -1, batch["stance"] == 1], -1).to(context.dtype)
        edge = self.edge_norm(sender + self.text_projection(batch["text"]) + self.stance_projection(stance))
        values = self.v(edge).view(b, n, width, self.heads, self.d_v).permute(0, 3, 1, 2, 4)
        mask = signs.ne(0)[:, None]
        isolated = ~mask.any(-1, keepdim=True)
        if self.kind == "A0":
            alpha = mask.to(h.dtype) / mask.sum(-1, keepdim=True).clamp_min(1)
        else:
            queries = self.q(hn).view(b, n, self.heads, self.d_k).permute(0, 2, 1, 3)
            keys = self.k(edge).view(b, n, width, self.heads, self.d_k).permute(0, 3, 1, 2, 4)
            scores = (queries[..., None, :] * keys).sum(-1) / math.sqrt(self.d_k)
            scores = scores + (signs > 0)[:, None] * self.edge_bias[None, :, None, None, 0] + (
                signs < 0)[:, None] * self.edge_bias[None, :, None, None, 1]
            scores = torch.where(isolated, torch.zeros_like(scores), scores.masked_fill(~mask, -torch.inf))
            alpha = scores.softmax(-1) * mask
        signed = (signs > 0)[:, None] * self.gamma[None, :, None, None, 0] + (
            signs < 0)[:, None] * self.gamma[None, :, None, None, 1]
        rho = F.softplus(self.strength(structure)).permute(0, 2, 1)[..., None]
        aggregate = rho * ((alpha * signed)[..., None] * values).sum(-2)
        social = self.out(aggregate.permute(0, 2, 1, 3).reshape(b, n, -1))
        h = h + self.dropout(social)
        h = h + self.dropout(self.ffn(h))
        return self.classifier(h)
