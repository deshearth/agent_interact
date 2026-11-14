from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GraphUpdate(nn.Module):
    """One synchronous exchange; no normalization across agent features."""
    def __init__(self, kind="A2", context_dim=16, extra_dim=0, d_model=128,
                 heads=4, d_k=16, d_v=32, ffn_dim=256, dropout=0.1):
        super().__init__()
        if kind not in ("A0", "A1", "A2"):
            raise ValueError(kind)
        self.kind, self.heads, self.d_k, self.d_v = kind, heads, d_k, d_v
        self.encoder = nn.Sequential(nn.Linear(2+context_dim+extra_dim+3, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)
        if kind != "A0":
            routing_dim = 5 if kind == "A1" else d_model
            self.q = nn.Linear(routing_dim, heads*d_k, bias=False)
            self.k = nn.Linear(routing_dim, heads*d_k, bias=False)
        self.v = nn.Linear(d_model, heads*d_v, bias=False)
        self.edge_bias = nn.Parameter(torch.zeros(heads, 2)) if kind != "A0" else None
        self.gamma = nn.Parameter(torch.tensor([[1., -1.]]).repeat(heads, 1))
        self.strength = nn.Linear(3, heads)
        self.out = nn.Linear(heads*d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def forward(self, s, context, J, extra, return_social=False):
        b, n = s.shape
        state = torch.stack([s == -1, s == 1], -1).to(context.dtype)
        structure = torch.stack([(J > 0).sum(-1), (J < 0).sum(-1), J.abs().sum(-1)], -1).to(context.dtype).log1p()
        h = self.encoder(torch.cat([state, context, extra, structure], -1))
        hn = self.norm(h)
        v = self.v(hn).view(b, n, self.heads, self.d_v).transpose(1, 2)
        mask = J.ne(0)[:, None, :, :]
        isolated = ~mask.any(-1, keepdim=True)
        if self.kind == "A0":
            alpha = mask.to(h.dtype) / mask.sum(-1, keepdim=True).clamp_min(1)
            alpha = alpha.expand(b, self.heads, n, n)
        else:
            route = torch.cat([state, structure], -1) if self.kind == "A1" else hn
            q = self.q(route).view(b, n, self.heads, self.d_k).transpose(1, 2)
            k = self.k(route).view(b, n, self.heads, self.d_k).transpose(1, 2)
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
            positive = (J > 0)[:, None].to(h.dtype)
            negative = (J < 0)[:, None].to(h.dtype)
            scores = scores + positive * self.edge_bias[None, :, None, None, 0] + negative * self.edge_bias[None, :, None, None, 1]
            scores = scores.masked_fill(~mask, -torch.inf)
            scores = torch.where(isolated, torch.zeros_like(scores), scores)
            alpha = scores.softmax(-1) * mask
        signed = (J > 0)[:, None] * self.gamma[None, :, None, None, 0] + (J < 0)[:, None] * self.gamma[None, :, None, None, 1]
        rho = F.softplus(self.strength(structure)).transpose(1, 2)[..., None]
        aggregate = rho * ((alpha * signed) @ v)
        social = self.out(aggregate.transpose(1, 2).reshape(b, n, -1))
        h = h + self.dropout(social)
        h = h + self.dropout(self.ffn(h))
        logits = self.classifier(h)
        return (logits, social, alpha) if return_social else logits


def make_graph(cfg, state):
    from ..data.features import extra_dim
    m = cfg["model"]
    if m["kind"] == "A1" and state != "opinion":
        raise ValueError("Primary A1 routing ablation is defined only for opinion state")
    model = GraphUpdate(kind=m["kind"], extra_dim=extra_dim(cfg, state),
                        **{k: m[k] for k in ("d_model", "heads", "d_k", "d_v", "ffn_dim", "dropout")})
    if sum(p.numel() for p in model.parameters()) >= m["max_parameters"]:
        raise ValueError("Model exceeds the declared parameter budget")
    return model
