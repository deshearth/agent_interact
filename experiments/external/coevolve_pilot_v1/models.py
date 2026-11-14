"""Small matched-budget models of the recorded neighbor-role/belief summaries."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Inputs:
    def __init__(self, arrays):
        self.state = torch.as_tensor(arrays['state'], dtype=torch.long)
        self.target = torch.as_tensor(arrays['target'], dtype=torch.long).flatten()
        self.identity = torch.as_tensor(arrays['identity'], dtype=torch.long)
        self.graph = torch.as_tensor(arrays['adjacency'], dtype=torch.bool)
        self.n = self.state.shape[1]
        self.qids = np.repeat(arrays['question'], self.n)
        self.partitions = np.repeat(arrays['partition'], self.n)
        self.own = self.state.flatten().numpy()
        belief = F.one_hot(self.state, 3).float()
        role = F.one_hot(self.identity, 15).float()
        self.nodes = torch.cat([belief, role], -1)
        degree = self.graph.sum(-1).float()
        time = torch.as_tensor(arrays['step'], dtype=torch.float32)[:, None].expand_as(degree)/9.
        self.context = torch.stack([time, degree.log1p()/math.log1p(self.n)], -1)
        self.receiver = torch.cat([self.nodes, self.context], -1).reshape(-1, 20)
        counts = self.graph.float() @ belief
        proportions = counts/degree[..., None].clamp_min(1)
        joint = (role[..., :, None]*belief[..., None, :]).reshape(*belief.shape[:2], 45)
        role_proportions = (self.graph.float() @ joint)/degree[..., None].clamp_min(1)
        self.pooled = torch.cat([self.nodes, self.context, proportions, role_proportions], -1).reshape(-1, 68)
        self.local = torch.cat([belief, proportions], -1).reshape(-1, 6)
        self.local_identity = torch.cat([belief, proportions, role], -1).reshape(-1, 21)

    def ids(self, partition):
        return np.flatnonzero(self.partitions == partition)

    def batch(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long)
        e, a = ids//self.n, ids % self.n
        receiver = self.receiver[ids]
        # Sender degree is not in the LLM's summary. Both sets receive only the
        # recipient's degree and round, identical across senders for this update.
        sender = torch.cat([self.nodes[e], self.context[e, a, None].expand(-1, self.n, -1)], -1)
        return receiver, sender, self.graph[e, a], self.pooled[ids]


class PooledMLP(nn.Module):
    def __init__(self, mean, scale, dropout=.1):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(68, 8), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(8, 20), nn.GELU(), nn.Dropout(dropout), nn.Linear(20, 3))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, receiver, sender, mask, pooled):
        return self.net((pooled-self.mean)/self.scale)


class TinyAttention(nn.Module):
    """Width 8, 4 heads, dk=1, dv=2, FFN=16; 795 trainable parameters."""
    def __init__(self, dropout=.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(20, 8), nn.GELU(), nn.Linear(8, 8))
        self.norm = nn.LayerNorm(8)
        self.q = nn.Linear(8, 4, bias=False); self.k = nn.Linear(8, 4, bias=False)
        self.v = nn.Linear(8, 8, bias=False); self.out = nn.Linear(8, 8, bias=False)
        self.strength = nn.Linear(1, 4)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 16), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(16, 8))
        self.classifier = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 3))
        nn.init.zeros_(self.classifier[-1].weight); nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, receiver, sender, mask, pooled=None):
        b, n, _ = sender.shape
        h = self.encoder(receiver); hn = self.norm(h)
        sh = self.norm(self.encoder(sender))
        q = self.q(hn).view(b, 4, 1)
        k = self.k(sh).transpose(1, 2)
        score = q*k
        mask = mask[:, None, :]
        isolated = ~mask.any(-1, keepdim=True)
        score = score.masked_fill(~mask, -torch.inf)
        score = torch.where(isolated, torch.zeros_like(score), score)
        alpha = score.softmax(-1)*mask
        values = self.v(sh).view(b, n, 4, 2).transpose(1, 2)
        rho = F.softplus(self.strength(receiver[:, -1:]))[..., None]
        aggregate = rho*(alpha[..., None]*values).sum(-2)
        h = h+self.dropout(self.out(aggregate.reshape(b, 8)))
        h = h+self.dropout(self.ffn(h))
        return self.classifier(h)


def make_model(kind, normalization, dropout):
    if kind == 'pooled_mlp':
        return PooledMLP(**normalization, dropout=dropout)
    if kind == 'tiny_attention':
        return TinyAttention(dropout=dropout)
    raise ValueError(kind)
