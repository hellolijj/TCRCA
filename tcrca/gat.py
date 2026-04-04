"""
Graph Attention refinement on the service call graph (Eq. (14)–(17)).

Directed edges follow the tensorizer convention: A[u, v] = 1 if u invokes v.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcrca.config import TCRCAConfig


class GATLayer(nn.Module):
    """
    One GAT layer. For each target v, attention is normalized over sources u with A[u, v] > 0.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, 2 * out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n, _ = x.shape
        h = self.W(x).view(n, self.heads, self.out_dim)
        # logits[u, v, head] for message u -> v
        hv = h.unsqueeze(0).expand(n, n, self.heads, self.out_dim)
        hu = h.unsqueeze(1).expand(n, n, self.heads, self.out_dim)
        cat = torch.cat([hv, hu], dim=-1)
        logits = (cat * self.attn.view(1, 1, self.heads, 2 * self.out_dim)).sum(-1)
        logits = F.leaky_relu(logits, negative_slope=0.2)

        mask = (adj > 0).float().unsqueeze(-1)
        logits = logits.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(logits, dim=0)
        alpha = self.dropout(alpha)

        out = torch.einsum("uvh,uhd->vhd", alpha, h)
        out = out.reshape(n, self.heads * self.out_dim)
        return out


class GATStack(nn.Module):
    """K-layer GAT (Eq. (14)–(15)) plus trace-level pooling (Eq. (16)–(17))."""

    def __init__(self, cfg: TCRCAConfig, in_dim: int):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(cfg.gat_layers):
            layers.append(GATLayer(dim, cfg.gat_hidden, cfg.gat_heads, cfg.dropout))
            dim = cfg.gat_hidden * cfg.gat_heads
        self.layers = nn.ModuleList(layers)
        self.trace_pool = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for layer in self.layers:
            h = F.elu(layer(h, adj))
        w = torch.softmax(self.trace_pool(h), dim=0)
        hg = (w * h).sum(dim=0, keepdim=True)
        return h, hg
