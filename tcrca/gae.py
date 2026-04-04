"""
Graph autoencoder for unsupervised anomaly detection (Sec. IV-C, Eq. (18)–(23)).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcrca.config import TCRCAConfig


def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization with self-loops (Kipf & Welling style)."""
    n = adj.size(0)
    a = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
    deg = a.sum(dim=1).clamp(min=1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    d_mat = torch.diag(d_inv_sqrt)
    return d_mat @ a @ d_mat


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm @ x @ self.weight + self.bias


class GraphAutoencoderTCRCA(nn.Module):
    """
    GCN encoder + structure decoder (Eq. (19)) + attribute decoder (Eq. (20)).
    """

    def __init__(self, cfg: TCRCAConfig, input_dim: int):
        super().__init__()
        self.cfg = cfg
        dims = [input_dim] + [cfg.gae_hidden] * cfg.gae_gcn_layers
        self.layers = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.hidden_dim = dims[-1]
        self.attr_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, input_dim),
        )
        self.ws = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(self.ws)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        a_norm = _normalize_adj(adj)
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, a_norm))
        return h

    def decode_structure(self, h: torch.Tensor) -> torch.Tensor:
        logits = (h @ self.ws) @ h.t()
        return torch.sigmoid(logits)

    def decode_features(self, h: torch.Tensor) -> torch.Tensor:
        return self.attr_decoder(h)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x, adj)
        a_hat = self.decode_structure(h)
        x_hat = self.decode_features(h)
        return h, a_hat, x_hat

    def reconstruction_loss(self, x: torch.Tensor, adj: torch.Tensor, a_hat: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Eq. (21)."""
        bce = F.binary_cross_entropy(a_hat, adj.clamp(0, 1))
        fro = F.mse_loss(x_hat, x)
        return self.cfg.lambda_structure * bce + self.cfg.lambda_feature * fro

    def node_anomaly_scores(self, x: torch.Tensor, adj: torch.Tensor, a_hat: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Eq. (22)–(23) (node-level only; trace-level max is applied outside)."""
        alpha = self.cfg.anomaly_alpha
        row_bce = ((a_hat - adj) ** 2).mean(dim=1)
        row_feat = ((x_hat - x) ** 2).mean(dim=1)
        return alpha * row_bce + (1.0 - alpha) * row_feat
