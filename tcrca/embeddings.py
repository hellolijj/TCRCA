"""
Multi-modal node embeddings (TCRCA Sec. IV-B, Eq. (7)–(13)).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcrca.config import TCRCAConfig


class NeuralLogHawkesEmbedding(nn.Module):
    """
    Log embedding x_i^log = [λ_i(t_1), …, λ_i(t_L)] (Eq. (7)–(8)).

    μ, α, β are learnable scalars (shared) for stability on small batches.
    """

    def __init__(self, num_bins: int):
        super().__init__()
        self.num_bins = num_bins
        self.mu = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, event_times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            event_times: [n, E] normalized timestamps in [0,1], padded with -1
        Returns:
            [n, L] intensity snapshots at uniform grid
        """
        n, _ = event_times.shape
        grid = torch.linspace(0.0, 1.0, steps=self.num_bins, device=event_times.device, dtype=event_times.dtype)
        out = torch.zeros((n, self.num_bins), device=event_times.device, dtype=event_times.dtype)
        alpha = F.softplus(self.alpha) + 1e-6
        beta = F.softplus(self.beta) + 1e-6
        mu = F.softplus(self.mu)
        for b in range(self.num_bins):
            t = grid[b]
            valid = event_times >= 0
            dt = t - event_times
            mask = valid & (dt > 0)
            contrib = torch.zeros_like(dt)
            contrib[mask] = alpha * torch.exp(-beta * dt[mask])
            out[:, b] = mu + contrib.sum(dim=1)
        return out


class _ScaleEncoder(nn.Module):
    """Transformer over downsampled metric sequence (Eq. (10))."""

    def __init__(self, d_in: int, d_model: int, nhead: int, nlayers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [n, t, d_in]
        if x.size(1) == 0:
            return torch.zeros(x.size(0), self.proj.out_features, device=x.device, dtype=x.dtype)
        h = self.proj(x)
        h = self.encoder(h)
        return h.mean(dim=1)


class MetricScaleformerEmbedding(nn.Module):
    """
    Multi-scale metric embedding (Eq. (9)–(11)).

    Each scale applies average pooling with stride 2^k followed by a shallow Transformer;
    fused with learned softmax weights w_j.
    """

    def __init__(self, cfg: TCRCAConfig, d_in: int):
        super().__init__()
        self.num_scales = cfg.num_scales
        self.encoders = nn.ModuleList(
            [
                _ScaleEncoder(
                    d_in=d_in,
                    d_model=cfg.metric_hidden,
                    nhead=cfg.metric_transformer_heads,
                    nlayers=cfg.metric_transformer_layers,
                    dropout=cfg.dropout,
                )
                for _ in range(self.num_scales)
            ]
        )
        self.fuse = nn.Linear(cfg.metric_hidden, cfg.metric_hidden)
        self.scale_logits = nn.Parameter(torch.zeros(self.num_scales))

    def _pool(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        # x: [n, T, D] -> downsample along T
        if x.size(1) <= 1:
            return x
        stride = 2**scale_idx
        # adaptive average pool to ceil(T/stride) using simple chunk pooling
        T = x.size(1)
        if stride >= T:
            return x.mean(dim=1, keepdim=True)
        # manual pooling for integer stride
        m = []
        for start in range(0, T, stride):
            chunk = x[:, start : start + stride, :]
            m.append(chunk.mean(dim=1, keepdim=True))
        return torch.cat(m, dim=1)

    def forward(self, metrics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused: [n, d_model]
            per_scale: [n, S, d_model] for MS-TGI reuse
        """
        per = []
        for s in range(self.num_scales):
            pooled = self._pool(metrics, s)
            per.append(self.encoders[s](pooled))
        H = torch.stack(per, dim=1)  # [n, S, d]
        w = torch.softmax(self.scale_logits, dim=0).view(1, self.num_scales, 1)
        fused = (w * H).sum(dim=1)
        fused = self.fuse(fused)
        return fused, H


class SpanMLPEmbedding(nn.Module):
    """Span embedding x_i^span = MLP(s_i) (Eq. (12))."""

    def __init__(self, cfg: TCRCAConfig, span_dim: int):
        super().__init__()
        h = cfg.span_mlp_hidden
        self.net = nn.Sequential(
            nn.Linear(span_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
        )

    def forward(self, span: torch.Tensor) -> torch.Tensor:
        return self.net(span)
