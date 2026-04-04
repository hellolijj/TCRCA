"""
Multi-Modal Service Call Graph (MSCG) construction (Sec. IV-B, Fig. 2 stage 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tcrca.config import TCRCAConfig
from tcrca.embeddings import MetricScaleformerEmbedding, NeuralLogHawkesEmbedding, SpanMLPEmbedding
from tcrca.gat import GATStack


class MSCGRefiner(nn.Module):
    """
    Builds unified node embeddings (Eq. (13)) and refines them with GAT (Eq. (14)–(17)).
    """

    def __init__(self, cfg: TCRCAConfig, metric_feature_dim: int, span_dim: int = 3):
        super().__init__()
        self.cfg = cfg
        self.hawkes = NeuralLogHawkesEmbedding(cfg.hawkes_bins)
        self.metric = MetricScaleformerEmbedding(cfg, d_in=metric_feature_dim)
        self.span = SpanMLPEmbedding(cfg, span_dim=span_dim)
        fuse_dim = cfg.hawkes_bins + cfg.metric_hidden + cfg.span_mlp_hidden
        self.pre_gat = nn.LayerNorm(fuse_dim)
        self.gat = GATStack(cfg, in_dim=fuse_dim)

    def forward(
        self,
        adjacency: torch.Tensor,
        metrics: torch.Tensor,
        log_event_times: torch.Tensor,
        span_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_nodes: contextualized node embeddings [n, d_gat]
            h_trace: trace-level vector [1, d_gat]
            metric_scales: [n, S, d_metric] for MS-TGI reuse
        """
        x_log = self.hawkes(log_event_times)
        x_met, h_scales = self.metric(metrics)
        x_span = self.span(span_features)
        x = torch.cat([x_log, x_met, x_span], dim=-1)
        x = self.pre_gat(x)
        h, hg = self.gat(x, adjacency)
        return h, hg, h_scales
