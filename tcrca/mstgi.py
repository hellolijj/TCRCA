"""
Multi-Scale Topology-Guided Granger Inference (MS-TGI, Sec. IV-D, Eq. (24)–(28)).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcrca.config import TCRCAConfig


class MultiScaleTopologyGuidedGranger(nn.Module):
    """
    Computes A_causal via scale-wise predictive attention with topology biasing (Eq. (24)–(27)).

    Attention is normalized over targets j for each source i (softmax along columns), matching
    ``softmax_j`` in the manuscript for e_{ij}^{(s)}.
    """

    def __init__(self, cfg: TCRCAConfig, scale_dim: int):
        super().__init__()
        self.cfg = cfg
        self.scale_logits = nn.Parameter(torch.zeros(cfg.num_scales))
        in_dim = 2 * scale_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.granger_mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.granger_mlp_hidden, 1),
        )

    def forward(self, h_scales: torch.Tensor, a_topo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_scales: [n, S, d]
            a_topo: [n, n] {0,1}
        Returns:
            A_causal: [n, n] mixed attention weights (rows sum to 1).
        """
        n, num_s, d = h_scales.shape
        gamma = self.cfg.gamma_topo_prior
        outs = []
        for s in range(num_s):
            hi = h_scales[:, s, :].unsqueeze(1).expand(n, n, d)
            hj = h_scales[:, s, :].unsqueeze(0).expand(n, n, d)
            pair = torch.cat([hi, hj], dim=-1)
            e = self.edge_mlp(pair).squeeze(-1)
            boosted = (1.0 + gamma * a_topo) * e
            alpha = torch.softmax(boosted, dim=1)
            outs.append(alpha)
        w = torch.softmax(self.scale_logits, dim=0).view(num_s, 1, 1)
        stacked = torch.stack(outs, dim=0)
        a_causal = (w * stacked).sum(dim=0)
        return a_causal

    def losses(
        self,
        a_causal: torch.Tensor,
        a_topo: torch.Tensor,
        h_scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        L_causal = L_task + λ1 L_sparse + λ2 L_topo (Eq. (28)).

        L_task: self-supervised prediction ||H - A_causal^T H||_F^2 (Granger-style influence).
        """
        l_sparse = a_causal.abs().sum()
        l_topo = ((a_causal - a_topo) ** 2).sum()

        h = h_scales.mean(dim=1)
        # If A_{ij} is influence i -> j, target j is predicted from sum_i A_{ij} h_i.
        pred = torch.matmul(a_causal.t(), h)
        l_pred = F.mse_loss(pred, h)

        total = (
            self.cfg.lambda_pred * l_pred
            + self.cfg.lambda_sparse * l_sparse
            + self.cfg.lambda_topo * l_topo
        )
        parts = {"l_pred": l_pred.detach(), "l_sparse": l_sparse.detach(), "l_topo": l_topo.detach()}
        return total, parts
