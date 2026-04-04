from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from tcrca.config import TCRCAConfig, TraceTensorBundle
from tcrca.fpg import FPGResult, build_fpg_result
from tcrca.gae import GraphAutoencoderTCRCA
from tcrca.mscg import MSCGRefiner
from tcrca.mstgi import MultiScaleTopologyGuidedGranger
from tcrca.tensorize import build_trace_tensor_bundle

logger = logging.getLogger(__name__)


@dataclass
class TCRCAOutput:
    trace_anomaly: float
    node_anomaly: torch.Tensor
    a_causal: torch.Tensor
    fpg: FPGResult
    bundle: TraceTensorBundle


class TCRCAPipeline(nn.Module):
    """
    Stages:
        1) MSCG — Sec. IV-B
        2) GAE anomaly scoring — Sec. IV-C
        3) MS-TGI — Sec. IV-D
        4) FPG + A-CPS — Sec. IV-E
    """

    def __init__(self, cfg: Optional[TCRCAConfig] = None):
        super().__init__()
        self.cfg = cfg or TCRCAConfig()
        self.mscg: Optional[MSCGRefiner] = None
        self.gae: Optional[GraphAutoencoderTCRCA] = None
        self.mstgi: Optional[MultiScaleTopologyGuidedGranger] = None
        self._inited = False

    def _build_from_bundle(self, bundle: TraceTensorBundle) -> None:
        d_metric = int(bundle.metrics.shape[-1])
        self.mscg = MSCGRefiner(self.cfg, metric_feature_dim=d_metric)
        gat_out = self.cfg.gat_heads * self.cfg.gat_hidden
        self.gae = GraphAutoencoderTCRCA(self.cfg, input_dim=gat_out)
        self.mstgi = MultiScaleTopologyGuidedGranger(self.cfg, scale_dim=self.cfg.metric_hidden)
        self._inited = True

    def forward_bundle(self, bundle: TraceTensorBundle) -> TCRCAOutput:
        if not self._inited:
            self._build_from_bundle(bundle)
        assert self.mscg is not None and self.gae is not None and self.mstgi is not None

        adj = bundle.adjacency
        h, _hg, h_scales = self.mscg(adj, bundle.metrics, bundle.log_event_times, bundle.span_features)
        _h_enc, a_hat, x_hat = self.gae(h, adj)
        a_causal = self.mstgi(h_scales, bundle.a_topo)

        node_scores = self.gae.node_anomaly_scores(h, adj, a_hat, x_hat)
        trace_score = float(node_scores.max().item())

        fpg = build_fpg_result(
            a_causal,
            node_scores,
            tau=self.cfg.tau_anomaly,
            theta=self.cfg.theta_causal,
            theta_a=self.cfg.acps_theta_a,
            theta_s=self.cfg.acps_theta_s,
            l_max=self.cfg.acps_l_max,
            acps_top_k=self.cfg.acps_top_k,
            root_top_k=self.cfg.root_cause_top_k,
        )

        node_scores = node_scores.detach()
        a_causal = a_causal.detach()

        return TCRCAOutput(
            trace_anomaly=trace_score,
            node_anomaly=node_scores,
            a_causal=a_causal,
            fpg=fpg,
            bundle=bundle,
        )

    def training_step(self, bundle: TraceTensorBundle) -> torch.Tensor:
        if not self._inited:
            self._build_from_bundle(bundle)
        assert self.mscg and self.gae and self.mstgi
        adj = bundle.adjacency
        h, _hg, h_scales = self.mscg(adj, bundle.metrics, bundle.log_event_times, bundle.span_features)
        _h_enc, a_hat, x_hat = self.gae(h, adj)
        l_gae = self.gae.reconstruction_loss(h, adj, a_hat, x_hat)
        a_causal = self.mstgi(h_scales, bundle.a_topo)
        l_causal, _parts = self.mstgi.losses(a_causal, bundle.a_topo, h_scales)
        return l_gae + l_causal

    def fit(self, bundle: TraceTensorBundle, epochs: int = 80, lr: float = 1e-3) -> None:
        if not self._inited:
            self._build_from_bundle(bundle)
        params: list[nn.Parameter] = []
        for m in (self.mscg, self.gae, self.mstgi):
            assert m is not None
            params += list(m.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        self.train()
        for ep in range(epochs):
            opt.zero_grad()
            loss = self.training_step(bundle)
            loss.backward()
            opt.step()
            if ep == 0 or ep == epochs - 1 or (ep + 1) % max(1, epochs // 5) == 0:
                logger.info("TCRCA train epoch %s loss=%.6f", ep + 1, float(loss.item()))
        self.eval()


def run_tcrca_from_tables(
    data: Dict[str, Any],
    cfg: Optional[TCRCAConfig] = None,
    *,
    train_epochs: int = 60,
    lr: float = 1e-3,
) -> TCRCAOutput:
    """
    High-level helper for pandas-based loaders (see ``tcrca.data``).
    """
    bundle = build_trace_tensor_bundle(data)
    pipe = TCRCAPipeline(cfg)
    pipe.fit(bundle, epochs=train_epochs, lr=lr)
    with torch.no_grad():
        return pipe.forward_bundle(bundle)
