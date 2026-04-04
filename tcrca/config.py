"""
Hyper-parameters aligned with TCRCA (Topology-Guided Causal Inference for RCA).

Defaults follow the manuscript: Eq. (21) weights, Eq. (22) anomaly mixing, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TCRCAConfig:
    # --- MSCG (Sec. IV-B) ---
    hawkes_bins: int = 8  # L in Eq. (8)
    metric_hidden: int = 64
    metric_transformer_layers: int = 2
    metric_transformer_heads: int = 4
    span_mlp_hidden: int = 32
    gat_layers: int = 2
    gat_heads: int = 4
    gat_hidden: int = 64
    dropout: float = 0.1

    # --- GAE anomaly detection (Sec. IV-C) ---
    gae_gcn_layers: int = 2
    gae_hidden: int = 128
    lambda_structure: float = 0.7  # λ_s in Eq. (21)
    lambda_feature: float = 0.3  # λ_f in Eq. (21)
    anomaly_alpha: float = 0.6  # α in Eq. (22)

    # --- MS-TGI (Sec. IV-D) ---
    num_scales: int = 4  # |S| (paper uses m = floor(log_s T); we cap scales)
    granger_mlp_hidden: int = 128
    gamma_topo_prior: float = 0.5  # γ in Eq. (26)
    lambda_sparse: float = 1e-3
    lambda_topo: float = 1e-2
    lambda_pred: float = 1.0  # self-supervised Granger-style prediction (L_task)

    # --- FPG / A-CPS (Sec. IV-E) ---
    tau_anomaly: float = 0.3
    theta_causal: float = 0.15
    acps_theta_a: float = 0.15
    acps_theta_s: float = 0.2
    acps_l_max: int = 6
    acps_top_k: int = 8
    root_cause_top_k: int = 5


@dataclass
class TraceTensorBundle:
    """Batched tensors for one trace-level MSCG instance (batch size = 1)."""

    adjacency: object  # torch.Tensor [n, n]
    a_topo: object  # torch.Tensor [n, n]
    metrics: object  # torch.Tensor [n, T, D]
    log_event_times: object  # torch.Tensor [n, E_max] (normalized); pad -1 for none
    span_features: object  # torch.Tensor [n, Sdim]
    node_ids: list = field(default_factory=list)
