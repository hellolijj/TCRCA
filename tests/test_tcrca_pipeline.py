import numpy as np
import pandas as pd
import torch

from tcrca.config import TCRCAConfig
from tcrca.fpg import anomaly_aware_causal_path_search, build_failure_propagation_graph
from tcrca.pipeline import TCRCAPipeline
from tcrca.tensorize import build_trace_tensor_bundle


def _tiny_dataframe_bundle():
    traces = pd.DataFrame(
        {
            "trace_id": ["t1", "t1", "t1"],
            "span_id": ["s1", "s2", "s3"],
            "parent_id": ["root", "s1", "s1"],
            "service_id": ["A", "B", "C"],
            "start_time": pd.date_range("2024-01-01", periods=3, freq="s"),
            "end_time": pd.date_range("2024-01-01", periods=3, freq="s") + pd.Timedelta(seconds=1),
            "duration": [1.0, 1.0, 1.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "service_id": ["A", "B", "C"] * 4,
            "timestamp": pd.date_range("2024-01-01", periods=12, freq="s"),
            "cpu": np.random.randn(12).astype("float32"),
            "mem": np.random.randn(12).astype("float32"),
        }
    )
    logs = pd.DataFrame(
        {
            "service_id": ["A", "B", "C", "A"],
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="500ms"),
            "message": ["x", "y", "z", "w"],
        }
    )
    return {"traces": traces, "metrics": metrics, "logs": logs}


def test_tensorize_and_forward_smoke():
    data = _tiny_dataframe_bundle()
    bundle = build_trace_tensor_bundle(data, max_metric_steps=8, max_log_events=8)
    assert bundle.adjacency.shape[0] == 3
    cfg = TCRCAConfig()
    cfg.gae_gcn_layers = 1
    cfg.gat_layers = 1
    cfg.metric_transformer_layers = 1
    pipe = TCRCAPipeline(cfg)
    pipe.fit(bundle, epochs=2, lr=1e-2)
    out = pipe.forward_bundle(bundle)
    assert out.node_anomaly.shape[0] == 3
    assert out.a_causal.shape == (3, 3)
    assert len(out.fpg.top_roots) <= cfg.root_cause_top_k


def test_acps_runs():
    a = torch.tensor([[0.0, 0.8, 0.0], [0.0, 0.0, 0.6], [0.0, 0.0, 0.0]])
    s = torch.tensor([0.1, 0.9, 0.7])
    paths = anomaly_aware_causal_path_search(a, s, v_obs=1, theta_a=0.1, theta_s=0.05, l_max=4, top_k=3)
    assert isinstance(paths, list)


def test_fpg_construction():
    a = torch.tensor([[0.0, 0.8, 0.0], [0.0, 0.0, 0.6], [0.0, 0.0, 0.0]])
    s = torch.tensor([0.1, 0.9, 0.7])
    vp, ep, _ = build_failure_propagation_graph(a, s, tau=0.2, theta=0.1)
    assert len(vp) >= 1
    assert isinstance(ep, list)
