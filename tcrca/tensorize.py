from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from tcrca.config import TraceTensorBundle

logger = logging.getLogger(__name__)


def _numeric_metric_block(df: pd.DataFrame) -> pd.DataFrame:
    drop = {"service_id", "timestamp", "trace_id", "span_id"}
    cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return pd.DataFrame()
    return df[cols]


def build_trace_tensor_bundle(
    data: Dict[str, Any],
    max_metric_steps: int = 32,
    max_log_events: int = 64,
    trace_id: Optional[str] = None,
) -> TraceTensorBundle:
    """
    Build a single-trace MSCG tensor bundle from merged observability tables.

    If ``trace_id`` is set, keep only that trace; otherwise use the most frequent ``trace_id``.
    When ``metrics`` / ``logs`` contain a ``trace_id`` column, rows are filtered to match.
    """
    traces = data.get("traces")
    if traces is None or traces.empty:
        raise ValueError("traces DataFrame is required and must be non-empty")

    metrics = data.get("metrics", pd.DataFrame())
    logs = data.get("logs", pd.DataFrame())

    tdf = traces.copy()
    if "trace_id" in tdf.columns and tdf["trace_id"].notna().any():
        if trace_id is not None:
            tdf = tdf[tdf["trace_id"] == trace_id]
            if tdf.empty:
                raise ValueError(f"No spans for trace_id={trace_id!r}")
        else:
            tid = tdf["trace_id"].value_counts().idxmax()
            tdf = tdf[tdf["trace_id"] == tid]

    sel_tid = None
    if "trace_id" in tdf.columns and tdf["trace_id"].notna().any():
        sel_tid = tdf["trace_id"].iloc[0]

    if sel_tid is not None and not metrics.empty and "trace_id" in metrics.columns:
        metrics = metrics[metrics["trace_id"] == sel_tid]
    if sel_tid is not None and not logs.empty and "trace_id" in logs.columns:
        logs = logs[logs["trace_id"] == sel_tid]

    # Stable node order: first appearance in trace table
    node_list: List = []
    seen = set()
    for sid in tdf["service_id"].tolist():
        if sid not in seen:
            seen.add(sid)
            node_list.append(sid)
    n = len(node_list)
    if n == 0:
        raise ValueError("No services found in traces")

    idx = {sid: i for i, sid in enumerate(node_list)}

    adj = torch.zeros((n, n), dtype=torch.float32)
    for _, row in tdf.iterrows():
        sid = row["service_id"]
        if sid not in idx:
            continue
        j = idx[sid]
        parent = row.get("parent_id")
        if pd.notna(parent) and parent != "root" and parent in idx:
            i = idx[parent]
            adj[i, j] = 1.0

    # Self-loops stabilize GAT/GCN when a service has no callers in this trace window.
    adj = adj + torch.eye(n, dtype=torch.float32)
    a_topo = adj.clone()

    # Metrics [n, T, D]
    D = 1
    if not metrics.empty and "service_id" in metrics.columns:
        mdf = metrics.copy()
        if "timestamp" in mdf.columns:
            mdf = mdf.sort_values("timestamp")
        num_df = _numeric_metric_block(mdf)
        D = max(1, num_df.shape[1])
    M = torch.zeros((n, max_metric_steps, D), dtype=torch.float32)
    if not metrics.empty and "service_id" in metrics.columns:
        for sid, i in idx.items():
            sdf = metrics[metrics["service_id"] == sid]
            if sdf.empty:
                continue
            sdf = sdf.sort_values("timestamp") if "timestamp" in sdf.columns else sdf
            block = _numeric_metric_block(sdf)
            if block.empty:
                continue
            arr = block.to_numpy(dtype=np.float32)
            arr = arr[-max_metric_steps:]
            tlen, dlen = arr.shape
            d_use = min(D, dlen)
            M[i, :tlen, :d_use] = torch.from_numpy(arr[:, :d_use])

    # Log event times (normalized per service), padded with -1
    log_t = torch.full((n, max_log_events), -1.0, dtype=torch.float32)
    if not logs.empty and "service_id" in logs.columns and "timestamp" in logs.columns:
        ldf = logs.copy().sort_values("timestamp")
        for sid, i in idx.items():
            lsub = ldf[ldf["service_id"] == sid]["timestamp"]
            if lsub.empty:
                continue
            ts = pd.to_datetime(lsub, errors="coerce").dropna()
            if ts.empty:
                continue
            tsec = ts.astype("int64").to_numpy(dtype=np.float64) / 1e9
            tsec = tsec[-max_log_events:]
            tsec = tsec - tsec.min()
            denom = float(tsec.max()) if float(tsec.max()) > 0 else 1.0
            tsec = tsec / denom
            le = min(len(tsec), max_log_events)
            log_t[i, :le] = torch.tensor(tsec[:le], dtype=torch.float32)

    # Span features: duration, depth proxy, error flag
    span_dim = 3
    spans = torch.zeros((n, span_dim), dtype=torch.float32)
    depth_map: Dict[Any, int] = {}

    def depth_of(pid) -> int:
        if pid is None or (isinstance(pid, float) and np.isnan(pid)) or pid == "root":
            return 0
        if pid not in depth_map:
            depth_map[pid] = 0
        return depth_map.get(pid, 0)

    for _, row in tdf.iterrows():
        sid = row["service_id"]
        if sid not in idx:
            continue
        i = idx[sid]
        st = row.get("start_time")
        et = row.get("end_time")
        dur = row.get("duration")
        try:
            if dur is None or (isinstance(dur, float) and np.isnan(dur)) and pd.notna(st) and pd.notna(et):
                stp = pd.to_datetime(st, errors="coerce")
                etp = pd.to_datetime(et, errors="coerce")
                if pd.notna(stp) and pd.notna(etp):
                    dur = (etp - stp).total_seconds()
        except Exception:
            dur = 0.0
        dur = float(dur) if dur is not None and not (isinstance(dur, float) and np.isnan(dur)) else 0.0
        pid = row.get("parent_id")
        d = depth_of(pid) + 1
        depth_map[row.get("span_id", sid)] = d
        err = 0.0
        if "error" in row and pd.notna(row["error"]):
            err = float(bool(row["error"]))
        elif "status" in row and str(row["status"]).upper() not in ("OK", "0", "SUCCESS"):
            err = 1.0
        spans[i, 0] = dur
        spans[i, 1] = float(d)
        spans[i, 2] = err

    return TraceTensorBundle(
        adjacency=adj,
        a_topo=a_topo,
        metrics=M,
        log_event_times=log_t,
        span_features=spans,
        node_ids=node_list,
    )
