"""
Offline evaluation driver: JSONL manifests + pandas observability tables.

See README for the recommended train-normal / test-fault protocol. By default each
manifest row instantiates a fresh model on that trace (sanity evaluation); for
paper-grade numbers you should add a checkpoint workflow or fixed-size subgraphs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from tcrca.config import TCRCAConfig
from tcrca.eval.metrics import (
    EvalRecord,
    aggregate_binary_alerts,
    edge_f1,
    edge_jaccard,
    mean_reciprocal_rank,
    paper_summary_metrics,
    predicted_edges_from_acps,
    root_hit_at_k,
    root_rank,
    table_vi_fpg_statistics,
)
from tcrca.pipeline import TCRCAOutput, TCRCAPipeline
from tcrca.tensorize import build_trace_tensor_bundle

logger = logging.getLogger(__name__)


def filter_tables_to_trace(data: Dict[str, pd.DataFrame], trace_id: str) -> Dict[str, pd.DataFrame]:
    """Restrict tables to one logical trace when ``trace_id`` columns exist."""
    out = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in data.items()}
    if "traces" in out and not out["traces"].empty and "trace_id" in out["traces"].columns:
        out["traces"] = out["traces"][out["traces"]["trace_id"] == trace_id]
    if "metrics" in out and not out["metrics"].empty and "trace_id" in out["metrics"].columns:
        out["metrics"] = out["metrics"][out["metrics"]["trace_id"] == trace_id]
    if "logs" in out and not out["logs"].empty and "trace_id" in out["logs"].columns:
        out["logs"] = out["logs"][out["logs"]["trace_id"] == trace_id]
    return out


def run_single_case(
    data: Dict[str, pd.DataFrame],
    *,
    cfg: TCRCAConfig,
    trace_id: Optional[str] = None,
    train_epochs: int = 40,
    lr: float = 1e-3,
) -> TCRCAOutput:
    """Fit (optional) and infer on a single trace window."""
    bundle = build_trace_tensor_bundle(data, trace_id=trace_id)
    pipe = TCRCAPipeline(cfg)
    if train_epochs > 0:
        pipe.fit(bundle, epochs=train_epochs, lr=lr)
    with torch.no_grad():
        return pipe.forward_bundle(bundle)


def _ranked_service_ids_fpg(out: TCRCAOutput, cfg: TCRCAConfig) -> List[Any]:
    """
    Root-cause ranking on the FPG candidate set V' using RScore (Eq. (32)–(33)).

    Services are ordered by descending ``root_scores`` on ``out.fpg.nodes`` only.
    """
    nodes = list(out.fpg.nodes)
    if not nodes:
        return _full_ranking_from_output(out, cfg)
    nodes.sort(key=lambda i: float(out.fpg.root_scores.get(int(i), 0.0)), reverse=True)
    return [out.bundle.node_ids[int(i)] for i in nodes]


def _full_ranking_from_output(out: TCRCAOutput, cfg: TCRCAConfig) -> List[Any]:
    """Fallback: Eq. (32)-style scores over all MSCG services (diagnostic / ablation)."""
    s = out.node_anomaly.detach().cpu().numpy().reshape(-1)
    a = out.a_causal.detach().cpu().numpy()
    n = len(out.bundle.node_ids)
    vanom = [j for j in range(n) if s[j] > cfg.tau_anomaly]
    if not vanom:
        vanom = list(range(n))
    scored = []
    for i in range(n):
        rscore = sum(float(a[i, j]) * float(s[j]) for j in vanom)
        scored.append((out.bundle.node_ids[i], rscore))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored]


def _eval_one_row(out: TCRCAOutput, row: Dict[str, Any], cfg: TCRCAConfig) -> EvalRecord:
    case_id = str(row.get("case_id", ""))
    true_root = row["root_service_id"]
    ranked_fpg = _ranked_service_ids_fpg(out, cfg)
    r = root_rank(ranked_fpg, true_root)
    rr = 1.0 / r if r else 0.0

    tv = table_vi_fpg_statistics(out.node_anomaly, out.fpg.acps_paths, cfg.tau_anomaly, top_paths_for_coverage=3)

    ej = eprec = erec = ef1 = None
    if row.get("ground_truth_edges"):
        gt_e = {tuple(e) for e in row["ground_truth_edges"]}
        pred_edges = predicted_edges_from_acps(out.fpg.acps_paths, out.bundle.node_ids)
        ej = edge_jaccard(pred_edges, gt_e)
        eprec, erec, ef1 = edge_f1(pred_edges, gt_e)

    extras: Dict[str, Any] = {
        "trace_anomaly": out.trace_anomaly,
        "ranking_scope": "FPG_V_prime_Eq33",
        **tv,
    }
    if row.get("include_full_mscg_ranking"):
        ranked_all = _full_ranking_from_output(out, cfg)
        extras["rank_full_mscg"] = root_rank(ranked_all, true_root)

    return EvalRecord(
        case_id=case_id,
        hit_at_1=root_hit_at_k(ranked_fpg, true_root, 1),
        hit_at_3=root_hit_at_k(ranked_fpg, true_root, 3),
        hit_at_5=root_hit_at_k(ranked_fpg, true_root, 5),
        rank=r,
        reciprocal_rank=rr,
        edge_jaccard=ej,
        edge_precision=eprec,
        edge_recall=erec,
        edge_f1=ef1,
        extras=extras,
    )


def evaluate_manifest(
    manifest_path: Path,
    data: Dict[str, pd.DataFrame],
    *,
    cfg: Optional[TCRCAConfig] = None,
    train_epochs: int = 40,
    lr: float = 1e-3,
) -> Tuple[List[EvalRecord], Dict[str, float]]:
    """
    Evaluate every JSONL record against observability tables ``data``.

    Each JSON line fields:
        - ``case_id`` (str)
        - ``trace_id`` (str, optional if data already single-trace)
        - ``root_service_id`` (str, matches ``service_id`` in spans)
        - ``ground_truth_edges`` (optional) list of [caller, callee] service ids
        - ``is_faulty_trace`` (optional bool) for alert aggregation
        - ``trace_anomaly_threshold`` (optional float) for predicted alert
    """
    cfg = cfg or TCRCAConfig()
    records: List[EvalRecord] = []
    alert_pairs: List[Tuple[bool, bool]] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = row.get("trace_id")
            case_data = filter_tables_to_trace(data, tid) if tid else data
            out = run_single_case(
                case_data,
                cfg=cfg,
                trace_id=tid,
                train_epochs=train_epochs,
                lr=lr,
            )
            records.append(_eval_one_row(out, row, cfg))

            if "is_faulty_trace" in row:
                thr = float(row.get("trace_anomaly_threshold", 0.0))
                pred_alert = out.trace_anomaly > thr
                alert_pairs.append((bool(row["is_faulty_trace"]), pred_alert))

    summary: Dict[str, float] = {}
    if records:
        summary.update(paper_summary_metrics(records, k_max=5))
        summary["hit@1"] = summary.get("A@1", 0.0)
        summary["hit@3"] = summary.get("A@3", 0.0)
        summary["hit@5"] = summary.get("A@5", 0.0)
        summary["MRR"] = mean_reciprocal_rank([r.rank for r in records])
        ej = [r.edge_jaccard for r in records if r.edge_jaccard is not None]
        if ej:
            summary["mean_edge_jaccard"] = float(sum(ej) / len(ej))
        ef = [r.edge_f1 for r in records if r.edge_f1 is not None]
        if ef:
            summary["mean_edge_f1"] = float(sum(ef) / len(ef))
        # Table VI–style batch means (Sec. V-E)
        keys = ("n_raw_alerts", "n_compact_paths", "compression_pct", "top3_path_coverage_pct")
        for key in keys:
            vals = [float(r.extras[key]) for r in records if key in r.extras]
            if vals:
                summary[f"mean_{key}"] = float(sum(vals) / len(vals))

    if alert_pairs:
        ys, ps = zip(*alert_pairs)
        conf = aggregate_binary_alerts(ys, ps)
        summary.update({f"alert_{k}": v for k, v in conf.items()})

    return records, summary
