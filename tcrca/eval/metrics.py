"""
Evaluation metrics aligned with TCRCA-style RCA experiments (Sec. V).

We report standard localization metrics (Top-k / MRR) and structural overlap for
causal paths / FPG edges, plus optional binary alert statistics when manifests
include fault labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class EvalRecord:
    """Per-instance metrics returned by the evaluation runner."""

    case_id: str
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    rank: Optional[int]
    reciprocal_rank: float
    edge_jaccard: Optional[float] = None
    edge_precision: Optional[float] = None
    edge_recall: Optional[float] = None
    edge_f1: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def root_hit_at_k(ranked_service_ids: Sequence[Any], true_service_id: Any, k: int) -> int:
    """1 if the ground-truth service appears in the first *k* predictions, else 0."""
    top = list(ranked_service_ids)[:k]
    return int(true_service_id in top)


def root_rank(ranked_service_ids: Sequence[Any], true_service_id: Any) -> Optional[int]:
    """1-based rank of ``true_service_id`` in ``ranked_service_ids``; ``None`` if absent."""
    for i, sid in enumerate(ranked_service_ids):
        if sid == true_service_id:
            return i + 1
    return None


def mean_reciprocal_rank(ranks: Sequence[Optional[int]]) -> float:
    """MRR over a list of 1-based ranks (missing rank contributes 0)."""
    if not ranks:
        return 0.0
    total = 0.0
    for r in ranks:
        if r is not None and r >= 1:
            total += 1.0 / r
    return total / len(ranks)


def rank_accuracy_curve(
    ranked_per_case: Sequence[Sequence[Any]],
    true_ids: Sequence[Any],
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, float]:
    """Mean hit@k for each k in ``ks``."""
    out: Dict[str, float] = {}
    n = len(true_ids)
    if n == 0:
        return {f"hit@{k}": 0.0 for k in ks}
    for k in ks:
        hits = sum(root_hit_at_k(ranked_per_case[i], true_ids[i], k) for i in range(n))
        out[f"hit@{k}"] = hits / n
    return out


def edge_jaccard(pred_edges: Set[Tuple[Any, Any]], gt_edges: Set[Tuple[Any, Any]]) -> float:
    if not pred_edges and not gt_edges:
        return 1.0
    inter = len(pred_edges & gt_edges)
    union = len(pred_edges | gt_edges)
    return float(inter / union) if union else 0.0


def edge_f1(pred_edges: Set[Tuple[Any, Any]], gt_edges: Set[Tuple[Any, Any]]) -> Tuple[float, float, float]:
    """Precision, recall, F1 over directed edge sets (service-id tuples)."""
    if not pred_edges and not gt_edges:
        return 1.0, 1.0, 1.0
    if not pred_edges:
        return 0.0, 0.0, 0.0
    if not gt_edges:
        return 0.0, 1.0, 0.0
    tp = len(pred_edges & gt_edges)
    prec = tp / len(pred_edges)
    rec = tp / len(gt_edges)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def aggregate_binary_alerts(
    y_faulty: Sequence[bool],
    predicted_alert: Sequence[bool],
) -> MutableMapping[str, float]:
    """
    Confusion matrix for trace-level anomaly screening (Task 1, Eq. (4)).

    ``y_faulty``: ground truth (True = injected / faulty trace).
    ``predicted_alert``: model declares anomaly when True.
    """
    tp = fp = tn = fn = 0
    for y, p in zip(y_faulty, predicted_alert):
        if y and p:
            tp += 1
        elif y and not p:
            fn += 1
        elif not y and p:
            fp += 1
        else:
            tn += 1
    n = len(y_faulty)
    out: Dict[str, float] = {
        "n": float(n),
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
    }
    out["precision_alert"] = tp / (tp + fp) if (tp + fp) else 0.0
    out["recall_alert"] = tp / (tp + fn) if (tp + fn) else 0.0
    out["fpr_alert"] = fp / (fp + tn) if (fp + tn) else 0.0
    out["fnr_alert"] = fn / (fn + tp) if (fn + tp) else 0.0
    return out


def predicted_edges_from_acps(
    acps_paths: Sequence[Tuple[float, Sequence[int]]],
    node_ids: Sequence[Any],
) -> Set[Tuple[Any, Any]]:
    edges: Set[Tuple[Any, Any]] = set()
    for _score, path in acps_paths:
        for u, v in zip(path[:-1], path[1:]):
            edges.add((node_ids[int(u)], node_ids[int(v)]))
    return edges


def hit_from_rank(rank: Optional[int], k: int) -> int:
    """Indicator I(rank ≤ k) for Eq. (40); rank is 1-based on the evaluation list."""
    return int(rank is not None and rank <= k)


def dataset_top_k_accuracy(records: Sequence[EvalRecord], k: int) -> float:
    """
    Top-k accuracy A@k (Eq. (40)): mean over cases of I(RC_a ∈ Top-k).
    Uses ``EvalRecord.rank`` (must match the paper's candidate ordering).
    """
    if not records:
        return 0.0
    return sum(hit_from_rank(r.rank, k) for r in records) / len(records)


def avg_top_k_accuracy(records: Sequence[EvalRecord], k_max: int = 5) -> float:
    """
    Average Top-k accuracy Avg@k (Eq. (41)): mean of A@1, …, A@k.
    Primary reporting uses k_max=5 as in the paper (Avg@5).
    """
    if not records or k_max < 1:
        return 0.0
    parts = [dataset_top_k_accuracy(records, i) for i in range(1, k_max + 1)]
    return sum(parts) / k_max


def _anomalous_node_indices(s: Any, tau: float) -> List[int]:
    arr = np.asarray(s, dtype=np.float64).reshape(-1)
    idx = [i for i, v in enumerate(arr) if v > tau]
    if not idx:
        idx = [int(np.argmax(arr))]
    return idx


def table_vi_fpg_statistics(
    node_anomaly: Any,
    acps_paths: Sequence[Tuple[float, Sequence[int]]],
    tau: float,
    top_paths_for_coverage: int = 3,
) -> Dict[str, float]:
    """
    Quantities analogous to Table VI (Sec. V-E): raw alerts, compact chains,
    compression rate, and Top-k path coverage over anomalous nodes.

    - *Avg. Alerts* proxy: count of nodes with s_i > τ (fallback: argmax if none).
    - *Avg. Chains* proxy: number of A-CPS paths returned (|P|).
    - *Compression (%)*: (n_raw − n_chains) / n_raw × 100 (0 if n_raw = 0).
    - *Top-k Cov. (%)*: fraction of anomalous nodes that lie on at least one of
      the top-``top_paths_for_coverage`` paths by path score.
    """
    s = np.asarray(node_anomaly.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    vanom = _anomalous_node_indices(s, tau)
    n_raw = len(vanom)
    vanom_set = set(vanom)
    n_chains = float(len(acps_paths))
    compression = float(100.0 * (n_raw - n_chains) / n_raw) if n_raw > 0 else 0.0

    sorted_paths = sorted(acps_paths, key=lambda x: x[0], reverse=True)
    covered: set[int] = set()
    for _sc, path in sorted_paths[:top_paths_for_coverage]:
        for nid in path:
            covered.add(int(nid))
    hit = len(vanom_set & covered)
    topk_cov = float(100.0 * hit / n_raw) if n_raw > 0 else 0.0

    return {
        "n_raw_alerts": float(n_raw),
        "n_compact_paths": n_chains,
        "compression_pct": compression,
        f"top{top_paths_for_coverage}_path_coverage_pct": topk_cov,
    }


def paper_summary_metrics(records: Sequence[EvalRecord], k_max: int = 5) -> Dict[str, float]:
    """Aggregate localization metrics using symbols from Sec. V-C (A@k, Avg@k, MRR)."""
    if not records:
        return {}
    out: Dict[str, float] = {
        "MRR": mean_reciprocal_rank([r.rank for r in records]),
    }
    for kk in (1, 3, 5):
        if kk <= k_max:
            out[f"A@{kk}"] = dataset_top_k_accuracy(records, kk)
    out[f"Avg@{k_max}"] = avg_top_k_accuracy(records, k_max)
    return out
