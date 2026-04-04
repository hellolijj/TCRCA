"""
Failure Propagation Graph (FPG) construction and Anomaly-aware Causal Path Search (A-CPS).

Sec. IV-E, Eq. (29)–(37) and Algorithm 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import heapq

import networkx as nx
import numpy as np
import torch


@dataclass
class FPGResult:
    nodes: List[int]
    edges: List[Tuple[int, int]]
    adjacency: np.ndarray
    root_scores: Dict[int, float]
    top_roots: List[int]
    acps_paths: List[Tuple[float, List[int]]]
    refined_graph: nx.DiGraph


def build_failure_propagation_graph(
    a_causal: torch.Tensor,
    anomaly_scores: torch.Tensor,
    tau: float,
    theta: float,
) -> Tuple[List[int], List[Tuple[int, int]], np.ndarray]:
    """
    Eqs. (29)–(31): GFPG = (V', E') from A_causal and anomaly scores s.
    """
    a = a_causal.detach().cpu().numpy()
    s = anomaly_scores.detach().cpu().numpy().reshape(-1)
    n = s.shape[0]
    vanom = {i for i in range(n) if s[i] > tau}
    if not vanom:
        vanom = {int(np.argmax(s))}

    extra = {i for i in range(n) if any(a[i, j] > theta for j in vanom)}
    v_prime = sorted(vanom | extra)

    e_prime: List[Tuple[int, int]] = []
    for i in range(n):
        for j in vanom:
            if a[i, j] > theta:
                e_prime.append((i, j))

    sub = np.zeros((n, n), dtype=np.float32)
    for i, j in e_prime:
        sub[i, j] = a[i, j]
    return v_prime, e_prime, sub


def root_cause_scores(
    a_causal: torch.Tensor,
    anomaly_scores: torch.Tensor,
    node_indices: Sequence[int],
    tau: float,
) -> Dict[int, float]:
    """Eq. (32): RScore_i = sum_{j in V_anom} A_{ij} * s_j (V_anom uses threshold τ)."""
    a = a_causal.detach().cpu().numpy()
    s = anomaly_scores.detach().cpu().numpy().reshape(-1)
    vanom = [j for j in range(len(s)) if s[j] > tau]
    if not vanom:
        vanom = [int(np.argmax(s))]
    out: Dict[int, float] = {}
    for i in node_indices:
        out[int(i)] = float(sum(a[i, j] * s[j] for j in vanom))
    return out


def anomaly_aware_causal_path_search(
    a_causal: torch.Tensor,
    anomaly_scores: torch.Tensor,
    v_obs: int,
    theta_a: float,
    theta_s: float,
    l_max: int,
    top_k: int,
) -> List[Tuple[float, List[int]]]:
    """
    Algorithm 1 (A-CPS): backward expansion from observed anomaly v_obs.

    Path score Eq. (34): sum_{(u,v) in P} A_{uv} * s_v.
    """
    a = a_causal.detach().cpu().numpy()
    s = anomaly_scores.detach().cpu().numpy().reshape(-1)
    n = s.shape[0]

    # Priority queue stores (-score, path) for max-first using min-heap
    q: List[Tuple[float, List[int]]] = []
    heapq.heappush(q, (0.0, [v_obs]))
    results: List[Tuple[float, List[int]]] = []

    seen_best: Dict[Tuple[int, ...], float] = {}

    while q and len(results) < top_k:
        neg_score, path = heapq.heappop(q)
        score = -neg_score
        tup = tuple(path)
        if tup in seen_best and seen_best[tup] >= score:
            continue
        seen_best[tup] = score

        if 1 < len(path) <= l_max:
            results.append((score, list(path)))

        if len(path) >= l_max:
            continue

        v = path[0]
        extended = False
        for u in range(n):
            if a[u, v] < theta_a or s[u] < theta_s:
                continue
            new_score = score + float(a[u, v] * s[v])
            new_path = [u] + path
            heapq.heappush(q, (-new_score, new_path))
            extended = True

        if not extended and len(path) == 1:
            # keep trivial path if nothing expands
            results.append((score, list(path)))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def refine_fpg_from_paths(
    paths: Sequence[Tuple[float, Sequence[int]]],
    a_causal: torch.Tensor,
    anomaly_scores: torch.Tensor,
) -> nx.DiGraph:
    """
    Eqs. (35)–(37): aggregate nodes/edges from P_final with frequency and influence weights.
    """
    a = a_causal.detach().cpu().numpy()
    s = anomaly_scores.detach().cpu().numpy().reshape(-1)
    g = nx.DiGraph()
    edge_freq: Dict[Tuple[int, int], int] = {}
    edge_inf: Dict[Tuple[int, int], float] = {}

    for _, p in paths:
        for v in p:
            g.add_node(int(v))
        for u, v in zip(p[:-1], p[1:]):
            e = (int(u), int(v))
            edge_freq[e] = edge_freq.get(e, 0) + 1
            edge_inf[e] = edge_inf.get(e, 0.0) + float(a[u, v] * s[v])

    for e, freq in edge_freq.items():
        g.add_edge(e[0], e[1], frequency=freq, influence=edge_inf.get(e, 0.0))
    return g


def build_fpg_result(
    a_causal: torch.Tensor,
    anomaly_scores: torch.Tensor,
    *,
    tau: float,
    theta: float,
    theta_a: float,
    theta_s: float,
    l_max: int,
    acps_top_k: int,
    root_top_k: int,
    v_obs: int | None = None,
) -> FPGResult:
    v_prime, e_prime, sub = build_failure_propagation_graph(a_causal, anomaly_scores, tau, theta)
    if v_obs is None:
        v_obs = int(torch.argmax(anomaly_scores).item())
    paths = anomaly_aware_causal_path_search(
        a_causal,
        anomaly_scores,
        v_obs=v_obs,
        theta_a=theta_a,
        theta_s=theta_s,
        l_max=l_max,
        top_k=acps_top_k,
    )
    refined = refine_fpg_from_paths(paths, a_causal, anomaly_scores)
    rscores = root_cause_scores(a_causal, anomaly_scores, v_prime, tau=tau)
    ranked = sorted(rscores.items(), key=lambda kv: kv[1], reverse=True)
    top_nodes = [nid for nid, _ in ranked[:root_top_k]]
    return FPGResult(
        nodes=v_prime,
        edges=e_prime,
        adjacency=sub,
        root_scores=rscores,
        top_roots=top_nodes,
        acps_paths=paths,
        refined_graph=refined,
    )
