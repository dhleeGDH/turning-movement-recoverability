"""Joint conservation estimator (BJ) over a missing cluster.

De-risk for model G: does estimating the WHOLE cluster jointly (using every observed
boundary link volume + internal flow-consistency) recover skill on h>=2 interior nodes,
where the per-node anchor B gives ~0? If a plain linear solve already lifts h>=2 above
zero, the gap is recoverable in principle and GraphPFN/cluster-attention is worth building.

BJ uses strictly more information than B:
  - B: observed APPROACH volume x historical split, per node (approach-only, point est).
  - BJ: observed approach AND exit boundary volumes as constraints, PLUS internal-link
        flow consistency, solved jointly with a historical prior as regulariser.

Constraints on cluster movement counts x (all conservation is exact at rho=0):
  approach:  for a cluster (intersection, from-link fl) group, if fl is observed
             (upstream O / boundary):  sum_g x = V_fl
  exit:      for a cluster (intersection, to-link el) group, if el is observed
             (downstream O / boundary): sum_g x = V_el
  internal:  for a link e with both endpoints U: sum(to_e at upstream) = sum(from_e at downstream)
Solved as augmented least squares: min || sqrt(lam)(x - x0) ; W(Cx - d) ||^2, x >= 0.
"""
from __future__ import annotations


from collections import defaultdict

import numpy as np

from ..graph.net_parser import NetworkData
from ..graph.observe import link_observed


def build_edge_maps(net: NetworkData):
    from_map: dict[str, list[int]] = defaultdict(list)   # edge -> movements with from_edge==edge
    for i, m in enumerate(net.movements):
        from_map[m.from_edge].append(i)
    return {k: np.array(v) for k, v in from_map.items()}


def link_volume(edge: str, Y_s: np.ndarray, from_map) -> float:
    """Total clean volume on a link = sum of movements leaving it (conservation-exact)."""
    idx = from_map.get(edge)
    return float(Y_s[idx].sum()) if idx is not None and len(idx) else 0.0


def estimate_joint(net: NetworkData, U: set[str], u_idx: np.ndarray,
                   Y_s_clean: np.ndarray, x0_full: np.ndarray, from_map,
                   lam: float = 0.05, w: float = 1.0) -> np.ndarray:
    """Return estimated counts for the cluster movements (order = u_idx)."""
    n = len(u_idx)
    loc = {g: k for k, g in enumerate(u_idx)}

    groups_from: dict[tuple, list[int]] = defaultdict(list)
    groups_to: dict[tuple, list[int]] = defaultdict(list)
    for g in u_idx:
        m = net.movements[g]
        groups_from[(m.intersection, m.from_edge)].append(g)
        groups_to[(m.intersection, m.to_edge)].append(g)

    rows, d = [], []

    # observed approach constraints (upstream O / boundary)
    for (I, fl), members in groups_from.items():
        if link_observed(net, fl, U):
            r = np.zeros(n)
            for gm in members:
                r[loc[gm]] = 1.0
            rows.append(r); d.append(link_volume(fl, Y_s_clean, from_map))

    # observed exit constraints (downstream O / boundary)
    for (I, el), members in groups_to.items():
        jd = net.links[el].to_j
        jdj = net.junctions.get(jd)
        exit_obs = (jdj is None) or (not jdj.is_intersection) or (jd not in U)
        if exit_obs:
            r = np.zeros(n)
            for gm in members:
                r[loc[gm]] = 1.0
            rows.append(r); d.append(link_volume(el, Y_s_clean, from_map))

    # internal-link consistency: e with both endpoint intersections in U
    for eid, link in net.links.items():
        if link.from_j in U and link.to_j in U:
            up = [g for g in groups_to.get((link.from_j, eid), [])]      # to_e at upstream
            dn = [g for g in groups_from.get((link.to_j, eid), [])]      # from_e at downstream
            if up and dn:
                r = np.zeros(n)
                for gm in up:
                    r[loc[gm]] += 1.0
                for gm in dn:
                    r[loc[gm]] -= 1.0
                rows.append(r); d.append(0.0)

    x0 = x0_full[u_idx].astype(float)
    C = np.array(rows) if rows else np.zeros((0, n))
    d = np.array(d)
    A_aug = np.vstack([np.sqrt(lam) * np.eye(n), w * C])
    b_aug = np.concatenate([np.sqrt(lam) * x0, w * d])
    x, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    return np.clip(x, 0.0, None)
