"""Source/sink (rho) injection (blueprint 2.5 / spec 1.4).

Structural bias applied to TARGET (U) movement counts only; neighbor-observed link
volumes stay clean. This breaks flow conservation exactly the way real mid-block
source/sink (driveways, parking) does, so a conservation-assuming estimator degrades
as rho grows.

rho = fractional net source/sink severity. sign = +1 (net inflow) / -1 (net outflow).
Bias is drawn once per intersection (structural, shared across that intersection's
movements) and accumulated along corridor order for multi-U configs.
"""
from __future__ import annotations


import os
import numpy as np

from ..graph.movement_graph import MovementGraph

def _routing(scale="5by5"):
    """Location of the SUMO networks written by od_gen; set TMR_ROUTING to override."""
    return os.path.join(os.environ.get("TMR_ROUTING", os.path.join(os.getcwd(), "routing")), scale)



def inject_rho(y_clean: np.ndarray,
               u_mask: np.ndarray,
               mg: MovementGraph,
               rho: float,
               sign: int,
               rng: np.random.Generator,
               sigma_frac: float = 0.25,
               cumulative: bool = True) -> np.ndarray:
    """Return injected counts. Only U nodes are modified.

    y_inj[m] = y_clean[m] * (1 + sign * bias_I)   for m at U intersection I
    cumulative=True : bias accumulates along corridor order (blueprint 2.5, for LINEAR
                      corridors). NOTE: over a large 2D block this compounds to ~k*rho and
                      explodes -- use cumulative=False for block clusters.
    cumulative=False: each U intersection gets an independent bias ~ N(rho, sigma*rho)
                      (per-intersection source/sink severity; interpretable for blocks).
    """
    y = y_clean.astype(float).copy()
    if rho <= 0:
        return y

    net = mg.net
    u_inters = sorted({net.movements[i].intersection for i in np.nonzero(u_mask)[0]},
                      key=lambda j: (net.junctions[j].x, net.junctions[j].y))
    offset = 0.0
    bias_of_inter: dict[str, float] = {}
    for j in u_inters:
        delta = rng.normal(rho, sigma_frac * rho)
        if cumulative:
            offset += delta
            bias_of_inter[j] = offset
        else:
            bias_of_inter[j] = delta

    for i in np.nonzero(u_mask)[0]:
        j = net.movements[i].intersection
        y[i] = y_clean[i] * (1.0 + sign * bias_of_inter[j])
    return np.clip(y, 0.0, None)


def inject_rho_splitbreak(y_clean, u_mask, mg, rho, sign, rng, sigma_frac=0.25):
    """Split-BREAKING source/sink: unlike inject_rho (one factor per site, ratios preserved),
    the per-site bias is modulated by turn type (L 1.5x, T 1.0x, R 0.5x), so the intersection's
    turn split changes, violating the split-preserving assumption of Eq. (4). Tests whether the
    regime characterization survives when the bias reshapes splits rather than scaling them."""
    y = y_clean.astype(float).copy()
    if rho <= 0:
        return y
    net = mg.net
    tw = {"l": 1.5, "s": 1.0, "r": 0.5}          # left 1.5x, through(straight) 1.0x, right 0.5x
    u_inters = sorted({net.movements[i].intersection for i in np.nonzero(u_mask)[0]},
                      key=lambda j: (net.junctions[j].x, net.junctions[j].y))
    bias = {j: rng.normal(rho, sigma_frac * rho) for j in u_inters}
    for i in np.nonzero(u_mask)[0]:
        m = net.movements[i]
        w = tw.get(getattr(m, "dir", "s"), 1.0)   # turn-type modulation breaks the split
        y[i] = y_clean[i] * (1.0 + sign * w * bias[m.intersection])
    return np.clip(y, 0.0, None)


def inject_persistent(y_clean_all: np.ndarray, net, rho: float, sign: int, seed: int,
                      sigma_site: float = 0.3, sigma_scen: float = 0.15) -> np.ndarray:
    """Persistent per-SITE source/sink present in EVERY scenario (so a site's history is
    source/sink-calibrated). Each intersection I gets a fixed bias level b_I ~ |N(rho,
    sigma_site*rho)| plus small per-scenario noise; all its movements scale by (1+sign*bias)
    (split ratios preserved). Turning counts inflate; link volumes stay clean (source/sink
    sits between the upstream sensor and the intersection). Returns Y_real [S, N]."""
    if rho <= 0:
        return y_clean_all.astype(float).copy()
    rng = np.random.default_rng(seed)
    S, N = y_clean_all.shape
    b = {j: abs(rng.normal(rho, sigma_site * rho)) for j in net.intersections}
    Y = y_clean_all.astype(float).copy()
    for i, m in enumerate(net.movements):
        bias = b.get(m.intersection, 0.0) + rng.normal(0.0, sigma_scen * rho, size=S)
        Y[:, i] = y_clean_all[:, i] * (1.0 + sign * bias)
    return np.clip(Y, 0.0, None)


if __name__ == "__main__":
    from ..graph.net_parser import parse_net
    from ..graph.movement_graph import build_movement_graph, assign_ou
    from ..graph.labels import extract_counts, counts_to_arrays

    base = _routing()
    net = parse_net(f"{base}/5by5.net.xml")
    mg = build_movement_graph(net)
    move, _ = extract_counts(f"{base}/simData/vehRouteData/Route1704000479d1.xml", net)
    y = counts_to_arrays(net, move)

    inters = sorted(net.intersections)
    um = assign_ou(net, {inters[len(inters)//2]})
    rng = np.random.default_rng(0)
    for rho in (0.0, 0.2, 0.4, 0.8):
        yi = inject_rho(y, um, mg, rho, sign=+1, rng=np.random.default_rng(0))
        d = yi[um] - y[um]
        print(f"rho={rho:.1f}: mean|delta| on U = {np.abs(d).mean():.1f} "
              f"(mean clean U count = {y[um].mean():.1f})")
