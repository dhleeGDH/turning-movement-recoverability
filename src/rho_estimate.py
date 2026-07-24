"""Empirical rho estimator for real turning-movement data (objective 3), + synthetic validation.

Method (blueprint 4): source/sink on an internal link L (J1 -> J2) shows up as a mismatch
between what the UPSTREAM intersection J1 sends onto L and what the DOWNSTREAM intersection
J2 receives from L. From a from/to/count table:
    upstream_exit(L)      = sum of counts at J1 with to_edge == L
    downstream_approach(L)= sum of counts at J2 with from_edge == L
    rho_hat(L) = downstream_approach / upstream_exit - 1     (net source +, sink -)
Clean/conserved flow => rho_hat = 0. This needs turning counts at ADJACENT intersections
sharing L (e.g. Bucheon 71572's 9+3-intersection layout).

Real Bucheon data is not on disk yet; this module (a) implements the estimator against the
from/to/count schema, (b) validates recovery of a known injected rho on synthetic, and
(c) provides an overlay hook for the operating-envelope figure.
"""

import numpy as np

from .graph.net_parser import NetworkData, parse_net
from .pilot import load_scenarios, net_path


def internal_links(net: NetworkData):
    """Links whose both endpoints are intersections (have an upstream & downstream turn set)."""
    out = []
    for eid, lk in net.links.items():
        j1, j2 = net.junctions.get(lk.from_j), net.junctions.get(lk.to_j)
        if j1 is not None and j2 is not None and j1.is_intersection and j2.is_intersection:
            out.append(eid)
    return out


def estimate_rho_per_link(net: NetworkData, counts: np.ndarray, links=None):
    """counts: [N] movement counts aligned to net.movements. Returns {link: rho_hat}."""
    from collections import defaultdict
    up = defaultdict(float)    # J1 sends onto L  (movements with to_edge==L)
    dn = defaultdict(float)    # J2 receives from L (movements with from_edge==L)
    for i, m in enumerate(net.movements):
        up[m.to_edge] += counts[i]
        dn[m.from_edge] += counts[i]
    links = links if links is not None else internal_links(net)
    rho = {}
    for L in links:
        u = up.get(L, 0.0)
        if u > 1e-6:
            rho[L] = dn.get(L, 0.0) / u - 1.0
    return rho


def inject_link_rho(net: NetworkData, y: np.ndarray, rho_true: dict, rng):
    """Simulate source/sink on links: inflate the DOWNSTREAM approach (movements FROM L) by
    (1+rho_L) with small measurement noise. Returns modified counts."""
    yy = y.astype(float).copy()
    for i, m in enumerate(net.movements):
        r = rho_true.get(m.from_edge)
        if r is not None:
            yy[i] *= (1.0 + r) * (1.0 + rng.normal(0, 0.03))
    return np.clip(yy, 0.0, None)


def validate(scale="9by9", n_links=40, seed=0):
    net = parse_net(net_path(scale))
    Y, _ = load_scenarios(net, scale)
    S, N = Y.shape
    links = internal_links(net)
    rng = np.random.default_rng(seed)

    # clean baseline: rho_hat should be ~0 (conservation)
    base = estimate_rho_per_link(net, Y[0], links)
    base_err = np.abs(list(base.values()))
    print(f"[{scale}] internal links={len(links)}  clean rho_hat: "
          f"mean|.|={base_err.mean():.4f} max|.|={base_err.max():.4f}  (expect ~0)")

    # assign known rho to a subset of links, inject, recover across scenarios
    chosen = list(rng.choice(links, size=min(n_links, len(links)), replace=False))
    rho_true = {L: float(rng.uniform(-0.6, 0.6)) for L in chosen}
    est = {L: [] for L in chosen}
    for s in range(S):
        yy = inject_link_rho(net, Y[s], rho_true, rng)
        rh = estimate_rho_per_link(net, yy, chosen)
        for L in chosen:
            if L in rh:
                est[L].append(rh[L])
    true_v = np.array([rho_true[L] for L in chosen])
    est_v = np.array([np.mean(est[L]) for L in chosen])
    mae = np.abs(est_v - true_v).mean()
    corr = np.corrcoef(true_v, est_v)[0, 1]
    print(f"recovery over {S} scenarios, {len(chosen)} injected links: "
          f"corr(true, est)={corr:.3f}  MAE={mae:.3f}")
    print(f"  sample: true={np.round(true_v[:6],2)}  est={np.round(est_v[:6],2)}")
    return dict(corr=corr, mae=mae, true=true_v, est=est_v, clean_max=base_err.max())


# F operating-envelope curves (from rho_robust.py / rho_in_history.py, 9by9 4x4)
ENVELOPE_CLEAN = {0.0: 0.68, 0.2: 0.37, 0.4: 0.18, 0.6: 0.08, 0.8: 0.05, 1.0: 0.02}
ENVELOPE_INHIST = {0.0: 0.62, 0.2: 0.68, 0.4: 0.66, 0.6: 0.62, 0.8: 0.60, 1.0: 0.61}


def make_overlay(rho_values=None, is_placeholder=True, out="results/fig_bucheon_overlay.png"):
    """Operating-envelope figure with the Bucheon rho distribution overlaid on the x-axis.
    Pass rho_values = np.array of |rho_hat| per real link/intersection to drop in real data;
    with none, draws a clearly-labelled PLACEHOLDER distribution so the figure is ready."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("results", exist_ok=True)

    if rho_values is None:
        rng = np.random.default_rng(0)
        rho_values = np.abs(rng.normal(0.0, 0.12, 200))     # PLACEHOLDER only
        is_placeholder = True

    fig, ax = plt.subplots(figsize=(7, 4.4))
    rr = sorted(ENVELOPE_CLEAN)
    ax.plot(rr, [ENVELOPE_INHIST[r] for r in rr], "-s", color="C2", lw=2,
            label="F skill: source/sink in history (robust)")
    ax.plot(rr, [ENVELOPE_CLEAN[r] for r in rr], "-o", color="C1", lw=2,
            label="F skill: unmodeled (graceful decay)")
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel(r"source/sink severity $\rho$"); ax.set_ylabel("F skill vs naive")
    ax.set_ylim(-0.05, 0.8)

    # Bucheon rho distribution as a normalized histogram along the x-axis (twin y)
    ax2 = ax.twinx()
    ax2.hist(np.clip(rho_values, 0, 1.0), bins=20, range=(0, 1.0),
             color="steelblue", alpha=0.35, density=True)
    ax2.set_ylabel("Bucheon $\\hat\\rho$ density", color="steelblue")
    med = float(np.median(rho_values))
    ax.axvline(med, color="steelblue", ls=":", lw=2,
               label=f"Bucheon median $\\hat\\rho$={med:.2f}")
    tag = "PLACEHOLDER (awaiting Bucheon data)" if is_placeholder else "Bucheon AI Hub 4x4"
    ax.set_title(f"Operating envelope with Bucheon $\\hat\\rho$ overlay ({tag})", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved overlay -> {out}  ({'PLACEHOLDER rho' if is_placeholder else 'real rho'})")


if __name__ == "__main__":
    validate()
    make_overlay()   # placeholder until real Bucheon data is provided
