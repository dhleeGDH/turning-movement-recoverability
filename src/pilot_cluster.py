"""Clustered-missing + h-axis study (backbone-free), extending src/pilot.py.

Adds three things over the single-U pilot:
  1. Contiguous U blocks of growing size (1x1 .. 3x3) -> interior nodes far from O.
  2. Observation-aware anchor B: when a U node's approach link comes from another U
     intersection (unobserved), B falls back to a historical approach mean -> collapses
     toward the naive floor A. This is exactly the loss module B/model G will recover.
  3. Per-node error stratified by h (movement-graph hops to nearest observed node) and by
     whether the approach is observed; skill and crossover rho* reported as functions of h.

Estimators (as in pilot.py): A = own regime-realized history (robust), B = conservation
anchor (observed approach x historical split, historical-approach fallback), C = oracle
(err 0) so skill = 1 - errB/errA and crossover = skill 0.
"""

import numpy as np

from .graph.movement_graph import assign_ou, build_movement_graph
from .graph.net_parser import parse_net
from .graph.observe import approach_observed_mask
from .graph.scenarios import generate_cluster_configs
from .inject.rho import inject_rho
from .pilot import load_scenarios, net_path

RHO_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2)


def _inject_all(Y, um, mg, rho, sign, seed):
    S = Y.shape[0]
    Yt = Y.astype(float).copy()
    for s in range(S):
        rng = np.random.default_rng((seed * 100003 + s) & 0x7FFFFFFF)
        Yt[s] = inject_rho(Y[s], um, mg, rho, sign, rng)
    return Yt


def depth_h(mg, um, obs_full):
    """h consistent with the observation model: h=1 for U nodes whose approach IS
    observed (they touch neighbour info directly, incl. via grid-boundary counts);
    h grows by BFS inward through U nodes with unobserved approaches. Fully enclosed
    nodes (no observed approach reachable) get max+1. This is monotone in 'depth into
    the unobserved region', unlike raw hops-to-O-intersection.
    """
    from collections import deque
    N = mg.n_nodes
    h = np.full(N, -1, dtype=np.int64)
    dq = deque()
    for n in np.nonzero(um)[0]:
        if obs_full[n]:
            h[n] = 1
            dq.append(n)
    while dq:
        n = dq.popleft()
        for nb in mg.G.neighbors(n):
            if um[nb] and h[nb] == -1:
                h[nb] = h[n] + 1
                dq.append(nb)
    unreached = um & (h == -1)
    if unreached.any():
        h[unreached] = (h[um & (h > 0)].max() + 1) if (um & (h > 0)).any() else 1
    return h


def _node_errors(Y, APPR, SPLIT, um, mg, rho, sign, seed):
    """Vectorised LOO-scenario errors per node. Returns (errA_n, errB_n) over U nodes."""
    S, N = Y.shape
    idx = np.nonzero(um)[0]
    obs = approach_observed_mask(mg.net, {mg.net.movements[i].intersection for i in idx})

    Yt = _inject_all(Y, um, mg, rho, sign, seed)
    loo = lambda M: (M.sum(0)[None, :] - M) / (S - 1)     # [S,N] leave-one-out means
    a_own = loo(Yt)
    hist_split = loo(SPLIT)
    hist_appr = loo(APPR)
    appr_used = np.where(obs[None, :], APPR, hist_appr)     # observed current else hist mean
    predA = a_own
    predB = appr_used * hist_split
    seA = (predA - Yt) ** 2
    seB = (predB - Yt) ** 2
    errA_n = np.sqrt(seA[:, idx].mean(0))                   # per-node RMSE across scenarios
    errB_n = np.sqrt(seB[:, idx].mean(0))
    h_n = depth_h(mg, um, obs)[idx]                         # obs is the full-N approach mask
    return errA_n, errB_n, h_n, obs[idx]


def _crossover(rhos, errA, errB):
    for i in range(1, len(rhos)):
        d0, d1 = errB[i - 1] - errA[i - 1], errB[i] - errA[i]
        if d0 < 0 <= d1:
            return rhos[i - 1] + (rhos[i] - rhos[i - 1]) * (-d0) / (d1 - d0)
    return None


def run(scale="5by5", n_seeds=4, sign=+1):
    net = parse_net(net_path(scale))
    mg = build_movement_graph(net)
    Y, APPR = load_scenarios(net, scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        SPLIT = np.where(APPR > 0, Y / APPR, 0.0)
    configs = generate_cluster_configs(net)
    rhos = list(RHO_GRID)

    # accumulate per-node results tagged with cluster size and h, across configs+seeds
    # store: for each rho, lists of (errA_n, errB_n, size, h, obs)
    rows = {r: {"eA": [], "eB": [], "size": [], "h": [], "obs": [], "seed": []} for r in rhos}
    for label, U, ksize in configs:
        um = assign_ou(net, U)
        for seed in range(n_seeds):
            for r in rhos:
                eA, eB, h, obs = _node_errors(Y, APPR, SPLIT, um, mg, r, sign,
                                              seed=hash((label, seed)) & 0x7FFFFFFF)
                rows[r]["eA"].append(eA); rows[r]["eB"].append(eB)
                rows[r]["size"].append(np.full(len(eA), ksize))
                rows[r]["h"].append(h); rows[r]["obs"].append(obs)
                rows[r]["seed"].append(np.full(len(eA), seed))
    for r in rhos:
        for k in rows[r]:
            rows[r][k] = np.concatenate(rows[r][k])

    def stratum_curves(mask_fn, val, seed=None):
        """mean errA/errB across rho for a stratum (optionally one seed)."""
        eA, eB = [], []
        for r in rhos:
            m = mask_fn(rows[r], val)
            if seed is not None:
                m = m & (rows[r]["seed"] == seed)
            eA.append(rows[r]["eA"][m].mean()); eB.append(rows[r]["eB"][m].mean())
        return np.array(eA), np.array(eB)

    by_size = lambda R, v: R["size"] == v
    by_h = lambda R, v: R["h"] == v

    def ci(vals):
        vals = np.array([v for v in vals if np.isfinite(v)])
        return (vals.mean(), vals.std()) if len(vals) else (float("nan"), float("nan"))

    # ---- Table 1: by cluster size, with per-seed CI ----
    print("== Skill(rho=0) & crossover rho* by CLUSTER SIZE (mean +/- sd over seeds) ==")
    print(f"{'size':>5} | {'n_nodes':>7} | {'skill@0':>13} | {'rho*':>11} | {'appr_obs%':>9}")
    print("-" * 58)
    sizes = sorted(set(rows[0.0]["size"].tolist()))
    size_summary = {}
    for sz in sizes:
        sk_seeds, rx_seeds = [], []
        for sd in range(n_seeds):
            eA, eB = stratum_curves(by_size, sz, seed=sd)
            sk_seeds.append(1 - eB[0] / eA[0] if eA[0] > 0 else np.nan)
            rx = _crossover(rhos, eA, eB); rx_seeds.append(rx if rx is not None else np.nan)
        skm, sks = ci(sk_seeds); rxm, rxs = ci(rx_seeds)
        m0 = rows[0.0]["size"] == sz
        obspct = 100 * rows[0.0]["obs"][m0].mean()
        eA_all, eB_all = stratum_curves(by_size, sz)
        size_summary[sz] = dict(skill=skm, rho=rxm, eA=eA_all, eB=eB_all)
        print(f"{sz:>5} | {int(m0.sum()):>7} | {skm:>6.2f} +/- {sks:>4.2f} | "
              f"{rxm:>5.2f} +/- {rxs:>3.2f} | {obspct:>8.0f}%")

    # ---- Table 2 (the money result): by h ----
    print("\n== Skill(rho=0) & crossover rho* by h (hops to nearest observed) ==")
    print(f"{'h':>3} | {'n_nodes':>7} | {'skill@0':>8} | {'rho*':>6} | {'appr_obs%':>9}")
    print("-" * 44)
    hvals = sorted(set(rows[0.0]["h"].tolist()))
    for hv in hvals:
        mh = rows[0.0]["h"] == hv
        eA0, eB0 = rows[0.0]["eA"][mh].mean(), rows[0.0]["eB"][mh].mean()
        skill0 = 1 - eB0 / eA0 if eA0 > 0 else float("nan")
        eA = [rows[r]["eA"][rows[r]["h"] == hv].mean() for r in rhos]
        eB = [rows[r]["eB"][rows[r]["h"] == hv].mean() for r in rhos]
        rx = _crossover(rhos, eA, eB)
        obspct = 100 * rows[0.0]["obs"][mh].mean()
        print(f"{hv:>3} | {int(mh.sum()):>7} | {skill0:>8.2f} | "
              f"{(f'{rx:.2f}' if rx else '  >max'):>6} | {obspct:>8.0f}%")

    print("\ninterpretation: as h grows, fewer approaches observed -> B collapses toward A")
    print("-> skill@0 drops and crossover rho* shrinks. THIS is the gap cluster-attention (G) must recover.")

    _money_plot(rhos, size_summary, out=f"results/fig_cluster_{scale}.png")
    return rows


def _money_plot(rhos, size_summary, out="results/fig_cluster_5by5.png"):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("results", exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: error vs rho with crossover, per cluster size (the operating envelope)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(size_summary)))
    for c, (sz, d) in zip(cmap, sorted(size_summary.items())):
        ax[0].plot(rhos, d["eB"], "-o", color=c, ms=3, label=f"B, cluster={sz}")
    ax[0].plot(rhos, next(iter(size_summary.values()))["eA"], "k--", lw=1, label="A (naive floor)")
    ax[0].set_xlabel(r"source/sink severity $\rho$"); ax[0].set_ylabel("RMSE (veh/h)")
    ax[0].set_title("Anchor B vs naive A: error grows, crossover shrinks with cluster")
    ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)

    # right: skill@0 and crossover rho* vs cluster size (the gap G must recover)
    szs = sorted(size_summary)
    sk = [size_summary[s]["skill"] for s in szs]
    rx = [size_summary[s]["rho"] for s in szs]
    ax2 = ax[1].twinx()
    l1, = ax[1].plot(szs, sk, "-s", color="C0", label="skill @ rho=0")
    l2, = ax2.plot(szs, rx, "-^", color="C3", label=r"crossover $\rho^*$")
    ax[1].set_xlabel("missing-cluster size (n intersections)")
    ax[1].set_ylabel("skill @ rho=0", color="C0"); ax2.set_ylabel(r"crossover $\rho^*$", color="C3")
    ax[1].set_title("Conservation-anchor skill collapses with cluster size")
    ax[1].axhline(0.5, ls=":", color="C0", alpha=.5)
    ax[1].legend(handles=[l1, l2], fontsize=8, loc="upper right"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nsaved money plot -> {out}")


if __name__ == "__main__":
    run()
