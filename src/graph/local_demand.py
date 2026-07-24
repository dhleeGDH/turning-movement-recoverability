"""Local demand anchor: harmonic graph-diffusion of OBSERVED approach volumes.

The global demand ratio (mean observed approach volume) recovered ~0.61 skill; ridge's
extra +0.15 came from spatial/local boundary information, approximated by treating
observed nodes' approach volumes as boundary conditions and diffusing them across the
movement graph (Jacobi iteration for the harmonic/graph-Laplacian interpolation), so each
interior node gets a locally boundary-informed demand estimate that varies across the
missing region -- unlike a single global scalar.

Observed values are fixed (Dirichlet boundary); unobserved nodes relax to the mean of
their neighbours. No label used -> no leakage.
"""

import numpy as np


def adjacency(mg):
    """Cache a scipy CSR adjacency (+ degree) on the MovementGraph."""
    if getattr(mg, "_adj_csr", None) is None:
        import networkx as nx
        A = nx.to_scipy_sparse_array(mg.G, nodelist=range(mg.n_nodes), format="csr", dtype=float)
        mg._adj_csr = A
        mg._deg = np.asarray(A.sum(1)).ravel()
    return mg._adj_csr, mg._deg


def diffuse_observed(mg, obs_mask: np.ndarray, values: np.ndarray,
                     n_iter: int = 30) -> np.ndarray:
    """Harmonic diffusion: observed nodes hold `values`; unobserved relax to neighbour mean."""
    A, deg = adjacency(mg)
    obs = obs_mask.astype(bool)
    v = np.where(obs, values.astype(float), 0.0)
    fill = float(values[obs].mean()) if obs.any() else 0.0
    v[~obs] = fill                                   # init interior at global observed mean
    deg_safe = np.where(deg > 0, deg, 1.0)
    for _ in range(n_iter):
        nbr_mean = (A @ v) / deg_safe                # mean of neighbours (current estimate)
        v = np.where(obs, values.astype(float), nbr_mean)   # keep observed fixed
    return v


def local_demand_ratio(mg, obs_appr: np.ndarray, appr_vol: np.ndarray,
                       demand_ref: float, n_iter: int = 30) -> np.ndarray:
    """Per-node local demand ratio = diffused local observed approach volume / global ref."""
    local = diffuse_observed(mg, obs_appr, appr_vol, n_iter)
    return local / (demand_ref + 1e-9)


def routed_flow(net, obs_appr, appr_vol, hist_split, n_iter=60):
    """Boundary-routing feature: propagate OBSERVED boundary link volumes inward along the
    flow graph, splitting at each intersection by historical turn ratios (forward flow
    assignment). Captures which boundary flows route to which interior movement -- the
    spatial structure ridge exploits but the scalar demand anchor misses. Inductive
    (historical splits, network-agnostic), no label used.

    Returns (routed_appr[N], routed_count[N]) per movement.
    """
    from collections import defaultdict
    mv = net.movements
    mfrom, mto = defaultdict(list), defaultdict(list)
    for i, m in enumerate(mv):
        mfrom[m.from_edge].append(i)
        mto[m.to_edge].append(i)
    # observed link volume (a link is observed iff its movements' approach is observed)
    obs_link, vol0 = {}, {}
    for e, idxs in mfrom.items():
        obs_link[e] = bool(obs_appr[idxs[0]])
        vol0[e] = float(appr_vol[idxs[0]]) if obs_link[e] else 0.0
    v = dict(vol0)
    internal = [e for e in mfrom if not obs_link[e]]
    for _ in range(n_iter):
        new = dict(v)
        for e in internal:
            s = 0.0
            for j in mto.get(e, []):
                s += v.get(mv[j].from_edge, 0.0) * hist_split[j]
            new[e] = s
        v = new
    import numpy as np
    r_appr = np.array([v.get(m.from_edge, 0.0) for m in mv], np.float64)
    r_count = r_appr * hist_split
    return r_appr, r_count


if __name__ == "__main__":
    from .net_parser import parse_net
    from .movement_graph import assign_ou, build_movement_graph
    from .observe import approach_observed_mask
    from .scenarios import generate_cluster_configs
    from ..pilot import load_scenarios, net_path

    net = parse_net(net_path("9by9"))
    mg = build_movement_graph(net)
    Y, APPR = load_scenarios(net, "9by9")
    realU = set([c for c in generate_cluster_configs(net) if "4x4" in c[0]][0][1])
    um = assign_ou(net, realU)
    oa = approach_observed_mask(net, realU)
    ref = float(APPR.mean())
    r = local_demand_ratio(mg, oa, APPR[0], ref)
    idx = np.nonzero(um)[0]
    from .movement_graph import MovementGraph  # noqa
    from ..pilot_cluster import depth_h
    h = depth_h(mg, um, oa)[idx]
    print(f"global ratio = {APPR[0][oa].mean()/ref:.3f}")
    for hv in sorted(set(h.tolist())):
        rv = r[idx][h == hv]
        print(f"h={hv}: local ratio mean={rv.mean():.3f} std={rv.std():.3f} (n={len(rv)})")
