"""Classical IPF / Furness turning-movement baseline, scored on the same held-out
protocol as F (naive-A skill, 9x9 4x4 cluster, held-out scenarios, 4 seeds).

Furness (iterative proportional fitting) distributes a *seed* turning matrix so its
row sums match approach (inflow) margins and its column sums match exit (outflow)
margins. It is the textbook method for turning-movement estimation, so it is the
natural classical baseline. Two regimes are reported:

  IPF-oracle   : true approach & exit totals of the unmeasured intersection are
                 handed to Furness (an upper bound -- shows what perfect margins buy).
  IPF-observed : margins come only from links whose volume is actually observable
                 (upstream/downstream intersection is O, or a boundary entry);
                 unobserved interior margins fall back to the historical mean.
                 This is the deployable classical method a traffic engineer could run.

Both use the historical per-movement mean as the seed. Because that seed already
satisfies the *historical* margins exactly, an interior intersection with no observed
links leaves Furness at the seed (= naive A, skill 0): the method collapses beyond the
observed boundary, exactly like the per-node conservation physics. Only observed
boundary margins move it. Skill = 1 - RMSE(pred)/RMSE(naive-A), pooled, and by h.

Pure numpy -- no GraphPFN / GPU needed:

    python3 -m src.ipf_baseline
"""

from collections import defaultdict

import numpy as np

from .graph.labels import exit_volume_per_movement
from .graph.local_demand import local_demand_ratio
from .graph.movement_graph import assign_ou, build_movement_graph
from .graph.net_parser import parse_net
from .graph.observe import approach_observed_mask, exit_observed_mask
from .graph.scenarios import generate_cluster_configs
from .pilot import load_scenarios, net_path
from .pilot_cluster import depth_h

SEEDS = [0, 1, 2, 3]
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}
EVAL_SCALE = "9by9"


def _load(scale, n_eval_scen, seed):
    net = parse_net(net_path(scale))
    mg = build_movement_graph(net)
    Y, APPR = load_scenarios(net, scale)
    Y = Y.astype(np.float64)
    S, N = Y.shape
    EXIT = exit_volume_per_movement(net, Y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(S)
    eval_scen, train_scen = perm[:n_eval_scen], perm[n_eval_scen:]
    hist_mean = Y[train_scen].mean(0)          # per-movement historical mean (= naive A)
    appr_hist = APPR[train_scen].mean(0)       # historical approach-link volume
    exit_hist = EXIT[train_scen].mean(0)       # historical exit-link volume
    demand_ref = float(APPR[train_scen].mean())  # global reference demand level
    return dict(net=net, mg=mg, Y=Y, APPR=APPR, EXIT=EXIT, N=N,
                train_scen=train_scen, eval_scen=eval_scen, hist_mean=hist_mean,
                appr_hist=appr_hist, exit_hist=exit_hist, demand_ref=demand_ref)


def _furness(seed, appr_of, exit_of, r_marg, c_marg, n_iter=200, tol=1e-9):
    """Distribute `seed` (over one intersection's movements) to match row/col margins.
    appr_of/exit_of: integer group id per movement. r_marg/c_marg: dict group->target."""
    x = np.where(seed > 0, seed, 1e-9).astype(float)
    appr_groups = {a: np.nonzero(appr_of == a)[0] for a in set(appr_of.tolist())}
    exit_groups = {e: np.nonzero(exit_of == e)[0] for e in set(exit_of.tolist())}
    for _ in range(n_iter):
        prev = x.copy()
        for a, members in appr_groups.items():
            cur = x[members].sum()
            if cur > 0 and r_marg.get(a) is not None:
                x[members] *= r_marg[a] / cur
        for e, members in exit_groups.items():
            cur = x[members].sum()
            if cur > 0 and c_marg.get(e) is not None:
                x[members] *= c_marg[e] / cur
        if np.abs(x - prev).max() < tol:
            break
    return x


def _predict(D, real_set, mode):
    """Return an [S_eval, |U|] array of IPF predictions over the eval scenarios,
    aligned to the U-movement index order (same idx as eval_model)."""
    net, mg = D["net"], D["mg"]
    um = assign_ou(net, real_set)
    idx = np.nonzero(um)[0]
    oa = approach_observed_mask(net, real_set)
    oe = exit_observed_mask(net, real_set)

    # group U movements by intersection; within each, approach=from_edge, exit=to_edge
    per_int = defaultdict(list)
    for gi in idx:
        per_int[net.movements[gi].intersection].append(int(gi))

    preds = []
    for s in D["eval_scen"]:
        s = int(s)
        # per-node local demand ratio (same anchor information F uses)
        ratio = (local_demand_ratio(mg, oa, D["APPR"][s], D["demand_ref"])
                 if mode == "demand" else None)
        pred_full = np.zeros(D["N"], float)
        for jid, gis in per_int.items():
            gis = np.array(gis)
            # one demand scalar per intersection keeps row & col margins mutually consistent
            rho_int = float(np.mean(ratio[gis])) if ratio is not None else None
            from_e = [net.movements[g].from_edge for g in gis]
            to_e = [net.movements[g].to_edge for g in gis]
            uf = {e: i for i, e in enumerate(sorted(set(from_e)))}
            ue = {e: i for i, e in enumerate(sorted(set(to_e)))}
            appr_of = np.array([uf[e] for e in from_e])
            exit_of = np.array([ue[e] for e in to_e])
            seed = D["hist_mean"][gis]
            # margins per approach/exit group
            r_marg, c_marg = {}, {}
            for e, ai in uf.items():
                g0 = gis[from_e.index(e)]
                if mode == "oracle":                       # true approach total (upper bound)
                    r_marg[ai] = float(D["Y"][s][gis[appr_of == ai]].sum())
                elif mode == "demand":                     # demand-anchor estimate (F's info):
                    # turn-consistent historical approach total, scaled by observed demand
                    r_marg[ai] = float(D["hist_mean"][gis[appr_of == ai]].sum() * rho_int)
                else:                                      # observed link vol, else historical
                    r_marg[ai] = float(D["APPR"][s][g0] if oa[g0] else D["appr_hist"][g0])
            for e, ei in ue.items():
                g0 = gis[to_e.index(e)]
                if mode == "oracle":
                    c_marg[ei] = float(D["Y"][s][gis[exit_of == ei]].sum())
                elif mode == "demand":
                    c_marg[ei] = float(D["hist_mean"][gis[exit_of == ei]].sum() * rho_int)
                else:
                    c_marg[ei] = float(D["EXIT"][s][g0] if oe[g0] else D["exit_hist"][g0])
            x = _furness(seed, appr_of, exit_of, r_marg, c_marg)
            pred_full[gis] = x
        preds.append(pred_full[idx])
    return np.array(preds), idx, um, oa


def eval_ipf(D, real_set, mode):
    net, mg = D["net"], D["mg"]
    preds, idx, um, oa = _predict(D, real_set, mode)
    h_nodes = depth_h(mg, um, oa)[idx]
    a_own = D["hist_mean"][idx]                       # naive A (same as train_F.eval_model)
    seA, seM = [], []
    for k, s in enumerate(D["eval_scen"]):
        true = D["Y"][int(s)][idx]
        seA.append((a_own - true) ** 2)
        seM.append((preds[k] - true) ** 2)
    seA, seM = np.mean(seA, 0), np.mean(seM, 0)
    by_h = {int(hv): float(1 - np.sqrt(seM[h_nodes == hv].mean()) / np.sqrt(seA[h_nodes == hv].mean()))
            for hv in sorted(set(h_nodes.tolist()))}
    overall = float(1 - np.sqrt(seM.mean()) / np.sqrt(seA.mean()))
    return {"by_h": by_h, "overall": overall}


def _ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), float("nan")
    return float(v.mean()), T95.get(len(v) - 1, 2.0) * v.std(ddof=1) / np.sqrt(len(v))


def run(scale=EVAL_SCALE, real_label="4x4", n_eval_scen=15):
    results = {}
    modes = [("oracle", "IPF-oracle"), ("demand", "IPF-demand"), ("observed", "IPF-observed")]
    by_h_acc = {name: defaultdict(list) for _, name in modes}
    for mode, name in modes:
        vals = []
        for seed in SEEDS:
            D = _load(scale, n_eval_scen, seed)
            real_set = set([c for c in generate_cluster_configs(D["net"])
                            if real_label in c[0]][0][1])
            r = eval_ipf(D, real_set, mode)
            vals.append(r["overall"])
            for h, v in r["by_h"].items():
                by_h_acc[name][h].append(v)
        results[name] = _ci(vals)
        print(f"{name}: {results[name][0]:+.3f} ± {results[name][1]:.3f}   {np.round(vals,3)}")

    print(f"\n== IPF/Furness baseline ({scale} {real_label}, {len(SEEDS)} seeds, 95% CI) ==")
    print(f"{'model':>14} | {'overall':>14} | by h (mean)")
    for name in ["IPF-oracle", "IPF-demand", "IPF-observed"]:
        m, h = results[name]
        byh = " ".join(f"h{k}:{np.mean(v):+.2f}" for k, v in sorted(by_h_acc[name].items()))
        print(f"{name:>14} | {m:+.3f} ±{h:.3f} | {byh}")

    out = {name: {"overall": list(results[name]),
                  "by_h": {int(k): float(np.mean(v)) for k, v in by_h_acc[name].items()}}
           for name in results}
    import json
    json.dump(out, open("data/ipf_baseline.json", "w"))
    print("\nsaved data/ipf_baseline.json")
    return out


if __name__ == "__main__":
    run()
