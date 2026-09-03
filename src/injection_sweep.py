"""Mid-block source and sink injection at the simulator level, and the crossover it locates.

For a central uninstrumented target, extra vehicles are inserted mid-block on the target's
internal approach links and driven through the network (src.sumo_inject). The neighbour
observable is the upstream intersection's discharge onto the shared approach link, which a
mid-block source does not enter. The conservation baseline

    B = upstream discharge on L  x  historical split

is therefore biased low, while the baseline formed from the target's own history

    A = realized mean of the target's turns

tracks the inflated truth. A sweep over the physical severity confirms baseline conservation
at rho = 0, recovers the injected severity with the field estimator rho_hat, and locates the
crossover between the two baselines, which the multiplicative bias model is compared against.

    python3 -m src.injection_sweep            # writes data/injection_sweep_<grid>.json
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .graph.net_parser import parse_net
from .rho_estimate import estimate_rho_per_link, internal_links
from .sumo_inject import (ROUTING, extract_YA, make_injected_routes, run_sumo,
                          sim_window)


def _base_ids(scale, k, tier="d1"):
    """Base scenario ids, optionally restricted to one demand tier (d1..d4) so ambient
    congestion is comparable and the injected severity is not confounded by base demand."""
    import glob
    import os
    ids = [os.path.basename(p)[5:-4] for p in
           sorted(glob.glob(f"{ROUTING}/{scale}/simData/vehRouteData/Route*.xml"))]
    if tier:
        ids = [i for i in ids if i.endswith(tier)]
    return ids[:k]


def _central_U(net):
    inters = sorted(net.intersections, key=lambda j: (net.junctions[j].x, net.junctions[j].y))
    xs = [net.junctions[j].x for j in inters]; ys = [net.junctions[j].y for j in inters]
    cx, cy = np.median(xs), np.median(ys)
    return min(inters, key=lambda j: (net.junctions[j].x - cx) ** 2 + (net.junctions[j].y - cy) ** 2)


def _upstream_exit(net, counts):
    up = defaultdict(float)
    for i, m in enumerate(net.movements):
        up[m.to_edge] += counts[i]
    return up


def run(scale="5by5", k_scen=20, rho_grid=(0.0, 0.3, 0.6, 0.9, 1.2, 1.5),
        tier="d1", out=None, verbose=True):
    net = parse_net(f"{ROUTING}/{scale}/{scale}.net.xml")
    U = _central_U(net)
    u_idx = [i for i, m in enumerate(net.movements) if m.intersection == U]
    # U's internal approach links (shared with an upstream instrumented neighbour)
    inter_set = set(net.intersections)
    u_appr = sorted({net.movements[i].from_edge for i in u_idx
                     if net.links[net.movements[i].from_edge].from_j in inter_set})
    ids = _base_ids(scale, k_scen, tier)
    begin, end = sim_window(scale, ids[0])
    if verbose:
        print(f"[{scale}] U={U}  U-movements={len(u_idx)}  internal approaches={len(u_appr)}  "
              f"scenarios={len(ids)}")

    # ---- baseline (clean) runs: Y_clean per scenario, clean split, clean upstream_exit ----
    Yc = []
    for sid in ids:
        _, _, c = extract_YA(net, run_sumo(scale, sid, None))
        Yc.append(c)
    Yc = np.array(Yc)                                 # [S, N] clean counts
    Sn = len(ids)

    # clean historical split of each U movement within its approach (train-averaged later)
    def split_of(counts):
        appr_tot = defaultdict(float)
        for i in u_idx:
            appr_tot[net.movements[i].from_edge] += counts[i]
        return {i: (counts[i] / appr_tot[net.movements[i].from_edge]
                    if appr_tot[net.movements[i].from_edge] > 0 else 0.0) for i in u_idx}

    links = internal_links(net)
    results = {}
    for rho in rho_grid:
        Yr = []           # injected truth per scenario
        rho_hats = []     # recovered severity on U approaches
        upexit = []       # deployment observable (upstream discharge) per scenario
        for si, sid in enumerate(ids):
            if rho <= 0:
                c = Yc[si]
            else:
                rng = np.random.default_rng(1000 + si)
                inj, _ = make_injected_routes(net, Yc[si], u_appr, rho, +1, rng, begin, end,
                                              tag=f"{sid}_r{rho}")
                _, _, c = extract_YA(net, run_sumo(scale, sid, inj))
            Yr.append(c)
            up = _upstream_exit(net, c)
            upexit.append(up)
            if rho > 0:
                rh = estimate_rho_per_link(net, c, u_appr)
                rho_hats += [rh[L] for L in u_appr if L in rh]
        Yr = np.array(Yr)

        # leave-one-scenario-out: A = regime-realised memory, B = upstream_exit x clean split
        eA, eB = [], []
        for s in range(Sn):
            tr = np.nonzero(np.arange(Sn) != s)[0]
            a_mem = Yr[tr].mean(0)                      # source-calibrated own history
            spl = {i: float(np.mean([split_of(Yc[t])[i] for t in tr])) for i in u_idx}
            for i in u_idx:
                yt = Yr[s][i]
                L = net.movements[i].from_edge
                predA = a_mem[i]
                predB = upexit[s][L] * spl[i]           # deployment observable x clean split
                eA.append((predA - yt) ** 2); eB.append((predB - yt) ** 2)
        rmseA = float(np.sqrt(np.mean(eA))); rmseB = float(np.sqrt(np.mean(eB)))
        # conservation advantage: >0 conservation (neighbour) beats memory; <0 memory wins
        cons_adv = 1.0 - rmseB / rmseA if rmseA > 0 else 0.0
        rhat = round(float(np.mean(rho_hats)), 3) if rho_hats else 0.0
        results[str(rho)] = {
            "rho_nominal": rho, "rho_hat_realized": rhat,
            "rmseA_memory": round(rmseA, 2), "rmseB_conservation": round(rmseB, 2),
            "conservation_advantage": round(cons_adv, 3),
            "u_turns_mean": round(float(Yr[:, u_idx].sum(1).mean()), 1),
        }
        if verbose:
            r = results[str(rho)]
            print(f"  rho_nom={rho:>4} rho_hat={rhat:>6}: errA(mem)={r['rmseA_memory']:>7}  "
                  f"errB(cons)={r['rmseB_conservation']:>7}  cons_adv={r['conservation_advantage']:>6}  "
                  f"Uturns={r['u_turns_mean']}")

    # crossover: realized rho_hat where memory overtakes conservation (cons_adv crosses 0 down)
    rg = sorted(results, key=lambda x: float(x))
    cross = None
    for a, b in zip(rg, rg[1:]):
        s0, s1 = results[a]["conservation_advantage"], results[b]["conservation_advantage"]
        if s0 >= 0 > s1:
            h0, h1 = results[a]["rho_hat_realized"], results[b]["rho_hat_realized"]
            cross = h0 + (h1 - h0) * (s0) / (s0 - s1)
            break
    summary = {"scale": scale, "U": U, "n_scenarios": Sn, "n_internal_approaches": len(u_appr),
               "crossover_rho_hat_star": round(cross, 3) if cross is not None else None,
               "max_realized_rho_hat": max(results[k]["rho_hat_realized"] for k in results),
               "grid": results}
    if verbose:
        print(f"\n[{scale}] physical-injection crossover rho_hat* = "
              f"{summary['crossover_rho_hat_star']}  (max realized rho_hat = "
              f"{summary['max_realized_rho_hat']}; baseline conservation clean)")
    if out is None:
        out = f"data/injection_sweep_{scale}.json"
    json.dump(summary, open(out, "w"), indent=2)
    if verbose:
        print("saved", out)
    return summary


if __name__ == "__main__":
    import sys
    sc = sys.argv[1] if len(sys.argv) > 1 else "5by5"
    kk = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    run(sc, kk)
