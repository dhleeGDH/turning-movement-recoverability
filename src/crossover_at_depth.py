"""Conservation crossover resolved by hop depth and scored on the contest the gate decides.

The crossover of Eq. (gate) is used at two strata that the single-intersection mask of the
earlier sweep cannot separate. Under that mask every query movement is adjacent to an
instrumented neighbour, which is the stratum in which the gate returns the diffused mode
unconditionally and never consults rho_hat.

This module re-runs the same physical mid-block injection on the 9x9 grid with the 4x4
held-out cluster of the synthetic setting, which supplies a boundary stratum at h = 1 and an
interior stratum at h >= 2. Both contests are scored on the same runs, so the two are
comparable:

    memory baseline  vs  upstream discharge x historical split
    hist x r_diff    vs  hist x r_glob                            (Eq. gate)

    python3 -m src.crossover_at_depth            # writes data/e6_crossover_at_depth.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .graph.local_demand import diffuse_observed
from .graph.movement_graph import assign_ou, build_movement_graph
from .graph.net_parser import parse_net
from .graph.observe import approach_observed_mask
from .graph.scenarios import generate_cluster_configs
from .pilot_cluster import depth_h
from .rho_estimate import estimate_rho_per_link
from .sumo_inject import (ROUTING, extract_YA, make_injected_routes, run_sumo,
                          sim_window)
from .injection_sweep import _base_ids

SCALE = "9by9"
K_SCEN = 20
SOURCE_GRID = (0.0, 0.05, 0.10, 0.15, 0.25, 0.40)
SINK_GRID = (0.05, 0.10, 0.20, 0.35, 0.55)
N_WORKERS = int(os.environ.get("E6_WORKERS", "16"))


def _setup():
    net = parse_net(f"{ROUTING}/{SCALE}/{SCALE}.net.xml")
    mg = build_movement_graph(net)
    real = set([c for c in generate_cluster_configs(net) if "4x4" in c[0]][0][1])
    um = assign_ou(net, real)
    idx = np.nonzero(um)[0]
    oa = approach_observed_mask(net, real)
    hq = depth_h(mg, um, oa)[idx]
    # Injection goes on the masked cluster's own internal approach links, which are the links
    # whose imbalance the rule reads at the target.
    inter = set(net.intersections)
    tgt = sorted({net.movements[i].from_edge for i in idx
                  if net.links[net.movements[i].from_edge].from_j in inter})
    return net, mg, real, um, idx, oa, hq, tgt


def _one(args):
    """One SUMO run. Returns (Y, APPR, counts) for a scenario, clean or injected."""
    sid, rho, sign, tgt, begin, end, base_counts = args
    net = parse_net(f"{ROUTING}/{SCALE}/{SCALE}.net.xml")
    if rho <= 0:
        vr = run_sumo(SCALE, sid, None)
    else:
        rng = np.random.default_rng(hash((sid, rho, sign)) % (2 ** 31))
        inj, _ = make_injected_routes(net, np.asarray(base_counts), tgt, rho, sign, rng,
                                      begin, end, tag=f"{sid}_r{rho}_s{sign}")
        vr = run_sumo(SCALE, sid, inj)
    Y, APPR, counts = extract_YA(net, vr)
    try:
        os.remove(vr)
    except OSError:
        pass
    return np.asarray(Y), np.asarray(APPR), np.asarray(counts)


def _ri(pred_by_scen, true_by_scen, naive):
    seN, seM = [], []
    for p, t in zip(pred_by_scen, true_by_scen):
        seN.append((naive - t) ** 2)
        seM.append((p - t) ** 2)
    seN = np.concatenate(seN); seM = np.concatenate(seM)
    return float(1 - np.sqrt(seM.mean()) / np.sqrt(seN.mean()))


def _score(net, mg, oa, idx, hq, Ys, APPRs, tgt_links):
    """Leave-one-scenario-out scoring of both contests, split by hop depth."""
    S = len(Ys)
    obs = oa.astype(bool)
    out = {}
    for hb, sel in (("h1", hq == 1), ("hge2", hq >= 2)):
        if sel.sum() == 0:
            continue
        ii = idx[sel]
        pd_, pg_, pm_, pc_, tt_ = [], [], [], [], []
        for s in range(S):
            tr = [t for t in range(S) if t != s]
            hist_mean = np.mean([Ys[t] for t in tr], axis=0)
            hist_appr = np.mean([APPRs[t] for t in tr], axis=0)
            cur = APPRs[s]
            r_diff = diffuse_observed(mg, oa, cur, 30) / (
                diffuse_observed(mg, oa, hist_appr, 30) + 1e-9)
            g = (cur[obs].mean() + 1e-9) / (hist_appr[obs].mean() + 1e-9)
            pd_.append(hist_mean[ii] * r_diff[ii])
            pg_.append(hist_mean[ii] * g)
            # old contest: memory vs upstream discharge x historical split
            up = defaultdict(float)
            for i, m in enumerate(net.movements):
                up[m.to_edge] += Ys[s][i]
            appr_tot = defaultdict(float)
            for t in tr:
                for i in ii:
                    appr_tot[(t, net.movements[i].from_edge)] += Ys[t][i]
            spl = {}
            for i in ii:
                L = net.movements[i].from_edge
                v = np.mean([Ys[t][i] / appr_tot[(t, L)] if appr_tot[(t, L)] > 0 else 0.0
                             for t in tr])
                spl[i] = v
            pm_.append(hist_mean[ii])
            pc_.append(np.array([up[net.movements[i].from_edge] * spl[i] for i in ii]))
            tt_.append(Ys[s][ii])
        naive = np.mean([Ys[t] for t in range(S)], axis=0)[ii]
        out[hb] = {
            "n_movements": int(sel.sum()),
            "RI_diffused": round(_ri(pd_, tt_, naive), 4),
            "RI_global": round(_ri(pg_, tt_, naive), 4),
            "RI_memory": round(_ri(pm_, tt_, naive), 4),
            "RI_conservation": round(_ri(pc_, tt_, naive), 4),
        }
        out[hb]["gate_advantage"] = round(out[hb]["RI_diffused"] - out[hb]["RI_global"], 4)
        out[hb]["old_advantage"] = round(out[hb]["RI_conservation"] - out[hb]["RI_memory"], 4)
    return out


def _cross(levels, adv):
    """Realized |rho_hat| where the advantage crosses zero downward, between adjacent points."""
    for a, b in zip(range(len(levels) - 1), range(1, len(levels))):
        if adv[a] > 0 >= adv[b]:
            x0, x1, y0, y1 = levels[a], levels[b], adv[a], adv[b]
            return round(x0 + (x1 - x0) * y0 / (y0 - y1), 4), (round(x0, 4), round(x1, 4))
    return None, None


def run(out="data/e6_crossover_at_depth.json"):
    net, mg, real, um, idx, oa, hq, tgt = _setup()
    ids = _base_ids(SCALE, K_SCEN)
    begin, end = sim_window(SCALE, ids[0])
    print(f"[{SCALE}] masked 4x4 cluster: {len(real)} intersections, {len(idx)} query movements "
          f"(h=1: {(hq==1).sum()}, h>=2: {(hq>=2).sum()}); injection links: {len(tgt)}; "
          f"scenarios: {len(ids)}")

    with ProcessPoolExecutor(N_WORKERS) as ex:
        clean = list(ex.map(_one, [(s, 0.0, +1, tgt, begin, end, None) for s in ids]))
    Yc = [c[2] for c in clean]
    print(f"  clean runs done ({len(Yc)})")

    res = {"scale": SCALE, "k_scen": K_SCEN, "n_query": int(len(idx)),
           "n_h1": int((hq == 1).sum()), "n_hge2": int((hq >= 2).sum()),
           "n_injection_links": len(tgt), "branches": {}}

    for name, grid, sign in (("source", SOURCE_GRID, +1), ("sink", SINK_GRID, -1)):
        rows = []
        for rho in grid:
            if rho <= 0:
                Ys = [c[0] for c in clean]; APPRs = [c[1] for c in clean]
                Cs = Yc; rhat = 0.0
            else:
                jobs = [(s, rho, sign, tgt, begin, end, Yc[i]) for i, s in enumerate(ids)]
                with ProcessPoolExecutor(N_WORKERS) as ex:
                    got = list(ex.map(_one, jobs))
                Ys = [g[0] for g in got]; APPRs = [g[1] for g in got]; Cs = [g[2] for g in got]
                rh = []
                for c in Cs:
                    d = estimate_rho_per_link(net, c, tgt)
                    rh += [abs(d[L]) for L in tgt if L in d]
                rhat = float(np.median(rh)) if rh else 0.0
            sc = _score(net, mg, oa, idx, hq, Ys, APPRs, tgt)
            rows.append({"rho_nominal": rho, "rho_hat_realized": round(rhat, 4), **sc})
            g1 = sc.get("h1", {}); g2 = sc.get("hge2", {})
            print(f"  {name} rho={rho:<5} |rho_hat|={rhat:.4f}  "
                  f"h1 gate_adv={g1.get('gate_advantage'):>7} old_adv={g1.get('old_advantage'):>7}  |  "
                  f"hge2 gate_adv={g2.get('gate_advantage'):>7} old_adv={g2.get('old_advantage'):>7}")
        lv = [r["rho_hat_realized"] for r in rows]
        res["branches"][name] = {"rows": rows}
        for hb in ("h1", "hge2"):
            for key, lab in (("gate_advantage", "gate"), ("old_advantage", "old")):
                adv = [r.get(hb, {}).get(key, 0.0) for r in rows]
                c, br = _cross(lv, adv)
                res["branches"][name][f"crossover_{hb}_{lab}"] = c
                res["branches"][name][f"bracket_{hb}_{lab}"] = br
    json.dump(res, open(out, "w"), indent=1)
    print("\nwrote " + out)
    for name in res["branches"]:
        b = res["branches"][name]
        print(f"  {name}: gate h1={b['crossover_h1_gate']} hge2={b['crossover_hge2_gate']} | "
              f"old h1={b['crossover_h1_old']} hge2={b['crossover_hge2_old']}")
    return res


if __name__ == "__main__":
    run()
