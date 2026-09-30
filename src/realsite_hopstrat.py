"""Relative Improvement of the diffused and the global demand ratio by stratum at the two real sites.

At each site the query movements are split into the boundary (h=1, the approach link observable)
and the interior (h>=2), with h the depth of src.pilot_cluster.depth_h. Each stratum is scored by
RI_ToD against the same-clock-hour lookup for
  (a) the diffused mode : base_m * r_m, r_m the boundary-diffused current/historical approach ratio,
  (b) the global mode   : base_m * g, g the network mean current/historical observed-approach volume.
The two modes share the base (same-clock-hour history) and the naive floor.

  * Site 1 (Bucheon): h=1 by single leave-one-intersection-out; h>=2 by the four 2x2 corner blocks,
    scored on their deep corners; same-clock-hour leave-one-window-out history; 30-sweep diffusion.
  * Site 2 (Xuancheng): the masked eight-intersection block of src.xc_network; eight-date
    leave-one-date-out, day hours 7-20, ratio clip [0.2, 5.0].

Each stratum also carries a date-clustered bootstrap interval (4000 draws, seed 0) of the RI
difference, and the difference of the two RIs after each is rounded to three decimals.

    python3 -m src.realsite_hopstrat      # writes data/m38_realsite_hopstrat.json
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

def _ri(se_a, se_n):
    a, n = np.asarray(se_a, float), np.asarray(se_n, float)
    if a.size == 0 or n.size == 0:
        return None
    return float(1 - np.sqrt(a.mean()) / np.sqrt(n.mean()))


def _cell_diff(rA, rG):
    """Difference of the two RIs as the table prints them (each rounded to three decimals first)."""
    if rA is None or rG is None:
        return None
    return round(round(rA, 3) - round(rG, 3), 3)


def _boot_diff(by_date, n_boot=4000, seed=0):
    """Date-clustered bootstrap 95% interval of RI(anchor) - RI(global).
    by_date: {date: ([se_anchor], [se_global], [se_naive])}; dates resampled with replacement."""
    ds = sorted(d for d, v in by_date.items() if v[0])
    if not ds:
        return None
    cat = {d: tuple(np.concatenate(x) for x in by_date[d]) for d in ds}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(ds), size=len(ds), replace=True)
        A = np.concatenate([cat[ds[i]][0] for i in pick])
        G = np.concatenate([cat[ds[i]][1] for i in pick])
        N = np.concatenate([cat[ds[i]][2] for i in pick])
        out.append(_ri(A, N) - _ri(G, N))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


# ------------------------------------------------------------------ Bucheon
def bucheon_part():
    from .bucheon import build_bucheon_net
    from .bucheon_apply import _link_vol, build_windowed
    from .corner_blocks import deep_corner_blocks as _corner_blocks
    from .graph.local_demand import diffuse_observed as _diffuse
    from .graph.movement_graph import build_movement_graph
    from .graph.observe import approach_observed_mask
    from .pilot_cluster import depth_h

    net = build_bucheon_net()
    mg = build_movement_graph(net)
    Yb, windows = build_windowed(net)
    W = Yb.shape[0]
    hours = np.array([int(w[8:10]) for w in windows])
    APPR, EXIT = _link_vol(net, Yb)
    inters = sorted(net.intersections)
    blocks = _corner_blocks()

    def hist_tod(wi, arr):
        same = (hours == hours[wi]).copy()
        same[wi] = False
        if same.any():
            return arr[same].mean(0)
        rest = np.ones(W, bool)
        rest[wi] = False
        return arr[rest].mean(0)

    # accumulate pooled squared errors per hop stratum
    acc = {"h1": {"A": [], "G": [], "N": []}, "hge2": {"A": [], "G": [], "N": []}}
    by_date = {"h1": defaultdict(lambda: ([], [], [])), "hge2": defaultdict(lambda: ([], [], []))}
    wdate = np.array([w[:8] for w in windows])
    nq = {"h1": 0, "hge2": 0}

    def score(U, query_idx, stratum):
        oa = approach_observed_mask(net, U)
        for wi in range(W):
            ht = hist_tod(wi, Yb)
            histA = hist_tod(wi, APPR)
            r_diff = _diffuse(mg, oa, APPR[wi], 30) / (_diffuse(mg, oa, histA, 30) + 1e-9)
            obs = oa.astype(bool)
            g = (float(APPR[wi][obs].mean()) / (float(histA[obs].mean()) + 1e-9)
                 if obs.any() else 1.0)
            q = query_idx
            true = Yb[wi][q]
            anchor = ht[q] * r_diff[q]
            glob = ht[q] * g
            naive = ht[q]
            acc[stratum]["A"].append((anchor - true) ** 2)
            acc[stratum]["G"].append((glob - true) ** 2)
            acc[stratum]["N"].append((naive - true) ** 2)
            bd = by_date[stratum][wdate[wi]]
            bd[0].append((anchor - true) ** 2)
            bd[1].append((glob - true) ** 2)
            bd[2].append((naive - true) ** 2)

    # h=1 boundary: single leave-one-intersection-out, all its movements are h=1
    n_h1_mov = 0
    for I in inters:
        U = {I}
        um = np.array([m.intersection in U for m in net.movements])
        oa = approach_observed_mask(net, U)
        h = depth_h(mg, um, oa)
        idxI = np.array([k for k in np.nonzero(um)[0] if net.movements[k].intersection == I])
        if idxI.size == 0:
            continue
        q1 = idxI[h[idxI] == 1]
        if q1.size == 0:
            continue
        n_h1_mov += q1.size
        score(U, q1, "h1")
    nq["h1"] = n_h1_mov

    # h>=2 interior: four 2x2 corner blocks, deep-corner movements at h>=2
    n_h2_mov = 0
    for tag, (blk, deep) in blocks.items():
        um = np.array([m.intersection in blk for m in net.movements])
        oa = approach_observed_mask(net, blk)
        h = depth_h(mg, um, oa)
        idxD = np.array([k for k in np.nonzero(um)[0] if net.movements[k].intersection == deep])
        q2 = idxD[h[idxD] >= 2]
        if q2.size == 0:
            continue
        n_h2_mov += q2.size
        score(blk, q2, "hge2")
    nq["hge2"] = n_h2_mov

    out = {"site": "Bucheon (real site 1)", "n_windows": int(W),
           "n_intersections": len(inters),
           "protocol": "single LOIO -> h=1 boundary; four 2x2 corner deep corners -> h>=2 "
                       "interior; same-clock-hour LOWO base; 30-iter diffusion; no ratio clip",
           "strata": {}}
    for st, lab in [("h1", "h=1 boundary"), ("hge2", "h>=2 interior")]:
        seA = np.concatenate(acc[st]["A"]) if acc[st]["A"] else np.array([])
        seG = np.concatenate(acc[st]["G"]) if acc[st]["G"] else np.array([])
        seN = np.concatenate(acc[st]["N"]) if acc[st]["N"] else np.array([])
        rA, rG = _ri(seA, seN), _ri(seG, seN)
        out["strata"][st] = {"label": lab, "n_query_movements": nq[st],
                             "n_scored_errors": int(seN.size),
                             "anchor_RI_ToD": None if rA is None else round(rA, 3),
                             "global_ratio_RI_ToD": None if rG is None else round(rG, 3),
                             "anchor_minus_global": None if (rA is None or rG is None)
                             else round(rA - rG, 3),
                             "anchor_minus_global_ci95": _boot_diff(by_date[st]),
                             "anchor_minus_global_of_printed_cells": _cell_diff(rA, rG)}
    return out


# ------------------------------------------------------------------ Xuancheng
def xc_part():
    from .graph.local_demand import diffuse_observed
    from .graph.observe import approach_observed_mask
    from .pilot_cluster import depth_h
    from .xc_network import load_site2

    B = load_site2()
    net, mg, Y15, APPR15 = B["net"], B["mg"], B["Y"], B["APPR"]
    date15, win15, inter_of = B["date"], B["window"], B["inter_of"]
    block = B["block"]
    udates = sorted(set(date15.tolist()))

    def resolution(binmin):
        k = binmin // 15
        agg = defaultdict(lambda: [np.zeros(Y15.shape[1]), np.zeros(Y15.shape[1])])
        for r in range(len(Y15)):
            key = (date15[r], int(win15[r]) // k)
            agg[key][0] += Y15[r]
            agg[key][1] += APPR15[r]
        keys = sorted(agg)
        Y = np.array([agg[key][0] for key in keys])
        AP = np.array([agg[key][1] for key in keys])
        dt = np.array([key[0] for key in keys])
        hr = np.array([key[1] * binmin // 60 for key in keys])
        return Y, AP, dt, hr

    def hop_strat(binmin, hrs=(7, 20)):
        Y, AP, dt, hr = resolution(binmin)
        day = (hr >= hrs[0]) & (hr < hrs[1])
        U = np.isin(inter_of, list(block))
        oa = approach_observed_mask(net, set(block))
        obs = oa.astype(bool)
        h = depth_h(mg, U, oa)          # h=1: the approach link is observable
        q = np.nonzero(U)[0]
        hq = h[q]
        acc = {"h1": {"A": [], "G": [], "N": []}, "hge2": {"A": [], "G": [], "N": []}}
        by_date = {"h1": defaultdict(lambda: ([], [], [])), "hge2": defaultdict(lambda: ([], [], []))}
        for d in udates:
            pool = (dt != d) & day
            ev = (dt == d) & day
            if not pool.any() or not ev.any():
                continue
            for wi in np.nonzero(ev)[0]:
                hh = hr[wi]
                same = pool & (hr == hh)
                base = Y[same].mean(0) if same.any() else Y[pool].mean(0)
                refA = AP[same].mean(0) if same.any() else AP[pool].mean(0)
                den = diffuse_observed(mg, oa, refA, 30)
                num = diffuse_observed(mg, oa, AP[wi], 30)
                ratio = np.clip((num[q] + 1e-6) / (den[q] + 1e-6), 0.2, 5.0)
                g = float(np.clip((AP[wi][obs].mean() + 1e-6) / (refA[obs].mean() + 1e-6),
                                  0.2, 5.0)) if obs.any() else 1.0
                base_q = base[q]
                true = Y[wi][q]
                anchor = base_q * ratio
                glob = base_q * g
                naive = base_q
                for st, sel in [("h1", hq == 1), ("hge2", hq >= 2)]:
                    if not sel.any():
                        continue
                    acc[st]["A"].append(((anchor - true) ** 2)[sel])
                    acc[st]["G"].append(((glob - true) ** 2)[sel])
                    acc[st]["N"].append(((naive - true) ** 2)[sel])
                    bd = by_date[st][d]
                    bd[0].append(((anchor - true) ** 2)[sel])
                    bd[1].append(((glob - true) ** 2)[sel])
                    bd[2].append(((naive - true) ** 2)[sel])
        res = {"binmin": binmin,
               "hop_histogram": {int(k): int(v) for k, v in zip(*np.unique(hq, return_counts=True))},
               "strata": {}}
        for st, lab in [("h1", "h=1 boundary"), ("hge2", "h>=2 interior")]:
            seA = np.concatenate(acc[st]["A"]) if acc[st]["A"] else np.array([])
            seG = np.concatenate(acc[st]["G"]) if acc[st]["G"] else np.array([])
            seN = np.concatenate(acc[st]["N"]) if acc[st]["N"] else np.array([])
            rA, rG = _ri(seA, seN), _ri(seG, seN)
            res["strata"][st] = {"label": lab,
                                 "n_query_movements": int((hq == 1).sum() if st == "h1" else (hq >= 2).sum()),
                                 "n_scored_errors": int(seN.size),
                                 "anchor_RI_ToD": None if rA is None else round(rA, 3),
                                 "global_ratio_RI_ToD": None if rG is None else round(rG, 3),
                                 "anchor_minus_global": None if (rA is None or rG is None)
                                 else round(rA - rG, 3),
                                 "anchor_minus_global_ci95": _boot_diff(by_date[st]),
                                 "anchor_minus_global_of_printed_cells": _cell_diff(rA, rG)}
        return res

    out = {"site": "Xuancheng (real site 2, 2023 Release B)",
           "block_variant": "feas",
           "n_days": len(udates), "days": udates,
           "n_block_intersections": len(block),
           "protocol": "8-date LODO, day hours 7-20, contiguous 8-int masked block; depth_h strata "
                       "(h=1 = observable approach link); same-clock-hour base; 30-iter diffusion; "
                       "ratio clip [0.2,5.0]",
           "by_resolution": {}}
    for bm in (60, 15):
        out["by_resolution"][f"{bm}min"] = hop_strat(bm)
    return out


def main():
    results = {"question": "Does the boundary-diffused demand ratio beat a global no-diffusion "
                           "growth ratio at h=1 (as synthetically 0.750 vs 0.626) and tie/lose "
                           "at h>=2, on the REAL sites?",
               "synthetic_reference": {"pooled": {"diffused": 0.629, "global": 0.629},
                                       "h1": {"diffused": 0.750, "global": 0.626},
                                       "hge2": "the diffused mode ties or loses (synthetic)"},
               "bucheon": bucheon_part(),
               "xuancheng": xc_part()}

    outp = "data/m38_realsite_hopstrat.json"
    json.dump(results, open(outp, "w"), indent=2)

    def line(site, st):
        r = st
        a, g, d = r["anchor_RI_ToD"], r["global_ratio_RI_ToD"], r["anchor_minus_global"]
        fa = f"{a:+.3f}" if a is not None else "  n/a "
        fg = f"{g:+.3f}" if g is not None else "  n/a "
        fd = f"{d:+.3f}" if d is not None else "  n/a "
        return (f"  {site:<26} {r['label']:<14} n={r['n_query_movements']:<4} "
                f"anchor {fa}  global {fg}  gap {fd}")

    print("\n============ REAL-SITE HOP-STRATIFIED RI_ToD: anchor vs global no-diffusion ratio ============")
    print("  synthetic ref  h=1: diffused +0.750 vs global +0.626 ; h>=2: diffusion ties/loses\n")
    b = results["bucheon"]
    print(f"[Bucheon]  {b['n_windows']} windows, {b['n_intersections']} intersections")
    print(line("Bucheon", b["strata"]["h1"]))
    print(line("Bucheon", b["strata"]["hge2"]))
    x = results["xuancheng"]
    print(f"\n[Xuancheng]  {x['n_days']} days, {x['n_block_intersections']}-int block")
    for bm in (60, 15):
        r = x["by_resolution"][f"{bm}min"]
        print(f"  @ {bm}min  hop hist {r['hop_histogram']}")
        print(line(f"Xuancheng@{bm}min", r["strata"]["h1"]))
        print(line(f"Xuancheng@{bm}min", r["strata"]["hge2"]))
    print(f"\nsaved data/m38_realsite_hopstrat.json")


if __name__ == "__main__":
    main()
