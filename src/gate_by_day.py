"""Eq.(gate) applied day by day at site 2, instead of once on the pooled imbalance.

The rule compares the median absolute relative imbalance of the target's links with the crossover
rho_hat* = 0.063. Site 2's pooled median is 0.062, and its per-weekday medians range from 0.052 to
0.066, so on three of the eight days the rule selects the global mode in the interior. This module
scores both modes on each date for the interior movements (h>=2) of the masked block, applies the
rule with that date's own imbalance, and pools the selected errors.

    python3 -m src.gate_by_day            # writes data/e8_gate_by_day.json
"""
from __future__ import annotations

import json

import numpy as np

from .graph.local_demand import diffuse_observed
from .graph.observe import approach_observed_mask
from .pilot_cluster import depth_h
from .xc_depth_ri import block, inter_of, mg, net, resolution, ri_from_se, udates

RHO_STAR = 0.063


def _per_day(binmin, hrs=(7, 20)):
    """Squared errors per date for the diffused mode, the global mode and the naive lookup."""
    Yb, APb, dtb, hrb = resolution(binmin)
    day = (hrb >= hrs[0]) & (hrb < hrs[1])
    U = np.isin(inter_of, list(block))
    oa = approach_observed_mask(net, set(block))
    obs = oa.astype(bool)
    h = depth_h(mg, U, oa)
    q = np.nonzero(U)[0]
    hq = h[q]
    seD = {d: [] for d in udates}
    seG = {d: [] for d in udates}
    seN = {d: [] for d in udates}
    for d in udates:
        pool = (dtb != d) & day
        ev = (dtb == d) & day
        if not pool.any() or not ev.any():
            continue
        for wi in np.nonzero(ev)[0]:
            same = pool & (hrb == hrb[wi])
            base = Yb[same].mean(0) if same.any() else Yb[pool].mean(0)
            refA = APb[same].mean(0) if same.any() else APb[pool].mean(0)
            den = diffuse_observed(mg, oa, refA, 30)
            num = diffuse_observed(mg, oa, APb[wi], 30)
            r_diff = np.clip((num[q] + 1e-6) / (den[q] + 1e-6), 0.2, 5.0)
            g = (APb[wi][obs].mean() + 1e-9) / (refA[obs].mean() + 1e-9)
            r_glob = np.clip(np.full(len(q), g), 0.2, 5.0)
            true = Yb[wi][q]
            seD[d].append((base[q] * r_diff - true) ** 2)
            seG[d].append((base[q] * r_glob - true) ** 2)
            seN[d].append((base[q] - true) ** 2)
    st = lambda s: {d: (np.vstack(s[d]) if s[d] else np.empty((0, len(q)))) for d in udates}
    return st(seD), st(seG), st(seN), hq


def run(out="data/e8_gate_by_day.json"):
    rho = json.load(open("data/e4_xc_rho_by_day.json"))["per_day"]
    res = {"rho_star": RHO_STAR, "per_day_rho": rho, "resolutions": {}}
    for binmin in (60, 15):
        D, G, N, hq = _per_day(binmin)
        cols = np.nonzero(hq >= 2)[0]                      # the interior stratum
        rows, selD, selN = [], [], []
        for d in udates:
            if D[d].size == 0:
                continue
            r = rho.get(str(d), rho.get(d))
            r = r if not isinstance(r, dict) else r.get("median")
            pick = "diffused" if abs(r) < RHO_STAR else "global"
            a = (D[d] if pick == "diffused" else G[d])[:, cols].ravel()
            n = N[d][:, cols].ravel()
            rows.append({"date": str(d), "median_abs_rho": r, "selected": pick,
                         "RI_diffused": round(ri_from_se(D[d][:, cols].ravel(), n), 4),
                         "RI_global": round(ri_from_se(G[d][:, cols].ravel(), n), 4),
                         "RI_selected": round(ri_from_se(a, n), 4),
                         "diffused_minus_global": round(ri_from_se(D[d][:, cols].ravel(), n)
                                                        - ri_from_se(G[d][:, cols].ravel(), n), 6)})
            selD.append(a); selN.append(n)
        pooled_sel = ri_from_se(np.concatenate(selD), np.concatenate(selN))
        allD = np.concatenate([D[d][:, cols].ravel() for d in udates if D[d].size])
        allG = np.concatenate([G[d][:, cols].ravel() for d in udates if G[d].size])
        allN = np.concatenate([N[d][:, cols].ravel() for d in udates if N[d].size])
        res["resolutions"][f"{binmin}min"] = {
            "n_interior_movements": int(len(cols)),
            "per_day": rows,
            "n_days_selecting_global": sum(1 for x in rows if x["selected"] == "global"),
            "global_days_diffused_minus_global_range": [
                min(x["diffused_minus_global"] for x in rows if x["selected"] == "global"),
                max(x["diffused_minus_global"] for x in rows if x["selected"] == "global")],
            "pooled_RI_under_the_rule": round(pooled_sel, 4),
            "pooled_RI_diffused_always": round(ri_from_se(allD, allN), 4),
            "pooled_RI_global_always": round(ri_from_se(allG, allN), 4),
        }
        r = res["resolutions"][f"{binmin}min"]
        print(f"\n== {binmin}-min interior ({r['n_interior_movements']} movements) ==")
        for x in rows:
            print(f"  {x['date']}  |rho|={x['median_abs_rho']:<7} -> {x['selected']:<9}"
                  f" diffused {x['RI_diffused']:+.3f}  global {x['RI_global']:+.3f}"
                  f"  selected {x['RI_selected']:+.3f}")
        print(f"  rule applied per day : {r['pooled_RI_under_the_rule']:+.4f}"
              f"   ({r['n_days_selecting_global']} of {len(rows)} days take the global mode)")
        print(f"  diffused on every day: {r['pooled_RI_diffused_always']:+.4f}")
        print(f"  global on every day  : {r['pooled_RI_global_always']:+.4f}")
    json.dump(res, open(out, "w"), indent=1)
    print("\nwrote " + out)
    return res


if __name__ == "__main__":
    run()
