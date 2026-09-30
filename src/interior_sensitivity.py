"""Camera-error sensitivity of the site-1 interior difference between the two demand modes.

The population is the eight deep-corner movements (h>=2) of the four 2x2 corner blocks, scored
hourly with the same-clock-hour leave-one-window-out history and 30-sweep diffusion as
src.realsite_hopstrat. The quantity is RI(diffused) - RI(global), both against the same-clock-hour
lookup. Each camera is one approach link, and one factor scales both the true counts and the
boundary volume of that approach. The error is injected on the evaluated date only, with the
history left clean, except the turn misclassification, which applies to every window:
  random per-camera factor, lognormal sd 0.19, eight draws;
  congestion-dependent factor, clip(1 - 0.19 v / v_ref, 0.7, 1), v_ref the 90th-percentile approach volume;
  additive volume-scaled missed detections at 0.19 and 0.30 of the approach volume;
  hour-varying lognormal detection factor, sd 0.19 and 0.30, eight draws;
  1% within-approach turn misclassification.
Folds are the six dates; the interval is a date-clustered bootstrap marginalised over draws.

    python3 -m src.interior_sensitivity   # writes data/m53_s4_interior_sensitivity.json
"""
import json
import numpy as np
from .bucheon import build_bucheon_net
from .bucheon_apply import build_windowed, _link_vol
from .corner_blocks import deep_corner_blocks as _corner_blocks
from .graph.local_demand import diffuse_observed as _diffuse
from .graph.movement_graph import build_movement_graph
from .graph.observe import approach_observed_mask
from .pilot_cluster import depth_h
OUT = "data/m53_s4_interior_sensitivity.json"
net = build_bucheon_net(); mg = build_movement_graph(net)
Yb, windows = build_windowed(net); W = Yb.shape[0]
hours = np.array([int(w[8:10]) for w in windows]); dates = np.array([w[:8] for w in windows]); ud = sorted(set(dates))
APPR, _ = _link_vol(net, Yb)
cam_of = np.array([m.from_edge for m in net.movements]); cams = sorted(set(cam_of))
cidx = {c: np.nonzero(cam_of == c)[0] for c in cams}
v_cam = np.array([[Yb[w, cidx[c]].sum() for c in cams] for w in range(W)])          # true approach volume
v_ref = float(np.percentile(v_cam[v_cam > 0], 90))

def hist(wi, arr):
    same = (hours == hours[wi]).copy(); same[wi] = False
    if same.any(): return arr[same].mean(0)
    rest = np.ones(W, bool); rest[wi] = False; return arr[rest].mean(0)

setups = []
for tag, (blk, deep) in _corner_blocks().items():
    um = np.array([m.intersection in blk for m in net.movements]); oa = approach_observed_mask(net, blk)
    h = depth_h(mg, um, oa); idxD = np.array([k for k in np.nonzero(um)[0] if net.movements[k].intersection == deep])
    q = idxD[h[idxD] >= 2]
    if q.size: setups.append((oa, q))
NQ = sum(len(q) for _, q in setups)

def errors(Ytrue, Ahist_src, Yhist_src, Aeval):
    """per-date lists of squared errors (A, G, N) over windows x query movements."""
    E = {d: ([], [], []) for d in ud}
    for oa, q in setups:
        obs = oa.astype(bool)
        for wi in range(W):
            ht = hist(wi, Yhist_src); hA = hist(wi, Ahist_src)
            r = _diffuse(mg, oa, Aeval[wi], 30) / (_diffuse(mg, oa, hA, 30) + 1e-9)
            g = float(Aeval[wi][obs].mean()) / (float(hA[obs].mean()) + 1e-9)
            t = Ytrue[wi][q]
            E[dates[wi]][0].append((ht[q] * r[q] - t) ** 2); E[dates[wi]][1].append((ht[q] * g - t) ** 2)
            E[dates[wi]][2].append((ht[q] - t) ** 2)
    return {d: tuple(np.concatenate(x) for x in v) for d, v in E.items()}

def perturbed(kind, rng, sev):
    Yd = Yb.copy()
    if kind == "misclass":
        for c in cams:
            ix = cidx[c]; k = len(ix)
            if k < 2: continue
            tot = Yd[:, ix].sum(1, keepdims=True)
            Yd[:, ix] = (1 - sev) * Yd[:, ix] + sev / (k - 1) * (tot - Yd[:, ix])
        return Yd
    for d in ud:
        rows = np.nonzero(dates == d)[0]
        if kind == "random":
            fac = rng.lognormal(0.0, sev, size=len(cams))
            for j, c in enumerate(cams): Yd[np.ix_(rows, cidx[c])] *= fac[j]
        elif kind == "tod":
            for wi in rows:
                fac = rng.lognormal(0.0, sev, size=len(cams))
                for j, c in enumerate(cams): Yd[wi, cidx[c]] *= fac[j]
        elif kind == "voldep":
            for wi in rows:
                for j, c in enumerate(cams):
                    Yd[wi, cidx[c]] *= np.clip(1.0 - sev * v_cam[wi, j] / v_ref, 0.70, 1.0)
        elif kind == "additive":
            for wi in rows:
                for j, c in enumerate(cams):
                    ix = cidx[c]; miss = sev * v_cam[wi, j]
                    Yd[wi, ix] = np.clip(Yd[wi, ix] - miss / len(ix), 0, None)
    return Yd

def ri(a, n): return float(1 - np.sqrt(np.mean(a)) / np.sqrt(np.mean(n)))

def run(kind, sev, K):
    draws = []
    for k in range(K):
        Yd = perturbed(kind, np.random.default_rng(1900 + k), sev) if kind != "none" else Yb
        Ad, _ = _link_vol(net, Yd)
        if kind == "misclass":           # systematic labelling: present in history, truth and naive alike
            draws.append(errors(Yd, Ad, Yd, Ad))
        else:                            # evaluated date perturbed, history clean
            draws.append(errors(Yd, APPR, Yb, Ad))
    pool = lambda E, j: np.concatenate([E[d][j] for d in ud])
    RA = np.mean([ri(pool(E, 0), pool(E, 2)) for E in draws]); RG = np.mean([ri(pool(E, 1), pool(E, 2)) for E in draws])
    fold = [np.mean([ri(E[d][0], E[d][2]) - ri(E[d][1], E[d][2]) for E in draws]) for d in ud]
    extra = {}
    if kind == "misclass":           # split perturbation on the query movements, split-fraction units (as m23)
        Yd = perturbed(kind, None, sev); qa = np.unique(np.concatenate([q for _, q in setups]))
        A = APPR[:, qa]; msk = A > 0
        d = (Yd[:, qa] - Yb[:, qa])[msk] / A[msk]
        extra = {"split_delta_rms": round(float(np.sqrt(np.mean(d ** 2))), 6),
                 "split_delta_rms_over_0095": round(float(np.sqrt(np.mean(d ** 2))) / 0.095, 4)}
    rng = np.random.default_rng(0); bs = []
    for _ in range(4000):
        E = draws[rng.integers(0, K)]; pick = rng.choice(ud, len(ud), replace=True)
        a = np.concatenate([E[d][0] for d in pick]); g = np.concatenate([E[d][1] for d in pick]); n = np.concatenate([E[d][2] for d in pick])
        bs.append(ri(a, n) - ri(g, n))
    return {"kind": kind, "severity": sev, "draws": K, "RI_diffused": round(RA, 3), "RI_global": round(RG, 3),
            "diffused_minus_global": round(RA - RG, 3), "global_minus_diffused": round(RG - RA, 3), "per_date_diff": [round(x, 3) for x in fold],
            "dates_diffused_ahead": f"{sum(x > 0 for x in fold)}/{len(fold)}",
            "boot95_diff": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], **extra}

res = {"population": f"site-1 interior of the depth table, {NQ} deep-corner h>=2 movements, {W} hourly windows", "v_ref_p90": round(v_ref, 1), "rows": []}
for kind, sev, K in [("none", 0, 1), ("random", 0.19, 8), ("voldep", 0.19, 1), ("additive", 0.19, 1), ("additive", 0.30, 1),
                     ("tod", 0.19, 8), ("tod", 0.30, 8), ("misclass", 0.01, 1)]:
    r = run(kind, sev, K); res["rows"].append(r); print(r, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
