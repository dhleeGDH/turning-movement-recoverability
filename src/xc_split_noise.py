"""Site-2 recovery under noise injected into the reconstructed turn split.

The site-2 counts are reconstructed from trajectories. A smoother reconstructed split would be easier
for the historical-split anchor to predict, so this module redistributes each approach total among
its turns with multiplicative split noise N(0, sigma), which preserves the approach volumes, and
re-measures the anchor RI_ToD over the same-clock-hour lookup at 15-min: at the boundary of a
single masked intersection and in the interior (h>=2) of the masked block. Three noise seeds per
level.

    python3 -m src.xc_split_noise         # writes data/xc_smoothing.json
"""
import numpy as np
from collections import defaultdict
import json

from .graph.local_demand import diffuse_observed
from .graph.observe import approach_observed_mask
from .pilot_cluster import depth_h
from .xc_network import load_site2

B = load_site2()
net, mg = B["net"], B["mg"]
Y15, APPR15 = B["Y"], B["APPR"]
date15, win15, inter_of = B["date"], B["window"], B["inter_of"]
core = B["core_ids"]; block = B["block"]
udates = sorted(set(date15.tolist()))

groups = defaultdict(list)                        # approach (from_edge) -> movement indices
for i, m in enumerate(net.movements):
    groups[m.from_edge].append(i)


def noise_Y(sigma, seed):
    """redistribute each approach total among its turns with multiplicative split noise N(0,sigma);
    approach totals (hence APPR) are preserved, only the split (turn fractions) is roughened."""
    if sigma == 0:
        return Y15
    rng = np.random.default_rng(seed)
    Yn = Y15.astype(float).copy()
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        sub = Yn[:, idxs]
        tot = sub.sum(1, keepdims=True)
        s = sub / np.maximum(tot, 1e-9)
        s2 = np.clip(s * (1 + rng.normal(0, sigma, sub.shape)), 0, None)
        s2 = s2 / np.maximum(s2.sum(1, keepdims=True), 1e-9)
        Yn[:, idxs] = s2 * tot
    return Yn


def ri(a, b):
    a, b = np.concatenate(a), np.concatenate(b)
    return float(1 - np.sqrt(a.mean()) / np.sqrt(b.mean()))


def resolution(Y, binmin):
    k = binmin // 15
    agg = defaultdict(lambda: [np.zeros(Y.shape[1]), np.zeros(Y.shape[1])])
    for r in range(len(Y)):
        key = (date15[r], int(win15[r]) // k)
        agg[key][0] += Y[r]; agg[key][1] += APPR15[r]
    keys = sorted(agg)
    return (np.array([agg[q][0] for q in keys]), np.array([agg[q][1] for q in keys]),
            np.array([q[0] for q in keys]), np.array([keys[i][1] * binmin // 60 for i in range(len(keys))]))


def boundary_ch(Y, binmin=15, hrs=(7, 20)):
    Yr, AP, dt, hr = resolution(Y, binmin); day = (hr >= hrs[0]) & (hr < hrs[1])
    seA, seN = [], []
    for j in core:
        q = np.nonzero(inter_of == j)[0]
        if not len(q):
            continue
        for d in udates:
            pool = (dt != d) & day; ev = (dt == d) & day
            if not pool.any() or not ev.any():
                continue
            for wi in np.nonzero(ev)[0]:
                same = pool & (hr == hr[wi])
                base = Yr[same].mean(0) if same.any() else Yr[pool].mean(0)
                ref = AP[same].mean(0) if same.any() else AP[pool].mean(0)
                ratio = np.clip((AP[wi][q] + 1e-6) / (ref[q] + 1e-6), 0.2, 5.0)
                seA.append((base[q] * ratio - Yr[wi][q]) ** 2); seN.append((base[q] - Yr[wi][q]) ** 2)
    return ri(seA, seN)


def interior(Y, binmin=15, hrs=(7, 20)):
    Yr, AP, dt, hr = resolution(Y, binmin); day = (hr >= hrs[0]) & (hr < hrs[1])
    U = np.isin(inter_of, list(block)); oa = approach_observed_mask(net, set(block))
    q = np.nonzero(U & (depth_h(mg, U, oa) >= 2))[0]
    seA, seN = [], []
    for d in udates:
        pool = (dt != d) & day; ev = (dt == d) & day
        if not pool.any() or not ev.any():
            continue
        for wi in np.nonzero(ev)[0]:
            same = pool & (hr == hr[wi])
            base = Yr[same].mean(0) if same.any() else Yr[pool].mean(0)
            refA = AP[same].mean(0) if same.any() else AP[pool].mean(0)
            den = diffuse_observed(mg, oa, refA, 30); num = diffuse_observed(mg, oa, AP[wi], 30)
            ratio = np.clip((num[q] + 1e-6) / (den[q] + 1e-6), 0.2, 5.0)
            seA.append((base[q] * ratio - Yr[wi][q]) ** 2); seN.append((base[q] - Yr[wi][q]) ** 2)
    return ri(seA, seN)


print("split-noise sensitivity (anchor RI_ToD, 15-min; approach volumes preserved)")
print(f"{'sigma':>6} {'boundary':>10} {'interior':>10}")
rows = []
for sigma in (0.0, 0.1, 0.2, 0.3, 0.5):
    bs = [boundary_ch(noise_Y(sigma, s)) for s in range(3)]
    it = [interior(noise_Y(sigma, s)) for s in range(3)]
    print(f"{sigma:>6.1f} {np.mean(bs):>+10.3f} {np.mean(it):>+10.3f}")
    rows.append({"sigma": sigma, "boundary_RI_ToD": round(float(np.mean(bs)), 3),
                 "interior_RI_ToD": round(float(np.mean(it)), 3)})
json.dump({"test": "split noise, approach volumes preserved; anchor RI_ToD, 15-min, 3 seeds",
           "interior": "depth_h >= 2 of the feas block", "rows": rows},
          open("data/xc_smoothing.json", "w"), indent=1)
