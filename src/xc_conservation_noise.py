"""Site-2 interior recovery under reconstruction noise that breaks approach conservation.

The split-noise test of src.xc_split_noise preserves the approach volumes, which is the
perturbation the demand anchor is immune to. This module instead injects independent per-movement
multiplicative noise, recomputes the approach volumes from the noisy counts, and re-measures the
interior (h>=2) anchor RI_ToD at 15-min, reporting the median absolute imbalance the noise induces.
Three noise seeds per level.

    python3 -m src.xc_conservation_noise  # writes data/xc_recon_noise.json
"""
from collections import defaultdict
import numpy as np
import json

from .graph.local_demand import diffuse_observed
from .graph.observe import approach_observed_mask
from .pilot_cluster import depth_h
from .xc_network import load_site2

B = load_site2()
net, mg = B["net"], B["mg"]
Y15, APPR15 = B["Y"], B["APPR"]
date15, win15, inter_of = B["date"], B["window"], B["inter_of"]
block = B["block"]
udates = sorted(set(date15.tolist()))

# approach (from_edge) -> movement indices, to recompute approach volumes from noisy turn counts
groups = defaultdict(list)
for i, m in enumerate(net.movements):
    groups[m.from_edge].append(i)
appr_of = np.zeros(len(net.movements), dtype=int)      # each movement -> index of a representative in its group
grp_members = []                                        # list of (member_indices) per group id
gid_of = np.zeros(len(net.movements), dtype=int)
for g, (fe, idxs) in enumerate(groups.items()):
    grp_members.append(np.array(idxs))
    for i in idxs:
        gid_of[i] = g


def noise_indep(sigma, seed):
    """Independent per-movement multiplicative noise on the reconstructed turn counts, with the
    multiplier centred at mean 1 (no systematic inflation). The counts are NOT renormalized and the
    approach volumes APPR the anchor diffuses are kept at their ORIGINAL clean values, so the noisy
    interior truth no longer sums to the boundary volume the anchor uses. This breaks the exact
    turn-to-approach conservation of the reconstruction, unlike the approach-preserving split test."""
    if sigma == 0:
        return Y15, APPR15
    rng = np.random.default_rng(seed)
    fac = np.exp(rng.normal(-0.5 * sigma * sigma, sigma, Y15.shape))   # E[fac]=1, unbiased
    Yn = np.clip(Y15 * fac, 0, None)
    return Yn, APPR15


def ri(a, b):
    a, b = np.concatenate(a), np.concatenate(b)
    return float(1 - np.sqrt(a.mean()) / np.sqrt(b.mean()))


def resolution(Y, AP, binmin):
    k = binmin // 15
    agg = defaultdict(lambda: [np.zeros(Y.shape[1]), np.zeros(Y.shape[1])])
    for r in range(len(Y)):
        key = (date15[r], int(win15[r]) // k)
        agg[key][0] += Y[r]; agg[key][1] += AP[r]
    keys = sorted(agg)
    return (np.array([agg[q][0] for q in keys]), np.array([agg[q][1] for q in keys]),
            np.array([q[0] for q in keys]), np.array([keys[i][1] * binmin // 60 for i in range(len(keys))]))


def interior(Y, AP, binmin=15, hrs=(7, 20)):
    Yr, APr, dt, hr = resolution(Y, AP, binmin); day = (hr >= hrs[0]) & (hr < hrs[1])
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
            refA = APr[same].mean(0) if same.any() else APr[pool].mean(0)
            den = diffuse_observed(mg, oa, refA, 30); num = diffuse_observed(mg, oa, APr[wi], 30)
            ratio = np.clip((num[q] + 1e-6) / (den[q] + 1e-6), 0.2, 5.0)
            seA.append((base[q] * ratio - Yr[wi][q]) ** 2); seN.append((base[q] - Yr[wi][q]) ** 2)
    return ri(seA, seN)


def measured_rho(Y, AP):
    """median |sum(noisy turns)/approach volume - 1| over approaches, showing how far the injected
    noise pushes the interior truth away from conservation with the (clean) approach volume."""
    r = []
    for members in grp_members:
        v_out = Y[:, members].sum(1)                       # sum of the (noisy) turns leaving the approach
        v_ap = AP[:, members[0]]                           # the original clean approach volume
        m = v_ap > 0
        if m.any():
            r.append(np.abs(v_out[m] / v_ap[m] - 1).mean())
    return float(np.median(r))


print("non-approach-preserving reconstruction noise (interior h>=2 anchor RI_ToD, 15-min)")
print(f"{'sigma':>6} {'interior RI':>12} {'median|rho|':>12}")
rows = []
for sigma in (0.0, 0.1, 0.2, 0.3, 0.5):
    its = [interior(*noise_indep(sigma, s)) for s in range(3)]
    rhos = [measured_rho(*noise_indep(sigma, s)) for s in range(3)]
    print(f"{sigma:>6.1f} {np.mean(its):>+12.3f} {np.mean(rhos):>12.3f}")
    rows.append({"sigma": sigma, "interior_RI_ToD": round(float(np.mean(its)), 3),
                 "median_abs_rho": round(float(np.mean(rhos)), 3)})
json.dump({"test": "per-movement noise breaking approach conservation; interior anchor RI_ToD, 15-min, 3 seeds",
           "interior": "depth_h >= 2 of the feas block", "rows": rows},
          open("data/xc_recon_noise.json", "w"), indent=1)
