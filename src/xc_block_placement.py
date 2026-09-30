"""Site-2 recovery under different placements and sizes of the masked block.

(1) Observation density: anchor RI_ToD at 15-min as a connected masked block grows from 1 to 18 of
    the 34 signals.
(2) Block placement: interior (h>=2) anchor RI_ToD at 60-min and 15-min for six contiguous
    eight-intersection blocks grown from the highest-degree seeds, with the primary block and
    repeated blocks marked.

    python3 -m src.xc_block_placement     # writes data/xc_robust.json
"""
import numpy as np
from collections import defaultdict, deque
import json

from .graph.local_demand import diffuse_observed
from .graph.observe import approach_observed_mask
from .pilot_cluster import depth_h
from .xc_network import load_site2

B = load_site2()
net, mg, Y15, APPR15 = B["net"], B["mg"], B["Y"], B["APPR"]
date15, win15, inter_of = B["date"], B["window"], B["inter_of"]
core = list(B["core_ids"]); udates = sorted(set(date15.tolist()))

# corridor adjacency among core (from augmented E2 edges)
nbr = defaultdict(set)
for (u, v), et in mg.edge_type.items():
    if et == "E2":
        ju, jv = inter_of[u], inter_of[v]
        if ju in B["core_ids"] and jv in B["core_ids"] and ju != jv:
            nbr[ju].add(jv); nbr[jv].add(ju)
deg = {j: len(nbr[j]) for j in core}


def grow(seed, k):
    blk = {seed}; q = deque([seed])
    while q and len(blk) < k:
        x = q.popleft()
        for y in sorted(nbr[x], key=lambda z: (-deg.get(z, 0), z)):
            if y not in blk and len(blk) < k:
                blk.add(y); q.append(y)
    return blk


def resolution(binmin):
    k = binmin // 15
    agg = defaultdict(lambda: [np.zeros(Y15.shape[1]), np.zeros(Y15.shape[1])])
    for r in range(len(Y15)):
        key = (date15[r], int(win15[r]) // k)
        agg[key][0] += Y15[r]; agg[key][1] += APPR15[r]
    keys = sorted(agg)
    return (np.array([agg[q][0] for q in keys]), np.array([agg[q][1] for q in keys]),
            np.array([q[0] for q in keys]), np.array([keys[i][1] * binmin // 60 for i in range(len(keys))]))


def ri(a, b):
    a, b = np.concatenate(a), np.concatenate(b)
    return float(1 - np.sqrt(a.mean()) / np.sqrt(b.mean()))


def block_ri(block, binmin=15, interior_only=False, hrs=(7, 20)):
    Y, AP, dt, hr = resolution(binmin)
    day = (hr >= hrs[0]) & (hr < hrs[1])
    U = np.isin(inter_of, list(block))
    oa = approach_observed_mask(net, set(block)); h = depth_h(mg, U, oa)
    q = np.nonzero(U & (h >= 2))[0] if interior_only else np.nonzero(U)[0]
    if len(q) == 0:
        return None, 0
    seA, seN = [], []
    for d in udates:
        pool = (dt != d) & day; ev = (dt == d) & day
        if not pool.any() or not ev.any():
            continue
        for wi in np.nonzero(ev)[0]:
            same = pool & (hr == hr[wi])
            base = Y[same].mean(0) if same.any() else Y[pool].mean(0)
            refA = AP[same].mean(0) if same.any() else AP[pool].mean(0)
            den = diffuse_observed(mg, oa, refA, 30); num = diffuse_observed(mg, oa, AP[wi], 30)
            ratio = np.clip((num[q] + 1e-6) / (den[q] + 1e-6), 0.2, 5.0)
            seA.append((base[q] * ratio - Y[wi][q]) ** 2); seN.append((base[q] - Y[wi][q]) ** 2)
    return ri(seA, seN), len(q)


out = {"density_sweep": [], "interior_by_block": []}
seed0 = max(core, key=lambda j: (deg.get(j, 0), j))
print("=== (1) observation-density sweep (anchor RI_ToD vs %observed, 15-min) ===")
N = len(core)
for k in [1, 3, 6, 10, 14, 18]:
    blk = grow(seed0, k); r, nq = block_ri(blk, 15, interior_only=False)
    print(f"  masked {len(blk):2d}/{N} ({100*(N-len(blk))/N:4.0f}% observed): anchor RI_ToD {r:+.3f}  (query {nq})")
    out["density_sweep"].append({"masked": len(blk), "n_core": N, "anchor_RI_ToD": round(r, 3), "n_query": nq})

print("\n=== (2) interior block-robustness (h>=2 anchor RI_ToD, 60- and 15-min, several 8-int blocks) ===")
seeds = sorted(core, key=lambda j: (-deg.get(j, 0), j))[:6]
seen = set()
for sd in seeds:
    blk = grow(sd, 8); key = tuple(sorted(blk))
    r60, nq = block_ri(blk, 60, interior_only=True); r15, _ = block_ri(blk, 15, interior_only=True)
    rec = {"seed": str(sd), "block": list(key), "duplicate_of_earlier_seed": key in seen,
           "is_primary_block": set(key) == set(B["block"]), "n_interior": nq,
           "interior_RI_ToD_60min": None if r60 is None else round(r60, 3),
           "interior_RI_ToD_15min": None if r15 is None else round(r15, 3)}
    seen.add(key); out["interior_by_block"].append(rec)
    print(f"  block@{sd}: interior n {nq}  RI_ToD 60min {rec['interior_RI_ToD_60min']}  15min {rec['interior_RI_ToD_15min']}"
          f"{'  (duplicate)' if rec['duplicate_of_earlier_seed'] else ''}{'  (primary)' if rec['is_primary_block'] else ''}")
alt = [x for x in out["interior_by_block"] if not x["duplicate_of_earlier_seed"] and not x["is_primary_block"]]
out["alternative_placements"] = {
    "n": len(alt), "n_interior_range": [min(x["n_interior"] for x in alt), max(x["n_interior"] for x in alt)],
    "RI_60min_range": [min(x["interior_RI_ToD_60min"] for x in alt), max(x["interior_RI_ToD_60min"] for x in alt)],
    "RI_15min_range": [min(x["interior_RI_ToD_15min"] for x in alt), max(x["interior_RI_ToD_15min"] for x in alt)]}
print("  -> alternative placements:", out["alternative_placements"])
json.dump(out, open("data/xc_robust.json", "w"), indent=1)
