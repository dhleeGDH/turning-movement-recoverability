"""Site-2 recovery RI_ToD by depth h within the masked eight-intersection block.

The depth is src.pilot_cluster.depth_h: h=1 for a query movement whose approach link is observable,
increasing by one at each step of a breadth-first search through the query set. For each depth
this module reports RI_ToD of the diffused demand anchor against the same-clock-hour lookup, the
number of movements and the spread over eight leave-one-date-out folds, at 60-min and 15-min.
It also records the depth histogram of six alternative eight-intersection blocks and, for any
block that reaches h>=4, the same breakdown on that block. Estimator, lookup, masking and
diffusion follow src.realsite_hopstrat.

    python3 -m src.xc_depth_ri            # writes data/m37_xc_depth_ri.json
"""
import json
from collections import defaultdict

import numpy as np

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
    """Aggregate the native 15-min arrays to binmin."""
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


def ri_from_se(seA, seN):
    a, n = np.asarray(seA), np.asarray(seN)
    if a.size == 0:
        return None
    return float(1 - np.sqrt(a.mean()) / np.sqrt(n.mean()))


def depth_breakdown(binmin, blk=None, hrs=(7, 20)):
    """Score the interior anchor on the block, retaining per-movement squared errors keyed by depth.

    Query set is ALL masked movements at the contiguous block; the anchor/naive are exactly the
    interior_block anchor (base * diffused-ratio) and naive (base = same-clock-hour ToD mean).
    """
    if blk is None:
        blk = block
    Y, AP, dt, hr = resolution(binmin)
    day = (hr >= hrs[0]) & (hr < hrs[1])
    U = np.isin(inter_of, list(blk))
    oa = approach_observed_mask(net, set(blk))
    h = depth_h(mg, U, oa)
    q = np.nonzero(U)[0]                      # all masked movements (h>=1)
    hq = h[q]                                 # hop depth per query movement

    # per (date, movement) squared errors, accumulated across event windows
    seA = {d: [] for d in udates}            # each entry: array over q for one window
    seN = {d: [] for d in udates}
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
            anchor = base[q] * ratio
            naive = base[q]
            true = Y[wi][q]
            seA[d].append((anchor - true) ** 2)
            seN[d].append((naive - true) ** 2)

    # stack per date -> [n_windows_d, n_q]
    stackA = {d: (np.vstack(seA[d]) if seA[d] else np.empty((0, len(q)))) for d in udates}
    stackN = {d: (np.vstack(seN[d]) if seN[d] else np.empty((0, len(q)))) for d in udates}

    out = {"binmin": binmin, "n_query_total": int(len(q)),
           "hop_histogram": {int(k): int(v) for k, v in
                             zip(*np.unique(hq, return_counts=True))},
           "by_depth": {}}
    for L in (1, 2, 3, 4, 5):
        cols = np.nonzero(hq == L)[0]
        rec = {"n_movements": int(len(cols))}
        if len(cols) == 0:
            rec["RI_ToD"] = None
            rec["folds"] = []
            out["by_depth"][L] = rec
            continue
        # pooled over all dates/windows for this depth
        allA = np.concatenate([stackA[d][:, cols].ravel() for d in udates if stackA[d].size])
        allN = np.concatenate([stackN[d][:, cols].ravel() for d in udates if stackN[d].size])
        rec["RI_ToD"] = round(ri_from_se(allA, allN), 3)
        # LODO fold spread: RI over each date's windows
        folds = []
        for d in udates:
            if stackA[d].size == 0:
                continue
            r = ri_from_se(stackA[d][:, cols].ravel(), stackN[d][:, cols].ravel())
            if r is not None:
                folds.append(round(r, 3))
        rec["folds"] = folds
        rec["fold_mean"] = round(float(np.mean(folds)), 3) if folds else None
        rec["fold_min"] = round(float(min(folds)), 3) if folds else None
        rec["fold_max"] = round(float(max(folds)), 3) if folds else None
        rec["n_folds_pos"] = int(sum(x > 0 for x in folds))
        rec["n_folds"] = len(folds)
        out["by_depth"][L] = rec

    # pooled h>=2 over the same block, for comparison with src.realsite_hopstrat
    cols2 = np.nonzero(hq >= 2)[0]
    a2 = np.concatenate([stackA[d][:, cols2].ravel() for d in udates if stackA[d].size])
    n2 = np.concatenate([stackN[d][:, cols2].ravel() for d in udates if stackN[d].size])
    out["pooled_h>=2_check"] = {"n": int(len(cols2)), "RI_ToD": round(ri_from_se(a2, n2), 3)}
    return out


def _corridor_nbr():
    nbr = defaultdict(set)
    for (u, v), et in mg.edge_type.items():
        if et == "E2":
            ju, jv = inter_of[u], inter_of[v]
            if ju in B["core_ids"] and jv in B["core_ids"] and ju != jv:
                nbr[ju].add(jv); nbr[jv].add(ju)
    core = list(B["core_ids"])
    deg = {j: len(nbr[j]) for j in core}
    return nbr, deg, core


def grow_block(seed, k=8):
    """Grow a contiguous k-intersection block from `seed` along corridor adjacency (matches
    xc_experiment / xc_robust BFS with the same deterministic tie-break)."""
    from collections import deque
    nbr, deg, _ = _corridor_nbr()
    blk = {seed}; dq = deque([seed])
    while dq and len(blk) < k:
        x = dq.popleft()
        for y in sorted(nbr[x], key=lambda z: (-deg.get(z, 0), z)):
            if y not in blk and len(blk) < k:
                blk.add(y); dq.append(y)
    return blk


def scan_alt_blocks():
    """Depth histogram of six alternative contiguous eight-intersection blocks."""
    _, deg, core = _corridor_nbr()
    seeds = sorted(core, key=lambda j: (-deg.get(j, 0), j))[:6]
    res = []
    for sd in seeds:
        blk = grow_block(sd, 8)
        U = np.isin(inter_of, list(blk))
        h = depth_h(mg, U, approach_observed_mask(net, set(blk)))
        hh = h[U]
        hist = {int(k): int(v) for k, v in zip(*np.unique(hh, return_counts=True))}
        res.append({"seed": str(sd), "n_ints": len(blk), "n_movements": int(U.sum()),
                    "max_h": int(hh.max()), "hop_histogram": hist})
    return res


def main():
    results = {
        "site": "Xuancheng (site 2, 2023 Release B)",
        "block_variant": "feas",
        "n_days": len(udates), "days": udates,
        "block_intersections": sorted(str(x) for x in block),
        "n_block_intersections": len(block),
        "protocol": "8-date leave-one-date-out; anchor=base*diffused-ratio, naive=same-clock-hour ToD mean",
        "reference": "trajectory-reconstructed turning counts",
        "depth_ri": {},
    }
    for bm in (60, 15):
        r = depth_breakdown(bm)
        results["depth_ri"][f"{bm}min"] = r
        print(f"\n=== hop-depth RI_ToD @ {bm}min (contiguous {len(block)}-int block) ===")
        print(f"  hop histogram (all masked movements): {r['hop_histogram']}")
        for L in (1, 2, 3, 4, 5):
            rec = r["by_depth"][L]
            if rec["n_movements"] == 0:
                print(f"  h={L}: n=0  (no movements at this depth)")
            else:
                print(f"  h={L}: n={rec['n_movements']:3d}  RI_ToD {rec['RI_ToD']:+.3f}  "
                      f"folds {rec['n_folds_pos']}/{rec['n_folds']}>0  "
                      f"[{rec['fold_min']:+.3f},{rec['fold_max']:+.3f}]")
        print(f"  pooled h>=2 check: {r['pooled_h>=2_check']}")

    results["alt_blocks_depth_scan"] = scan_alt_blocks()
    print("\n=== alt 8-int block hop-depth scan (max h per block) ===")
    for b in results["alt_blocks_depth_scan"]:
        print(f"  seed {b['seed']}: max_h={b['max_h']}  hist {b['hop_histogram']}")

    # blocks that reach h>=4 receive the same per-depth breakdown under the same protocol
    deep_seeds = [b["seed"] for b in results["alt_blocks_depth_scan"] if b["max_h"] >= 4]
    results["deep_blocks_depth_ri"] = {}
    for sd in deep_seeds:
        blk = grow_block(sd, 8)
        results["deep_blocks_depth_ri"][sd] = {}
        print(f"\n=== DEEP block seed {sd} (max h>=4): per-depth RI_ToD ===")
        for bm in (60, 15):
            r = depth_breakdown(bm, blk=blk)
            results["deep_blocks_depth_ri"][sd][f"{bm}min"] = r
            row = []
            for L in (1, 2, 3, 4, 5):
                rec = r["by_depth"][L]
                if rec["n_movements"] == 0:
                    row.append(f"h{L}:n0")
                else:
                    row.append(f"h{L}:n{rec['n_movements']} RI{rec['RI_ToD']:+.3f}")
            print(f"  @{bm}min  " + "  ".join(row))

    outp = "data/m37_xc_depth_ri.json"
    json.dump(results, open(outp, "w"), indent=2)
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
