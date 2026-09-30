"""Site-1 interior recovery per corner block under leave-one-date-out folds.

The interior stratum of the main text pools the deep-corner movements of the four 2x2
corner blocks. This module scores each block separately, with the history and the
same-clock-hour naive drawn from the other dates only, and reports per-fold Relative
Improvement for the diffused anchor and for the global growth ratio. The top level also carries
the two aggregates of the interior movements of the four blocks: the fold mean averaged over the
blocks, and the pooled RI over every block, date and window. The two aggregates can order the
modes differently.

    python3 -m src.interior_corners_lodo  # writes data/e12_interior_corners_lodo.json
"""
import json

import numpy as np

from .bucheon import GRID_LAYOUT, build_bucheon_net
from .bucheon_apply import _link_vol, build_windowed
from .graph.local_demand import diffuse_observed
from .graph.movement_graph import build_movement_graph
from .graph.observe import approach_observed_mask
from .pilot_cluster import depth_h


def _corner_blocks():
    """The four 2x2 corner blocks of the 4x4 grid, keyed TL/TR/BL/BR."""
    nr = max(r for r, _ in GRID_LAYOUT.values()) + 1
    nc = max(c for _, c in GRID_LAYOUT.values()) + 1
    blocks = {}
    for (r0, c0, tag) in [(0, 0, "TL"), (0, nc - 2, "TR"), (nr - 2, 0, "BL"), (nr - 2, nc - 2, "BR")]:
        blocks[tag] = {f"J{n}" for n, (r, c) in GRID_LAYOUT.items()
                       if r in (r0, r0 + 1) and c in (c0, c0 + 1)}
    return blocks


def run(out="data/e12_interior_corners_lodo.json"):
    net = build_bucheon_net(); mg = build_movement_graph(net)
    Yb, windows = build_windowed(net, "data/bucheon_cam_turn_60min.json")
    W = Yb.shape[0]
    hours = np.array([int(w[8:10]) for w in windows])
    dates = np.array([w[:8] for w in windows])
    udates = sorted(set(dates.tolist()))
    APPR, _ = _link_vol(net, Yb)
    res = {"n_windows": int(W), "n_dates": len(udates), "blocks": {}}
    allA, allG, allN = [], [], []
    for bname, U in _corner_blocks().items():
        um = np.array([m.intersection in U for m in net.movements])
        oa = approach_observed_mask(net, U)
        h = depth_h(mg, um, oa)
        q2 = np.where(um & (h >= 2))[0]
        folds = {"anchor": [], "global": []}
        for d in udates:
            pool = dates != d
            seA, seG, seN = [], [], []
            for wi in np.where(dates == d)[0]:
                same = pool & (hours == hours[wi])
                base = same if same.any() else pool
                ht = Yb[base].mean(0)
                histA = APPR[base].mean(0)
                r_diff = diffuse_observed(mg, oa, APPR[wi], 30) / (diffuse_observed(mg, oa, histA, 30) + 1e-9)
                obs = oa.astype(bool)
                g = (float(APPR[wi][obs].mean()) / (float(histA[obs].mean()) + 1e-9)
                     if obs.any() else 1.0)
                true = Yb[wi][q2]
                seA.append((ht[q2] * r_diff[q2] - true) ** 2)
                seG.append((ht[q2] * g - true) ** 2)
                seN.append((ht[q2] - true) ** 2)
            allA.extend(seA); allG.extend(seG); allN.extend(seN)
            rn = float(np.sqrt(np.concatenate(seN).mean()))
            folds["anchor"].append(round(1 - float(np.sqrt(np.concatenate(seA).mean())) / rn, 3))
            folds["global"].append(round(1 - float(np.sqrt(np.concatenate(seG).mean())) / rn, 3))
        res["blocks"][bname] = {
            "n_h2": int(len(q2)),
            "per_fold_anchor_RI_ToD": folds["anchor"],
            "per_fold_global_RI_ToD": folds["global"],
            "fold_mean_anchor": round(float(np.mean(folds["anchor"])), 3),
            "fold_mean_global": round(float(np.mean(folds["global"])), 3),
        }
        print(f"  {bname}: n_h2={len(q2)}  anchor={res['blocks'][bname]['fold_mean_anchor']:+.3f}  "
              f"global={res['blocks'][bname]['fold_mean_global']:+.3f}")
    bl = [b for b in res["blocks"].values() if b.get("n_h2")]
    rn = float(np.sqrt(np.concatenate(allN).mean()))
    res["n_h2_total"] = int(sum(b["n_h2"] for b in bl))
    res["block_average_fold_mean_anchor"] = round(float(np.mean([b["fold_mean_anchor"] for b in bl])), 3)
    res["block_average_fold_mean_global"] = round(float(np.mean([b["fold_mean_global"] for b in bl])), 3)
    res["anchor_pooled"] = round(1 - float(np.sqrt(np.concatenate(allA).mean())) / rn, 3)
    res["global_pooled"] = round(1 - float(np.sqrt(np.concatenate(allG).mean())) / rn, 3)
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)
    return res


if __name__ == "__main__":
    run()
