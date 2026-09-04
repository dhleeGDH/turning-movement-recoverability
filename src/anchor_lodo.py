"""Site-1 boundary recovery of the two anchor modes under leave-one-date-out folds.

For each of the six dates, the same-clock-hour history and the naive lookup come from the
other five dates, one intersection is masked at a time, and its h=1 boundary movements are
scored at the hourly and the 15-min resolution. The diffused mode applies the harmonic
ratio of the anchor; the global mode applies the single clock-hour growth ratio over the
observed approaches. Both modes share the history, the folds, and the masking.

    python3 -m src.anchor_lodo            # writes data/e13_boundary_lodo_modes.json
"""
import itertools
import json
import statistics as st

import numpy as np

from .bucheon import build_bucheon_net
from .bucheon_apply import _link_vol, build_windowed
from .graph.local_demand import diffuse_observed
from .graph.movement_graph import build_movement_graph
from .graph.observe import approach_observed_mask
from .pilot_cluster import depth_h


def _perm_p(folds):
    obs = st.mean(folds)
    cnt = sum(1 for s in itertools.product((1, -1), repeat=len(folds))
              if abs(st.mean([a * b for a, b in zip(folds, s)])) >= obs - 1e-12)
    return cnt / 2 ** len(folds)


def run(out="data/e13_boundary_lodo_modes.json"):
    res = {}
    for bin_min in (60, 15):
        net = build_bucheon_net(); mg = build_movement_graph(net)
        Yb, windows = build_windowed(net, f"data/bucheon_cam_turn_{bin_min}min.json")
        ch = np.array([w[8:10] for w in windows]); date = np.array([w[:8] for w in windows])
        udates = sorted(set(date.tolist())); inters = sorted(net.intersections)
        APPR, _ = _link_vol(net, Yb)
        pf = {"diffused": [], "global": []}
        for d in udates:
            se = {"diffused": [], "global": [], "naive": []}
            for I in inters:
                um = np.array([m.intersection == I for m in net.movements])
                idx = np.nonzero(um)[0]
                oa = approach_observed_mask(net, {I})
                hsel = depth_h(mg, um, oa)[idx] == 1
                if not hsel.any():
                    continue
                for wi in np.nonzero(date == d)[0]:
                    pool = date != d
                    same = pool & (ch == ch[wi])
                    base = same if same.any() else pool
                    hist = Yb[base].mean(0)
                    refA = APPR[base].mean(0)
                    r_g = (APPR[wi][oa].mean() + 1e-9) / (refA[oa].mean() + 1e-9)
                    r_d = diffuse_observed(mg, oa, APPR[wi], 30) / (diffuse_observed(mg, oa, refA, 30) + 1e-9)
                    true = Yb[wi][idx]
                    se["diffused"].append((((hist * r_d)[idx] - true) ** 2)[hsel])
                    se["global"].append((((hist * r_g)[idx] - true) ** 2)[hsel])
                    se["naive"].append(((hist[idx] - true) ** 2)[hsel])
            rn = float(np.sqrt(np.concatenate(se["naive"]).mean()))
            for k in ("diffused", "global"):
                pf[k].append(round(1 - float(np.sqrt(np.concatenate(se[k]).mean())) / rn, 3))
        res[f"{bin_min}min"] = {
            k: {"per_fold_RI_ToD": pf[k],
                "fold_mean": round(float(np.mean(pf[k])), 3),
                "folds_positive": int(sum(x > 0 for x in pf[k])),
                "exact_signflip_p_two": round(_perm_p(pf[k]), 4)}
            for k in ("diffused", "global")}
        for k in ("diffused", "global"):
            r = res[f"{bin_min}min"][k]
            print(f"  [{bin_min}min h=1 LODO] {k:8} mean {r['fold_mean']:+.3f} "
                  f"{r['per_fold_RI_ToD']}  {r['folds_positive']}/6  p_two={r['exact_signflip_p_two']}")
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)
    return res


if __name__ == "__main__":
    run()
