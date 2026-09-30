"""Per-intersection residual same-clock-hour demand variability at site 2, and its transfer.

The site-2 counterpart of src.screen_input_transfer on the 34 signals of the block's district, at
15-min, with corridor adjacency taken from the corridor edges of src.xc_network. The median and
range give the site-2 point of Figure 3(b).

    python3 -m src.xc_screen_input_transfer   # writes data/m19_xc_transfer.json
"""
import json
from collections import defaultdict

import numpy as np

from .xc_network import load_site2

B = load_site2()
net, mg = B["net"], B["mg"]
Y15 = B["Y"]; date15, win15, inter_of = B["date"], B["window"], B["inter_of"]
core = list(B["core_ids"])
hr = (win15.astype(int) * 15) // 60                         # clock hour of each 15-min window

# corridor adjacency among core intersections (from augmented E2 edges)
nbr = defaultdict(set)
for (u, v), et in mg.edge_type.items():
    if et == "E2":
        ju, jv = inter_of[u], inter_of[v]
        if ju in B["core_ids"] and jv in B["core_ids"] and ju != jv:
            nbr[ju].add(jv); nbr[jv].add(ju)


def per_intersection_demandcv(j):
    cols = inter_of == j
    if not cols.any():
        return np.nan
    resid = []
    for s in set(hr.tolist()):
        rows = hr == s
        t = Y15[rows][:, cols].sum(1)
        if len(t) >= 2 and t.mean() > 0:
            resid.append(t / t.mean() - 1)
    return float(np.concatenate(resid).std()) if resid else np.nan


per = {j: per_intersection_demandcv(j) for j in core}
per = {j: v for j, v in per.items() if not np.isnan(v)}
js = sorted(per); vals = np.array([per[j] for j in js])

pred, act = [], []
for j in js:
    nb = [k for k in nbr[j] if k in per]
    if nb:
        pred.append(np.mean([per[k] for k in nb])); act.append(per[j])
pred, act = np.array(pred), np.array(act)
mae = float(np.abs(pred - act).mean())
mae_median = float(np.abs(act - np.median(vals)).mean())
corr = float(np.corrcoef(act, pred)[0, 1]) if len(act) > 2 else float("nan")

res = {"site": "Xuancheng", "bin_min": 15, "n_intersections": len(js),
       "demandcv_res_median": round(float(np.median(vals)), 3),
       "demandcv_res_iqr": [round(float(np.percentile(vals, 25)), 3),
                            round(float(np.percentile(vals, 75)), 3)],
       "demandcv_res_range": [round(float(vals.min()), 3), round(float(vals.max()), 3)],
       "across_intersection_spread_sd": round(float(vals.std()), 3),
       "neighbour_pred_mae": round(mae, 3), "site_median_pred_mae": round(mae_median, 3),
       "neighbour_pred_corr": round(corr, 3)}
print(f"Xuancheng per-intersection demandCV_res (15-min, {len(js)} intersections):")
print(f"  median {res['demandcv_res_median']}, IQR {res['demandcv_res_iqr']}, "
      f"range {res['demandcv_res_range']}, across-intersection SD {res['across_intersection_spread_sd']}")
print(f"  leave-one-intersection-out neighbour prediction: MAE {mae:.3f} "
      f"(vs site-median baseline MAE {mae_median:.3f}, corr {corr:.2f})")
json.dump(res, open("data/m19_xc_transfer.json", "w"), indent=2)
print("saved data/m19_xc_transfer.json")
