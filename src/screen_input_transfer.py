"""Per-intersection residual same-clock-hour demand variability at site 1, and its transfer.

For each site-1 intersection the residual demandCV_res of its total approach volume is computed at
15-min, together with its within-site spread and a leave-one-intersection-out prediction of each
target's value from its grid-adjacent intersections. The median and range give the site-1 point of
Figure 3(b).

    python3 -m src.screen_input_transfer  # writes data/m19_screen_input_transfer.json
"""
from __future__ import annotations

import json

import numpy as np

from .bucheon import build_bucheon_net
from .bucheon_apply import build_windowed


def _rc(j):                                     # 'J7' -> (row, col) on the 4x4 grid
    n = int(j[1:]); return ((n - 1) // 4, (n - 1) % 4)


def per_intersection_demandcv(Yb, clockhour, inter_of, j):
    """residual same-clock-hour demandCV_res for one intersection's total approach volume."""
    cols = inter_of == j
    resid = []
    for s in set(clockhour.tolist()):
        t = Yb[clockhour == s][:, cols].sum(1)
        if len(t) >= 2 and t.mean() > 0:
            resid.append(t / t.mean() - 1)
    return float(np.concatenate(resid).std()) if resid else np.nan


def run(out="data/m19_screen_input_transfer.json", bin_min=15):
    net = build_bucheon_net()
    Yb, windows = build_windowed(net, f"data/bucheon_cam_turn_{bin_min}min.json")
    clockhour = np.array([w[8:10] for w in windows])
    inter_of = np.array([m.intersection for m in net.movements])
    inters = sorted(set(inter_of.tolist()))

    per = {j: per_intersection_demandcv(Yb, clockhour, inter_of, j) for j in inters}
    per = {j: v for j, v in per.items() if not np.isnan(v)}
    js = sorted(per)
    vals = np.array([per[j] for j in js])

    def neighbours(j):
        r, c = _rc(j)
        return [k for k in js if abs(_rc(k)[0] - r) + abs(_rc(k)[1] - c) == 1]

    # leave-one-intersection-out: predict target demandCV_res from mean of adjacent instrumented ones
    pred, act = [], []
    for j in js:
        nb = neighbours(j)
        if nb:
            pred.append(np.mean([per[k] for k in nb])); act.append(per[j])
    pred, act = np.array(pred), np.array(act)
    mae = float(np.abs(pred - act).mean())
    spread = float(vals.std())
    corr = float(np.corrcoef(act, pred)[0, 1]) if len(act) > 2 else float("nan")
    # baseline: predict every target by the site median (no spatial information)
    mae_median = float(np.abs(act - np.median(vals)).mean())

    res = {"site": "Bucheon", "bin_min": bin_min, "n_intersections": len(js),
           "demandcv_res_median": round(float(np.median(vals)), 3),
           "demandcv_res_iqr": [round(float(np.percentile(vals, 25)), 3),
                                round(float(np.percentile(vals, 75)), 3)],
           "demandcv_res_range": [round(float(vals.min()), 3), round(float(vals.max()), 3)],
           "across_intersection_spread_sd": round(spread, 3),
           "neighbour_pred_mae": round(mae, 3),
           "site_median_pred_mae": round(mae_median, 3),
           "neighbour_pred_corr": round(corr, 3),
           "per_intersection": {j: round(per[j], 3) for j in js}}
    print(f"Bucheon per-intersection demandCV_res ({bin_min}-min, {len(js)} intersections):")
    print(f"  median {res['demandcv_res_median']}, IQR {res['demandcv_res_iqr']}, "
          f"range {res['demandcv_res_range']}, across-intersection SD {spread:.3f}")
    print(f"  leave-one-intersection-out neighbour prediction: MAE {mae:.3f} "
          f"(vs site-median baseline MAE {mae_median:.3f}, corr {corr:.2f})")
    print(f"  -> a neighbour predicts the target's screen input to within {mae:.3f}, "
          f"small against the between-site contrast (Bucheon vs Xuancheng ~0.09)")
    json.dump(res, open(out, "w"), indent=2); print("saved", out)


if __name__ == "__main__":
    run()
