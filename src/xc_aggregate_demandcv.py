"""Network-aggregate residual same-clock-hour demand variability at site 2.

The screening rule compares a target's network aggregate against the threshold, so the site-2
value must be computed on the same basis as site 1: the standard deviation of the within-
clock-hour residual of the window total, over the movements the site-2 results are scored on.
This reads xc_feas_arrays.npz, the arrays behind the site-2 hop-stratified results.

    python3 -m src.xc_aggregate_demandcv
"""
import json
from collections import defaultdict

import numpy as np

BASE = "data"


def demandcv_res(Y, date, window, binmin, step):
    agg = defaultdict(float)
    f = step // binmin
    for i in range(Y.shape[0]):
        agg[(date[i], window[i] // f)] += Y[i].sum()
    slots_per_hour = 60 // step
    by_hour = defaultdict(list)
    for (_, slot), v in agg.items():
        by_hour[slot // slots_per_hour if slots_per_hour > 1 else slot].append(v)
    resid = [np.array(v) / np.mean(v) - 1 for v in by_hour.values()
             if len(v) >= 2 and np.mean(v) > 0]
    return float(np.concatenate(resid).std()), len(agg)


def run(out="data/xc_aggregate_demandcv.json"):
    d = np.load(f"{BASE}/xc_feas_arrays.npz", allow_pickle=True)
    Y, date, window = d["Y"], d["date"], d["window"]
    binmin = int(d["binmin"])
    res = {"site": "Xuancheng, feasible block",
           "source": "xc_feas_arrays.npz (the arrays behind the site-2 hop-stratified results)",
           "definition": "std of the within-clock-hour residual of the window total, as at site 1",
           "n_movements": int(Y.shape[1]), "n_dates": int(len(set(date.tolist()))),
           "by_resolution": {}}
    for step in (60, 15):
        v, n = demandcv_res(Y, date, window, binmin, step)
        res["by_resolution"][f"{step}min"] = {"demandCV_res": round(v, 3), "n_windows": n}
        print(f"  {step:>2}min: demandCV_res = {v:.3f}  (windows {n})")
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)
    return res


if __name__ == "__main__":
    run()
