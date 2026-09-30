"""Stability of the site-2 conservation imbalance across the days that define it.

The movements of site 2 come from road-identifier sequences over a given network, so each
movement's approach and exit links are fixed by the topology; the degree of freedom that remains
is the sample of days. This module recomputes the per-link median |rho_hat| on each weekday, with
the same estimator, the same internal links and the same daytime window (07-20 h) as the pooled
figure, and reports the spread.

    python3 -m src.xc_rho_by_day          # writes data/e4_xc_rho_by_day.json
"""
from __future__ import annotations

import json

import numpy as np

from .rho_estimate import estimate_rho_per_link, internal_links
from .xc_network import load_site2

_B = load_site2()
net, Y15, date15, win15 = _B["net"], _B["Y"], _B["date"], _B["window"]
udates = sorted(set(date15.tolist()))


def run(out="data/e4_xc_rho_by_day.json"):
    hr15 = (win15.astype(int) * 15) // 60
    day_mask = (hr15 >= 7) & (hr15 < 20)
    il = internal_links(net)

    def med_q(counts):
        rl = estimate_rho_per_link(net, counts, il)
        v = np.array([abs(x) for x in rl.values()])
        return v, [float(np.percentile(v, p)) for p in (25, 50, 75)]

    pooled_v, pooled_q = med_q(Y15[day_mask].sum(0))
    per_day = {}
    for d in udates:
        m = day_mask & (date15 == d)
        if m.sum() == 0:
            continue
        v, q = med_q(Y15[m].sum(0))
        per_day[str(d)] = {"n_links": int(len(v)), "q25": round(q[0], 4),
                           "median": round(q[1], 4), "q75": round(q[2], 4)}
        print(f"  {d}: median|rho|={q[1]:.4f}  IQR=[{q[0]:.4f}, {q[2]:.4f}]  n={len(v)}")

    meds = [v["median"] for v in per_day.values()]
    print(f"\npooled daytime median |rho| = {pooled_q[1]:.4f}  quartiles "
          f"[{pooled_q[0]:.4f}, {pooled_q[1]:.4f}, {pooled_q[2]:.4f}]")
    print(f"per-day median: min {min(meds):.4f}  max {max(meds):.4f}  "
          f"mean {float(np.mean(meds)):.4f}  sd {float(np.std(meds, ddof=1)):.4f}")
    res = {"site": "Xuancheng (site 2)", "n_internal_links": int(len(pooled_v)),
           "note": "per-LINK |rho_hat| median; the pooled figure aggregates all days first",
           "pooled": {"q25": round(pooled_q[0], 4), "median": round(pooled_q[1], 4),
                      "q75": round(pooled_q[2], 4)},
           "per_day": per_day,
           "per_day_median_min": round(min(meds), 4), "per_day_median_max": round(max(meds), 4),
           "per_day_median_sd": round(float(np.std(meds, ddof=1)), 4)}
    json.dump(res, open(out, "w"), indent=2)
    print("saved", out)
    return res


if __name__ == "__main__":
    run()
