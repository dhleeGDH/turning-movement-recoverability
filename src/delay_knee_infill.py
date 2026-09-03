"""Design points inside the gap between 1.85 and 5.27 percentage points of split error.

The coarse sweep steps from eps = 0.15 to eps = 0.30, which realizes 1.85 and 5.27 points, so
the interval that carries the operational threshold holds no measurement of its own. This
module adds three levels inside that gap on the geometry that resolves a response.

    python3 -m src.delay_knee_infill            # writes data/e9_delay_knee_infill.json
"""
import json

from .microsim_delay import run

GRID = "6by6"
EPS = (0.0, 0.19, 0.23, 0.26, 0.30)

if __name__ == "__main__":
    out = run(scale=GRID, eps_grid=EPS)
    json.dump({GRID: out}, open("data/e9_delay_knee_infill.json", "w"), indent=1)
    print("\nwrote data/e9_delay_knee_infill.json")
    for k, v in sorted(out.items(), key=lambda kv: float(kv[0])):
        if isinstance(v, dict) and "split_err_pp" in v:
            resolved = "resolved" if abs(v["paired_pct_mean"]) > v["paired_pct_ci95"] else "not resolved"
            print(f"  eps {k:<5} split error {v['split_err_pp']:>6} pp   "
                  f"paired {v['paired_pct_mean']:+8.2f} % +/- {v['paired_pct_ci95']:.2f}   {resolved}")
