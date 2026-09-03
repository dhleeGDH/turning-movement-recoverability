"""Delay response read against the split error each scenario actually realizes.

The sweep perturbs the split by a random draw scaled by eps, so the error a scenario realizes
is not the mean of its level. A mean delay change reported against a mean split error at each
level therefore pools scenarios that landed at very different errors, which widens the
per-level intervals and breaks their monotonicity. This module pools the (realized split
error, paired delay change) pairs over every level and scenario, and reads the response on the
axis the operational claim is about.

    python3 -m src.delay_pooled            # writes data/e10_delay_pooled.json
"""
import json

from .microsim_delay import run

GRID = "6by6"
EPS = (0.0, 0.10, 0.15, 0.19, 0.23, 0.26, 0.30, 0.40, 0.50)

if __name__ == "__main__":
    out = run(scale=GRID, eps_grid=EPS)
    json.dump({GRID: out}, open("data/e10_delay_pooled.json", "w"), indent=1)
    print("\nwrote data/e10_delay_pooled.json")
