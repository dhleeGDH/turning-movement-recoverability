"""Real Bucheon rho estimation from the aggregated 통과차량 counts.

camera code = {intersection}{approach}010 (verified: 63 = 16x4 - 1 for the 3-leg site 13).
Determines the approach->compass convention (LEG_DIR) data-drivenly: the correct convention
makes internal shared links roughly conserve (upstream-exit ~ downstream-approach), so we
pick the {1,2,3,4}->{N,E,S,W} permutation minimising median |rho_hat|. Then reports the real
rho distribution and overlays it on F's operating envelope (real Fig 9).

    PYTHONPATH=. uv run --project ... NOT needed -- pure numpy/pandas (system python3):
    python3 -m src.bucheon_rho
"""

import itertools
import json

import numpy as np

from .bucheon import DIRS, GRID_LAYOUT, OPP, build_bucheon_net
from .rho_estimate import estimate_rho_per_link, internal_links

LEFT = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT = {v: k for k, v in LEFT.items()}
INV = {v: k for k, v in GRID_LAYOUT.items()}


def parse_camera(code: str):
    rest = code[:-3]                       # strip camera suffix "010"
    return int(rest[:-1]), int(rest[-1])   # (intersection, approach)


def _neighbor(n, d):
    r, c = GRID_LAYOUT[n]; dc, dr = DIRS[d]
    return INV.get((r - dr, c + dc))


def counts_from_cam_turn(net, cam_turn, leg_dir):
    key = {(m.from_edge, m.to_edge): i for i, m in enumerate(net.movements)}
    counts = np.zeros(len(net.movements))
    for cam, turns in cam_turn.items():
        I, a = parse_camera(cam)
        da = leg_dir.get(a)
        if da is None or I not in GRID_LAYOUT:
            continue
        src = _neighbor(I, da)
        for mv, c in turns.items():
            if mv == "t":
                b = OPP[da]
            elif mv == "l":
                b = LEFT[OPP[da]]
            elif mv == "r":
                b = RIGHT[OPP[da]]
            else:
                continue                    # skip u-turn / blank
            dst = _neighbor(I, b)
            if src is None or dst is None:
                continue
            k = key.get((f"J{src}->J{I}", f"J{I}->J{dst}"))
            if k is not None:
                counts[k] += c
    return counts


def find_leg_dir(net, cam_turn):
    """Search {1,2,3,4}->compass permutations; pick the one whose internal links conserve best."""
    best = None
    for perm in itertools.permutations(["N", "E", "S", "W"]):
        ld = {1: perm[0], 2: perm[1], 3: perm[2], 4: perm[3]}
        counts = counts_from_cam_turn(net, cam_turn, ld)
        rho = estimate_rho_per_link(net, counts)
        if not rho:
            continue
        med = float(np.median(np.abs(list(rho.values()))))
        matched = int((counts > 0).sum())
        if best is None or med < best[0]:
            best = (med, ld, matched)
    return best


def run():
    net = build_bucheon_net()
    cam_turn = json.load(open("data/bucheon_cam_turn.json"))
    print(f"cameras={len(cam_turn)}  internal links={len(internal_links(net))}")
    med, leg_dir, matched = find_leg_dir(net, cam_turn)
    print(f"\nbest LEG_DIR (min median|rho_hat|): {leg_dir}")
    print(f"  median|rho_hat|={med:.3f}  matched movements with count>0 = {matched}")

    counts = counts_from_cam_turn(net, cam_turn, leg_dir)
    rho = estimate_rho_per_link(net, counts)
    vals = np.array(list(rho.values()))
    print(f"\n== Bucheon real rho_hat over {len(vals)} internal links ==")
    for q in (10, 25, 50, 75, 90):
        print(f"  p{q}: {np.percentile(vals, q):+.3f}")
    print(f"  mean|rho|={np.abs(vals).mean():.3f}  |rho|>0.2: {(np.abs(vals)>0.2).mean():.0%}  "
          f">0.4: {(np.abs(vals)>0.4).mean():.0%}")

    # real overlay
    from .rho_estimate import make_overlay
    make_overlay(rho_values=np.abs(vals), is_placeholder=False,
                 out="results/fig_bucheon_overlay_real.png")
    json.dump({k: float(v) for k, v in rho.items()}, open("data/bucheon_rho.json", "w"))
    return rho


if __name__ == "__main__":
    run()
