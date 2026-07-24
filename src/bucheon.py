"""Bucheon 4x4 real-data pipeline (AI Hub dataSetSn=522, "교차로 신호 데이터").

The '통과차량 데이터' CSV is per-vehicle passage: video_id encodes (교차로ID, 접근로ID) +
camera + timestamp; car_info.movement is the turn (r/t/l/u); departure_time the pass time.
Aggregating by (intersection, approach-leg, turn, interval) gives turning-movement counts,
from which rho_hat per shared link follows (upstream-exit vs downstream-approach mismatch).

This module builds the deterministic parts now:
  - build_bucheon_net(): a 4x4 grid NetworkData (16 intersections, internal links, movements
    with l/s/r turn types), with the single 3-leg intersection (우체국삼거리) handled.
  - passage_to_counts(): aggregate a 통과차량 DataFrame into counts aligned to net.movements,
    given TWO site-specific mappings that must be confirmed against real video_ids:
       * parse_camera(video_id) -> (intersection_id, approach_leg)   [camera-ID digit split]
       * LEG_DIR: approach_leg -> compass direction it faces          [leg convention]
Then reuse src.rho_estimate.estimate_rho_per_link.

CONFIRM-AGAINST-DATA (marked CONFIG): GRID_LAYOUT, THREE_LEG, parse_camera, LEG_DIR.
"""

import numpy as np

from .graph.net_parser import Junction, Link, Movement, NetworkData

# ---- CONFIG (confirm against page-2 map + sample video_ids) ----
# intersection number (1..16) -> (row, col) on the 4x4 grid; default row-major from the map.
GRID_LAYOUT = {n: ((n - 1) // 4, (n - 1) % 4) for n in range(1, 17)}
THREE_LEG = {13}                    # 우체국(앞)삼거리 — confirm which number is the 3-leg one
DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _turn(a_from: str, b_to: str) -> str:
    """Turn type for a vehicle entering from neighbour direction a_from and exiting toward b_to.
    heading_in = opposite(a_from); straight if b_to == opposite(a_from); else l/r by rotation."""
    if b_to == OPP[a_from]:
        return "s"
    head = OPP[a_from]
    # left = +90 (CCW), right = -90 (CW) from heading
    left = {"N": "W", "W": "S", "S": "E", "E": "N"}
    return "l" if left[head] == b_to else "r"


def build_bucheon_net(layout=None, three_leg=None) -> NetworkData:
    layout = layout or GRID_LAYOUT
    three_leg = THREE_LEG if three_leg is None else three_leg
    pos = layout
    inv = {v: k for k, v in pos.items()}
    junctions, links = {}, {}
    for n, (r, c) in pos.items():
        junctions[f"J{n}"] = Junction(f"J{n}", "traffic_light", float(c), float(-r))

    def neighbor(n, d):
        r, c = pos[n]; dc, dr = DIRS[d]
        return inv.get((r - dr, c + dc))   # d points to neighbor; row grows downward

    # internal directed links between adjacent intersections (both directions)
    for n in pos:
        for d in DIRS:
            m = neighbor(n, d)
            if m is not None and not ({n, m} & set() ):
                # skip a leg if it belongs to a 3-leg intersection's missing arm (config)
                lid = f"J{n}->J{m}"
                if lid not in links:
                    links[lid] = Link(lid, f"J{n}", f"J{m}", num_lanes=1, length=1.0, speed=1.0)

    # movements: at each intersection, approach from neighbour d_in -> exit toward d_out
    movements = []
    for n in pos:
        legs = [d for d in DIRS if neighbor(n, d) is not None]
        if n in three_leg and len(legs) == 4:
            legs = legs[:3]                # drop one arm for the 3-leg site (confirm which)
        for a in legs:                     # approach: vehicle comes FROM neighbour in dir a
            src = neighbor(n, a)
            from_edge = f"J{src}->J{n}"
            for b in legs:                 # exit toward neighbour in dir b
                if b == a:
                    continue               # no U-turn back to same leg (u handled separately)
                dst = neighbor(n, b)
                to_edge = f"J{n}->J{dst}"
                if from_edge in links and to_edge in links:
                    movements.append(Movement(from_edge, to_edge, f"J{n}", _turn(a, b)))
    return NetworkData(junctions=junctions, links=links, movements=movements)


def passage_to_counts(df, net: NetworkData, parse_camera, leg_dir, interval=None):
    """Aggregate a 통과차량 DataFrame -> counts[N] aligned to net.movements.

    parse_camera(video_id) -> (intersection_id:int, approach_leg)  # CONFIRM digit split
    leg_dir: dict approach_leg -> compass 'N'/'E'/'S'/'W'           # CONFIRM convention
    df columns per manual: video_id, car_info.movement (r/t/l/u), departure_time, ...
    """
    from collections import defaultdict
    key_to_idx = {(m.from_edge, m.to_edge): i for i, m in enumerate(net.movements)}
    counts = np.zeros(len(net.movements))
    turn_exit = {  # (approach compass, turn) -> exit compass
        **{(a, "t"): OPP[a] for a in DIRS},
        **{(a, "l"): {"N": "W", "W": "S", "S": "E", "E": "N"}[OPP[a]] for a in DIRS},
        **{(a, "r"): {"N": "E", "E": "S", "S": "W", "W": "N"}[OPP[a]] for a in DIRS},
    }
    inv = {v: k for k, v in GRID_LAYOUT.items()}
    for _, row in df.iterrows():
        mv = str(row.get("car_info.movement", "")).strip()
        if mv not in ("r", "t", "l"):     # skip u-turn / blanks for link routing
            continue
        try:
            inter, leg = parse_camera(str(row["video_id"]))
        except Exception:
            continue
        a = leg_dir.get(leg)
        if a is None or inter not in GRID_LAYOUT:
            continue
        b = turn_exit.get((a, mv))
        r, c = GRID_LAYOUT[inter]; dc, dr = DIRS.get(b, (None, None))
        if dc is None:
            continue
        src = inv.get((r - DIRS[a][1], c + DIRS[a][0]))
        dst = inv.get((r - dr, c + dc))
        if src is None or dst is None:
            continue
        k = key_to_idx.get((f"J{src}->J{inter}", f"J{inter}->J{dst}"))
        if k is not None:
            counts[k] += 1
    return counts


if __name__ == "__main__":
    from collections import Counter
    net = build_bucheon_net()
    per = Counter(m.intersection for m in net.movements)
    dirs = Counter(m.dir for m in net.movements)
    print(f"Bucheon 4x4: intersections={len(net.intersections)} links={len(net.links)} "
          f"movements={len(net.movements)}")
    print(f"movements/intersection: {sorted(set(per.values()))}  dir={dict(dirs)}")
    print(f"3-leg site J{list(THREE_LEG)}: {per.get('J'+str(list(THREE_LEG)[0]))} movements "
          f"(4-leg sites have 12)")
