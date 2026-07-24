"""Missing-configuration generator: contiguous rectangular blocks of U intersections
on the grid, for the clustered-missing / h-axis study.

Grid coords come straight from junction (x, y). A k-block places U intersections
in a contiguous r x c rectangle; interior intersections sit multiple hops from the
nearest observed one, which is what makes the anchor B degrade.
"""

import numpy as np

from .net_parser import NetworkData


def grid_coords(net: NetworkData) -> dict[str, tuple[int, int]]:
    inter = net.intersections
    xs = sorted(set(round(j.x) for j in inter.values()))
    ys = sorted(set(round(j.y) for j in inter.values()))
    col = {x: i for i, x in enumerate(xs)}
    row = {y: i for i, y in enumerate(ys)}
    return {jid: (row[round(j.y)], col[round(j.x)]) for jid, j in inter.items()}


def rect_block(net: NetworkData, r0: int, c0: int, h: int, w: int) -> set[str]:
    rc = grid_coords(net)
    want = {(r0 + dr, c0 + dc) for dr in range(h) for dc in range(w)}
    return {jid for jid, pos in rc.items() if pos in want}


def generate_cluster_configs(net: NetworkData) -> list[tuple[str, set[str], int]]:
    """A spread of block shapes/positions. Returns (label, U set, n_intersections)."""
    rc = grid_coords(net)
    nrows = max(r for r, _ in rc.values()) + 1
    ncols = max(c for _, c in rc.values()) + 1
    shapes = [("1x1", 1, 1), ("1x2", 1, 2), ("2x2", 2, 2),
              ("2x3", 2, 3), ("3x3", 3, 3), ("4x4", 4, 4), ("5x5", 5, 5)]
    configs = []
    for label, h, w in shapes:
        # need at least a 1-cell O border for interior nodes to have observed neighbours
        if h > nrows - 2 or w > ncols - 2:
            continue
        # place at a few positions: top-left, center-ish
        placements = {(0, 0)}
        placements.add(((nrows - h) // 2, (ncols - w) // 2))
        placements.add((nrows - h, ncols - w))
        for (r0, c0) in sorted(placements):
            U = rect_block(net, r0, c0, h, w)
            if len(U) == h * w:                 # fully inside grid
                configs.append((f"{label}@({r0},{c0})", U, h * w))
    return configs


if __name__ == "__main__":
    from .net_parser import parse_net
    from .movement_graph import assign_ou, build_movement_graph
    from .observe import approach_observed_mask

    net = parse_net("/home/dhlee/routing/5by5/5by5.net.xml")
    mg = build_movement_graph(net)
    print(f"{'config':>12} | {'n_int':>5} {'n_U':>4} | {'appr_obs':>10} | {'h dist':>12} | comps")
    print("-" * 62)
    for label, U, k in generate_cluster_configs(net):
        um = assign_ou(net, U)
        obs = approach_observed_mask(net, U)[um]
        h = mg.hop_to_observed(um)[um]
        comps = mg.component_labels(um).max() + 1
        hs = ",".join(f"{v}:{int((h==v).sum())}" for v in sorted(set(h.tolist())))
        print(f"{label:>12} | {k:>5} {int(um.sum()):>4} | "
              f"{int(obs.sum()):>4}/{int(um.sum()):<4} | {hs:>12} | {comps}")
