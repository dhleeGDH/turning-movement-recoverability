"""The four 2x2 corner blocks of the site-1 grid, each with its deep (outer-corner) intersection.

The deep corner is the block intersection whose grid neighbours all lie inside the block, so that
with the block masked its movements sit at depth two or more.
"""
from .bucheon import GRID_LAYOUT


def deep_corner_blocks():
    """{tag: (set of four intersections, deep corner)} for TL, TR, BL, BR."""
    nr = max(r for r, _ in GRID_LAYOUT.values()) + 1
    nc = max(c for _, c in GRID_LAYOUT.values()) + 1
    pos2node = {(r, c): f"J{n}" for n, (r, c) in GRID_LAYOUT.items()}
    blocks = {}
    for (r0, c0, tag) in [(0, 0, "TL"), (0, nc - 2, "TR"), (nr - 2, 0, "BL"), (nr - 2, nc - 2, "BR")]:
        block = {f"J{n}" for n, (r, c) in GRID_LAYOUT.items()
                 if r in (r0, r0 + 1) and c in (c0, c0 + 1)}
        deep = None
        for n, (r, c) in GRID_LAYOUT.items():
            if f"J{n}" not in block:
                continue
            outside = sum(1 for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                          if (r + dr, c + dc) in pos2node and pos2node[(r + dr, c + dc)] not in block)
            if outside == 0:
                deep = f"J{n}"
        blocks[tag] = (block, deep)
    return blocks
