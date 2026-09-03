"""Extract per-scenario turning-movement counts and link volumes from vehRoute XML.

v1 = hourly aggregate over the whole sim window (sidesteps 0-2a time-resolution).
Both quantities come from the same source (vehicle routes), so at rho=0 they are
conservation-consistent by construction -- exactly what Go/No-Go 1 needs.

  movement_count[(from_edge,to_edge)] : vehicles making that turn (consecutive edge pair)
  link_volume[edge]                   : vehicles traversing that link
"""
from __future__ import annotations


import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from .net_parser import NetworkData

def _routing(scale="5by5"):
    """Location of the generated SUMO networks; set TMR_ROUTING to override."""
    return os.path.join(os.environ.get("TMR_ROUTING", os.path.join(os.getcwd(), "routing")), scale)



def extract_counts(route_xml: str | Path,
                   net: NetworkData) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    move: dict[tuple[str, str], int] = defaultdict(int)
    link: dict[str, int] = defaultdict(int)

    # iterparse keeps memory flat over thousands of vehicles
    for _, el in ET.iterparse(str(route_xml), events=("end",)):
        if el.tag != "route":
            continue
        edges = el.get("edges", "").split()
        for e in edges:
            link[e] += 1
        for a, b in zip(edges, edges[1:]):
            move[(a, b)] += 1
        el.clear()

    # keep only movements that exist in the network topology (drop TAZ/connector pairs)
    valid = {m.key for m in net.movements}
    move = {k: v for k, v in move.items() if k in valid}
    return dict(move), dict(link)


def counts_to_arrays(net: NetworkData,
                     move: dict[tuple[str, str], int]):
    """Align movement counts to net.movements order -> numpy array (missing=0)."""
    import numpy as np
    return np.array([move.get(m.key, 0) for m in net.movements], dtype=np.float64)


def exit_volume_per_movement(net: NetworkData, Y: "np.ndarray"):
    """[S,N] exit-link (to_edge) volume for each movement, derived from Y by conservation:
    volume on link e = sum of movements leaving e. APPR (from_pilot) is the from-link
    analogue; this is the to-link one, needed as an extra GraphPFN feature."""
    import numpy as np
    from collections import defaultdict
    from_map = defaultdict(list)
    for i, m in enumerate(net.movements):
        from_map[m.from_edge].append(i)
    to_edge = [m.to_edge for m in net.movements]
    vol = np.zeros_like(Y)
    for e, members in from_map.items():
        v = Y[:, members].sum(1)                       # volume on link e per scenario
        cols = [k for k, te in enumerate(to_edge) if te == e]
        if cols:
            vol[:, cols] = v[:, None]
    return vol


if __name__ == "__main__":
    import sys
    import numpy as np
    from .net_parser import parse_net

    base = _routing()
    net = parse_net(f"{base}/5by5.net.xml")
    rp = sys.argv[1] if len(sys.argv) > 1 else f"{base}/simData/vehRouteData/Route1704000479d1.xml"
    move, link = extract_counts(rp, net)

    y = counts_to_arrays(net, move)
    print(f"movements with count>0: {int((y>0).sum())}/{len(y)}  "
          f"total turns={int(y.sum())}  mean={y.mean():.1f} max={int(y.max())}")

    # conservation sanity: for each intersection, sum of movements OUT of an approach
    # link should ~equal that link's volume (minus vehicles ending mid-link).
    errs = []
    for jid, jv in net.intersections.items():
        for m in net.movements_at(jid):
            fl = m.from_edge
            out_sum = sum(move.get((fl, mm.to_edge), 0)
                          for mm in net.movements_at(jid) if mm.from_edge == fl)
            if link.get(fl, 0) > 0:
                errs.append(abs(out_sum - link[fl]) / link[fl])
    errs = np.array(errs)
    print(f"approach conservation |sum(turns)-linkvol|/linkvol: "
          f"mean={errs.mean():.3f} median={np.median(errs):.3f} "
          f"(0 = perfect; >0 from mid-link origins/dests)")
