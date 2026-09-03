"""Parse a SUMO .net.xml into the static structures the movement graph needs.

This does not depend on sumolib. Plain xml.etree suffices, since only
junctions, normal edges (with lanes), and connections are needed.

Outputs (all keyed by SUMO ids):
  intersections : dict[jid -> Junction]   (traffic_light junctions = real intersections)
  links         : dict[eid -> Link]       (normal directed edges; internal ":..." excluded)
  movements     : list[Movement]          (one per unique (from_edge, to_edge), lanes aggregated)

A Movement is a graph node. Its turn type (l/s/r) comes straight from SUMO's
connection dir attribute, its intersection from link topology.
"""
from __future__ import annotations


import os
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

def _routing(scale="5by5"):
    """Location of the generated SUMO networks; set TMR_ROUTING to override."""
    return os.path.join(os.environ.get("TMR_ROUTING", os.path.join(os.getcwd(), "routing")), scale)



@dataclass
class Junction:
    id: str
    type: str
    x: float
    y: float

    @property
    def is_intersection(self) -> bool:
        # signalized junctions are the controllable intersections in these grids
        return self.type == "traffic_light"


@dataclass
class Link:
    id: str
    from_j: str
    to_j: str
    num_lanes: int
    length: float
    speed: float

    def bearing(self, jmap: dict[str, Junction]) -> float:
        """Compass-ish heading from tail to head junction, radians in (-pi, pi]."""
        a, b = jmap[self.from_j], jmap[self.to_j]
        return math.atan2(b.y - a.y, b.x - a.x)


@dataclass
class Movement:
    """One turning movement = graph node. Lanes are aggregated away."""
    from_edge: str
    to_edge: str
    intersection: str          # junction id where the turn happens
    dir: str                   # 'l' | 's' | 'r' (SUMO connection dir)
    n_lane_conns: int = 0      # how many lane-level connections were merged

    @property
    def key(self) -> tuple[str, str]:
        return (self.from_edge, self.to_edge)


@dataclass
class NetworkData:
    junctions: dict[str, Junction]
    links: dict[str, Link]
    movements: list[Movement]

    @property
    def intersections(self) -> dict[str, Junction]:
        return {j: v for j, v in self.junctions.items() if v.is_intersection}

    def movements_at(self, jid: str) -> list[Movement]:
        return [m for m in self.movements if m.intersection == jid]

    def summary(self) -> str:
        n_int = len(self.intersections)
        dirs = {}
        for m in self.movements:
            dirs[m.dir] = dirs.get(m.dir, 0) + 1
        return (f"intersections(tl)={n_int}  links={len(self.links)}  "
                f"movements={len(self.movements)}  dir={dirs}")


def _is_internal(eid: str) -> bool:
    return eid.startswith(":")


def parse_net(path: str | Path) -> NetworkData:
    root = ET.parse(str(path)).getroot()

    junctions: dict[str, Junction] = {}
    for j in root.findall("junction"):
        jid = j.get("id")
        jtype = j.get("type", "")
        if jtype == "internal":
            continue
        junctions[jid] = Junction(jid, jtype, float(j.get("x")), float(j.get("y")))

    links: dict[str, Link] = {}
    for e in root.findall("edge"):
        eid = e.get("id")
        if e.get("function") == "internal" or _is_internal(eid):
            continue
        lanes = e.findall("lane")
        if not lanes:
            continue
        length = max(float(l.get("length", 0.0)) for l in lanes)
        speed = max(float(l.get("speed", 0.0)) for l in lanes)
        links[eid] = Link(eid, e.get("from"), e.get("to"),
                          num_lanes=len(lanes), length=length, speed=speed)

    # aggregate lane-level connections into movements keyed by (from, to)
    agg: dict[tuple[str, str], Movement] = {}
    for c in root.findall("connection"):
        fe, te = c.get("from"), c.get("to")
        if fe is None or te is None or _is_internal(fe) or _is_internal(te):
            continue
        if fe not in links or te not in links:
            continue
        # the intersection is where 'from' ends and 'to' starts
        inter = links[fe].to_j
        assert links[te].from_j == inter, (
            f"connection {fe}->{te} not junction-consistent: "
            f"{links[fe].to_j} vs {links[te].from_j}")
        key = (fe, te)
        if key not in agg:
            agg[key] = Movement(fe, te, inter, c.get("dir", "?"), 0)
        agg[key].n_lane_conns += 1

    movements = list(agg.values())
    return NetworkData(junctions=junctions, links=links, movements=movements)


if __name__ == "__main__":
    import sys
    net = parse_net(sys.argv[1] if len(sys.argv) > 1
                    else os.path.join(_routing(), "5by5.net.xml"))
    print(net.summary())
    # per-intersection movement counts
    from collections import Counter
    per = Counter(m.intersection for m in net.movements)
    vals = sorted(per.values())
    print(f"movements/intersection: min={vals[0]} max={vals[-1]} "
          f"median={vals[len(vals)//2]}  (n_intersections_with_mvmts={len(per)})")
    # show one intersection's movements
    jid = max(per, key=per.get)
    print(f"\nexample intersection {jid} ({per[jid]} movements):")
    for m in net.movements_at(jid)[:14]:
        print(f"  {m.from_edge:>8} -> {m.to_edge:<8} dir={m.dir} lanes={m.n_lane_conns}")
