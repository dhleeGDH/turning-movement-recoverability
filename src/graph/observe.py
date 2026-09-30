"""Observation model: given an O/U configuration, which link volumes are observable,
and which movement-node approach volumes the conservation anchor B can actually use.

A directed link e (Ju -> Jd) is observed iff its UPSTREAM intersection Ju is O
(Ju measures its own outflow onto e), or e enters from a boundary (dead_end)
treated as an externally counted network entry.

For movement m at U intersection I, B needs m's approach link fl (fl: Ju -> I). B can
use the *observed* current-scenario approach volume iff fl is observed, i.e. iff the
upstream neighbour Ju is O (or boundary). Inside a missing cluster, interior approaches
come from other U intersections and are therefore unobserved -> B must fall back to a
historical approach mean, collapsing toward the naive floor A. This is the gap module B
(cluster-aware attention, model G) is meant to recover.
"""
from __future__ import annotations


import os
import numpy as np

from .net_parser import NetworkData

def _routing(scale="5by5"):
    """Location of the SUMO networks written by od_gen; set TMR_ROUTING to override."""
    return os.path.join(os.environ.get("TMR_ROUTING", os.path.join(os.getcwd(), "routing")), scale)



def link_observed(net: NetworkData, eid: str, u_intersections: set[str],
                  cordon_observed: bool = True) -> bool:
    up = net.links[eid].from_j
    j = net.junctions.get(up)
    if j is None:
        return cordon_observed           # unknown tail -> boundary; observed only if cordon_observed
    if not j.is_intersection:
        return cordon_observed           # dead_end / boundary entry: externally counted iff cordon_observed
    return up not in u_intersections     # observed iff upstream intersection is O


def approach_observed_mask(net: NetworkData, u_intersections: set[str],
                           cordon_observed: bool = True) -> np.ndarray:
    """Per movement node: True iff its approach (from) link volume is observable.
    cordon_observed=False treats network-boundary (external) inflows as UNobserved, i.e. a
    truly uninstrumented district whose external inflows are not externally counted."""
    return np.array([link_observed(net, m.from_edge, u_intersections, cordon_observed)
                     for m in net.movements])


def exit_observed_mask(net: NetworkData, u_intersections: set[str]) -> np.ndarray:
    """Per movement node: True iff its exit (to) link volume is observable (downstream O/boundary)."""
    out = []
    for m in net.movements:
        jd = net.links[m.to_edge].to_j
        jdj = net.junctions.get(jd)
        out.append((jdj is None) or (not jdj.is_intersection) or (jd not in u_intersections))
    return np.array(out)


if __name__ == "__main__":
    from .net_parser import parse_net
    from .movement_graph import assign_ou, build_movement_graph

    net = parse_net(os.path.join(_routing(), "5by5.net.xml"))
    mg = build_movement_graph(net)
    inters = sorted(net.intersections, key=lambda j: (net.junctions[j].x, net.junctions[j].y))

    for label, block in [("single", inters[12:13]),
                         ("2x1", inters[12:14]),
                         ("center 3", [inters[6], inters[12], inters[18]])]:
        U = set(block)
        um = assign_ou(net, U)
        obs = approach_observed_mask(net, U)
        h = mg.hop_to_observed(um)
        uobs = obs[um]
        print(f"{label:>8}: |U nodes|={int(um.sum())} "
              f"approaches observed={int(uobs.sum())}/{int(um.sum())} "
              f"h(U): {sorted(set(h[um].tolist()))}")
