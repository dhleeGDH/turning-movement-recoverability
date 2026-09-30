"""Site-2 network and movement tables for the masked eight-intersection block.

The movement graph of site 2 is read from data/xc_feas_network.json, which records the
junctions, links and turning movements of the 34 signalized intersections of the block's
district, derived from the Xuancheng road network (see data/XUANCHENG_TABLES.md), together with
the corridor edges that join two signals across unsignalized road segments. The counts and
approach volumes come from data/xc_feas_arrays.npz.

    from src.xc_network import load_site2
"""
import json

import numpy as np

from .graph.movement_graph import build_movement_graph
from .graph.net_parser import Junction, Link, Movement, NetworkData

BASE = "data"


def load_site2(base=BASE):
    """dict(net, mg, Y, APPR, date, window, binmin, inter_of, core_ids, block) for site 2."""
    d = json.load(open(f"{base}/xc_feas_network.json"))
    junctions = {j: Junction(j, t, 0.0, 0.0) for j, t in d["junctions"].items()}
    links = {k: Link(k, a, b, n, length, speed) for k, (a, b, n, length, speed) in d["links"].items()}
    movements = [Movement(f, t, j, dr, n) for f, t, j, dr, n in d["movements"]]
    net = NetworkData(junctions, links, movements)
    mg = build_movement_graph(net)
    for i, j in d["corridor_edges"]:
        mg.G.add_edge(i, j)
        mg.edge_type[(min(i, j), max(i, j))] = "E2"
    a = np.load(f"{base}/xc_feas_arrays.npz", allow_pickle=True)
    return dict(net=net, mg=mg, Y=a["Y"].astype(float), APPR=a["APPR"].astype(float), date=a["date"],
                window=a["window"], binmin=int(a["binmin"]), inter_of=a["inter_of"],
                core_ids=set(d["core_ids"]), block=set(d["block"]))
