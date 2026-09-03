"""Build the movement graph: nodes = movements, edges = E1 (intra) + E2 (corridor).

Also handles O/U assignment (which intersections are observed) and the missing
connected components that cluster-aware attention (module B) will key on.
"""
from __future__ import annotations


from dataclasses import dataclass

import os
import networkx as nx
import numpy as np

from .net_parser import NetworkData

def _routing(scale="5by5"):
    """Location of the generated SUMO networks; set TMR_ROUTING to override."""
    return os.path.join(os.environ.get("TMR_ROUTING", os.path.join(os.getcwd(), "routing")), scale)



@dataclass
class MovementGraph:
    net: NetworkData
    G: nx.Graph                       # nodes = movement index, edges = E1 | E2
    node_index: dict[tuple[str, str], int]
    inter_of_node: np.ndarray         # [n_nodes] intersection id (object array)
    edge_type: dict                   # (u,v) -> 'E1' | 'E2'

    @property
    def n_nodes(self) -> int:
        return self.net.movements.__len__()

    def component_labels(self, u_mask: np.ndarray) -> np.ndarray:
        """Connected-component id for each U node over the subgraph induced by U.

        O nodes get label -1. Two U nodes share a label iff connected through
        E1|E2 using only U nodes -> this is the cluster-attention scope.
        """
        labels = np.full(self.n_nodes, -1, dtype=np.int64)
        u_nodes = set(np.nonzero(u_mask)[0].tolist())
        sub = self.G.subgraph(u_nodes)
        for cid, comp in enumerate(nx.connected_components(sub)):
            for n in comp:
                labels[n] = cid
        return labels

    def hop_to_observed(self, u_mask: np.ndarray) -> np.ndarray:
        """Per U node: hop distance to the nearest O node over E1|E2. O nodes -> 0.

        This is the blueprint's h axis (2.2.1). Multi-source BFS from all O nodes.
        """
        from collections import deque
        h = np.full(self.n_nodes, -1, dtype=np.int64)
        dq = deque()
        for n in range(self.n_nodes):
            if not u_mask[n]:
                h[n] = 0
                dq.append(n)
        while dq:
            n = dq.popleft()
            for nb in self.G.neighbors(n):
                if h[nb] == -1:
                    h[nb] = h[n] + 1
                    dq.append(nb)
        return h


def build_movement_graph(net: NetworkData) -> MovementGraph:
    node_index = {m.key: i for i, m in enumerate(net.movements)}
    inter_of_node = np.array([m.intersection for m in net.movements], dtype=object)

    G = nx.Graph()
    G.add_nodes_from(range(len(net.movements)))
    edge_type: dict = {}

    # E1: movements sharing an intersection form a clique
    by_inter: dict[str, list[int]] = {}
    for i, m in enumerate(net.movements):
        by_inter.setdefault(m.intersection, []).append(i)
    for nodes in by_inter.values():
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                u, v = nodes[a], nodes[b]
                G.add_edge(u, v)
                edge_type[(u, v)] = "E1"

    # E2: corridor flow -- m.to_edge feeds m'.from_edge
    from_by_edge: dict[str, list[int]] = {}
    for i, m in enumerate(net.movements):
        from_by_edge.setdefault(m.from_edge, []).append(i)
    for i, m in enumerate(net.movements):
        for j in from_by_edge.get(m.to_edge, []):
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if not G.has_edge(*key):
                G.add_edge(*key)
                edge_type[key] = "E2"

    return MovementGraph(net=net, G=G, node_index=node_index,
                         inter_of_node=inter_of_node, edge_type=edge_type)


def assign_ou(net: NetworkData, u_intersections: set[str]) -> np.ndarray:
    """Boolean U mask over movement nodes: True iff the movement's intersection is U."""
    return np.array([m.intersection in u_intersections for m in net.movements])


if __name__ == "__main__":
    from collections import Counter
    from .net_parser import parse_net

    net = parse_net(os.path.join(_routing(), "5by5.net.xml"))
    mg = build_movement_graph(net)
    et = Counter(mg.edge_type.values())
    print(f"nodes={mg.n_nodes}  edges={mg.G.number_of_edges()}  by_type={dict(et)}")
    deg = [d for _, d in mg.G.degree()]
    print(f"degree: min={min(deg)} max={max(deg)} mean={np.mean(deg):.1f}")

    # single-intersection missing (a center one) -> one component, h up to interior
    inters = sorted(net.intersections)
    center = inters[len(inters) // 2]
    um = assign_ou(net, {center})
    lab = mg.component_labels(um)
    h = mg.hop_to_observed(um)
    print(f"\nsingle U={center}: |U nodes|={int(um.sum())} "
          f"components={lab.max()+1} maxhop={h[um].max()}")

    # clustered missing: a 2x2 block of intersections
    block = set(inters[:2] + inters[5:7])
    um2 = assign_ou(net, block)
    lab2 = mg.component_labels(um2)
    print(f"clustered U={sorted(block)}: |U nodes|={int(um2.sum())} "
          f"components={lab2.max()+1} maxhop={mg.hop_to_observed(um2)[um2].max()}")
