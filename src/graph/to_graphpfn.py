"""Convert the movement graph and one scenario into a GraphPFN GraphDataset.

Runs only inside the GraphPFN uv env (needs dgl/torch/graphpfn). Node = movement,
single scalar target = movement count, task_type = 'regression'.

Inductive masks:  train = O (observed intersections' movements, provide ICL context),
                  test  = U (query),  val = small held-out slice of O (for finetune LR search).

Feature design (NO own label / own count -> passes no-leakage):
  static:   turn-type one-hot(l,s,r); from/to bearing sin&cos; #lanes; link length
  neighbour: observed approach-link volume, observed exit-link volume (NaN if unobserved)
  anchor:   observed-approach x historical split (NaN if approach unobserved)
Continuous features go under FeatureArrays['num']; GraphPFN/LimiX handle NaNs.
"""

import math

import numpy as np

from .movement_graph import MovementGraph
from .net_parser import NetworkData


def static_features(net: NetworkData) -> np.ndarray:
    """[N, 7] turn one-hot(3) + from-bearing(sin,cos) + #lanes + length. Scenario-independent."""
    jm = net.junctions
    rows = []
    for m in net.movements:
        oneh = [float(m.dir == d) for d in ("l", "s", "r")]
        b = net.links[m.from_edge].bearing(jm)
        lanes = net.links[m.from_edge].num_lanes
        length = net.links[m.from_edge].length
        rows.append(oneh + [math.sin(b), math.cos(b), lanes, length])
    return np.asarray(rows, np.float32)


def assemble_features(static: np.ndarray, appr_vol: np.ndarray, exit_vol: np.ndarray,
                      obs_appr: np.ndarray, obs_exit: np.ndarray,
                      hist_split: np.ndarray,
                      hist_mean: np.ndarray | None = None,
                      demand_ratio: float | None = None,
                      routed_appr: np.ndarray | None = None,
                      routed_count: np.ndarray | None = None,
                      zero_groups: set | None = None) -> np.ndarray:
    """[N, d] node features using a finite 'value-or-0 + observed-flag' encoding for the
    neighbour volumes (source/sink makes them unobservable). Unobserved -> value 0 and
    flag 0, so the missing value never leaks and the transforms stay NaN-free. NO own
    label -> passes no-leakage.

    hist_mean / demand_ratio (optional): the interior signal is
    dominated by the scenario demand level, which GraphPFN failed to infer from context. It is
    injected explicitly: per-movement historical mean, the scenario demand ratio, and
    anchor2 = hist_mean * demand_ratio (the demand-scaled-historical predictor, ~0.61 skill).
    """
    zg = zero_groups or set()
    z = lambda a, g: (np.zeros_like(a) if g in zg else a)   # ablate a feature group -> zeros
    appr = z(np.where(obs_appr, appr_vol, 0.0), "neighbor")
    exit_ = z(np.where(obs_exit, exit_vol, 0.0), "neighbor")
    anchor = z(np.where(obs_appr, appr_vol * hist_split, 0.0), "neighbor")
    cols = [z(static, "static"), appr, obs_appr.astype(np.float32),
            exit_, obs_exit.astype(np.float32), anchor]
    if hist_mean is not None and demand_ratio is not None:
        n = static.shape[0]
        dr = np.asarray(demand_ratio, np.float32)
        if dr.ndim == 0:                              # scalar (global) -> broadcast
            dr = np.full(n, float(dr), np.float32)
        anchor2 = hist_mean * dr                      # per-node demand-scaled historical
        cols += [z(hist_mean.astype(np.float32), "anchor"),
                 z(dr, "anchor"), z(anchor2.astype(np.float32), "anchor")]
    if routed_appr is not None and routed_count is not None:
        cols += [z(routed_appr.astype(np.float32), "routed"),
                 z(routed_count.astype(np.float32), "routed")]
    return np.column_stack(cols).astype(np.float32)


def build_dgl_graph(mg: MovementGraph, N: int):
    import dgl
    import torch
    edges = list(mg.G.edges())
    if edges:
        u = np.array([e[0] for e in edges] + [e[1] for e in edges])
        v = np.array([e[1] for e in edges] + [e[0] for e in edges])
    else:
        u = v = np.array([], dtype=int)
    return dgl.graph((torch.tensor(u, dtype=torch.int32), torch.tensor(v, dtype=torch.int32)),
                     num_nodes=N)


def build_graphdataset(net: NetworkData, mg: MovementGraph, *,
                       y_counts: np.ndarray, appr_vol: np.ndarray, exit_vol: np.ndarray,
                       obs_appr: np.ndarray, obs_exit: np.ndarray, u_mask: np.ndarray,
                       hist_split: np.ndarray, static: np.ndarray,
                       val_frac: float = 0.15, seed: int = 0, name: str = "tm"):
    from graphpfn import GraphDataset

    N = len(net.movements)
    feats = assemble_features(static, appr_vol, exit_vol, obs_appr, obs_exit, hist_split)
    g = build_dgl_graph(mg, N)

    o_idx = np.nonzero(~u_mask)[0]
    val_mask = np.zeros(N, bool)
    if val_frac > 0:
        rng = np.random.default_rng(seed)
        val = rng.choice(o_idx, size=max(1, int(len(o_idx) * val_frac)), replace=False)
        val_mask[val] = True
    train_mask = (~u_mask) & (~val_mask)     # val_frac=0 -> train = all O, eval = U exactly

    masks = {"train": train_mask, "val": val_mask, "test": u_mask.copy()}
    return GraphDataset(name=name, graph=g,
                        features={"num": feats},
                        targets=y_counts.astype(np.float32),
                        masks=masks, task_type="regression")
