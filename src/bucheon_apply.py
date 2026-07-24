"""Apply the method to real Bucheon data (objective: does SUMO-trained F transfer?).

Bucheon 4x4, 33 hourly windows (= real 'scenarios' with demand variation). Leave-one-
intersection-out: mask each intersection, estimate its turning movements from observed
neighbours, per window; skill vs naive A (Bucheon per-movement historical mean). Ladder:
  A (naive)          : Bucheon historical mean
  anchor-only        : demand-scaled historical (hist_mean x demand_ratio), no NN
  E (vanilla GraphPFN): ICL with anchor features, NO SUMO training
  F (SUMO-trained)   : GraphPFN domain-pretrained on synthetic grids, applied zero-shot

    PYTHONPATH=. uv run --project external/graphpfn python -m src.bucheon_apply
"""

import collections
import json

import numpy as np

from .bucheon import GRID_LAYOUT, DIRS, OPP, build_bucheon_net
from .bucheon_rho import LEFT, RIGHT, INV, parse_camera, _neighbor
from .graph.movement_graph import build_movement_graph
from .graph.to_graphpfn import assemble_features, build_dgl_graph, static_features


def build_windowed(net, window_json="data/bucheon_cam_turn_window.json"):
    wd = json.load(open(window_json))
    windows = wd["windows"]; data = wd["data"]
    legdir = {int(k): {int(a): d for a, d in v.items()}
              for k, v in json.load(open("data/bucheon_legdir.json")).items()}
    key = {(m.from_edge, m.to_edge): i for i, m in enumerate(net.movements)}
    N = len(net.movements)
    Y = np.zeros((len(windows), N))
    for wi, w in enumerate(windows):
        for cam, byw in data.items():
            turns = byw.get(w)
            if not turns:
                continue
            I, a = parse_camera(cam); da = legdir.get(I, {}).get(a)
            if da is None:
                continue
            src = _neighbor(I, da)
            for mv, c in turns.items():
                b = OPP[da] if mv == "t" else LEFT[OPP[da]] if mv == "l" else RIGHT[OPP[da]]
                dst = _neighbor(I, b)
                if src is None or dst is None:
                    continue
                k = key.get((f"J{src}->J{I}", f"J{I}->J{dst}"))
                if k is not None:
                    Y[wi, k] += c
    return Y, windows


def _link_vol(net, Y):  # from-link and to-link volume per movement, per window
    fromm = collections.defaultdict(list)
    for i, m in enumerate(net.movements):
        fromm[m.from_edge].append(i)
    APPR = np.zeros_like(Y); EXIT = np.zeros_like(Y)
    to_edge = [m.to_edge for m in net.movements]
    for e, idx in fromm.items():
        v = Y[:, idx].sum(1)
        APPR[:, idx] = v[:, None]
        cols = [k for k, te in enumerate(to_edge) if te == e]
        if cols:
            EXIT[:, cols] = v[:, None]
    return APPR, EXIT


def run(sumo_scale="9by9", n_steps=800, device="cuda:0", n_windows=None):
    import torch
    from graphpfn import GraphDataset
    from graphpfn.inference.preprocessing import NumTransform, apply_num_transform, impute_nans, standardize_targets
    from graphpfn.inference.util import apply_model
    from .models import cluster_attention as CA
    from .dart_pretrain import _fresh_model, _load, _prep, train
    CA.patch_encoder(); CA.clear()
    dev = torch.device(device)

    # ---- Bucheon data ----
    net = build_bucheon_net(); mg = build_movement_graph(net); N = len(net.movements)
    Yb, windows = build_windowed(net)
    if n_windows:
        Yb = Yb[:n_windows]
    APPR, EXIT = _link_vol(net, Yb)
    static = static_features(net)
    graph = build_dgl_graph(mg, N).to(dev)
    inters = sorted(net.intersections)
    W = Yb.shape[0]
    print(f"[Bucheon] {len(inters)} intersections, {N} movements, {W} windows")

    HIST_MEAN = Yb.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        HIST_SPLIT = np.nan_to_num(np.where(APPR > 0, Yb / APPR, 0).mean(0))
    DEMAND_REF = float(APPR.mean())

    def prep_feats(missing, wi, train_mask_np):
        from .graph.observe import approach_observed_mask, exit_observed_mask
        oa = approach_observed_mask(net, missing); oe = exit_observed_mask(net, missing)
        hist_mean = HIST_MEAN; hist_split = HIST_SPLIT; demand_ref = DEMAND_REF
        obs_av = APPR[wi][oa]
        ratio = (float(obs_av.mean()) if obs_av.size else demand_ref) / (demand_ref + 1e-9)
        f = assemble_features(static, APPR[wi], EXIT[wi], oa, oe, hist_split,
                              hist_mean=hist_mean, demand_ratio=ratio)
        f = impute_nans(apply_num_transform(f, NumTransform.QUANTILE_NORMAL, 0))
        return torch.tensor(np.asarray(f), dtype=torch.float32, device=dev), hist_mean

    # ---- train F on SUMO (transfer source) ----
    print(f"training F on SUMO {sumo_scale} ({n_steps})...")
    Ds = _load(sumo_scale, 15, 0)
    from .graph.to_graphpfn import build_dgl_graph as bdg
    gs = bdg(Ds["mg"], Ds["N"]).to(dev)
    n_feat = _prep(Ds, set([sorted(Ds['net'].intersections)[0]]), int(Ds["train_scen"][0]),
                   np.ones(Ds["N"], bool), dev).shape[-1]
    modelF = train(_fresh_model(dev, n_feat), {sumo_scale: Ds}, {sumo_scale: gs}, False,
                   n_steps, 2e-5, np.random.default_rng(0), dev, verbose=False)
    modelE = _fresh_model(dev, n_feat)   # vanilla GraphPFN (no SUMO training)

    # ---- Bucheon LOIO ----
    def eval_model(model):
        seA, seM, seAnc = [], [], []
        model.eval()
        with torch.no_grad():
            for I in inters:
                um = np.array([m.intersection == I for m in net.movements])
                idx = np.nonzero(um)[0]; tmask = ~um
                a_own = Yb.mean(0)[idx]
                ds = GraphDataset(name="b", graph=graph, features={"num": np.zeros((N, 1), np.float32)},
                                  targets=Yb[0].astype(np.float32),
                                  masks={"train": tmask, "val": np.zeros(N, bool), "test": um.copy()},
                                  task_type="regression")
                for wi in range(W):
                    feats, hist_mean = prep_feats({I}, wi, tmask)
                    yc = torch.tensor(Yb[wi][tmask], dtype=torch.float32, device=dev)
                    yc_std, stats = standardize_targets(y_train=yc)
                    pred = apply_model(model=model, dataset=ds, graph=graph, features=feats,
                                       y_train=yc_std, train_mask=torch.tensor(tmask, device=dev),
                                       regression_target_stats=stats, amp=True, device=dev)
                    true = Yb[wi][idx]
                    seA.append((a_own - true) ** 2); seM.append((np.asarray(pred)[idx] - true) ** 2)
        seA = np.concatenate(seA); seM = np.concatenate(seM)
        return 1 - np.sqrt(seM.mean()) / np.sqrt(seA.mean())

    # anchor-only baseline (no NN)
    def anchor_skill():
        hist_mean = Yb.mean(0); demand_ref = float(APPR.mean())
        seA, seM = [], []
        from .graph.observe import approach_observed_mask
        for I in inters:
            um = np.array([m.intersection == I for m in net.movements]); idx = np.nonzero(um)[0]
            oa = approach_observed_mask(net, {I})
            with np.errstate(divide="ignore", invalid="ignore"):
                hist_split = np.nan_to_num(np.where(APPR > 0, Yb / APPR, 0).mean(0))
            a_own = hist_mean[idx]
            for wi in range(W):
                ratio = (float(APPR[wi][oa].mean()) if oa.any() else demand_ref)/(demand_ref+1e-9)
                anc = (hist_mean * ratio)[idx]
                true = Yb[wi][idx]
                seA.append((a_own-true)**2); seM.append((anc-true)**2)
        seA=np.concatenate(seA); seM=np.concatenate(seM)
        return 1-np.sqrt(seM.mean())/np.sqrt(seA.mean())

    print("\n== Bucheon real-data leave-one-intersection-out (skill vs naive A) ==")
    print(f"  anchor-only (no NN)     : {anchor_skill():.3f}")
    print(f"  E (vanilla GraphPFN)    : {eval_model(modelE):.3f}")
    print(f"  F (SUMO-trained)        : {eval_model(modelF):.3f}")


if __name__ == "__main__":
    run()
