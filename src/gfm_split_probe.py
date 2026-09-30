"""Turn-split probe of the graph foundation model (DART on GraphPFN).

The split of a masked movement is its predicted count over the predicted counts on the same
approach link. The gain is RI(model split x v) - RI(historical split x v), where v is the
approach volume entering the target and the naive baseline is the historical mean; the interval is
a bootstrap over the resampling units (20000 draws) with a Student-t interval alongside.

  synthetic : 9x9 grid, units = 4 seeds x 3 held-out 4x4 blocks; the model is trained per seed on
              that seed's training scenarios (600 steps, lr 2e-5) and applied in context to each
              block; evaluation scenarios never enter training.
  site 1    : the model is trained once on the 9x9 grid (800 steps, seed 0) and applied zero-shot
              to each leave-one-intersection-out fold of Bucheon at 15-min, 16 folds, with v the
              observed and the diffused approach volume.

The run is deterministic: it restarts itself with PYTHONHASHSEED=0, seeds the NumPy, Python and
torch generators at the start of each part and seed, and enables torch deterministic algorithms.
Scenario data are read from $TMR_ROUTING (see README).

    PYTHONPATH=. python -m src.gfm_split_probe      # writes data/m52_gfm_split_probe.json
"""
import os, sys
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, "-B", "-m", "src.gfm_split_probe"])
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, random, time
import numpy as np, torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


RNG = np.random.default_rng(0)          # bootstrap generator of _cis, drawn in call order


def _cis(gains, n_boot=20000):
    """Bootstrap-percentile and Student-t 95% intervals of the mean gain over the resampling units."""
    g = np.asarray(gains, float)
    n = len(g)
    mean = float(g.mean())
    idx = RNG.integers(0, n, size=(n_boot, n))
    boot_means = g[idx].mean(1)
    lo_b, hi_b = np.percentile(boot_means, [2.5, 97.5])
    sd = float(g.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    try:
        from scipy.stats import t as _t
        tcrit = float(_t.ppf(0.975, n - 1)) if n > 1 else 0.0
    except Exception:
        tcrit = 1.96
    lo_t, hi_t = mean - tcrit * se, mean + tcrit * se
    return {"point_gain": round(mean, 4), "n_units": n, "sd_over_units": round(sd, 4), "se": round(se, 4),
            "ci95_bootstrap": [round(float(lo_b), 4), round(float(hi_b), 4)],
            "ci95_t": [round(float(lo_t), 4), round(float(hi_t), 4)],
            "pos_frac": round(float((g > 0).mean()), 3), "per_unit_gain": [round(float(x), 4) for x in g]}


def _syn_features(D, oa, s):
    """Anchor features of a synthetic scenario and the diffused approach volume v_in."""
    mg = D["mg"]
    appr_hist = D["hist_mean"] / np.clip(D["hist_split"], 1e-9, None)
    v_in = diffuse_observed(mg, oa, D["APPR"][s])
    v_in_h = diffuse_observed(mg, oa, appr_hist)
    ratio = v_in / np.clip(v_in_h, 1e-9, None)
    logv = np.log1p(v_in)
    return np.column_stack([ratio, logv, D["hist_split"], D["static"]]), v_in


def _seed_all(seed):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
from . import dart_pretrain as TF
from .graph.scenarios import generate_cluster_configs
from .graph.movement_graph import assign_ou, build_movement_graph
from .graph.observe import approach_observed_mask, exit_observed_mask
from .bucheon import build_bucheon_net
from .bucheon_apply import build_windowed, _link_vol
from .graph.local_demand import diffuse_observed
from .graph.to_graphpfn import assemble_features, build_dgl_graph, static_features
from .models import cluster_attention as CA
from .dart_pretrain import _fresh_model, _load, _prep, train
from graphpfn import GraphDataset
from graphpfn.inference.preprocessing import NumTransform, apply_num_transform, impute_nans, standardize_targets
from graphpfn.inference.util import apply_model

OUT = "data/m52_gfm_split_probe.json"


def synthetic():
    TF.CA.patch_encoder(); dev = torch.device("cuda:0")
    gains, units, T0 = [], [], time.time()
    for seed in (0, 1, 2, 3):
        _seed_all(seed)
        D = TF._load("9by9", 15, seed); net = D["net"]
        g = TF.build_dgl_graph(D["mg"], D["N"]).to(dev)
        confs = [c for c in generate_cluster_configs(net) if "4x4" in c[0]]
        nf = TF._prep(D, set(confs[0][1]), int(D["train_scen"][0]), np.ones(D["N"], bool), dev).shape[-1]
        model = TF.train(TF._fresh_model(dev, nf), {"9by9": D}, {"9by9": g}, False, 600, 2e-5,
                         np.random.default_rng(seed), dev, verbose=False)
        for name, real_set, _ in confs:
            real_set = set(real_set)
            r = TF.eval_model(model, D, D["mg"], g, real_set, False, dev, return_arrays=True)
            um = assign_ou(net, real_set); qidx = np.nonzero(um)[0]
            oa = approach_observed_mask(net, real_set)
            appr_of = np.array([net.movements[k].from_edge for k in qidx])
            pr_all, base_all, vin_all, y_all, naive_all = [], [], [], [], []
            for j, s in enumerate(D["eval_scen"]):
                s = int(s)
                _, v_in = _syn_features(D, oa, s)
                pred = np.clip(r["preds"][j], 0, None)
                split = np.zeros_like(pred)
                for a in set(appr_of.tolist()):
                    sel = appr_of == a; tot = pred[sel].sum()
                    split[sel] = pred[sel] / tot if tot > 0 else D["hist_split"][qidx][sel]
                pr_all.append(split); base_all.append(D["hist_split"][qidx]); vin_all.append(v_in[qidx])
                y_all.append(D["Y"][s][qidx]); naive_all.append(D["hist_mean"][qidx])
            gs, bs, vi, yy, nv = (np.concatenate(x) for x in (pr_all, base_all, vin_all, y_all, naive_all))
            ri = lambda p: float(1 - np.sqrt(np.mean((p - yy) ** 2)) / np.sqrt(np.mean((nv - yy) ** 2)))
            gain = ri(gs * vi) - ri(bs * vi)
            gains.append(gain); units.append({"seed": seed, "block": name, "gain": round(gain, 4),
                                               "RI_gfm_split": round(ri(gs * vi), 4), "RI_hist_split": round(ri(bs * vi), 4),
                                               "DART_count_overall_RI": round(r["overall"], 4)})
            print(units[-1], flush=True)
        del model; torch.cuda.empty_cache()
    res = {"probe": "GFM (DART/GraphPFN) split, synthetic 9x9, leave-one-4x4-block-out x seed", "n_units": len(gains),
           "units": units, "ci": _cis(gains), "seconds": round(time.time() - T0, 1)}
    print(json.dumps(res["ci"]), res["seconds"])
    return res


def bucheon():
    T0 = time.time(); CA.patch_encoder(); CA.clear(); dev = torch.device("cuda:0"); _seed_all(0)
    net = build_bucheon_net(); mg = build_movement_graph(net); N = len(net.movements)
    Yb, windows = build_windowed(net, "data/bucheon_cam_turn_15min.json")
    APPR, EXIT = _link_vol(net, Yb); static = static_features(net); graph = build_dgl_graph(mg, N).to(dev)
    inter_of = np.array([m.intersection for m in net.movements]); inters = sorted(set(inter_of.tolist())); W = Yb.shape[0]
    HIST_MEAN = Yb.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        HIST_SPLIT_APPLY = np.nan_to_num(np.where(APPR > 0, Yb / APPR, 0).mean(0))
    DEMAND_REF = float(APPR.mean())
    hist_split = (Yb / np.clip(APPR, 1e-9, None)).mean(0); appr_hist = APPR.mean(0)   # historical split and approach volume

    def prep_feats(missing, wi):
        oa = approach_observed_mask(net, missing); oe = exit_observed_mask(net, missing)
        obs_av = APPR[wi][oa]; ratio = (float(obs_av.mean()) if obs_av.size else DEMAND_REF) / (DEMAND_REF + 1e-9)
        f = assemble_features(static, APPR[wi], EXIT[wi], oa, oe, HIST_SPLIT_APPLY, hist_mean=HIST_MEAN, demand_ratio=ratio)
        f = impute_nans(apply_num_transform(f, NumTransform.QUANTILE_NORMAL, 0))
        return torch.tensor(np.asarray(f), dtype=torch.float32, device=dev)

    Ds = _load("9by9", 15, 0); gs = build_dgl_graph(Ds["mg"], Ds["N"]).to(dev)
    n_feat = _prep(Ds, set([sorted(Ds["net"].intersections)[0]]), int(Ds["train_scen"][0]), np.ones(Ds["N"], bool), dev).shape[-1]
    model = train(_fresh_model(dev, n_feat), {"9by9": Ds}, {"9by9": gs}, False, 800, 2e-5, np.random.default_rng(0), dev, verbose=False)
    model.eval(); T1 = time.time()
    res = {"observed": [], "diffused": []}; units = []
    with torch.no_grad():
        for I in inters:
            um = inter_of == I; idx = np.nonzero(um)[0]; tmask = ~um
            if not um.any(): continue
            oa = approach_observed_mask(net, {I}); appr_of = np.array([net.movements[k].from_edge for k in idx])
            ds = GraphDataset(name="b", graph=graph, features={"num": np.zeros((N, 1), np.float32)}, targets=Yb[0].astype(np.float32),
                              masks={"train": tmask, "val": np.zeros(N, bool), "test": um.copy()}, task_type="regression")
            S, B, V_obs, V_dif, Yt = [], [], [], [], []
            for wi in range(W):
                yc = torch.tensor(Yb[wi][tmask], dtype=torch.float32, device=dev); yc_std, stats = standardize_targets(y_train=yc)
                pred = np.clip(np.asarray(apply_model(model=model, dataset=ds, graph=graph, features=prep_feats({I}, wi), y_train=yc_std,
                                                      train_mask=torch.tensor(tmask, device=dev), regression_target_stats=stats, amp=True, device=dev))[idx], 0, None)
                split = np.zeros_like(pred)
                for a in set(appr_of.tolist()):
                    sel = appr_of == a; tot = pred[sel].sum(); split[sel] = pred[sel] / tot if tot > 0 else hist_split[idx][sel]
                S.append(split); B.append(hist_split[idx]); V_obs.append(APPR[wi][idx]); V_dif.append(diffuse_observed(mg, oa, APPR[wi])[idx]); Yt.append(Yb[wi][idx])
            S, B, Vo, Vd, Yt = (np.concatenate(x) for x in (S, B, V_obs, V_dif, Yt))
            naive = np.tile(hist_split[idx] * appr_hist[idx], W)
            ri = lambda p: float(1 - np.sqrt(np.mean((p - Yt) ** 2)) / np.sqrt(np.mean((naive - Yt) ** 2)))
            g_o = ri(S * Vo) - ri(B * Vo); g_d = ri(S * Vd) - ri(B * Vd)
            res["observed"].append(g_o); res["diffused"].append(g_d)
            units.append({"intersection": str(I), "gain_observed": round(g_o, 4), "gain_diffused": round(g_d, 4)}); print(units[-1], flush=True)
    out = {"probe": "GFM (DART/GraphPFN, SUMO-trained, zero-shot) split, Bucheon 15-min LOIO", "units": units,
           "observed": _cis(res["observed"]), "diffused": _cis(res["diffused"]), "train_seconds": round(T1 - T0, 1), "seconds": round(time.time() - T0, 1)}
    print(json.dumps({k: out[k] for k in ("observed", "diffused")}))
    return out


if __name__ == "__main__":
    res = {"deterministic": True, "synthetic": synthetic(), "bucheon": bucheon(),
           "detection_floor_RI": {"synthetic": 0.028, "site1": 0.018}}
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)
