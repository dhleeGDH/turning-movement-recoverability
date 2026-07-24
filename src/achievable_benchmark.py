"""Does the estimated ceiling move when the proxy conditions on the
full feature vector z_m instead of the two-variable summary [hist_mean, demand_ratio]?

E[Var(x|summary)] >= E[Var(x|o)], so a two-variable-summary proxy can only understate the
true ceiling. Re-estimating skill* with proxies conditioned on the full z_m (static, the
observed approach/exit volumes and their observed-flags, the anchor, hist_mean, demand_ratio,
and anchor2) tests whether the diffused demand ratio is a lossy summary of the observables. If
the ceiling is unchanged, the saturation argument is strengthened; if it rises, F has headroom.

    python3 -m src.achievable_benchmark
"""

import json

import numpy as np

from .graph.local_demand import local_demand_ratio
from .graph.movement_graph import assign_ou
from .graph.scenarios import generate_cluster_configs
from .graph.to_graphpfn import assemble_features
from .dart_pretrain import _load, _obs

GRIDS = ["6by6", "5by5", "9by9"]
F_SKILL = {"6by6": None, "5by5": None, "9by9": 0.66}
# two-variable-summary ceiling from src.benchmark_proxies (GBM / random-forest)
SUMMARY_CEIL = {"6by6": (0.46, 0.436), "5by5": (0.521, 0.492), "9by9": (0.679, 0.681)}


def _cluster(net):
    allc = generate_cluster_configs(net)
    cfgs = ([c for c in allc if "4x4" in c[0]] or [c for c in allc if "2x2" in c[0]] or allc)
    return set(cfgs[0][1])


def _full_features(scale):
    """Return (X_full [S*Q, d], y, naive, Q, S) with X_full the full z_m at query nodes."""
    D = _load(scale, 15, 0)
    net, mg = D["net"], D["mg"]
    missing = _cluster(net)
    um = assign_ou(net, missing); idx = np.nonzero(um)[0]
    oa, oe = _obs(net, missing)
    scen = np.concatenate([D["train_scen"], D["eval_scen"]])
    X, y, naive = [], [], []
    for s in scen:
        s = int(s)
        ratio = local_demand_ratio(mg, oa, D["APPR"][s], D["demand_ref"])
        f = assemble_features(D["static"], D["APPR"][s], D["EXIT"][s], oa, oe,
                              D["hist_split"], hist_mean=D["hist_mean"], demand_ratio=ratio)
        X.append(f[idx]); y.append(D["Y"][s][idx]); naive.append(D["hist_mean"][idx])
    return (np.concatenate(X), np.concatenate(y), np.concatenate(naive),
            len(idx), len(scen))


def _full_features_cluster(D, missing):
    """Full z_m query rows for a specific missing cluster, over all scenarios."""
    net, mg = D["net"], D["mg"]
    um = assign_ou(net, missing); idx = np.nonzero(um)[0]
    oa, oe = _obs(net, missing)
    scen = np.concatenate([D["train_scen"], D["eval_scen"]])
    X, y, naive = [], [], []
    for s in scen:
        s = int(s)
        ratio = local_demand_ratio(mg, oa, D["APPR"][s], D["demand_ref"])
        f = assemble_features(D["static"], D["APPR"][s], D["EXIT"][s], oa, oe,
                              D["hist_split"], hist_mean=D["hist_mean"], demand_ratio=ratio)
        X.append(f[idx]); y.append(D["Y"][s][idx]); naive.append(D["hist_mean"][idx])
    return np.concatenate(X), np.concatenate(y), np.concatenate(naive)


def held_out_cluster(scale="9by9", out="data/recover_bound_holdcluster.json"):
    """Recompute the full-feature ceiling under the held-out-cluster protocol used
    for scoring, rather than within-site scenario CV. Leave-one-4x4-placement-out: train the
    proxy on the other placements' query rows, predict the held-out placement's."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    D = _load(scale, 15, 0); net = D["net"]
    placements = [c[1] for c in generate_cluster_configs(net) if "4x4" in c[0]]
    feats = [_full_features_cluster(D, set(p)) for p in placements]
    ypool = np.concatenate([f[1] for f in feats]); npool = np.concatenate([f[2] for f in feats])
    pred = np.zeros_like(ypool); off = 0
    for k in range(len(placements)):
        Xte, yte, _ = feats[k]
        tr = [j for j in range(len(placements)) if j != k]
        Xtr = np.concatenate([feats[j][0] for j in tr]); ytr = np.concatenate([feats[j][1] for j in tr])
        est = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)
        est.fit(Xtr, ytr); pred[off:off + len(yte)] = est.predict(Xte); off += len(yte)
    ceil_hoc = float(1 - np.sqrt(np.mean((pred - ypool) ** 2)) / np.sqrt(np.mean((npool - ypool) ** 2)))
    res = {"scale": scale, "n_placements": len(placements),
           "held_out_cluster_ceiling": round(ceil_hoc, 3),
           "within_site_ceiling": 0.708, "summary_ceiling": 0.679, "F_skill": 0.66}
    print(f"[{scale}] full-feature ceiling: within-site={res['within_site_ceiling']}  "
          f"held-out-cluster={res['held_out_cluster_ceiling']}  (summary={res['summary_ceiling']}, F={res['F_skill']})")
    print("Gap from within-site to held-out-cluster ceiling = protocol; residual over F = genuine headroom.")
    json.dump(res, open(out, "w"), indent=2); print("saved", out)
    return res


def run(out="data/achievable_benchmark.json"):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import KFold

    def models():
        return {"GBM": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05),
                "random-forest": RandomForestRegressor(n_estimators=300, n_jobs=-1)}
    res = {}
    for scale in GRIDS:
        X, y, naive, Q, S = _full_features(scale)
        gid = np.repeat(np.arange(S), Q)
        row = {"n_features": int(X.shape[1])}
        for name in models():
            pred = np.zeros_like(y)
            for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(S)):
                trm = np.isin(gid, tr); tem = np.isin(gid, te)
                est = models()[name]; est.fit(X[trm], y[trm]); pred[tem] = est.predict(X[tem])
            row[name] = round(float(1 - np.sqrt(np.mean((pred - y) ** 2))
                                    / np.sqrt(np.mean((naive - y) ** 2))), 3)
        sg, sr = SUMMARY_CEIL[scale]
        row["summary_GBM"] = sg; row["summary_RF"] = sr
        row["delta_GBM"] = round(row["GBM"] - sg, 3)
        res[scale] = row
        print(f"{scale}: full-z_m GBM={row['GBM']} RF={row['random-forest']} "
              f"(d={row['n_features']})  vs 2-var summary GBM={sg} RF={sr}  "
              f"delta_GBM={row['delta_GBM']:+.3f}" + (f"  [F={F_SKILL[scale]}]" if F_SKILL[scale] else ""))
    json.dump(res, open(out, "w"), indent=2)
    print("\nsaved", out)
    print("If delta ~ 0, the diffused demand ratio is a sufficient summary of the observables "
          "for an unobserved query node and the ceiling does not move.")


if __name__ == "__main__":
    run()
