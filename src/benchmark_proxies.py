"""Test whether the recoverability bound is an artifact of the gradient-boosted proxy.

Re-estimate skill* with several conditional-model classes of increasing flexibility. If the
FLEXIBLE proxies (GBM, MLP, random forest, GP) agree on a bound that a LOW-flexibility linear
model undershoots, the estimate has plateaued in flexibility and is the true irreducible-variance
ceiling, not an underfit; F sitting at that plateau is then genuine information saturation rather
than 'F matches another flexible model'.

    python3 -m src.benchmark_proxies
"""

import json

import numpy as np

from .graph.local_demand import local_demand_ratio
from .graph.movement_graph import assign_ou
from .graph.observe import approach_observed_mask
from .graph.scenarios import generate_cluster_configs
from .dart_pretrain import _load

GRIDS = ["6by6", "5by5", "9by9"]           # 9by9 is the headline
F_SKILL = {"6by6": None, "5by5": None, "9by9": 0.66}


def _features(scale):
    D = _load(scale, 15, 0)
    net, mg = D["net"], D["mg"]
    allc = generate_cluster_configs(net)
    cfgs = ([c for c in allc if "4x4" in c[0]] or [c for c in allc if "2x2" in c[0]] or allc)
    real = set(cfgs[0][1]); um = assign_ou(net, real); idx = np.nonzero(um)[0]
    oa = approach_observed_mask(net, real); hist = D["hist_mean"]
    scen = np.concatenate([D["train_scen"], D["eval_scen"]])
    X, y, naive = [], [], []
    for s in scen:
        s = int(s)
        r = local_demand_ratio(mg, oa, D["APPR"][s], D["demand_ref"])
        for m in idx:
            X.append([*D["static"][m], hist[m], r[m]]); y.append(D["Y"][s][m]); naive.append(hist[m])
    return (np.asarray(X, float), np.asarray(y, float), np.asarray(naive, float),
            len(idx), len(scen))


def run(out="data/benchmark_proxies.json"):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def models():
        return {
            "linear-ridge (low flex)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "GBM": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05),
            "random-forest": RandomForestRegressor(n_estimators=300, n_jobs=-1),
            "MLP (deep)": make_pipeline(StandardScaler(),
                                        MLPRegressor(hidden_layer_sizes=(128, 128),
                                                     max_iter=800, early_stopping=True)),
            "GP (subsample)": make_pipeline(StandardScaler(),
                                            GaussianProcessRegressor(kernel=RBF() + WhiteKernel(),
                                                                     normalize_y=True, alpha=1e-6)),
        }
    res = {}
    rng = np.random.default_rng(0)
    for scale in GRIDS:
        X, y, naive, nq, nsc = _features(scale)
        gid = np.repeat(np.arange(nsc), nq)
        row = {}
        for name, _ in models().items():
            pred = np.zeros_like(y)
            for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(nsc)):
                trm = np.isin(gid, tr); tem = np.isin(gid, te)
                est = models()[name]
                Xtr, ytr = X[trm], y[trm]
                if name.startswith("GP") and Xtr.shape[0] > 1500:      # GP is O(n^3)
                    sub = rng.choice(Xtr.shape[0], 1500, replace=False); Xtr, ytr = Xtr[sub], ytr[sub]
                est.fit(Xtr, ytr); pred[tem] = est.predict(X[tem])
            row[name] = round(float(1 - np.sqrt(np.mean((pred - y) ** 2))
                                    / np.sqrt(np.mean((naive - y) ** 2))), 3)
        res[scale] = row
        print(f"{scale}: " + "  ".join(f"{k.split()[0]}={v}" for k, v in row.items())
              + (f"   [F={F_SKILL[scale]}]" if F_SKILL[scale] else ""))
    json.dump(res, open(out, "w"), indent=2)
    print("\nsaved", out)
    print("Interpretation: if the flexible proxies (GBM/RF/MLP/GP) cluster above linear-ridge and")
    print("agree with each other, the bound has plateaued in flexibility and F sits at the true ceiling.")


if __name__ == "__main__":
    run()
