"""Calibration of the bound estimator on a known generative process.

Build a synthetic target x = mu(o) + eps with an analytically known conditional variance
Var(eps|o) = sigma^2, using the real 9x9 observables o. The true recoverability bound is then
skill*_true = 1 - sigma / RMSE(naive), computable exactly. Re-estimate skill* with the GBM proxy
(the paper's procedure) and with a linear model. If GBM recovers skill*_true while the linear model
undershoots it, the proxy is calibrated and flexibility is necessary, so the anchor sitting at the
bound is not a linear-underfit artifact.

    python3 -m src.gaussian_calibration
"""

import json

import numpy as np

from .graph.local_demand import local_demand_ratio
from .graph.movement_graph import assign_ou
from .graph.observe import approach_observed_mask
from .graph.scenarios import generate_cluster_configs
from .dart_pretrain import _load


def run(scale="9by9", target_bound=0.65, out="data/gaussian_calibration.json"):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)

    D = _load(scale, 15, 0); net, mg = D["net"], D["mg"]
    allc = generate_cluster_configs(net)
    real = set(([c for c in allc if "4x4" in c[0]] or allc)[0][1])
    um = assign_ou(net, real); idx = np.nonzero(um)[0]
    oa = approach_observed_mask(net, real); hist = D["hist_mean"]
    scen = np.concatenate([D["train_scen"], D["eval_scen"]])
    X, base, mu = [], [], []
    for s in scen:
        s = int(s)
        r = local_demand_ratio(mg, oa, D["APPR"][s], D["demand_ref"])
        for m in idx:
            st = D["static"][m]
            X.append([*st, hist[m], r[m]]); base.append(hist[m])
            # known nonlinear conditional mean (GBM must learn the r x static interaction)
            mu.append(hist[m] * (0.35 + 0.65 * r[m]) * (1.0 + 0.15 * np.sin(3.0 * st[3])))
    X = np.asarray(X, float); base = np.asarray(base, float); mu = np.asarray(mu, float)
    nq, nsc = len(idx), len(scen); gid = np.repeat(np.arange(nsc), nq)

    # choose sigma so the true bound = target_bound: skill* = 1 - sigma/rmse_naive,
    # rmse_naive^2 = mean((mu-base)^2) + sigma^2  => solve for sigma
    v_signal = float(np.mean((mu - base) ** 2))
    # skill* = 1 - sigma/sqrt(v_signal+sigma^2) = target  => sigma^2 = v_signal*(1-t)^2/(1-(1-t)^2)
    t = target_bound; k = (1 - t) ** 2; sigma = float(np.sqrt(v_signal * k / (1 - k)))
    x = mu + rng.normal(0, sigma, size=mu.shape)
    rmse_naive = float(np.sqrt(np.mean((x - base) ** 2)))
    true_bound = 1 - sigma / rmse_naive

    def cv_bound(est_fn):
        pred = np.zeros_like(x)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(nsc)):
            trm = np.isin(gid, tr); tem = np.isin(gid, te)
            est = est_fn(); est.fit(X[trm], x[trm]); pred[tem] = est.predict(X[tem])
        return float(1 - np.sqrt(np.mean((pred - x) ** 2)) / rmse_naive)

    gbm = cv_bound(lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05))
    lin = cv_bound(lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)))
    res = {"scale": scale, "sigma": round(sigma, 2), "true_bound": round(true_bound, 3),
           "GBM_estimated_bound": round(gbm, 3), "linear_estimated_bound": round(lin, 3),
           "GBM_error": round(gbm - true_bound, 3), "linear_error": round(lin - true_bound, 3)}
    print(json.dumps(res, indent=2))
    print(f"\nTrue bound {true_bound:.3f} | GBM recovers {gbm:.3f} (err {gbm-true_bound:+.3f}) | "
          f"linear {lin:.3f} (err {lin-true_bound:+.3f}).")
    print("GBM near-zero error = calibrated; linear negative error = it underfits mu and understates the bound.")
    json.dump(res, open(out, "w"), indent=2); print("saved", out)


if __name__ == "__main__":
    run()
