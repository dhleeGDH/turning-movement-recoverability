"""Calibration of the recoverability-ceiling estimator under a count-appropriate
(integer, overdispersed, heteroskedastic) conditional distribution, plus an uncertainty interval on
the tree-ensemble ceiling estimate itself.

The Gaussian calibration (src.gaussian_calibration) builds x = mu(o) + N(0, sigma^2) on the real 9x9
observables o, where the true recoverability ceiling RI* = 1 - sqrt(E[Var(x|o)]) / RMSE(naive) is known
because Var(x|o) = sigma^2 is constant and known. There it found the gradient-boosted proxy recovers
RI* to within 0.014 while a linear proxy understates it by 0.461. Real turning counts are instead
integer, overdispersed and plausibly heteroskedastic, so a proxy calibrated on a Gaussian
ceiling need not recover a count/negative-binomial ceiling.

This module repeats the calibration with the same nonlinear conditional mean mu(o) on the same real 9x9
observables, but draws the target from a negative binomial (gamma-Poisson) with mean mu and dispersion
alpha matched to the real Bucheon count overdispersion (Var = mu + alpha*mu^2, estimated by method of
moments on the Bucheon hourly counts against their same-clock-hour mean). Because the negative-binomial
conditional variance mu + alpha*mu^2 is known in closed form, the true count ceiling
    RI*_count = 1 - sqrt( E[ mu + alpha*mu^2 ] ) / RMSE(naive)
is still exact (heteroskedastic but analytically known). The gradient-boosted proxy and a linear proxy
are scored against it, giving the count analogue of the 0.014 / 0.461 Gaussian numbers.

Part B reports a scenario-bootstrap 95% interval on the tree-ensemble ceiling estimate RI_hat* computed
on the real 9x9 process (the paper's ~0.68-0.70 ceiling), so the paper can state a ceiling interval
rather than a point and check whether "F near-optimal" (anchor ~0.63 vs ceiling) survives that interval.

    PYTHONPATH=. uv run --project external/graphpfn python -m src.count_calibration
"""

import json

import numpy as np

from .bucheon import build_bucheon_net
from .bucheon_apply import build_windowed
from .graph.local_demand import local_demand_ratio
from .graph.movement_graph import assign_ou
from .graph.observe import approach_observed_mask
from .graph.scenarios import generate_cluster_configs
from .dart_pretrain import _load


# ----------------------------------------------------------------------------- Bucheon dispersion
def bucheon_dispersion(window_json="data/bucheon_cam_turn_window.json", floor=0.5):
    """Method-of-moments negative-binomial dispersion alpha of the real Bucheon hourly counts against
    their same-clock-hour mean: Var(y|hour) = mu + alpha*mu^2 => alpha = E[(y-mu)^2 - mu] / E[mu^2]."""
    net = build_bucheon_net()
    Y, windows = build_windowed(net, window_json)
    hourkey = np.array([w[8:10] for w in windows])
    mu = np.zeros_like(Y)
    for w in range(len(windows)):
        same = (hourkey == hourkey[w]); same[w] = False
        mu[w] = Y[same].mean(0) if same.any() else Y.mean(0)
    m, y = mu.ravel(), Y.ravel()
    keep = m > floor
    m, y = m[keep], y[keep]
    alpha = float((np.mean((y - m) ** 2 - m)) / np.mean(m ** 2))
    return max(alpha, 0.0), float(m.mean()), int(m.size)


# ----------------------------------------------------------------------------- 9x9 synthetic process
def build_process(scale="9by9"):
    """Rows over (scenario x sensorless query node) on the real 9x9 observables o.

    Returns X (features [static, hist_mean, demand_ratio]), base (naive historical mean), mu (known
    nonlinear conditional mean, same form as the Gaussian calibration), Yreal (the real synthetic
    turning counts, used for the ceiling-uncertainty part), and grouping (gid, nsc, nq)."""
    D = _load(scale, 15, 0); net, mg = D["net"], D["mg"]
    allc = generate_cluster_configs(net)
    real = set(([c for c in allc if "4x4" in c[0]] or allc)[0][1])
    um = assign_ou(net, real); idx = np.nonzero(um)[0]
    oa = approach_observed_mask(net, real); hist = D["hist_mean"]
    scen = np.concatenate([D["train_scen"], D["eval_scen"]])
    X, base, mu, Yreal = [], [], [], []
    for s in scen:
        s = int(s)
        r = local_demand_ratio(mg, oa, D["APPR"][s], D["demand_ref"])
        for m in idx:
            st = D["static"][m]
            X.append([*st, hist[m], r[m]]); base.append(hist[m])
            # known nonlinear conditional mean (identical to src.gaussian_calibration): the GBM must
            # learn the demand-ratio x static-geometry interaction; a linear proxy cannot.
            mu.append(hist[m] * (0.35 + 0.65 * r[m]) * (1.0 + 0.15 * np.sin(3.0 * st[3])))
            Yreal.append(D["Y"][s][m])
    X = np.asarray(X, float); base = np.asarray(base, float)
    mu = np.clip(np.asarray(mu, float), 0.0, None); Yreal = np.asarray(Yreal, float)
    nq, nsc = len(idx), len(scen); gid = np.repeat(np.arange(nsc), nq)
    return X, base, mu, Yreal, gid, nsc, nq


# ----------------------------------------------------------------------------- proxies / CV
def _oof(est_fn, X, target, gid, nsc):
    """Grouped 5-fold (by scenario) out-of-fold predictions of `target` from X."""
    from sklearn.model_selection import KFold
    pred = np.zeros_like(target, dtype=float)
    fold = np.zeros(nsc, dtype=int)
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(nsc))):
        trm = np.isin(gid, tr); tem = np.isin(gid, te)
        est = est_fn(); est.fit(X[trm], target[trm]); pred[tem] = est.predict(X[tem])
        fold[te] = k
    return pred, fold


def _gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)


def _lin():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


# ----------------------------------------------------------------------------- count calibration
def count_calibration(X, base, mu, gid, nsc, alpha, n_rep=5, seed=0):
    """Draw x ~ NegBin(mean=mu, Var=mu+alpha*mu^2) via the gamma-Poisson mixture, score GBM and linear
    proxies against the analytically known count ceiling. Averaged over n_rep independent draws."""
    rng = np.random.default_rng(seed)
    v_noise = float(np.mean(mu + alpha * mu ** 2))          # E[Var(x|o)] = E[mu + alpha*mu^2] (exact)
    v_signal = float(np.mean((mu - base) ** 2))             # naive residual attributable to signal
    true_bound = 1.0 - np.sqrt(v_noise) / np.sqrt(v_signal + v_noise)
    gbm_b, lin_b, tb = [], [], []
    for _ in range(n_rep):
        lam = rng.gamma(shape=1.0 / alpha, scale=alpha * mu)   # gamma with mean mu, var alpha*mu^2
        x = rng.poisson(lam).astype(float)                     # integer, overdispersed, heteroskedastic
        rmse_naive = float(np.sqrt(np.mean((x - base) ** 2)))
        tb.append(1.0 - np.sqrt(v_noise) / rmse_naive)         # ceiling vs the realized-draw naive RMSE
        gp, _ = _oof(_gbm, X, x, gid, nsc)
        lp, _ = _oof(_lin, X, x, gid, nsc)
        gbm_b.append(1.0 - np.sqrt(np.mean((gp - x) ** 2)) / rmse_naive)
        lin_b.append(1.0 - np.sqrt(np.mean((lp - x) ** 2)) / rmse_naive)
    gbm_b, lin_b, tb = np.array(gbm_b), np.array(lin_b), np.array(tb)
    tbm = float(tb.mean())
    return {
        "alpha": round(float(alpha), 4),
        "var_to_mean_at_meanmu": round(float(1 + alpha * mu[mu > 0].mean()), 3),
        "true_count_ceiling": round(tbm, 3),
        "true_count_ceiling_analytic": round(float(true_bound), 3),
        "GBM_estimated_ceiling": round(float(gbm_b.mean()), 3),
        "linear_estimated_ceiling": round(float(lin_b.mean()), 3),
        "GBM_error": round(float(gbm_b.mean() - tbm), 3),
        "linear_error": round(float(lin_b.mean() - tbm), 3),
        "GBM_error_sd_over_reps": round(float(gbm_b.std(ddof=1)), 4),
        "n_rep": n_rep,
    }


# ----------------------------------------------------------------------------- ceiling uncertainty
def ceiling_uncertainty(X, base, Yreal, gid, nsc, nq, n_boot=2000, seed=0):
    """Scenario-bootstrap 95% interval on the tree-ensemble ceiling estimate RI_hat* for the real 9x9
    process (GBM out-of-fold skill of the real turning counts over the naive historical mean). Resamples
    whole scenarios with replacement and re-evaluates the fixed out-of-fold predictions, so the interval
    reflects finite-scenario sampling variability of the ceiling estimate. Also reports per-fold skills."""
    pred, fold = _oof(_gbm, X, Yreal, gid, nsc)
    def skill(mask):
        return 1.0 - np.sqrt(np.mean((pred[mask] - Yreal[mask]) ** 2)) \
                     / np.sqrt(np.mean((base[mask] - Yreal[mask]) ** 2))
    point = float(skill(np.ones_like(Yreal, dtype=bool)))
    # per-fold skills (scenario-grouped CV folds)
    fold_of_row = np.repeat(fold, nq)
    per_fold = [round(float(skill(fold_of_row == k)), 3) for k in range(5)]
    # scenario bootstrap
    rng = np.random.default_rng(seed)
    rows_by_scen = [np.nonzero(gid == s)[0] for s in range(nsc)]
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, nsc, size=nsc)
        rows = np.concatenate([rows_by_scen[s] for s in pick])
        boot[b] = 1.0 - np.sqrt(np.mean((pred[rows] - Yreal[rows]) ** 2)) \
                        / np.sqrt(np.mean((base[rows] - Yreal[rows]) ** 2))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "RI_hat_point": round(point, 3),
        "RI_hat_ci95": [round(float(lo), 3), round(float(hi), 3)],
        "RI_hat_boot_sd": round(float(boot.std(ddof=1)), 4),
        "per_fold_skill": per_fold,
        "per_fold_sd": round(float(np.std(per_fold, ddof=1)), 4),
        "n_boot": n_boot, "n_scenarios": nsc, "n_query_per_scenario": nq,
    }


# ----------------------------------------------------------------------------- driver
def run(out="data/count_calibration.json"):
    alpha, bmean, bn = bucheon_dispersion()
    print(f"Bucheon overdispersion: alpha={alpha:.4f} (Var=mu+alpha*mu^2), mean count {bmean:.1f}, "
          f"n={bn}  => Var/mean ~ {1 + alpha * bmean:.2f} at the Bucheon mean.")
    X, base, mu, Yreal, gid, nsc, nq = build_process("9by9")
    print(f"9x9 synthetic process: {nsc} scenarios x {nq} query nodes = {len(mu)} rows; "
          f"mean(mu)={mu[mu>0].mean():.2f}")

    # primary calibration at the matched dispersion, plus a heavier-dispersion sensitivity check
    primary = count_calibration(X, base, mu, gid, nsc, alpha)
    sens = {f"alpha_x{mult}": count_calibration(X, base, mu, gid, nsc, alpha * mult)
            for mult in (3, 10)}
    unc = ceiling_uncertainty(X, base, Yreal, gid, nsc, nq)

    res = {
        "bucheon_dispersion": {"alpha": round(alpha, 4), "mean_count": round(bmean, 1),
                               "var_to_mean_at_mean": round(1 + alpha * bmean, 2), "n_rows": bn},
        "gaussian_reference": {"true_bound": 0.65, "GBM_error": -0.014, "linear_error": -0.461,
                               "source": "data/gaussian_calibration.json"},
        "count_calibration_matched": primary,
        "count_calibration_sensitivity": sens,
        "ceiling_uncertainty_real": unc,
    }
    print("\n== Count (negative-binomial) calibration, dispersion matched to Bucheon ==")
    print(f"  true count ceiling      RI*_count = {primary['true_count_ceiling']:.3f}")
    print(f"  GBM recovers                        {primary['GBM_estimated_ceiling']:.3f} "
          f"(error {primary['GBM_error']:+.3f}; Gaussian was -0.014)")
    print(f"  linear recovers                     {primary['linear_estimated_ceiling']:.3f} "
          f"(error {primary['linear_error']:+.3f}; Gaussian was -0.461)")
    print("  sensitivity (heavier dispersion):")
    for k, v in sens.items():
        print(f"    {k}: true={v['true_count_ceiling']:.3f} GBM_err={v['GBM_error']:+.3f} "
              f"lin_err={v['linear_error']:+.3f}")
    print("\n== Uncertainty on the real tree-ensemble ceiling estimate RI_hat* ==")
    print(f"  RI_hat* = {unc['RI_hat_point']:.3f}  95% CI {unc['RI_hat_ci95']}  "
          f"(per-fold {unc['per_fold_skill']})")
    json.dump(res, open(out, "w"), indent=2)
    print("\nsaved", out)


if __name__ == "__main__":
    run()
