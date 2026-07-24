"""Calibration of the recoverability-ceiling estimator under a
regime-switching (mixture) conditional distribution.

The Gaussian calibration (src.gaussian_calibration) and the count/negative-binomial calibration
(src.count_calibration) both draw the synthetic target from a unimodal conditional law on the real
9x9 observables o, so the conditional distribution target|o is single-component. Traffic demand is
instead a regime-switching domain: incidents, adverse weather and special events flip whole
windows into a disrupted, higher-mean and higher-variance state, so the conditional distribution is a
mixture (bimodal / heavy-tailed, non-Gaussian) rather than a single Gaussian or negative binomial. A
proxy calibrated only on unimodal ceilings need not recover a regime-switching ceiling.

This module keeps the same nonlinear conditional mean mu(o) on the same real 9x9 observables (from
src.count_calibration.build_process) but draws the target from a K=2 regime-switching mixture:

    latent regime z ~ Bernoulli(w_disrupted), assigned per scenario window (an incident/weather/event
    disrupts a whole window, not an isolated movement), then per-row target ~ NegBin:
        z = 0  "normal"    : mean = mu(o),         Var = mu + alpha_norm  * mu^2
        z = 1  "disrupted" : mean = g * mu(o),     Var = g*mu + alpha_disr * (g*mu)^2

The normal regime carries the Bucheon-matched overdispersion; the disrupted regime is a higher-mean,
heavier-dispersion state, so the pooled target|o is bimodal / heavy-tailed.

Because the regime z is LATENT (not part of the observables o), given o the target is a mixture and the
conditional variance follows the law of total variance in closed form:

    m_k(o), v_k(o)  = component means / variances,   w = (w_norm, w_disr)
    mbar(o)         = sum_k w_k m_k(o)                              (regime-averaged conditional mean)
    Var(target|o)   = sum_k w_k v_k(o)              (within-regime, "aleatoric")
                    + sum_k w_k (m_k(o) - mbar(o))^2 (between-regime mean spread, = w_norm w_disr (m1-m0)^2)

so E[Var(target|o)] is exact and the true regime-switching ceiling
    RI*_regime = 1 - sqrt( E[Var(target|o)] ) / sqrt( E[(mbar-base)^2] + E[Var(target|o)] )
is analytically known (heteroskedastic AND mixture, but closed form). The best any predictor that sees
only o can do is mbar(o); the between-regime term is irreducible because the regime is unobservable, and
it is correctly folded into the ceiling. The same GBM conditional-mean proxy and the same linear proxy as
the existing calibration are scored against RI*_regime, giving the regime-switching analogue of the
0.014 (Gaussian) / 0.024 (NB) GBM recovery errors.

    python3 -m src.known_ceiling_calibration
"""

import json

import numpy as np

from .count_calibration import _gbm, _lin, _oof, bucheon_dispersion, build_process


# ----------------------------------------------------------------------------- mixture DGP moments
def mixture_moments(mu, base, w_disr, g, alpha_norm, alpha_disr):
    """Closed-form conditional moments of the K=2 regime-switching mixture on observables o.

    Returns (mbar, var_cond, v_noise, v_signal, true_bound, between_frac) where var_cond is the per-row
    conditional variance Var(target|o) via the law of total variance (regime latent given o)."""
    w0, w1 = 1.0 - w_disr, w_disr
    m0, m1 = mu, g * mu
    v0 = m0 + alpha_norm * m0 ** 2                 # NegBin var of the normal regime
    v1 = m1 + alpha_disr * m1 ** 2                 # NegBin var of the disrupted regime
    mbar = w0 * m0 + w1 * m1
    within = w0 * v0 + w1 * v1                     # E[Var(target|o,z)]  (aleatoric)
    between = w0 * w1 * (m1 - m0) ** 2             # Var(E[target|o,z])  (regime spread, irreducible)
    var_cond = within + between
    v_noise = float(np.mean(var_cond))
    v_signal = float(np.mean((mbar - base) ** 2))
    true_bound = 1.0 - np.sqrt(v_noise) / np.sqrt(v_signal + v_noise)
    between_frac = float(np.mean(between) / np.mean(var_cond))   # share of E[Var|o] from regime switching
    return mbar, var_cond, v_noise, v_signal, float(true_bound), between_frac


def _draw_regime(rng, mu, gid, nsc, w_disr, g, alpha_norm, alpha_disr):
    """Draw one regime-switching target vector: latent regime PER SCENARIO, then per-row NegBin."""
    z_scen = (rng.random(nsc) < w_disr)            # regime flipped per whole window (incident/weather)
    z = z_scen[gid]                                # broadcast the window regime to its rows
    mean = np.where(z, g * mu, mu)
    alpha = np.where(z, alpha_disr, alpha_norm)
    lam = rng.gamma(shape=1.0 / alpha, scale=alpha * mean)      # gamma-Poisson mixture -> NegBin
    x = rng.poisson(lam).astype(float)
    return x, z_scen


def _excess_kurtosis(x):
    x = x - x.mean()
    m2 = np.mean(x ** 2)
    return float(np.mean(x ** 4) / (m2 ** 2) - 3.0)


# ----------------------------------------------------------------------------- calibration
def regime_calibration(X, base, mu, gid, nsc, w_disr, g, alpha_norm, alpha_disr, n_rep=12, seed=0):
    """Score GBM and linear proxies against the analytically known regime-switching mixture ceiling,
    averaged over n_rep independent draws (each with a fresh per-scenario regime assignment).

    The recovery error is scored against the exact analytic ceiling RI*_regime (the known ground truth,
    = 1 - sqrt(E[Var(target|o)]) / sqrt(v_signal + E[Var(target|o)])). The realized-draw ceiling is also
    reported, but because the between-regime variance dominates E[Var|o] and only nsc per-scenario regimes
    are drawn, the realized naive RMSE is volatile and its ceiling is a noisy Monte-Carlo estimate of the
    analytic value, so the analytic ceiling is the correct scoring target (the count/NB case had them
    coincide, 0.584 vs 0.585, only because its noise was unimodal)."""
    rng = np.random.default_rng(seed)
    mbar, _, v_noise, v_signal, true_bound, between_frac = mixture_moments(
        mu, base, w_disr, g, alpha_norm, alpha_disr)
    gbm_b, lin_b, tb, kurt, disr_frac = [], [], [], [], []
    for _ in range(n_rep):
        x, z_scen = _draw_regime(rng, mu, gid, nsc, w_disr, g, alpha_norm, alpha_disr)
        rmse_naive = float(np.sqrt(np.mean((x - base) ** 2)))
        tb.append(1.0 - np.sqrt(v_noise) / rmse_naive)          # ceiling vs the realized-draw naive RMSE
        gp, _ = _oof(_gbm, X, x, gid, nsc)
        lp, _ = _oof(_lin, X, x, gid, nsc)
        gbm_b.append(1.0 - np.sqrt(np.mean((gp - x) ** 2)) / rmse_naive)
        lin_b.append(1.0 - np.sqrt(np.mean((lp - x) ** 2)) / rmse_naive)
        kurt.append(_excess_kurtosis(x))
        disr_frac.append(float(z_scen.mean()))
    gbm_b, lin_b, tb = np.array(gbm_b), np.array(lin_b), np.array(tb)
    tbm = float(tb.mean())
    gbm_est, lin_est = float(gbm_b.mean()), float(lin_b.mean())
    return {
        "mixing_weights": {"w_normal": round(1 - w_disr, 3), "w_disrupted": round(w_disr, 3)},
        "disrupted_mean_multiplier_g": round(float(g), 3),
        "alpha_normal": round(float(alpha_norm), 4),
        "alpha_disrupted": round(float(alpha_disr), 4),
        "component_var_to_mean_at_meanmu": {
            "normal": round(float(1 + alpha_norm * mu[mu > 0].mean()), 3),
            "disrupted": round(float(1 + alpha_disr * g * mu[mu > 0].mean()), 3)},
        "between_regime_var_fraction": round(between_frac, 3),   # share of E[Var|o] from regime switching
        "pooled_excess_kurtosis": round(float(np.mean(kurt)), 3),  # >0 => heavy-tailed / non-Gaussian
        "realized_disrupted_scenario_frac": round(float(np.mean(disr_frac)), 3),
        "true_regime_ceiling": round(true_bound, 3),                    # exact analytic ground truth
        "true_regime_ceiling_analytic": round(true_bound, 3),
        "true_regime_ceiling_realized_mean": round(tbm, 3),            # noisy MC estimate (diagnostic)
        "GBM_estimated_ceiling": round(gbm_est, 3),
        "linear_estimated_ceiling": round(lin_est, 3),
        "GBM_error": round(gbm_est - true_bound, 3),                    # vs analytic ground truth
        "linear_error": round(lin_est - true_bound, 3),
        "GBM_recovery_abs_error": round(abs(gbm_est - true_bound), 3),
        "linear_recovery_abs_error": round(abs(lin_est - true_bound), 3),
        "GBM_error_vs_realized": round(gbm_est - tbm, 3),
        "linear_error_vs_realized": round(lin_est - tbm, 3),
        "GBM_est_sd_over_reps": round(float(gbm_b.std(ddof=1)), 4),
        "n_rep": n_rep,
    }


# ----------------------------------------------------------------------------- driver
def run(out="data/known_ceiling_calibration.json"):
    alpha, bmean, bn = bucheon_dispersion()
    print(f"Bucheon overdispersion: alpha={alpha:.4f}, mean count {bmean:.1f}, n={bn} "
          f"=> normal-regime carries this matched dispersion.")

    # regime-switching parameters: rare, high-mean, heavier-dispersion disrupted state
    W_DISR = 0.15          # ~15% of windows disrupted (incident / weather / event)
    G = 2.2                # disrupted-regime demand surge (mean x2.2)
    ALPHA_NORM = alpha     # normal regime matched to Bucheon overdispersion
    ALPHA_DISR = alpha * 6 # disrupted regime markedly heavier tailed

    res = {
        "bucheon_dispersion": {"alpha": round(alpha, 4), "mean_count": round(bmean, 1),
                               "var_to_mean_at_mean": round(1 + alpha * bmean, 2), "n_rows": bn},
        "reference_calibrations": {
            "gaussian": {"GBM_error": -0.014, "linear_error": -0.461,
                         "source": "data/gaussian_calibration.json"},
            "negative_binomial": {"GBM_error": -0.024, "linear_error": -0.402,
                                  "source": "data/count_calibration.json"}},
        "regime_dgp": {"K_components": 2, "regime_level": "per_scenario_window",
                       "w_disrupted": W_DISR, "disrupted_mean_multiplier": G,
                       "alpha_normal": round(ALPHA_NORM, 4), "alpha_disrupted": round(ALPHA_DISR, 4)},
    }

    for scale in ("9by9", "8by8"):
        X, base, mu, Yreal, gid, nsc, nq = build_process(scale)
        print(f"\n[{scale}] synthetic process: {nsc} scenarios x {nq} query nodes = {len(mu)} rows; "
              f"mean(mu)={mu[mu>0].mean():.2f}")
        primary = regime_calibration(X, base, mu, gid, nsc, W_DISR, G, ALPHA_NORM, ALPHA_DISR)
        res[f"regime_calibration_{scale}"] = primary
        print(f"  true regime ceiling RI*_regime = {primary['true_regime_ceiling']:.3f} (analytic; "
              f"realized-MC {primary['true_regime_ceiling_realized_mean']:.3f}); "
              f"pooled excess kurtosis {primary['pooled_excess_kurtosis']:.2f}, "
              f"between-regime var frac {primary['between_regime_var_fraction']:.2f}")
        print(f"  GBM recovers     {primary['GBM_estimated_ceiling']:.3f} "
              f"(|err| {primary['GBM_recovery_abs_error']:.3f}; Gaussian 0.014, NB 0.024)")
        print(f"  linear recovers  {primary['linear_estimated_ceiling']:.3f} "
              f"(|err| {primary['linear_recovery_abs_error']:.3f})")

    # sensitivity on the 9x9: mixing weight and disruption severity
    X, base, mu, Yreal, gid, nsc, nq = build_process("9by9")
    sens = {}
    for w in (0.10, 0.30):
        sens[f"w_disrupted_{w}"] = regime_calibration(X, base, mu, gid, nsc, w, G, ALPHA_NORM, ALPHA_DISR,
                                                       n_rep=10)
    for g in (1.6, 3.0):
        sens[f"g_{g}"] = regime_calibration(X, base, mu, gid, nsc, W_DISR, g, ALPHA_NORM, ALPHA_DISR,
                                            n_rep=10)
    res["regime_calibration_9by9_sensitivity"] = sens
    print("\n== sensitivity (9x9): mixing weight & disruption severity ==")
    for k, v in sens.items():
        print(f"  {k}: true={v['true_regime_ceiling']:.3f} GBM|err|={v['GBM_recovery_abs_error']:.3f} "
              f"lin|err|={v['linear_recovery_abs_error']:.3f} kurt={v['pooled_excess_kurtosis']:.2f}")

    json.dump(res, open(out, "w"), indent=2)
    print("\nsaved", out)


if __name__ == "__main__":
    run()
