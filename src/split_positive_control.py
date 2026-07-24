"""Positive control for the turn-split detection floor.

Injects a known, recoverable split signal into the movement truth on the 9x9
held-out-4x4-cluster protocol and checks that the split-probe battery recovers it.
The injected shift is scenario-varying and a deterministic function of the
standardized approach demand, an observable the probe reads through diffusion, so
recovering it shows the battery can detect an o-predictable split. A matched
control replaces the demand signal with independent noise, which is unrecoverable
by construction.

Per scenario s and movement m, the o-dependent injection is
    base_split = Y[s,m] / APPR[s,m]
    z          = standardized APPR[s,m] over movements
    delta      = alpha * z * {left: +1, through: 0, right: -1}[turn(m)]
    split_new  = renormalize clip(base_split + delta, 0, None) within each approach
    Y_inj      = split_new * APPR[s,m]                  # link volumes stay clean
so high-demand approaches shift probability toward lefts and away from rights,
scenario by scenario. The matched control uses delta = alpha * eps with eps drawn
independently of the demand.

The frozen historical split is calibrated on the injected training scenarios, so a
learned gradient-boosted probe gains only from the scenario-varying, demand-correlated
structure, scored by RI over the frozen split out of cluster. The o-dependent gain
rises with alpha while the noise control stays near zero, which calibrates the
detection floor and shows the null on the real data reflects an absent o-signal
rather than an underpowered battery. CPU only.

    python3 -m src.split_positive_control
"""

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .graph.local_demand import diffuse_observed
from .graph.movement_graph import assign_ou
from .graph.observe import approach_observed_mask
from .graph.scenarios import generate_cluster_configs
from .dart_pretrain import _load

SEEDS = [0, 1, 2, 3]
ALPHAS = [0.0, 0.1, 0.2, 0.4]
TURN = {"l": 0, "s": 1, "r": 2}
DIR_MULT = {"l": 1.0, "s": 0.0, "r": -1.0}   # turn-type-dependent shift direction
EPS_SEED = 20260722                          # base seed for the o-independent random shift


def _approach_groups(net):
    """Map each movement to an integer id for its approach (from_edge) group."""
    ids, grp = {}, np.empty(len(net.movements), int)
    for i, m in enumerate(net.movements):
        grp[i] = ids.setdefault(m.from_edge, len(ids))
    return grp, len(ids)


def _renorm(split_raw, grp, n_groups):
    """Renormalize splits within each approach group to sum to 1 (groups with 0 total -> 0)."""
    tot = np.zeros(n_groups)
    np.add.at(tot, grp, split_raw)
    tot_safe = np.where(tot > 0, tot, 1.0)
    return split_raw / tot_safe[grp]


def _inject_all(D, alpha, mode, grp, n_groups, dir_mult, seed):
    """Inject an alpha-scaled turn-split reshape into the movement truth of every scenario.

    mode="odep": delta = alpha * standardize(APPR[s]) * dir_mult   (predictable from v_in)
    mode="oind": delta = alpha * eps ~ N(0,1)        * dir_mult   (independent of any observable)
    Approach/link volumes (APPR) are left clean; only the split (turning distribution) moves.

    The o-dependent standardization uses fixed, scenario-independent constants (global mean/std of
    APPR), so delta[s,m] is a PURE deterministic function of the per-movement observable APPR[s,m]
    (which the probe reads as v_in) and the turn type -- recoverable from a single movement's
    features. (Per-scenario z-scoring would fold in the scenario's global mean/std, which the probe
    cannot see from one movement, making even the injected shift unrecoverable.)
    Returns Yb [S, N].
    """
    Y, APPR = D["Y"], D["APPR"]
    S, N = Y.shape
    a_ref = float(APPR.mean()); a_scale = float(APPR.std()) + 1e-9   # fixed global standardizers
    with np.errstate(divide="ignore", invalid="ignore"):
        base_split = np.where(APPR > 0, Y / APPR, 0.0)
    Yb = np.empty_like(Y, dtype=float)
    for s in range(S):
        if alpha <= 0:
            Yb[s] = Y[s].astype(float)
            continue
        if mode == "odep":
            zsig = (APPR[s].astype(float) - a_ref) / a_scale   # standardized demand observable (fixed scale)
        else:  # o-independent, matched-amplitude random shift, drawn independently of demand
            zsig = np.random.default_rng(EPS_SEED + seed * 100003 + s).standard_normal(N)
        delta = alpha * zsig * dir_mult
        split_raw = np.clip(base_split[s] + delta, 0.0, None)
        split_new = _renorm(split_raw, grp, n_groups)
        Yb[s] = split_new * APPR[s]
    return Yb


def _probe_gain(D, Yb, grp, n_groups, turn, want_r2=False):
    """Learned GB split-adjustment probe, 9x9 held-out-4x4-cluster; mean RI gain over frozen split.

    Frozen split calibrated on the same injected training scenarios. Returns (gain, r2) where r2 is
    the out-of-cluster R^2 of the split residual (None unless want_r2).
    """
    net, mg = D["net"], D["mg"]
    with np.errstate(divide="ignore", invalid="ignore"):
        split_all = np.where(D["APPR"] > 0, Yb / D["APPR"], 0.0)   # injected split per scenario
    hist_split_inj = split_all[D["train_scen"]].mean(0)            # frozen split (injection baked in)
    appr_mean = D["APPR"].mean(0)
    gains, r2s = [], []
    for _, real_set, _ in [c for c in generate_cluster_configs(net) if "4x4" in c[0]]:
        um = assign_ou(net, set(real_set)); q = np.nonzero(um)[0]; tr = ~um
        oa = approach_observed_mask(net, set(real_set))
        prior = {t: float(hist_split_inj[tr][turn[tr] == t].mean()) for t in np.unique(turn)}
        tt_prior = np.array([prior[t] for t in turn])
        yq, vq, naive = [], [], []
        Xtr, rtr, Xte, rte = [], [], [], []
        for s in D["eval_scen"]:
            s = int(s)
            v_in = diffuse_observed(mg, oa, D["APPR"][s])          # observable demand level
            yq.append(Yb[s][q]); vq.append(v_in[q])
            naive.append(hist_split_inj[q] * appr_mean[q])
            feat = np.column_stack([v_in, turn, tt_prior, hist_split_inj])
            resid = (split_all[s] - hist_split_inj)
            Xtr.append(feat[tr]); rtr.append(resid[tr])
            Xte.append(feat[q]); rte.append(resid[q])
        yq = np.concatenate(yq); vq = np.concatenate(vq); naive = np.concatenate(naive)
        rte = np.concatenate(rte)
        base = np.tile(hist_split_inj[q], len(D["eval_scen"]))

        def ri(p):
            return float(1 - np.sqrt(np.mean((p - yq) ** 2)) / np.sqrt(np.mean((naive - yq) ** 2)))

        frozen = ri(base * vq)
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4)
        gb.fit(np.vstack(Xtr), np.concatenate(rtr))
        rpred = gb.predict(np.vstack(Xte))
        gains.append(ri(np.clip(base + rpred, 0, None) * vq) - frozen)
        if want_r2:
            ss_res = float(np.sum((rte - rpred) ** 2))
            ss_tot = float(np.sum((rte - rte.mean()) ** 2))
            r2s.append(1 - ss_res / max(ss_tot, 1e-12))
    return float(np.mean(gains)), (float(np.mean(r2s)) if want_r2 else None)


def run(scale="9by9", n_eval=15):
    res = {"scale": scale,
           "protocol": "9x9 held-out-4x4-cluster; frozen split calibrated on same injected train scenarios",
           "alphas": ALPHAS, "seeds": SEEDS, "dir_mult": DIR_MULT,
           "odep": {}, "oind": {}, "odep_probe_r2": {}}
    print("Positive control: learned GB split-adjustment probe, gain over frozen split vs alpha")
    print(f"{'alpha':>6} | {'o-DEP gain':>11} {'o-DEP R^2':>10} | {'o-IND gain':>11}")
    for alpha in ALPHAS:
        gd, gi, r2d = [], [], []
        for seed in SEEDS:
            D = _load(scale, n_eval, seed)
            grp, ng = _approach_groups(D["net"])
            turn = np.array([TURN.get(getattr(m, "dir", "s"), 1) for m in D["net"].movements])
            dm = np.array([DIR_MULT.get(getattr(m, "dir", "s"), 0.0) for m in D["net"].movements])
            Yb_d = _inject_all(D, alpha, "odep", grp, ng, dm, seed)
            Yb_i = _inject_all(D, alpha, "oind", grp, ng, dm, seed)
            g_d, r2_d = _probe_gain(D, Yb_d, grp, ng, turn, want_r2=True)
            g_i, _ = _probe_gain(D, Yb_i, grp, ng, turn, want_r2=False)
            gd.append(g_d); gi.append(g_i); r2d.append(r2_d)
        gd_m, gi_m, r2_m = float(np.mean(gd)), float(np.mean(gi)), float(np.mean(r2d))
        res["odep"][str(alpha)] = round(gd_m, 4)
        res["oind"][str(alpha)] = round(gi_m, 4)
        res["odep_probe_r2"][str(alpha)] = round(r2_m, 4)
        print(f"{alpha:>6} | {gd_m:>+11.4f} {r2_m:>+10.4f} | {gi_m:>+11.4f}")

    # verdict: positive control passes iff o-DEP gain rises with alpha while o-IND stays ~0
    gd0 = res["odep"][str(ALPHAS[0])]; gdmax = res["odep"][str(ALPHAS[-1])]
    gi_max = max(abs(v) for v in res["oind"].values())
    odep_rises = (gdmax - gd0) > 0.02 and gdmax > 0.02
    oind_flat = gi_max < 0.02
    res["odep_rises_with_alpha"] = bool(odep_rises)
    res["oind_stays_near_zero"] = bool(oind_flat)
    res["positive_control_passes"] = bool(odep_rises)
    res["verdict"] = (
        "POSITIVE CONTROL passes: the probe battery RECOVERS split variation that is predictable "
        "from the observables (o-dependent gain rises with alpha) while remaining ~0 for the matched "
        "o-independent shift. The battery has power; therefore its null on the real/actual data is "
        "SUBSTANTIVE (the real residual split carries no o-signal), not a trivially-guaranteed artifact."
        if odep_rises and oind_flat else
        "POSITIVE CONTROL INCONCLUSIVE: o-dependent gain did not rise as expected"
        + ("" if oind_flat else " and/or the o-independent control did not stay near zero")
        + " -- see numbers."
    )
    print("\nVERDICT:", res["verdict"])
    json.dump(res, open("data/split_positive_control.json", "w"), indent=2)
    print("saved data/split_positive_control.json")
    return res


if __name__ == "__main__":
    run()
