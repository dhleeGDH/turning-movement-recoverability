"""S1 pilot on 5by5 (no GraphPFN backbone): Go/No-Go 1 (skill at rho=0) and
Go/No-Go 2 (crossover rho* where the neighbor-based method loses to naive).

Models at this stage (backbone-free):
  A (naive floor)   : target movement's OWN historical mean, computed on the site's
                      realized regime (i.e. source/sink-inclusive). Uses no neighbors and
                      no conservation -> ROBUST to source/sink, but ignores demand variation.
  B (method, anchor): clean observed approach volume x historical split ratio.
                      Uses cross-sectional neighbor info via conservation (spec 1.5 anchor)
                      -> great when conservation holds, structurally BIASED under source/sink.
  C (oracle ceiling): the true (regime-realized) count -> err 0. So skill = 1 - errB/errA,
                      and the A-vs-B crossover is exactly skill = 0.

Source/sink model: a *persistent* mid-block source/sink at the U site, present in every
scenario (so the site's own history A is calibrated to it), but invisible to the upstream
sensor that feeds B's approach volume. Severity rho, net direction sign.

Validation: leave-one-scenario-out for all historical stats; every intersection is used
as the single U (missing) target in turn (spatial LOO), averaged over seeds.
"""

import glob
import os

import numpy as np

from .eval.metrics import geh_pass_rate, rmse, skill_score
from .graph.labels import counts_to_arrays, extract_counts
from .graph.movement_graph import assign_ou, build_movement_graph
from .graph.net_parser import parse_net
from .inject.rho import inject_rho

BASE = os.path.join(os.environ.get("TMR_ROUTING", os.path.join(os.getcwd(), "routing")), "5by5")


def scale_dir(scale: str) -> str:
    return os.path.join(os.environ.get("TMR_ROUTING", os.path.join(os.getcwd(), "routing")), scale)


def net_path(scale: str) -> str:
    return f"{scale_dir(scale)}/{scale}.net.xml"


def load_scenarios(net, scale: str = "5by5", force: bool = False):
    """Return Y [S,N] movement counts and APPR [S,N] approach-link volume per movement."""
    cache = f"data/{scale}_scenarios.npz"
    if os.path.exists(cache) and not force:
        d = np.load(cache)
        return d["Y"], d["APPR"]

    routes = sorted(glob.glob(f"{scale_dir(scale)}/simData/vehRouteData/Route*.xml"))
    from_edge = [m.from_edge for m in net.movements]
    Y, APPR = [], []
    for rp in routes:
        move, link = extract_counts(rp, net)
        Y.append(counts_to_arrays(net, move))
        APPR.append(np.array([link.get(fe, 0) for fe in from_edge], float))
    Y = np.array(Y)
    APPR = np.array(APPR)
    os.makedirs("data", exist_ok=True)
    np.savez(cache, Y=Y, APPR=APPR)
    print(f"[{scale}] cached {Y.shape[0]} scenarios -> {cache}")
    return Y, APPR


def _inject_all(Y, APPR, um, mg, rho, sign, seed):
    """Persistent source/sink at the U site across ALL scenarios.

    Returns Yt [S,N] with U-site counts perturbed per scenario (regime-realized truth).
    Neighbor approach volume APPR is left clean (source/sink invisible upstream).
    """
    S = Y.shape[0]
    Yt = Y.astype(float).copy()
    for s in range(S):
        rng = np.random.default_rng((seed * 100003 + s) & 0x7FFFFFFF)
        Yt[s] = inject_rho(Y[s], um, mg, rho, sign, rng)
    return Yt


def run(rho_grid=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2), n_seeds=4, sign=+1):
    net = parse_net(f"{BASE}/5by5.net.xml")
    mg = build_movement_graph(net)
    Y, APPR = load_scenarios(net)
    S, N = Y.shape
    inters = sorted(net.intersections)

    # clean historical split ratio per movement per scenario (B uses clean neighbor world).
    # NOTE: computed per held-out scenario (train only) to avoid using the current U truth.
    with np.errstate(divide="ignore", invalid="ignore"):
        SPLIT = np.where(APPR > 0, Y / APPR, 0.0)

    print(f"5by5: S={S} scenarios, N={N} movements, {len(inters)} intersections\n")
    header = f"{'rho':>5} | {'errA':>7} {'errB':>7} | {'skill':>6} | {'GEH<5 (A/B)':>13}"
    print(header); print("-" * len(header))

    results = {}
    for rho in rho_grid:
        eA, eB, gA, gB = [], [], [], []
        for ji, jid in enumerate(inters):            # every intersection as U (spatial LOO)
            um = assign_ou(net, {jid})
            idx = np.nonzero(um)[0]
            for seed in range(n_seeds):
                Yt = _inject_all(Y, APPR, um, mg, rho, sign, seed=1000 * ji + seed)
                for s in range(S):                    # leave-one-scenario-out
                    train = np.arange(S) != s
                    a_own = Yt[train].mean(0)          # A: own-site regime-realized history
                    hist_split = SPLIT[train].mean(0)  # B prior: clean split, train only (no current-U leak)
                    y_true = Yt[s][idx]
                    predA = a_own[idx]
                    predB = APPR[s][idx] * hist_split[idx]        # clean approach x clean split
                    eA.append(rmse(predA, y_true)); eB.append(rmse(predB, y_true))
                    gA.append(geh_pass_rate(predA, y_true)); gB.append(geh_pass_rate(predB, y_true))

        mA, mB = np.mean(eA), np.mean(eB)
        sk = skill_score(mA, mB, 0.0)                 # oracle err = 0 -> skill = 1 - errB/errA
        results[rho] = dict(errA=mA, errB=mB, skill=sk, gehA=np.mean(gA), gehB=np.mean(gB))
        print(f"{rho:>5.1f} | {mA:>7.1f} {mB:>7.1f} | {sk:>6.2f} | "
              f"{np.mean(gA):>5.2f}/{np.mean(gB):.2f}")

    # crossover rho* where errB first exceeds errA
    rgrid = sorted(results)
    cross = None
    for i in range(1, len(rgrid)):
        r0, r1 = rgrid[i - 1], rgrid[i]
        d0 = results[r0]["errB"] - results[r0]["errA"]
        d1 = results[r1]["errB"] - results[r1]["errA"]
        if d0 < 0 <= d1:
            cross = r0 + (r1 - r0) * (-d0) / (d1 - d0)   # linear interp
            break
    print(f"\nGo/No-Go 1: skill(rho=0) = {results[0.0]['skill']:.2f}  "
          f"(target >= 0.50 -> {'PASS' if results[0.0]['skill'] >= 0.5 else 'FAIL'})")
    print(f"Go/No-Go 2: crossover rho* = "
          f"{f'{cross:.2f}' if cross is not None else 'not in grid'}  "
          f"({'PASS (in range)' if cross is not None else 'extend grid'})")
    return results


if __name__ == "__main__":
    run()
