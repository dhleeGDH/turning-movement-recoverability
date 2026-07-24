"""Conservation-consistent source/sink injection at the SUMO level.

The pilot's rho axis multiplied the extracted TARGET labels by (1+rho) and left the
neighbour observables untouched -- a post-hoc label perturbation, not a physical flow.
Here the source/sink is real SUMO demand: extra vehicles that depart mid-block on an
internal link L = (J1 -> J2) and then drive on through J2 to a network boundary. Because
J1 never routed them onto L, the upstream-exit count of L is unchanged, while the
downstream-approach count of L rises -- exactly the mismatch the field estimator
    rho_hat(L) = downstream_approach(L) / upstream_exit(L) - 1
measures. Every injected vehicle carries a full valid route, so global flow conservation
holds. Running SUMO with [base routes + injected routes] and re-extracting gives Y and
APPR that both move consistently, and rho_hat recovers the injected severity.

  python3 -m src.sumo_inject          # self-test: recover a known rho on one scenario
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .graph.labels import counts_to_arrays, extract_counts
from .graph.net_parser import NetworkData, parse_net
from .rho_estimate import estimate_rho_per_link, internal_links

ROUTING = "/home/dhlee/routing"
WORK = "/tmp/claude-1000/-home-dhlee-estimation/93a9334d-e0f0-4ca5-abcc-f81e963b8cde/scratchpad/sumo_inject"


def _adjacency(net: NetworkData) -> dict[str, list[str]]:
    """Valid successor edges of each edge (edges reachable by one real turning movement)."""
    from collections import defaultdict
    adj = defaultdict(list)
    for m in net.movements:
        adj[m.from_edge].append(m.to_edge)
    return dict(adj)


def path_to_boundary(adj: dict[str, list[str]], start: str, rng, max_hops: int = 14) -> list[str]:
    """Build a topologically valid edge path from `start` until an exit edge (no successor)."""
    path = [start]
    cur = start
    for _ in range(max_hops):
        succ = adj.get(cur)
        if not succ:                      # exit edge -> leaves the grid
            break
        cur = succ[int(rng.integers(len(succ)))]
        path.append(cur)
    return path


def _baseline_downstream_flow(net: NetworkData, base_counts: np.ndarray) -> dict[str, float]:
    """Downstream-approach volume per internal link L (movements with from_edge == L)."""
    from collections import defaultdict
    dn = defaultdict(float)
    for i, m in enumerate(net.movements):
        dn[m.from_edge] += base_counts[i]
    return dn


def make_injected_routes(net: NetworkData, base_counts: np.ndarray, target_links: list[str],
                         rho: float, sign: int, rng, begin: int, end: int,
                         tag: str = "") -> tuple[str, dict]:
    """Write an injected .rou.xml. For each target link L, add round(rho*baseline_flow(L))
    vehicles that depart on L and drive to a boundary. sign=+1 source (extra vehicles).
    sign=-1 (sink) is modelled by removing routes downstream; the source branch is
    implemented physically and the sink derived as its mirror in the analysis. Returns (path, meta).
    """
    adj = _adjacency(net)
    dn = _baseline_downstream_flow(net, base_counts)
    veh = []
    injected_per_link = {}
    vid = 0
    for L in target_links:
        n = int(round(rho * dn.get(L, 0.0)))
        injected_per_link[L] = n
        for _ in range(n):
            route = path_to_boundary(adj, L, rng)
            if len(route) < 2:            # L is itself an exit edge; skip
                continue
            depart = float(rng.uniform(begin, end))
            veh.append((depart, vid, " ".join(route)))
            vid += 1
    veh.sort()                            # SUMO requires non-decreasing depart order
    os.makedirs(WORK, exist_ok=True)
    uniq = tag or f"p{os.getpid()}"
    path = f"{WORK}/injected_{uniq}.rou.xml"
    with open(path, "w") as f:
        f.write('<routes>\n')
        f.write('  <vType id="inj" length="5" accel="2.6" decel="4.5" sigma="0.5"/>\n')
        for depart, i, edges in veh:
            f.write(f'  <vehicle id="inj{i}" type="inj" depart="{depart:.1f}" '
                    f'departLane="free" departSpeed="max">\n')
            f.write(f'    <route edges="{edges}"/>\n  </vehicle>\n')
        f.write('</routes>\n')
    return path, {"injected_per_link": injected_per_link, "n_injected": len(veh)}


def sim_window(scale: str, base_id: str) -> tuple[int, int]:
    """Read the simulation begin/end from the base scenario's flow definitions."""
    base_rou = f"{ROUTING}/{scale}/ODtables/{base_id}.rou.xml"
    begins, ends = [], []
    for _, el in ET.iterparse(base_rou, events=("end",)):
        if el.tag == "flow":
            if el.get("begin"):
                begins.append(float(el.get("begin")))
            if el.get("end"):
                ends.append(float(el.get("end")))
        el.clear()
    if begins and ends:
        return int(min(begins)), int(max(ends))
    return 21600, 25200


def run_sumo(scale: str, base_id: str, extra_route: str | None) -> str:
    """Run SUMO for a base scenario (optionally + an extra route-file). Return vehroute path."""
    d = f"{ROUTING}/{scale}"
    net = f"{d}/{scale}.net.xml"
    taz = f"{d}/tazfile.taz.xml"
    tls = f"{d}/additional/{base_id}.xml"
    base_rou = f"{d}/ODtables/{base_id}.rou.xml"
    b, e = sim_window(scale, base_id)
    os.makedirs(WORK, exist_ok=True)
    out = f"{WORK}/Route_{base_id}_{'inj' if extra_route else 'base'}_p{os.getpid()}.xml"
    route_files = base_rou + ("," + extra_route if extra_route else "")
    add_files = ",".join([p for p in (taz, tls) if os.path.exists(p)])
    cmd = ["sumo", "-n", net, "-r", route_files, "--vehroute-output", out,
           "--begin", str(b), "--end", str(e), "--time-to-teleport", "-1",
           "--no-step-log", "true", "--no-warnings", "true", "--xml-validation", "never"]
    if add_files:
        cmd += ["-a", add_files]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def extract_YA(net: NetworkData, vehroute: str):
    """Return (Y [N], APPR [N], raw_counts [N]) from a vehroute file."""
    move, link = extract_counts(vehroute, net)
    Y = counts_to_arrays(net, move)
    from_edge = [m.from_edge for m in net.movements]
    APPR = np.array([link.get(fe, 0) for fe in from_edge], float)
    return Y, APPR, Y


def self_test(scale="5by5", rho=0.4, seed=0):
    net = parse_net(f"{ROUTING}/{scale}/{scale}.net.xml")
    base_id = "1704000479d1" if scale == "5by5" else "1703841287d1"
    rng = np.random.default_rng(seed)
    begin, end = sim_window(scale, base_id)

    # baseline SUMO run (reproduce), rho_hat ~ 0
    vr0 = run_sumo(scale, base_id, None)
    Y0, A0, c0 = extract_YA(net, vr0)
    links = internal_links(net)
    r0 = estimate_rho_per_link(net, c0, links)
    e0 = np.abs(list(r0.values()))
    print(f"[{scale}] baseline: turns={int(Y0.sum())}  internal links={len(links)}  "
          f"clean |rho_hat| mean={e0.mean():.3f} max={e0.max():.3f}  (expect ~0)")

    # inject a known rho on a handful of internal links, recover it
    target = list(rng.choice(links, size=min(8, len(links)), replace=False))
    inj, meta = make_injected_routes(net, c0, target, rho, +1, rng, begin, end)
    print(f"  injected {meta['n_injected']} source vehicles on {len(target)} links "
          f"(target rho={rho})")
    vr1 = run_sumo(scale, base_id, inj)
    Y1, A1, c1 = extract_YA(net, vr1)
    r1 = estimate_rho_per_link(net, c1, links)
    rec = np.array([r1[L] for L in target if L in r1])
    print(f"  after injection: turns={int(Y1.sum())} (+{int(Y1.sum()-Y0.sum())})  "
          f"recovered rho_hat on target links: mean={rec.mean():.3f} "
          f"median={np.median(rec):.3f}  (target {rho})")
    off = [L for L in links if L not in target]
    ro = np.array([abs(r1[L]) for L in off if L in r1])
    print(f"  non-target links |rho_hat| mean={ro.mean():.3f}  (should stay ~0)")


if __name__ == "__main__":
    self_test()
