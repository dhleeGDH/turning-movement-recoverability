"""Delay response to split-allocation error, measured in microsimulation.

The Webster translation of an allocation error into delay is an approximation. This module
measures the response directly. On a synthetic SUMO grid the four green phases of the target
intersection are re-timed from the true approach volumes and from volumes perturbed by a
controlled split error, holding the cycle length and the yellow phases fixed. Each timing is
applied by overriding the target's tlLogic, the scenario is re-simulated, and the delay is
taken as the summed per-trip time loss from tripinfo. Each scenario is compared with its own
true-split run, so the reported interval is paired.

    python3 -m src.microsim_delay
"""
from __future__ import annotations

import json
import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import sumolib

from .sumo_inject import ROUTING, WORK, sim_window

SAT = 1900.0 / 3600.0    # saturation flow veh/s/lane


def _green_phases(prog):
    """Indices of green (movement) phases: those without y/Y and with some G, longest states."""
    out = []
    for i, ph in enumerate(prog.getPhases()):
        s = ph.state
        if ("y" not in s and "Y" not in s) and ("G" in s or "g" in s):
            out.append(i)
    return out


def _phase_approach_edges(net, tls, prog, gph):
    """For each green phase, the set of incoming edges it gives right-of-way (state G/g)."""
    conns = tls.getConnections()      # list of (inLane, outLane, linkIndex)
    link_edge = {}
    for inLane, outLane, li in conns:
        link_edge[li] = inLane.getEdge().getID()
    out = []
    for i in gph:
        s = prog.getPhases()[i].state
        edges = {link_edge[li] for li in range(len(s))
                 if li in link_edge and s[li] in ("G", "g")}
        out.append(edges)
    return out


def _webster_green(vols, green_total, n, min_g=5.0):
    """Split green_total across n phases proportional to flow ratio y=vol/sat, with a floor."""
    y = np.array([max(v, 1.0) / SAT for v in vols], float)
    g = green_total * y / y.sum()
    g = np.maximum(g, min_g)
    return g * (green_total / g.sum())     # renormalise after flooring


def _write_tls_add(path, tls_id, prog, gph, new_green, edgedata_out, freq=99999):
    phases = prog.getPhases()
    durs = [ph.duration for ph in phases]
    for k, gi in enumerate(gph):
        durs[gi] = float(round(new_green[k], 1))
    with open(path, "w") as f:
        f.write('<additional>\n')
        f.write(f'  <tlLogic id="{tls_id}" type="static" programID="ovr" offset="0">\n')
        for ph, d in zip(phases, durs):
            f.write(f'    <phase duration="{d}" state="{ph.state}"/>\n')
        f.write('  </tlLogic>\n')
        f.write(f'  <edgeData id="ed" freq="{freq}" file="{edgedata_out}"/>\n')
        f.write('</additional>\n')


def _run(scale, base_id, add_file, tag, u_edges=None, ed_path=None):
    """Run SUMO; return (network timeLoss, U-approach total waiting time). U-approach waiting is
    the undiluted local delay at the retimed intersection, read from edgeData."""
    d = f"{ROUTING}/{scale}"
    b, e = sim_window(scale, base_id)
    trip = f"{WORK}/mds_trip_{tag}.xml"
    cmd = ["sumo", "-n", f"{d}/{scale}.net.xml", "-r", f"{d}/ODtables/{base_id}.rou.xml",
           "-a", f"{d}/tazfile.taz.xml,{add_file}", "--tripinfo-output", trip,
           "--begin", str(b), "--end", str(e), "--time-to-teleport", "-1",
           "--no-step-log", "true", "--no-warnings", "true", "--xml-validation", "never"]
    subprocess.run(cmd, check=True, capture_output=True)
    tl = 0.0
    for _, el in ET.iterparse(trip, events=("end",)):
        if el.tag == "tripinfo":
            tl += float(el.get("timeLoss", 0.0)); el.clear()
    uwait = 0.0
    if u_edges and ed_path and os.path.exists(ed_path):
        for _, el in ET.iterparse(ed_path, events=("end",)):
            if el.tag == "edge" and el.get("id") in u_edges:
                uwait += float(el.get("waitingTime", 0) or 0)
            if el.tag == "edge":
                el.clear()
    return tl, uwait


def run(scale="5by5", base_ids=None, eps_grid=(0.0, 0.15, 0.3, 0.5), seed=0,
        out="data/microsim_delay.json"):
    net = sumolib.net.readNet(f"{ROUTING}/{scale}/{scale}.net.xml", withPrograms=True)
    tls_list = net.getTrafficLights()
    tls = tls_list[len(tls_list) // 2]
    prog = list(tls.getPrograms().values())[0]
    gph = _green_phases(prog)
    appr_edges = _phase_approach_edges(net, tls, prog, gph)
    cycle = sum(ph.duration for ph in prog.getPhases())
    green_total = sum(prog.getPhases()[i].duration for i in gph)
    os.makedirs(WORK, exist_ok=True)
    if base_ids is None:
        import glob
        base_ids = [os.path.basename(p)[5:-4] for p in
                    sorted(glob.glob(f"{ROUTING}/{scale}/simData/vehRouteData/Route*.xml"))
                    if p.endswith("d1.xml")][:6]
    print(f"[{scale}] U={tls.getID()}  green phases={len(gph)}  cycle={cycle}s  "
          f"green_total={green_total}s  scenarios={len(base_ids)}")

    rng = np.random.default_rng(seed)
    results = {str(eps): [] for eps in eps_grid}
    for bid in base_ids:
        # true approach volumes per green phase from a clean edgeData run
        ed = f"{WORK}/mds_ed_{bid}.xml"
        add0 = f"{WORK}/mds_add0_{bid}.xml"
        _write_tls_add(add0, tls.getID(), prog, gph, [prog.getPhases()[i].duration for i in gph], ed)
        _run(scale, bid, add0, f"true_{bid}")
        evol = {}
        for _, el in ET.iterparse(ed, events=("end",)):
            if el.tag == "edge":
                evol[el.get("id")] = float(el.get("entered", 0) or 0); el.clear()
        vols_true = [sum(evol.get(e, 0.0) for e in edges) for edges in appr_edges]
        if sum(vols_true) < 1:
            continue
        u_edges = set().union(*appr_edges)
        gt = _webster_green(vols_true, green_total, len(gph))
        for eps in eps_grid:
            if eps == 0:
                vols = vols_true
            else:
                vols = [max(v * (1 + eps * rng.standard_normal()), 1.0) for v in vols_true]
            g = _webster_green(vols, green_total, len(gph))
            edp = f"{WORK}/mds_ed_{bid}_{eps}.xml"
            add = f"{WORK}/mds_add_{bid}_{eps}.xml"
            _write_tls_add(add, tls.getID(), prog, gph, g, edp)
            tl, uwait = _run(scale, bid, add, f"e{eps}_{bid}", u_edges=u_edges, ed_path=edp)
            sperr = float(np.mean(np.abs(g / green_total - gt / green_total)) * 100)
            results[str(eps)].append((tl, sperr, uwait))

    summary = {}
    base_per = [x[2] for x in results["0.0"]]                  # per-scenario true-timing wait
    base_u = np.mean(base_per) if base_per else 0.0
    T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31}
    for eps in eps_grid:
        rs = results[str(eps)]
        d = np.mean([x[0] for x in rs]); sp = np.mean([x[1] for x in rs])
        uw = np.mean([x[2] for x in rs])
        # paired per-scenario percentage change against that scenario's own true-timing run
        paired = [100.0 * (rs[i][2] - base_per[i]) / base_per[i]
                  for i in range(min(len(rs), len(base_per))) if base_per[i] > 0]
        pm = float(np.mean(paired)) if paired else 0.0
        pci = (T95.get(len(paired) - 1, 2.0) * float(np.std(paired, ddof=1)) / np.sqrt(len(paired))
               if len(paired) > 1 else float("nan"))
        summary[str(eps)] = {"mean_timeLoss_s": round(float(d), 1),
                             "split_err_pp": round(float(sp), 2),
                             "U_approach_wait_s": round(float(uw), 1),
                             "U_delay_vs_true_pct": round(float(100 * (uw - base_u) / base_u), 2)
                             if base_u > 0 else 0.0,
                             "paired_pct_mean": round(pm, 2),
                             "paired_pct_ci95": round(pci, 2) if pci == pci else None,
                             "split_err_pp_per_scenario": [round(float(x[1]), 3) for x in rs],
                             "n_scenarios": len(paired),
                             "paired_pct_per_scenario": [round(x, 2) for x in paired]}
        print(f"  eps={eps:>4}: split_err={summary[str(eps)]['split_err_pp']:>5}pp  "
              f"U-approach wait={summary[str(eps)]['U_approach_wait_s']:>9}s  "
              f"(+{summary[str(eps)]['U_delay_vs_true_pct']}% vs true-split timing)")
    json.dump(summary, open(out, "w"), indent=2)
    print("saved", out)
    return summary


if __name__ == "__main__":
    run()
