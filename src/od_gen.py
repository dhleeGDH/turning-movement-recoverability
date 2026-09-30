"""Controlled OD generation for the CV(APPR)/demandCV decoupling study.

Two independent knobs, both realised as real SUMO demand (marouter -> sumo):
  total_scale        : overall demand magnitude of a scenario -> varying it across
                       scenarios sets demandCV (site total-demand variability).
  major_minor_ratio  : demand on the major corridor relative to the minor streams ->
                       spatial concentration, which sets the dispersion of approach
                       volumes CV(APPR) WITHOUT changing total demand.
Holding one knob and varying the other decouples the two axes the pilot confounded.

Reuses the 5by5 four-stream OD template from routing/5by5/1_OD_generation.py.
"""

import os
import subprocess
from itertools import product

import numpy as np

from .sumo_inject import ROUTING, WORK

# 5by5 four-stream template: (origins, destinations) TAZ ids
STREAMS_5by5 = [
    ([2, 4, 6, 8, 10], [30, 28, 26, 24, 22]),      # major corridor
    ([12, 14, 16, 18, 20], [40, 38, 36, 34, 32]),  # minor 1
    ([29, 27, 25, 23, 21], [1, 3, 5, 7, 9]),       # minor 2
    ([39, 37, 35, 33, 31], [11, 13, 15, 17, 19]),  # minor 3
]


def write_fma(path, total_demand, major_minor_ratio, seed,
              begin_h=6.55, end_h=8.0):
    """Write an O;D2 matrix whose total OD demand is `total_demand` REGARDLESS of the ratio,
    so the concentration knob does not leak into total demand. Each stream has 25 O-D pairs;
    major pair value v_maj, each minor pair v_maj/ratio, chosen so
    25*v_maj + 3*25*(v_maj/ratio) = total_demand. Small per-stream N(.,./21) jitter."""
    rng = np.random.default_rng(seed)
    r = major_minor_ratio
    v_maj = total_demand / (25.0 + 75.0 / r)
    v_min = v_maj / r
    vals = [v_maj, v_min, v_min, v_min]
    lines = []
    for (orig, dest), v in zip(STREAMS_5by5, vals):
        vj = max(0.5, rng.normal(v, v / 21.0))
        for o, d in product(orig, dest):
            lines.append(f"\t{o}\t{d}\t{vj:.2f}\n")
    hdr = ("$O;D2\n* From-Time To-Time\n"
           f"{begin_h} {end_h}\n* Factor\n1.00\n* gen\n* gen\n* gen\n")
    with open(path, "w") as f:
        f.write(hdr + "".join(lines))
    return path


def marouter(scale, fma_path, out_rou):
    net = f"{ROUTING}/{scale}/{scale}.net.xml"
    taz = f"{ROUTING}/{scale}/tazfile.taz.xml"
    subprocess.run(["marouter", "-n", net, "-m", fma_path, "--additional-files", taz,
                    "-o", out_rou, "--ignore-errors", "--no-warnings", "true",
                    "--xml-validation", "never"], check=True, capture_output=True)
    return out_rou


def run_sumo_rou(scale, rou_path, tag, begin=21600, end=28800):
    """Run SUMO on a produced route file (net-default signal timing). Return vehroute path."""
    net = f"{ROUTING}/{scale}/{scale}.net.xml"
    taz = f"{ROUTING}/{scale}/tazfile.taz.xml"
    os.makedirs(f"{WORK}/od", exist_ok=True)
    out = f"{WORK}/od/Route_{tag}.xml"
    subprocess.run(["sumo", "-n", net, "-r", rou_path, "-a", taz,
                    "--vehroute-output", out, "--begin", str(begin), "--end", str(end),
                    "--time-to-teleport", "-1", "--no-step-log", "true",
                    "--no-warnings", "true", "--xml-validation", "never"],
                   check=True, capture_output=True)
    return out


def gen_route_sim(scale, tag, total_demand, major_minor_ratio, seed):
    """Full chain for one scenario: OD -> marouter -> sumo. Returns vehroute path."""
    os.makedirs(f"{WORK}/od", exist_ok=True)
    fma = f"{WORK}/od/{tag}.fma"
    rou = f"{WORK}/od/{tag}.rou.xml"
    write_fma(fma, total_demand, major_minor_ratio, seed)
    marouter(scale, fma, rou)
    return run_sumo_rou(scale, rou, tag)


if __name__ == "__main__":
    from .graph.net_parser import parse_net
    from .sumo_inject import extract_YA
    net = parse_net(f"{ROUTING}/5by5/5by5.net.xml")
    for mm in (1.0, 3.0):
        vr = gen_route_sim("5by5", f"test_m{mm}", total_demand=4000, major_minor_ratio=mm, seed=1)
        Y, A, c = extract_YA(net, vr)
        # spatial dispersion of approach volumes at the central intersection
        print(f"ratio={mm}: total turns={int(Y.sum())}  nonzero={int((Y>0).sum())}/{len(Y)}  "
              f"mean approach vol={A[A>0].mean():.0f}  approach-vol CV={A[A>0].std()/A[A>0].mean():.3f}")
