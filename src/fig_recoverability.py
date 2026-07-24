"""The recoverability money figure (2 panels):
  A: synthetic achievable-accuracy benchmark -- MMSE bound vs demandCV, with F at the bound.
  B: real Bucheon resolution reversal -- bound and observed F skill vs a time-of-day lookup,
     bound ~0 at hourly (nothing beats ToD) opening at finer resolution where F captures it.

    python3 -m src.fig_recoverability
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12})

os.makedirs("results", exist_ok=True)
ACC, POS, NEG, MUT, ORA = "#22687e", "#2f8f5f", "#c0524a", "#8a94a3", "#a9781a"


def run(out="results/fig_recoverability.png"):
    syn = json.load(open("data/recover_bound_syn.json"))
    grids = sorted(syn, key=lambda g: syn[g]["demandCV"])
    dcv = [syn[g]["demandCV"] for g in grids]
    bnd = [syn[g]["bound_realistic_obs"] for g in grids]
    ora = [syn[g]["oracle_demand_bound"] for g in grids]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.6, 5.6))

    # ---- Panel A: synthetic envelope ----
    axA.plot(dcv, bnd, "-o", color=ACC, lw=2.4, ms=8, label="estimated achievable-accuracy benchmark $\\widehat{\\mathrm{RI}^{*}}$ (realistic obs.)")
    axA.plot(dcv, ora, "--s", color=ORA, lw=1.8, ms=6, label="oracle demand-only bound")
    axA.axhline(0.76, ls=":", color=MUT, lw=1.4)
    axA.text(0.055, 0.775, "transductive ridge 0.76 (peeks at test)", color=MUT, fontsize=12, va="bottom")
    axA.scatter([0.329], [0.629], s=340, marker="*", color=POS, zorder=7, edgecolors="w", lw=1.0,
                label="demand anchor (deployable, 9$\\times$9)")
    axA.scatter([0.329], [0.66], s=95, marker="D", color=NEG, zorder=6, edgecolors="w", lw=0.8,
                label="DART (capacity probe)")
    axA.annotate("the deployable anchor\nreaches the benchmark", (0.329, 0.629), (0.085, 0.55),
                 color=POS, fontsize=12.5, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=POS, lw=1.6))
    axA.set_xlabel("Demand variability ($\\mathrm{demandCV}_{\\mathrm{tot}}$)", fontsize=13)
    axA.set_ylabel("RI vs naive (held-out cluster)", fontsize=13)
    axA.set_title("(a) Synthetic: achievable-accuracy benchmark")
    axA.set_ylim(0, 0.98); axA.grid(alpha=.3); axA.legend(loc="lower right")

    # ---- Panel B: real resolution reversal ----
    rb = json.load(open("data/recover_bound.json"))
    res = ["hourly", "15min", "5min"]
    x = np.arange(3)
    bound = [rb[r]["bound_skill_vs_ToD"] for r in res]
    F = [-0.204, 0.253, 0.357]; Fe = [0.203, 0.076, 0.033]
    axB.axhline(0, color="k", lw=1.0)
    # RI*>=0 floor: the benchmark is a non-negative achievable-accuracy bound and never enters the shaded band
    axB.axhspan(-0.42, 0.0, facecolor=NEG, alpha=.10, hatch="//", edgecolor="none")
    axB.text(2.02, -0.02, "$\\mathrm{RI}^{*}\\geq 0$ floor", color=MUT, fontsize=10.5, ha="right", va="top")
    axB.plot(x, bound, "-o", color=ACC, lw=2.2, ms=9, zorder=6, markeredgecolor="w", markeredgewidth=1.2,
             label="estimated achievable-accuracy benchmark $\\widehat{\\mathrm{RI}^{*}}$ (non-negative)")
    axB.errorbar(x, F, yerr=Fe, fmt="--D", color=POS, lw=2.4, ms=8, capsize=4, zorder=5,
                 label="observed DART (can be negative)")
    # the only negative hourly point is observed DART, not the benchmark
    axB.annotate("observed DART, not the benchmark:\nDART loses to a same-hour lookup here",
                 xy=(0.0, -0.204), xytext=(0.18, -0.36),
                 color=POS, fontsize=11.5, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=POS, lw=1.5))
    axB.annotate("benchmark at the floor ($\\approx$0):\nno detected headroom over a same-hour lookup",
                 xy=(0.0, 0.013), xytext=(0.14, 0.30),
                 color=ACC, fontsize=11.5, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=ACC, lw=1.5))
    axB.annotate("benchmark opens $\\rightarrow$ DART gains", xy=(1.0, 0.253), xytext=(0.9, 0.50),
                 color=POS, fontsize=12, ha="left", va="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                 arrowprops=dict(arrowstyle="->", color=POS, lw=1.6))
    axB.set_xlim(-0.3, 2.3)
    axB.set_xticks(x); axB.set_xticklabels(["hourly", "15-min", "5-min"])
    axB.set_ylabel("RI vs time-of-day lookup", fontsize=13)
    axB.set_xlabel("temporal resolution (same real records)", fontsize=13)
    axB.set_title("(b) Real Bucheon: the resolution reversal")
    axB.set_ylim(-0.44, 0.56); axB.grid(alpha=.3); axB.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out, dpi=300); fig.savefig(out[:-4] + ".pdf")
    print("saved", out, "(+pdf)")


if __name__ == "__main__":
    run()
