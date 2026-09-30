"""Figure 3: boundary recovery against the residual same-clock-hour demand variability.

Panel (a) plots the site-1 boundary RI of the diffused mode over the clock-hour lookup at three
resolutions, under leave-one-date-out folds (data/e13_boundary_lodo_modes.json,
data/e14_boundary_lodo_5min.json). Panel (b) places the two sites at the matched 15-min resolution
on the per-intersection axis: the median and range of demandCV_res across intersections
(data/m19_screen_input_transfer.json, data/m19_xc_transfer.json) against the boundary RI of each
site (site 1 from data/e13_boundary_lodo_modes.json; site 2 from the h=1 stratum of
data/m38_realsite_hopstrat.json).

    python3 -m src.fig_reconcile          # writes results/fig_reconcile.png and .pdf
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "axes.titlesize": 15,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 13})

# boundary RI over a same-clock-hour (ToD) lookup, by temporal resolution.
# (demandCV_res, RI_ToD, resolution-label)
BUCHEON = [(0.064, 0.726, "hourly"), (0.176, 0.779, "15-min"), (0.222, 0.836, "5-min")]  # leave-one-date-out folds
BUC, XC = "#c0524a", "#22687e"


def _plot(ax, pts, color, label, marker, filled=True, lab_dy=-15, lab_va="top", lab_dx=7):
    # convention shared with Figure 3: filled = directionally validated, open = marginal/exploratory.
    # resolution labels sit off the connecting line (Bucheon below, Xuancheng above).
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    face = color if filled else "white"
    ax.plot(x, y, "-", color=color, lw=2.2, marker=marker, ms=10, zorder=4, label=label,
            markerfacecolor=face, markeredgecolor=color, markeredgewidth=1.6)
    for cvv, riv, res in pts:
        ax.annotate(res, (cvv, riv), textcoords="offset points", xytext=(lab_dx, lab_dy),
                    fontsize=10.5, color=color, va=lab_va, ha="left")


def main(out="results/fig_reconcile.png"):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---- panel (a): within-site resolution sweep at Bucheon, network-aggregate axis ----
    axA.axhspan(-0.45, 0.0, facecolor="#8a94a3", alpha=0.10, zorder=0)
    axA.axhline(0.0, color="#555", lw=1.2)
    axA.axvspan(0.056, 0.063, facecolor="#6b7280", alpha=0.22, zorder=0)
    axA.annotate("clean-trace crossing\n$[0.056,\\,0.063]$", (0.063, 0.08), (0.105, 0.04),
                 color="#6b7280", fontsize=10.5, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", lw=0.8, color="#6b7280"))
    axA.axvline(0.0606, color="#6b7280", ls="--", lw=1.1, zorder=1)
    axA.annotate("corrected for this record's\ncounting noise: $0.061$", (0.0606, 0.45), (0.098, 0.45),
                 color="#6b7280", fontsize=10.5, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#6b7280"))
    x = [p[0] for p in BUCHEON]; y = [p[1] for p in BUCHEON]
    axA.plot(x, y, "-o", color=BUC, mfc=BUC, mec=BUC, mew=1.8, lw=1.8, ms=9)
    for xi, yi, lab in BUCHEON:
        axA.annotate(lab, (xi, yi), textcoords="offset points", xytext=(7, -15),
                     fontsize=10.5, color=BUC)
    axA.set_xlabel("$\\gamma$ (network aggregate)")
    axA.set_ylabel("boundary RI, diffused mode\nover the clock-hour lookup")
    axA.set_title("(a) within site: Bucheon across resolutions", fontsize=13)
    axA.grid(ls=":", alpha=0.4); axA.set_xlim(0.03, 0.255); axA.set_ylim(-0.05, 0.95)

    # ---- panel (b): cross-site at matched 15-min, per-intersection axis ----
    axB.axhline(0.0, color="#555", lw=1.2)
    axB.axhspan(-0.45, 0.0, facecolor="#8a94a3", alpha=0.10, zorder=0)
    # (median, lo, hi, RI, label, colour, marker, filled)
    CROSS = [(0.226, 0.147, 0.251, 0.779, "Bucheon", BUC, "o", True),
             (0.316, 0.228, 0.789, 0.575, "Xuancheng", XC, "s", True)]
    for med, lo, hi, ri, lab, col, mk, filled in CROSS:
        axB.errorbar([med], [ri], xerr=[[med - lo], [hi - med]], fmt=mk, color=col,
                     mfc=col if filled else "white", mec=col, mew=1.8, ms=11,
                     elinewidth=1.6, capsize=4, label=lab)
    axB.set_ylabel("boundary RI, diffused mode\nover the clock-hour lookup")
    axB.set_xlabel("$\\gamma$ across intersections")
    axB.set_title("(b) across sites at matched 15-min", fontsize=13)
    axB.grid(ls=":", alpha=0.4); axB.set_xlim(0.10, 0.83); axB.set_ylim(-0.05, 0.95)
    axB.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")
    print("saved", out, "(+pdf)")


if __name__ == "__main__":
    main()
