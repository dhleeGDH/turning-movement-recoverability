"""Two-site regime figure: boundary RI over a same-clock-hour lookup versus the residual
same-clock-hour demand variability demandCV_res, for both real sites across temporal resolution.
The two sites bracket the operating regime: low-residual Bucheon crosses zero as resolution rises
(the reversal), high-residual Xuancheng is positive throughout. RI magnitude is site-dependent
(volume/sensing modality), so the sites bracket the demandCV_res direction rather than trace one
curve.

    python3 -m src.fig_reconcile
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12})

# boundary RI over a same-clock-hour (ToD) lookup, by temporal resolution.
# (demandCV_res, RI_ToD, resolution-label)
BUCHEON = [(0.064, -0.20, "hourly"), (0.176, 0.25, "15-min"), (0.222, 0.36, "5-min")]   # reversal
XUANCHENG = [(0.087, 0.60, "hourly"), (0.164, 0.60, "15-min")]                           # flat, high
BUC, XC = "#c0524a", "#22687e"


def _plot(ax, pts, color, label, marker):
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    ax.plot(x, y, "-", color=color, lw=2.2, marker=marker, ms=10, zorder=4, label=label,
            markeredgecolor="w", markeredgewidth=1.0)
    for cvv, riv, res in pts:
        ax.annotate(res, (cvv, riv), textcoords="offset points", xytext=(6, -14),
                    fontsize=10.5, color=color)


def main(out="results/fig_reconcile.png"):
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.axhspan(-0.45, 0.0, facecolor="#8a94a3", alpha=0.10, zorder=0)
    ax.axhline(0.0, color="#555", lw=1.2, ls="-")
    ax.text(0.235, -0.055, "$\\mathrm{RI}\\leq 0$: no recovery over the lookup", color="#6b7280", fontsize=11,
            ha="right", va="top")

    _plot(ax, BUCHEON, BUC, "Bucheon (camera, low-residual)", "o")
    _plot(ax, XUANCHENG, XC, "Xuancheng (trajectory, high-residual)", "s")

    ax.annotate("crosses zero as resolution\nrises: the reversal", (0.064, -0.20), (0.11, -0.30),
                color=BUC, fontsize=11.5, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=BUC, lw=1.5))
    ax.annotate("already above threshold:\npositive at every resolution", (0.087, 0.60), (0.175, 0.50),
                color=XC, fontsize=11.5, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=XC, lw=1.5))

    ax.set_xlabel("Residual same-clock-hour demand variability $\\mathrm{demandCV}_{\\mathrm{res}}$")
    ax.set_ylabel("boundary RI over a same-clock-hour lookup")
    ax.set_title("Boundary RI versus residual demand variability at two real sites")
    ax.grid(ls=":", alpha=0.4); ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.70), framealpha=0.9)
    ax.set_xlim(0.03, 0.245); ax.set_ylim(-0.45, 0.72)
    fig.tight_layout(); fig.savefig(out, dpi=300); fig.savefig(out[:-4] + ".pdf")
    print("saved", out, "(+pdf)")


if __name__ == "__main__":
    main()
