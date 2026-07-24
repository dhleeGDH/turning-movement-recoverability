"""Figure 1, characterization-centric: a turning count factors into a demand level and a turn split;
the level is recoverable by a closed-form anchor, the split is not recovered by any tested estimator
class (the eight probe classes are listed in the main-text table, none named in the figure so that no
single instrument reads as a proposed method).

    python3 -m src.fig_characterization
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

GREEN, RED, GREY, INK = "#2f7d4f", "#b23a48", "#8a94a3", "#222222"


def _box(ax, x, y, w, h, text, fc, ec, fs=10.5, weight="normal", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, weight=weight, zorder=3)


def _arrow(ax, xy1, xy2, color=GREY, lw=1.6):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=lw, color=color, zorder=1))


def main(out="results/fig_characterization.png"):
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    # decomposition header
    _box(ax, 3.1, 5.25, 3.8, 0.62, r"turning count  $x_m = v_a \cdot s_m$",
         "#eef1f4", INK, fs=13, weight="bold")
    _arrow(ax, (4.3, 5.25), (2.6, 4.50)); _arrow(ax, (5.7, 5.25), (7.4, 4.50))

    # LEFT: demand level, recoverable (header -> mechanism -> outcome)
    _box(ax, 0.35, 3.72, 4.3, 0.62, r"demand level  $v_a$   (RECOVERABLE)",
         "#e7f3ec", GREEN, fs=11.5, weight="bold", tc=GREEN)
    _box(ax, 0.35, 2.30, 4.3, 1.10,
         "closed-form demand anchor\nrescales the historical count\nby a boundary-diffused ratio",
         "#f5faf7", GREEN, fs=10.5)
    _box(ax, 0.35, 1.15, 4.3, 0.72,
         "reaches the achievable-accuracy\nbenchmark (the ceiling)",
         "#e7f3ec", GREEN, fs=10.5, weight="bold", tc=GREEN)
    _arrow(ax, (2.5, 3.72), (2.5, 3.42), GREEN); _arrow(ax, (2.5, 2.30), (2.5, 1.89), GREEN)

    # RIGHT: turn split, not recovered (header -> the eight probes -> outcome)
    _box(ax, 5.35, 3.72, 4.3, 0.62, r"turn split  $s_m$   (NOT RECOVERED)",
         "#f6e7e9", RED, fs=11.5, weight="bold", tc=RED)
    _box(ax, 5.35, 2.30, 4.3, 1.10,
         "eight tested estimator classes,\nfrom a learned adjustment to a\ngraph foundation model (Table 4)",
         "#fbf4f5", RED, fs=10.5)
    _box(ax, 5.35, 1.15, 4.3, 0.72,
         "none beats the frozen split\n(fixed-time, from $o$)",
         "#f6e7e9", RED, fs=10.5, weight="bold", tc=RED)
    _arrow(ax, (7.5, 3.72), (7.5, 3.42), RED); _arrow(ax, (7.5, 2.30), (7.5, 1.89), RED)

    # bottom conclusion bar (two clean lines that fit inside the box)
    _box(ax, 0.55, 0.18, 8.9, 0.72,
         "the anchor is near the estimated ceiling, and achievable accuracy is\n"
         r"governed by the residual demand variability $\mathrm{demandCV}_{\mathrm{res}}$",
         "#eef1f4", GREY, fs=10.5, weight="bold")
    _arrow(ax, (2.5, 1.15), (3.1, 0.90), GREY); _arrow(ax, (7.5, 1.15), (6.9, 0.90), GREY)

    fig.savefig(out, dpi=300, bbox_inches="tight"); fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")
    print("saved", out, "(+pdf)")


if __name__ == "__main__":
    main()
