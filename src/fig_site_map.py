"""Real-map site figure for Bucheon: OpenStreetMap basemap with the 16 intersections
placed at their real geocoded coordinates (Nominatim; names from the dataset guideline).
Node 8 (구터미널사거리) is affine-interpolated from the other 15 (mis-geocoded).
Web-Mercator global-pixel placement (exact). OSM attribution shown.

    python3 -m src.fig_site_map
"""

import json
import math
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .bucheon import THREE_LEG

ACC, NEG, MUT = "#22687e", "#c0524a", "#8a94a3"
TILE = 256
NAMES = {1: "Sangdong Stn.", 3: "Seokcheon", 4: "Munye", 12: "Central Park",
         13: "Post office (3-leg)", 16: "Boksagol"}  # a few labels for context


def _gpix(lat, lon, z):
    n = TILE * 2 ** z
    x = (lon + 180.0) / 360.0 * n
    latr = math.radians(lat)
    y = (1.0 - math.log(math.tan(latr) + 1.0 / math.cos(latr)) / math.pi) / 2.0 * n
    return x, y


def _tile(z, x, y, cache="/tmp/osm_cache"):
    os.makedirs(cache, exist_ok=True)
    fp = os.path.join(cache, f"{z}_{x}_{y}.png")
    if not os.path.exists(fp):
        subprocess.run(["curl", "-sSL", "--max-time", "20", "-A",
                        "turning-movement-research/1.0 (academic figure)", "-o", fp,
                        f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"], check=True)
    return Image.open(fp).convert("RGB")


def fig_site_bucheon_map(out="results/fig_site_bucheon.png", z=16, margin=90):
    g = json.load(open("data/bucheon_geo.json"))
    coords = {int(k): v for k, v in g["coords"].items()}
    interp = set(g.get("interpolated", []))
    GL = {n: ((n - 1) // 4, (n - 1) % 4) for n in range(1, 17)}
    gp = {n: _gpix(lat, lon, z) for n, (lat, lon) in coords.items()}
    xs = [p[0] for p in gp.values()]; ys = [p[1] for p in gp.values()]
    xL, xR = min(xs) - margin, max(xs) + margin
    yT, yB = min(ys) - margin, max(ys) + margin        # smaller gy = north = top
    # tile range covering the window
    tx0, tx1 = int(xL // TILE), int(xR // TILE)
    ty0, ty1 = int(yT // TILE), int(yB // TILE)
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE))
    for i, tx in enumerate(range(tx0, tx1 + 1)):
        for j, ty in enumerate(range(ty0, ty1 + 1)):
            mosaic.paste(_tile(z, tx, ty), (i * TILE, j * TILE))
    ext = [tx0 * TILE, (tx1 + 1) * TILE, (ty1 + 1) * TILE, ty0 * TILE]  # L,R,B,T (gy down)

    fig, ax = plt.subplots(figsize=(7.4, 7.9))
    ax.imshow(np.asarray(mosaic), extent=ext, origin="upper", zorder=0, alpha=0.94)
    # edges: grid-adjacent node pairs, drawn at real positions
    for n, (r, c) in GL.items():
        for m, (r2, c2) in GL.items():
            if n < m and abs(r - r2) + abs(c - c2) == 1:
                ax.plot([gp[n][0], gp[m][0]], [gp[n][1], gp[m][1]], "-",
                        color=ACC, lw=2.0, alpha=0.8, zorder=2)
    for n, (x, y) in gp.items():
        three = n in THREE_LEG
        fc = NEG if three else ACC
        ax.scatter([x], [y], s=460, c=fc, edgecolors="w", lw=1.4, zorder=3)
        ax.text(x, y, str(n), ha="center", va="center", color="w", fontsize=12,
                fontweight="bold", zorder=4)
    ax.set_xlim(xL, xR); ax.set_ylim(yB, yT)          # yB (south) bottom, yT (north) top
    ax.scatter([], [], s=150, c=ACC, label="4-leg signalized intersection")
    ax.scatter([], [], s=150, c=NEG, label="3-leg junction (node 13)")
    ax.text(0.99, 0.01, "© OpenStreetMap contributors", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color="#333", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.015), ncol=1, frameon=False)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    fig_site_bucheon_map()
