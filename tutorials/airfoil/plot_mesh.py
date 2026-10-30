#!/usr/bin/env python3
"""Geometry and mesh figures of the airfoil tutorial.

    python plot_mesh.py naca0012.msh [--out DIR]

Writes airfoil_geometry (the C-shaped domain, its boundary zones and the
airfoil) and airfoil_mesh (the whole mesh, the airfoil, and the leading and
trailing edges), each for light and dark pages.
"""
import argparse
import os
import sys

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import matplotlib.pyplot as plt  # noqa: E402
from tutfig import label_box, render  # noqa: E402
from gmsh_mesh import read_msh  # noqa: E402


def naca0012(n=200):
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    y = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    return np.concatenate([x[::-1], x[1:]]), np.concatenate([y[::-1], -y[1:]])


def zone_segments(xyz, zones, name):
    return [xyz[f][:, :2] for f in zones[name]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mesh")
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    args = ap.parse_args()
    xyz, cells, zones = read_msh(args.mesh)
    polys = [xyz[c][:, :2] for c in cells]
    R = np.abs(xyz[:, 1]).max()
    xa, ya = naca0012()

    def geometry(t):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
        far, wall = t["series"][0], t["series"][1]
        ax.add_collection(LineCollection(zone_segments(xyz, zones, "farfield"), colors=far, linewidths=2.5))
        ax.fill(xa, ya, color=wall)
        ax.plot([1, 1 + R], [0, 0], color=t["muted"], lw=1, ls="--")
        ax.annotate("", (1 + R, -0.12 * R), (1, -0.12 * R), arrowprops={"arrowstyle": "<->", "color": t["ink2"]})
        ax.text(1 + R / 2, -0.17 * R, f"R = {R:g} c", ha="center", va="top", color=t["ink2"])
        ax.text(1 - 0.72 * R, 0.78 * R, '"farfield"', color=far, ha="center", fontsize=11)
        ax.text(1 + 0.55 * R, 0.08 * R, "wake cut\n(interior)", color=t["muted"], ha="center", fontsize=9)
        ax.annotate('"airfoil"', (1, 0), (-0.25 * R, -0.45 * R), color=wall, fontsize=11,
                    arrowprops={"arrowstyle": "->", "color": wall})
        ax.set_aspect("equal")
        ax.set_xlim(1 - 1.1 * R, 1 + 1.1 * R)
        ax.set_ylim(-1.1 * R, 1.1 * R)
        ax.set_title("Domain (chords)")
        ax.grid(False)
        ax.set_xlabel("x / c")
        ax.set_ylabel("y / c")

        alpha = np.radians(1.0)
        bx.fill(xa, ya, color=wall, alpha=0.9)
        bx.plot(xa, ya, color=wall, lw=1.5)
        for y0 in (-0.12, 0.0, 0.12):
            bx.add_patch(FancyArrowPatch((-0.45, y0 - 0.06), (-0.45 + 0.3 * np.cos(alpha), y0 - 0.06 + 0.3 * np.sin(alpha)),
                                         arrowstyle="-|>", mutation_scale=14, color=t["ink2"], lw=1.5))
        bx.text(-0.3, 0.16, "M = 0.5, α = 1°\nRe = 5000", ha="center", color=t["ink"], bbox=label_box(t))
        bx.annotate("", (1.0, -0.17), (0.0, -0.17), arrowprops={"arrowstyle": "<->", "color": t["ink2"]})
        bx.text(0.5, -0.19, "chord c = 1", ha="center", va="top", color=t["ink2"])
        bx.text(0.3, 0.1, "NACA 0012, 12% thick,\nclosed trailing edge", color=t["ink2"], fontsize=9)
        bx.set_aspect("equal")
        bx.set_xlim(-0.6, 1.2)
        bx.set_ylim(-0.3, 0.3)
        bx.set_title("Airfoil and free stream")
        bx.grid(False)
        bx.set_xlabel("x / c")
        return fig

    def mesh(t):
        fig, axs = plt.subplots(2, 2, figsize=(10, 7.2))
        views = [((1 - 1.05 * R, 1 + 1.05 * R), (-1.05 * R, 1.05 * R), "Whole domain"),
                 ((-0.25, 1.4), (-0.38, 0.38), "Airfoil"),
                 ((-0.03, 0.09), (-0.045, 0.045), "Leading edge"),
                 ((0.94, 1.06), (-0.045, 0.045), "Trailing edge")]
        colors = {"airfoil": t["series"][1], "farfield": t["series"][0]}
        for ax, (xl, yl, title) in zip(axs.flat, views):
            ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors=t["mesh"], rasterized=True,
                                             linewidths=0.15 if title == "Whole domain" else 0.3))
            for name, segs in zones.items():
                ax.add_collection(LineCollection([s[:, :2] for s in (xyz[f] for f in segs)],
                                                 colors=colors.get(name, t["series"][2]), linewidths=2.2, label=f'"{name}"', rasterized=True))
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.grid(False)
            ax.tick_params(labelsize=8)
        axs[0, 0].legend(loc="upper left", fontsize=9)
        n_wall = len(zones["airfoil"])
        fig.suptitle(f"{len(cells):,} quadrilaterals, {n_wall} faces on the airfoil", color=t["ink"])
        fig.tight_layout()
        return fig

    render(geometry, args.out, "airfoil_geometry")
    render(mesh, args.out, "airfoil_mesh")


if __name__ == "__main__":
    main()
