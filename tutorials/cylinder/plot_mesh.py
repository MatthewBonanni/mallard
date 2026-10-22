#!/usr/bin/env python3
"""The cylinder tutorial's Gmsh mesh: the whole disk with its boundary zones,
and the cells near the cylinder (the quadrilateral ring shaded).

    python plot_mesh.py cylinder.msh [--out DIR]

Writes the figure cylinder_mesh.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import matplotlib.pyplot as plt  # noqa: E402
from gmsh_mesh import read_msh  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from tutfig import label_box, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mesh")
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    args = ap.parse_args()
    xyz, cells, zones = read_msh(args.mesh)
    polys = [xyz[c, :2] for c in cells]
    quads = np.array([len(c) == 4 for c in cells])
    print(f"{len(cells)} cells: {quads.sum()} quadrilaterals, {(~quads).sum()} triangles")

    def draw(t):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 1.35]))
        for a, lim in ((ax, (-21, 21, -21, 21)), (bx, (-1.2, 3.2, -1.3, 1.3))):
            face = [to_rgba(t["teal"], 0.35) if q else "none" for q in quads]
            a.add_collection(PolyCollection(polys, facecolors=face, edgecolors=t["mesh"], linewidths=0.25,
                                            rasterized=True))
            for name, color in (("cylinder", t["flame"]), ("farfield", t["blue"])):
                a.add_collection(LineCollection([xyz[f, :2] for f in zones[name]], colors=color, linewidths=2.5))
            a.set_xlim(lim[0], lim[1])
            a.set_ylim(lim[2], lim[3])
            a.set_aspect("equal")
            a.axis("off")
        ax.text(0, 22.5, "farfield", color=t["blue"], ha="center", fontsize=12, bbox=label_box(t))
        ax.add_patch(plt.Rectangle((-1.2, -1.3), 4.4, 2.6, fill=False, edgecolor=t["ink"], lw=1))
        bx.text(-0.95, 1.05, "cylinder", color=t["flame"], fontsize=12, bbox=label_box(t))
        bx.text(1.6, 0.85, "triangles, fine through the wake", color=t["ink"], fontsize=11, bbox=label_box(t))
        bx.annotate("quadrilateral ring (O-grid)", (0.0, -0.8), (0.9, -1.15), color=t["ink"], fontsize=11,
                    arrowprops=dict(arrowstyle="-|>", color=t["ink2"]), bbox=label_box(t))
        return fig

    render(draw, args.out, "cylinder_mesh")


if __name__ == "__main__":
    main()
