#!/usr/bin/env python3
"""Plot the mesh-quality fields of `Mallard --check --quality-vtu PREFIX` (2D).

    plot_mesh_quality.py PREFIX_000000.vtu OUT.png [--fields NON_ORTHOGONALITY SKEWNESS] [--worst 10]
        [--xlim X0 X1] [--ylim Y0 Y1]

One panel per field: the cells coloured by the field with their edges drawn,
and the worst cells outlined, circled and numbered from the worst. Grey text
on a transparent background, so the figure reads on light and dark pages.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from mallard_vtu import read_vtu_cells

LABELS = {"NON_ORTHOGONALITY": "non-orthogonality [deg]", "SKEWNESS": "skewness",
          "ASPECT_RATIO": "aspect ratio", "VOLUME_RATIO": "volume ratio"}
GREY = "#8a8f98"
HIGHLIGHT = "#ff4f3a"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("vtu")
    ap.add_argument("out")
    ap.add_argument("--fields", nargs="+", default=["NON_ORTHOGONALITY", "SKEWNESS"])
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--xlim", type=float, nargs=2)
    ap.add_argument("--ylim", type=float, nargs=2)
    ap.add_argument("--width", type=float, default=11.0)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    pts, conn, offs, _, arrays = read_vtu_cells(args.vtu)
    starts = np.concatenate([[0], offs[:-1]])
    polys = [pts[conn[s:e], :2] for s, e in zip(starts, offs)]
    centers = np.array([p.mean(axis=0) for p in polys])

    plt.rcParams.update({"text.color": GREY, "axes.labelcolor": GREY, "xtick.color": GREY,
                         "ytick.color": GREY, "axes.edgecolor": GREY, "font.size": 10})
    n = len(args.fields)
    fig, axes = plt.subplots(1, n, figsize=(args.width, 0.85 * args.width / n), squeeze=False,
                             layout="constrained")
    for ax, field in zip(axes[0], args.fields):
        values = arrays[field]
        cells = PolyCollection(polys, array=values, cmap="viridis", edgecolors=(0.5, 0.5, 0.5, 0.35),
                               linewidths=0.15)
        ax.add_collection(cells)
        worst = np.argsort(-values, kind="stable")[: args.worst]
        ax.add_collection(PolyCollection([polys[c] for c in worst], facecolors="none", edgecolors=HIGHLIGHT,
                                         linewidths=1.2))
        ax.scatter(*centers[worst].T, s=160, facecolors="none", edgecolors=HIGHLIGHT, linewidths=1.4)
        for rank, c in enumerate(worst, 1):
            ax.annotate(str(rank), centers[c], xytext=(7, 5), textcoords="offset points", color=HIGHLIGHT,
                        fontsize=8, fontweight="bold")
        bar = fig.colorbar(cells, ax=ax, fraction=0.04, pad=0.02)
        bar.set_label(LABELS.get(field, field))
        bar.outline.set_edgecolor(GREY)
        ax.set_title(f"{LABELS.get(field, field)}: max {values.max():.3g}, mean {values.mean():.3g}\n"
                     f"the {len(worst)} worst cells circled", color=GREY)
        ax.set_aspect("equal")
        ax.autoscale_view()
        if args.xlim:
            ax.set_xlim(*args.xlim)
        if args.ylim:
            ax.set_ylim(*args.ylim)
        ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
