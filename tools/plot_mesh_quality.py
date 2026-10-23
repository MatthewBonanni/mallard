#!/usr/bin/env python3
"""Plot the mesh-quality fields of `mallard check --quality-vtu PREFIX` (2D).

    plot_mesh_quality.py PREFIX_000000.vtu OUT.png [--fields NON_ORTHOGONALITY SKEWNESS] [--worst 10]
        [--xlim X0 X1] [--ylim Y0 Y1]

One panel per field: the cells coloured by the field, and the worst cells
outlined, circled and numbered from the worst.
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


def read_quality(path):
    """Cell polygons (2D) and the quality fields of a --quality-vtu file."""
    pts, conn, offs, _, arrays = read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    return [pts[conn[s:e], :2] for s, e in zip(starts, offs)], arrays


def draw_quality(ax, polys, values, n_worst, highlight, edges, cmap="viridis", fontsize=8, among=None):
    """The cells coloured by values, the n_worst largest (of the cells among, a mask) circled and numbered;
    returns the cells' collection."""
    cells = PolyCollection(polys, array=values, cmap=cmap, edgecolors=edges, linewidths=0.15, rasterized=True)
    ax.add_collection(cells)
    order = np.argsort(-values, kind="stable")
    worst = (order if among is None else order[among[order]])[:n_worst]
    centers = np.array([polys[c].mean(axis=0) for c in worst])
    ax.add_collection(PolyCollection([polys[c] for c in worst], facecolors="none", edgecolors=highlight,
                                     linewidths=1.2))
    ax.scatter(*centers.T, s=160, facecolors="none", edgecolors=highlight, linewidths=1.4)
    for rank, center in enumerate(centers, 1):
        ax.annotate(str(rank), center, xytext=(7, 5), textcoords="offset points", color=highlight,
                    fontsize=fontsize, fontweight="bold")
    ax.set_aspect("equal")
    ax.autoscale_view()
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("vtu")
    ap.add_argument("out")
    ap.add_argument("--fields", nargs="+", default=["NON_ORTHOGONALITY", "SKEWNESS"])
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--xlim", type=float, nargs=2)
    ap.add_argument("--ylim", type=float, nargs=2)
    ap.add_argument("--width", type=float, default=11.0)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    polys, arrays = read_quality(args.vtu)
    n = len(args.fields)
    fig, axes = plt.subplots(1, n, figsize=(args.width, 0.85 * args.width / n), squeeze=False,
                             layout="constrained")
    for ax, field in zip(axes[0], args.fields):
        values = arrays[field]
        cells = draw_quality(ax, polys, values, args.worst, "#e4572e", (0.5, 0.5, 0.5, 0.35))
        fig.colorbar(cells, ax=ax, fraction=0.04, pad=0.02).set_label(LABELS.get(field, field))
        ax.set_title(f"{LABELS.get(field, field)}: max {values.max():.3g}, mean {values.mean():.3g}")
        if args.xlim:
            ax.set_xlim(*args.xlim)
        if args.ylim:
            ax.set_ylim(*args.ylim)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")


if __name__ == "__main__":
    main()
