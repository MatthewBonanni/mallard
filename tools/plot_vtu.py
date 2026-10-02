#!/usr/bin/env python3
"""Render a cell field from a Mallard VTU file to PNG."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mallard_vtu import read_vtu


def load(path, var):
    pts, tris, tri_cell, arrays = read_vtu(path)
    return pts, tris, arrays[var][tri_cell]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtu")
    ap.add_argument("out")
    ap.add_argument("--var", default="RHO")
    ap.add_argument("--contours", type=int, default=0)
    args = ap.parse_args()
    pts, tris, vals = load(args.vtu, args.var)
    fig, ax = plt.subplots(figsize=(7, 6))
    tpc = ax.tripcolor(pts[:, 0], pts[:, 1], tris, facecolors=vals, cmap="viridis")
    fig.colorbar(tpc, ax=ax, label=args.var)
    ax.set_aspect("equal")
    ax.set_title(f"{args.var}  min={vals.min():.4g} max={vals.max():.4g}")
    fig.savefig(args.out, dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
