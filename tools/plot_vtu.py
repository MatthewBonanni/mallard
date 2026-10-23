#!/usr/bin/env python3
"""Render a cell field from a Mallard VTU file to PNG.
    plot_vtu.py SNAPSHOT.vtu OUTPUT.png [--var RHO] [--contours N] [--line]
        [--slice z] [--at VALUE]
2D meshes are drawn cell by cell; --line plots the field against x instead
(1D cases on a strip, averaging the cells at equal x). 3D meshes are drawn on
the plane through the cells nearest the middle of the --slice axis (or --at).
Vector fields (U) are drawn as their magnitude.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mallard_vtu import VTK_3D_TYPES, read_vtu, read_vtu_cells

AXES = {"x": 0, "y": 1, "z": 2}


def scalar(values):
    return np.linalg.norm(values, axis=1) if values.ndim == 2 else values


def cell_centers(path, var):
    pts, conn, offs, types, arrays = read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    sizes = offs - starts
    centers = np.empty((len(offs), 3))
    for n in np.unique(sizes):
        sel = np.nonzero(sizes == n)[0]
        centers[sel] = pts[conn[starts[sel, None] + np.arange(n)]].mean(axis=1)
    return centers, scalar(arrays[var]), np.isin(types, list(VTK_3D_TYPES)).any(), arrays.get("TIME")


def plot_line(ax, centers, vals, var):
    x = np.round(centers[:, 0], 12)
    xs, inverse = np.unique(x, return_inverse=True)
    mean = np.bincount(inverse, weights=vals) / np.bincount(inverse)
    ax.plot(xs, mean, ".-", ms=3, lw=1)
    ax.set_xlabel("x")
    ax.set_ylabel(var)
    ax.grid(alpha=0.3)


def plot_slice(ax, centers, vals, var, axis, at):
    k = AXES[axis]
    coord = centers[:, k]
    target = 0.5 * (coord.min() + coord.max()) if at is None else at
    nearest = coord[np.argmin(np.abs(coord - target))]
    extent = np.ptp(centers, axis=0).max()
    h = extent / max(len(vals), 1) ** (1 / 3)
    sel = np.abs(coord - nearest) <= 0.5 * h
    i, j = [a for a in range(3) if a != k]
    tpc = ax.tripcolor(centers[sel, i], centers[sel, j], vals[sel], shading="gouraud", cmap="viridis")
    ax.set_xlabel("xyz"[i])
    ax.set_ylabel("xyz"[j])
    ax.set_aspect("equal")
    return tpc, f"{axis} = {nearest:.4g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vtu")
    ap.add_argument("out")
    ap.add_argument("--var", default="RHO")
    ap.add_argument("--contours", type=int, default=0)
    ap.add_argument("--line", action="store_true", help="plot against x (1D cases)")
    ap.add_argument("--slice", choices=list(AXES), default="z", help="3D: axis normal to the plotted plane")
    ap.add_argument("--at", type=float, help="3D: position of the plane (default: the middle)")
    args = ap.parse_args()

    centers, vals, is_3d, time = cell_centers(args.vtu, args.var)
    title = f"{args.var}  min={vals.min():.4g} max={vals.max():.4g}"
    if time is not None:
        title = f"t = {time:.4g}   " + title
    if args.line:
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_line(ax, centers, vals, args.var)
    elif is_3d:
        fig, ax = plt.subplots(figsize=(7, 6))
        tpc, where = plot_slice(ax, centers, vals, args.var, args.slice, args.at)
        fig.colorbar(tpc, ax=ax, label=args.var)
        title += f"   ({where})"
    else:
        pts, tris, tri_cell, arrays = read_vtu(args.vtu)
        fig, ax = plt.subplots(figsize=(7, 6))
        tpc = ax.tripcolor(pts[:, 0], pts[:, 1], tris, facecolors=scalar(arrays[args.var])[tri_cell],
                           cmap="viridis")
        if args.contours:
            ax.tricontour(centers[:, 0], centers[:, 1], vals, args.contours, colors="k", linewidths=0.4)
        fig.colorbar(tpc, ax=ax, label=args.var)
        ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
