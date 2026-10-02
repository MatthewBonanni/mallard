#!/usr/bin/env python3
"""Compare the viscous shock tube (examples/viscous_shock_tube, Re = 200) at
t = 1 with the grid-converged reference of Zhou et al. (arXiv:1705.09062):
density contours near the floor with the reference triple point and vortex
height, and the wall density against their Table 1.

    plot_viscous_shock_tube.py SNAPSHOT.vtu OUTPUT.png

The wall density is that of the first row of cells. Needs a Cartesian quad
mesh (cells are binned into rows and columns).
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mallard_vtu import read_vtu

# Zhou et al., Table 1: wall density at t = 1 on the 1500 x 750 grid
REF_WALL = np.array([
    (0.3030, 39.8418), (0.4490, 37.0662), (0.5230, 52.6465), (0.5730, 42.4400), (0.5930, 40.5506),
    (0.6123, 47.3367), (0.6370, 39.3203), (0.6577, 36.9558), (0.6830, 49.6513), (0.7070, 77.9810),
    (0.7317, 108.3916), (0.7543, 92.5760), (0.7790, 64.5319), (0.7957, 59.4386), (0.8183, 95.3607),
    (0.8617, 117.6452), (0.9437, 96.4287), (0.9670, 98.2689), (0.9883, 81.8465), (0.9943, 82.7077)])
REF_TRIPLE_POINT = (0.58, 0.137)
REF_VORTEX_HEIGHT = 0.166


def structured(pts, tris, tri_cell, values):
    """Cell values as a (ny, nx) array and the cell-center coordinates."""
    n_cells = tri_cell.max() + 1
    centroid = np.zeros((n_cells, 2))
    count = np.zeros(n_cells)
    np.add.at(centroid, tri_cell, pts[tris].mean(axis=1))
    np.add.at(count, tri_cell, 1)
    centroid /= count[:, None]
    xs = np.unique(np.round(centroid[:, 0], 9))
    ys = np.unique(np.round(centroid[:, 1], 9))
    if len(xs) * len(ys) != n_cells:
        raise SystemExit("not a Cartesian quad mesh")
    ix = np.searchsorted(xs, np.round(centroid[:, 0], 9))
    iy = np.searchsorted(ys, np.round(centroid[:, 1], 9))
    grid = np.empty((len(ys), len(xs)))
    grid[iy, ix] = values
    return xs, ys, grid


def triple_point(xs, ys, rho):
    """Intersection of the lambda's front leg with the reflected shock above
    it, each fitted with a straight line to the density-gradient ridges
    (x between 0.4 and 0.7) within 0.03 of where the legs merge."""
    gx = np.abs(np.gradient(rho, xs, axis=1))
    sel = (xs > 0.4) & (xs < 0.7)
    xsel = xs[sel]
    ridges = []
    for j in np.nonzero(ys < 0.3)[0]:
        row = gx[j, sel]
        strong = row > 0.25 * row.max()
        peaks = np.nonzero(strong & (row >= np.roll(row, 1)) & (row >= np.roll(row, -1)))[0]
        ridges.append((ys[j], xsel[peaks]))
    split = [y for y, p in ridges if len(p) >= 2 and p[-1] - p[0] > 0.01]
    if not split:
        return None
    y_split = max(split)
    front = [(y, p[0]) for y, p in ridges if y_split - 0.03 < y <= y_split and len(p) >= 2]
    shock = [(y, p[-1]) for y, p in ridges if y_split + 0.01 < y < y_split + 0.04]
    (af, bf), (as_, bs) = [np.polyfit(*np.array(pts).T, 1) for pts in (front, shock)]  # x = a y + b
    y = (bs - bf) / (af - as_)
    return af * y + bf, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("output")
    args = ap.parse_args()

    pts, tris, tri_cell, data = read_vtu(args.snapshot)
    xs, ys, rho = structured(pts, tris, tri_cell, data["RHO"])
    wall = rho[0]
    at_ref = np.interp(REF_WALL[:, 0], xs, wall)
    err = at_ref - REF_WALL[:, 1]
    print(f"t = {data.get('TIME', float('nan')):.4f}, {len(xs)} x {len(ys)} cells")
    print("   x     rho_ref   rho      diff")
    for (x, r), m in zip(REF_WALL, at_ref):
        print(f"{x:.4f}  {r:8.3f}  {m:8.3f}  {m - r:+7.3f}")
    print(f"RMS difference {np.sqrt(np.mean(err ** 2)):.2f}, max |difference| {np.abs(err).max():.2f}")
    print(f"wall density range: min {wall.min():.2f} at x = {xs[wall.argmin()]:.4f}, "
          f"max {wall.max():.2f} at x = {xs[wall.argmax()]:.4f}")
    tp = triple_point(xs, ys, rho)
    if tp:
        print(f"triple point ~ ({tp[0]:.3f}, {tp[1]:.3f}); reference {REF_TRIPLE_POINT}")

    plt.rcParams.update({"font.size": 10})
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 8.5), height_ratios=[1.25, 1])
    sel_x = xs >= 0.3
    sel_y = ys <= 0.3
    sub = rho[np.ix_(sel_y, sel_x)]
    ax0.contourf(xs[sel_x], ys[sel_y], sub, levels=60, cmap="viridis")
    ax0.contour(xs[sel_x], ys[sel_y], sub, levels=30, colors="k", linewidths=0.3)
    ax0.plot(*REF_TRIPLE_POINT, "r+", ms=14, mew=2, label="reference triple point (0.58, 0.137)")
    ax0.axhline(REF_VORTEX_HEIGHT, color="r", ls="--", lw=1, label="reference vortex height 0.166")
    ax0.set_aspect("equal")
    ax0.set_xticks(np.arange(0.3, 1.001, 0.05), minor=True)
    ax0.set_yticks(np.arange(0.0, 0.301, 0.01), minor=True)
    ax0.grid(which="both", color="w", lw=0.2, alpha=0.5)
    ax0.legend(loc="upper left", fontsize=8)
    ax0.set_title("Density at t = 1")
    ax1.plot(xs, wall, "k-", lw=1, label=f"Mallard, {len(xs)} x {len(ys)}")
    ax1.plot(REF_WALL[:, 0], REF_WALL[:, 1], "ro", ms=4, label="Zhou et al., 1500 x 750 (Table 1)")
    ax1.set_xlim(0.3, 1.0)
    ax1.set_xlabel("x")
    ax1.set_ylabel("wall density")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
