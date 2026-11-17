#!/usr/bin/env python3
"""Streamwise velocity of the channel_inlet run across the channel in the column of cells nearest x = 3.5
against plane Poiseuille flow u = 6 U y (1 - y), compared as cell averages.

    plot_channel_inlet.py OUTPUT.png SNAPSHOT [--U 0.1] [--x 3.5]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools"))
from mallard.vtu import read_vtu_cells  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("snapshot")
    ap.add_argument("--U", type=float, default=0.1)
    ap.add_argument("--x", type=float, default=3.5)
    args = ap.parse_args()
    pts, conn, offs, _, data = read_vtu_cells(args.snapshot)
    starts = np.concatenate([[0], offs[:-1]])
    x = np.array([pts[conn[s:e], 0].mean() for s, e in zip(starts, offs)])
    a = np.array([pts[conn[s:e], 1].min() for s, e in zip(starts, offs)])
    b = np.array([pts[conn[s:e], 1].max() for s, e in zip(starts, offs)])
    centers = np.unique(np.round(x, 12))
    nearest = centers[np.argmin(np.abs(centers - args.x))]
    column = np.abs(x - nearest) < 1e-9
    a, b, u = a[column], b[column], data["U_X"][column]
    # Average of 6 U y (1 - y) over [a, b]
    exact = 6 * args.U * ((b ** 2 - a ** 2) / 2 - (b ** 3 - a ** 3) / 3) / (b - a)
    error = np.abs(u - exact).max() / args.U
    print(f"{args.snapshot}: x = {nearest:g}, {u.size} cells, max error / U = {error:.3e}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    y = np.linspace(0.0, 1.0, 200)
    ax.plot(6 * args.U * y * (1 - y), y, "k-", lw=1, label="plane Poiseuille flow")
    ax.plot(u, 0.5 * (a + b), "o", ms=4, label=f"Mallard at x = {nearest:g} (max error {error:.1e} U)")
    ax.set_xlabel("u_x")
    ax.set_ylabel("y")
    ax.legend(fontsize=9)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
