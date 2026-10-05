#!/usr/bin/env python3
"""Axial velocity of an axisymmetric Hagen-Poiseuille run against the exact
profile u = U (1 - r^2 / R^2), compared as r-weighted cell averages.

    plot_poiseuille.py OUTPUT.png RUN.vtu [RUN.vtu ...] [--U 0.01] [--R 1]
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mallard_vtu import read_vtu_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--U", type=float, default=0.01)
    ap.add_argument("--R", type=float, default=1.0)
    args = ap.parse_args()
    U, R = args.U, args.R

    fig, ax = plt.subplots(figsize=(7, 4.5))
    r = np.linspace(0.0, R, 200)
    ax.plot(U * (1 - (r / R) ** 2), r, "k-", lw=1, label="exact")
    for path in args.runs:
        pts, conn, offs, _, data = read_vtu_cells(path)
        starts = np.concatenate([[0], offs[:-1]])
        y = [pts[conn[s:e], 1] for s, e in zip(starts, offs)]
        a = np.array([v.min() for v in y])
        b = np.array([v.max() for v in y])
        # r-weighted average of U (1 - r^2 / R^2) over [a, b] (rectangular cells)
        exact = U * (1 - (a ** 2 + b ** 2) / (2 * R ** 2))
        error = np.abs(data["U_X"] - exact).mean() / U
        print(f"{path}: {len(a)} cells, L1 error / U = {error:.3e}")
        ax.plot(data["U_X"], 0.5 * (a + b), ".", ms=4, label=f"{path} ({error:.1e})")
    ax.set_xlabel("u_x")
    ax.set_ylabel("r")
    ax.legend(fontsize=8)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
