"""Numerical soot foil and cell size of a 2D cellular detonation run (examples/detonation_2d).

    python tools/soot_foil.py SERIES.pvd D_CJ [--every 10] [--png FOIL.png]

Reads the outputs of a run on a generated "cartesian" mesh (P, and P_MAX: the
largest pressure of each cell so far). Reports:

  D        the mean front speed over the second half of the run against D_CJ,
           from the front position of every output (per row of cells, the
           rightmost cell with p above twice the initial minimum, averaged
           across the channel);
  n_tp     the triple points on the front in every EVERY-th output: peaks
           along y (prominence --prominence, default 15%, of the mean) of the largest pressure
           within 3 mm behind the front, with the walls as symmetry planes,
           and the cell width they imply, lambda = 2 W / n_tp (two triple
           points, one running each way, per cell width).

With --png, writes the foil of the last output (log of P_MAX / p0, as a soot
foil) up to the front.
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import grid_fields  # noqa: E402


def front_position(x, p, p0):
    """Mean over the rows of the rightmost cell with p > 2 p0."""
    above = p > 2.0 * p0
    idx = x.size - 1 - np.argmax(above[::-1, :], axis=0)
    return x[idx].mean()


def triple_points(x, y, p, p0, x_front, depth=0.003, prominence=0.15):
    """Triple points on the front: peaks along y of the largest pressure within DEPTH behind the front."""
    from scipy.signal import find_peaks
    near = (x > x_front - depth) & (x <= x_front + 0.001)
    profile = p[near].max(axis=0) / p0
    mirrored = np.concatenate([profile[::-1], profile, profile[::-1]])  # walls are symmetry planes
    peaks, _ = find_peaks(mirrored, prominence=prominence * profile.mean())
    n = y.size
    return int(((peaks >= n) & (peaks < 2 * n)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pvd")
    ap.add_argument("D_cj", type=float)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--prominence", type=float, default=0.15)
    ap.add_argument("--png")
    args = ap.parse_args()
    base = os.path.dirname(args.pvd)
    files = re.findall(r'file="([^"]+)"', open(args.pvd).read())
    rows, p0 = [], None
    for f in files:
        t, x, y, fields = grid_fields(os.path.join(base, f), ["P"])
        p0 = fields["P"].min() if p0 is None else p0
        rows.append((t, front_position(x, fields["P"], p0)))
    rows = np.array(rows)
    half = rows[rows[:, 0] >= 0.5 * rows[-1, 0]]
    D = np.polyfit(half[:, 0], half[:, 1], 1)[0]
    print(f"front: {rows[0, 1] * 1e3:.1f} mm to {rows[-1, 1] * 1e3:.1f} mm in {rows[-1, 0] * 1e6:.1f} us")
    print(f"mean speed over the second half: D = {D:.1f} m/s, D / D_CJ - 1 = {D / args.D_cj - 1:+.4f}")

    W = None
    print(f"{'t [us]':>7} {'x_f [mm]':>8} {'n_tp':>5} {'lambda [mm]':>11}")
    for (t, xf), f in list(zip(rows, files))[::args.every]:
        t, x, y, fields = grid_fields(os.path.join(base, f), ["P"])
        W = y[-1] + 0.5 * (y[1] - y[0])
        n = triple_points(x, y, fields["P"], p0, xf, prominence=args.prominence)
        print(f"{t * 1e6:7.1f} {xf * 1e3:8.1f} {n:5d} {2 * W / n * 1e3 if n else float('nan'):11.1f}")
    t, x, y, fields = grid_fields(os.path.join(base, files[-1]), ["P_MAX"])
    foil = fields["P_MAX"]
    x_front = rows[-1, 1]
    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keep = x < x_front
        fig, ax = plt.subplots(figsize=(14, 14 * W / (x[keep][-1] - x[keep][0]) + 0.8))
        img = np.log(foil[keep] / p0).T
        lo, hi = np.percentile(img[:, img.shape[1] // 4:], [2, 99.5])
        ax.imshow(img, origin="lower", cmap="gray_r", vmin=lo, vmax=hi, aspect="equal",
                  extent=[x[keep][0] * 1e3, x[keep][-1] * 1e3, 0, W * 1e3])
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.tight_layout()
        fig.savefig(args.png, dpi=150)


if __name__ == "__main__":
    main()
