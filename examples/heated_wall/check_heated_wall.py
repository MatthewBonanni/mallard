#!/usr/bin/env python3
"""The heated-wall example (examples/heated_wall): the energy in the closed box
against the heat that entered through the floor, q A t, and where it entered.

    check_heated_wall.py PREFIX      (e.g. solut/box)

Globally: the energy gained at each snapshot against q A t (a relative
difference above 1e-10 fails). Locally, at the first snapshot after the start
(1 ms, when the heat has conducted 0.15 mm into the 2.5 mm floor cells): the
heat taken up by the floor cells of each of 8 bins along x, sum rho cp (T -
T_far) V with T_far the mean temperature of the upper half (which the box's
uniform compression warms), against the integral of q over the bin times t. A
uniform q of the same total would give every bin the same heat, while q gives
the bins under the cosine's peak about 2.6 times those under its trough; a
bin more than 2% off fails. Exits 1 on any failure.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools"))
from mallard import vtu  # noqa: E402

Q_TOTAL = 2.0e4 * 0.1   # W per metre of depth: the integral of q over the 0.1 m floor (the cosine integrates to 0)
LX, CP = 0.1, 1.4 * 287.0 / 0.4
BINS = 8


def q_integral(a, b):
    """The integral of q = 2e4 (1 + 0.5 cos(2 pi x / LX)) over [a, b]."""
    return 2.0e4 * ((b - a) + 0.5 * LX / (2.0 * np.pi) * (np.sin(2.0 * np.pi * b / LX) - np.sin(2.0 * np.pi * a / LX)))


def cells(path):
    pts, conn, offs, _, arrays = vtu.read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    quads = np.array([conn[s:e] for s, e in zip(starts, offs)])
    x, y = pts[quads][..., 0], pts[quads][..., 1]
    area = 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
    return x.mean(axis=1), y.mean(axis=1), area, arrays


def energy(path):
    _, _, area, a = cells(path)
    return a["TIME"], np.sum(a["RHOE"] * area)


def local_heat(path, t0):
    """(heat taken up by the floor cells of each bin, the integral of q over it times t)."""
    cx, cy, area, a = cells(path)
    floor = cy < cy.min() + 1e-9
    t_far = a["T"][cy > 0.5 * cy.max()].mean()
    heat = a["RHO"] * CP * (a["T"] - t_far) * area
    edges = np.linspace(0.0, LX, BINS + 1)
    taken, given = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        taken.append(heat[floor & (cx > lo) & (cx < hi)].sum())
        given.append(q_integral(lo, hi) * (a["TIME"] - t0))
    return np.array(taken), np.array(given)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prefix")
    args = ap.parse_args()
    files = vtu.series(os.path.dirname(args.prefix) or ".", os.path.basename(args.prefix) + "_[0-9]*")
    t0, e0 = energy(files[0])
    worst = 0.0
    for f in files[1:]:
        t, e = energy(f)
        expected = Q_TOTAL * (t - t0)
        worst = max(worst, abs((e - e0) - expected) / expected)
        print(f"t = {1e3 * t:6.2f} ms: energy gained {e - e0:.12g} J/m, q A t {expected:.12g} J/m")
    print(f"largest relative difference {worst:.2e}")
    taken, given = local_heat(files[1], t0)
    ratio = taken / given
    print("heat of each bin's floor cells / (integral of q over the bin) t:", " ".join(f"{r:.3f}" for r in ratio))
    print(f"peak bins / trough bins: {taken[[0, -1]].mean() / taken[[3, 4]].mean():.2f} "
          f"(q: {given[[0, -1]].mean() / given[[3, 4]].mean():.2f}, a uniform q: 1)")
    sys.exit(0 if worst < 1e-10 and np.all(np.abs(ratio - 1.0) < 0.02) else 1)


if __name__ == "__main__":
    main()
