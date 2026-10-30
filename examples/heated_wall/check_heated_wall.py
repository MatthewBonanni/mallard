#!/usr/bin/env python3
"""The heated-wall example (examples/heated_wall): the energy in the closed box
against the heat that entered through the floor, q A t.

    check_heated_wall.py PREFIX      (e.g. solut/box)

Prints the energy gained at each snapshot against q A t and the largest
relative difference; exits 1 if it exceeds 1e-10.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools"))
from mallard import vtu  # noqa: E402

Q_TOTAL = 2.0e4 * 0.1   # W per metre of depth: the integral of q over the 0.1 m floor (the cosine integrates to 0)


def energy(path):
    pts, conn, offs, _, arrays = vtu.read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    quads = np.array([conn[s:e] for s, e in zip(starts, offs)])
    x, y = pts[quads][..., 0], pts[quads][..., 1]
    area = 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
    return arrays["TIME"], np.sum(arrays["RHOE"] * area)


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
    sys.exit(0 if worst < 1e-10 else 1)


if __name__ == "__main__":
    main()
