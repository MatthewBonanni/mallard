#!/usr/bin/env python3
"""The area-averaged outlet example (examples/average_outlet): the outlet's
area-averaged pressure against the 0.9 bar target, with and without average.

    check_average_outlet.py AVERAGE_PREFIX [LOCAL_PREFIX]   (e.g. solut/outlet solut/local)

Prints, for each snapshot from t = 2.5 ms on, the area average of the
pressure over the outlet's cells, its range and the outlet's Mach range;
exits 1 if the average outlet's mean is not within 0.1% of the target.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools"))
from mallard import vtu  # noqa: E402

TARGET, T_START, GAMMA, R = 0.9e5, 2.5e-3, 1.4, 287.0


def outlet(path):
    pts, conn, offs, _, a = vtu.read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    quads = np.array([conn[s:e] for s, e in zip(starts, offs)])
    x, y = pts[quads][..., 0], pts[quads][..., 1]
    area = 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
    last = x.mean(axis=1) > x.max() - 0.6 * (x.max(axis=1) - x.min(axis=1)).max()
    height = (y.max(axis=1) - y.min(axis=1))[last]   # face length of each outlet cell
    p, mach = a["P"][last], a["U_X"][last] / np.sqrt(GAMMA * R * a["T"][last])
    return a["TIME"], np.sum(p * height) / np.sum(height), p, mach


def report(prefix):
    files = vtu.series(os.path.dirname(prefix) or ".", os.path.basename(prefix) + "_[0-9]*")
    worst = 0.0
    for f in files:
        t, mean, p, mach = outlet(f)
        if t < T_START:
            continue
        worst = max(worst, abs(mean / TARGET - 1.0))
        print(f"{prefix} t = {1e3 * t:.2f} ms: mean p {mean:.1f} Pa ({100 * (mean / TARGET - 1):+.3f}%), "
              f"p {p.min():.0f}-{p.max():.0f} Pa, Mach {mach.min():.2f}-{mach.max():.2f}")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("average")
    ap.add_argument("local", nargs="?")
    args = ap.parse_args()
    worst = report(args.average)
    print(f"average: largest departure of the mean from the target {100 * worst:.3f}%")
    if args.local:
        print(f"local: largest departure {100 * report(args.local):.3f}%")
    sys.exit(0 if worst < 1e-3 else 1)


if __name__ == "__main__":
    main()
