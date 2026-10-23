#!/usr/bin/env python3
"""Compare a sector run with the full annulus (examples/rotational_sector).

    rotational_sector.py SECTOR.vtu ANNULUS.vtu --sectors N

Each cell of the annulus is matched with the sector cell it is a turned copy
of, and its density, momentum (turned by k 360/N degrees in copy k) and energy
compared: the largest differences over all cells and per copy are printed,
with the annulus's own copies compared with its first, the limit of the
comparison. 2D, or 3D sectors about the z axis.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools"))
from mallard.vtu import read_vtu_cells  # noqa: E402


def turn(x, y, angle):
    c, s = np.cos(angle), np.sin(angle)
    return c * x - s * y, s * x + c * y


def read_cells(path):
    pts, conn, offs, types, arrays = read_vtu_cells(path)
    starts = np.concatenate([[0], offs[:-1]])
    polys = [pts[conn[a:b]] for a, b in zip(starts, offs)]
    return polys, np.array([p.mean(axis=0) for p in polys]), arrays


def compare(sector_path, annulus_path, sectors):
    """Annulus cell polygons and fields, the copy of each annulus cell, and the
    largest difference of each annulus cell from its sector cell (turned) and
    from its own first copy (turned)."""
    alpha = 2 * np.pi / sectors
    _, cs, fs = read_cells(sector_path)
    pa, ca, fa = read_cells(annulus_path)
    h = 1e-6 * np.ptp(ca[:, 0])
    key = lambda x: (round(x[0] / h), round(x[1] / h), round(x[2] / h))
    index = {key(x): i for i, x in enumerate(cs)}
    theta = np.mod(np.arctan2(ca[:, 1], ca[:, 0]), 2 * np.pi)
    copies = np.floor(theta / alpha + 1e-9).astype(int) % sectors
    match = np.empty(len(ca), dtype=int)
    for i, (x, k) in enumerate(zip(ca, copies)):
        xs, ys = turn(x[0], x[1], -k * alpha)
        match[i] = index[key((xs, ys, x[2]))]

    def difference(f, rows):
        mx, my = turn(f["RHOU"][rows, 0], f["RHOU"][rows, 1], copies * alpha)
        return np.max(np.abs(np.stack([fa["RHO"] - f["RHO"][rows], fa["RHOE"] - f["RHOE"][rows],
                                       fa["RHOU"][:, 0] - mx, fa["RHOU"][:, 1] - my,
                                       fa["RHOU"][:, 2] - f["RHOU"][rows, 2]])), axis=0)

    first = {match[i]: i for i in range(len(ca)) if copies[i] == 0}
    return pa, fa, copies, difference(fs, match), difference(fa, np.array([first[m] for m in match]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sector")
    ap.add_argument("annulus")
    ap.add_argument("--sectors", type=int, required=True)
    args = ap.parse_args()
    _, _, copies, d, own = compare(args.sector, args.annulus, args.sectors)
    print(f"sector against the annulus: largest difference {d.max():.3e}")
    for k in range(args.sectors):
        print(f"  copy {k}: {d[copies == k].max():.3e}")
    print(f"annulus copies against its first: {own.max():.3e}")


if __name__ == "__main__":
    main()
