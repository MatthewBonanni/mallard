"""Read Mallard's output for the tutorial scripts: the snapshots of a series
PREFIX (all in PREFIX.vtkhdf, or PREFIX.pvd in builds without HDF5) and their
cells."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from mallard.vtu import read_index, read_vtu_cells  # noqa: E402


def series(prefix):
    """(time, path) of every snapshot of the series PREFIX."""
    try:
        out = read_index(prefix)
    except FileNotFoundError:
        raise SystemExit(f"no {prefix}.vtkhdf (or .pvd)")
    if not out:
        raise SystemExit(f"{prefix}: no snapshots")
    return out


def last(prefix):
    return series(prefix)[-1][1]


def read(path):
    """Points, connectivity, cell offsets (end of each cell) and cell arrays of a snapshot."""
    pts, conn, offs, _, arrays = read_vtu_cells(path)
    return pts, conn, offs, arrays


def cells(pts, conn, offs):
    """The node coordinates of each cell (a list of arrays) and the cell centroids (vertex averages)."""
    starts = np.concatenate([[0], offs[:-1]])
    polys = [pts[conn[s:e]] for s, e in zip(starts, offs)]
    return polys, np.array([p.mean(0) for p in polys])
