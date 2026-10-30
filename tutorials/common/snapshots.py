"""Read Mallard's output for the tutorial scripts: the snapshots of a series
PREFIX (indexed by PREFIX.vtkhdf.series, or PREFIX.pvd in builds without
HDF5; a boundary zone's snapshots are all in PREFIX.vtkhdf) and their cells."""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from mallard.vtu import read_vtu_cells  # noqa: E402


def series(prefix):
    """(time, path) of every snapshot of the series PREFIX."""
    if os.path.exists(prefix + ".vtkhdf.series"):
        from mallard.vtkhdf import read_series
        out = read_series(prefix + ".vtkhdf.series")
    elif os.path.exists(prefix + ".vtkhdf"):
        # One file of every snapshot: read() reads its last
        from mallard.vtkhdf import times
        out = [(t, prefix + ".vtkhdf") for t in times(prefix + ".vtkhdf")]
    elif os.path.exists(prefix + ".pvd"):
        text = open(prefix + ".pvd").read()
        out = [(float(t), os.path.join(os.path.dirname(prefix), f))
               for t, f in re.findall(r'timestep="([^"]+)"[^>]*file="([^"]+)"', text)]
    else:
        raise SystemExit(f"no {prefix}.vtkhdf.series (or .pvd)")
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
