"""Read Mallard's VTU output for the tutorial scripts: the snapshots of a
series (the .pvd file) and their cells, whether one rank wrote a .vtu file
or several wrote the pieces of a .pvtu."""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from mallard.vtu import read_vtu_cells  # noqa: E402


def series(pvd):
    """(time, path) of every snapshot listed in a .pvd file."""
    text = open(pvd).read()
    out = [(float(t), os.path.join(os.path.dirname(pvd), f))
           for t, f in re.findall(r'timestep="([^"]+)"[^>]*file="([^"]+)"', text)]
    if not out:
        raise SystemExit(f"{pvd}: no snapshots")
    return out


def last(pvd):
    return series(pvd)[-1][1]


def read(path):
    """Points, connectivity, cell offsets (end of each cell) and cell arrays of a .vtu or .pvtu file."""
    if not path.endswith(".pvtu"):
        pts, conn, offs, _, arrays = read_vtu_cells(path)
        return pts, conn, offs, arrays
    pieces = re.findall(r'Source="([^"]+)"', open(path).read())
    parts = [read_vtu_cells(os.path.join(os.path.dirname(path), p)) for p in pieces]
    pts, conn, offs, arrays = [], [], [], {}
    n_pts = n_conn = 0
    for p, c, o, _, a in parts:
        pts.append(p)
        conn.append(c + n_pts)
        offs.append(o + n_conn)
        n_pts += len(p)
        n_conn += len(c)
        for k, v in a.items():
            if np.ndim(v) > 0:
                arrays.setdefault(k, []).append(v)
            else:
                arrays[k] = v
    arrays = {k: (np.concatenate(v) if isinstance(v, list) else v) for k, v in arrays.items()}
    return np.vstack(pts), np.concatenate(conn), np.concatenate(offs), arrays


def cells(pts, conn, offs):
    """The node coordinates of each cell (a list of arrays) and the cell centroids (vertex averages)."""
    starts = np.concatenate([[0], offs[:-1]])
    polys = [pts[conn[s:e]] for s, e in zip(starts, offs)]
    return polys, np.array([p.mean(0) for p in polys])
