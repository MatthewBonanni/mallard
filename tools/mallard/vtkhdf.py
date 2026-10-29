"""Reader for Mallard's VTKHDF output (format = "vtkhdf", the default; needs h5py).

A snapshot is PREFIX_NNNNNN.vtkhdf (indexed by PREFIX.vtkhdf.series), or with
series = true, one step of PREFIX.vtkhdf, which holds them all. Volume output
is an UnstructuredGrid, boundary zones and probes a PolyData.
"""
import json
import os

import h5py
import numpy as np

# Newest format version (the root attribute "version") this reader knows
VTKHDF_FORMAT_VERSION = 1


def _check_version(path, f):
    version = int(f.attrs.get("version", 1))
    if version > VTKHDF_FORMAT_VERSION:
        raise ValueError(f"{path} is a Mallard VTKHDF file of format version {version}, written by a newer "
                         f"Mallard; this script reads versions up to {VTKHDF_FORMAT_VERSION}: update tools/")


def times(path):
    """The times of the snapshots in a file: one for a snapshot file, all of them for a series."""
    with h5py.File(path, "r") as f:
        g = f["VTKHDF"]
        if "Steps" in g:
            return [float(t) for t in g["Steps/Values"][...]]
        return [float(g["FieldData/TIME"][0])]


def read_cells(path, step=-1):
    """A snapshot like mallard.vtu.read_vtu_cells: points (n, 3),
    connectivity, offsets (end of each cell), VTK cell types and the cell
    arrays (vectors as (n_cells, 3)) plus "TIME". In a series, step picks the
    snapshot (default the last). A PolyData gives its polygons, or its lines
    in 2D, as cells of VTK type 7 (polygon) or 3 (line)."""
    with h5py.File(path, "r") as f:
        _check_version(path, f)
        g = f["VTKHDF"]
        kind = g.attrs["Type"]
        kind = kind.decode() if isinstance(kind, bytes) else str(kind)
        pts = g["Points"][...].astype(float)
        if kind == "UnstructuredGrid":
            cells = g
            types = g["Types"][...]
        else:
            name = "Polygons" if int(g["Polygons/NumberOfCells"][0]) > 0 else "Lines"
            cells = g[name]
            types = None
        conn = cells["Connectivity"][...]
        starts = cells["Offsets"][...]
        n = starts.size - 1
        if types is None:
            types = np.full(n, 7 if name == "Polygons" else 3, dtype=np.uint8)
        out = {}
        if "Steps" in g:
            k = range(int(g["Steps"].attrs["NSteps"]))[step]
            out["TIME"] = float(g["Steps/Values"][k])
            for name, data in g["CellData"].items():
                at = int(g["Steps/CellDataOffsets"][name][k])
                out[name] = np.asarray(data[at:at + n], dtype=float)
        else:
            out["TIME"] = float(g["FieldData/TIME"][0])
            for name, data in g["CellData"].items():
                out[name] = np.asarray(data[...], dtype=float)
    return pts, conn, starts[1:], types, out


def read_points(path, step=-1):
    """A probe file (a PolyData of points or curves): points (n, 3) and the
    point arrays of a snapshot (default the last), plus "TIME"."""
    with h5py.File(path, "r") as f:
        g = f["VTKHDF"]
        pts = g["Points"][...].astype(float)
        n = pts.shape[0]
        k = range(int(g["Steps"].attrs["NSteps"]))[step]
        out = {"TIME": float(g["Steps/Values"][k])}
        for name, data in g["PointData"].items():
            at = int(g["Steps/PointDataOffsets"][name][k])
            out[name] = np.asarray(data[at:at + n], dtype=float)
    return pts, out


def read_series(path):
    """(time, file) of every snapshot of an index PREFIX.vtkhdf.series."""
    with open(path) as f:
        index = json.load(f)
    base = os.path.dirname(path)
    return [(float(e["time"]), os.path.join(base, e["name"])) for e in index["files"]]


def provenance(path):
    """The provenance block of a VTKHDF file, as a dict; None if it has none."""
    with h5py.File(path, "r") as f:
        value = f["/"].attrs.get("provenance")
    if value is None:
        return None
    value = value[0]
    return json.loads(value.decode() if isinstance(value, bytes) else value)
