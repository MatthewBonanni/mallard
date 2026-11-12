"""Reader for Mallard's VTKHDF output (format = "vtkhdf", the default; needs h5py).

A snapshot is one step of the series file PREFIX.vtkhdf, which holds them all
(series = true, the default), or with series = false PREFIX_NNNNNN.vtkhdf
(indexed by PREFIX.vtkhdf.series). Volume output is an UnstructuredGrid,
boundary zones and probes a PolyData. The snapshots of a series file are
SeriesSnapshot paths, which every reader of mallard.vtu and mallard.vtkhdf
reads as that snapshot.
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


class SeriesSnapshot(str):
    """One snapshot of a series file: the file's path, with position its place in the file."""

    def __new__(cls, path, position):
        out = super().__new__(cls, path)
        out.position = position
        return out


def position_of(path, step=None):
    """The place in a series file of the snapshot path names: step if given,
    else a SeriesSnapshot's position, else the last."""
    return step if step is not None else getattr(path, "position", -1)


def is_series_file(path):
    """Whether a VTKHDF file holds a series (a Steps group), not one snapshot."""
    with h5py.File(path, "r") as f:
        return "Steps" in f["VTKHDF"]


def snapshots(path):
    """The snapshots of a VTKHDF file: a SeriesSnapshot per step of a series file, else the file."""
    with h5py.File(path, "r") as f:
        g = f["VTKHDF"]
        if "Steps" not in g:
            return [path]
        return [SeriesSnapshot(path, k) for k in range(int(g["Steps"].attrs["NSteps"]))]


def times(path):
    """The times of the snapshots in a file: one for a snapshot file, all of them for a series."""
    with h5py.File(path, "r") as f:
        g = f["VTKHDF"]
        if "Steps" in g:
            return [float(t) for t in g["Steps/Values"][...]]
        return [float(g["FieldData/TIME"][0])]


def read_cells(path, step=None):
    """A snapshot like mallard.vtu.read_vtu_cells: points (n, 3),
    connectivity, offsets (end of each cell), VTK cell types and the cell
    arrays (vectors as (n_cells, 3)) plus "TIME". In a series file, step (or
    a SeriesSnapshot's position) picks the snapshot (default the last). A PolyData gives its polygons, or its lines
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
            k = range(int(g["Steps"].attrs["NSteps"]))[position_of(path, step)]
            out["TIME"] = float(g["Steps/Values"][k])
            for name, data in g["CellData"].items():
                at = int(g["Steps/CellDataOffsets"][name][k])
                out[name] = np.asarray(data[at:at + n], dtype=float)
        else:
            out["TIME"] = float(g["FieldData/TIME"][0])
            for name, data in g["CellData"].items():
                out[name] = np.asarray(data[...], dtype=float)
    from .vtu import vectors
    return pts, conn, starts[1:], types, vectors(out)


def read_points(path, step=None):
    """A probe file (a PolyData of points or curves): points (n, 3) and the
    point arrays of a snapshot (default the last), plus "TIME"."""
    with h5py.File(path, "r") as f:
        g = f["VTKHDF"]
        pts = g["Points"][...].astype(float)
        n = pts.shape[0]
        k = range(int(g["Steps"].attrs["NSteps"]))[position_of(path, step)]
        out = {"TIME": float(g["Steps/Values"][k])}
        for name, data in g["PointData"].items():
            at = int(g["Steps/PointDataOffsets"][name][k])
            out[name] = np.asarray(data[at:at + n], dtype=float)
    return pts, out


def field_metadata(path):
    """{field: (units, long_name)} of the cell and point arrays of a VTKHDF file."""
    from .h5 import field_metadata as h5_metadata
    out = {}
    for name, value in h5_metadata(path).items():
        for group in ("VTKHDF/CellData/", "VTKHDF/PointData/"):
            if name.startswith(group):
                out[name[len(group):]] = value
    return out


def read_series(path):
    """(time, file) of every snapshot of an index PREFIX.vtkhdf.series or a series file PREFIX.vtkhdf."""
    return [(t, f) for t, _, f in read_series_steps(path)]


def read_series_steps(path):
    """(time, step, file) of every entry of a .vtkhdf.series index, or of every
    snapshot of a series file PREFIX.vtkhdf (a SeriesSnapshot); step is None
    where the file records none (before Mallard 0.10)."""
    if path.endswith(".vtkhdf"):
        from .h5 import snapshot_blocks
        blocks = snapshot_blocks(path)
        entries = snapshots(path)
        steps = [b.get("step") for b in blocks] if len(blocks) == len(entries) else [None] * len(entries)
        return list(zip(times(path), steps, entries))
    with open(path) as f:
        index = json.load(f)
    base = os.path.dirname(path)
    return [(float(e["time"]), e.get("step"), os.path.join(base, e["name"])) for e in index["files"]]


def provenance(path):
    """The run block of a VTKHDF file (named provenance before Mallard 0.10), as a dict; None if it has none."""
    from .h5 import run_block
    return run_block(path)

