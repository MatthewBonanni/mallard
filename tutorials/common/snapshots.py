"""Read Mallard's output for the tutorial scripts through the mallard package: the
series PREFIX (PREFIX.vtkhdf, or PREFIX.pvd in builds without HDF5) and the cells
of its snapshots."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import mallard  # noqa: E402


def series(prefix):
    """The series PREFIX, opened (a mallard.File)."""
    for ext in (".vtkhdf", ".pvd"):
        if os.path.exists(prefix + ext):
            f = mallard.open(prefix + ext)
            if not f.snapshots:
                raise SystemExit(f"{prefix}{ext}: no snapshots")
            return f
    raise SystemExit(f"no {prefix}.vtkhdf (or .pvd)")


def read(f, snapshot=-1):
    """The geometry (a mallard.Geometry) and {field: values} of a snapshot of an open file (default the last)."""
    return f.geometry(snapshot), {name: f.read(name, snapshot) for name in f.fields}


def cells(g):
    """The point coordinates of each cell (a list of arrays) and the cell centroids (vertex averages)."""
    return [g.points[g.connectivity[a:b]] for a, b in zip(g.offsets[:-1], g.offsets[1:])], g.vertex_averages()
