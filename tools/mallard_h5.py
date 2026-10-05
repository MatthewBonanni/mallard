"""Reader for Mallard's HDF5 solution output (format = "hdf5"; needs h5py).

A series is PREFIX_mesh.h5 (written once) and one PREFIX_NNNNNN.h5 of cell
fields per snapshot, every array in global cell order, with the XDMF indexes
PREFIX_NNNNNN.xmf and PREFIX.xmf (the time series) for ParaView.
"""
import functools
import os
import re

import h5py
import numpy as np

# XDMF Mixed topology codes -> (VTK cell type, node count)
XDMF_CELLS = {4: (5, 3), 5: (9, 4), 6: (10, 4), 7: (14, 5), 8: (13, 6), 9: (12, 8)}


@functools.lru_cache(maxsize=4)
def _read_mesh(path, mtime):
    del mtime  # part of the cache key only
    with h5py.File(path, "r") as f:
        x = f["nodes/coordinates"][...].astype(float)
        topology = f["cells/topology"][...]
        starts = f["cells/offsets"][...]
    pts = np.zeros((x.shape[0], 3))
    pts[:, :x.shape[1]] = x
    codes = topology[starts[:-1]]
    vtk_types = np.zeros(codes.size, dtype=np.uint8)
    sizes = np.zeros(codes.size, dtype=np.int64)
    for code, (vtk, n) in XDMF_CELLS.items():
        vtk_types[codes == code] = vtk
        sizes[codes == code] = n
    keep = np.ones(topology.size, dtype=bool)
    keep[starts[:-1]] = False
    return pts, topology[keep], np.cumsum(sizes), vtk_types


def mesh_path(path):
    """The mesh file of a snapshot file."""
    with h5py.File(path, "r") as f:
        name = f.attrs["mesh"][0]
    return os.path.join(os.path.dirname(path), name.decode() if isinstance(name, bytes) else str(name))


def read_mesh(path):
    """Points (n, 3), connectivity, offsets (end of each cell) and VTK cell types of a mesh file."""
    return _read_mesh(os.path.abspath(path), os.path.getmtime(path))


def read_fields(path, names=None):
    """Cell fields (vectors as (n_cells, 3)) of a snapshot, plus "TIME"."""
    out = {}
    with h5py.File(path, "r") as f:
        group = f["fields"]
        for name in (group.keys() if names is None else names):
            out[name] = np.asarray(group[name][...], dtype=float)
        out["TIME"] = float(f.attrs["time"])
    return out


def read_h5_cells(path):
    """A snapshot (.h5, or its .xmf) like mallard_vtu.read_vtu_cells: points
    (n, 3), connectivity, offsets, VTK cell types, and the cell arrays plus "TIME"."""
    if path.endswith(".xmf"):
        path = path[:-4] + ".h5"
    pts, conn, offs, types = read_mesh(mesh_path(path))
    return pts, conn, offs, types, read_fields(path)


def read_series(path):
    """(time, snapshot file) of every snapshot of a series index PREFIX.xmf."""
    text = open(path).read()
    entries = re.findall(r'<Grid Name="([^"]+)" GridType="Uniform">\s*<Time Value="([^"]+)"/>', text)
    base = os.path.dirname(path)
    return [(float(t), os.path.join(base, name + ".h5")) for name, t in entries]
