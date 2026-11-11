"""Every file Mallard writes or reads, through its C++ reader (src/io/output_reader.h)."""
import builtins
import ctypes
import json
import os
from dataclasses import dataclass

import numpy as np

from . import _lib


@dataclass(frozen=True)
class Field:
    name: str
    components: int
    units: str        # UDUNITS ("kg m-3", "1"); "" where the file records none
    long_name: str
    location: str     # "cell", "point" or "column"


@dataclass(frozen=True)
class Snapshot:
    t: float
    step: int | None  # None where the file records none (written before Mallard 0.10)
    file: str         # the file holding it: the file itself, or a series member
    index: int        # its index in that file
    run_id: str | None
    block: dict | None  # its snapshot block: reason, written_at, run, fields' units and stored precision, ...


@dataclass(frozen=True)
class Geometry:
    points: np.ndarray        # (n_points, 3); z is zero in 2D
    offsets: np.ndarray       # (n_cells + 1,): cell i is connectivity[offsets[i]:offsets[i + 1]]
    connectivity: np.ndarray
    types: np.ndarray         # VTK cell types (5 triangle, 9 quadrilateral, 12 hexahedron, ...)

    @property
    def n_cells(self):
        return self.offsets.size - 1


def _array(lib, ptr, n, dtype):
    return np.ctypeslib.as_array(ptr, shape=(n,)).astype(dtype, copy=True) if n else np.zeros(0, dtype)


class File:
    """What a file holds (mallard info --format json, as attributes) and its values.

    kind: "mesh", "snapshot", "restart", "probe" or "table"; snapshots in time order; fields by name.
    """

    def __init__(self, lib, handle):
        self._lib = lib
        self._handle = handle
        self.contents = json.loads(lib.lib.mallard_contents(handle).decode())
        c = self.contents
        self.path = c["path"]
        self.kind = c["kind"]
        self.format = c["format"]
        self.version = c["version"]
        self.dim = c["dim"]
        self.precision = c["precision"]
        self.compressed = c["compressed"]
        self.global_order = c["global_order"]
        self.mesh = c["mesh"]
        self.run = c["run"]
        self.notes = c["notes"]
        self.species = c["species"]
        self.fields = {f["name"]: Field(f["name"], f["components"], f["units"], f["long_name"], f["location"])
                       for f in c["fields"]}
        self.snapshots = [Snapshot(s["t"], s["step"], s["file"], s["index"], s["run_id"], s["snapshot"])
                          for s in c["snapshots"]]

    @property
    def times(self):
        return np.array([s.t for s in self.snapshots])

    def _snapshot(self, snapshot):
        n = len(self.snapshots)
        if not -n <= snapshot < n:
            raise IndexError(f"{self.path} has {n} snapshots, not {snapshot + 1 if snapshot >= 0 else snapshot}")
        return snapshot % n

    def _values(self, array):
        lib = self._lib
        try:
            n = ctypes.c_uint64()
            ptr = lib.lib.mallard_array_values(array, ctypes.byref(n))
            return _array(lib, ptr, n.value, np.float64)
        finally:
            lib.lib.mallard_array_free(array)

    def _shape(self, values, field, leading=()):
        k = self.fields[field].components if field in self.fields else 1
        return values.reshape(*leading, -1, k) if k > 1 else values.reshape(*leading, -1)

    def read(self, field, snapshot=-1):
        """A field's values at a snapshot (default the last), in the file's order: (n,) or (n, components)."""
        array = self._lib.check(self._lib.lib.mallard_read(self._handle, field.encode(), self._snapshot(snapshot)))
        return self._shape(self._values(array), field)

    def read_all(self, field):
        """A field at every snapshot: (n_snapshots, n) or (n_snapshots, n, components); a table's whole column."""
        array = self._lib.check(self._lib.lib.mallard_read_all(self._handle, field.encode()))
        values = self._values(array)
        if self.kind == "table":
            return values
        n = len(self.snapshots)
        if n == 0 or values.size % n:
            raise ValueError(f"{self.path}: the snapshots of {field} differ in size; read them one by one with read()")
        return self._shape(values, field, (n,))

    def geometry(self, snapshot=-1):
        """The points and cells the values are on (a probe HDF5 file: its points only)."""
        index = 0 if self.kind == "mesh" else self._snapshot(snapshot)
        lib = self._lib
        array = lib.check(lib.lib.mallard_read_geometry(self._handle, index))
        try:
            dp = ctypes.POINTER(ctypes.c_double)()
            op, cp = ctypes.POINTER(ctypes.c_uint64)(), ctypes.POINTER(ctypes.c_uint64)()
            tp = ctypes.POINTER(ctypes.c_uint8)()
            n = [ctypes.c_uint64() for _ in range(4)]
            lib.lib.mallard_array_geometry(array, ctypes.byref(dp), ctypes.byref(n[0]), ctypes.byref(op),
                                           ctypes.byref(n[1]), ctypes.byref(cp), ctypes.byref(n[2]), ctypes.byref(tp),
                                           ctypes.byref(n[3]))
            return Geometry(_array(lib, dp, n[0].value, np.float64).reshape(-1, 3),
                            _array(lib, op, n[1].value, np.int64), _array(lib, cp, n[2].value, np.int64),
                            _array(lib, tp, n[3].value, np.uint8))
        finally:
            lib.lib.mallard_array_free(array)

    def __getitem__(self, field):
        return self.read(field)

    def close(self):
        if self._handle:
            self._lib.lib.mallard_close(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def __repr__(self):
        return (f"<mallard.File {self.path}: {self.kind}, {self.format}, {len(self.snapshots)} snapshots, "
                f"fields {', '.join(self.fields)}>")


def open(path, dim=None):
    """Opens any file Mallard writes or reads: snapshots (.vtkhdf, .vtu, .pvtu, .h5 and the series indexes
    .vtkhdf.series, .pvd, .xmf), restart files (compressed too), probe files (.csv, .h5, .vtkhdf, .vtp), CSV
    tables (forces, integrals, spectra) and mesh files. Output files are read whatever the build's dimension;
    mesh files by the library of their dimension (dim, else each one found in turn)."""
    path = os.fspath(path)
    libs = _lib.libraries()
    if dim is not None:
        libs = [lib for lib in libs if lib.dim == dim] or libs
    error = None
    for lib in libs:
        handle = lib.lib.mallard_open(path.encode())
        if handle:
            return File(lib, handle)
        error = error or lib.lib.mallard_error().decode()
    raise RuntimeError(error)


def run_files(directory):
    """The output files of a run directory: series indexes, snapshots no index lists, restart, probe and CSV files."""
    lib = _lib.libraries()[0]
    array = lib.check(lib.lib.mallard_run_files(os.fspath(directory).encode()))
    try:
        return json.loads(lib.lib.mallard_array_text(array).decode())
    finally:
        lib.lib.mallard_array_free(array)


def run_record(path):
    """The records of a run record ([output] run_log: a run directory's mallard.jsonl, or the file), in order:
    start, progress, output, end, ... as dicts."""
    path = os.fspath(path)
    with builtins.open(os.path.join(path, "mallard.jsonl") if os.path.isdir(path) else path) as f:
        return [json.loads(line) for line in f if line.strip()]
