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
    """A field of a file, as the file records it.

    - `name`: `"RHO"`, `"U"`, `"MEAN_P"`, a table's column (`"Fx_pressure"`).
    - `components`: 1, or 3 for vectors (z is zero in 2D).
    - `units`: [UDUNITS](https://docs.unidata.ucar.edu/udunits/current/) units in SI base units (`"kg m-3"`,
      `"1"` for dimensionless); `""` where the file records none.
    - `long_name`: `"density"`, `"velocity"`, ...
    - `location`: `"cell"`, `"point"` or `"column"` (tables).
    """
    name: str
    components: int
    units: str
    long_name: str
    location: str


@dataclass(frozen=True)
class Snapshot:
    """One time of a file: a snapshot of a series, a sample of a probe file, a block of a table's rows.

    - `t`: the simulated time, in s.
    - `step`: the step, or `None` where the file records none (files written before Mallard 0.10).
    - `file`: the file holding it: the file itself, or a member of a series index.
    - `index`: its index in that file (series files, probe files, tables).
    - `run_id`: the run that wrote it, or `None` if the file does not say.
    - `block`: its snapshot block, as a dict (`reason`, `written_at`, `fields` with their stored precision,
      `statistics` of averaged fields, ...), or `None` (files written before Mallard 0.10).
    """
    t: float
    step: int | None
    file: str
    index: int
    run_id: str | None
    block: dict | None


@dataclass(frozen=True)
class Geometry:
    """The points and cells a file's values are on, in the file's order (the order of `File.read`).

    - `points`: `(n_points, 3)` coordinates, in m (z is zero in 2D).
    - `offsets`: `(n_cells + 1,)`: cell `i` is the points `connectivity[offsets[i]:offsets[i + 1]]`.
    - `connectivity`: point indices of every cell, one cell after the other.
    - `types`: the [VTK cell type](https://vtk.org/doc/nightly/html/vtkCellType_8h.html) of each cell
      (5 triangle, 9 quadrilateral, 10 tetrahedron, 12 hexahedron, 13 prism, 14 pyramid, 3 line, 4 polyline,
      7 polygon, 1 vertex); probe HDF5 files hold sample points and no cells.
    """
    points: np.ndarray
    offsets: np.ndarray
    connectivity: np.ndarray
    types: np.ndarray

    @property
    def n_cells(self):
        """The number of cells."""
        return self.offsets.size - 1

    def vertex_averages(self):
        """The average of each cell's points, `(n_cells, 3)`: the centroid of a triangle, tetrahedron or
        parallelogram, close to it for other cells; for plotting cell values at points."""
        counts = np.diff(self.offsets)
        cell = np.repeat(np.arange(self.n_cells), counts)
        sums = np.zeros((self.n_cells, 3))
        np.add.at(sums, cell, self.points[self.connectivity])
        return sums / counts[:, None]


def _array(ptr, n, dtype):
    return np.ctypeslib.as_array(ptr, shape=(n,)).astype(dtype, copy=True) if n else np.zeros(0, dtype)


class File:
    """An open file: what it holds, and its values. `mallard.open` returns one.

    Its attributes are those of `mallard info FILE --format json`:

    - `path`, `format` (`"VTKHDF"`, `"VTU"`, `"XDMF series index (XDMF output snapshots)"`, `"restart"`, ...),
      `version` (the Mallard format version, or `None`), `dim` (2, 3 or `None`), `precision` (`"double"`,
      `"single"` or `None`), `compressed`.
    - `kind`: `"mesh"`, `"snapshot"`, `"restart"`, `"probe"` or `"table"`.
    - `fields`: `{name: Field}`, in the file's order; `species`: the species of `Y_`, `X_` and `RHOY_` fields.
    - `snapshots`: `[Snapshot]`, in time order; `times`: their times as an array.
    - `mesh`: the mesh it holds or refers to: `cells`, `points`, `cell_types`, `bounds`, `boundary_zones`,
      `cell_zones`, and `file` (the mesh file of XDMF output).
    - `global_order`: whether cell values come in the global cell order of the run's mesh (restart files, and
      XDMF and VTKHDF output of the whole mesh), so that two such files compare cell by cell.
    - `run`: the run block (the code, build, resolved input, mesh, start and `run_id` of the run that wrote
      it), as a dict; `None` if it has none.
    - `notes`: what a reader of the file should know (no units, written before a format change, ...).
    - `contents`: all of the above as the JSON dict `mallard info` prints.

    A file is also a context manager: `with mallard.open(path) as f:` closes it at the end.
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
        """The time of each snapshot, in s, as an array."""
        return np.array([s.t for s in self.snapshots])

    def _snapshot(self, snapshot):
        n = len(self.snapshots)
        if not -n <= snapshot < n:
            raise IndexError(f"{self.path} has {n} snapshots; there is no snapshot {snapshot}")
        return snapshot % n

    def _values(self, array):
        lib = self._lib.lib
        try:
            n = ctypes.c_uint64()
            ptr = lib.mallard_array_values(array, ctypes.byref(n))
            return _array(ptr, n.value, np.float64)
        finally:
            lib.mallard_array_free(array)

    def _shape(self, values, field, leading=()):
        k = self.fields[field].components if field in self.fields else 1
        return values.reshape(*leading, -1, k) if k > 1 else values.reshape(*leading, -1)

    def read(self, field, snapshot=-1):
        """A field's values at a snapshot, as stored: `(n,)`, or `(n, components)` for vectors, in the order
        of `geometry()` (cells, points, or a table's rows of that snapshot). `snapshot` is a position in
        `snapshots`, negative from the end (default: the last). Values come as float64 whatever precision
        wrote them; restart files hold the conservative variables (`RHO`, `RHOU_X`, ..., `RHOE`, `RHOY_<s>`)."""
        array = self._lib.check(self._lib.lib.mallard_read(self._handle, field.encode(), self._snapshot(snapshot)))
        return self._shape(self._values(array), field)

    def read_all(self, field):
        """A field at every snapshot: `(n_snapshots, n)` or `(n_snapshots, n, components)`; for a table, its
        whole column `(n_rows,)`. One read for tables and probe HDF5 files."""
        array = self._lib.check(self._lib.lib.mallard_read_all(self._handle, field.encode()))
        values = self._values(array)
        if self.kind == "table":
            return values
        n = len(self.snapshots)
        if n == 0 or values.size % n:
            raise ValueError(f"{self.path}: the snapshots of {field} differ in size; read them one by one with read()")
        return self._shape(values, field, (n,))

    def geometry(self, snapshot=-1):
        """The `Geometry` of a snapshot (default the last): the cells of snapshots and mesh files, the points
        and cells of probe files. Restart files and tables hold none: the error says where to look."""
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
            return Geometry(_array(dp, n[0].value, np.float64).reshape(-1, 3), _array(op, n[1].value, np.int64),
                            _array(cp, n[2].value, np.int64), _array(tp, n[3].value, np.uint8))
        finally:
            lib.lib.mallard_array_free(array)

    def __getitem__(self, field):
        """`f["RHO"]` is `f.read("RHO")`: the last snapshot."""
        return self.read(field)

    def close(self):
        """Releases the file; `del` and the end of a `with` block do too."""
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
        return f"<mallard.File {self.path}: {self.kind}, {self.format}, {len(self.snapshots)} snapshots>"


def open(path, dim=None):
    """Opens any file Mallard writes or reads and returns a `File`: snapshots (`.vtkhdf`, `.vtu`, `.pvtu`,
    `.h5` and the series indexes `.vtkhdf.series`, `.pvd`, `.xmf`), restart files (compressed too), probe
    files (`.csv`, `.h5`, `.vtkhdf`, `.pvd` of `.vtp`), CSV tables (forces, integrals, probe reductions,
    spectra) and mesh files (`.msh`, `.cas`, `.cgns`, Mallard HDF5 meshes, OpenFOAM cases). Output files are
    read whatever the dimension of the library; a mesh file is read by the library of its dimension: `dim`, or
    each library found in turn. Raises `RuntimeError` with the reader's message (a missing file, a file
    Mallard did not write, a format version newer than the library reads)."""
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
    """The output files of a run directory, sorted: its series indexes and the snapshots no index lists,
    restart files, probe and CSV files, one level of subdirectories (`solut/`) included; members of an index
    are listed through it."""
    lib = _lib.libraries()[0]
    array = lib.check(lib.lib.mallard_run_files(os.fspath(directory).encode()))
    try:
        return json.loads(lib.lib.mallard_array_text(array).decode())
    finally:
        lib.lib.mallard_array_free(array)


def run_record(path):
    """The records of a run record (`[output] run_log`): a run directory's `mallard.jsonl`, or the file
    itself. A list of dicts in the order written, each with its `type`: `start` (the run block, placement,
    setup times), `progress`, `output` (a snapshot written: its files with their sizes and SHA-256), `event`,
    `divergence`, `warning`, `error` and `end` (how the run ended), as the
    [run record](output.md#run-record-mallardjsonl) describes them."""
    path = os.fspath(path)
    with builtins.open(os.path.join(path, "mallard.jsonl") if os.path.isdir(path) else path) as f:
        return [json.loads(line) for line in f if line.strip()]
