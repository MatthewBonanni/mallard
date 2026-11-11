# Python API

Every public function and class of the Python package `mallard`
(`tools/mallard`), from their docstrings. How to install it and what it is for:
[Python](python.md).

```python
import mallard

f = mallard.open("solut/flow.vtkhdf.series")   # mallard.File
f.read("P"), f.read_all("P"), f.geometry()
mallard.run_record("."), mallard.run_files(".")
mallard.info(path), mallard.diff(a, b), mallard.post.extract(files, ...), mallard.post.convergence(runs, ...)
```

`mallard.vtu`, `mallard.vtkhdf`, `mallard.h5` and `mallard.tables` are the
readers the scripts in `tools/` use ([Tools](../../tools/README.md)); they need
h5py for HDF5 files, and are not part of this reference.

<!-- BEGIN GENERATED reference -->

## Reading files

### `mallard.open(path, dim=None)`

Opens any file Mallard writes or reads and returns a `File`: snapshots (`.vtkhdf`, `.vtu`, `.pvtu`,
`.h5` and the series indexes `.vtkhdf.series`, `.pvd`, `.xmf`), restart files (compressed too), probe
files (`.csv`, `.h5`, `.vtkhdf`, `.pvd` of `.vtp`), CSV tables (forces, integrals, probe reductions,
spectra) and mesh files (`.msh`, `.cas`, `.cgns`, Mallard HDF5 meshes, OpenFOAM cases). Output files are
read whatever the dimension of the library; a mesh file is read by the library of its dimension: `dim`, or
each library found in turn. Raises `RuntimeError` with the reader's message (a missing file, a file
Mallard did not write, a format version newer than the library reads).

### `mallard.File`

An open file: what it holds, and its values. `mallard.open` returns one.

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

#### `file.times`

The time of each snapshot, in s, as an array.

#### `file.read(field, snapshot=-1)`

A field's values at a snapshot, as stored: `(n,)`, or `(n, components)` for vectors, in the order
of `geometry()` (cells, points, or a table's rows of that snapshot). `snapshot` is a position in
`snapshots`, negative from the end (default: the last). Values come as float64 whatever precision
wrote them; restart files hold the conservative variables (`RHO`, `RHOU_X`, ..., `RHOE`, `RHOY_<s>`).

#### `file.read_all(field)`

A field at every snapshot: `(n_snapshots, n)` or `(n_snapshots, n, components)`; for a table, its
whole column `(n_rows,)`. One read for tables and probe HDF5 files.

#### `file.geometry(snapshot=-1)`

The `Geometry` of a snapshot (default the last): the cells of snapshots and mesh files, the points
and cells of probe files. Restart files and tables hold none: the error says where to look.

#### `file[field]`

`f["RHO"]` is `f.read("RHO")`: the last snapshot.

#### `file.close()`

Releases the file; `del` and the end of a `with` block do too.

### `mallard.Field`

A field of a file, as the file records it.

- `name`: `"RHO"`, `"U"`, `"MEAN_P"`, a table's column (`"F_x"`).
- `components`: 1, or 3 for vectors (z is zero in 2D).
- `units`: [UDUNITS](https://docs.unidata.ucar.edu/udunits/current/) units in SI base units (`"kg m-3"`,
  `"1"` for dimensionless); `""` where the file records none.
- `long_name`: `"density"`, `"velocity"`, ...
- `location`: `"cell"`, `"point"` or `"column"` (tables).

### `mallard.Snapshot`

One time of a file: a snapshot of a series, a sample of a probe file, a block of a table's rows.

- `t`: the simulated time, in s.
- `step`: the step, or `None` where the file records none (files written before Mallard 0.10).
- `file`: the file holding it: the file itself, or a member of a series index.
- `index`: its index in that file (series files, probe files, tables).
- `run_id`: the run that wrote it, or `None` if the file does not say.
- `block`: its snapshot block, as a dict (`reason`, `written_at`, `fields` with their stored precision,
  `statistics` of averaged fields, ...), or `None` (files written before Mallard 0.10).

### `mallard.Geometry`

The points and cells a file's values are on, in the file's order (the order of `File.read`).

- `points`: `(n_points, 3)` coordinates, in m (z is zero in 2D).
- `offsets`: `(n_cells + 1,)`: cell `i` is the points `connectivity[offsets[i]:offsets[i + 1]]`.
- `connectivity`: point indices of every cell, one cell after the other.
- `types`: the [VTK cell type](https://vtk.org/doc/nightly/html/vtkCellType_8h.html) of each cell
  (5 triangle, 9 quadrilateral, 10 tetrahedron, 12 hexahedron, 13 prism, 14 pyramid, 3 line, 4 polyline,
  7 polygon, 1 vertex); probe HDF5 files hold sample points and no cells.

#### `geometry.n_cells`

The number of cells.

#### `geometry.vertex_averages()`

The average of each cell's points, `(n_cells, 3)`: the centroid of a triangle, tetrahedron or
parallelogram, close to it for other cells; for plotting cell values at points.

### `mallard.run_files(directory)`

The output files of a run directory, sorted: its series indexes and the snapshots no index lists,
restart files, probe and CSV files, one level of subdirectories (`solut/`) included; members of an index
are listed through it.

### `mallard.run_record(path)`

The records of a run record (`[output] run_log`): a run directory's `mallard.jsonl`, or the file
itself. A list of dicts in the order written, each with its `type`: `start` (the run block, placement,
setup times), `progress`, `output` (a snapshot written: its files with their sizes and SHA-256), `event`,
`divergence`, `warning`, `error` and `end` (how the run ended), as the
[run record](output.md#run-record-mallardjsonl) describes them.

## Commands

### `mallard.info(path, **options)`

`mallard info PATH`: what a file holds (as `mallard.open(path).contents`, with each field's `min` and
`max` at the last snapshot), or a run directory's files and its start and end.

### `mallard.diff(a, b, **options)`

`mallard diff A B`: norms of the differences between two files, field by field, with the largest ones
and where they are; `remap=True` compares files on different meshes. Options: `fields`, `tol`, `step`, `time`,
`snapshot`, `remap`, `mesh_a`, `mesh_b` (`mallard diff --help`). Differences beyond tolerance are a
result, not an error: the report's `result` and `exit_status` give the verdict.

### `mallard.command(*args, nprocs=None, mpirun='mpirun', threads=None, cwd=None, verdicts=(), **options)`

Runs `mallard ARGS... OPTIONS --format json` and returns its JSON output, parsed.

Keyword arguments are the command's options, as `mallard COMMAND --help` lists them: `fields=["RHO", "P"]`
is `--fields RHO,P`, `latest=True` is `--latest`, `mesh_a="a.msh"` is `--mesh-a a.msh`, `input="case.toml"`
is `-i case.toml`; `None` and `False` leave an option out. `nprocs` runs the command on that many MPI ranks
with `mpirun -n` (`mpirun` names another launcher), `threads` sets `--kokkos-num-threads`, and `cwd` is the
directory it runs in (relative paths in a file's run block, such as its mesh file, resolve from there).
The executable is `MALLARD_EXECUTABLE`, else `mallard` on `PATH`. Raises `CommandError` when the command
exits with an error; an exit status in `verdicts` is a result instead (`mallard diff`'s 1: a difference
beyond tolerance), returned with the JSON the command printed.

### `mallard.CommandError`

A command that exited with an error (exit status 1: an error; 2: a usage error, such as an unknown
option).

- `returncode`: the exit status.
- `argv`: the command line run.
- `stdout`, `stderr`: what it printed.
- `result`: the JSON it printed, parsed, or `None`.

## `mallard.post`

### `mallard.post.extract(files, **options)`

`mallard post extract FILE...`: probes, region integrals and snapshots of existing output, as the run
would have written them. `files` is a path or a list of them (restart files, snapshots of the whole mesh,
series). Options: `input`, `probe`, `region`, `name`, `fields`, `reductions`, `sampling`, `snapshot`,
`steps`, `times`, `every`, `latest`, `output` (`mallard post extract --help`). Returns the report: the
files written (under `output`, default `extract/`, for `mallard.open`), their times and fields, and the
reductions of the last snapshot.

### `mallard.post.convergence(runs, **options)`

`mallard post convergence RUN_DIR...`: for each quantity of the runs' forces, integrals and probe tables,
the observed order of accuracy, the value extrapolated to zero cell size and the grid convergence index.
Options: `quantity`, `time`, `average`, `order`, `safety_factor`, `sizes`, `field`, `exact`
(`mallard post convergence --help`). A quantity that diverges is a result, not an error (exit status 1
with a report): each quantity's verdict says so.

<!-- END GENERATED reference -->
