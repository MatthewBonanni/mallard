# Reading output

How Mallard reads back the files it writes, for `mallard info` and the
post-processing commands (`post extract`, `post derive`, `post average`,
`diff`, `post render`) and the Python package built on them. The user-facing
page is [Command line](../guide/cli.md#mallard-info).

## One reader, in C++

`src/io/output_reader.h` is the only reader of Mallard's file formats.
`output::open(path)` reads what a file holds without its field values:

- its kind (mesh, snapshot, restart, probe, table), format and format version;
- the mesh it holds or refers to: cell and point counts, cell types, bounds,
  boundary and cell zones;
- its snapshots (time, step where recorded, and the file and index holding
  each), so a series index (`.vtkhdf.series`, `.pvd`, `.xmf`), a series file
  (`series = true`) and a probe file read the same way;
- its fields, with components, units and long names as the file records them;
- its provenance block.

`output::read(file, field, snapshot)` loads a field's values as doubles,
whatever precision wrote them, and `read_ranges` their extrema.

Formats: VTKHDF (unstructured grids and the poly data of probes and region
snapshots, one file per snapshot or a series file), XDMF output (HDF5 files) and its
mesh file and XDMF indexes, VTU, PVTU, PVD, VTP, restart files of every
version (compressed too), probe HDF5, CSV, and the mesh files a run reads
(Gmsh, Fluent, CGNS, OpenFOAM, Mallard HDF5 meshes).

The reader lives next to the writers: it takes the format versions from
`file_format.h`, the units table's encodings from `field_units.h`, and
restart files are read by the same header parser as `read_restart`
(`read_restart_header`), so a format change touches one place in C++.
Output files are read whatever the build's dimension and precision; only
mesh files go through this build's mesh readers, so the front end runs
`mallard info mesh.msh` on the solver of the mesh's dimension.

## Why C++, and Python on top

The post-processing commands evaluate fields on the case's own mesh with the
solver's reconstruction, gradients and gas model, in parallel with MPI and on
GPUs: they run in the solver, so they read in C++. A second reader in Python
would duplicate every format and drift from it.

The Python package ([#517](https://github.com/MatthewBonanni/mallard/issues/517))
binds this reader and the commands built on it; it does not reimplement them.
Until it does, `tools/mallard/*.py` keep reading the formats with h5py and
numpy, and `tools/dev/check_output_units.py` (run in CI) checks that
`mallard info --json` and the Python readers report the same fields, units
and times for every format a run writes.

## What later commands need from it

- **Cell order.** `Contents::global_order` says whether cell values come in
  the global cell order of the mesh the run used (restart, XDMF and VTKHDF
  output of the whole mesh). Such values map onto the case's mesh cell by
  cell, as `post derive` needs to evaluate fields with the solver's
  reconstruction. VTU pieces and region snapshots are in their own order.
- **The case.** Provenance version 2 records the resolved input
  (`input.resolved`, every key explicit) and the mesh's SHA-256: enough to
  set the case up again (`Solver::init`), check that its mesh is the file's,
  and fill its state from `read()` of the conservative variables (restart
  files) or `RHO`, `U`, `P` and the species (snapshots). Version 1 blocks hold
  the input as written (`input.text`), which a run of the same release
  resolves the same way.
- **Partial reads.** `read()` loads a whole field. Reading only a rank's
  cells (as `read_restart` does for its `cells`) is added with the first
  MPI-parallel command, behind the same interface.

## `mallard info`

`mallard info FILE` prints `open()`'s contents and the ranges of the last
snapshot; `--json` the same as one object; `--input` the input of the
file's provenance. On a mesh file it also reports mesh quality with the
function and report of `mallard check` (`--quality PREFIX` writes the
fields). On a run directory it lists the run's output files and the run's
start and end from `mallard.jsonl`.
