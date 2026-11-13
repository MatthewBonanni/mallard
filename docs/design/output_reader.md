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
  each, each with its snapshot block and run id), so a series index
  (`.vtkhdf.series`, `.pvd`, `.xmf`), a series file (`series = true`), a
  probe file and a table read the same way;
- its fields, with components, units and long names as the file records them;
- its run block (`Contents::run`).

`output::read(file, field, snapshot)` loads a field's values as doubles,
whatever precision wrote them, `read_all` a field at every snapshot (a table's
column and a probe file's samples in one read), `read_ranges` their extrema,
and `read_geometry` the points and cells the values are on (VTK cell types, in
the file's order; the mesh file of XDMF output, the cells of a mesh file).
`to_json` writes `Contents` as `mallard info --format json` prints it. A series
index reads its members' values and cells without opening them whole.

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

## Step, time and run: recorded, never parsed from names

Every snapshot's step, time and run come from what the file records: the
snapshot block (`snapshot_block.h`: one JSON object per snapshot, in the
format's own slot, [Snapshot metadata](../guide/output.md#snapshot-metadata)),
else the format's own slots (field data `TIME` and `STEP`, the `Steps` group,
the index's `timestep` and `step`, XDMF's `<Time>` and `step` information,
the restart header, CSV columns). The reader never parses a file name: an
XDMF grid's file comes from its data items, a series' members from its index,
the mesh file of XDMF output from its root attribute `mesh`, and a run
directory's members and mesh files from the indexes and XDMF files that name
them. The run's own consumers follow the same rule: `--restart latest` takes
the restart files of each writer from the run record's output index (and
falls back to listing `<prefix>_*.restart`, ordered by what they record, for
records before 0.10), and the writers resume a series from their own index.

Two blocks, because they answer different questions. The run block is the
same in every file of a run (code, build, input, mesh, where and when it
started, its `run_id` and what it restarted from); the snapshot block is
this snapshot's (step, time, reason, when it was written, by which run, its
fields' units and stored precision, the averaging window). Keeping them apart
keeps the run block identical across a run's files, so comparing it tells
whether two files come from the same run, and lets a series continued by
several runs say which run wrote each snapshot.

Older files fall back explicitly, and `Contents::notes` says how: a run block
under its earlier name (`provenance`, schema `mallard-provenance`) is read as
one, with a note that it records no run id; a file without snapshot blocks
takes step and time from the format's slots, and where it records no step
(VTU, VTKHDF and the indexes before 0.10) its snapshots are numbered by their
position from 0 and their step is -1. Rows of a CSV file belong to the run
whose `# mallard-run` line precedes them; samples of a probe HDF5 file to the
run its `mallard_runs` attribute names from their first row.

The schema leaves room for what later issues add to it: a field's sampling
(cell average or point value, #671), its averaging kind (Reynolds or Favre,
#672) and a probe's requested and actual location (#673) are further keys of
`fields` entries or of the block, which readers ignore until they know them.

## Why C++, and Python on top

The post-processing commands evaluate fields on the case's own mesh with the
solver's reconstruction, gradients and gas model, in parallel with MPI and on
GPUs: they run in the solver, so they read in C++. A second reader in Python
would duplicate every format and drift from it.

The Python package ([Python](../guide/python.md), `tools/mallard`; its layout
in [The Python package](python_package.md)) therefore reads through this reader. `Mallard_ENABLE_PYTHON` (on by default) builds
`libmallard2d`/`libmallard3d`, a shared library of `src/io/reader_c_api.h`: a
C interface over `open`, `read`, `read_all`, `read_geometry` and `run_files`,
with `Contents` passed as the JSON of `to_json`, values as arrays of doubles,
and errors as a message, so that no C++ exception or type crosses it. The
package loads it with ctypes and copies the arrays into numpy:

- **C and ctypes, not a binding library.** pybind11 or nanobind would give
  richer types, but add a build dependency and tie the library to one Python
  ABI. A C interface of a dozen functions is enough for files, whose contents
  are JSON and whose values are flat arrays, and any Python 3.10+ loads it.
- **Commands as processes.** `mallard.post.extract` and the other commands run
  the `mallard` executable with `--format json` instead of calling the solver
  inside the Python process: the front end picks the solver of the case's
  dimension, `mpirun` and GPUs work as on the command line, and a command
  cannot leave Kokkos or MPI state in the interpreter. The function and the
  command share every line of code and option, so they cannot disagree.
- **Position-independent code.** A shared library needs every library it links
  compiled with `-fPIC`; the option sets `CMAKE_POSITION_INDEPENDENT_CODE` for
  the whole build. It is on by default, so that every installer installs the
  package; `-DMallard_ENABLE_PYTHON=OFF` (Spack: `~python`) leaves it out.
- **One version.** `tools/mallard/_version.py` holds Mallard's version, and a
  build with the option checks it against `project(VERSION)`.

`tools/mallard`'s older modules (`vtu`, `vtkhdf`, `h5`, `tables`) read the
formats with h5py and numpy for the scripts in `tools/`, and
`tools/dev/check_output_units.py` (run in CI) checks that
`mallard info --format json` and those modules report the same fields, units
and times for every format a run writes.

## What later commands need from it

- **Cell order.** `Contents::global_order` says whether cell values come in
  the global cell order of the mesh the run used (restart, XDMF and VTKHDF
  output of the whole mesh). Such values map onto the case's mesh cell by
  cell, as `post derive` needs to evaluate fields with the solver's
  reconstruction. VTU pieces and region snapshots are in their own order.
- **The case.** Run blocks from version 2 on record the resolved input
  (`input.resolved`, every key explicit) and the mesh's SHA-256: enough to
  set the case up again (`Solver::init`), check that its mesh is the file's,
  and fill its state from `read()` of the conservative variables (restart
  files) or `RHO`, `U`, `P` and the species (snapshots). Version 1 blocks hold
  the input as written (`input.text`), which a run of the same release
  resolves the same way.
- **Partial reads.** `read()` loads a whole field, and `post extract` takes
  each rank's cells from it: memory per rank of one field of the whole mesh.
  Reading only a rank's cells (as `read_restart` does for its `cells`) goes
  behind the same interface when a mesh needs it.

## `mallard info`

`mallard info FILE` prints `open()`'s contents and the ranges of the last
snapshot; `--format json` the same as one object (`run`, and per snapshot its
`snapshot` block and `run_id`); `--input` the input of the file's run block.
The text has two metadata sections: **Run** (the run block, the same for
every file of the run) and **This snapshot** (the last snapshot's block,
with units). On a mesh file it also reports mesh quality from
`mallard check`'s summary (`--mesh-fields PREFIX` writes the fields). The text
follows the output principles of #657: tables rather than sentences, units
on every dimensional quantity (column headers included) and `dimensionless`
on the others, and each number labeled with what it is (which snapshot, per
file or for the run, simulated or wall time). The tables come from
`logging::table`, a plain column layout that #657's report layer can take
over. On a run directory it lists the run's output files and the run's
start and end from `mallard.jsonl`.

## `mallard diff`

`mallard diff A B` reads both files with `open()` and `read()`, so any two
formats compare, and matches cells by `Contents::global_order`: two files in
the mesh's global order compare cell by cell, two in their own order value by
value, and a mix is refused rather than silently misaligned. Restart files are
compared bit for bit through their `RestartHeader` (attributes, arrays, face
records read raw from `faces_offset`); the run block and snapshot blocks are left out.
The mesh, for `--remap` and for the coordinates of the largest differences, is
set up from the run block's recorded input (`[mesh]` and `[[periodic]]`, a
mesh file relative to `run.directory` and checked against `mesh.sha256`) with
`remap::source_mesh`, and `remap::Source` maps B onto A's cells. It runs on
one rank: reading is whole-field (see Partial reads), and the remap is only
collective over ranks that each hold part of both meshes.

## `mallard post extract`

`post extract` writes the probes of a run on the state of its output, so that
a probe added after the run, or a region given on the command line, means what
it would have meant during the run. It does not reimplement sampling: it sets
the case up as a run does (`Solver::init`) and lets the run's own `Probes` and
`[[write_data]]` writers write at each snapshot.

- **The case.** The input is `-i`'s, else the resolved input of the file's
  run block. The command rewrites it before the setup: it keeps the probes
  chosen (`--probe`) and adds the `--region` probe, every one written each
  time (`interval = 1`) to the output directory under the file names the
  input gives; it replaces `[[write_data]]` by the `--snapshot` writer and
  drops `[[forces]]`, `[integrals]`, `[statistics]`, the run log and
  `[initialize]`. Keys the rewrite leaves unread (a named region only a
  dropped writer used) are not an error.
- **The state.** `Solver::set_state_reader` gives `init` a function that
  reads each snapshot on the local cells instead of `[initialize]`: a restart
  file through `read_restart`, as a restart does; a snapshot through `read()`,
  mapped by global cell id (`Contents::global_order`). The solver takes the
  conservative variables where the file holds them, else computes them from
  `RHO` or `P` and `T`, `U` and the mass fractions with its own gas model.
  `write_output_of_state` then loads the next snapshot and writes, so the
  mesh, regions, probe geometry and sample locations are set up once for a
  whole series.
- **Memory.** With a state reader the setup allocates no time stepping:
  `allocate_time_stepping()` (face states and fluxes, the integrator's stages
  and right-hand sides, the last good state) is the one call it skips, and the
  rewrite turns off `[output] health`, whose ledger follows steps. The
  reconstruction stays the run's, because point samples, gradients and
  `Q`/`VORTICITY` read its polynomials.
- **Steps.** Each snapshot's step and time are those the reader finds
  recorded (`Snapshot::step` and `t`: the snapshot block, a restart file's
  header), never the file name; only the snapshots of files written before
  0.10 that record no step are numbered by position, as the report says.
- **Same values.** A probe's samples are evaluated and gathered as in the run,
  so restart files give the run's probe rows bit for bit on any number of
  ranks (`PostExtractTest`); snapshots of primitives agree to rounding.
- **The front end** runs it on the solver of `-i`'s dimension, else of the
  file's (restart files) or its run block's build (`Case::output_case`).

## `mallard post derive`

`post derive` sets the case up as `post extract` does (the same input
rewrite, state reader and skipped time stepping) and writes the derived fields
of each snapshot with the run's `[[write_data]]` writer, so a field is computed
by the run's own reconstruction, gradients and kernels.

- **Same values.** From restart files the fields equal the run's bit for bit
  (`PostDerive.RestartFilesGiveTheRunsDerivedFieldsBitForBit`); from VTKHDF
  snapshots of `RHO`, `U`, `P` and `T` the gradient fields were bitwise too
  in both cases of the benchmark below, and `MACH` differs in the last bit, from converting
  the stored primitives back to the conservative variables.
- **Cost.** The case's setup is the reconstruction's: for TENO its stencils,
  pseudo-inverses and smoothness matrices, the same tables as a run's
  (`[numerics.face_reconstruction] cache_file` stores them), and their memory. Each snapshot then costs one gradient pass and the write.

### Cost against ParaView

`benchmarks/post_derive` (described in `benchmarks/README.md`) derives
`MACH`, `Q_CRITERION`, `LAMBDA2`, `SCHLIEREN` and `VORTICITY` with
`mallard post derive` and with the equivalent `pvbatch` pipeline: ParaView's
`Gradient` filter on the cell data (with its Q-criterion and vorticity),
`Calculator` for `MACH` and `SCHLIEREN`, and numpy for λ₂, reading the same
VTKHDF snapshots and writing VTKHDF. Apple M4 Pro, macOS, 2 threads each
(Kokkos Threads; `VTK_SMP_MAX_THREADS=2`), ParaView 6.0.1; wall time and peak
memory (`/usr/bin/time -l`), the best of 3 (2D) or 2 (3D) runs:

| Case | Files | `post derive` | `post derive`, TENO cache | `pvbatch` |
|---|---|---|---|---|
| 2D double Mach reflection, 115,200 triangles, TENO5 (10 MB per snapshot) | 1 | 4.6 s, 1.1 GB | 0.64 s, 1.0 GB | 3.8 s, 0.21 GB |
|  | 11 | 5.0 s, 1.1 GB | 1.0 s, 1.0 GB | 4.4 s, 0.24 GB |
| 3D Taylor-Green vortex, 64³ hexahedra, TENO5 (37 MB per snapshot) | 1 | 32 s, 9.6 GB | 4.5 s, 9.5 GB | 3.3 s, 0.32 GB |
|  | 11 | 33 s, 9.6 GB | 7.2 s, 9.5 GB | 5.8 s, 0.37 GB |

- **Setup dominates.** Without a cache, the TENO setup is about 4 s in 2D and
  28 s in 3D, against ParaView's 3-4 s start (Python, the reader and the
  pipeline). Each further snapshot costs about 0.04 s in 2D and 0.27 s in 3D
  for `post derive` (the TENO gradient pass), 0.06 s and 0.24 s for `pvbatch`:
  in 2D `post derive` is faster from about 40 snapshots on, in 3D not at all.
- **With the run's cache.** Reading the TENO cache (0.76 GB in 2D, 8.4 GB in
  3D) instead of computing it makes `post derive` 6 and 4 times faster than
  `pvbatch` in 2D; in 3D it is slower by a quarter to a third.
- **Memory** is the TENO tables' (about 32 KB per hexahedron at order 5):
  5 times ParaView's in 2D, 30 times in 3D.

Agreement, as relative L2 norms of the difference over the norm of
`post derive`'s field:

| Field | against the run's `[[write_data]]` | against `pvbatch`, 3D, interior | against `pvbatch`, 2D |
|---|---|---|---|
| `MACH` | bitwise (restart files); last bit (snapshots) | 1e-16 | 1e-16 |
| `Q_CRITERION` | bitwise | 1.5e-2 | 0.87 |
| `LAMBDA2` | bitwise | 1.4e-2 | 0.69 |
| `SCHLIEREN` | bitwise | 1.1e-2 | 0.21 |
| `VORTICITY` | bitwise | 8.4e-3 | 0.57 |

`MACH` is pointwise and agrees to rounding. The gradient fields differ by
discretization: ParaView's cell-data gradient is a linear least-squares fit of
the neighbors' values, `post derive`'s the TENO5 polynomials of the run. On
the smooth Taylor-Green vortex they agree to 1-2% away from the box's faces
(2 cells), which ParaView does not know are periodic (14% for `Q_CRITERION`
over the whole box); behind the shocks of the double Mach reflection the
gradient of a discontinuity depends on the method, and they differ at order one.
