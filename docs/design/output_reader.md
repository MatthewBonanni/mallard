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
package loads it with ctypes, and its numpy arrays view the library's arrays
without a copy: each holds the library's result, which the last of them to go
frees. A read peaks at the size of what it returns
([measurements](python_package.md#performance)):

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
  `Q`/`VORTICITY` read its polynomials, but TENO sets up only its central
  stencils (`TENO::sector_stencils` off), which those read: no sector
  stencils, smoothness matrices or the moments they are formed from, no
  troubled-cell scratch, and no species reconstruction of mixtures, all of
  which only face values read. The central stencils' tables, and so
  gradients, are bitwise the run's
  (`CentralStencilsAloneAreTheFullSetupsBitForBit`). With `share_tables`,
  cells share basis means when the means alone agree on the sharing grid,
  where the run also compares smoothness matrices or moments, so point
  samples can differ from the run's below that grid. These tables are never
  written to `cache_file`, and a run's cache is read without its sector
  stencils. `STENCIL_*` fields summarize the sectors, so writing any keeps
  the full setup and gives the run's fields bit for bit
  (`StencilFieldsAreTheRunsBitForBit`); the rewrite drops `[output]
  write_stencils`, the run's file.
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
- **Cost.** The case's setup is the reconstruction's: for TENO the central
  stencils and their pseudo-inverses, as a run computes them
  (`[numerics.face_reconstruction] cache_file` stores them with the rest of a
  run's tables), and their memory. Each snapshot then costs one gradient pass
  and the write.

### Cost against ParaView

`benchmarks/post_derive` (described in `benchmarks/README.md`) derives
`MACH`, `Q_CRITERION`, `LAMBDA2`, `SCHLIEREN` and `VORTICITY` with
`mallard post derive` and with the equivalent `pvbatch` pipeline: ParaView's
`Gradient` filter on the cell data (with its Q-criterion and vorticity),
`Calculator` for `MACH` and `SCHLIEREN`, and numpy for λ₂, reading the same
VTKHDF snapshots and writing VTKHDF. One CPU-only Linux container on an AMD
EPYC 7763, 2 threads each (Kokkos Threads; `VTK_SMP_MAX_THREADS=2`), `mallard`
built with GCC 15.2 (Release, HDF5), ParaView 6.0.1's Linux binaries; wall
time and peak resident memory (`/usr/bin/time`), the best of 3 (2D) or 2 (3D)
runs:

| Case | Files | `post derive` | `post derive`, TENO cache | `pvbatch` |
|---|---|---|---|---|
| 2D double Mach reflection, 115,200 triangles, TENO5 (10 MB per snapshot) | 1 | 5.4 s, 0.17 GB | 2.2 s, 0.18 GB | 3.2 s, 0.42 GB |
|  | 11 | 6.0 s, 0.18 GB | 2.6 s, 0.18 GB | 5.2 s, 0.44 GB |
| 3D Taylor-Green vortex, 64³ hexahedra, TENO5 (37 MB per snapshot) | 1 | 56 s, 0.96 GB | 22 s, 0.97 GB | 3.9 s, 0.51 GB |
|  | 11 | 58 s, 0.96 GB | 24 s, 0.97 GB | 10.1 s, 0.55 GB |

- **Setup dominates.** Without a cache, the TENO setup takes about 5 s in 2D
  and 55 s in 3D, against ParaView's 3 to 4 s start (Python, the reader and
  the pipeline). Each further snapshot costs `post derive` 0.05 s in 2D and
  0.2 s in 3D, `pvbatch` 0.2 and 0.6 s, so `post derive` is the faster one
  from about 15 snapshots on in 2D, and from about 130 in 3D.
- **What the 3D setup is.** Of its CPU time, about half fits each cell's
  central stencil (pseudo-inverses and Lebesgue constants), a fifth packs the
  shared tables and a tenth computes the cells' moments, which host backends
  do four cells at a time, each with the operations it has alone (a quarter
  of the time before). Fitting each stencil shape once (#986) and packing
  each shape's table once (#1079) remove most of the rest.
- **With the run's cache.** Reading the TENO cache (0.76 GB in 2D, 8.4 GB in
  3D) instead of computing it makes `post derive` faster than `pvbatch` from
  the first snapshot in 2D; in 3D it takes 22 s, mostly ranking every cell's
  table again to share them (#1079).
- **Memory.** `post derive` sets TENO up with its central stencils alone, the
  only tables the gradient reads, so its memory is mostly the mesh's (face
  quadrature): less than ParaView's in 2D, about twice it in 3D. The peak is
  the TENO setup's; after it, the solver holds only what the requested outputs
  read (below).

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

### Buffers only for their readers

The vertex gradient's weights (`make_vertex_gradient`) take about as much
memory as the gradients they fill. `Solver::init_vertex_gradient` builds them
once every writer, probe and model is set up, and only if one of them reads
them: the right-hand side's viscous fluxes and upwind sensor in runs that step,
the velocity gradients of schemes without cell gradients (TENO's come from its
polynomials), mixture transport, LES, probes of gradients, mass-fraction
gradients (`FLAME_INDEX`, `SCALAR_DISSIPATION`) and viscous forces. A read
without them throws, so a reader added without a line there fails loudly.
Likewise gas mixtures stage `Y_*` and `X_*` on the host, and compute the mole
fractions at each snapshot, only when a writer or snapshot writes them
(`Solver::init_mixture_fractions`; the divergence report stages `Y_*` itself).

Measured on a CPU-only Linux container (AMD EPYC 7763, GCC 15.2, Kokkos
Threads, 2 threads), resident memory from `/proc`, outputs identical bit for bit
(`mallard diff`) before and after:

| Case | Command | Before | After |
|---|---|---|---|
| 3D Taylor-Green vortex, 64³, TENO5, Navier-Stokes | `post derive --fields MACH,Q_CRITERION,LAMBDA2,SCHLIEREN,VORTICITY`, 2 restart files | 776 MB after setup, 953 MB peak | 648 MB after setup, 953 MB peak |
| 2D H2/air, 512², TENO5, Euler mixture, 10 restart files | `post derive --fields MIXTURE_FRACTION,MACH` | 430 MB peak | 359 MB peak |
|  | `post extract --snapshot vtu --fields Y_H2,T` | 371 MB peak | 352 MB peak |
|  | `post extract --snapshot vtu --fields X_H2` | 370 MB peak | 352 MB peak |
|  | `post average --conditional '{on = "MIXTURE_FRACTION", fields = ["T", "U_X", "Y_H2"]}'` | 424 MB peak | 351 MB peak |

Without the weights' fit, the 3D `post derive` of one file takes 52.8 s instead
of 56.1 s (best of 3); the 2D commands' 14 s, mostly the TENO setup, are
unchanged within noise.
