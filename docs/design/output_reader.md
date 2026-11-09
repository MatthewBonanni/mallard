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

The Python package ([#517](https://github.com/MatthewBonanni/mallard/issues/517))
binds this reader and the commands built on it; it does not reimplement them.
Until it does, `tools/mallard/*.py` keep reading the formats with h5py and
numpy, and `tools/dev/check_output_units.py` (run in CI) checks that
`mallard info --format json` and the Python readers report the same fields, units
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
- **Partial reads.** `read()` loads a whole field. Reading only a rank's
  cells (as `read_restart` does for its `cells`) is added with the first
  MPI-parallel command, behind the same interface.

## `mallard info`

`mallard info FILE` prints `open()`'s contents and the ranges of the last
snapshot; `--format json` the same as one object (`run`, and per snapshot its
`snapshot` block and `run_id`); `--input` the input of the file's run block.
The text has two metadata sections: **Run** (the run block, the same for
every file of the run) and **This snapshot** (the last snapshot's block,
with units). On a mesh file it also reports mesh quality from
`mallard check`'s summary (`--quality PREFIX` writes the fields). The text
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
