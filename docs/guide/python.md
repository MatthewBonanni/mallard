# Python

The Python package `mallard` reads every file Mallard writes or reads into numpy
arrays, and runs Mallard's commands as functions. Use it for your own analysis
in scripts and notebooks: plots, time histories, comparisons with references,
anything ParaView is not the right tool for. It does not reimplement the
formats: `mallard.open` goes through the solver's own reader, compiled into a
library, `libmallard2d` or `libmallard3d`, so the package reads a file exactly
as `mallard info` does, and `mallard.post.extract` runs `mallard post extract`
itself. Its only dependency is numpy.

```python
import mallard

f = mallard.open("solut/sod.vtkhdf.series")
rho = f.read_all("RHO")                       # (11, 800): every snapshot
x = f.geometry().vertex_averages()[:, 0]      # where each cell is
```

Every function and class: [Python API](python_api.md).

## Installing

The package is `tools/mallard` of the Mallard repository (`share/mallard/tools/mallard`
of an install), and its library comes with a build with `-DMallard_ENABLE_PYTHON=ON`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF -DKokkos_ENABLE_THREADS=ON \
      -DMallard_ENABLE_HDF5=ON -DMallard_ENABLE_PYTHON=ON
cmake --build build -j
```

The build writes `build/src/libmallard2d.so` (`.dylib` on macOS) next to
`build/src/mallard`; `cmake --install build --prefix PREFIX` installs it in
`PREFIX/lib`. A build with `-DMallard_DIM=3` gives `libmallard3d`; either reads
the output of both dimensions. Then install the package into your Python
environment, with [uv](https://docs.astral.sh/uv/) or pip:

```bash
uv venv && uv pip install -e tools        # in the repository; or: pip install -e tools
uv pip install PREFIX/share/mallard/tools  # from an install
```

`-e` (editable) keeps the package in the repository, so it follows `git pull`.
The package looks for the `mallard` executable on `PATH` and for its library
next to it (or in `../lib`); two environment variables say where they are
otherwise:

| Variable | Default | What |
|---|---|---|
| `MALLARD_EXECUTABLE` | `mallard` on `PATH` | The executable the commands run, and next to which the library is looked for |
| `MALLARD_LIBRARY` | next to the executable, then in `../lib` | The library, or both libraries separated by `:` |

The package is versioned with Mallard (`mallard.__version__`), and a build
checks that they match. Use the package of the same checkout or install as the
library.

## Quick start

The [Sod shock tube](https://matthewbonanni.github.io/mallard/docs/examples/#sod)
runs in about a second and writes a VTKHDF series:

```bash
mallard new sod sod && cd sod && mallard -i input.toml
```

```python
import mallard

f = mallard.open("solut/sod.vtkhdf.series")
f                  # <mallard.File solut/sod.vtkhdf.series: snapshot, VTKHDF series index (VTKHDF snapshots), 11 snapshots>
list(f.fields)     # ['P', 'RHO', 'T', 'U_X']
f.fields["RHO"]    # Field(name='RHO', components=1, units='kg m-3', long_name='density', location='cell')
f.times            # array([0.  , 0.02, 0.04, ..., 0.2 ])
f.read("RHO")      # the last snapshot: (800,)
f.read("RHO", 0)   # the first
f.read_all("RHO")  # (11, 800)

g = f.geometry()
g.points.shape, g.n_cells    # ((1005, 3), 800)
centers = g.vertex_averages()
```

![Density of each snapshot of the Sod shock tube along the tube, and its history at x = 0.8, read with the Python package](../images/guide/python-sod-light.svg#gh-light-mode-only)
![Density of each snapshot of the Sod shock tube along the tube, and its history at x = 0.8, read with the Python package](../images/guide/python-sod-dark.svg#gh-dark-mode-only)

The figure is [`tools/docs/python.py`](../../tools/docs/python.py): the rows of
cells along the tube from `vertex_averages()`, every snapshot from one
`read_all`.

## Reading files

`mallard.open(path)` takes any file Mallard writes or reads and returns a
`File`. Its attributes are what `mallard info FILE --format json` prints:

| Attribute | What |
|---|---|
| `kind` | `"snapshot"`, `"probe"`, `"table"`, `"restart"` or `"mesh"` |
| `format`, `version`, `precision`, `compressed` | The format (`"VTKHDF"`, `"VTU"`, `"PVD series index (VTU snapshots)"`, ...), its [format version](output.md#formats-and-compatibility), `"double"` or `"single"`, and compression |
| `fields` | `{name: Field}`: each field's `components`, `units`, `long_name` and `location` (`cell`, `point`, `column`) |
| `snapshots`, `times` | Each snapshot's `t`, `step`, `file`, `run_id` and [snapshot block](output.md#snapshot-metadata) (`block`); their times |
| `mesh` | Cell and point counts, cell types, bounds, boundary and cell zones |
| `global_order` | Whether cell values are in the global cell order of the run's mesh |
| `run` | The [run block](output.md#run-block) of the run that wrote the file: code, build, resolved input, mesh, `run_id` |
| `notes` | What a reader should know about the file: no units, written before a format change, ... |

### Values

`f.read(field, snapshot=-1)` gives a field at one snapshot, `f.read_all(field)`
at every one; `snapshot` counts from 0 in time order, and from the end when
negative. Scalars come as `(n,)`, vectors (`U`, `VORTICITY` in 3D,
`TRACTION_PRESSURE`) as `(n, 3)`, with z zero in 2D; `read_all` adds a first
axis of snapshots. Values are float64 whatever precision wrote them. `f["P"]`
is `f.read("P")`.

```python
u = f.read_all("U_X")            # (11, 800)
mean_u = u.mean(axis=0)          # the mean over the snapshots, cell by cell
```

The order is the file's: the order of `f.geometry()`, so values and cells line
up. Files with `global_order` (restart files, VTKHDF and XDMF snapshots of the
whole mesh) are in the global cell order of the run's mesh, the same on any
number of ranks, and two such files of a run compare cell by cell. VTU pieces
and [region snapshots](output.md#snapshots-of-regions) are in their own order.

### Geometry

`f.geometry(snapshot=-1)` gives the cells the values are on: `points` `(n, 3)`,
and each cell's points `connectivity[offsets[i]:offsets[i + 1]]` with its
[VTK cell type](https://vtk.org/doc/nightly/html/vtkCellType_8h.html) in
`types`. `vertex_averages()` places each cell for a plot. To draw the cells,
for instance with matplotlib:

```python
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

g = f.geometry()
cells = [g.points[g.connectivity[a:b], :2] for a, b in zip(g.offsets[:-1], g.offsets[1:])]
fig, ax = plt.subplots()
ax.add_collection(PolyCollection(cells, array=f.read("P"), cmap="viridis"))
ax.autoscale()
ax.set_aspect("equal")
```

Probe files give their points (and their cells: a curve's lines, a surface's
polygons). Restart files and tables hold no geometry, and say so: open the
run's mesh file for a restart file's cells, and read a table's `x`, `y`, `z`
columns.

### Every format

| Files | `kind` | What `read` gives |
|---|---|---|
| `PREFIX.vtkhdf.series`, `PREFIX.vtkhdf` (`series = true`), `PREFIX_<step>.vtkhdf` | `snapshot` | Cell values of each snapshot |
| `PREFIX.pvd` and `.pvtu`, `.vtu` | `snapshot` | Cell values; a `.pvtu`'s pieces one after the other |
| `PREFIX.xmf` (XDMF output), `PREFIX_<step>.h5` | `snapshot` | Cell values; the geometry from `PREFIX_mesh.h5` |
| Region and boundary snapshots (`region =` in `[[write_data]]`) | `snapshot` | The region's cells, in their own order |
| `<prefix>_<step>.restart`, compressed too | `restart` | The conservative variables: `RHO`, `RHOU_X`, ..., `RHOE`, `RHOY_<species>`, and `[statistics]` sums |
| Probe `.h5`, `.vtkhdf`, `.pvd` of `.vtp` | `probe` | The samples at each time: `(n_points,)` per snapshot |
| `.csv`: probes, reductions, forces, integrals, spectra | `table` | A column's rows of one time (`read`) or all of them (`read_all`) |
| Mesh files: `.msh`, `.cas`, `.cgns`, Mallard HDF5 meshes, OpenFOAM cases | `mesh` | No values; the geometry |

Files whose write did not finish (`NAME.partial`) are refused, as by every
reader. Files of earlier releases open as
[Formats and compatibility](output.md#formats-and-compatibility) says, and
`notes` tells how their steps and runs were found.

### Restart files and meshes

```python
r = mallard.open("solut/restart_000400.restart")
r.fields.keys()                    # dict_keys(['RHO', 'RHOU_X', 'RHOU_Y', 'RHOE'])
u = r.read("RHOU_X") / r.read("RHO")
m = mallard.open("cylinder.msh")   # a mesh file: m.geometry(), m.mesh["boundary_zones"]
```

A mesh file is read by the library of its dimension: `mallard.open(path, dim=3)`,
or each library found in turn. Output files are read by either.

## Probes, monitors and statistics

Probes, forces, integrals and spectra write CSV tables or probe files
([Probes and monitors](probes.md)); statistics are fields of snapshots and
restart files ([Statistics](statistics.md)). The
[measuring tutorial](../tutorials/measuring.md) writes all of them; in its run
directory:

```python
import mallard

forces = mallard.open("forces_cylinder.csv")
t = forces.read_all("t")                                # every row
lift = forces.read_all("Fy_pressure") + forces.read_all("Fy_viscous")
forces.fields["Fy_pressure"].units                      # 'kg s-2': per unit depth, in 2D

wake = mallard.open("probe_wake.csv")                   # one row per time of the point probe
v = wake.read_all("U_Y")                                # (n_times,)

box = mallard.open("probe_box_reductions.csv")
mass = box.read_all("integral_RHO")                     # the mass in the control volume

stats = mallard.open("solut/box.vtkhdf.series")         # the box's cells at the end
mean_u = stats.read("MEAN_U_X")
uu = stats.read("COV_U_X_U_X")
stats.snapshots[-1].block["statistics"]                 # {'average_start': 60.0, 'average_end': 200.0, 'samples': ...}
phases = [stats.read(f"PHASE_{k}_U_Y") for k in range(8)]
```

A table's `read_all(column)` is the whole column; `read(column, k)` the rows of
its `k`-th time (a curve probe writes one row per point and time). Probe HDF5
and VTKHDF files hold the samples of each time as a snapshot:
`read_all("P")` is `(n_times, n_points)`, and `geometry().points` where they
are. A table continued by several runs (restarts) holds the rows of each, and
each snapshot's `run_id` says which run wrote it.

## Run records and run directories

```python
records = mallard.run_record(".")      # mallard.jsonl, or the file [output] run_log names
start = records[0]                     # the run block, placement, setup times
progress = [r for r in records if r["type"] == "progress"]
cfl = [r["cfl"] for r in progress]
end = records[-1]                      # end["outcome"]: 'finished', 'stopped', 'diverged', ...
written = [r["path"] for r in records if r["type"] == "output"]

mallard.run_files(".")                 # the run's output files: series indexes, restart, probe and CSV files
```

Each record is a dict, as the [run record](output.md#run-record-mallardjsonl)
describes it. A run that restarted appends to the record, so `start` records
divide it into the runs that wrote it.

## Running commands

Mallard's commands are functions that run the `mallard` executable and return
its JSON report as a dict. Their keyword arguments are the command's options,
as `mallard COMMAND --help` lists them: `fields=["RHO", "P"]` is
`--fields RHO,P`, `latest=True` is `--latest`, `input="case.toml"` is `-i case.toml`.

```python
report = mallard.info("solut/sod.vtkhdf.series")          # as mallard.open(...).contents, with min and max

d = mallard.diff("a/solut/flow_000400.vtkhdf", "b/solut/flow_000400.vtkhdf")
d["result"], [(f["name"], f["relative"]["linf"]) for f in d["fields"]]

out = mallard.post.extract("solut/flow.vtkhdf.series", region="domain", fields=["RHO"],
                           reductions=["integral"], latest=True, output="extract")
mallard.open(out["written"][0]["path"])                  # the file it wrote

conv = mallard.post.convergence(["coarse", "medium", "fine"], quantity="forces_cylinder:Fx_pressure")
```

`nprocs=4` runs a command on four MPI ranks (`mpirun -n 4`, or the launcher
`mpirun=` names), `threads=8` sets `--kokkos-num-threads`, and `cwd=` the
directory it runs in. A command that fails raises `mallard.CommandError`, with
its exit status, output and error message; `mallard.diff` and
`mallard.post.convergence` return differences beyond tolerance and diverging
quantities as results (`d["exit_status"]` is 1), since those are their
verdicts. `mallard.command` runs any other command that has `--format json`:
`mallard.command("check", "input.toml")`, `mallard.command("list", "fields", "input.toml")`.

## Pitfalls

- **A file of the run still being written.** Snapshots and probe files are
  complete once their name appears; a CSV table grows as the run appends to it,
  so `read_all` reads the rows written so far.
- **Mesh files of the other dimension.** A 2D mesh needs `libmallard2d` and a 3D
  mesh `libmallard3d`: build both, or pass the right `dim`. Output files are
  read by either.
- **Cell order.** Values of a region snapshot or a VTU piece are not in the
  mesh's global order; line them up with their own `geometry()`, not with the
  whole mesh's.
- **Memory.** A field is read whole: `read_all` of a large series holds every
  snapshot at once. Loop over `read(field, k)` to hold one.
- **Old builds.** The library reads the format versions of its own Mallard and
  earlier, and refuses newer files with a message saying so: use the package and
  library of the newer Mallard.

## See also

- [Python API](python_api.md): every function and class
- [Output](output.md): what a run writes, units, run block and snapshot metadata
- [Command line](cli.md): `mallard info`, `mallard diff`, `mallard post extract`, `mallard post convergence`
- [Tools](../../tools/README.md): the plotting and animation scripts in `tools/`
- Design notes: [Reading output](../design/output_reader.md)
