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

f = mallard.open("solut/sod.vtkhdf")
rho = f.read_all("RHO")                       # (11, 800): every snapshot
x = f.geometry().vertex_averages()[:, 0]      # where each cell is
```

Every function and class: [Python API](python_api.md). With the solver
bindings, the package also runs cases in the Python process, step by step
([Running cases](#running-cases)); the [C++ API](cpp_api.md) is the same from C++.

## Installing

Every installer but the [relocatable archive](../installation.md#relocatable-archive)
installs the package with Mallard, so `import mallard` works after its usual
command:

| Installer | Command | Opt out |
|---|---|---|
| [Container](../installation.md#container) | `docker run --rm -v "$PWD:/work" ghcr.io/matthewbonanni/mallard python3 script.py` | |
| [Spack](../installation.md#spack) | `spack install mallard && spack load mallard` | `~python` |
| [From source](../installation.md#from-source) | `cmake --install build`, then `uv pip install PREFIX/share/mallard/tools` | `-DMallard_ENABLE_PYTHON=OFF -DMallard_ENABLE_LIBRARY=OFF` |

The package is `tools/mallard` of the Mallard repository
(`share/mallard/tools/mallard` of an install; its distribution name, for uv,
is `mallard-python`). A build writes its library,
`build/src/libmallard2d.so` (`.dylib` on macOS), next to `build/src/mallard`,
and `cmake --install` installs it in `PREFIX/lib`; a build with
`-DMallard_DIM=3` gives `libmallard3d`, and either reads the output of both
dimensions. From a build or an install, add the package to a Python
environment with [uv](https://docs.astral.sh/uv/):

```bash
uv venv && uv pip install -e tools        # in the repository
uv pip install PREFIX/share/mallard/tools  # from an install
```

`-e` (editable) keeps the package in the repository, so it follows `git pull`.
`-DMallard_PYTHON_INSTALL_DIR=DIR` makes `cmake --install` also copy the
package into `DIR`, a `site-packages` directory of your Python (the containers
and Spack install it this way). The package looks for the `mallard` executable
on `PATH` and for its library next to it (or in `../lib`); two environment
variables say where they are otherwise:

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

f = mallard.open("solut/sod.vtkhdf")
f                  # <mallard.File solut/sod.vtkhdf: snapshot, VTKHDF, 11 snapshots>
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
| `PREFIX.vtkhdf`, `PREFIX.vtkhdf.series` and `PREFIX_<step>.vtkhdf` (`series = false`) | `snapshot` | Cell values of each snapshot |
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

stats = mallard.open("solut/box.vtkhdf")                # the box's cells at the end
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
report = mallard.info("solut/sod.vtkhdf")                 # as mallard.open(...).contents, with min and max

d = mallard.diff("a/solut/flow.vtkhdf", "b/solut/flow.vtkhdf")
d["result"], [(f["name"], f["relative"]["linf"]) for f in d["fields"]]

out = mallard.post.extract("solut/flow.vtkhdf", region="domain", fields=["RHO"],
                           reductions=["integral"], latest=True, output="extract")
mallard.open(out["written"][0]["path"])                  # the file it wrote

derived = mallard.post.derive("solut/flow.vtkhdf", ["MACH", "Q_CRITERION"], latest=True)
mallard.open(derived["written"][0]["path"]).read("MACH")

avg = mallard.post.average("solut/flow.vtkhdf", times="40:80", fields=["U_X", "U_Y", "P"], favre=True)
mallard.open(avg["written"][0]).read("RMS_U_X")         # avg["window"], each snapshot's weight

conv = mallard.post.convergence(["coarse", "medium", "fine"], quantity="forces_cylinder:Fx_pressure")

mallard.report("runs/cylinder")["report"]                # rebuilds runs/cylinder/report.html
archive = mallard.pack("runs/cylinder", output="cylinder.tar.gz", with_outputs="all")
archive["rerun"]["command"], [f["path"] for f in archive["files"] if f["role"] == "output"]
check = mallard.pack("cylinder.tar.gz", verify=True)     # check["result"]: 'ok' or 'differs'
```

`nprocs=4` runs a command on four MPI ranks (`mpirun -n 4`, or the launcher
`mpirun=` names), `threads=8` sets `--kokkos-num-threads`, and `cwd=` the
directory it runs in. A command that fails raises `mallard.CommandError`, with
its exit status, output and error message; `mallard.diff` and
`mallard.post.convergence` return differences beyond tolerance and diverging
quantities as results (`d["exit_status"]` is 1), since those are their
verdicts, and so does `mallard.pack(archive, verify=True)` with the files that
differ from the archive's manifest. `mallard.command` runs any other command that has `--format json`:
`mallard.command("check", "input.toml")`, `mallard.command("list", "fields", "input.toml")`.

## Running cases

`mallard.Case`, `mallard.Session` and `mallard.Solver` run a case in the
Python process, as the [C++ API](cpp_api.md) does in a program: the same run as
`mallard -i input.toml`, with the same output, run log and results bit for bit,
stepped from Python, with its fields as arrays. They need the solver bindings,
an extension module of the package (`mallard._core2d`, `mallard._core3d`) that
the containers and Spack install; from source, configure with
`-DMallard_ENABLE_PYTHON_BINDINGS=ON`. The rest of the package works without
them, and loads them only when a case is built.

```python
import mallard

mallard.Session(["--kokkos-num-threads=4"])           # optional: Kokkos' defaults otherwise
solver = mallard.Solver(mallard.Case.from_toml_file("input.toml"))
solver.setup()
while solver.advance_to(solver.time() + 0.01):
    p = solver.field("P")                              # numpy, the solver's own memory
    print(solver.time(), p.max(), solver.monitor("integrals_kinetic_energy").value)
solver.finish()
```

- **The input is the record.** A case is TOML: `Case.from_toml_file(path)` or
  `Case.from_toml(text)` for one a script writes; the run log and every output
  file record its exact text, so `mallard -i` on it reruns the run. There is
  no other way to configure a run from Python but writing that text:
  `case.set(key, value)` returns the case with one key set (`"run.cfl"`,
  `"probes[0].interval"`, the array's length appending an element; a bool,
  int, float, str, or a list or dict of them),
  changing only that key's line, so `toml()` stays the whole input, and
  `case.save(path)` writes it to a file and returns the case of that file
  ([C++](cpp_api.md#building-a-case-in-code)).

    ```python
    import os
    base = mallard.Case.from_toml_file("input.toml")
    for cfl in (0.3, 0.5, 0.7):
        os.makedirs(f"cfl_{cfl}", exist_ok=True)
        case = base.set("run.cfl", cfl).set("write_data[0].prefix", f"cfl_{cfl}/flow")
        mallard.Solver(case.set("output.run_log", f"cfl_{cfl}/mallard.jsonl").save(f"cfl_{cfl}/input.toml")).run()
    ```

- **Typed configs.** `mallard.config` has a class per table of the input,
  generated from the key catalog, with a keyword per key, typed and documented
  as [the input reference](../input.md) documents it; `case.set(table)` sets
  every key it holds through `case.set(key, value)`, so the TOML stays the
  record. Classes, element indices and nesting are those of
  [C++](cpp_api.md#typed-configs); a table keyword also takes a dict of its
  keys, a key with a physical dimension a number in SI units, a string with its
  unit or `(value, unit)`, and `config.Inline({...})` writes an inline table
  as given. Every region is a `config.Region` (`config.Regions("jet", ...)`
  in `[regions]`); a list of regions mixes them with names:
  `region=config.Region(type="union", regions=["jet", config.Region(type="sphere", center=[0.5, 0.5], radius=0.1)])`.

    ```python
    from mallard import config
    case = (mallard.Case.from_toml_file("input.toml")
            .set(config.Run(cfl=0.5, t_stop=(2.0, "ms")))
            .set(config.Numerics(face_reconstruction={"type": "TENO", "order": 5}))
            .set(config.Probes(1, name="wake", fields=["P"], point=[0.5, 0.1], file="wake.csv")))
    ```

- **Checking.** `case.check()` is `mallard check` on the case, as a report:
  `status`, `ok`, `diagnostics`, `unused_keys` (`key`, `line`,
  `suggestion`), `boundaries`, `time_step`, `resolution`, `memory`, `cost`,
  `resolved`, `keys`, and the whole report as `text` and `json`
  ([C++](cpp_api.md#checking-a-case)); `check(ranks=8, memory="a100-80")`
  takes `mallard check`'s options, `log=` a sink for the setup's output.
  `keys` is the resolved input key by key: each `mallard.InputKey` has its
  `path`, `catalog_path`, `value` (TOML text, in plain units), `set`, and its
  record in the key catalog, `type`, `dimension`, `unit`, `required`,
  `default` and `values`: `{k.path: k for k in case.check().keys}`.
- **Calls.** `setup()`, `step()`, `step(n)`, `advance_to(t)`, `finish()`, `run()`,
  `time()`, `step_count()`, `finished()`, `monitors()`, `monitor(name)`,
  `statistics()`, as in [C++](cpp_api.md#running-a-case); failures raise
  `mallard.Error`, with the message `mallard` prints and its `code`.
- **Fields without copies.** `field(name)` is a read-only NumPy array of the
  solver's memory on a CPU build; on a GPU it is a `mallard.DeviceArray` of
  device memory, which `cupy.asarray`, `torch.from_dlpack` and Numba take
  without a copy (`__cuda_array_interface__`, DLPack); `field(name, memory="host")`
  copies it to NumPy. The names are those of the
  [C++ API](cpp_api.md#fields): `RHO`, `P`, `conservatives`, `MEAN_U_X`,
  `CENTROID`, ... (`field_names()`).
- **Setting the state.** `writable_field(name)` is a writable view of the
  state: change it in place (`rho = solver.writable_field("RHO"); rho *= 1.01`),
  then `commit_state()` or step; the run log records a `state_set` event.
- **Callbacks.** `on_step(f)` calls `f(solver)` after every step and
  `on_stage(f)` calls `f(solver, stage)` at each stage of the time integrator
  (`stage.step`, `stage.index`, `stage.time`); `remove_callback(id)`. An
  exception in a callback ends the run as an error and is raised from the call
  that stepped. `on_progress(f)` calls `f(progress)` with each progress row
  (`step`, `time`, `dt`, `fraction`, `eta`, `diagnostics`, `written`, `json`),
  `on_output(f)` calls `f(output)` with each file written (`kind`, `path`,
  `reason`, `files`, `json`) and `on_record(f)` calls `f(record)` with every
  record of the run log (`type`, `json`); they keep the GPU's step queue, and
  get their record only ([C++](cpp_api.md#progress-output-and-the-run-log)).
- **Console output.** `set_log(sink)` sends what the solver prints to
  `sink(message)` (`message.level`: `"info"`, `"warning"` or `"error"`;
  `message.code`, `message.text`) instead of the terminal;
  `set_log(lambda message: None)` silences it, `set_log(None)` prints again.
- **One session per process.** Kokkos starts once per process, and a process
  runs cases of one dimension: run others in another process
  (`multiprocessing`, a job array). The session ends when Python exits.
- **MPI.** Start the script with `mpirun -n 4 python3 script.py` on a build
  with MPI: every call is collective, fields hold each rank's owned cells
  (`GLOBAL_ID` places them), and `monitors()` are the same on every rank.
  Callbacks run on every rank, but `on_output` and `on_record` on rank 0, which
  writes the run log.
  To run on some of the ranks only, for instance beside another code, give the
  session an [mpi4py](https://mpi4py.readthedocs.io) communicator
  (`uv pip install "mallard-python[mpi]"`): every rank of it starts the session
  and runs the same cases, and the other ranks are free.

    ```python
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    comm = world.Split(color=int(world.Get_rank() < 4))   # the first 4 ranks run Mallard
    mallard.Session(comm=comm)
    ```

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
- **Memory.** A field is read whole, into the array returned without a copy:
  `read_all` of a large series holds every snapshot at once. Loop over
  `read(field, k)` to hold one.
- **Old builds.** The library reads the format versions of its own Mallard and
  earlier, and refuses newer files with a message saying so: use the package and
  library of the newer Mallard.

## See also

- [Python API](python_api.md): every function and class
- [From run to figure](../tutorials/post_processing.md): a tutorial that takes three runs from
  `mallard info` through `mallard post extract`, `post derive`, `post average` and `post convergence` to figures
  drawn with the package
- [Output](output.md): what a run writes, units, run block and snapshot metadata
- [Command line](cli.md): `mallard info`, `mallard diff`, `mallard post extract`, `post derive`, `post average`,
  `post convergence`, `mallard pack`
- [Tools](../../tools/README.md): the plotting and animation scripts in `tools/`
- Design notes: [Reading output](../design/output_reader.md), [The Python package](../design/python_package.md)
