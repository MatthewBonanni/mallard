# The C++ library API

How a program drives Mallard in process instead of running `mallard -i input.toml`
([#345](https://github.com/MatthewBonanni/mallard/issues/345)), and how the Python
bindings (#345) and the plugin API
([#336](https://github.com/MatthewBonanni/mallard/issues/336)) build on it. The
user-facing page is [C++ API](../guide/cpp_api.md); the Python side is laid out in
[The Python package](python_package.md).

## Invariant: the TOML input is the run's record

Every run, however it is driven, is fully described by one TOML input, and that
text is the source of truth:

- **A run is built from TOML.** `mallard::Case` holds the TOML text and nothing
  else; `Solver` builds the run from that text through the same `Input` as
  `mallard -i`. No setter, override or side channel changes a run's
  configuration outside the TOML. A typed config built in code (later slices) is
  a view that emits TOML, never a second path into the solver.
- **The run records the exact text.** Provenance (the run log's start record,
  the run block of every output and restart file) holds the input text and its
  SHA-256, byte for byte as given, so `mallard -i` on it reruns the run.
- **Round trip.** A case built in code emits TOML, and running that TOML gives
  the same run bit for bit. Each slice that adds a way to build a case adds a
  test of this.
- **Programs that save runs write the file first.** The GUI writes `input.toml`,
  and its worker builds the run from that file on disk
  (`Case::from_toml_file`); what the GUI holds in memory never reaches the
  solver otherwise. A case from a string (`Case::from_toml`) is for programs
  that generate inputs; it is recorded the same way.

Runtime control is not configuration: stepping, `advance_to`, stopping early and
(later) callbacks decide *when* the solver acts, not *what* it solves, as
`mallard ctl` and triggers do today. A runtime setting that changes the
solution (`mallard ctl set`) goes through `apply_setting`, which records it in
the run log as it does now, so the record stays complete.

## Run setup today

`mallard -i input.toml` (`run_main.cpp`, `solve`) starts MPI (`comm::Session`),
binds the rank to a device (`binding::select_device`), initializes Kokkos, then
`Solver::init(file)`: parse into `Input` (which records every lookup), upgrade
deprecated keys, apply the numerics preset, `setup_from_input` (run parameters,
mesh, physics and boundaries, numerics, halo rebuilds, fields, statistics,
sources, output, initial state, health checks), fail on keys no lookup read
(`unused_keys`), record the resolved input and provenance. `Solver::run` then
writes the start's output, steps until a stop condition ([run] `n_steps`,
`t_stop`, `t_wall_stop`, a trigger, a signal or `mallard ctl stop`) and writes
the end's output, summary and end record. Steps run on the host one by one, or
are enqueued on the device ahead of the host up to a *horizon* (the next step
the host must act at: an output, a monitor, a trigger, `n_steps`, `t_stop`),
in CUDA builds as captured graphs (`solver_graph.cpp`).

## Public surface

One installed header per concept under `include/mallard/`, namespace
`mallard`, depending only on the standard library: no Kokkos, MPI, toml11 or
Mallard internals in public headers, so programs compile without them and the
solver's internals change freely. The first slice is one header,
`mallard/mallard.h`; later ones split it (`case.h`, `solver.h`, `session.h`,
`config.h`, `plugin.h`) behind it.

| Type | Role |
|---|---|
| `Session` | MPI and Kokkos for the process (below) |
| `Case` | A run's TOML: `from_toml_file(path)`, `from_toml(text)`, `toml()`; later typed configs (below) |
| `Solver` | `setup()`, `step()`, `advance_to(t)`, `finish()`, `run()`, `time()`, `step_count()`, `finished()`, `monitors()` |
| `Monitor` | Latest value of an integral, force or probe value: name (as triggers name it), UDUNITS unit, value, step |
| `Error` | Every failure: the diagnostic code and the message `mallard` prints |

Later slices add, each additive to the API version:

- **State access**: `Solver::fields()` gives named views of the owned cells
  (`RHO`, `U`, `P`, `Y_<species>`; conservative and primitive), as a
  `mallard::Field` with extents, strides, scalar type and memory space
  (host or device pointer), read-only by default; `set_state(...)` writes the
  conservative state at a step boundary only (between steps, never inside one),
  through the same path a restart takes, and the run log records that the state
  was set. These map to DLPack and `__cuda_array_interface__` in Python without
  copies. The geometry (cell centroids, volumes, global ids) comes the same way.
- **Callbacks**: `on_step(f)` after every step the host takes, `on_output(f)`,
  `on_progress(f)` with the progress record as a struct (the same fields as the
  run log's `progress` record), `on_stage(f)` at each Runge-Kutta stage. A run
  with a per-step or per-stage callback cannot enqueue steps ahead, so it takes
  every step on the host as a run with host triggers does today
  (`step_horizon`); `on_progress` and `on_output` keep the device queue, since
  the host already acts at those steps.
- **Diagnostics**: statistics means and their convergence, forces, the run
  status (`RunStatus`, as `mallard ctl status` reports it), the check report
  (`mallard check`) and the resolved input with its key catalog records.
- **Logging sink**: where the text the solver prints goes (stdout by default,
  a callback, or nowhere), and the run log's records as structs.

Collective semantics: under MPI every call of `Solver` is collective, and
values (`monitors()`) are the same on every rank. A solver runs on
`MPI_COMM_WORLD`, as the executable does; taking a communicator (an MPI
`MPI_Comm` as an opaque handle, mpi4py's in Python) comes with the bindings,
through `comm.cpp`'s communicator, which `SelfScope` already swaps.

## Typed configs and the TOML front end: one path

The typed config objects (`mallard::config::Mesh`, `Physics`, `Boundary`,
`Numerics`, `Probe`, ...) do not get a second setup path. They are a typed,
documented way to write the `Input` tree: each object writes its keys into a
`toml` table under its section, and `Case` holds the resulting TOML text, which
`Solver::setup` parses exactly as a file. So:

- unused-key detection, the error messages with lines (of the emitted TOML),
  deprecated-key upgrades, presets, `mallard check` and the resolved input work
  unchanged, with no code in setup knowing where the input came from;
- `Case::toml()` is always available and is what provenance records, which is
  the invariant above;
- the typed objects are generated from the key catalog (`docs/input-keys`,
  which records every key, its type, default, dimension and accepted values from
  the lookups setup makes), so they cannot drift from what setup reads. A
  generator in `tools/dev/` writes `include/mallard/config.h` from the catalog;
  CI checks it is current as it checks the catalog.

Escape hatch: `Case::set(key, value)` with a TOML value edits the same text by
key path, for keys the generated objects do not yet cover. It returns a new
`Case` (cases are values), so a sweep builds many cases from one.

## Stepping and the run's loop

`Solver::run` is split into `start_run` (the start's output and progress row),
`advance_run(until_step, until_time)` (one pass of the loop at a time until the
stop condition or the limit) and `finish_run` (the end's output, summary, end
record); `run()` is the three in a row, so the executable's run is unchanged.
The API's limits enter the run as two more limits of the horizon
(`step_horizon`: the device queue never runs past them) and of the landing
time (`next_target_time`: the last step lands on `until_time` exactly as on
`t_stop` or an output time). Stepping by one runs every step on the host side
of the queue, which takes the same steps bit for bit. `finish()` before the stop
condition ends the run with the outcome `stopped`, cause `caller`.

## Kokkos and MPI ownership

`mallard::Session` initializes MPI if it is not initialized (then MPI before
Kokkos, so Kokkos can map ranks to devices, as `run_main.cpp` does), binds the
rank's device, and initializes Kokkos if it is not initialized; its destructor
finalizes only what it initialized. A program that already uses Kokkos or MPI
initializes them itself and still creates a `Session` (or none: `Solver`
checks that Kokkos is initialized). Kokkos initializes once per process: a
`Session` after Kokkos was finalized throws, which is why the GUI (below) runs
solvers in worker processes. Signal handling and the control socket stay with
the executable: a library must not take over its host's signals. The run's
control files (`STOP`) work as in the executable.

## ABI and stability policy

- **API version.** `MALLARD_API_VERSION` (header) and `mallard::api_version()`
  (library). Before Mallard 1.0, each minor release may change the API; the
  changelog lists every change under `changed` or `removed`, and the CMake
  package's version file accepts only the same minor version
  (`SameMinorVersion`), so a program is rebuilt against each release. From 1.0,
  the API is stable within a major version (additions only), with deprecations
  announced one minor release ahead.
- **ABI.** Public classes hold a `std::unique_ptr` to their implementation
  (pimpl) and public structs only standard types, so internals change without
  changing the ABI. The ABI is guaranteed only within a patch release before
  1.0; a program and the library must use the same C++ standard library. The
  library gets an SONAME with the API version at 1.0.
- **What is public.** Only `include/mallard/`. `src/` headers are internals,
  installed for no one; plugins (#336) see only `include/mallard/plugin.h`.

## The CMake package

`cmake --install` installs `lib/libmallard2d` (or `3d`), `include/mallard/`,
and `lib/cmake/Mallard/` with `MallardConfig.cmake`, `MallardConfigVersion.cmake`
and `MallardTargets<dim>d.cmake`. The 2D and 3D builds install into the same
prefix side by side, like the two executables; the config file includes the
targets of every dimension present:

```cmake
find_package(Mallard REQUIRED COMPONENTS 2d)   # or 3d, or both
target_link_libraries(app PRIVATE Mallard::mallard2d)
```

One header serves both (`mallard::dimension()` tells them apart at run time). A
program links one dimension; linking both in one process would load two copies
of the solver's globals and is refused at configure time in a later slice.

The library is the existing `libmallard2d`/`libmallard3d`, which already holds
the reader's C interface for the Python package: one shared library per
dimension holds all of Mallard (the solver links statically into it, position
independent), and its dependencies (Kokkos, MPI, HDF5) stay private, so a
program needs no Kokkos or MPI to compile or link. `Mallard_ENABLE_LIBRARY`
(on by default) builds it; `Mallard_ENABLE_PYTHON` needs it. The test suite
compiles the API's sources with the solver statically (`MallardApi`, an object
library), so one copy of the solver runs in the test process.

## Consumers

### The Python bindings (#345)

A nanobind extension module of `mallard-python` (`mallard._core`), built by the
same CMake build and linked against `libmallard<dim>d` only through this API:
`mallard.Case`, `mallard.Solver`, `mallard.Session` mirror the C++ classes; fields
become NumPy arrays on the host and DLPack / `__cuda_array_interface__` objects on
the device without copies; `mpi4py` communicators pass in as handles. The
reader's C interface then moves into the extension
([The Python package](python_package.md)). The bindings add no logic of their
own, so a Python run and a C++ run of the same case are the same run, which the
bindings' tests check bit for bit against `mallard -i`.

### The plugin API (#336)

Plugins are shared libraries built against the installed package
(`find_package(Mallard)`, target `Mallard::plugin2d`/`3d`, which adds
`include/mallard/plugin.h` and Kokkos' headers, since plugins run device code)
and named in the input (`plugins = ["libmybc.so"]`), so the TOML stays the
record. Each exports `mallard_plugin_api_version()`, checked at load against the
library's `MALLARD_API_VERSION`, and registers boundary conditions, source terms,
SGS models and output hooks into registries the setup consults by `type`
name, as it consults the built-in ones. The plugin's keys are read through the
same `Input` (a restricted view that records lookups), so unknown plugin keys
are errors and appear in `mallard check`. Plugins and in-process programs
share the package, the version policy and the callbacks' types: an output hook
is the plugin form of `on_output`.

### The GUI

The GUI (a Tauri shell around a Python server) drives Mallard today by writing
`input.toml` and running the CLI: `mallard check --format json`,
`mallard list keys`, `mallard input resolve` for validation and previews, and
`mallard -i` for runs, whose progress it follows by reading the run log
(`mallard.jsonl`). Through the Python bindings it uses this API instead, with
no Rust API:

1. **In process: validation and previews.** The server imports the bindings
   and validates the case it edits directly: `Case` from the edited text,
   then the check report (`mallard check`'s, as structs: errors with their key
   and line, unused keys with suggestions, memory and cost estimates) and the
   resolved input with each key's record from the key catalog (type, default,
   unit, accepted values), which the typed configs use as their schema. Mesh and
   setup previews (the mesh's geometry and zones, regions, boundary assignment,
   the initial state) come from a setup that stops before the run
   (`Solver::check`'s path), as arrays. These need Kokkos but no run; their
   setups are small, on the host backend, and nothing they do can corrupt a run.
2. **Runs in a worker process.** A run never runs in the server's process. The
   GUI writes `input.toml`, then starts a worker (`python -m mallard.worker
   input.toml`, a thin script over the bindings, or `mallard -i` itself), locally
   or through its SSH, Kubernetes and batch backends, under `mpirun` for MPI
   and on GPUs. A worker crash (a GPU fault, an abort in MPI, an out-of-memory
   kill) cannot take down the GUI; the GUI kills a worker with a signal (the
   executable's checkpoint-and-stop handling, or `mallard ctl`), and Kokkos and
   MPI initialize once per worker, which their once-per-process rule requires.
   The worker builds the run only from the file (`Case::from_toml_file`), so
   the file the GUI saved is what ran.
3. **Structured progress and state.** The worker streams the run as records,
   not text: `on_progress` gives the progress record (the run log's fields,
   with units from the start record), `monitors()` the monitored values,
   `on_output` the files written, and `fields()` the state for live views,
   downsampled or on a probe's points. The worker sends them to the GUI over a
   pipe or socket as the run log's JSON records, so the GUI has one parser
   whether it follows a worker or a run log on a cluster. Until those slices
   land, the first slice already allows this pattern: the worker drives
   `step()` itself and reads `time()`, `step_count()` and `monitors()` between
   steps.
4. **The TOML stays the record.** The worker runs the file on disk; provenance
   records its text; the run directory holds `input.toml`, the run log and the
   outputs, so any GUI run reruns with `mallard -i input.toml` and appears in
   the GUI's history like a run started from the command line.

## Alternatives rejected

- **Typed configs as a second setup path** (setup reads C++ structs; TOML
  parses into them). It would need every key read twice or a rewrite of setup,
  lose unused-key detection and line numbers for code-built cases, and let a
  code-built run have no TOML record. Emitting TOML keeps one path.
- **Exporting `Solver` and `src/` headers.** The `Solver` class has hundreds
  of members and changes with nearly every PR; exposing it would freeze it or
  break every program each release, and would force Kokkos, toml11 and
  Mallard's compile definitions (`Mallard_DIM`, `Mallard_USE_DOUBLE`) onto every
  consumer.
- **A static library package.** Consumers would have to link Kokkos, yaml-cpp,
  HDF5, CGNS, MPI and NCCL themselves with matching options, and the bundled
  yaml-cpp and Kokkos are not installed. One shared library with private
  dependencies needs none of that.
- **A separate shared library next to `libmallard<dim>d`.** The Python
  extension would then load two copies of the solver's globals (communicators,
  logging, provenance). One library per dimension holds everything.
- **One library for both dimensions.** The dimension is a compile-time
  constant throughout the solver (`N_DIM`); one library would compile everything
  twice under two namespaces. Two libraries match the two executables.
- **A C API first, C++ over it.** The Python bindings are the main second
  consumer and nanobind binds C++ directly; a C layer would add a translation of
  every type for no consumer. The reader's C interface goes away with the
  bindings.
- **The API owning Kokkos and MPI unconditionally.** Programs that couple
  Mallard with other Kokkos or MPI codes initialize them themselves; `Session`
  initializes only what is missing.
- **Running solvers in the GUI's server process.** Kokkos cannot reinitialize,
  a crash or abort would end the GUI, and cluster runs need a separate process
  anyway.
- **`step()` advancing to the next host stop** (the device queue's horizon)
  instead of one step. Faster on GPUs, but the step count of a call would depend
  on outputs and the backend. `advance_to(t)` and, later, `step(n)` give
  batched stepping with the same results.

## Slices

1. **This one:** `Session`, `Case::from_toml_file`/`from_toml`, `Solver` with
   `setup`, `step`, `advance_to`, `finish`, `run`, `time`, `step_count`,
   `monitors`; the CMake package; an example built against an install; tests
   that the API's runs match `mallard -i` bit for bit.
2. State access (`fields()`, geometry, `set_state` at step boundaries) and
   `step(n)`.
3. Callbacks (`on_step`, `on_progress`, `on_output`, `on_stage`) and the logging
   sink, with structured progress records.
4. Python bindings (nanobind) over 1-3, in `mallard-python`; the reader moves
   into the extension.
5. Typed configs generated from the key catalog, `Case::set`, and the check
   report and resolved input as structs; round-trip tests (code-built case to
   TOML to `mallard -i`, bit for bit).
6. Plugin API (#336) on the same package.
