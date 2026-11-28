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

- **State access** (slice 2): `Solver::field(name, memory)` gives a view of
  the owned cells of a field as a `mallard::Field` (data pointer, scalar type,
  device and device ordinal, shape, strides in elements, unit): the
  conservative variables and partial densities (the state), the primitives
  (computed from the state when asked for), their blocks, the statistics
  averages and the geometry (`CENTROID`, `VOLUME`, `GLOBAL_ID`). With
  `Memory::solver` it is the solver's own memory, on the device in GPU builds,
  never copied; `Memory::host` copies to host memory only where the solver's
  memory is a device's. The state read has a reacting run's pending chemistry
  half step applied (`Solver::boundary_state`, which the next step undoes,
  bit for bit). A `Field` is exactly what DLPack and
  `__cuda_array_interface__` describe, so the bindings wrap it without copies.
  `writable_field` hands out the state's views for writing; `commit_state()`
  (or the next step) takes the state as a restart does
  (`Solver::state_set`: host copies written back, the chemistry's pending half
  step dropped, primitives, halo cells and the last valid state from it, the
  next step on the host side of the queue, kernel by kernel, which takes the
  queue's steps bit for bit), and the run log records an event `state_set`
  naming the fields. Setting the state is runtime control like `mallard ctl
  set`: the input stays the record of the configuration, the run log that of
  where the run departed from it.
- **Callbacks** (slice 2): `on_step(f)` after every step, `on_stage(f)` at
  each stage of the time integrator, before the stage's right-hand side.
  Neither runs inside captured work. The solver holds one hook of each
  (`set_step_hook`, `set_stage_hook`), empty without callbacks, so a run
  without them does nothing more per step. A step hook adds `step + 1` to the
  horizon, as host triggers do: the device takes one queued step at a time,
  still as captured CUDA graphs, and the hook runs after the host's
  `after_step`. A stage hook makes `device_steps_unsupported` give a reason,
  so steps run kernel by kernel on the host, where `take_step` wraps the
  right-hand side's function to call it; removing it brings the queue and the
  graphs back. Callbacks may not step or end the run (re-entrance throws).
  Record callbacks (slice 3): `on_progress(f)` gets each progress row as
  `mallard::Progress`, the run log's `progress` record as a struct (and its
  JSON), from a hook `print_progress` calls on every rank after it builds the
  record (`set_progress_hook`; with a hook it builds the whole record, as with a
  run log); `on_output(f)` and `on_record(f)` get the run log's output records
  and all its records from a hook in `run_log` (`set_record_hook`), on rank 0,
  which writes the log and alone knows every file. All three run at steps the
  host acts on anyway (progress rows and outputs are horizons of the device
  queue), so they keep the queue and the graphs and add nothing to a step; they
  get their record only, since the device may be ahead of it, and may not call
  the solver. `step(n)` is `advance_run(step + n, inf)`: one horizon, the
  steps queued as for `advance_to`.
- **Diagnostics**: statistics means (slice 2: `MEAN_`/`COV_` fields and
  `Solver::statistics()`, the averaging window) and their convergence, the run
  status (`RunStatus`, as `mallard ctl status` reports it), the check report
  (`mallard check`) and the resolved input with its key catalog records
  (`CheckReport::keys`: `resolve_input`'s `KeyRecord` of each key setup read,
  as `mallard check --catalog` writes them, with the key's value).
- **Logging sink** (slice 3): `Solver::set_log(sink)` sends what the solver
  prints (`logging::line` and the setup phases, line by line; warnings and
  errors as messages with their code) to a callback instead of stdout and
  stderr, without color; a sink that does nothing silences it. `logging` holds
  one sink and `run_log` one record hook for the process: each `Solver` call
  that runs the solver sets its own for the call and restores the previous
  ones after, so solvers in one process keep theirs.

Collective semantics: under MPI every call of `Solver` is collective, and
values (`monitors()`) are the same on every rank. A solver runs on
`MPI_COMM_WORLD`, as the executable does, or on a communicator the `Session`
takes (a pointer to an `MPI_Comm`, so the header needs no `mpi.h`; mpi4py's
`MPI._addressof` in Python). `comm::Session` then duplicates it and
`comm::communicator()`, which every collective and `SelfScope` go through,
returns the copy, so the solver's messages never match the program's.

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
  the lookups setup makes), so they cannot drift from what setup reads.
  `tools/dev/input_config.py` writes `include/mallard/config.h` and the
  Python module `tools/mallard/config.py` from the catalog and the key rows of
  `docs/input.md` (each setter's documentation); `precheck.sh` runs it with
  `--check`, so a catalog rebuild that changes a key fails until both are
  regenerated.

Slice 5 implements them as collectors, not a schema of their own. A class per
table (`config::Run`, `config::NumericsFaceReconstruction`) holds `(key, Value)`
pairs relative to its table in the order set (`config::Table`, hand-written in
`mallard.h`); a child table's setter takes the child by value and prefixes its
keys, so the same class serves a table wherever it sits. Top-level tables carry
their path, an element of a top-level array of tables its index (`Probes(1)`),
a named table its name (`Regions("jet")`); tables inside those have no path
and are placed only by their parent. `Case::set(table)` is one
`Case::set(key, value)` per pair, so every typed value goes through the same
text edit and setup checks it as a file's. Types come from the catalog: `bool`,
`std::int64_t`, `double`, `std::string_view`, `mallard::Quantity` for a key
with a dimension (a number in SI units or the key's documented unit, or
`"5 mm"`, which setup converts and checks), `Value` for keys of several types
(expressions, arrays, a region's name or inline table). The catalog does not
say which tables are arrays of tables; the generator takes those the catalog
types as arrays, the `[[...]]` headings of `docs/input.md`, and the tables an
input of the catalog writes as an array. Tables whose keys are all species
(`X`, `Y`) are maps: `.X("O2", 0.21)`. The catalog holds only keys some input
sets up, so a table's setters cover those keys; the parent's `Value` setter and
`Case::set(key, value)` cover the rest. Regions are a recursive grammar (the
parts of booleans are regions) found in many tables, which a catalog of key
paths cannot enumerate: every region is one class, `config::Region`, whose
keys are those of every region of the catalog (a region takes the same keys
wherever it is; setup checks them against its type and place), and
`tools/dev/catalog_inputs/{2d,3d}/regions.toml` set every region type with
each of its keys, so each key a region takes has a setter. Appending to an array of tables (`probes[n]` of `n`
probes, the index a typed element takes) is part of `set_value`: the element is
created and its `[[...]]` header written at the end of the text.

The check report (`Case::check()`) is `Solver::check` on the case's input, as
`mallard check` runs it, converted to public structs that mirror the sections
of its JSON (diagnostics, unused keys, boundaries, time step, resolution,
memory, cost, resolved input), with the whole report's text and JSON beside
them; an input error is a diagnostic of the report, as in `mallard check`.

`Case::set(key, value)` (slice 3) edits the text by key path, for every key
until the generated objects exist and for those they do not cover. It returns
a new `Case` (cases are values), so a sweep builds many cases from one. The
edit (`input_upgrade::set_value`, beside the upgrade's own rewrites) changes
only the key's line, or adds the key after its table's last key (a new table
at the end), and checks that the parsed result is the input with that key set;
where the line cannot change alone (a dotted key, a key of an inline table, a
value over several lines) it reformats the whole text, comments kept. Values
are `mallard::Value`: booleans, integers, floats written to read back exactly
(`float_text`), strings and arrays, or any TOML text. Keys are not checked
when set: setup checks them as it checks a file's, with the same messages.
`Case::save(path)` writes the text and returns the case of the file, for
programs whose runs build from the input on disk.

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
2. State access (`field()`, geometry, statistics, `writable_field` and
   `commit_state` at step boundaries) and callbacks (`on_step`, `on_stage`).
3. `on_progress`, `on_output`, `on_record` and the logging sink, with
   structured progress and output records; `step(n)`; `Case::set` and
   `Case::save`, with round-trip tests (code-built case to TOML to
   `mallard -i`, bit for bit).
4. Python bindings (nanobind) over 1-2, in `mallard-python`
   ([The Python package](python_package.md)), and `Case::dimension()`; then
   a `Session` on a communicator (mpi4py's in Python, `mallard-python[mpi]`),
   and the reader moved into the extension.
5. Typed configs generated from the key catalog over `Case::set`, and the
   check report (with the resolved input's text) as structs; then the resolved
   input key by key with its catalog records, and one `Region` class with
   every region key.
6. Plugin API (#336) on the same package.
