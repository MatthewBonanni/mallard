# C++ API

Mallard is also a library: a C++ program can run a case itself, step by step,
instead of starting `mallard -i input.toml`. Use it to couple Mallard with
another code, to run your own control logic between steps (an optimization
loop, a controller acting on a probe), or to embed a solver in a larger
application. The run is the same run: the same setup, the same output files,
run log and restart files, and the same results bit for bit as `mallard -i` on
the same input.

```cpp
#include <iostream>
#include <mallard/mallard.h>

int main(int argc, char ** argv) {
    mallard::Session session(argc, argv);  // MPI and Kokkos, for the whole program
    mallard::Solver solver(mallard::Case::from_toml_file("input.toml"));
    solver.setup();
    while (solver.advance_to(solver.time() + 0.01)) {
        if (auto p = solver.monitor("wake_mean_P")) std::cout << solver.time() << " " << p->value << "\n";
    }
    return solver.finish();
}
```

The [example](../../examples/cpp_api) builds a program like this against an
installed Mallard. The design and the plans for the next steps (Python
bindings, typed configuration objects, plugins) are in
[The C++ library API](../design/cpp_api.md).

## Building against Mallard

`cmake --install` installs the library (`lib/libmallard2d`, or `libmallard3d`
from a 3D build), the header `include/mallard/mallard.h` and a CMake package.
Install both dimensions into one prefix if you need both. In your project:

```cmake
find_package(Mallard REQUIRED COMPONENTS 2d)
target_link_libraries(my_program PRIVATE Mallard::mallard2d)
```

and configure with `-DCMAKE_PREFIX_PATH=<Mallard's install prefix>`. The
library carries Kokkos, MPI and the rest inside it: your program needs only a
C++17 compiler, not Kokkos or MPI. The header is the same for both
dimensions; `mallard::dimension()` says which one a program linked. A build
with `-DMallard_ENABLE_LIBRARY=OFF` leaves the library out.

## Running a case

- **`mallard::Session`** initializes MPI and Kokkos, and finalizes them when it
  ends; create one at the start of `main` and keep it for the whole program.
  It takes the program's options (`--kokkos-num-threads=N`,
  `--kokkos-device-id=N`) as `mallard` does, and leaves the others in `argv`.
  If your program initializes Kokkos or MPI itself, the session uses them as
  they are. Kokkos initializes once per process: run a second set of cases in
  the same `Session`, not a new one.
- **`mallard::Case`** is the input: `Case::from_toml_file("input.toml")`, or
  `Case::from_toml(text)` for an input your program writes. The input is the
  run's record: the run log and every output file record its exact text, so
  `mallard -i` on it reruns the run. Paths inside it are relative to the
  working directory, as for `mallard -i`.
- **`mallard::Solver`** runs it:

| Call | What it does |
|---|---|
| `setup()` | The setup of `mallard -i`: mesh, physics, numerics, output, the initial state, then the start of the run (its first output and progress row). The other calls do it first if you do not |
| `step()` | One step. Returns `false` once the case's stop condition (`[run]` `n_steps` or `t_stop`, a [trigger](running.md#triggers), a [stop request](running.md)) is reached |
| `step(n)` | `n` steps, as `n` calls of `step()` take them but queued on the GPU ahead of the host; returns as `step()` does, stopping early at the stop condition |
| `advance_to(t)` | Steps until the time is `t`, the last step landing exactly on it as on `t_stop`; returns as `step()` does |
| `finish()` | Ends the run: the final output, the summary and the run log's end record; returns the exit status `mallard -i` would. Before the stop condition, the run ends as stopped by the program (`outcome = "stopped"`, `cause = "caller"`) |
| `run()` | The whole run of `mallard -i`: steps to the stop condition, then `finish()` |
| `time()`, `step_count()`, `finished()` | Where the run is |
| `monitors()`, `monitor(name)` | The latest value of every `[integrals]` column, `[[forces]]` component and `[[probes]]` value, by the name [triggers](running.md#triggers) give it (`integrals_kinetic_energy`, `forces_cylinder_Fx_pressure`, `wake_mean_P`), with its unit and the step it was written at |

The case still needs a stop condition (`n_steps` or `t_stop`), which ends the
run whatever the program asks; set it to the longest run you want.
`advance_to` adds landing times, so a run stepped with it takes different steps
from one with only `t_stop`, as an output time does; `step()` alone takes
exactly the steps of `mallard -i`.

## Building a case in code

`case.set(key, value)` returns the case with one key set: its TOML text with
that key's line changed, or the key added after its table's last key (a new
table, with its keys, at the end). The rest of the text stays as it was, so
`toml()` is still the whole input, which the run records and `mallard -i`
reruns bit for bit. Keys are paths of bare keys, an element of an array of
tables by its index; values are booleans, integers, floats (written so they
read back exactly), strings and arrays of them (`{0.5, 0.5}`, a
`std::vector`), or `mallard::Value::toml("...")` for any other TOML value. Keys
are checked when the run is set up, as a file's are: a misspelled key is an
error. `case.save(path)` writes the text to a file and returns the case of that
file, for runs that should build from the input on disk.

```cpp
const mallard::Case base = mallard::Case::from_toml_file("input.toml");
for (const double cfl : {0.3, 0.5, 0.7}) {
    const std::string dir = "cfl_" + std::to_string(cfl);
    std::filesystem::create_directories(dir);
    const mallard::Case c = base.set("run.cfl", cfl)
                                .set("write_data[0].prefix", dir + "/flow")
                                .set("output.run_log", dir + "/mallard.jsonl");
    mallard::Solver solver(c.save(dir + "/input.toml"));
    solver.run();
}
```

A key the line cannot hold alone (a dotted key, a key of an inline table, a
value written over several lines) has the whole text reformatted, comments
kept. An index equal to its array's length appends an element: `probes[2]` of a
case with two probes adds a `[[probes]]` table at the end.

### Typed configs

`mallard/config.h` has a class per table of the input, generated from the key
catalog (the keys setup reads, `docs/input-keys/`), with a setter per key, typed
and documented as [the input reference](../input.md) documents the key.
`case.set(table)` sets every key the table holds, in order, through
`case.set(key, value)`: a typed config writes TOML, which setup then reads and
checks as a file's, so the run still builds from the TOML and records it.

```cpp
#include <mallard/config.h>
namespace config = mallard::config;

const mallard::Case c = mallard::Case::from_toml_file("input.toml")
    .set(config::Run().cfl(0.5).t_stop(mallard::Quantity(2.0, "ms")))
    .set(config::Numerics().face_reconstruction(config::NumericsFaceReconstruction().type("TENO").order(5)))
    .set(config::Probes(1).name("wake").fields({"P"}).point({0.5, 0.1}).file("wake.csv"));
```

- **Classes.** A table's class is its path in CamelCase: `[run]` is
  `config::Run`, `[numerics.face_reconstruction]` is
  `config::NumericsFaceReconstruction`. An element of an array of tables takes
  its index (`config::Probes(1)`, `config::WriteData(0)`; the array's length
  appends one), a named table its name (`config::Regions("jet")`). A table
  inside an array of tables or a named table is set through its parent's
  setter (`config::Probes(0).region(config::ProbesRegion().type("circle"))`),
  an array of tables inside one through a `std::vector` of its elements.
- **Values.** Setters take `bool`, `std::int64_t`, `double` or
  `std::string_view` as the key's type is; a key with a physical dimension
  takes a `mallard::Quantity`: a plain number in SI units (or the unit the key
  documents), or `Quantity(5.0, "mm")`, written `"5 mm"` and converted by setup
  ([units](input_files.md#units)). Keys without one type (expressions,
  arrays, a region's name or inline table) take a `mallard::Value`; a species
  table (`X`, `Y`) takes each species and its value (`.X("O2", 0.21)`).
- **Coverage.** The catalog has the keys some input of the repository sets up,
  so a key the catalog lacks for a table (a region's parameters of another
  type) has no setter: set it with `Value::toml("{...}")` on the parent's
  setter, or with `case.set(key, value)`. `tools/dev/input_config.py`
  regenerates the header from the catalog, and CI fails when it is out of date.

## Checking a case

`case.check()` is [`mallard check`](input_files.md#checking-an-input) on the
case, as a `mallard::CheckReport`: the setup without the run (no files are
written), then `status` (`"ok"`, `"warning"`, `"error"`), `ok`,
`diagnostics` (each with its `level`, `code`, `message` and `docs`),
`unused_keys` (each `key`, `line` and `suggestion`), `boundaries`,
`time_step`, `resolution` (the estimates of y+, cells per flame thickness,
...), `memory` and `cost`, the `resolved` input (every key explicit), and the
whole report as `mallard check` prints it (`text`) and as
`mallard check --format json` writes it (`json`). An invalid input gives a
report of its errors, not an exception. `CheckOptions` takes `mallard check`'s
`ranks`, `throughput`, `steps` and `memory`, and a `log` sink for the setup's
console output. It needs a `Session`, and is collective under MPI.

```cpp
const mallard::CheckReport r = c.check();
for (const mallard::CheckUnusedKey & k : r.unused_keys) std::cerr << k.key << ": did you mean " << k.suggestion << "\n";
if (r.ok) std::cout << "dt " << r.time_step->dt << " s, " << r.cost->wall_per_time << " wall s per s\n";
```

## Fields

Between steps, `field(name)` gives a view of a field on the rank's owned
cells, in the solver's memory: on a GPU it points to device memory, without a
copy, so a CUDA or HIP code reads it in place; `field(name, mallard::Memory::host)`
gives it in host memory (the solver's own on a CPU build, a copy on a GPU one).
A `mallard::Field` describes the array as DLPack and `__cuda_array_interface__`
do: `data`, `type` (`float64`, or `float32` in a single-precision build),
`device` (`host`, `cuda`, `hip`, `sycl`) and `device_id`, `shape` and
`strides` (in elements), and the field's `unit`.

| Name | Shape | What |
|---|---|---|
| `RHO`, `RHOU_X`, ..., `RHOE` | cells | Conservative variables: the state |
| `RHOY_<species>` | cells | Partial densities of a mixture's species: the state too |
| `U_X`, ..., `P`, `T`, `H` | cells | Primitive variables, computed from the state when asked for |
| `conservatives`, `species`, `primitives` | cells x variables | The blocks the variables above are columns of |
| `MEAN_<field>`, `COV_<a>_<b>` | cells | The [statistics](statistics.md) averages so far |
| `CENTROID`, `VOLUME`, `GLOBAL_ID` | cells (x dimensions) | Cell centroids, volumes (areas in 2D) and global ids (the order of the output files) |

`field_names()` lists those of the case. A view stays valid until the next
call that steps or ends the run. The state is that at the end of the last
step; in a reacting run, with the chemistry's pending half step applied, as in
the output files.

To set the state, take `writable_field(name)` of a conservative variable,
partial density or block, write it, and call `commit_state()` (the next step
commits it if you do not). The next step starts from that state as from a
restart: primitives, halo cells and the time step are computed from it, and the
run log records an event `state_set` naming the fields set, so the record of
the run says where it departed from its input. Time and step count are not
part of the state, and the output written at setup (the start's snapshot) holds
the initial state of the input.

```cpp
solver.setup();
mallard::Field rho = solver.writable_field("RHO", mallard::Memory::host);
auto * values = static_cast<double *>(rho.data);
for (int64_t i = 0; i < rho.shape[0]; i++) values[i * rho.strides[0]] *= 1.01;
solver.commit_state();
```

## Callbacks

`on_step(f)` calls `f(solver)` after every step, `on_stage(f)` calls
`f(solver, stage)` at each stage of the time integrator, before the stage's
right-hand side is evaluated (`stage.step`, `stage.index` from 0, and
`stage.time`, `t_n + c_i dt`). Both return an id for `remove_callback(id)`.
A per-step callback may read fields and monitors and set the state; a per-stage
callback reads the stage's state through `field()` (conservative variables,
partial densities and the geometry). Neither may step or end the run.

Callbacks cost nothing while none is registered. With a per-step callback, the
GPU takes steps one at a time instead of queuing them ahead of the host (still
as captured CUDA graphs), as with a trigger the host evaluates; with a
per-stage callback, steps run kernel by kernel on the host side, so register
one only for the steps that need it. Neither changes the results: a run with
callbacks that only read takes the steps of `mallard -i` bit for bit.

```cpp
solver.on_step([](mallard::Solver & s) {
    if (auto p = s.monitor("wake_mean_P"); p && p->value > 2e5) std::cout << "high pressure at " << s.time() << "\n";
});
```

### Progress, output and the run log

`on_progress(f)` calls `f(progress)` with each row of the progress table: the
run log's progress record (`step`, `time`, `dt`, `cfl`, `fraction` of the run
done, `wall`, `wall_per_step`, `cells_per_second`, `eta`, the row's
`diagnostics` such as `min_rho` and `max_mach`, the files `written` since the
last row) and its `json`. `on_output(f)` calls `f(output)` with each file or
series completely written (`kind`, `path`, `reason`, `writer`, the `files` with
their sizes, `json`), and `on_record(f)` calls `f(record)` with every record of
the [run log](output.md#run-record-mallardjsonl) as it is written (`type`, `json`), whether or
not the case writes a run log file. These run where the host acts anyway, so
they keep the GPU's step queue and CUDA graphs; they get their record only and
may not call the solver. Progress rows come every `[output] check_interval`
steps and at the start and end of the run.

```cpp
solver.on_progress([](const mallard::Progress & p) { std::cout << p.step << " " << p.fraction << "\n"; });
solver.on_output([](const mallard::Output & o) { std::cout << "wrote " << o.path << "\n"; });
```

## Console output

`set_log(sink)` sends what the solver prints (the setup, the progress table,
the summary, warnings and errors) to `sink` instead of stdout and stderr: each
line, and each warning and error with its diagnostic code, as a
`mallard::LogMessage` (`level`, `code`, `text`), without color. A sink that
does nothing silences the solver; an empty one (the default) prints again. The
run log and output files are the same either way.

```cpp
std::ofstream log("solver.log");
solver.set_log([&](const mallard::LogMessage & m) { log << m.text << "\n"; });
```

## Statistics

`statistics()` gives the [statistics](statistics.md) means' averaging so far:
`samples`, the window from `start` to `end`, and the closed `batches` of their
confidence intervals; the averages themselves are the fields `MEAN_<field>`
and `COV_<a>_<b>`. The latest integrals and probe values are the
[monitors](#running-a-case).

## Errors

Every failure throws `mallard::Error`: `what()` is the message `mallard`
prints, with the fix, and `code()` its [diagnostic code](running.md#warnings-and-errors)
(`E-INPUT`, `E-KEY-UNUSED`, `E-DIVERGED`, ...). An input is checked as its file
would be, so a misspelled or unused key is an error, with suggestions.

```cpp
try {
    solver.setup();
} catch (const mallard::Error & e) {
    std::cerr << "error " << e.code() << ": " << e.what() << "\n";
}
```

## MPI

Build Mallard with `-DMallard_ENABLE_MPI=ON` and start the program with
`mpirun` as you would `mallard`. Every call of `Solver` is collective: make it on
every rank. `monitors()` gives the same values on every rank; fields hold each
rank's owned cells, which `GLOBAL_ID` places in the whole mesh. Callbacks run on
every rank, but `on_output` and `on_record` run on rank 0, which writes the run
log and knows every file. Rank 0 prints the console output, and every rank its
errors, to its sink.

To run on some of the processes only, for instance beside another code, initialize
MPI in your program and give the session a communicator:
`mallard::Session session(argc, argv, &comm);` with `comm` an `MPI_Comm`. The
solvers then communicate on a copy of it, never on `MPI_COMM_WORLD`, and every
process of it creates the session and runs the same cases.

## Compatibility

`MALLARD_API_VERSION` in the header and `mallard::api_version()` give the
version of the API. Until Mallard 1.0 it may change in any minor release (the
[changelog](../../CHANGELOG.md) lists each change), so rebuild your program for
each release: `find_package(Mallard)` accepts only an install of the minor
version it was given, if one is given.
