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
installed Mallard. The design and the plans for the next steps (state access,
callbacks, Python bindings, typed configuration objects, plugins) are in
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
every rank. `monitors()` gives the same values on every rank.

## Compatibility

`MALLARD_API_VERSION` in the header and `mallard::api_version()` give the
version of the API. Until Mallard 1.0 it may change in any minor release (the
[changelog](../../CHANGELOG.md) lists each change), so rebuild your program for
each release: `find_package(Mallard)` accepts only an install of the minor
version it was given, if one is given.
