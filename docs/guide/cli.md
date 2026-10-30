# Command line

Everything Mallard does goes through one command, `mallard`: `mallard -i input.toml`
runs a case, and subcommands create, check and control cases, convert meshes and
run the tests. `mallard --help` lists them by what they are for, and
`mallard <command> --help` (`-h` works too, anywhere in the arguments) shows a
command's page: what it does, every option with its default, examples and its
exit status. This page collects those pages as `mallard` prints them.

```sh
mallard new cylinder && cd cylinder     # a ready-to-run case
mallard check input.toml                # set it up without running: keys, mesh, time step, cost
mallard -i input.toml                   # run it
mallard ctl . status                    # from another terminal: how far it is
```

A few things hold for every command:

- **Exit status:** 0 on success, 1 on failure and 2 on a usage error (an unknown
  option or command, a missing argument), which prints one line naming the
  command's `--help`. A run stopped on request exits with 3.
- **Dimension:** `mallard` runs `mallard2d` or `mallard3d`, whichever the case
  needs, by its mesh; `--dim 2|3`, before or after the command, states it.
- **Parallel runs:** start `mallard` under `mpirun` (or `srun`); Kokkos options
  (`--kokkos-num-threads=8`, `--kokkos-device-id=1`) go after the command
  ([Running in parallel](parallel.md)).

How the commands are laid out, and which are planned:
[design notes](../design/cli.md).

## Pages

<!-- BEGIN GENERATED pages -->

### mallard

```text
Usage: mallard -i input.toml [--restart FILE] [Kokkos options]
       mallard <command> [options]

Mallard solves the compressible Euler, Navier-Stokes and reacting-flow equations
on unstructured meshes, in 2D and 3D. mallard -i input.toml runs the case an
input file describes; the commands below create, check, control and maintain
cases.

mallard runs mallard2d or mallard3d, by the dimension of the case: that of its
mesh file, or, for a generated mesh, 3D when [mesh] sets Nz or Lz. --dim 2|3
states it.

Parallel runs start under MPI, with one rank per GPU or per group of cores:
mpirun -n 4 mallard -i input.toml. Kokkos options set the threads and devices of
each rank; on OpenMP builds, so do OMP_NUM_THREADS, OMP_PROC_BIND and
OMP_PLACES.

Commands:
  Run
    mallard -i input.toml  Run a case
    mallard ctl            Ask a running case for its status, a restart file or
                           a stop, or change its settings
    mallard reactor        Integrate a 0D reactor, or benchmark the chemistry
  Set up
    mallard new            Create a ready-to-run case from a template
    mallard input upgrade  Bring an input file from an older release up to date
  Inspect
    mallard check          Set up a case without running it and report on it:
                           keys, mesh, time step, memory, cost
  Mesh
    mallard mesh convert   Write a mesh file, or the mesh of an input file, as a
                           Mallard HDF5 mesh
  Maintain
    mallard test           Run the unit and regression tests
    mallard --version      Print the version

Options:
  -i FILE         The input file (TOML). Required.
  --restart FILE  Start from this restart file: replaces the input's
                  [initialize] with type = "restart". Default: as [initialize]
                  says.
  --kokkos-*      Kokkos options, e.g. --kokkos-num-threads=8 (threads per rank)
                  or --kokkos-device-id=1 (the GPU; by default each rank gets
                  its own).
  -h, --help      Show this page.

Options of mallard before any command:
  --dim 2|3  --dim 2 runs mallard2d, --dim 3 mallard3d, for any command.
             Default: the dimension of the case's mesh.

Examples:
  $ mallard -i input.toml
  $ mallard -i input.toml --kokkos-num-threads=8
  $ mpirun -n 4 mallard -i input.toml
  $ mallard -i input.toml --restart solut/restart_010000.restart

Exit status:
  0  the run reached a stop condition (n_steps, t_stop, t_wall_stop)
  1  an error, divergence included
  2  usage error: an unknown option or command, no -i
  3  stopped on request (signal, mallard ctl stop, STOP file), restart written

mallard <command> --help shows the page of a command.

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard
```

### mallard ctl

```text
Usage: mallard ctl <run-dir> status | checkpoint | stop
       mallard ctl <run-dir> set <key>=<value> [<key>=<value> ...]

Sends a request to the run started in run-dir, through its control socket
mallard.sock, and prints the reply. It must run on the node of the run's rank 0.

status: step, t, dt, CFL number, progress, wall time per step, cells per second,
ETA and memory. checkpoint: write a restart file now; the reply names it. stop:
write a restart file and end the run (exit status 3). set: change settings from
the next step on: n_steps, t_stop, t_wall_stop (0 removes a limit; one must
remain; times in seconds or with a unit, e.g. t_wall_stop=2h), check_interval,
write_data.<prefix or index>.interval or .time_interval, probes.<name>.interval
or .time_interval, forces.<zone>.interval, integrals.interval. Other keys are
refused.

Options:
  -h, --help  Show this page.

Examples:
  $ mallard ctl runs/cylinder status
  $ mallard ctl runs/cylinder set t_stop=3.0 write_data.flow.time_interval=0.05
  $ mallard ctl . stop

Exit status:
  0  the run carried out the request
  1  no run answers in run-dir, or the run refused the request
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-ctl
```

### mallard reactor

```text
Usage: mallard reactor -i input.toml [Kokkos options]

Integrates the adiabatic constant-volume reactor of the input's [reactor] table
(initial T, p and composition, end_time) with the solver's chemistry kernels,
and writes its temperature, pressure and mass fractions as CSV to reactor.output
(default reactor.csv). With a [benchmark] table, it advances states sampled
along the reactor's trajectory, replicated over many cells, and writes cells per
second to benchmark.output (default benchmark.csv). Runs on one rank.

Options:
  -i FILE     The input file, with a [reactor] table, or a [benchmark] table
              too. Required.
  --kokkos-*  Kokkos options, e.g. --kokkos-num-threads=8 (threads per rank) or
              --kokkos-device-id=1 (the GPU; by default each rank gets its own).
  -h, --help  Show this page.

Examples:
  $ mallard reactor -i input.toml
  $ mallard reactor -i benchmark.toml --kokkos-num-threads=8

Exit status:
  0  done
  1  an error in the input or the integration, or more than one rank
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-reactor
```

### mallard new

```text
Usage: mallard new <template> [directory] [--quick] [--force]
       mallard new --list

Creates a ready-to-run case from one of Mallard's examples or tutorials:
input.toml, run.sh, the mesh or initial-state script it needs, post.sh and a
README. The directory defaults to the template's name. Without a template, lists
them.

Options:
  -l, --list  List the templates, with what each shows and what it costs.
  --quick     A coarse, short version that runs in seconds to a minute: for
              trying the steps, not for the physical result.
  --force     Write into a directory that is not empty.
  -h, --help  Show this page.

Examples:
  $ mallard new --list
  $ mallard new cylinder
  $ mallard new cylinder runs/cyl --quick

Exit status:
  0  the case was created, or --list
  1  the case could not be written
  2  usage error, no template given, or no template of that name

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-new
```

### mallard input

```text
Usage: mallard input <command> [options]

Commands on input files.

Commands:
  mallard input upgrade    Bring an input file from an older release up to date

mallard input <command> --help shows the page of a command.

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-input
```

### mallard input upgrade

```text
Usage: mallard input upgrade old.toml > new.toml
       mallard input upgrade --check old.toml

Rewrites the renamed, moved and removed keys and options of an input file for
Mallard <version>, keeping its comments and layout, and writes it to standard
output; the changes go to standard error. A renamed key keeps its value and
comment, a moved key goes to the end of its new table, and a removed key is
commented out with a note. A file with nothing to change comes back byte for
byte; one with a renamed, moved or removed dotted key or key of an inline table
is reformatted, comments kept; a renamed option changes in place, also inside
inline tables.

Some renames depend on the case's dimension, which the command reads from the
input's mesh as a run does; when the mesh is not at hand, --dim states it.

Renamed keys still work, with a warning, for at least one minor release before
they are removed. mallard check lists deprecated keys as warnings and removed
ones as errors.

Options:
  --check     Write nothing: list the changes, with exit status 1 if there are
              any.
  --dim 2|3   The case's dimension, for renames that depend on it. Default: from
              the input's mesh, as a run finds it.
  -h, --help  Show this page.

Examples:
  $ mallard input upgrade old.toml > new.toml
  $ mallard input upgrade --check input.toml
  $ mallard input upgrade --dim 3 old.toml > new.toml

Exit status:
  0  the file was written (with --check: it is up to date)
  1  the file cannot be read or parsed (with --check: or it is out of date)
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-input-upgrade
```

### mallard check

```text
Usage: mallard check input.toml [options]

Runs the whole setup of a case (mesh, partition, boundaries, numerics, initial
solution) without time stepping and without writing any file, then reports:
unknown or unused keys, boundary zones without a condition, mesh quality with
the worst cells, the stable time step and the cell that sets it, memory per rank
and, with --throughput or --steps, the cost. Under mpirun -n N it checks the run
on N ranks.

Options:
  -i FILE                   The input file, instead of the argument.
  --restart FILE            Check the run as started from this restart file.
  --ranks N                 Memory and cost per rank for a run on N ranks; from
                            a serial check, memory is scaled to the largest part
                            the run's partitioner would make. Default: the ranks
                            of this check.
  --throughput CELLS_PER_S  Cost (wall time and GPU-hours per unit simulated
                            time, per flow-through and to t_stop) at this many
                            cell updates per second per rank: the Throughput of
                            a run's summary over its ranks. Default: no cost.
  --steps N                 Cost at the throughput measured over N steps, after
                            one untimed step. Default: no cost.
  --quality PREFIX          Write the mesh-quality fields to
                            PREFIX_000000.vtkhdf (PREFIX_000000.vtu in builds
                            without HDF5), the default snapshot format. Default:
                            none.
  --catalog KEYS.json       Write every key the setup looked up, with its type,
                            unit, default and accepted values, as JSON. Default:
                            none.
  --kokkos-*                Kokkos options, e.g. --kokkos-num-threads=8 (threads
                            per rank) or --kokkos-device-id=1 (the GPU; by
                            default each rank gets its own).
  -h, --help                Show this page.

Examples:
  $ mallard check input.toml
  $ mallard check input.toml --ranks 64 --throughput 5e7
  $ mpirun -n 4 mallard check input.toml --quality quality

Exit status:
  0  no errors (there may be warnings)
  1  errors: in the input, the mesh or the setup
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-check
```

### mallard mesh

```text
Usage: mallard mesh <command> [options]

Commands on meshes.

Commands:
  mallard mesh convert     Write a mesh file, or the mesh of an input file, as a
                           Mallard HDF5 mesh

mallard mesh <command> --help shows the page of a command.

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-mesh
```

### mallard mesh convert

```text
Usage: mallard mesh convert INPUT OUTPUT.h5

Writes a mesh as a Mallard HDF5 mesh file, which large runs read in parallel.
INPUT is a mesh file (Gmsh or Fluent .msh, Fluent .cas, CGNS .cgns, OpenFOAM
polyMesh directory), or an input file (.toml) whose [mesh] table describes the
mesh (with its [mesh.transform]). Under MPI, the ranks write in parallel (needs
parallel HDF5); each reads the whole mesh file but generates only its block.
Needs a build with HDF5.

Options:
  -h, --help  Show this page.

Examples:
  $ mallard mesh convert wing.msh wing.h5
  $ mpirun -n 8 mallard mesh convert input.toml mesh.h5

Exit status:
  0  written
  1  an error reading or writing, or a build without HDF5
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-mesh-convert
```

### mallard test

```text
Usage: mallard test [--dim 2|3] [GoogleTest options] [Kokkos options]

Runs Mallard's test suites, mallard-test2d and mallard-test3d, one after the
other: those next to mallard (an install tree) or in the test directory of its
build tree. Multi-rank tests run under MPI: mpirun -n 3 mallard test
--gtest_filter='MPITest.*'.

Options:
  --dim 2|3                Run only the 2D or the 3D suite. Default: every suite
                           there is.
  --gtest_filter=PATTERN   Tests by name: --gtest_filter='Sod*:MPI*-*Slow*'.
  --gtest_list_tests       List the tests, run none: --gtest_list_tests.
  --gtest_repeat=N         Run N times: --gtest_repeat=50.
  --gtest_fail_fast        Stop at the first failure: --gtest_fail_fast.
  --gtest_brief            Print only failures and the summary: --gtest_brief=1.
  --gtest_output=xml:FILE  A JUnit report too: --gtest_output=xml:tests.xml.
  --gtest_*                Other --gtest_* options are passed to GoogleTest.
  --kokkos-*               Kokkos options, e.g. --kokkos-num-threads=8 (threads
                           per rank) or --kokkos-device-id=1 (the GPU; by
                           default each rank gets its own).
  -h, --help               Show this page.

Examples:
  $ mallard test
  $ mallard test --dim 3 --gtest_filter='Sod*' --kokkos-num-threads=2
  $ mallard test --gtest_list_tests

Exit status:
  0  every test passed
  1  a test failed, or there is no test suite
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-test
```

### mallard --version

```text
Usage: mallard --version

Prints the version of this build: the git description of the commit it was built
from, else the release.

Options:
  -h, --help  Show this page.

Examples:
  $ mallard --version

Exit status:
  0  always

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-version
```

<!-- END GENERATED pages -->
