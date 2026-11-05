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
mallard -i input.toml --restart latest  # after a stop: resume from the newest restart file
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

## Shell completion

Tab completes `mallard`'s commands, their options and the values those take: input
files after `-i`, `latest` and `*.restart` files after `--restart`, template names
after `mallard new`, run directories and requests after `mallard ctl`, `2` or `3`
after `--dim`. An install puts the scripts where bash (with the bash-completion
package), zsh and fish look under its prefix:

| Shell | Installed as | Found without setup when |
|---|---|---|
| bash | `share/bash-completion/completions/mallard` | the prefix is `/usr` or `/usr/local`, or in `XDG_DATA_DIRS` |
| zsh | `share/zsh/site-functions/_mallard` | that directory is in `fpath` (it is for `/usr/local` on most systems) |
| fish | `share/fish/vendor_completions.d/mallard.fish` | the prefix is `/usr` or `/usr/local`, or in `XDG_DATA_DIRS` |

Otherwise, for a build tree, a Spack install or a prefix such as `~/mallard`,
load it from the shell's startup file; `mallard completion` prints it:

```sh
source <(mallard completion bash)     # ~/.bashrc
source <(mallard completion zsh)      # ~/.zshrc, after compinit
mallard completion fish | source      # ~/.config/fish/config.fish
```

The containers load the bash completion in every interactive shell. The script is
made from the same table as the help pages, so it completes exactly the commands of
the `mallard` that printed it; reload it after updating Mallard.

How the commands are laid out, and which are planned:
[design notes](../design/cli.md).

## Pages

<!-- BEGIN GENERATED pages -->

### mallard

```text
Usage: mallard -i input.toml [--restart FILE|latest] [Kokkos options]
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

--restart latest resumes a run from where it stopped, whatever stopped it: a job
script runs the same command for the first job and every continuation. It skips,
with a warning, a restart file that does not read completely, and fails if no
restart file does.

Commands:
  Run
    mallard -i input.toml  Run a case
    mallard ctl            Ask a running case for its status, a restart file or
                           a stop, or change its settings
    mallard reactor        Integrate a 0D reactor, or benchmark the chemistry
  Set up
    mallard new            Create a ready-to-run case from a template
    mallard input upgrade  Bring an input file from an older release up to date
    mallard input resolve  Write the input as a run uses it: every key explicit,
                           defaults and preset included
  Inspect
    mallard check          Set up a case without running it and report on it:
                           keys, mesh, time step, memory, cost
    mallard info           What a mesh, snapshot, restart, probe or CSV file, or
                           a run directory, holds
  Mesh
    mallard mesh convert   Write a mesh file, or the mesh of an input file, as a
                           Mallard HDF5 mesh
  Post
    mallard post extract   Probes, region integrals and snapshots of existing
                           output, as the run would have written them
  Maintain
    mallard test           Run the unit and regression tests
    mallard completion     Print the shell completion script for bash, zsh or
                           fish
    mallard --version      Print the version

Options:
  -i FILE                The input file (TOML). Required.
  --restart FILE|latest  Start from this restart file: replaces the input's
                         [initialize] with type = "restart". latest: the newest
                         complete restart file of the input's restart writers
                         (of the run that started last, then of the highest
                         step), or [initialize] if there is none. Default: as
                         [initialize] says.
  --kokkos-*             Kokkos options, e.g. --kokkos-num-threads=8 (threads
                         per rank) or --kokkos-device-id=1 (the GPU; by default
                         each rank gets its own).
  -h, --help             Show this page.

Options of mallard before any command:
  --dim 2|3  --dim 2 runs mallard2d, --dim 3 mallard3d, for any command.
             Default: the dimension of the case's mesh.

Examples:
  $ mallard -i input.toml
  $ mallard -i input.toml --kokkos-num-threads=8
  $ mpirun -n 4 mallard -i input.toml
  $ mallard -i input.toml --restart solut/restart_010000.restart
  $ mpirun -n 4 mallard -i input.toml --restart latest

Exit status:
  0  the run reached a stop condition (n_steps, t_stop, t_wall_stop)
  1  an error, divergence included (the divergence report says why)
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
  mallard input resolve    Write the input as a run uses it: every key explicit,
                           defaults and preset included

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

### mallard input resolve

```text
Usage: mallard input resolve input.toml > full.toml

Sets up the case as mallard check does and writes to standard output the input
the run uses: every key the setup reads, with the value it takes, whether the
input sets it, a numerics preset fills it or it is the default. Renamed keys are
under their new names, quantities with units are plain numbers in the units the
key documents, keys are sorted and comments dropped. Keys the setup does not
read are left out, and so are tables the input leaves out, whose presence can
turn a feature on.

The result runs bitwise the same as the input and resolves to itself. Every
output file's provenance holds the same text (input.resolved), so a run's
results carry the full input that made them.

Options:
  --restart FILE  The input of the run as started from this restart file.
  --kokkos-*      Kokkos options, e.g. --kokkos-num-threads=8 (threads per rank)
                  or --kokkos-device-id=1 (the GPU; by default each rank gets
                  its own).
  -h, --help      Show this page.

Examples:
  $ mallard input resolve input.toml > full.toml
  $ diff <(mallard input resolve a.toml) <(mallard input resolve b.toml)

Exit status:
  0  the resolved input was written
  1  errors in the input or the setup, as mallard check reports them
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-input-resolve
```

### mallard check

```text
Usage: mallard check input.toml [options]

Runs the whole setup of a case (mesh, partition, boundaries, numerics, initial
solution) without time stepping and without writing any file, then reports:
unknown or unused keys, boundary zones without a condition, mesh quality with
the worst cells, the stable time step and the cell that sets it, memory per rank
and, with --throughput or --steps, the cost. It lists every region the case uses
and the mesh's own (domain, boundary zones, cell zones) with its dimension,
measure and cells, and whether a surface or curve is exact or extracted; a
region that covers no cell is an error. Under mpirun -n N it checks the run on N
ranks.

Options:
  -i FILE                   The input file, instead of the argument.
  --restart FILE|latest     Check the run as started from this restart file, or
                            as --restart latest would.
  --ranks N                 Memory and cost per rank for a run on N ranks; from
                            a serial check, memory is scaled to the largest part
                            the run's partitioner would make. Default: the ranks
                            of this check.
  --throughput CELLS_PER_S  Cost (wall time and GPU-hours per unit simulated
                            time, per flow-through and to t_stop) at this many
                            cell updates per second per rank: the Throughput of
                            a run's summary over its ranks. Default: no cost.
  --steps N                 Cost at the throughput measured over N steps, after
                            one untimed step; the memory then includes the
                            steps' peak. Default: no cost.
  --memory SIZE|GPU         Size the run for this memory per rank: a size (40GB,
                            512MiB; host memory per rank in CPU builds) or a GPU
                            (a100-40, a100-80, h100-80, h200, mi250x, mi300x):
                            the most cells per rank, the fewest ranks the mesh
                            fits on and up to how many the halo stays below 10%.
                            Default: the GPU's memory in GPU builds, else none.
  --quality PREFIX          Write the mesh-quality fields, and those of the
                            resolution estimates (Y_PLUS, DELTA_OVER_ETA,
                            CELLS_PER_DELTA_L), to PREFIX_000000.vtkhdf
                            (PREFIX_000000.vtu in builds without HDF5), the
                            default snapshot format. Default: none.
  --catalog KEYS.json       Write every key the setup looked up, with its type,
                            unit, default and accepted values, as JSON. Default:
                            none.
  --view DIR                Write the case view to DIR/case.vtkhdf for ParaView:
                            every region the case uses (filled regions as their
                            cells with their fractions, surfaces and curves with
                            their normals or tangents and measures) and every
                            boundary zone, one block each (builds without HDF5:
                            DIR/case.vtm). Default: none.
  --kokkos-*                Kokkos options, e.g. --kokkos-num-threads=8 (threads
                            per rank) or --kokkos-device-id=1 (the GPU; by
                            default each rank gets its own).
  -h, --help                Show this page.

Examples:
  $ mallard check input.toml
  $ mallard check input.toml --ranks 64 --throughput 5e7
  $ mpirun -n 4 mallard check input.toml --quality quality
  $ mallard check input.toml --view view   # then open view/case.vtkhdf in ParaView

Exit status:
  0  no errors (there may be warnings)
  1  errors: in the input, the mesh or the setup
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-check
```

### mallard info

```text
Usage: mallard info FILE [--format text|json] [--input] [--quality PREFIX]
       mallard info RUN_DIR [--format text|json] [--input]

Reports what a file Mallard writes or reads holds, without an input file: its
kind, format and format version; the mesh (cells, points, cell types, bounds,
boundary and cell zones); its times and steps; its fields with their units,
descriptions and ranges (of the last snapshot); the species; and its provenance
(Mallard version and commit, build, when, where and on how many ranks it ran,
the input file and mesh with their SHA-256). Files without provenance (before
Mallard 0.9) or from a newer Mallard are reported with a note.

FILE is a snapshot (.vtkhdf, .vtu, .pvtu, .h5) or a series index
(.vtkhdf.series, .pvd, .xmf), a restart file (compressed too), a probe file
(.csv, .h5, .vtkhdf, .vtp, .pvd), a CSV file of forces or integrals, or a mesh
file (Gmsh or Fluent .msh, .cas, .cgns, an OpenFOAM case, a Mallard HDF5 mesh).
Output files are read by any build, of any dimension and precision; a mesh file
by the solver of its dimension, which also reports its quality as mallard check
does.

RUN_DIR is a directory a run wrote to: its start and end from mallard.jsonl, and
every output file in it and one level down, one table row each (the snapshots of
a series as their index).

Options:
  --format text|json  text: sections and tables, with the units of every
                      quantity; json: the same as one JSON object, for scripts.
                      Default: text.
  --input             Print the input file that produced the file, from its
                      provenance: the resolved input (every key the run read,
                      defaults filled in) for files of Mallard 0.10 and later,
                      the input as written for 0.9. For a run directory, its
                      run's.
  --quality PREFIX    Mesh files: write the mesh-quality fields to
                      PREFIX_000000.vtkhdf (PREFIX_000000.vtu in builds without
                      HDF5), as mallard check --quality does. Default: none.
  --kokkos-*          Kokkos options, e.g. --kokkos-num-threads=8 (threads per
                      rank) or --kokkos-device-id=1 (the GPU; by default each
                      rank gets its own).
  -h, --help          Show this page.

Examples:
  $ mallard info solut/flow.vtkhdf.series
  $ mallard info solut/restart_001000.restart --format json
  $ mallard info solut/flow_000100.vtkhdf --input > input.toml   # rerun it
  $ mallard info wing.msh --quality quality
  $ mallard info runs/cylinder

Exit status:
  0  the file was read
  1  a file cannot be read (missing, not Mallard's, newer), or --input finds no provenance
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-info
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

### mallard post

```text
Usage: mallard post <command> [options]

Commands on a run's output: probes, integrals and snapshots of it.

Commands:
  mallard post extract     Probes, region integrals and snapshots of existing
                           output, as the run would have written them

mallard post <command> --help shows the page of a command.

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-post
```

### mallard post extract

```text
Usage: mallard post extract FILE... [-i input.toml] [--probe NAMES] [-o DIR]
       mallard post extract FILE... --region REGION [--fields LIST] [--reductions LIST]
       mallard post extract FILE... --snapshot vtkhdf|vtu [--fields LIST] [--region REGION]

Sets up the case of a restart file, snapshot or series and writes probes of its
stored state, without rerunning: the input's [[probes]] (points, curves, cut
planes, surfaces with their fluxes, region integrals, means and extremes), a
region given with --region, or snapshots of chosen fields and regions. It runs
the run's own code, so a probe gives at each step what the run's probe gave
there: the same values for restart files and for snapshots of the conservative
variables, the same to rounding for snapshots of RHO, U and P.

FILE is a restart file, a snapshot of the whole mesh (.vtkhdf, the .h5 of XDMF
output) or a series of them (.vtkhdf.series, a series file, .xmf); several
files, or a glob, extract in the order given. The case comes from -i, else from
FILE's provenance, and its mesh must be the file's: run it where the run ran, so
that relative paths (the mesh file) resolve. --steps, --times, --every and
--latest choose the snapshots. VTKHDF files record no step: their snapshots are
numbered by their position in FILE.

Files go to the output directory under the names the input gives them, so a
run's files are never overwritten. The report lists each file with its times and
fields, and every reduction of the last snapshot with its units. Under mpirun -n
N the probes are evaluated on N ranks.

Options:
  -i FILE                The input file of the case: its mesh, physics,
                         numerics, [regions] and [[probes]]. Default: the input
                         in the provenance of FILE (Mallard 0.10 and later
                         record it resolved).
  --probe NAMES          The [[probes]] of the input to extract, by name,
                         separated by commas (--probe wake,box). Default: every
                         one, unless --region or --snapshot is given.
  --region REGION        Extract a region as a [[probes]] entry of --fields with
                         this region: the name of a region of the input's
                         [regions] or of the mesh (domain, a boundary or cell
                         zone), or an inline TOML table, a region ('{type =
                         "rectangle", center = [0.5, 0.5], size = [0.4, 0.2]}')
                         or a probe type ('{type = "line", start = [0, 0], end =
                         [1, 0], n_points = 50}'). Default: none.
  --name NAME            The name of the --region probe and of --snapshot:
                         probe_NAME.csv, probe_NAME_reductions.csv,
                         NAME_<step>.vtkhdf. Default: extract.
  --fields LIST          The variables of --region and --snapshot, separated by
                         commas, as [[probes]] and [[write_data]] take them
                         (RHO,U,P,T,MASS_FLUX). Default: RHO,U,P,T.
  --reductions LIST      The reductions of --region, separated by commas:
                         integral, mean, favre, rms, min, max. Default:
                         integral, mean, min and max of a region; none of a
                         probe type.
  --sampling cell|point  Whether --region samples cell averages or the
                         reconstruction's point values. Default: cell.
  --snapshot vtkhdf|vtu  Also write a snapshot of --fields (of --region, if
                         given) at each step extracted, as [[write_data]] writes
                         it: OUTPUT/NAME_<step>.vtkhdf, or .vtu files with a
                         .pvd index. Default: none.
  --steps LIST           Extract the snapshots of these steps, separated by
                         commas (--steps 100,200). Default: every snapshot.
  --times T|T1:T2        Extract the snapshot nearest to simulated time T, or
                         every one with T1 <= t <= T2, in seconds. Default:
                         every snapshot.
  --every N              Extract every Nth of the snapshots selected, starting
                         with the first. Default: 1.
  --latest               Extract only the last snapshot selected.
  -o, --output DIR       The directory the files go to, under the names the
                         input gives them (a probe's file and reductions_file).
                         Default: extract.
  --format text|json     text: sections and tables, with the units of every
                         quantity; json: the same as one JSON object, for
                         scripts. Default: text.
  --kokkos-*             Kokkos options, e.g. --kokkos-num-threads=8 (threads
                         per rank) or --kokkos-device-id=1 (the GPU; by default
                         each rank gets its own).
  -h, --help             Show this page.

Examples:
  $ mallard post extract solut/flow.vtkhdf.series      # every [[probes]] of the run
  $ mallard post extract solut/restart_*.restart --probe wake -o wake
  $ mallard post extract solut/flow.vtkhdf.series --latest --region domain --fields RHO,RHOE
  $ mallard post extract solut/flow.xmf --times 0.1:0.5 --fields MASS_FLUX --reductions integral \
        --region '{type = "cut_plane", point = [0.5, 0, 0], normal = [1, 0, 0]}'
  $ mallard post extract solut/flow.vtkhdf.series --latest --snapshot vtkhdf --fields P,T

Exit status:
  0  every snapshot was extracted
  1  a file cannot be read, the case cannot be set up, or the file is not of its mesh
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-post-extract
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

### mallard completion

```text
Usage: mallard completion bash|zsh|fish

Prints a script that makes the shell complete mallard's commands, their options
and the values those take: input files for -i, latest and restart files for
--restart, template names for mallard new, run directories for mallard ctl and
its requests. It is made from the same table as the help pages, so it knows
every command of this build; templates come from mallard new when completing.

An install (cmake --install, Spack, the containers) puts the scripts where the
shells look: share/bash-completion/completions/mallard,
share/zsh/site-functions/_mallard and
share/fish/vendor_completions.d/mallard.fish under its prefix. For a build tree,
or a prefix the shell does not search, load it from the shell's startup file, as
the examples show.

Options:
  -h, --help  Show this page.

Examples:
  $ source <(mallard completion bash)                  # in ~/.bashrc
  $ source <(mallard completion zsh)                   # in ~/.zshrc, after compinit
  $ mallard completion fish | source                   # in ~/.config/fish/config.fish
  $ mallard completion zsh > ~/.zfunc/_mallard         # or once, into a directory of fpath

Exit status:
  0  the script was printed
  2  usage error, or a shell other than bash, zsh and fish

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-completion
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
