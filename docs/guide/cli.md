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
  option or command, a missing argument), which names the command's `--help`.
  A run stopped on request exits with 3.
- **Errors and warnings** go to stderr with a stable code and the docs that
  explain it (`error E-USAGE: ...`, [Warnings and errors](running.md#warnings-and-errors)).
- **Dimension:** `mallard` runs `mallard2d` or `mallard3d`, whichever the case
  needs, by its mesh; `--dim 2|3`, before or after the command, states it.
- **Parallel runs:** start `mallard` under `mpirun` (or `srun`); Kokkos options
  (`--kokkos-num-threads=8`, `--kokkos-device-id=1`) go after the command
  ([Running in parallel](parallel.md)).
- **From Python:** the commands with `--format json` (`info`, `diff`, `check`,
  `list`, `post extract`, `post convergence`) are functions of the
  [Python package](python.md), `mallard.info(path)` and `mallard.post.extract(files, latest=True)`,
  which run the command and return its JSON as a dict.

## Shell completion

Tab completes `mallard`'s commands, their options and the values those take: input
files after `-i`, `latest` and `*.restart` files after `--restart`, template names
after `mallard new`, run directories and requests after `mallard ctl`, `2` or `3`
after `--dim`, sections and keys (`numerics.riemann_solver`) after `mallard list
keys`, and output fields after `--fields`, each item of the comma-separated list
(`--fields RHO,U,<Tab>`). An install puts the scripts where bash (with the
bash-completion package), zsh and fish look under its prefix:

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
made from the same table as the help pages and from the catalogs `mallard list keys`
and `mallard list fields` print, so it completes exactly the commands, keys and
fields of the `mallard` that printed it; reload it after updating Mallard.

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
    mallard -i input.toml     Run a case
    mallard ctl               Ask a running case for its status, a restart file
                              or a stop, or change its settings
    mallard reactor           Integrate a 0D reactor, or benchmark the chemistry
  Set up
    mallard new               Create a ready-to-run case from a template
    mallard input upgrade     Bring an input file from an older release up to
                              date
    mallard input resolve     Write the input as a run uses it: every key
                              explicit, defaults and preset included
  Inspect
    mallard check             Set up a case without running it and report on it:
                              keys, mesh, time step, memory, cost, environment
    mallard info              What a mesh, snapshot, restart, probe or CSV file,
                              or a run directory, holds
    mallard list fields       The output fields there are, or exactly those of
                              one case
    mallard list species      The species of a case
    mallard list keys         The input keys there are, or exactly those of one
                              case: type, units, default, values
    mallard list templates    The case templates of mallard new
    mallard diff              Compare two solutions field by field: norms of the
                              difference, where it is largest, bitwise
  Mesh
    mallard mesh convert      Write a mesh file, or the mesh of an input file,
                              as a Mallard HDF5 mesh
  Post
    mallard post extract      Probes, region integrals and snapshots of existing
                              output, as the run would have written them
    mallard post derive       Derived fields of existing output (MACH, P0,
                              Q_CRITERION, ...), as the run would have written
                              them
    mallard post average      Means, RMS, covariances, PDFs and conditional
                              statistics over saved snapshots, as [statistics]
                              keeps them
    mallard post convergence  Order of accuracy, extrapolated value and grid
                              convergence index (GCI) of runs at several
                              resolutions
  Share
    mallard pack              One archive that reproduces a run, with a manifest
                              of hashes
  Maintain
    mallard test              Run the unit and regression tests
    mallard completion        Print the shell completion script for bash, zsh or
                              fish
    mallard --version         Print the version

Options:
  -i FILE                  The input file (TOML). Required.
  --restart FILE|latest    Start from this restart file: replaces the input's
                           [initialize] with type = "restart". latest: the
                           newest complete restart file of the input's restart
                           writers (of the run that started last, then of the
                           highest step), or [initialize] if there is none.
                           Default: as [initialize] says.
  --verbose                A line for every file written, between the progress
                           rows. Default: the progress table's written column
                           counts them, and mallard.jsonl and the summary name
                           them.
  --log-format text|jsonl  text: the setup, the progress table and the summary
                           on stdout; jsonl: on stdout only the run record's
                           progress records, one JSON object per line as
                           mallard.jsonl has them, for live monitoring and
                           plotting, and the text on stderr. Default: text.
  --kokkos-*               Kokkos options, e.g. --kokkos-num-threads=8 (threads
                           per rank) or --kokkos-device-id=1 (the GPU; by
                           default each rank gets its own).
  -h, --help               Show this page.

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
       mallard list templates

Creates a ready-to-run case from one of Mallard's examples or tutorials:
input.toml, run.sh, the mesh or initial-state script it needs, post.sh and a
README. The directory defaults to the template's name. Without a template, lists
them, as mallard list templates does.

Options:
  -l, --list  Deprecated: mallard list templates. List the templates.
  --quick     A coarse, short version that runs in seconds to a minute: for
              trying the steps, not for the physical result.
  --force     Write into a directory that is not empty.
  -h, --help  Show this page.

Examples:
  $ mallard list templates
  $ mallard new cylinder
  $ mallard new cylinder runs/cyl --quick

Exit status:
  0  the case was created, or the templates were listed
  1  the case could not be written
  2  usage error, no template given, or no template of that name

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-new
```

### mallard input

```text
Usage: mallard input <command> [options]

Commands on input files.

Commands:
  mallard input upgrade       Bring an input file from an older release up to
                              date
  mallard input resolve       Write the input as a run uses it: every key
                              explicit, defaults and preset included

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
output file's run block holds the same text (input.resolved), so a run's results
carry the full input that made them.

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
       mallard check [--format text|json]

Runs the whole setup of a case (mesh, partition, boundaries, numerics, initial
solution) without time stepping and without writing any file, then reports:
unknown or unused keys, boundary zones without a condition, mesh quality with
the worst cells, the stable time step and the cell that sets it, memory per rank
and, with --throughput or --steps, the cost. Each row has a status, each error
or warning a stable code (E-REGION-OUTSIDE, W-CFL-LIMIT) and the docs that
explain it. It lists every region the case uses and the mesh's own (domain,
boundary zones, cell zones) with its dimension, measure and cells, and whether a
surface or curve is exact or extracted; a region that covers no cell is an
error. Under mpirun -n N it checks the run on N ranks.

Without an input, checks the environment only; with one, the report ends with
the same Environment section: the build (dimension, precision, the solvers
installed), Kokkos and its execution spaces, every rank's host, device and
memory, CPUs and threads, MPI (library, ranks per host), GPU-aware MPI and NCCL
(built, and whether the run uses them), HDF5 (serial or parallel), CGNS, zlib
and KaMinPar. Each problem is a diagnostic with its fix: ranks sharing a GPU or
cores, OpenMP threads free to migrate, a GPU-aware build on an MPI that cannot
read device memory (an error), device halos staged through host memory. Run it
under the job's launcher (srun, mpirun) to check its nodes.

Options:
  -i FILE                   The input file, instead of the argument.
  --format text|json        text: a section per part of the check, each row with
                            its status (ok, WARN, ERROR) and units, then the
                            diagnostics with their codes; json: the same report
                            as one JSON object, for scripts. Default: text.
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
  --mesh-fields PREFIX      Write the mesh and stencil fields of every cell to
                            PREFIX_000000.vtkhdf (PREFIX_000000.vtu in builds
                            without HDF5), the default snapshot format: the mesh
                            quality (NON_ORTHOGONALITY, SKEWNESS, ASPECT_RATIO,
                            VOLUME_RATIO), the stencils as built for the run
                            (STENCIL_SIZE, STENCIL_SECTORS, STENCIL_DEGREE,
                            STENCIL_CONDITION, STENCIL_WALL, STENCIL_SEAM,
                            STENCIL_PARTITION) and the resolution estimates
                            (Y_PLUS, DELTA_OVER_ETA, CELLS_PER_DELTA_L).
                            Default: none.
  --stencils all            With --mesh-fields: also write every cell's
                            stencils, member by member with their roles, to
                            PREFIX_stencils.h5 (HDF5 builds); about 0.6 KB per
                            cell in 3D with TENO order 5. Default: none.
  --cells IDS               With --mesh-fields: write the stencils of these
                            cells (global ids, comma separated) to
                            PREFIX_stencils.h5, and each as a field STENCIL_<id>
                            (1 central, 2 sector, 3 both, 4 the cell itself).
                            Default: none.
  --region NAME|{TABLE}     With --mesh-fields: as --cells, for the cells of a
                            region: a name of [regions] or of the mesh, or an
                            inline table ({type = "rectangle", center = [0, 0],
                            size = [0.1, 0.1]}); STENCIL_<id> fields for the
                            first 64 cells. Default: none.
  --weights                 With --stencils, --cells or --region: add each
                            member's weights to PREFIX_stencils.h5: TENO's
                            pseudo-inverse rows, MUSCL's gradient weights.
                            Default: none.
  --quality PREFIX          Deprecated: --mesh-fields PREFIX.
  --catalog KEYS.json       Write every key the setup looked up, with its type,
                            unit, default and accepted values, as JSON. Default:
                            none.
  --view DIR                Write the case view, without running the case:
                            DIR/case.vtkhdf for ParaView (builds without HDF5:
                            DIR/case.vtm), one block each for the mesh, every
                            boundary zone, region the case uses (filled regions
                            as their cells with their fractions, surfaces and
                            curves with their normals or tangents and measures),
                            probe and periodic pair; and DIR/case-light.svg and
                            DIR/case-dark.svg, the annotated image (3D: an
                            isometric view over three orthographic ones) with
                            every boundary labeled by its name and condition and
                            every region and probe by its name, a legend, axes,
                            a scale bar and the domain's extent in its units.
                            Default: none.
  --kokkos-*                Kokkos options, e.g. --kokkos-num-threads=8 (threads
                            per rank) or --kokkos-device-id=1 (the GPU; by
                            default each rank gets its own).
  -h, --help                Show this page.

Examples:
  $ mallard check input.toml
  $ srun -N 2 --ntasks-per-node 4 mallard check   # the environment of a job
  $ mallard check input.toml --ranks 64 --throughput 5e7
  $ mpirun -n 4 mallard check input.toml --mesh-fields mesh
  $ mallard check input.toml --mesh-fields mesh --cells 120,4031   # their stencils
  $ mallard check input.toml --view view   # view/case-light.svg; view/case.vtkhdf in ParaView

Exit status:
  0  no errors (there may be warnings)
  1  errors: in the input, the mesh, the setup or the environment
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-check
```

### mallard info

```text
Usage: mallard info FILE [--format text|json] [--input] [--mesh-fields PREFIX] [--view DIR]
       mallard info RUN_DIR [--format text|json] [--input] [--verify] [--html]

Reports what a file Mallard writes or reads holds, without an input file: its
kind, format and format version; the mesh (cells, points, cell types, bounds,
boundary and cell zones); its times and steps; its fields with their units,
descriptions, ranges (of the last snapshot) and stored precision; the species;
its run block, the same for every file of the run (Mallard version and commit,
build, run id and the run it restarted from, when, where and on how many ranks
it ran, the input file and mesh with their SHA-256); and the last snapshot's
block (step, time, reason, when and by which run it was written). Step, time and
run come from what the file records, never from its name. Files without a run
block (before Mallard 0.9), without snapshot blocks (before 0.10) or from a
newer Mallard are reported with a note that says what is read instead.

FILE is a snapshot (.vtkhdf, .vtu, .pvtu, .h5) or a series index
(.vtkhdf.series, .pvd, .xmf), a restart file (compressed too), a probe file
(.csv, .h5, .vtkhdf, .vtp, .pvd), a CSV file of forces or integrals, or a mesh
file (Gmsh or Fluent .msh, .cas, .cgns, an OpenFOAM case, a Mallard HDF5 mesh).
Output files are read by any build, of any dimension and precision; a mesh file
by the solver of its dimension, which also reports its quality as mallard check
does.

RUN_DIR is a directory a run wrote to: its start from mallard.jsonl and, from
the same records as its report.html, its outcome, health (lowest values,
repairs, conservation), cost and warnings, each with a status; then every output
file in it and one level down, one table row each (the snapshots of a series as
their index). Each file the run log records as written is checked against its
record: missing, or a size (with --verify, a SHA-256) other than the one
recorded, is reported. A file still named NAME.partial is one whose write did
not finish. With --html, writes the run report instead: a page of the run's
outcome, identity, configuration, health, monitors, cost and output, built from
mallard.jsonl alone.

Options:
  --format text|json    text: sections and tables, with the units of every
                        quantity; json: the same as one JSON object, for
                        scripts. Default: text.
  --input               Print the input file that produced the file, from its
                        run block: the resolved input (every key the run read,
                        defaults filled in) for files of Mallard 0.10 and later,
                        the input as written for 0.9. For a run directory, its
                        run's.
  --mesh-fields PREFIX  Mesh files: write the mesh-quality fields to
                        PREFIX_000000.vtkhdf (PREFIX_000000.vtu in builds
                        without HDF5), as mallard check --mesh-fields does; the
                        stencil fields need the case, from mallard check.
                        Default: none.
  --view DIR            Mesh files: write the case view of the mesh alone (its
                        cells, boundary and cell zones and periodic pairs) to
                        DIR, as mallard check --view does: DIR/case.vtkhdf
                        (DIR/case.vtm without HDF5), DIR/case-light.svg and
                        DIR/case-dark.svg. Default: none.
  --quality PREFIX      Deprecated: --mesh-fields PREFIX.
  --verify              Run directories: also check each file's contents against
                        the SHA-256 its output record holds, not only its size;
                        reads every file once. Default: sizes only.
  --html                Run directories: write RUN_DIR/report.html, the run
                        report, from RUN_DIR/mallard.jsonl, the page the run
                        writes itself: for a run still going, killed, or of a
                        Mallard before 0.10. Default: off.
  --kokkos-*            Kokkos options, e.g. --kokkos-num-threads=8 (threads per
                        rank) or --kokkos-device-id=1 (the GPU; by default each
                        rank gets its own).
  -h, --help            Show this page.

Examples:
  $ mallard info solut/flow.vtkhdf.series
  $ mallard info solut/restart_001000.restart --format json
  $ mallard info solut/flow_000100.vtkhdf --input > input.toml   # rerun it
  $ mallard info wing.msh --mesh-fields wing
  $ mallard info wing.msh --view view   # view/case-light.svg: the zones, labeled
  $ mallard info runs/cylinder
  $ mallard info runs/cylinder --html            # rebuild runs/cylinder/report.html

Exit status:
  0  the file was read
  1  a file cannot be read (missing, not Mallard's, newer, its write unfinished), a run directory's file differs from its output record, --input finds no run block, or --html no mallard.jsonl
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-info
```

### mallard list

```text
Usage: mallard list <command> [options]

Catalogs: the fields, species, input keys and case templates there are.

Commands:
  mallard list fields         The output fields there are, or exactly those of
                              one case
  mallard list species        The species of a case
  mallard list keys           The input keys there are, or exactly those of one
                              case: type, units, default, values
  mallard list templates      The case templates of mallard new

mallard list <command> --help shows the page of a command.

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-list
```

### mallard list fields

```text
Usage: mallard list fields [INPUT] [--format text|json]

Without an input file, lists every field the [[write_data]], [[probes]] and
[statistics] fields take, by group, with its units, its description and what a
case needs to have it, and the patterns (Y_<s>, MEAN_<A>, COV_<A>_<B>, ...). Its
"available in" column says where each one can be had: written by [[write_data]],
sampled by [[probes]], averaged by [statistics], or derived after the run by
mallard post derive.

With an input file, sets the case up as mallard check does (reading its mesh,
writing nothing) and lists exactly the fields [[write_data]] fields accept for
it: its species, its physics, chemistry, LES and statistics, and its dimension,
with where each one can be had; then the common fields it does not have, with
what they need.

Options:
  --format text|json  text: one table per group (of fields) or table (of keys),
                      with units; json: the same as one JSON object, for
                      scripts. Default: text.
  --kokkos-*          Kokkos options, e.g. --kokkos-num-threads=8 (threads per
                      rank) or --kokkos-device-id=1 (the GPU; by default each
                      rank gets its own).
  -h, --help          Show this page.

Examples:
  $ mallard list fields
  $ mallard list fields input.toml
  $ mallard list fields input.toml --format json

Exit status:
  0  listed
  1  the case cannot be set up (the errors are printed)
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-list-fields
```

### mallard list species

```text
Usage: mallard list species INPUT [--format text|json]

Sets the case up as mallard check does and lists its species, one table row
each, in the order of its mechanism: the <s> of its Y_<s>, X_<s> and RHOY_<s>
fields. A single gas has none.

Options:
  --format text|json  text: one table per group (of fields) or table (of keys),
                      with units; json: the same as one JSON object, for
                      scripts. Default: text.
  --kokkos-*          Kokkos options, e.g. --kokkos-num-threads=8 (threads per
                      rank) or --kokkos-device-id=1 (the GPU; by default each
                      rank gets its own).
  -h, --help          Show this page.

Examples:
  $ mallard list species flame.toml
  $ mallard list species flame.toml --format json

Exit status:
  0  listed
  1  the case cannot be set up (the errors are printed)
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-list-species
```

### mallard list keys

```text
Usage: mallard list keys [SECTION] [INPUT] [--format text|json]

Without an input file, lists the key catalog docs/input-keys.json: every key the
examples and tutorials read, by table, with its type, units (of a number with a
physical dimension), default (or required) and accepted values; (2D) or (3D)
marks a key only that dimension reads.

With an input file, sets the case up as mallard check does (reading its mesh,
writing nothing) and lists the keys its setup reads, the same columns and
whether the input sets each: what mallard check --catalog writes, as tables.

SECTION keeps the keys of one table (numerics, [numerics], write_data) or of one
key path (numerics.face_reconstruction). An argument that ends in .toml or names
a file is the input.

Options:
  --format text|json  text: one table per group (of fields) or table (of keys),
                      with units; json: the same as one JSON object, for
                      scripts. Default: text.
  --kokkos-*          Kokkos options, e.g. --kokkos-num-threads=8 (threads per
                      rank) or --kokkos-device-id=1 (the GPU; by default each
                      rank gets its own).
  -h, --help          Show this page.

Examples:
  $ mallard list keys
  $ mallard list keys numerics
  $ mallard list keys write_data input.toml
  $ mallard list keys input.toml --format json

Exit status:
  0  listed
  1  the case cannot be set up (the errors are printed), or the catalog is not found
  2  usage error, or a section with no key

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-list-keys
```

### mallard list templates

```text
Usage: mallard list templates

Lists the templates mallard new creates cases from, with what each shows and
what it costs.

Options:
  -h, --help  Show this page.

Examples:
  $ mallard list templates
  $ mallard new cylinder

Exit status:
  0  listed

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-list-templates
```

### mallard diff

```text
Usage: mallard diff FILE_A FILE_B [--fields LIST] [--tol TOL] [--step N|--time T] [--format text|json]
       mallard diff FILE_A FILE_B --remap [--mesh-a FILE] [--mesh-b FILE] [options]

Compares two snapshot, restart, probe or CSV files of the same case, of any
format each (a restart file against a VTKHDF snapshot, say): for every field
both hold, how many values differ bit for bit, the L1, L2 and Linf norms of B -
A (mean, root mean square and max over the cells, in the field's units) and the
same relative to A's norms, and where the largest difference is (the cell, its
coordinates and both values there). Restart files are also compared bit for bit
in the rest of their state: step, time, scalar attributes, named arrays and the
face records of characteristic boundaries; the run and snapshot blocks, which
record when and where each ran, are not compared.

Each field passes if its max |B - A| / max |A| is at most --tol (0 by default),
and so does t; the result is identical (every value bitwise the same), within
tolerance or beyond it, on the first and last line and as the exit status. Cells
are matched by their global index: restart files and VTKHDF or XDMF snapshots of
the whole mesh are in the mesh's order whatever the ranks that wrote them, so a
serial run compares with an MPI run. Files on different meshes are refused,
unless --remap.

Uses: regression checks in CI, comparing two settings, checking that a restart
reproduces a run, serial against MPI, and against a finer mesh with --remap.

Options:
  --fields LIST       Compare only these fields, comma separated (RHO,P,U); each
                      file must hold them. Default: every field both files hold.
  --tol TOL           The tolerance on each field's max |B - A| / max |A|, and
                      on t relative to the larger |t|: a number for every field,
                      then FIELD=TOL for some (1e-12,P=1e-6). Restart attributes
                      and arrays take it too. Default: 0, so any difference in
                      value fails.
  --step N            Compare the snapshot of each file at this step, as the
                      files record it, for series and files of several times.
                      Default: the last snapshot.
  --time T            Compare the snapshot of each file nearest this simulated
                      time, in s; the Time table shows the times compared.
                      Default: the last snapshot.
  --snapshot N|last   Compare the snapshot of each file at this position,
                      counting from 0 (in time order); for files that record no
                      step. Default: last.
  --remap             Compare files on different meshes: B's cell values
                      remapped conservatively onto A's mesh (as [initialize]
                      type = "remap" does, degree 2, limited), then compared
                      cell by cell. Needs both meshes: as the files' run blocks
                      record them, or --mesh-a and --mesh-b.
  --mesh-a FILE       A's mesh file, where its run block does not record it or
                      the file has moved: for --remap and for the coordinates of
                      the largest differences. Default: from A's run block.
  --mesh-b FILE       B's mesh file, for --remap. Default: from B's run block.
  --format text|json  text: sections and tables, with the units of every
                      quantity; json: the same as one JSON object, for scripts.
                      Default: text.
  --kokkos-*          Kokkos options, e.g. --kokkos-num-threads=8 (threads per
                      rank) or --kokkos-device-id=1 (the GPU; by default each
                      rank gets its own).
  -h, --help          Show this page.

Examples:
  $ mallard diff run1/solut/restart_001000.restart run2/solut/restart_001000.restart
  $ mallard diff serial/flow_000100.vtkhdf mpi/flow_000100.vtkhdf --tol 1e-12
  $ mallard diff a/flow.vtkhdf.series b/flow.vtkhdf.series --step 300 --fields RHO,P
  $ mallard diff fine/restart_000500.restart coarse/restart_000500.restart --remap --tol 1e-2
  $ mallard diff a.restart b.restart --format json | jq '.fields[] | {name, status}'

Exit status:
  0  identical, or every difference within tolerance
  1  a difference beyond tolerance
  2  usage error
  3  a file cannot be read, or the files cannot be compared (no field in common, different meshes without --remap)

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-diff
```

### mallard mesh

```text
Usage: mallard mesh <command> [options]

Commands on meshes.

Commands:
  mallard mesh convert        Write a mesh file, or the mesh of an input file,
                              as a Mallard HDF5 mesh

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

Commands on a run's output: probes, integrals, snapshots, derived fields and
averages of it, and the grid convergence of several runs.

Commands:
  mallard post extract        Probes, region integrals and snapshots of existing
                              output, as the run would have written them
  mallard post derive         Derived fields of existing output (MACH, P0,
                              Q_CRITERION, ...), as the run would have written
                              them
  mallard post average        Means, RMS, covariances, PDFs and conditional
                              statistics over saved snapshots, as [statistics]
                              keeps them
  mallard post convergence    Order of accuracy, extrapolated value and grid
                              convergence index (GCI) of runs at several
                              resolutions

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
FILE's run block, and its mesh must be the file's: run it where the run ran, so
that relative paths (the mesh file) resolve. --steps, --times, --every and
--latest choose the snapshots, by the steps and times the files record (files
written before 0.10 that record no step: by their position in FILE).

Files go to the output directory under the names the input gives them, so a
run's files are never overwritten. The report lists each file with its times and
fields, and every reduction of the last snapshot with its units. Under mpirun -n
N the probes are evaluated on N ranks.

Options:
  -i FILE                The input file of the case: its mesh, physics,
                         numerics, [regions] and [[probes]]. Default: the input
                         in the run block of FILE (Mallard 0.10 and later record
                         it resolved), whose mesh file, mechanism and TENO
                         cache_file resolve from the run's directory.
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
  --fields LIST          The fields of --region and --snapshot, separated by
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

### mallard post derive

```text
Usage: mallard post derive FILE... --fields LIST [-i input.toml] [--with-state] [-o DIR]
       mallard post derive FILE... --fields CP --reference TABLE
       mallard post derive FILE... --fields MIXTURE_FRACTION --mixture TABLE

Sets up the case of a restart file, snapshot or series and writes snapshots of
derived fields of its stored state, without rerunning, so the fields need not be
chosen before a long run. It runs the run's own code, with its reconstruction
and gradients: a field is what [[write_data]] would have written at that step,
the same values for restart files and snapshots of the conservative variables,
the same to rounding for snapshots of RHO, U and P. --fields takes the fields
mallard list fields says it writes, the gradient fields VORTICITY and
Q_CRITERION among them, not what only the run itself records (CFL, P_MAX,
TENO_SIGMA); those that need input the run did not give (the reference of CP,
the streams of MIXTURE_FRACTION) take it from --reference and --mixture.

FILE, the case and the choice of snapshots are as for mallard post extract: the
case comes from -i, else from FILE's run block, and its mesh must be the file's;
--steps, --times, --every and --latest choose the snapshots by the steps and
times the files record. The snapshots go to the output directory, so a run's
files are never overwritten. Under mpirun -n N the fields are computed on N
ranks.

Options:
  --fields LIST          The fields to write, separated by commas
                         (MACH,P0,Q_CRITERION): those mallard list fields says
                         mallard post derive writes. Required.
  -i FILE                The input file of the case: its mesh, physics, numerics
                         and [output] reference and mixture. Default: the input
                         in the run block of FILE (Mallard 0.10 and later record
                         it resolved), whose mesh file, mechanism and TENO
                         cache_file resolve from the run's directory.
  --reference TABLE      The [output] reference of CP, as an inline TOML table
                         ('{rho = 1.2, u = [30, 0], p = 101325}'), in place of
                         the input's. Default: the input's.
  --mixture TABLE        The [output.mixture] table of the mixture fields,
                         inline ('{fuel = "fuel_inlet", oxidizer = {X = {O2 = 1,
                         N2 = 3.76}}}'), in place of the input's. Default: the
                         input's.
  --with-state           Also write RHO, U, P and T in each snapshot, so that it
                         stands alone.
  --region REGION        Write the fields on a region only: a name of the
                         input's [regions] or of the mesh, or an inline TOML
                         table, as [[write_data]] region takes it. Default: the
                         whole mesh.
  --snapshot vtkhdf|vtu  The format of the snapshots: OUTPUT/NAME_<step>.vtkhdf
                         with a .vtkhdf.series, or .vtu files with a .pvd index.
                         Default: vtkhdf, or vtu in builds without HDF5.
  --name NAME            The name of the snapshots: NAME_<step>.vtkhdf. Default:
                         derived.
  --steps LIST           Derive the snapshots of these steps, separated by
                         commas (--steps 100,200). Default: every snapshot.
  --times T|T1:T2        Derive the snapshot nearest to simulated time T, or
                         every one with T1 <= t <= T2, in seconds. Default:
                         every snapshot.
  --every N              Derive every Nth of the snapshots selected, starting
                         with the first. Default: 1.
  --latest               Derive only the last snapshot selected.
  -o, --output DIR       The directory the snapshots go to. Default: derived.
  --format text|json     text: sections and tables; json: the same as one JSON
                         object, for scripts. Default: text.
  --kokkos-*             Kokkos options, e.g. --kokkos-num-threads=8 (threads
                         per rank) or --kokkos-device-id=1 (the GPU; by default
                         each rank gets its own).
  -h, --help             Show this page.

Examples:
  $ mallard post derive solut/flow.vtkhdf.series --fields MACH,P0,Q_CRITERION
  $ mallard post derive solut/restart_*.restart --latest --fields ENTROPY,DILATATION --with-state
  $ mallard post derive solut/flow.vtkhdf.series --fields CP \
        --reference '{rho = 1.2, u = [30, 0], p = 101325}'
  $ mallard post derive solut/flame.vtkhdf.series --fields MIXTURE_FRACTION,FLAME_INDEX \
        --mixture '{fuel = "fuel_inlet", oxidizer = "air_inlet"}'

Exit status:
  0  every snapshot was derived
  1  a file cannot be read, the case cannot be set up, a field needs input it lacks, or the file is not of its mesh
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-post-derive
```

### mallard post average

```text
Usage: mallard post average FILE... [--fields LIST] [--products LIST] [--favre] [--t-start T] [-o DIR]
       mallard post average FILE... --homogeneous DIRS [--region REGION]
       mallard post average FILE... --conditional TABLE
       mallard post average FILE --statistics --homogeneous DIRS

Sets up the case of a series of restart files or snapshots and averages their
states in time, for runs without [statistics]: per cell, the means MEAN_<a> and
covariances COV_<a>_<b> that [statistics] keeps, by its own code, RMS_<a> (the
square root of a variance) and with --favre the Favre means and covariances.
Each snapshot weighs the time since the previous one, as a run's samples do, so
restart files at the run's sampling steps, with --t-start at its t_start, give
the run's averages bit for bit. --fields takes what [statistics] fields takes,
derived fields included (mallard list fields), and in reacting runs HRR and
OMEGA_<species>. --conditional adds the PDFs and conditional statistics of
[[statistics.conditional]], such as conditional means on the mixture fraction,
written as CSV.

The averages are a snapshot of the last step, OUTPUT/NAME_<step>.vtkhdf, as a
run writes its [statistics]. --homogeneous averages them further over directions
in which the flow is statistically uniform and writes a CSV profile (the kept
coordinates, n_cells, then the columns), which synthetic inflow profile reads;
--statistics does that to the [statistics] fields one snapshot already holds.
The states are read one at a time, so memory does not grow with the number of
snapshots.

FILE, the case and the choice of snapshots are as for mallard post extract: the
case comes from -i, else from FILE's run block, and its mesh must be the file's;
the snapshots' steps and times are those the files record, in increasing time.
Under mpirun -n N the cells are split over N ranks.

Options:
  --fields LIST          The variables to average, separated by commas, as
                         [statistics] fields takes them, derived fields included
                         (RHO,U_X,P,MACH). Default: RHO, U_X, U_Y, U_Z (3D), P
                         and T.
  --products LIST        The covariances to keep, as pairs A*B separated by
                         commas (U_X*U_Y,P*U_X), or none; a variable not in
                         --fields is averaged too. Default: the variance of
                         every field and the covariances of the velocity
                         components (the Reynolds stresses).
  --favre                Also Favre (density-weighted) means FAVRE_MEAN_<a> and
                         covariances FAVRE_COV_<a>_<b> of the fields and
                         products other than RHO, exactly, the triple
                         correlations included; adds RHO to the fields.
  --t-start T            Where the averaging window starts, in s: the first
                         snapshot weighs the time since then (each other, the
                         time since the previous one). The run's [statistics]
                         t_start reproduces its averages. Default: as long
                         before the first snapshot as the second is after it, so
                         that evenly spaced snapshots weigh the same.
  --homogeneous DIRS     Also average over these directions, separated by commas
                         (x,z: profiles in y of a channel), weighted by cell
                         volume, and write a CSV profile, OUTPUT/NAME.csv,
                         instead of a snapshot. Covariances include the
                         dispersive part, the spatial variation of the time
                         means. Default: none.
  --conditional TABLE    Also the PDF of one or two variables and conditional
                         means and variances of fields on them, as an inline
                         TOML table or an array of them, as
                         [[statistics.conditional]] takes them ({on =
                         "MIXTURE_FRACTION", fields = ["T", "HRR"]}), written to
                         OUTPUT/conditional_<on>.csv unless the table gives a
                         file. Without --fields and --products, only these.
                         Default: none.
  --statistics           Average over --homogeneous directions the [statistics]
                         fields (MEAN_, COV_) that one snapshot of FILE holds,
                         instead of averaging its states in time.
  --rtol R               With --homogeneous: cells share a row where their
                         centers agree to R times the mesh's extent in each kept
                         direction. Default: 1e-6.
  --region REGION        Average over a region only: a name of the input's
                         [regions] or of the mesh, or an inline TOML table, as
                         [[write_data]] region takes it; with --homogeneous, a
                         region of the mesh's dimension, its cells weighted by
                         the fraction of them in it. A region defined by the
                         solution (where, isosurface, or a boolean of them) is
                         placed on each snapshot's state, and the time averages
                         of the integrals, means, covariances and Favre averages
                         over it are written to OUTPUT/NAME_reductions.csv
                         instead of a snapshot, or with --homogeneous their
                         profile, each cell weighted by the time average of its
                         measure in it. Default: the whole mesh.
  --reference TABLE      The [output] reference of CP, as an inline TOML table,
                         in place of the input's, as for mallard post derive.
                         Default: the input's.
  --mixture TABLE        The [output.mixture] table of the mixture fields,
                         inline, in place of the input's, as for mallard post
                         derive. Default: the input's.
  -i FILE                The input file of the case: its mesh, physics and
                         numerics. Default: the input in the run block of FILE
                         (Mallard 0.10 and later record it resolved), whose mesh
                         file, mechanism and TENO cache_file resolve from the
                         run's directory.
  --snapshot vtkhdf|vtu  The format of the snapshot of the averages:
                         OUTPUT/NAME_<step>.vtkhdf with a .vtkhdf.series, or a
                         .vtu file with a .pvd index. Default: vtkhdf, or vtu in
                         builds without HDF5.
  --name NAME            The name of the output: NAME_<step>.vtkhdf, or
                         NAME.csv. Default: average.
  --steps LIST           Average the snapshots of these steps, separated by
                         commas (--steps 100,200). Default: every snapshot.
  --times T|T1:T2        Average the snapshot nearest to simulated time T, or
                         every one with T1 <= t <= T2, in seconds. Default:
                         every snapshot.
  --every N              Average every Nth of the snapshots selected, starting
                         with the first. Default: 1.
  --latest               Only the last snapshot selected (with --statistics).
  -o, --output DIR       The directory the output goes to. Default: average.
  --format text|json     text: sections and tables; json: the same as one JSON
                         object, for scripts. Default: text.
  --kokkos-*             Kokkos options, e.g. --kokkos-num-threads=8 (threads
                         per rank) or --kokkos-device-id=1 (the GPU; by default
                         each rank gets its own).
  -h, --help             Show this page.

Examples:
  $ mallard post average solut/restart_*.restart --t-start 40
  $ mallard post average solut/flow.vtkhdf.series --times 40:80 --favre
  $ mallard post average solut/flow.vtkhdf.series --fields U_X,U_Y,U_Z,MACH --homogeneous x,z
  $ mallard post average stats/channel.vtkhdf.series --latest --statistics --homogeneous x,z
  $ mallard post average solut/flame.vtkhdf.series --conditional \
        '{on = "MIXTURE_FRACTION", fields = ["T", "HRR"], weight = "mass"}'

Exit status:
  0  the averages were written
  1  a file cannot be read, the case cannot be set up, a field needs input it lacks, or the file is not of its mesh
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-post-average
```

### mallard post convergence

```text
Usage: mallard post convergence RUN_DIR RUN_DIR [RUN_DIR...] [--quantity NAMES] [--time T | --average T0:T1]
       mallard post convergence RUN_DIR|FILE... --field NAME --exact EXPR

Estimates the discretization error of a case from runs of it at several
resolutions: for each quantity, the order of accuracy the runs show, the value
extrapolated to zero cell size and the fine run's grid convergence index (GCI),
an error band, by the procedure of Celik et al. (2008) with Roache's factors of
safety. A quantity's verdict says in words whether it converges, and how far to
trust the band.

Each RUN_DIR is a directory a run ran in. What identifies it is what its run
record (mallard.jsonl) records, never its name: its run, its cell count and its
input. The runs are ordered by cell count, whatever the order given, and their
refinement ratios follow from it. The quantities are the columns of the forces,
integrals and probe tables each run wrote ([[forces]], [integrals], [[probes]]),
or mallard post extract wrote into its directory from its snapshots, by default
their last values. Inputs that differ other than in [mesh], [run], [output] and
[[write_data]] are a warning.

Three runs give an observed order; more give one per three successive runs,
which agree once the runs are in the asymptotic range. Two runs assume the
design order with a factor of safety of 3, a warning. With --field and --exact
the report is the error norms of a field against an exact solution and their
observed orders; then FILE can also be a snapshot or restart file of the whole
mesh.

Options:
  --quantity NAMES    The quantities to study, separated by commas: TABLE:COLUMN
                      (integrals:rho2, forces_wall:F_x), a column of every table
                      (F_x), or a table (forces_wall), as the report names them.
                      Default: every column of every forces, integrals and probe
                      table that all the runs wrote.
  --time T            Take each quantity at simulated time T, in seconds (a row
                      every run wrote), or with --field the snapshot of time T.
                      Default: each run's last value (snapshot).
  --average T0:T1     Take the time average of each quantity over T0 <= t <= T1,
                      in seconds (trapezoidal, over the rows of the window): for
                      unsteady flows. Default: each run's last value.
  --order P           The order of accuracy to compare the observed orders with,
                      and the order a study of two runs assumes. Default: the
                      design order of the runs' reconstruction (FO 1, MUSCL 2,
                      TENO its order).
  --safety-factor F   The GCI's factor of safety. Default: 1.25 with three or
                      more runs, 3 with two (Roache).
  --sizes LIST        A representative cell size of each run, in the order
                      given, separated by commas: for refinements that are not
                      uniform, or time-step studies. Default: (1 /
                      cells)^(1/dim) from each run's cell count.
  --field NAME        Instead of quantities, the L1, L2 and Linf errors of this
                      field of each run's last snapshot (NAME[k] for a component
                      of a vector) against --exact. Needs --exact.
  --exact EXPR        The exact solution of --field, an expression of x, y, z
                      and t, averaged over each cell (quadrature exact to degree
                      6) at the snapshot's time.
  --format text|json  text: sections and tables, with the units of every
                      quantity; json: the same as one JSON object, for scripts.
                      Default: text.
  --kokkos-*          Kokkos options, e.g. --kokkos-num-threads=8 (threads per
                      rank) or --kokkos-device-id=1 (the GPU; by default each
                      rank gets its own).
  -h, --help          Show this page.

Examples:
  $ mallard post convergence coarse medium fine          # every force, integral and probe
  $ mallard post convergence coarse medium fine --quantity forces_cylinder:F_x
  $ mallard post convergence n32 n64 n128 --average 10:20 --quantity F_x,F_y
  $ mallard post convergence n16 n32 n64 --field RHO --exact '1 + 0.2 * sin(2 * pi * (x - t))'
  $ mallard post convergence coarse fine --order 2 --format json

Exit status:
  0  no errors (there may be warnings: oscillating, two runs, an order far from the design's)
  1  a quantity diverges, a run cannot be read, or the runs are not of one case
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-post-convergence
```

### mallard pack

```text
Usage: mallard pack RUN_DIR [-o FILE] [--with-outputs small|all|none]
       mallard pack --verify FILE

Packs the last run of a run directory into one tar archive, for a paper's
supplement or a bug report: what it needs to run again and what it wrote. What
goes in comes from its run record (mallard.jsonl, Mallard 0.10 and later), never
from file names: the input as the run read it (from its run block) and the
resolved input, the mesh file (a generated mesh is made again from the input),
every other file it read (mechanisms, tables, profiles, probe points, the
restart file or remap source it started from), its start record (the run block:
code, build, where and how it ran), its run record, and the outputs
--with-outputs asks for, as its output records and resolved input name them.
Each file is checked against the SHA-256 the run recorded of it, so a file
changed since the run read or wrote it is refused.

The archive holds one directory, named as the archive: manifest.json (each file
with its role, size and SHA-256, and the command that reruns the run), run.json
(the start record), resolved.toml, case/ (the input, mesh and files read, laid
out as they were around the run directory, so the input's relative paths hold)
and output/ (the outputs, by their paths in the run directory). Unpacked
anywhere with tar -xf, the rerun command of the manifest, in its directory,
reproduces the run: bit for bit on the same build and rank count. A file the
input names by an absolute path is packed too, under absolute/, but the rerun
reads it from that path.

--verify checks an archive: every file against its manifest, and the input, mesh
and files read against the run block's own hashes. The archive is
gzip-compressed in builds with zlib (Mallard_ENABLE_ZLIB), else plain tar; any
tar reads both.

Options:
  -o, --output FILE         The archive: .tar.gz (gzip-compressed, in builds
                            with zlib) or .tar. Default: the run directory's
                            name with .tar.gz (.tar in builds without zlib), in
                            the current directory.
  --with-outputs small|all|none
                            The run's outputs to pack besides its run record:
                            small, its forces, integrals and probes (CSV files,
                            probe snapshots, spectra); all, also its snapshots
                            and restart files; none. Default: small.
  --verify                  Check the archive FILE instead: every file against
                            the size and SHA-256 its manifest records, no file
                            missing or added, and the input, mesh and files read
                            against the SHA-256 the run recorded.
  -h, --help                Show this page.

Examples:
  $ mallard pack runs/cylinder                         # cylinder.tar.gz
  $ mallard pack runs/cylinder -o cylinder_re100.tar.gz --with-outputs all
  $ mallard pack --verify cylinder.tar.gz
  $ tar -xf cylinder.tar.gz && cd cylinder/case && mallard -i input.toml

Exit status:
  0  the archive was written, or every file of it checks
  1  the run record or a file cannot be read, a file differs from the run's record of it, or (--verify) a file is missing, added or differs
  2  usage error

Documentation: https://matthewbonanni.github.io/mallard/docs/guide/cli/#mallard-pack
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
its requests, sections and keys of the key catalog for mallard list keys, output
fields of the field catalog for --fields, each item after a comma too. It is
made from the same table as the help pages and from the catalogs mallard list
keys and mallard list fields print, so it knows every command, key and field of
this build; templates come from mallard new when completing.

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
