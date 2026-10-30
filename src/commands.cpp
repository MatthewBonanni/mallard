/**
 * @file commands.cpp
 * @brief The command table of the mallard executable: every command's help
 *        page, options and arguments, and the function that runs it.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <iostream>
#include <string>

#include "build_info.h"
#include "case_template.h"
#include "cli.h"
#include "commands.h"
#include "front_end.h"
#include "input_upgrade.h"
#include "run_control.h"

namespace cli {

namespace {

constexpr Option KOKKOS = {"--kokkos-*", "", "",
                           "Kokkos options, e.g. --kokkos-num-threads=8 (threads per rank) or "
                           "--kokkos-device-id=1 (the GPU; by default each rank gets its own)."};

constexpr Option RUN_OPTIONS[] = {
    {"-i", "", "FILE", "The input file (TOML). Required."},
    {"--restart", "", "FILE",
     "Start from this restart file: replaces the input's [initialize] with type = \"restart\". "
     "Default: as [initialize] says."},
    KOKKOS,
};

constexpr Option CHECK_OPTIONS[] = {
    {"-i", "", "FILE", "The input file, instead of the argument."},
    {"--restart", "", "FILE", "Check the run as started from this restart file."},
    {"--ranks", "", "N",
     "Memory and cost per rank for a run on N ranks; from a serial check, memory is scaled to the "
     "largest part the run's partitioner would make. Default: the ranks of this check."},
    {"--throughput", "", "CELLS_PER_S",
     "Cost (wall time and GPU-hours per unit simulated time, per flow-through and to t_stop) at this "
     "many cell updates per second per rank: the Throughput of a run's summary over its ranks. "
     "Default: no cost."},
    {"--steps", "", "N", "Cost at the throughput measured over N steps, after one untimed step. Default: no cost."},
    {"--quality", "", "PREFIX",
     "Write the mesh-quality fields to PREFIX_000000.vtkhdf (PREFIX_000000.vtu in builds without HDF5), "
     "the default snapshot format. Default: none."},
    {"--catalog", "", "KEYS.json",
     "Write every key the setup looked up, with its type, unit, default and accepted values, as JSON. "
     "Default: none."},
    {"--view", "", "DIR",
     "Write the case view to DIR/case.vtkhdf for ParaView: every region the case uses (filled regions as their "
     "cells with their fractions, surfaces and curves with their normals or tangents and measures) and every "
     "boundary zone, one block each (builds without HDF5: DIR/case.vtm). Default: none."},
    KOKKOS,
};

constexpr Option NEW_OPTIONS[] = {
    {"--list", "-l", "", "List the templates, with what each shows and what it costs."},
    {"--quick", "", "",
     "A coarse, short version that runs in seconds to a minute: for trying the steps, not for the "
     "physical result."},
    {"--force", "", "", "Write into a directory that is not empty."},
};

constexpr Option UPGRADE_OPTIONS[] = {
    {"--check", "", "", "Write nothing: list the changes, with exit status 1 if there are any."},
    {"--dim", "", "2|3",
     "The case's dimension, for renames that depend on it. Default: from the input's mesh, as a run finds it."},
};

constexpr Option REACTOR_OPTIONS[] = {
    {"-i", "", "FILE", "The input file, with a [reactor] table, or a [benchmark] table too. Required."},
    KOKKOS,
};

constexpr Option TEST_OPTIONS[] = {
    {"--dim", "", "2|3", "Run only the 2D or the 3D suite. Default: every suite there is."},
    {"--gtest_*", "", "",
     "GoogleTest options, e.g. --gtest_filter='Sod*' (tests by name) or --gtest_list_tests; mallard test "
     "--help lists them all after this page."},
    KOKKOS,
};

constexpr Option GLOBAL_OPTIONS[] = {
    {"--dim", "", "2|3",
     "--dim 2 runs mallard2d, --dim 3 mallard3d, for any command. Default: the dimension of the case's "
     "mesh."},
};

constexpr std::string_view CHECK_ARGUMENTS[] = {"[INPUT]"};
constexpr std::string_view NEW_ARGUMENTS[] = {"[TEMPLATE]", "[DIRECTORY]"};
constexpr std::string_view CTL_ARGUMENTS[] = {"RUN_DIR", "COMMAND", "[KEY=VALUE...]"};
constexpr std::string_view FILE_ARGUMENTS[] = {"FILE"};
constexpr std::string_view CONVERT_ARGUMENTS[] = {"INPUT", "OUTPUT.h5"};

int new_main(const Args & args) {
    return case_template::new_case(args.positional, args.has("--list"), args.has("--quick"), args.has("--force"));
}

int ctl_main(const Args & args) {
    const std::string & verb = args.positional[1];
    std::string request = verb;
    if (verb == "set") {
        if (args.positional.size() < 3) return usage_error(*command("ctl"), "set needs KEY=VALUE");
        for (size_t i = 2; i < args.positional.size(); i++) request += " " + args.positional[i];
    } else if (verb != "status" && verb != "checkpoint" && verb != "stop") {
        return usage_error(*command("ctl"), "unknown command " + verb + " (status, checkpoint, stop, set)");
    } else if (args.positional.size() > 2) {
        return usage_error(*command("ctl"), "unexpected argument " + args.positional[2]);
    }
    return run_control::ctl(args.positional[0], request);
}

int upgrade_main(const Args & args) {
    const std::string & file = args.positional[0];
    const std::string dim = args.value_or("--dim", "");
    if (!dim.empty() && dim != "2" && dim != "3") {
        return cli::usage_error(*cli::command("input upgrade"), "--dim takes 2 or 3, not " + dim);
    }
    return input_upgrade::upgrade_file(file, args.has("--check"),
                                       dim.empty() ? front_end::input_dimension(file).dim : dim[0] - '0');
}

int version_main(const Args &) {
    std::cout << "Mallard " << mallard_version() << std::endl;
    return 0;
}

constexpr Command COMMANDS[] = {
    {
        .name = "",
        .group = "Run",
        .summary = "Run a case",
        .synopsis = "mallard -i input.toml [--restart FILE] [Kokkos options]\n"
                    "mallard <command> [options]",
        .description =
            "Mallard solves the compressible Euler, Navier-Stokes and reacting-flow equations on unstructured "
            "meshes, in 2D and 3D. mallard -i input.toml runs the case an input file describes; the commands "
            "below create, check, control and maintain cases.\n"
            "\n"
            "mallard runs mallard2d or mallard3d, by the dimension of the case: that of its mesh file, or, for "
            "a generated mesh, 3D when [mesh] sets Nz or Lz. --dim 2|3 states it.\n"
            "\n"
            "Parallel runs start under MPI, with one rank per GPU or per group of cores: mpirun -n 4 mallard -i "
            "input.toml. Kokkos options set the threads and devices of each rank; on OpenMP builds, so do "
            "OMP_NUM_THREADS, OMP_PROC_BIND and OMP_PLACES.",
        .options = RUN_OPTIONS,
        .arguments = {},
        .examples = "$ mallard -i input.toml\n"
                    "$ mallard -i input.toml --kokkos-num-threads=8\n"
                    "$ mpirun -n 4 mallard -i input.toml\n"
                    "$ mallard -i input.toml --restart solut/restart_010000.restart",
        .exit_status = "0  the run reached a stop condition (n_steps, t_stop, t_wall_stop)\n"
                       "1  an error, divergence included\n"
                       "2  usage error: an unknown option or command, no -i\n"
                       "3  stopped on request (signal, mallard ctl stop, STOP file), restart written",
        .dim = Case::input,
        .run = run_main,
    },
    {
        .name = "ctl",
        .group = "Run",
        .summary = "Ask a running case for its status, a restart file or a stop, or change its settings",
        .synopsis = "mallard ctl <run-dir> status | checkpoint | stop\n"
                    "mallard ctl <run-dir> set <key>=<value> [<key>=<value> ...]",
        .description =
            "Sends a request to the run started in run-dir, through its control socket mallard.sock, and "
            "prints the reply. It must run on the node of the run's rank 0.\n"
            "\n"
            "status: step, t, dt, CFL number, progress, wall time per step, cells per second, ETA and memory. "
            "checkpoint: write a restart file now; the reply names it. stop: write a restart file and end the "
            "run (exit status 3). set: change settings from the next step on: n_steps, t_stop, t_wall_stop "
            "(0 removes a limit; one must remain; times in seconds or with a unit, e.g. t_wall_stop=2h), "
            "check_interval, write_data.<prefix or index>.interval or .time_interval, "
            "probes.<name>.interval or .time_interval, forces.<zone>.interval, integrals.interval. Other keys "
            "are refused.",
        .options = {},
        .arguments = CTL_ARGUMENTS,
        .examples = "$ mallard ctl runs/cylinder status\n"
                    "$ mallard ctl runs/cylinder set t_stop=3.0 write_data.flow.time_interval=0.05\n"
                    "$ mallard ctl . stop",
        .exit_status = "0  the run carried out the request\n"
                       "1  no run answers in run-dir, or the run refused the request\n"
                       "2  usage error",
        .run = ctl_main,
    },
    {
        .name = "reactor",
        .group = "Run",
        .summary = "Integrate a 0D reactor, or benchmark the chemistry",
        .synopsis = "mallard reactor -i input.toml [Kokkos options]",
        .description =
            "Integrates the adiabatic constant-volume reactor of the input's [reactor] table (initial T, p and "
            "composition, end_time) with the solver's chemistry kernels, and writes its temperature, pressure "
            "and mass fractions as CSV to reactor.output (default reactor.csv). With a [benchmark] table, it "
            "advances states sampled along the reactor's trajectory, replicated over many cells, and writes "
            "cells per second to benchmark.output (default benchmark.csv). Runs on one rank.",
        .options = REACTOR_OPTIONS,
        .arguments = {},
        .examples = "$ mallard reactor -i input.toml\n"
                    "$ mallard reactor -i benchmark.toml --kokkos-num-threads=8",
        .exit_status = "0  done\n"
                       "1  an error in the input or the integration, or more than one rank\n"
                       "2  usage error",
        .run = reactor_main,
    },
    {
        .name = "new",
        .group = "Set up",
        .summary = "Create a ready-to-run case from a template",
        .synopsis = "mallard new <template> [directory] [--quick] [--force]\n"
                    "mallard new --list",
        .description =
            "Creates a ready-to-run case from one of Mallard's examples or tutorials: input.toml, run.sh, the "
            "mesh or initial-state script it needs, post.sh and a README. The directory defaults to the "
            "template's name. Without a template, lists them.",
        .options = NEW_OPTIONS,
        .arguments = NEW_ARGUMENTS,
        .examples = "$ mallard new --list\n"
                    "$ mallard new cylinder\n"
                    "$ mallard new cylinder runs/cyl --quick",
        .exit_status = "0  the case was created, or --list\n"
                       "1  the case could not be written\n"
                       "2  usage error, no template given, or no template of that name",
        .dim = Case::template_name,
        .run = new_main,
    },
    {
        .name = "input upgrade",
        .group = "Set up",
        .summary = "Bring an input file from an older release up to date",
        .synopsis = "mallard input upgrade old.toml > new.toml\n"
                    "mallard input upgrade --check old.toml",
        .description =
            "Rewrites the renamed, moved and removed keys and options of an input file for Mallard {version}, "
            "keeping its comments and layout, and writes it to standard output; the changes go to standard "
            "error. A renamed key keeps its value and comment, a moved key goes to the end of its new table, "
            "and a removed key is commented out with a note. A file with nothing to change comes back byte for "
            "byte; one with a renamed, moved or removed dotted key or key of an inline table is reformatted, "
            "comments kept; a renamed option changes in place, also inside inline tables.\n"
            "\n"
            "Some renames depend on the case's dimension, which the command reads from the input's mesh as a run "
            "does; when the mesh is not at hand, --dim states it.\n"
            "\n"
            "Renamed keys still work, with a warning, for at least one minor release before they are removed. "
            "mallard check lists deprecated keys as warnings and removed ones as errors.",
        .options = UPGRADE_OPTIONS,
        .arguments = FILE_ARGUMENTS,
        .examples = "$ mallard input upgrade old.toml > new.toml\n"
                    "$ mallard input upgrade --check input.toml\n"
                    "$ mallard input upgrade --dim 3 old.toml > new.toml",
        .exit_status = "0  the file was written (with --check: it is up to date)\n"
                       "1  the file cannot be read or parsed (with --check: or it is out of date)\n"
                       "2  usage error",
        .dim = Case::front_end,
        .run = upgrade_main,
    },
    {
        .name = "check",
        .group = "Inspect",
        .summary = "Set up a case without running it and report on it: keys, mesh, time step, memory, cost",
        .synopsis = "mallard check input.toml [options]",
        .description =
            "Runs the whole setup of a case (mesh, partition, boundaries, numerics, initial solution) without "
            "time stepping and without writing any file, then reports: unknown or unused keys, boundary zones "
            "without a condition, mesh quality with the worst cells, the stable time step and the cell that "
            "sets it, memory per rank and, with --throughput or --steps, the cost. It lists every region the case "
            "uses and the mesh's own (domain, boundary zones, cell zones) with its dimension, measure and cells, "
            "and whether a surface or curve is exact or extracted; a region that covers no cell is an error. Under "
            "mpirun -n N it checks the run on N ranks.",
        .options = CHECK_OPTIONS,
        .arguments = CHECK_ARGUMENTS,
        .examples = "$ mallard check input.toml\n"
                    "$ mallard check input.toml --ranks 64 --throughput 5e7\n"
                    "$ mpirun -n 4 mallard check input.toml --quality quality\n"
                    "$ mallard check input.toml --view view   # then open view/case.vtkhdf in ParaView",
        .exit_status = "0  no errors (there may be warnings)\n"
                       "1  errors: in the input, the mesh or the setup\n"
                       "2  usage error",
        .dim = Case::input,
        .run = check_main,
    },
    {
        .name = "mesh convert",
        .group = "Mesh",
        .summary = "Write a mesh file, or the mesh of an input file, as a Mallard HDF5 mesh",
        .synopsis = "mallard mesh convert INPUT OUTPUT.h5",
        .description =
            "Writes a mesh as a Mallard HDF5 mesh file, which large runs read in parallel. INPUT is a mesh file "
            "(Gmsh or Fluent .msh, Fluent .cas, CGNS .cgns, OpenFOAM polyMesh directory), or an input file "
            "(.toml) whose [mesh] table describes the mesh (with its [mesh.transform]). Under MPI, the ranks "
            "write in parallel (needs parallel HDF5); each reads the whole mesh file but generates only its "
            "block. Needs a build with HDF5.",
        .options = {},
        .arguments = CONVERT_ARGUMENTS,
        .examples = "$ mallard mesh convert wing.msh wing.h5\n"
                    "$ mpirun -n 8 mallard mesh convert input.toml mesh.h5",
        .exit_status = "0  written\n"
                       "1  an error reading or writing, or a build without HDF5\n"
                       "2  usage error",
        .dim = Case::mesh,
        .run = mesh_convert_main,
    },
    {
        .name = "test",
        .group = "Maintain",
        .summary = "Run the unit and regression tests",
        .synopsis = "mallard test [--dim 2|3] [GoogleTest options] [Kokkos options]",
        .description =
            "Runs Mallard's test suites, mallard-test2d and mallard-test3d, one after the other: those next to "
            "mallard (an install tree) or in the test directory of its build tree. Multi-rank tests run under "
            "MPI: mpirun -n 3 mallard test --gtest_filter='MPITest.*'.",
        .options = TEST_OPTIONS,
        .arguments = {},
        .examples = "$ mallard test\n"
                    "$ mallard test --dim 3 --gtest_filter='Sod*' --kokkos-num-threads=2\n"
                    "$ mallard test --gtest_list_tests",
        .exit_status = "0  every test passed\n"
                       "1  a test failed, or there is no test suite\n"
                       "2  usage error",
        .dim = Case::front_end,
        .run = front_end::test_main,
        .run_after_help = true,
    },
    {
        .name = "--version",
        .group = "Maintain",
        .summary = "Print the version",
        .synopsis = "mallard --version",
        .description =
            "Prints the version of this build: the git description of the commit it was built from, else the "
            "release.",
        .options = {},
        .arguments = {},
        .examples = "$ mallard --version",
        .exit_status = "0  always",
        .run = version_main,
    },
};

constexpr Family FAMILIES[] = {
    {"input", "Commands on input files."},
    {"mesh", "Commands on meshes."},
};

constexpr std::string_view GROUPS[] = {"Run", "Set up", "Inspect", "Mesh", "Post", "Share", "Maintain"};

} // namespace

std::span<const Command> commands() { return COMMANDS; }
std::span<const Family> families() { return FAMILIES; }
std::span<const std::string_view> groups() { return GROUPS; }
std::span<const Option> global_options() { return GLOBAL_OPTIONS; }

} // namespace cli
