"""Mallard's commands as functions: each runs the mallard executable with `--format json` and returns what it
printed, parsed, so that a function and its command cannot disagree."""
import json
import os
import subprocess

from . import _lib


class CommandError(RuntimeError):
    """A command that exited with an error (exit status 1: an error; 2: a usage error, such as an unknown
    option).

    - `returncode`: the exit status.
    - `argv`: the command line run.
    - `stdout`, `stderr`: what it printed.
    - `result`: the JSON it printed, parsed, or `None`.
    """

    def __init__(self, argv, returncode, stdout, stderr, result):
        super().__init__(f"{' '.join(argv)} exited with status {returncode}: {stderr.strip() or stdout.strip()}")
        self.argv = argv
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.result = result


def _flags(options):
    out = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        out.append("-i" if key == "input" else "--" + key.replace("_", "-"))
        if value is True:
            continue
        if isinstance(value, (list, tuple)):
            out.append(",".join(str(v) for v in value))
        else:
            out.append(os.fspath(value) if isinstance(value, os.PathLike) else str(value))
    return out


def _paths(paths):
    return [os.fspath(paths)] if isinstance(paths, (str, os.PathLike)) else [os.fspath(p) for p in paths]


def command(*args, nprocs=None, mpirun="mpirun", threads=None, cwd=None, verdicts=(), **options):
    """Runs `mallard ARGS... OPTIONS --format json` and returns its JSON output, parsed.

    Keyword arguments are the command's options, as `mallard COMMAND --help` lists them: `fields=["RHO", "P"]`
    is `--fields RHO,P`, `latest=True` is `--latest`, `mesh_a="a.msh"` is `--mesh-a a.msh`, `input="case.toml"`
    is `-i case.toml`; `None` and `False` leave an option out. `nprocs` runs the command on that many MPI ranks
    with `mpirun -n` (`mpirun` names another launcher), `threads` sets `--kokkos-num-threads`, and `cwd` is the
    directory it runs in (relative paths in a file's run block, such as its mesh file, resolve from there).
    The executable is `MALLARD_EXECUTABLE`, else `mallard` on `PATH`. Raises `CommandError` when the command
    exits with an error; an exit status in `verdicts` is a result instead (`mallard diff`'s 1: a difference
    beyond tolerance), returned with the JSON the command printed."""
    argv = [_lib.executable()] + [os.fspath(a) for a in args] + _flags(options) + ["--format", "json"]
    if threads is not None:
        argv.append(f"--kokkos-num-threads={threads}")
    if nprocs is not None:
        argv = [mpirun, "-n", str(nprocs)] + argv
    p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    try:
        result = json.loads(p.stdout) if p.stdout.strip() else None
    except json.JSONDecodeError:
        result = None
    if (p.returncode != 0 and p.returncode not in verdicts) or result is None:
        raise CommandError(argv, p.returncode, p.stdout, p.stderr, result)
    return result


def info(path, **options):
    """`mallard info PATH`: what a file holds (as `mallard.open(path).contents`, with each field's `min` and
    `max` at the last snapshot), or a run directory's files and its start and end."""
    return command("info", path, **options)


def diff(a, b, **options):
    """`mallard diff A B`: norms of the differences between two files, field by field, with the largest ones
    and where they are; `remap=True` compares files on different meshes. Options: `fields`, `tol`, `step`, `time`,
    `snapshot`, `remap`, `mesh_a`, `mesh_b` (`mallard diff --help`). Differences beyond tolerance are a
    result, not an error: the report's `result` and `exit_status` give the verdict."""
    return command("diff", a, b, verdicts=(1,), **options)
