"""Mallard's commands as functions: each runs the mallard executable with --format json and returns what it
printed, parsed, so that a function and its command cannot disagree.

Keyword arguments are the command's options (mallard COMMAND --help): fields=["RHO", "P"] is --fields RHO,P,
latest=True is --latest, input="case.toml" is -i case.toml. nprocs runs the command under mpirun, threads sets
--kokkos-num-threads.
"""
import json
import os
import subprocess

from . import _lib


class CommandError(RuntimeError):
    """A command that exited with an error: its exit status, output, and what it printed as JSON (None if nothing)."""

    def __init__(self, args, returncode, stdout, stderr, result):
        super().__init__(f"{' '.join(args)} exited with status {returncode}: {stderr.strip() or stdout.strip()}")
        self.args_list = args
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


def command(*args, nprocs=None, mpirun="mpirun", threads=None, cwd=None, **options):
    """Runs mallard ARGS... OPTIONS --format json and returns its JSON output, parsed.
    Raises CommandError when it exits with an error (its result attribute holds the JSON it printed, if any)."""
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
    if p.returncode != 0:
        raise CommandError(argv, p.returncode, p.stdout, p.stderr, result)
    if result is None:
        raise CommandError(argv, p.returncode, p.stdout, p.stderr, None)
    return result


def info(path, **options):
    """mallard info: what a file, or a run directory, holds."""
    return command("info", path, **options)


def diff(a, b, **options):
    """mallard diff: the differences between two files of the same mesh, or of two meshes (remap=True)."""
    return command("diff", a, b, **options)
