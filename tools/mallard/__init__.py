"""Mallard's output as numpy arrays, and its commands as functions.

mallard.open reads every file Mallard writes or reads through the solver's own reader (libmallard2d/3d, a build
with -DMallard_ENABLE_PYTHON=ON); mallard.info, mallard.diff and mallard.post run the mallard executable. The
modules vtkhdf, vtu, h5 and tables are the h5py and numpy readers the scripts in tools/ use.
"""
from . import post
from ._version import __version__
from .commands import CommandError, command, diff, info
from .reader import Field, File, Geometry, Snapshot, open, run_files, run_record

__all__ = ["CommandError", "Field", "File", "Geometry", "Snapshot", "__version__", "command", "diff", "info",
           "open", "post", "run_files", "run_record"]
