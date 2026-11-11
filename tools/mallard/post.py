"""mallard post: commands on a run's output (see mallard.commands for how options are given)."""
from .commands import _paths, command


def extract(files, **options):
    """mallard post extract: probes, region integrals and snapshots of existing output, as the run would have
    written them. Returns the report; the files it wrote are under output (default extract/), for mallard.open."""
    return command("post", "extract", *_paths(files), **options)


def convergence(runs, **options):
    """mallard post convergence: observed order, extrapolated value and GCI of runs at several resolutions."""
    return command("post", "convergence", *_paths(runs), **options)
