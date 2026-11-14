"""`mallard post`: commands on a run's output, as functions (options as for `mallard.command`)."""
from .commands import _paths, command


def extract(files, **options):
    """`mallard post extract FILE...`: probes, region integrals and snapshots of existing output, as the run
    would have written them. `files` is a path or a list of them (restart files, snapshots of the whole mesh,
    series). Options: `input`, `probe`, `region`, `name`, `fields`, `reductions`, `sampling`, `snapshot`,
    `steps`, `times`, `every`, `latest`, `output` (`mallard post extract --help`). Returns the report: the
    files written (under `output`, default `extract/`, for `mallard.open`), their times and fields, and the
    reductions of the last snapshot."""
    return command("post", "extract", *_paths(files), **options)


def convergence(runs, **options):
    """`mallard post convergence RUN_DIR...`: for each quantity of the runs' forces, integrals and probe tables,
    the observed order of accuracy, the value extrapolated to zero cell size and the grid convergence index.
    Options: `quantity`, `time`, `average`, `order`, `safety_factor`, `sizes`, `field`, `exact`
    (`mallard post convergence --help`). A quantity that diverges is a result, not an error (exit status 1
    with a report): each quantity's verdict says so."""
    return command("post", "convergence", *_paths(runs), verdicts=(1,), **options)
