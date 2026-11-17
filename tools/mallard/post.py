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


def derive(files, fields, **options):
    """`mallard post derive FILE... --fields LIST`: snapshots of derived fields (`MACH`, `P0`, `Q_CRITERION`, ...)
    of existing output, as the run would have written them. `files` is a path or a list of them (restart files,
    snapshots, series); `fields` the fields to derive, as `mallard list fields` names them. Options: `input`,
    `reference`, `mixture`, `with_state`, `region`, `snapshot`, `name`, `steps`, `times`, `every`, `latest`,
    `output` (`mallard post derive --help`). Returns the report: the snapshots derived and the files written
    (under `output`, default `derived/`, for `mallard.open`), with their fields and units."""
    return command("post", "derive", *_paths(files), fields=fields, **options)


def average(files, **options):
    """`mallard post average FILE...`: means `MEAN_<a>`, `RMS_<a>` and covariances `COV_<a>_<b>` over saved
    snapshots, as `[statistics]` keeps them. `files` is a path or a list of them. Options: `fields`, `products`,
    `favre`, `t_start`, `homogeneous`, `conditional`, `statistics`, `rtol`, `region`, `reference`, `mixture`,
    `input`, `snapshot`, `name`, `steps`, `times`, `every`, `latest`, `output` (`mallard post average --help`);
    `conditional`, `region`, `reference` and `mixture` take inline TOML tables as strings. Returns the report:
    the averaging window, each snapshot with its weight, the fields and covariances, and the paths written
    (under `output`, default `average/`)."""
    return command("post", "average", *_paths(files), **options)


def convergence(runs, **options):
    """`mallard post convergence RUN_DIR...`: for each quantity of the runs' forces, integrals and probe tables,
    the observed order of accuracy, the value extrapolated to zero cell size and the grid convergence index.
    Options: `quantity`, `time`, `average`, `order`, `safety_factor`, `sizes`, `field`, `exact`
    (`mallard post convergence --help`). A quantity that diverges is a result, not an error (exit status 1
    with a report): each quantity's verdict says so."""
    return command("post", "convergence", *_paths(runs), verdicts=(1,), **options)
