"""Reader for Mallard's CSV output: probes, probe reductions, wall forces and
domain integrals. Each file is a header row, a "# units: ..." line with the
units of each column (UDUNITS syntax, "1" for dimensionless), and the rows,
with a "# provenance {...}" line where each run starts writing."""
import numpy as np


def read_csv(path):
    """The columns of a CSV file as {name: array}, and their units as
    {name: units} (empty for files written before Mallard recorded units).
    Rows of a run continued from a restart follow those of the first run."""
    header, units, rows = None, {}, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if header is None:
                header = line.split(",")
            elif line.startswith("# units: "):
                units = dict(zip(header, line[len("# units: "):].split(",")))
            elif line and not line.startswith("#"):
                rows.append([float(v) for v in line.split(",")])
    data = np.array(rows, dtype=float).reshape(-1, len(header or []))
    return {name: data[:, i] for i, name in enumerate(header or [])}, units


def units(path):
    """{column: units} of a CSV file."""
    return read_csv(path)[1]
