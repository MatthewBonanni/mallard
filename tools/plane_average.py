"""Average the [statistics] fields of a run over homogeneous directions.

    python tools/plane_average.py FLOW.pvtu --average x z [--out profile.csv]

Reads a VTU or PVTU snapshot holding MEAN_<field> and COV_<a>_<b> arrays
(written by [statistics]) of a mesh of axis-aligned quads or hexahedra, such as
the generated `cartesian` meshes (stretched or not), and averages them over the
listed directions as a function of the others: cells are grouped by their
center coordinates in the kept directions, equal to a relative tolerance, and
weighted by their volume. Covariances become those over time and the averaged
directions together,

    <a'b'> = avg(COV_a_b + MEAN_a MEAN_b) - avg(MEAN_a) avg(MEAN_b),

which adds the dispersive part (the spatial variation of the time means) to the
averaged time covariances; a covariance whose two means are not in the file is
averaged as it is. Writes a CSV with the kept coordinates, the number of cells
averaged, then each MEAN_ and COV_ column.
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mallard_vtu import read_vtu_cells  # noqa: E402

AXES = "xyz"


def read_cells(path):
    """Cell centers, volumes (axis-aligned boxes) and statistics arrays of a VTU or all pieces of a PVTU."""
    if path.endswith(".pvtu"):
        pieces = re.findall(r'Source="([^"]+)"', open(path).read())
        parts = [read_cells(os.path.join(os.path.dirname(path), p)) for p in pieces]
        names = parts[0][2].keys()
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                {n: np.concatenate([p[2][n] for p in parts]) for n in names})
    pts, conn, offs, _, arrays = read_vtu_cells(path)
    n_nodes = np.diff(np.concatenate([[0], offs]))
    if not np.all(n_nodes == n_nodes[0]) or n_nodes[0] not in (4, 8):
        raise SystemExit(f"{path}: expected a mesh of quads or hexahedra")
    nodes = pts[conn.reshape(-1, n_nodes[0])]
    lo, hi = nodes.min(axis=1), nodes.max(axis=1)
    dim = 3 if n_nodes[0] == 8 else 2
    centers = 0.5 * (lo + hi)[:, :dim]
    volumes = np.prod((hi - lo)[:, :dim], axis=1)
    stats = {}
    for name, values in arrays.items():
        if not (name.startswith("MEAN_") or name.startswith("COV_")):
            continue
        if values.ndim == 2:  # a vector, e.g. MEAN_U
            for d in range(dim):
                stats[f"{name}_{AXES[d].upper()}"] = values[:, d]
        else:
            stats[name] = values
    return centers, volumes, stats


def group(values, rtol):
    """Group index of each value: a sorted value joins the group of its predecessor if within rtol of their scale."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    tol = rtol * max(np.ptp(values), np.abs(values).max(), 1e-300)
    starts = np.concatenate([[True], np.diff(v) > tol])
    ids = np.empty(len(values), dtype=np.int64)
    ids[order] = np.cumsum(starts) - 1
    return ids


def covariance_means(name, means):
    """The two MEAN_ names of COV_<a>_<b>, or None."""
    body = name[len("COV_"):]
    for k in range(1, len(body)):
        if body[k] == "_" and f"MEAN_{body[:k]}" in means and f"MEAN_{body[k + 1:]}" in means:
            return f"MEAN_{body[:k]}", f"MEAN_{body[k + 1:]}"
    return None


def plane_average(centers, volumes, stats, average, rtol=1e-6):
    """Kept coordinates (n_groups, n_kept), cell counts and averaged columns."""
    dim = centers.shape[1]
    kept = [d for d in range(dim) if AXES[d] not in average]
    if not kept:
        raise SystemExit("--average must leave at least one direction")
    ids = np.zeros(len(volumes), dtype=np.int64)
    for d in kept:
        g = group(centers[:, d], rtol)
        ids = ids * (g.max() + 1) + g
    _, ids = np.unique(ids, return_inverse=True)
    n_groups = ids.max() + 1
    weight = np.bincount(ids, volumes, n_groups)

    def avg(values):
        return np.bincount(ids, volumes * values, n_groups) / weight

    coords = np.stack([avg(centers[:, d]) for d in kept], axis=1)
    columns = {}
    means = {n: v for n, v in stats.items() if n.startswith("MEAN_")}
    for name, values in means.items():
        columns[name] = avg(values)
    for name, values in stats.items():
        if not name.startswith("COV_"):
            continue
        pair = covariance_means(name, means)
        if pair is None:
            columns[name] = avg(values)
        else:
            a, b = pair
            columns[name] = avg(values + means[a] * means[b]) - columns[a] * columns[b]
    order = np.lexsort(coords.T[::-1])
    return ([AXES[d] for d in kept], coords[order], np.bincount(ids, minlength=n_groups)[order],
            {n: c[order] for n, c in columns.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", help="VTU or PVTU file with MEAN_ and COV_ arrays")
    ap.add_argument("--average", nargs="+", required=True, choices=list(AXES), help="homogeneous directions")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="relative tolerance for equal cell-center coordinates (default 1e-6)")
    ap.add_argument("--out", help="output CSV (default: standard output)")
    args = ap.parse_args()

    centers, volumes, stats = read_cells(args.snapshot)
    if not stats:
        raise SystemExit(f"{args.snapshot}: no MEAN_ or COV_ arrays")
    axes, coords, counts, columns = plane_average(centers, volumes, stats, args.average, args.rtol)
    names = list(columns)
    out = open(args.out, "w") if args.out else sys.stdout
    out.write(",".join(axes + ["n_cells"] + names) + "\n")
    for k in range(len(counts)):
        row = [f"{c:.17g}" for c in coords[k]] + [str(counts[k])] + [f"{columns[n][k]:.17g}" for n in names]
        out.write(",".join(row) + "\n")
    if args.out:
        out.close()


if __name__ == "__main__":
    main()
