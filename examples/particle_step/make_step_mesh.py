#!/usr/bin/env python3
"""Write the Gmsh 2.2 quadrilateral mesh of particle_step: a planar backward-facing step of height 1 and
expansion ratio 5:3 (the geometry of Fessler & Eaton 1999).

    python make_step_mesh.py step.msh [--cells-per-h 24] [--inlet 20] [--fine 14] [--length 30] [--growth 1.04]

The step corner is at the origin: the inlet channel spans y = 1 to 2.5 from x = -inlet to 0, the channel after
the step y = 0 to 2.5 from x = 0 to length. Cells are squares of side 1 / cells-per-h for 0 <= x <= fine, and
grow by the factor growth per cell upstream (the inlet's fully developed flow does not change along x) and
downstream (they damp the eddies before the outlet), to at most 6 times as long as they are tall.
Physical curves: "inlet", "outlet" and "wall" (both walls and the face of the step).
"""
import argparse

import numpy as np


def graded(length, dx0, growth, dx_max):
    """Node offsets 0 ... length, the first cell dx0, then growing geometrically up to dx_max."""
    dx = [dx0]
    while sum(dx) < length:
        dx.append(min(dx[-1] * growth, dx_max))
    s = np.concatenate([[0.0], np.cumsum(dx)])
    return s * length / s[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--cells-per-h", type=int, default=24)
    ap.add_argument("--inlet", type=float, default=20.0)
    ap.add_argument("--fine", type=float, default=14.0)
    ap.add_argument("--length", type=float, default=30.0)
    ap.add_argument("--growth", type=float, default=1.04)
    args = ap.parse_args()

    n = args.cells_per_h
    dx0 = 1.0 / n
    y_low = np.linspace(0.0, 1.0, n + 1)
    y_up = np.linspace(1.0, 2.5, 3 * n // 2 + 1)
    x_in = -graded(args.inlet, dx0, args.growth, 6 * dx0)[::-1]
    x_out = np.concatenate([np.linspace(0.0, args.fine, round(args.fine * n) + 1),
                            args.fine + graded(args.length - args.fine, dx0, args.growth, 6 * dx0)[1:]])
    x = np.concatenate([x_in, x_out[1:]])
    i0 = len(x_in) - 1  # the column of nodes at x = 0
    y = np.concatenate([y_low, y_up[1:]])
    j1 = n  # the row of nodes at y = 1

    ids = -np.ones((len(x), len(y)), dtype=np.int64)
    for i in range(len(x)):
        for j in range(len(y)):
            if i >= i0 or j >= j1:
                ids[i, j] = 0
    ids[ids == 0] = np.arange(1, np.count_nonzero(ids == 0) + 1)
    nodes = [f"{ids[i, j]} {x[i]:.15g} {y[j]:.15g} 0" for i in range(len(x)) for j in range(len(y)) if ids[i, j] > 0]
    nodes.sort(key=lambda s: int(s.split()[0]))

    elements = []
    ni, nj = len(x) - 1, len(y) - 1
    for j in range(j1, nj):
        elements.append(f"1 2 1 1 {ids[0, j]} {ids[0, j + 1]}")
    for j in range(nj):
        elements.append(f"1 2 2 2 {ids[ni, j]} {ids[ni, j + 1]}")
    for i in range(ni):
        elements.append(f"1 2 3 3 {ids[i, nj]} {ids[i + 1, nj]}")
        jb = j1 if i < i0 else 0
        elements.append(f"1 2 3 3 {ids[i, jb]} {ids[i + 1, jb]}")
    for j in range(j1):
        elements.append(f"1 2 3 3 {ids[i0, j]} {ids[i0, j + 1]}")
    n_quads = 0
    for i in range(ni):
        for j in range(j1 if i < i0 else 0, nj):
            elements.append(f"3 2 0 1 {ids[i, j]} {ids[i + 1, j]} {ids[i + 1, j + 1]} {ids[i, j + 1]}")
            n_quads += 1

    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
             "$PhysicalNames", "3", '1 1 "inlet"', '1 2 "outlet"', '1 3 "wall"', "$EndPhysicalNames",
             "$Nodes", str(len(nodes)), *nodes, "$EndNodes",
             "$Elements", str(len(elements)), *[f"{k + 1} {e}" for k, e in enumerate(elements)], "$EndElements"]
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.output}: {n_quads} quads, x from {x[0]:g} to {x[-1]:g}, {len(x_in) - 1} columns upstream")


if __name__ == "__main__":
    main()
