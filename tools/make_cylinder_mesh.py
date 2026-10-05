#!/usr/bin/env python3
"""Write a Gmsh 2.2 O-grid of quadrilaterals around a circular cylinder.

    make_cylinder_mesh.py OUTPUT.msh [--n-theta 256] [--n-r 128] [--r-far 20] [--dr0 0.004] [--half]

The cylinder has diameter 1 and is centered at the origin. Radial spacing
grows geometrically from dr0 at the wall to the far-field radius. Physical
curves: "cylinder" (the wall) and "farfield" (the outer circle). With --half,
only the upper half (y >= 0, n-theta cells over 180 degrees) with the two
cuts along y = 0 as "axis": the meridian plane of a sphere for axisymmetric
runs.
"""
import argparse

import numpy as np
from scipy.optimize import brentq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--n-theta", type=int, default=256)
    ap.add_argument("--n-r", type=int, default=128)
    ap.add_argument("--r-far", type=float, default=20.0)
    ap.add_argument("--dr0", type=float, default=0.004)
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()

    r0, n_r, n_t = 0.5, args.n_r, args.n_theta
    # Geometric growth ratio q with dr0 * (q^n_r - 1) / (q - 1) = r_far - r0
    span = args.r_far - r0
    q = brentq(lambda q: args.dr0 * (q ** n_r - 1) / (q - 1) - span, 1.0 + 1e-9, 2.0)
    r = r0 + args.dr0 * (q ** np.arange(n_r + 1) - 1) / (q - 1)
    if args.half:
        n_nodes_t = n_t + 1
        theta = np.linspace(0.0, np.pi, n_nodes_t)
    else:
        n_nodes_t = n_t
        theta = np.linspace(0.0, 2.0 * np.pi, n_t, endpoint=False)

    def node(i, j):  # i: angle, j: radius
        return j * n_nodes_t + (i % n_nodes_t) + 1

    names = ['1 1 "cylinder"', '1 2 "farfield"'] + (['1 3 "axis"'] if args.half else [])
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
             "$PhysicalNames", str(len(names))] + names + ["$EndPhysicalNames",
             "$Nodes", str((n_r + 1) * n_nodes_t)]
    for j in range(n_r + 1):
        for i in range(n_nodes_t):
            y = 0.0 if args.half and i in (0, n_t) else r[j] * np.sin(theta[i])
            lines.append(f"{node(i, j)} {r[j] * np.cos(theta[i]):.15g} {y:.15g} 0")
    lines.append("$EndNodes")

    elements = []
    for i in range(n_t):
        elements.append(f"1 2 1 1 {node(i, 0)} {node(i + 1, 0)}")
        elements.append(f"1 2 2 2 {node(i, n_r)} {node(i + 1, n_r)}")
    if args.half:
        for j in range(n_r):
            elements.append(f"1 2 3 3 {node(0, j)} {node(0, j + 1)}")
            elements.append(f"1 2 3 3 {node(n_t, j)} {node(n_t, j + 1)}")
    for j in range(n_r):
        for i in range(n_t):
            elements.append(f"3 2 0 1 {node(i, j)} {node(i, j + 1)} {node(i + 1, j + 1)} {node(i + 1, j)}")
    lines += ["$Elements", str(len(elements))]
    lines += [f"{k + 1} {e}" for k, e in enumerate(elements)]
    lines.append("$EndElements")
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.output}: {n_r * n_t} quads, growth ratio {q:.4f}, first cell {args.dr0}")


if __name__ == "__main__":
    main()
