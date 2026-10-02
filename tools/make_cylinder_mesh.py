#!/usr/bin/env python3
"""Write a Gmsh 2.2 O-grid of quadrilaterals around a circular cylinder.

    make_cylinder_mesh.py OUTPUT.msh [--n-theta 256] [--n-r 128] [--r-far 20] [--dr0 0.004]

The cylinder has diameter 1 and is centered at the origin. Radial spacing
grows geometrically from dr0 at the wall to the far-field radius. Physical
curves: "cylinder" (the wall) and "farfield" (the outer circle).
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
    args = ap.parse_args()

    r0, n_r, n_t = 0.5, args.n_r, args.n_theta
    # Geometric growth ratio q with dr0 * (q^n_r - 1) / (q - 1) = r_far - r0
    span = args.r_far - r0
    q = brentq(lambda q: args.dr0 * (q ** n_r - 1) / (q - 1) - span, 1.0 + 1e-9, 2.0)
    r = r0 + args.dr0 * (q ** np.arange(n_r + 1) - 1) / (q - 1)
    theta = np.linspace(0.0, 2.0 * np.pi, n_t, endpoint=False)

    def node(i, j):  # i: angle, j: radius
        return j * n_t + (i % n_t) + 1

    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
             "$PhysicalNames", "2", '1 1 "cylinder"', '1 2 "farfield"', "$EndPhysicalNames",
             "$Nodes", str((n_r + 1) * n_t)]
    for j in range(n_r + 1):
        for i in range(n_t):
            lines.append(f"{node(i, j)} {r[j] * np.cos(theta[i]):.15g} {r[j] * np.sin(theta[i]):.15g} 0")
    lines.append("$EndNodes")

    elements = []
    for i in range(n_t):
        elements.append(f"1 2 1 1 {node(i, 0)} {node(i + 1, 0)}")
        elements.append(f"1 2 2 2 {node(i, n_r)} {node(i + 1, n_r)}")
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
