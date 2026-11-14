#!/usr/bin/env python3
"""Write a Gmsh 2.2 mesh of quadrilaterals of the upper half of a planar
converging-diverging nozzle.

    make_nozzle_mesh.py OUTPUT.msh [--nx 320] [--ny 20]

The half-height is h(x) = 1 + 0.6 (x / 2 - 1)^2 on 0 <= x <= 4: throat of
height 1 at x = 2, inlet and exit 1.6 throat heights. Physical curves:
"inlet" (x = 0), "outlet" (x = 4), "axis" (y = 0) and "wall".
"""
import argparse


def height(x):
    return 1.0 + 0.6 * (0.5 * x - 1.0) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--nx", type=int, default=320)
    ap.add_argument("--ny", type=int, default=20)
    args = ap.parse_args()
    nx, ny = args.nx, args.ny

    def node(i, j):
        return j * (nx + 1) + i + 1

    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "4",
             '1 1 "axis"', '1 2 "outlet"', '1 3 "wall"', '1 4 "inlet"', "$EndPhysicalNames",
             "$Nodes", str((nx + 1) * (ny + 1))]
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = 4.0 * i / nx
            lines.append(f"{node(i, j)} {x:.17g} {height(x) * j / ny:.17g} 0")
    lines.append("$EndNodes")
    elements = []
    for i in range(nx):
        elements.append(f"1 2 1 1 {node(i, 0)} {node(i + 1, 0)}")
        elements.append(f"1 2 3 3 {node(i + 1, ny)} {node(i, ny)}")
    for j in range(ny):
        elements.append(f"1 2 4 4 {node(0, j + 1)} {node(0, j)}")
        elements.append(f"1 2 2 2 {node(nx, j)} {node(nx, j + 1)}")
    for j in range(ny):
        for i in range(nx):
            elements.append(f"3 2 0 1 {node(i, j)} {node(i + 1, j)} {node(i + 1, j + 1)} {node(i, j + 1)}")
    lines += ["$Elements", str(len(elements))]
    lines += [f"{k + 1} {e}" for k, e in enumerate(elements)]
    lines.append("$EndElements")
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
