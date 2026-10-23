#!/usr/bin/env python3
"""Write a Gmsh 2.2 O-grid of the annulus r_in <= r <= r_out.

    make_annulus_mesh.py OUTPUT.msh [--n-r 8] [--n-theta 128] [--r-in 1] [--r-out 1.384]
        [--triangles] [--order 1|2|3] [--jitter 0]

Quadrilaterals (or, with --triangles, each split along alternating diagonals)
between equispaced circles; nodes lie on the circles. Physical curves: "inner"
and "outer". With --order 2 or 3 the boundary curves are written as
high-order lines (line3, line4) whose extra nodes lie on the circles, so that
Mallard curves the walls from the file; --jitter moves interior nodes randomly
by that fraction of the radial spacing (the walls stay circles).
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--n-r", type=int, default=8)
    ap.add_argument("--n-theta", type=int, default=128)
    ap.add_argument("--r-in", type=float, default=1.0)
    ap.add_argument("--r-out", type=float, default=1.384)
    ap.add_argument("--triangles", action="store_true")
    ap.add_argument("--order", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    n_r, n_t = args.n_r, args.n_theta
    r = np.linspace(args.r_in, args.r_out, n_r + 1)
    theta = 2.0 * np.pi * np.arange(n_t) / n_t
    rng = np.random.default_rng(args.seed)
    dr = (args.r_out - args.r_in) / n_r

    nodes = []

    def node(i, j):  # i: angle, j: radius
        return j * n_t + (i % n_t) + 1

    for j in range(n_r + 1):
        for i in range(n_t):
            rr, tt = r[j], theta[i]
            if 0 < j < n_r and args.jitter > 0:
                rr += args.jitter * dr * rng.uniform(-1, 1)
                tt += args.jitter * dr / rr * rng.uniform(-1, 1)
            nodes.append((rr * np.cos(tt), rr * np.sin(tt)))

    elements = []
    for j, tag in ((0, 1), (n_r, 2)):
        for i in range(n_t):
            a, b = node(i, j), node(i + 1, j)
            if args.order == 1:
                elements.append(f"1 2 {tag} {tag} {a} {b}")
                continue
            t0, t1 = theta[i], theta[i] + 2.0 * np.pi / n_t
            extra = []
            for k in range(1, args.order):
                t = t0 + (t1 - t0) * k / args.order
                nodes.append((r[j] * np.cos(t), r[j] * np.sin(t)))
                extra.append(len(nodes))
            etype = 8 if args.order == 2 else 26
            elements.append(f"{etype} 2 {tag} {tag} {a} {b} " + " ".join(map(str, extra)))
    for j in range(n_r):
        for i in range(n_t):
            q = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            if not args.triangles:
                elements.append("3 2 0 1 " + " ".join(map(str, q)))
            elif (i + j) % 2 == 0:
                elements.append(f"2 2 0 1 {q[0]} {q[1]} {q[2]}")
                elements.append(f"2 2 0 1 {q[0]} {q[2]} {q[3]}")
            else:
                elements.append(f"2 2 0 1 {q[0]} {q[1]} {q[3]}")
                elements.append(f"2 2 0 1 {q[1]} {q[2]} {q[3]}")

    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
             "$PhysicalNames", "2", '1 1 "inner"', '1 2 "outer"', "$EndPhysicalNames",
             "$Nodes", str(len(nodes))]
    lines += [f"{k + 1} {x:.17g} {y:.17g} 0" for k, (x, y) in enumerate(nodes)]
    lines += ["$EndNodes", "$Elements", str(len(elements))]
    lines += [f"{k + 1} {e}" for k, e in enumerate(elements)]
    lines.append("$EndElements")
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    n_cells = n_r * n_t * (2 if args.triangles else 1)
    print(f"wrote {args.output}: {n_cells} cells, boundary order {args.order}")


if __name__ == "__main__":
    main()
