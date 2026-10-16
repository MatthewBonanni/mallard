#!/usr/bin/env python3
"""Geometry and mesh figures of the cone tutorial.

    python plot_mesh.py cone.msh [--out DIR]

Writes cone_geometry (a side view of the domain with the cone, the shock
and the boundary zones) and cone_mesh (the boundary zones in 3D and the
triangles of the symmetry plane z = 0), each for light and dark pages.
"""
import argparse
import os
import sys

import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import matplotlib.pyplot as plt  # noqa: E402
from tutfig import label_box, render  # noqa: E402
from gmsh_mesh import read_msh  # noqa: E402
from taylor_maccoll import solve  # noqa: E402

ZONES = ["cone", "inlet", "outlet", "outer", "symmetry_y", "symmetry_z"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mesh")
    ap.add_argument("--out", help="output directory (default: docs/images/tutorials)")
    ap.add_argument("--mach", type=float, default=2.0)
    ap.add_argument("--cone", type=float, default=15.0)
    args = ap.parse_args()
    xyz, cells, zones = read_msh(args.mesh)
    beta, _ = solve(args.mach, args.cone)
    x0, x1, R = xyz[:, 0].min(), xyz[:, 0].max(), xyz[:, 1].max()
    tc = np.tan(np.radians(args.cone))

    def geometry(t):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        s = t["series"]
        ax.plot([x0, x1], [R, R], color=s[0], lw=3)
        ax.plot([x0, x0], [0, R], color=s[2], lw=3)
        ax.plot([x1, x1], [tc * x1, R], color=s[3], lw=3)
        ax.plot([x0, 0], [0, 0], color=s[4], lw=3)
        ax.fill([0, x1, x1], [0, tc * x1, 0], color=s[1])
        ax.plot([0, x1], [0, np.tan(beta) * x1], color=t["ink2"], lw=1.5, ls="--")
        ax.text(0.62, np.tan(beta) * 0.62 + 0.03, f"shock, {np.degrees(beta):.1f}°", rotation=np.degrees(beta) * 0.9,
                color=t["ink2"], ha="center", va="bottom")
        ax.text(0.5, 0.03, f'"cone", {args.cone:g}°', color=t["surface"], fontsize=10, ha="center")
        ax.text((x0 + x1) / 2, R + 0.03, '"outer"', color=s[0], ha="center", va="bottom")
        ax.text(x0 - 0.02, R / 2, '"inlet"', color=s[2], ha="right", va="center", rotation=90)
        ax.text(x1 + 0.02, (R + tc) / 2, '"outlet"', color=s[3], ha="left", va="center", rotation=90)
        ax.text(x0 / 2, -0.03, "axis", color=s[4], ha="center", va="top", fontsize=9)
        for y in np.linspace(0.12, R - 0.12, 4):
            ax.annotate("", (x0 + 0.13, y), (x0 + 0.01, y), arrowprops={"arrowstyle": "-|>", "color": t["ink2"]})
        ax.text(-0.13, 0.62, f"M = {args.mach:g}", color=t["ink"], bbox=label_box(t))
        ax.text(0.08, R - 0.12, '"symmetry_y" (y = 0) and "symmetry_z" (z = 0):\nthe planes of this quarter domain',
                color=t["ink2"], fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(x0 - 0.12, x1 + 0.12)
        ax.set_ylim(-0.1, R + 0.1)
        ax.set_xlabel("x (cone length 1)")
        ax.set_ylabel("r = (y² + z²)^½")
        ax.set_title("Side view: the domain is the quarter cylinder y ≥ 0, z ≥ 0 around the axis")
        ax.grid(False)
        return fig

    def mesh(t):
        fig = plt.figure(figsize=(11, 4.6))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        colors = dict(zip(ZONES, [t["series"][i] for i in (1, 2, 3, 0, 4, 6)]))
        for name in ("cone", "inlet", "symmetry_y", "symmetry_z"):
            tris = [xyz[f] for f in zones[name]]
            sym = name.startswith("symmetry")
            ax.add_collection3d(Poly3DCollection(tris, facecolor=colors[name], edgecolor=t["mesh"],
                                                 linewidth=0.05 if sym else 0.1, alpha=0.25 if sym else 1.0,
                                                 label=f'"{name}"', rasterized=True))
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, R)
        ax.set_zlim(0, R)
        ax.set_box_aspect((x1 - x0, R, R))
        ax.view_init(elev=24, azim=-58)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0, 0, 0, 0))
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("Boundary zones (outer and outlet hidden)")
        bx = fig.add_subplot(1, 2, 2)
        tris = [xyz[f][:, :2] for f in zones["symmetry_z"]]
        bx.add_collection(PolyCollection(tris, facecolors="none", edgecolors=t["mesh"], linewidths=0.2, rasterized=True))
        bx.plot([0, x1], [0, np.tan(beta) * x1], color=t["series"][1], lw=1.2, ls="--", label="exact shock")
        bx.set_xlim(x0, x1)
        bx.set_ylim(0, R)
        bx.set_aspect("equal")
        bx.grid(False)
        bx.legend(loc="upper left")
        bx.set_title(f'Faces on "symmetry_z" ({len(cells):,} tetrahedra in all)')
        bx.set_xlabel("x")
        bx.set_ylabel("y")
        fig.tight_layout()
        return fig

    render(geometry, args.out, "cone_geometry")
    render(mesh, args.out, "cone_mesh")


if __name__ == "__main__":
    main()
