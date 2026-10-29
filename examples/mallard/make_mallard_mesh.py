#!/usr/bin/env python3
"""Write a Gmsh 2.2 triangle mesh around a 2D flying mallard silhouette.

    make_mallard_mesh.py OUTPUT.msh [--h-wall 0.0015] [--h-far 0.0045] [--preview duck.png]

The duck flies in the -x direction (bill tip at the origin, length about 1.1)
with its wing raised, inside the rectangle [x0, x1] x [y0, y1]. Its outline is
a periodic cubic spline through hand-placed control points. The cell size grows
linearly with the distance from the body from h_wall to h_far. Physical curves:
"duck" (the body), "inflow" (left), "outflow" (right), "top" and "bottom".
Needs the gmsh Python module (pip install gmsh).
"""
import argparse

import numpy as np
from scipy.interpolate import splev, splprep

# Side profile facing +x (mirrored below), counterclockwise from the bill tip
CONTROL_POINTS = [
    (1.000, 0.100),  # bill tip
    (0.985, 0.112), (0.945, 0.118), (0.895, 0.128),  # bill top
    (0.868, 0.150), (0.852, 0.195),  # forehead
    (0.815, 0.232), (0.760, 0.240),  # crown
    (0.712, 0.215), (0.690, 0.165),  # back of the head
    (0.665, 0.115), (0.610, 0.088),  # nape
    (0.545, 0.085), (0.500, 0.100),  # shoulder
    (0.465, 0.150), (0.410, 0.245), (0.335, 0.350),  # wing leading edge
    (0.240, 0.450), (0.130, 0.540),
    (0.020, 0.600),  # wing tip
    (0.045, 0.565), (0.105, 0.500),  # primaries
    (0.160, 0.420), (0.205, 0.320), (0.250, 0.210),  # wing trailing edge
    (0.290, 0.120),  # wing root
    (0.200, 0.082), (0.100, 0.062),  # rump
    (0.020, 0.052), (-0.060, 0.046),  # tail top
    (-0.100, 0.030),  # tail tip
    (-0.070, 0.006), (0.000, -0.020),  # under tail
    (0.080, -0.058), (0.180, -0.092),  # belly
    (0.300, -0.110), (0.420, -0.100),
    (0.530, -0.068), (0.610, -0.025),  # breast
    (0.680, 0.020), (0.740, 0.048),  # neck front
    (0.800, 0.064), (0.850, 0.072),  # chin
    (0.910, 0.078), (0.970, 0.082),  # bill underside
]


def outline(n=600):
    """Closed outline facing -x, bill tip at the origin; (n, 2), counterclockwise."""
    p = np.array(CONTROL_POINTS + [CONTROL_POINTS[0]])
    tck, _ = splprep([p[:, 0], p[:, 1]], s=0, per=1)
    x, y = splev(np.linspace(0, 1, n, endpoint=False), tck)
    xy = np.column_stack([1.0 - np.asarray(x), np.asarray(y) - CONTROL_POINTS[0][1]])
    return xy[::-1]  # mirroring reversed the orientation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--h-wall", type=float, default=0.0015, help="Cell size at the body")
    ap.add_argument("--h-far", type=float, default=0.0045, help="Cell size far from the body")
    ap.add_argument("--growth", type=float, default=0.05, help="Cell size increase per unit distance")
    ap.add_argument("--domain", type=float, nargs=4, default=[-0.5, 3.0, -1.1, 1.5],
                    metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--n-spline", type=int, default=400, help="Spline points along the outline")
    ap.add_argument("--preview", help="Also save a plot of the outline")
    args = ap.parse_args()

    duck = outline(args.n_spline)
    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill(duck[:, 0], duck[:, 1], color="k")
        ax.set_aspect("equal")
        fig.savefig(args.preview, dpi=150)

    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("mallard")
    geo = gmsh.model.geo
    x0, x1, y0, y1 = args.domain
    corners = [geo.addPoint(x, y, 0) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
    bottom, outflow, top, inflow = (geo.addLine(corners[i], corners[(i + 1) % 4]) for i in range(4))
    duck_pts = [geo.addPoint(x, y, 0) for x, y in duck]
    body = geo.addSpline(duck_pts + [duck_pts[0]])
    outer = geo.addCurveLoop([bottom, outflow, top, inflow])
    inner = geo.addCurveLoop([body])
    surface = geo.addPlaneSurface([outer, inner])
    geo.synchronize()
    for name, curve in [("duck", body), ("inflow", inflow), ("outflow", outflow), ("top", top), ("bottom", bottom)]:
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [curve]), name)
    gmsh.model.addPhysicalGroup(2, [surface])

    field = gmsh.model.mesh.field
    dist = field.add("Distance")
    field.setNumbers(dist, "CurvesList", [body])
    field.setNumber(dist, "Sampling", 4000)
    size = field.add("MathEval")
    field.setString(size, "F", f"Min({args.h_far}, {args.h_wall} + {args.growth} * F{dist})")
    field.setAsBackgroundMesh(size)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(2)
    gmsh.write(args.output)
    _, tags, _ = gmsh.model.mesh.getElements(2)
    print(f"wrote {args.output}: {sum(len(t) for t in tags)} triangles")
    gmsh.finalize()


if __name__ == "__main__":
    main()
