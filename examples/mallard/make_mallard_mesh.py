#!/usr/bin/env python3
"""Write a Gmsh 2.2 triangle mesh around a 2D flying mallard silhouette.

    make_mallard_mesh.py OUTPUT.msh [--h-wall 0.0015] [--h-far 0.0045] [--order 1|2|3] [--preview duck.png]

The duck flies in the -x direction (bill tip at the origin, length about 1.1)
with its wings raised, inside the rectangle [x0, x1] x [y0, y1]. Its outline is
a periodic cubic spline through hand-placed control points. The cell size grows
linearly with the distance from the body from h_wall to h_far. Physical curves:
"duck" (the body), "inflow" (left), "outflow" (right), "top" and "bottom".
With --order 2 or 3 the elements are curved onto the spline, so Mallard's
curved boundaries follow the duck exactly instead of its straight edges.
Needs the gmsh Python module (pip install gmsh).
"""
import argparse

import numpy as np
from scipy.interpolate import splev, splprep

# Side profile facing +x (mirrored below), counterclockwise from the bill tip,
# traced from the logo's duck
CONTROL_POINTS = [
    (1.000, 0.100), (1.021, 0.125), (0.951, 0.135), (0.900, 0.144), (0.865, 0.172),
    (0.824, 0.202), (0.785, 0.206), (0.736, 0.199), (0.683, 0.172), (0.639, 0.130),
    (0.566, 0.055), (0.526, 0.099), (0.513, 0.147), (0.494, 0.215), (0.457, 0.264),
    (0.374, 0.318), (0.280, 0.370), (0.220, 0.405), (0.173, 0.424), (0.181, 0.371),
    (0.206, 0.328), (0.237, 0.273), (0.102, 0.313), (0.037, 0.341), (-0.038, 0.368),
    (-0.100, 0.389), (-0.073, 0.305), (0.015, 0.222), (-0.034, 0.219), (0.013, 0.157),
    (0.064, 0.125), (0.013, 0.120), (0.088, 0.052), (0.150, 0.020), (0.089, 0.008),
    (0.140, -0.029), (0.192, -0.049), (0.145, -0.077), (0.104, -0.110), (0.078, -0.143),
    (0.050, -0.190), (0.028, -0.226), (0.003, -0.252), (0.076, -0.271), (0.158, -0.277),
    (0.233, -0.286), (0.369, -0.287), (0.484, -0.262), (0.563, -0.226), (0.645, -0.174),
    (0.692, -0.125), (0.731, -0.050), (0.765, 0.033), (0.831, 0.078), (0.899, 0.084),
    (0.963, 0.088),
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
    ap.add_argument("--order", type=int, default=1, choices=[1, 2, 3], help="Element order (2, 3: curved)")
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
    if args.order > 1:
        gmsh.model.mesh.setOrder(args.order)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
        gmsh.model.mesh.optimize("HighOrder")
    gmsh.write(args.output)
    _, tags, _ = gmsh.model.mesh.getElements(2)
    print(f"wrote {args.output}: {sum(len(t) for t in tags)} triangles")
    gmsh.finalize()


if __name__ == "__main__":
    main()
