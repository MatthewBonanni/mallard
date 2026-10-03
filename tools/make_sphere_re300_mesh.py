#!/usr/bin/env python3
"""Gmsh mesh around a sphere for viscous flow: prism boundary layer and tetrahedra.

    make_sphere_re300_mesh.py OUTPUT.msh [--scale 1] [--h-wall 0.035] [--dn0 0.005]
        [--n-layers 10] [--growth 1.2] [--h-near 0.05] [--h-wake 0.07] [--h-far 1.5]
        [--x-min -15] [--x-max 30] [--half-width 15]

The sphere has diameter 1 and is centered at the origin, with the free stream
along +x. The whole domain is meshed (no symmetry planes): the box
[x-min, x-max] x [-half-width, half-width]^2. Layers of prisms grow
geometrically from dn0 on the sphere; tetrahedra fill the rest, with edge
length h-near around the sphere, h-wake through the near wake (a cylinder of
radius 1.2 to x = 5), coarsening to 2.5 h-wake at x = 12 and to h-far at the
box: 0.85M cells. --scale s refines for mesh studies: it divides the sizes on
and around the sphere, in the wake and of the first layer by s, and makes s
times as many layers of the same total thickness (s = 1.4: 2.06M cells).
Physical surfaces: "sphere", "inflow" (x = x-min), "outflow" (x = x-max) and
"lateral" (the four sides). Needs the gmsh Python module.
"""
import argparse

import gmsh


def sphere_surfaces(geo, r, lc):
    """Eight octant patches of a sphere of radius r at the origin."""
    c = geo.addPoint(0, 0, 0, lc)
    p = {k: geo.addPoint(*v, lc) for k, v in {
        "+x": (r, 0, 0), "-x": (-r, 0, 0), "+y": (0, r, 0), "-y": (0, -r, 0), "+z": (0, 0, r), "-z": (0, 0, -r)}.items()}
    arcs = {}

    def arc(a, b):
        if (b, a) in arcs:
            return -arcs[(b, a)]
        if (a, b) not in arcs:
            arcs[(a, b)] = geo.addCircleArc(p[a], c, p[b])
        return arcs[(a, b)]

    surfaces = []
    for sx in "+-":
        for sy in "+-":
            for sz in "+-":
                a, b, d = sx + "x", sy + "y", sz + "z"
                # Counterclockwise seen from outside for positive octant orientation
                if (sx == "+") ^ (sy == "+") ^ (sz == "+"):
                    a, b = b, a
                loop = geo.addCurveLoop([arc(a, b), arc(b, d), arc(d, a)])
                surfaces.append(geo.addSurfaceFilling([loop]))
    return surfaces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--scale", type=float, default=1.0, help="Refinement factor applied to every size")
    ap.add_argument("--h-wall", type=float, default=0.035, help="Triangle size on the sphere")
    ap.add_argument("--dn0", type=float, default=0.005, help="First prism layer height")
    ap.add_argument("--n-layers", type=int, default=10)
    ap.add_argument("--growth", type=float, default=1.2)
    ap.add_argument("--h-near", type=float, default=0.05, help="Tetrahedron size around the sphere")
    ap.add_argument("--h-wake", type=float, default=0.07, help="Tetrahedron size in the near wake")
    ap.add_argument("--h-far", type=float, default=1.5)
    ap.add_argument("--x-min", type=float, default=-15.0)
    ap.add_argument("--x-max", type=float, default=30.0)
    ap.add_argument("--half-width", type=float, default=15.0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    s = args.scale
    h_wall, h_near, h_wake = args.h_wall / s, args.h_near / s, args.h_wake / s
    dn0 = args.dn0 / s
    growth = args.growth ** (1.0 / s)
    n_layers = round(args.n_layers * s)
    heights, total = [], 0.0
    for k in range(n_layers):
        total += dn0 * growth ** k
        heights.append(total)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.NumThreads", args.threads)
    gmsh.model.add("sphere_re300")
    geo = gmsh.model.geo

    sphere = sphere_surfaces(geo, 0.5, h_wall)
    gmsh.option.setNumber("Geometry.ExtrudeReturnLateralEntities", 0)
    layers = geo.extrudeBoundaryLayer([(2, t) for t in sphere], [1] * n_layers, [-h for h in heights], True)
    top = [layers[i - 1][1] for i in range(len(layers)) if layers[i][0] == 3]
    bl_volumes = [t for d, t in layers if d == 3]

    x0, x1, w = args.x_min, args.x_max, args.half_width
    corners = [geo.addPoint(x, y, z, args.h_far) for x in (x0, x1) for y in (-w, w) for z in (-w, w)]

    def quad(a, b, c, d):
        lines = []
        for i, j in ((a, b), (b, c), (c, d), (d, a)):
            lines.append(edge(i, j))
        return geo.addPlaneSurface([geo.addCurveLoop(lines)])

    edges = {}

    def edge(i, j):
        if (j, i) in edges:
            return -edges[(j, i)]
        if (i, j) not in edges:
            edges[(i, j)] = geo.addLine(corners[i], corners[j])
        return edges[(i, j)]

    # corners index = 4 ix + 2 iy + iz
    inflow = quad(0, 1, 3, 2)
    outflow = quad(4, 6, 7, 5)
    lateral = [quad(0, 4, 5, 1), quad(2, 3, 7, 6), quad(0, 2, 6, 4), quad(1, 5, 7, 3)]
    outer = geo.addSurfaceLoop([inflow, outflow] + lateral)
    inner = geo.addSurfaceLoop(top)
    fluid = geo.addVolume([outer, inner])
    geo.synchronize()

    for name, tags in {"sphere": sphere, "inflow": [inflow], "outflow": [outflow], "lateral": lateral}.items():
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, tags), name)
    gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, bl_volumes + [fluid]), "fluid")

    f = gmsh.model.mesh.field
    r_bl = 0.5 + heights[-1]
    near = f.add("MathEval")
    f.setString(near, "F",
                f"{h_near} + Max(0, Sqrt(x*x + y*y + z*z) - {r_bl}) * 0.25")
    # Near wake: a cylinder of radius 1.2 to x = 5, then linear growth to 2.5 h_wake at x = 12
    # and h_far beyond; radial growth outside the cylinder
    wake = f.add("MathEval")
    f.setString(wake, "F",
                f"{h_wake} * (1 + 1.5 * Min(1, Max(0, (x - 5) / 7)))"
                f" + 0.2 * Max(0, Sqrt(y*y + z*z) - 1.2 - 0.08 * Max(0, x))"
                f" + 0.2 * Max(0, -0.5 - x) + 0.3 * Max(0, x - 12)")
    both = f.add("Min")
    f.setNumbers(both, "FieldsList", [near, wake])
    capped = f.add("MathEval")
    f.setString(capped, "F", f"{args.h_far}")
    final = f.add("Min")
    f.setNumbers(final, "FieldsList", [both, capped])
    f.setAsBackgroundMesh(final)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(3)
    counts = {}
    for t, tags in zip(*gmsh.model.mesh.getElements(3)[:2]):
        counts[gmsh.model.mesh.getElementProperties(t)[0]] = len(tags)
    print(", ".join(f"{n} {name}" for name, n in counts.items()),
          f"({sum(counts.values())} cells); prism layers {n_layers}, first {dn0:.4g}, total {heights[-1]:.4g}")
    gmsh.write(args.output)
    gmsh.finalize()


if __name__ == "__main__":
    main()
