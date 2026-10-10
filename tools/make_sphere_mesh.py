#!/usr/bin/env python3
"""Gmsh tetrahedral mesh of a quarter domain around a sphere, for supersonic flow.

    make_sphere_mesh.py OUTPUT.msh [--h-wall 0.02] [--h-shock 0.025] [--h-far 0.12]
        [--x-min -1.5] [--x-max 2.5] [--r-max 2.5] [--shock-x -0.62]

The sphere has diameter 1 and is centered at the origin, with the free stream
along +x. Only y >= 0, z >= 0 is meshed (the flow is axisymmetric), so the
planes y = 0 and z = 0 are symmetry planes. The mesh is finest on the sphere
and through the shock layer in front of it, a shell between the sphere and a
paraboloid through x = shock-x, and coarsens to h-far outside. Physical
surfaces: "sphere", "inflow" (x = x-min), "outflow" (x = x-max), "lateral"
(the outer quarter cylinder), "symmetry_y" (y = 0) and "symmetry_z" (z = 0).
With --order 2 or 3 the elements are quadratic or cubic, their extra nodes on
the CAD surfaces: Mallard keeps the corners and curves the sphere from its
high-order faces.
Needs the gmsh Python module (pip install gmsh).
"""
import argparse

import gmsh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--h-wall", type=float, default=0.02)
    ap.add_argument("--h-shock", type=float, default=0.025)
    ap.add_argument("--h-far", type=float, default=0.12)
    ap.add_argument("--x-min", type=float, default=-1.5)
    ap.add_argument("--x-max", type=float, default=2.5)
    ap.add_argument("--r-max", type=float, default=2.5)
    ap.add_argument("--shock-x", type=float, default=-0.62,
                    help="Expected shock nose position; the shock layer is refined around it")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--order", type=int, default=1, choices=[1, 2, 3],
                    help="Element order: 2 and 3 place the extra nodes on the sphere, so Mallard curves the wall")
    args = ap.parse_args()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.NumThreads", args.threads)
    gmsh.model.add("sphere")
    occ = gmsh.model.occ
    L = args.x_max - args.x_min
    cyl = occ.addCylinder(args.x_min, 0, 0, L, 0, 0, args.r_max)
    quarter = occ.addBox(args.x_min - 1, 0, 0, L + 2, args.r_max + 1, args.r_max + 1)
    domain, _ = occ.intersect([(3, cyl)], [(3, quarter)])
    sphere = occ.addSphere(0, 0, 0, 0.5)
    fluid, _ = occ.cut(domain, [(3, sphere)])
    occ.synchronize()

    surfaces = {"sphere": [], "inflow": [], "outflow": [], "lateral": [], "symmetry_y": [], "symmetry_z": []}
    eps = 1e-6
    for dim, tag in gmsh.model.getBoundary(fluid, oriented=False):
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, tag)
        if x1 - x0 < eps and abs(x0 - args.x_min) < eps:
            surfaces["inflow"].append(tag)
        elif x1 - x0 < eps and abs(x0 - args.x_max) < eps:
            surfaces["outflow"].append(tag)
        elif y1 - y0 < eps and abs(y0) < eps:
            surfaces["symmetry_y"].append(tag)
        elif z1 - z0 < eps and abs(z0) < eps:
            surfaces["symmetry_z"].append(tag)
        elif max(abs(x0), abs(x1), y1, z1) < 0.5 + 1e-3:
            surfaces["sphere"].append(tag)
        else:
            surfaces["lateral"].append(tag)
    for name, tags in surfaces.items():
        assert tags, f"no surface for {name}"
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, tags), name)
    gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, [t for _, t in fluid]), "fluid")

    # Shock layer: inside the paraboloid x > shock_x - 0.15 + k r^2 with a margin
    # in front of the expected shock, and within r < 1.6 of the axis
    x_s = args.shock_x - 0.12
    f = gmsh.model.mesh.field
    shell = f.add("MathEval")
    f.setString(shell, "F",
                f"{args.h_shock} + ({args.h_far} - {args.h_shock}) * "
                f"Min(1, Max(0, ({x_s} + 0.45 * (y*y + z*z) - x) / 0.35, (Sqrt(y*y + z*z) - 1.6) / 0.6,"
                f" (x - 1.2) / 0.8))")
    wall = f.add("Distance")
    f.setNumbers(wall, "SurfacesList", surfaces["sphere"])
    near = f.add("Threshold")
    f.setNumber(near, "InField", wall)
    f.setNumber(near, "SizeMin", args.h_wall)
    f.setNumber(near, "SizeMax", args.h_far)
    f.setNumber(near, "DistMin", 0.02)
    f.setNumber(near, "DistMax", 0.6)
    both = f.add("Min")
    f.setNumbers(both, "FieldsList", [shell, near])
    f.setAsBackgroundMesh(both)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(3)
    if args.order > 1:
        gmsh.model.mesh.setOrder(args.order)
    types, tags, _ = gmsh.model.mesh.getElements(3)
    print(f"{sum(len(t) for t in tags)} tetrahedra")
    gmsh.write(args.output)
    gmsh.finalize()


if __name__ == "__main__":
    main()
