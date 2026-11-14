#!/usr/bin/env python3
"""Gmsh tetrahedral mesh of a quarter domain around a prolate spheroid.

    make_ellipsoid_mesh.py OUTPUT.msh [--scale 1] [--h-wall 0.06] [--h-far 0.8]
        [--a 1.0] [--b 0.5] [--x-min -6] [--x-max 8] [--half-width 6]

The spheroid has semi-axes a along x and b along y and z (length 2 a,
diameter 2 b = 1 by default), centered at the origin, with the free stream
along +x. Only the quarter y >= 0, z >= 0 of the box [x-min, x-max] x
[-half-width, half-width]^2 is meshed (the flow is axisymmetric), so the
planes y = 0 and z = 0 are symmetry planes. Tetrahedra have edge length
h-wall on the spheroid and coarsen to h-far at 2.5 from it; --scale s divides
both by s for mesh studies. Physical surfaces: "ellipsoid", "inflow"
(x = x-min), "outflow" (x = x-max), "lateral" (y, z = half-width),
"symmetry_y" (y = 0) and "symmetry_z" (z = 0). The faces on the spheroid are
flat: Mallard curves them onto the analytic surface ([[mesh.curved]]).
Needs the gmsh Python module (pip install gmsh).
"""
import argparse

import gmsh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--scale", type=float, default=1.0, help="Refinement factor dividing every size")
    ap.add_argument("--h-wall", type=float, default=0.06)
    ap.add_argument("--h-far", type=float, default=0.8)
    ap.add_argument("--a", type=float, default=1.0, help="Semi-axis along the flow")
    ap.add_argument("--b", type=float, default=0.5, help="Semi-axis across the flow")
    ap.add_argument("--x-min", type=float, default=-6.0)
    ap.add_argument("--x-max", type=float, default=8.0)
    ap.add_argument("--half-width", type=float, default=6.0)
    ap.add_argument("--threads", type=int, default=1, help="Gmsh threads; more are faster but not reproducible")
    args = ap.parse_args()
    h_wall, h_far = args.h_wall / args.scale, args.h_far / args.scale

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.NumThreads", args.threads)
    gmsh.model.add("ellipsoid")
    occ = gmsh.model.occ
    L = args.x_max - args.x_min
    domain = [(3, occ.addBox(args.x_min, 0, 0, L, args.half_width, args.half_width))]
    body = occ.addSphere(0, 0, 0, 1.0)
    occ.dilate([(3, body)], 0, 0, 0, args.a, args.b, args.b)
    fluid, _ = occ.cut(domain, [(3, body)])
    occ.synchronize()

    surfaces = {"ellipsoid": [], "inflow": [], "outflow": [], "lateral": [], "symmetry_y": [], "symmetry_z": []}
    eps = 1e-4
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
        elif max(abs(x0), abs(x1), y1, z1) < 0.5 * (args.a + min(args.x_max, -args.x_min, args.half_width)):
            surfaces["ellipsoid"].append(tag)
        else:
            surfaces["lateral"].append(tag)
    for name, tags in surfaces.items():
        assert tags, f"no surface for {name}"
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, tags), name)
    gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, [t for _, t in fluid]), "fluid")

    f = gmsh.model.mesh.field
    wall = f.add("Distance")
    f.setNumbers(wall, "SurfacesList", surfaces["ellipsoid"])
    f.setNumber(wall, "Sampling", 200)
    near = f.add("Threshold")
    f.setNumber(near, "InField", wall)
    f.setNumber(near, "SizeMin", h_wall)
    f.setNumber(near, "SizeMax", h_far)
    f.setNumber(near, "DistMin", 0.05)
    f.setNumber(near, "DistMax", 2.5)
    f.setAsBackgroundMesh(near)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(3)
    types, tags, _ = gmsh.model.mesh.getElements(3)
    print(f"{sum(len(t) for t in tags)} tetrahedra")
    gmsh.write(args.output)
    gmsh.finalize()


if __name__ == "__main__":
    main()
