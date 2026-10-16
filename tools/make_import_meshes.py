#!/usr/bin/env python3
"""Write the mesh-import test fixtures: one geometry in every format Mallard reads.

    make_import_meshes.py OUTDIR [--engine podman|docker] [--image opencfd/openfoam-default:2406] [--no-foam]

The geometry is a channel [0, 2] x [0, 1] with a cylinder of radius 0.2 at
(0.5, 0.5): triangles around the cylinder, transfinite quadrilaterals in
[1, 2] x [0, 1]. Boundary zones "inlet" (x = 0), "outlet" (x = 2), "walls"
(y = 0 and y = 1) and "cylinder". Gmsh (the gmsh Python module) writes:

  cyl2d.msh, cyl2d.cgns          the 2D mesh (Gmsh 4.1, CGNS)
  cyl2d_o2.msh, cyl2d_o2.cgns    the same at order 2 (curved cylinder)
  cyl2d_zones.cgns               the 2D mesh as two CGNS zones joined 1-to-1
  cyl3d.msh, cyl3d.cgns          extruded 0.3 in z in two layers (prisms and
                                 hexahedra), with "back" (z = 0) and "front"
  cyl3d_zones.cgns               the 3D mesh as two CGNS zones
  cyl3d_tet.msh, cyl3d_tet.cgns  the 3D geometry in tetrahedra

OpenFOAM (in a container, unless --no-foam) converts Gmsh 2.2 copies with
gmshToFoam and writes Fluent meshes with foamMeshToFluent:

  cyl3d_foam/constant/polyMesh   the 3D mesh
  cyl3d_fluent.msh               the 3D mesh, Fluent ASCII
  cyl2d_foam/constant/polyMesh   the 2D mesh extruded one cell thick, with
                                 the empty patch "frontAndBack"
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import gmsh

E = 1e-6


def build(dim, n_layers=2, tets=False, slab=False):
    gmsh.model.add("cyl")
    occ = gmsh.model.occ
    square = occ.addRectangle(0, 0, 0, 1, 1)
    disk = occ.addDisk(0.5, 0.5, 0, 0.2, 0.2)
    occ.cut([(2, square)], [(2, disk)])
    occ.addRectangle(1, 0, 0, 1, 1)
    occ.fragment(occ.getEntities(2), [])
    occ.synchronize()

    def box(d, x0, y0, x1, y1, z0=-E, z1=E):
        return [t for _, t in gmsh.model.getEntitiesInBoundingBox(x0 - E, y0 - E, z0, x1 + E, y1 + E, z1, d)]

    quad = box(2, 1, 0, 2, 1)
    for c in box(1, 1, 0, 2, 1):
        gmsh.model.mesh.setTransfiniteCurve(c, 7)
    gmsh.model.mesh.setTransfiniteSurface(quad[0])
    gmsh.model.mesh.setRecombine(2, quad[0])
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0 / 6)
    if dim == 2:
        zones = {"inlet": box(1, 0, 0, 0, 1), "outlet": box(1, 2, 0, 2, 1),
                 "walls": box(1, 0, 0, 2, 0) + box(1, 0, 1, 2, 1), "cylinder": box(1, 0.3, 0.3, 0.7, 0.7)}
        for name, tags in zones.items():
            gmsh.model.addPhysicalGroup(1, tags, name=name)
        gmsh.model.addPhysicalGroup(2, [t for _, t in gmsh.model.getEntities(2)], name="fluid")
        gmsh.model.mesh.generate(2)
        return
    lz = 0.3
    if tets:
        gmsh.model.mesh.setRecombine(2, quad[0], 0)
        gmsh.model.mesh.setTransfiniteSurface(quad[0], "Left")
        occ.extrude(occ.getEntities(2), 0, 0, lz)
        occ.synchronize()
    else:
        surfaces = [(2, t) for _, t in gmsh.model.getEntities(2)]
        occ.extrude(surfaces, 0, 0, lz, numElements=[n_layers], recombine=True)
        occ.synchronize()
    Z = (-E, lz + E)
    zones = {"inlet": box(2, 0, 0, 0, 1, *Z), "outlet": box(2, 2, 0, 2, 1, *Z),
             "walls": box(2, 0, 0, 2, 0, *Z) + box(2, 0, 1, 2, 1, *Z),
             "cylinder": box(2, 0.3, 0.3, 0.7, 0.7, *Z),
             "back": box(2, 0, 0, 2, 1, -E, E), "front": box(2, 0, 0, 2, 1, lz - E, lz + E)}
    if slab:
        zones["frontAndBack"] = zones.pop("back") + zones.pop("front")
    for name, tags in zones.items():
        gmsh.model.addPhysicalGroup(2, tags, name=name)
    gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], name="fluid")
    gmsh.model.mesh.generate(3)


def write(path, version=4.1):
    gmsh.option.setNumber("Mesh.MshFileVersion", version)
    gmsh.write(path)


def foam(engine, image, outdir, msh, case, empty_patch=None):
    """gmshToFoam on a Gmsh 2.2 file in a fresh case; returns the case directory."""
    work = tempfile.mkdtemp(dir=outdir)
    os.makedirs(os.path.join(work, "system"))
    with open(os.path.join(work, "system", "controlDict"), "w") as f:
        f.write("FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
                "application none; startFrom latestTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;\n"
                "writeControl timeStep; writeInterval 1;\n")
    header = "FoamFile { version 2.0; format ascii; class dictionary; object %s; }\n"
    with open(os.path.join(work, "system", "fvSchemes"), "w") as f:
        f.write(header % "fvSchemes" + "ddtSchemes { default steadyState; } gradSchemes { default Gauss linear; }\n"
                "divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; }\n"
                "interpolationSchemes { default linear; } snGradSchemes { default corrected; }\n")
    with open(os.path.join(work, "system", "fvSolution"), "w") as f:
        f.write(header % "fvSolution")
    shutil.copy(msh, os.path.join(work, "mesh.msh"))
    cmds = ["gmshToFoam mesh.msh"]
    if empty_patch:
        cmds.append(f"foamDictionary constant/polyMesh/boundary -entry entry0/{empty_patch}/type -set empty")
    cmds.append("checkMesh")
    cmds.append("foamMeshToFluent")
    script = "cd /case; source /usr/lib/openfoam/openfoam2406/etc/bashrc; (" + " && ".join(cmds) + ") > log 2>&1"
    run = subprocess.run([engine, "run", "--rm", "-v", f"{os.path.abspath(work)}:/case", "-w", "/case", image,
                          "bash", "-c", script])
    if run.returncode != 0:
        with open(os.path.join(work, "log")) as f:
            raise RuntimeError("OpenFOAM conversion failed:\n" + f.read()[-3000:])
    out = os.path.join(outdir, case)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "constant"))
    shutil.copytree(os.path.join(work, "constant", "polyMesh"), os.path.join(out, "constant", "polyMesh"))
    fluent = os.path.join(work, "fluentInterface", "case.msh")
    return work, fluent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--engine", default="podman")
    ap.add_argument("--image", default="docker.io/opencfd/openfoam-default:2406")
    ap.add_argument("--no-foam", action="store_true")
    args = ap.parse_args()
    out = args.outdir
    os.makedirs(out, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    p = lambda name: os.path.join(out, name)

    build(2)
    write(p("cyl2d.msh"))
    gmsh.write(p("cyl2d.cgns"))
    gmsh.model.mesh.partition(2)
    gmsh.write(p("cyl2d_zones.cgns"))
    gmsh.model.mesh.unpartition()
    gmsh.model.mesh.setOrder(2)
    write(p("cyl2d_o2.msh"))
    gmsh.write(p("cyl2d_o2.cgns"))
    gmsh.model.remove()

    tmp = tempfile.mkdtemp(dir=out)
    build(3)
    write(p("cyl3d.msh"))
    gmsh.write(p("cyl3d.cgns"))
    write(os.path.join(tmp, "cyl3d_22.msh"), 2.2)
    gmsh.model.mesh.partition(2)
    gmsh.write(p("cyl3d_zones.cgns"))
    gmsh.model.remove()

    build(3, tets=True)
    write(p("cyl3d_tet.msh"))
    gmsh.write(p("cyl3d_tet.cgns"))
    gmsh.model.remove()

    # The 2D mesh one cell thick, its front and back one zone for OpenFOAM's empty patch
    build(3, n_layers=1, slab=True)
    write(os.path.join(tmp, "cyl2d_slab_22.msh"), 2.2)
    gmsh.finalize()

    if not args.no_foam:
        work, fluent = foam(args.engine, args.image, tmp, os.path.join(tmp, "cyl3d_22.msh"), "cyl3d_foam")
        shutil.move(os.path.join(tmp, "cyl3d_foam"), p("cyl3d_foam"))
        shutil.copy(fluent, p("cyl3d_fluent.msh"))
        foam(args.engine, args.image, tmp, os.path.join(tmp, "cyl2d_slab_22.msh"), "cyl2d_foam", "frontAndBack")
        shutil.move(os.path.join(tmp, "cyl2d_foam"), p("cyl2d_foam"))
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
