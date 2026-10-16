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

This script writes the 2D mesh as Fluent (no free tool does; written from
Gmsh's nodes and elements following the Fluent file format):

  cyl2d_fluent.msh, cyl2d_fluent_bin.msh   ASCII and binary

OpenFOAM (in a container, unless --no-foam) converts Gmsh 2.2 copies with
gmshToFoam and writes Fluent meshes with foamMeshToFluent:

  cyl3d_foam/constant/polyMesh   the 3D mesh (and cyl3d_foam_bin/, converted to
                                 binary by foamFormatConvert)
  cyl3d_fluent.msh               the 3D mesh, Fluent ASCII
  cyl3d_fluent_bin.msh           the same re-encoded as binary sections
  cyl2d_foam/constant/polyMesh   the 2D mesh extruded one cell thick (by this
                                 script, as a Gmsh 2.2 file), with
                                 the empty patch "frontAndBack" (and _bin/)
"""
import argparse
import os
import re
import struct
import shutil
import subprocess
import tempfile

import gmsh

E = 1e-6


def build(dim, n_layers=2, tets=False):
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
    for name, tags in zones.items():
        gmsh.model.addPhysicalGroup(2, tags, name=name)
    gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], name="fluid")
    gmsh.model.mesh.generate(3)


def fluent_2d(path):
    """The current 2D Gmsh mesh as an ASCII Fluent mesh: nodes, cells, and faces (edges) by zone."""
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    index = {int(t): i + 1 for i, t in enumerate(tags)}
    cells = []
    for etype, _, enodes in zip(*gmsh.model.mesh.getElements(2)):
        n = gmsh.model.mesh.getElementProperties(etype)[3]
        cells += [[index[int(t)] for t in enodes[k:k + n]] for k in range(0, len(enodes), n)]
    edge_cells = {}
    for c, nodes in enumerate(cells):
        for k in range(len(nodes)):
            a, b = nodes[k], nodes[(k + 1) % len(nodes)]
            edge_cells.setdefault(tuple(sorted((a, b))), []).append((c + 1, (a, b)))
    edge_zone = {}
    names = {}
    for dim, tag in gmsh.model.getPhysicalGroups(1):
        names[tag] = gmsh.model.getPhysicalName(dim, tag)
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            for etype, _, enodes in zip(*gmsh.model.mesh.getElements(1, entity)):
                for k in range(0, len(enodes), 2):
                    edge_zone[tuple(sorted((index[int(enodes[k])], index[int(enodes[k + 1])])))] = tag
    interior = [e for e, c in edge_cells.items() if len(c) == 2]
    zones = [(3, "interior", "interior-fluid", 2, interior)]
    for z, tag in enumerate(sorted(names)):
        zones.append((4 + z, "wall", names[tag], 3, [e for e, c in edge_cells.items() if len(c) == 1 and edge_zone[e] == tag]))
    n_faces = sum(len(z[4]) for z in zones)
    with open(path, "w") as f:
        f.write('(0 "Mallard test fixture: Fluent 2D mesh written from Gmsh")\n(2 2)\n')
        f.write(f"(10 (0 1 {len(tags):x} 0 2))\n(10 (1 1 {len(tags):x} 1 2)(\n")
        for i in range(len(tags)):
            f.write(f"{xyz[3 * i]:.17g} {xyz[3 * i + 1]:.17g}\n")
        f.write("))\n")
        f.write(f"(12 (0 1 {len(cells):x} 0 0))\n(12 (2 1 {len(cells):x} 1 0)(\n")
        f.write(" ".join("1" if len(c) == 3 else "3" for c in cells) + "\n))\n")
        f.write(f"(13 (0 1 {n_faces:x} 0 0))\n")
        first = 1
        for zid, _, _, bc, edges in zones:
            f.write(f"(13 ({zid:x} {first:x} {first + len(edges) - 1:x} {bc:x} 2)(\n")
            for e in edges:
                (c0, (a, b)), *rest = edge_cells[e]
                c1 = rest[0][0] if rest else 0
                f.write(f"{a:x} {b:x} {c0:x} {c1:x}\n")
            f.write("))\n")
            first += len(edges)
        f.write("(45 (2 fluid fluid)())\n")
        for zid, ztype, name, _, _ in zones:
            f.write(f"(45 ({zid} {ztype} {name})())\n")


def slab_22(path, lz=0.3):
    """The current 2D Gmsh mesh extruded one cell thick in z, as a Gmsh 2.2 file for gmshToFoam: prisms
    and hexahedra, its boundary zones as quadrilaterals, and both ends in the zone "frontAndBack"."""
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    index = {int(t): i + 1 for i, t in enumerate(tags)}
    n = len(tags)
    cells = []
    for etype, _, enodes in zip(*gmsh.model.mesh.getElements(2)):
        k = gmsh.model.mesh.getElementProperties(etype)[3]
        cells += [[index[int(t)] for t in enodes[i:i + k]] for i in range(0, len(enodes), k)]
    groups = {}
    for dim, tag in gmsh.model.getPhysicalGroups(1):
        name = gmsh.model.getPhysicalName(dim, tag)
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            for _, _, enodes in zip(*gmsh.model.mesh.getElements(1, entity)):
                groups.setdefault(name, []).extend(
                    [index[int(enodes[i])], index[int(enodes[i + 1])]] for i in range(0, len(enodes), 2))
    names = list(groups) + ["frontAndBack", "fluid"]
    elements = []
    for z, name in enumerate(groups):
        for a, b in groups[name]:
            elements.append((3, z + 1, [a, b, b + n, a + n]))
    for c in cells:
        elements.append((2 if len(c) == 3 else 3, len(groups) + 1, c))
        elements.append((2 if len(c) == 3 else 3, len(groups) + 1, [v + n for v in c]))
    for c in cells:
        elements.append((6 if len(c) == 3 else 5, len(groups) + 2, c + [v + n for v in c]))
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n" % len(names))
        for i, name in enumerate(names):
            f.write('%d %d "%s"\n' % (3 if name == "fluid" else 2, i + 1, name))
        f.write("$EndPhysicalNames\n$Nodes\n%d\n" % (2 * n))
        for layer in (0.0, lz):
            for i in range(n):
                f.write("%d %.17g %.17g %.17g\n" % (i + 1 + (n if layer else 0), xyz[3 * i], xyz[3 * i + 1], layer))
        f.write("$EndNodes\n$Elements\n%d\n" % len(elements))
        for i, (etype, phys, nodes) in enumerate(elements):
            f.write("%d %d 2 %d %d %s\n" % (i + 1, etype, phys, phys, " ".join(map(str, nodes))))
        f.write("$EndElements\n")


def fluent_binary(src, dst):
    """Re-encode the node (10), cell (12) and face (13) sections of an ASCII Fluent file as binary (3010, 2012, 2013)."""
    text = open(src).read()
    out = []
    pos = 0
    for m in re.finditer(r"\((1[023]) \(([0-9a-fA-F ]+)\)\s*\(", text):
        start = m.start()
        if start < pos:
            continue
        index, header = m.group(1), m.group(2)
        body_end = text.index(")", m.end())
        values = text[m.end():body_end].split()
        close = text.index(")", body_end + 1)
        out.append(text[pos:start])
        h = header.split()
        if index == "10":
            data = struct.pack(f"<{len(values)}d", *map(float, values))
            out.append(f"(3010 ({header})(".encode().decode("latin-1") + data.decode("latin-1") + ")End of Binary Section 3010)")
        else:
            ints = [int(v, 16) for v in values]
            code = "2012" if index == "12" else "2013"
            data = struct.pack(f"<{len(ints)}i", *ints)
            out.append(f"({code} ({header})(" + data.decode("latin-1") + f")End of Binary Section {code})")
        pos = close + 1
    out.append(text[pos:])
    with open(dst, "wb") as f:
        f.write("".join(out).encode("latin-1"))


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
                "writeControl timeStep; writeInterval 1; writePrecision 17;\n")
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
    cmds.append("cp -r constant/polyMesh ascii_polyMesh")
    cmds.append("foamDictionary system/controlDict -entry writeFormat -add binary")
    cmds.append("foamFormatConvert -constant")
    script = "cd /case; source /usr/lib/openfoam/openfoam2406/etc/bashrc; (" + " && ".join(cmds) + ") > log 2>&1"
    run = subprocess.run([engine, "run", "--rm", "-v", f"{os.path.abspath(work)}:/case", "-w", "/case", image,
                          "bash", "-c", script])
    if run.returncode != 0:
        with open(os.path.join(work, "log")) as f:
            raise RuntimeError("OpenFOAM conversion failed:\n" + f.read()[-3000:])
    for name, src in ((case, "ascii_polyMesh"), (case + "_bin", os.path.join("constant", "polyMesh"))):
        out = os.path.join(outdir, name)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(os.path.join(out, "constant"))
        shutil.copytree(os.path.join(work, src), os.path.join(out, "constant", "polyMesh"),
                        ignore=shutil.ignore_patterns("sets"))
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
    fluent_2d(p("cyl2d_fluent.msh"))
    tmp = tempfile.mkdtemp(dir=out)
    slab_22(os.path.join(tmp, "cyl2d_slab_22.msh"))
    fluent_binary(p("cyl2d_fluent.msh"), p("cyl2d_fluent_bin.msh"))
    gmsh.model.mesh.partition(2)
    gmsh.write(p("cyl2d_zones.cgns"))
    gmsh.model.mesh.unpartition()
    gmsh.model.mesh.setOrder(2)
    write(p("cyl2d_o2.msh"))
    gmsh.write(p("cyl2d_o2.cgns"))
    gmsh.model.remove()

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

    gmsh.finalize()

    if not args.no_foam:
        work, fluent = foam(args.engine, args.image, tmp, os.path.join(tmp, "cyl3d_22.msh"), "cyl3d_foam")
        for suffix in ("", "_bin"):
            shutil.move(os.path.join(tmp, "cyl3d_foam" + suffix), p("cyl3d_foam" + suffix))
        shutil.copy(fluent, p("cyl3d_fluent.msh"))
        fluent_binary(p("cyl3d_fluent.msh"), p("cyl3d_fluent_bin.msh"))
        foam(args.engine, args.image, tmp, os.path.join(tmp, "cyl2d_slab_22.msh"), "cyl2d_foam", "frontAndBack")
        for suffix in ("", "_bin"):
            shutil.move(os.path.join(tmp, "cyl2d_foam" + suffix), p("cyl2d_foam" + suffix))
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
