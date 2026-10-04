#!/usr/bin/env python3
"""Interpolate a Mallard restart file onto another Gmsh mesh of the same domain.

    interpolate_restart.py OLD.msh OLD.restart NEW.msh NEW.restart [--neighbors 8]

Starts a run on a refined mesh from a developed flow computed on a coarser
one, skipping the start-up transient. Each new cell takes the inverse-distance
weighted average of the conservative variables of the nearest old cells
(centroid to centroid), so the result is only as smooth as the old mesh.
Cells are numbered as Mallard numbers them, in the order of the mesh file's
volume elements. Reads ASCII Gmsh 2.2 meshes of linear elements; the restart
keeps the old file's step and time.
"""
import argparse
import struct

import numpy as np
from scipy.spatial import cKDTree

NODES_PER_TYPE = {2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5}  # Gmsh element type -> nodes
VOLUME_TYPES = {4, 5, 6, 7}


def cell_centroids(path):
    """Vertex averages of the volume elements of an ASCII Gmsh 2.2 file, in file order."""
    with open(path) as f:
        line = f.readline()
        while line and not line.startswith("$MeshFormat"):
            line = f.readline()
        version = f.readline().split()[0]
        if not version.startswith("2.2"):
            raise SystemExit(f"{path}: only Gmsh 2.2 files are supported")
        while not line.startswith("$Nodes"):
            line = f.readline()
        n_nodes = int(f.readline())
        nodes = np.loadtxt(f, max_rows=n_nodes)
        index = np.zeros(int(nodes[:, 0].max()) + 1, dtype=np.int64)
        index[nodes[:, 0].astype(np.int64)] = np.arange(n_nodes)
        xyz = nodes[:, 1:4]
        while not line.startswith("$Elements"):
            line = f.readline()
        n_elements = int(f.readline())
        centroids = []
        for _ in range(n_elements):
            e = f.readline().split()
            etype, n_tags = int(e[1]), int(e[2])
            if etype not in VOLUME_TYPES:
                continue
            v = index[np.array(e[3 + n_tags:3 + n_tags + NODES_PER_TYPE[etype]], dtype=np.int64)]
            centroids.append(xyz[v].mean(0))
    return np.array(centroids)


def read_restart(path):
    with open(path, "rb") as f:
        magic = f.read(16)
        version, real_size = struct.unpack("<II", f.read(8))
        n_cells, n_vars, step = struct.unpack("<QQQ", f.read(24))
        (t,) = struct.unpack("<d", f.read(8))
        if not magic.startswith(b"MALLARD-RESTART") or version not in (2, 3) or real_size != 8:
            raise SystemExit(f"{path}: expected a double-precision restart file of version 2 or 3")

        def name():
            (length,) = struct.unpack("<I", f.read(4))
            return f.read(length).decode()

        names = [name() for _ in range(n_vars)]
        attributes = []
        if version == 3:
            (n_attributes,) = struct.unpack("<Q", f.read(8))
            attributes = [(name(), struct.unpack("<d", f.read(8))[0]) for _ in range(n_attributes)]
        fields = np.fromfile(f, dtype="<f8", count=n_vars * n_cells).reshape(n_vars, n_cells)
    return {"version": version, "step": step, "t": t, "names": names, "attributes": attributes, "fields": fields}


def write_restart(path, restart, fields):
    n_vars, n_cells = fields.shape
    with open(path, "wb") as f:
        f.write(b"MALLARD-RESTART\0")
        f.write(struct.pack("<II", restart["version"], 8))
        f.write(struct.pack("<QQQ", n_cells, n_vars, restart["step"]))
        f.write(struct.pack("<d", restart["t"]))
        for name in restart["names"]:
            f.write(struct.pack("<I", len(name)) + name.encode())
        if restart["version"] == 3:
            f.write(struct.pack("<Q", len(restart["attributes"])))
            for name, value in restart["attributes"]:
                f.write(struct.pack("<I", len(name)) + name.encode() + struct.pack("<d", value))
        fields.astype("<f8").tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old_mesh")
    ap.add_argument("old_restart")
    ap.add_argument("new_mesh")
    ap.add_argument("new_restart")
    ap.add_argument("--neighbors", type=int, default=8)
    args = ap.parse_args()

    old = read_restart(args.old_restart)
    old_c = cell_centroids(args.old_mesh)
    if len(old_c) != old["fields"].shape[1]:
        raise SystemExit(f"{args.old_restart} has {old['fields'].shape[1]} cells, {args.old_mesh} {len(old_c)}")
    new_c = cell_centroids(args.new_mesh)
    dist, near = cKDTree(old_c).query(new_c, k=args.neighbors)
    w = 1.0 / np.maximum(dist, 1e-12) ** 2
    w /= w.sum(1, keepdims=True)
    fields = np.einsum("vnk,nk->vn", old["fields"][:, near], w)
    write_restart(args.new_restart, old, fields)
    print(f"{args.new_restart}: {len(new_c)} cells from {len(old_c)}, t = {old['t']}, step {old['step']}")


if __name__ == "__main__":
    main()
