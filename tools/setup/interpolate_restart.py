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
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mallard import restart  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old_mesh")
    ap.add_argument("old_restart")
    ap.add_argument("new_mesh")
    ap.add_argument("new_restart")
    ap.add_argument("--neighbors", type=int, default=8)
    args = ap.parse_args()

    old = restart.read(args.old_restart)
    old_c = cell_centroids(args.old_mesh)
    if len(old_c) != old.n_cells:
        raise SystemExit(f"{args.old_restart} has {old.n_cells} cells, {args.old_mesh} {len(old_c)}")
    new_c = cell_centroids(args.new_mesh)
    dist, near = cKDTree(old_c).query(new_c, k=args.neighbors)
    w = 1.0 / np.maximum(dist, 1e-12) ** 2
    w /= w.sum(1, keepdims=True)
    values = np.einsum("vnk,nk->vn", old.values.astype(float)[:, near], w)
    restart.write(args.new_restart, dict(zip(old.names, values)), old.step, old.t, old.attributes,
                  real_size=old.real_size)
    print(f"{args.new_restart}: {len(new_c)} cells from {len(old_c)}, t = {old.t}, step {old.step}")


if __name__ == "__main__":
    main()
