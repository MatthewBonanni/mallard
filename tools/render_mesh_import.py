#!/usr/bin/env python3
"""Draw the mesh-import figures of docs/design/mesh_import.md from what Mallard reads.

    render_mesh_import.py MESH_CONVERT [--meshes test/data/meshes] [--out docs/images]

MESH_CONVERT is a 2D build's mallard-mesh-convert (needs -DMallard_ENABLE_HDF5=ON,
and -DMallard_ENABLE_CGNS=ON for the CGNS panel). Each fixture of
tools/make_import_meshes.py (Gmsh, CGNS, Fluent, OpenFOAM) is converted to a
Mallard HDF5 mesh, read back with h5py and drawn with its boundary zones
coloured and labelled:

  mesh_import_formats.png    the four imports side by side
  mesh_import_transform.png  the Gmsh mesh as read, and with [mesh.transform]
                             scale = [1.5, 1], rotate = 30, translate = [0.5, 0.25]
                             (+ marks the origin, about which it rotates)

The figures have transparent backgrounds and mid-tone colours, so they read on
light and dark pages alike.
"""
import argparse
import os
import subprocess
import tempfile

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

GREY = "#8a8a8a"
ZONE_COLOURS = {"inlet": "#3b82f6", "outlet": "#ef4444", "walls": "#22a35a", "cylinder": "#f59e0b"}
TRANSFORM = "[mesh.transform]\nscale = [1.5, 1]\nrotate = 30\ntranslate = [0.5, 0.25]\n"


def convert(tool, source, workdir, transform=""):
    """Mallard's read of a mesh file (through mallard-mesh-convert), as arrays."""
    name = os.path.basename(source.rstrip("/"))
    out = os.path.join(workdir, name + (".t" if transform else "") + ".h5")
    toml = os.path.join(workdir, name + ".toml")
    with open(toml, "w") as f:
        f.write('[mesh]\ntype = "file"\nfilename = "%s"\n%s' % (os.path.abspath(source), transform))
    subprocess.run([tool, toml, out], check=True, stdout=subprocess.DEVNULL)
    with h5py.File(out, "r") as h:
        mesh = {
            "x": h["nodes/coordinates"][:],
            "cell_offsets": h["cells/offsets"][:],
            "cell_nodes": h["cells/nodes"][:],
            "face_offsets": h["boundary/offsets"][:],
            "face_nodes": h["boundary/nodes"][:],
            "face_zone": h["boundary/zone"][:],
            "zones": [z.decode() if isinstance(z, bytes) else str(z) for z in h["boundary"].attrs["zone_names"]],
        }
    return mesh


def draw(ax, mesh, title, labels=True):
    x = mesh["x"]
    off, nodes = mesh["cell_offsets"], mesh["cell_nodes"]
    polys = [x[nodes[off[c]:off[c + 1]]] for c in range(len(off) - 1)]
    ax.add_collection(PolyCollection(polys, facecolors=(0.55, 0.55, 0.55, 0.12), edgecolors=GREY, linewidths=0.5))
    foff, fnodes = mesh["face_offsets"], mesh["face_nodes"]
    for z, name in enumerate(mesh["zones"]):
        segs = [x[fnodes[foff[f]:foff[f + 1]]] for f in range(len(foff) - 1) if mesh["face_zone"][f] == z]
        if not segs:
            continue
        colour = ZONE_COLOURS.get(name, "#a855f7")
        ax.add_collection(LineCollection(segs, colors=colour, linewidths=3.0, capstyle="round"))
        if labels:
            mid = np.concatenate(segs).mean(axis=0)
            if name == "walls":  # the top wall
                top = [s for s in segs if s[:, 1].mean() > np.median(x[:, 1])]
                mid = np.concatenate(top).mean(axis=0) if top else mid
            ax.annotate(name, mid, ha="center", va="center", fontsize=9, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.25", fc=colour, ec="none"))
    ax.set_title(title, color=GREY, fontsize=11)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.04)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("convert")
    ap.add_argument("--meshes", default="test/data/meshes")
    ap.add_argument("--out", default="docs/images")
    args = ap.parse_args()
    work = tempfile.mkdtemp()
    m = lambda name: os.path.join(args.meshes, name)
    formats = [("Gmsh (cyl2d.msh)", "cyl2d.msh"), ("CGNS (cyl2d.cgns)", "cyl2d.cgns"),
               ("Fluent (cyl2d_fluent.msh)", "cyl2d_fluent.msh"), ("OpenFOAM (cyl2d_foam/)", "cyl2d_foam")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6))
    for ax, (title, name) in zip(axes.flat, formats):
        mesh = convert(args.convert, m(name), work)
        n_cells = len(mesh["cell_offsets"]) - 1
        draw(ax, mesh, f"{title}: {n_cells} cells")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mesh_import_formats.png"), dpi=150, transparent=True)

    before = convert(args.convert, m("cyl2d.msh"), work)
    after = convert(args.convert, m("cyl2d.msh"), work, TRANSFORM)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    draw(axes[0], before, "As read")
    draw(axes[1], after, "scale = [1.5, 1], rotate = 30, translate = [0.5, 0.25]")
    for ax in axes:
        ax.axis("on")
        ax.grid(True, color=GREY, alpha=0.3, lw=0.5)
        ax.plot(0, 0, "+", color=GREY, ms=12, mew=1.5)
        ax.tick_params(colors=GREY, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GREY)
            spine.set_alpha(0.5)
    xy = np.concatenate([before["x"], after["x"]])
    axes[0].set_xlim(xy[:, 0].min() - 0.15, xy[:, 0].max() + 0.15)
    axes[0].set_ylim(xy[:, 1].min() - 0.15, xy[:, 1].max() + 0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mesh_import_transform.png"), dpi=150, transparent=True)


if __name__ == "__main__":
    main()
