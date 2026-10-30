"""Figures of the probes section of the user guide, from short Mallard runs
(seconds each on a CPU build): probe shapes over a 2D mesh, boundary probes on
curved walls (polylines with arc length), and a 3D wall's heat flux.

    python tools/docs/probes.py --mallard build/src/mallard [--mallard3d build-3d/src/mallard] [names...]
"""
import argparse
import csv
import functools
import math
import re
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import FancyArrowPatch, Polygon
import matplotlib.pyplot as plt

import figstyle
from figstyle import bare, label
from runs import PHYSICS, polygons, run


def read_csv(path, step=None):
    """Rows of a probe CSV as a dict of arrays (only the given step, if any)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if step is not None:
        rows = [r for r in rows if int(float(r["step"])) == step]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def read_vtp(path):
    """Points, polygons and cell arrays of a probe's VTK PolyData file (raw appended, UInt64 headers)."""
    raw = Path(path).read_bytes()
    head, data = raw.split(b"<AppendedData encoding=\"raw\">", 1)
    data = data[data.index(b"_") + 1:]
    text = head.decode()
    arrays = {}
    for m in re.finditer(r'<DataArray type="(\w+)" Name="(\w+)" NumberOfComponents="(\d+)" format="appended" '
                         r'offset="(\d+)"/>', text):
        kind, name, comps, offset = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        (size,) = struct.unpack_from("<Q", data, offset)
        dtype = np.float64 if kind == "Float64" else np.int64
        arrays[name] = np.frombuffer(data, dtype=dtype, count=size // 8, offset=offset + 8).reshape(-1, comps)
    return arrays


# --------------------------------------------------------------- probe shapes

FIELD = "1.0 + 0.4 * exp(-12 * ((x - 0.45)^2 + (y - 0.62)^2)) - 0.2 * exp(-20 * ((x - 1.1)^2 + (y - 0.35)^2))"


@functools.cache
def shapes_data(mallard):
    work = Path(tempfile.mkdtemp())
    probes = {
        "points": 'shape = "points"\npoints = [[0.2, 0.2], [0.3, 0.8], [1.25, 0.25], [1.3, 0.75]]',
        "line": "start = [0.1, 0.5]\nend = [1.4, 0.5]\nn_points = 40",
        "arc": 'shape = "arc"\ncenter = [0.55, 0.5]\nradius = 0.33\nangles = [20.0, 160.0]\nn_points = 40',
        "curve": 'shape = "curve"\nx = "0.1 + 1.3 * s"\ny = "0.16 + 0.05 * sin(6 * pi * s)"\nn_points = 80',
        "polyline": 'shape = "polyline"\nvertices = [[1.05, 0.62], [1.4, 0.92], [1.05, 0.92]]\nn_points = 12',
    }
    toml = f"""
[run]
n_steps = 1
dt = 1.0e-9
[mesh]
type = "cartesian_tri"
Nx = 24
Ny = 16
Lx = 1.5
Ly = 1.0
[initialize]
type = "analytical"
rho = "1.0"
u = ["0.0", "0.0"]
p = "{FIELD}"
[[boundaries]]
name = "left"
type = "symmetry"
[[boundaries]]
name = "right"
type = "symmetry"
[[boundaries]]
name = "top"
type = "symmetry"
[[boundaries]]
name = "bottom"
type = "symmetry"
[numerics.face_reconstruction]
type = "TENO"
order = 4
troubled_threshold = 1.0
[[write_data]]
prefix = "mesh"
format = "vtu"
interval = 1
variables = ["P"]
{PHYSICS}"""
    for name, geometry in probes.items():
        toml += f'\n[[probes]]\nname = "{name}"\nvariables = ["P"]\nsampling = "point"\n{geometry}\n'
    run(mallard, toml, work)
    polys, cells = polygons(work / "mesh_000000.vtu")
    return polys, cells["P"], {name: read_csv(work / f"probe_{name}.csv", step=0) for name in probes}


def probe_shapes(P, args):
    polys, field, data = shapes_data(args.mallard)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    lo, hi = 0.8, 1.4
    cmap = plt.get_cmap(P["cmap"])
    # The cell averages, faint, under the probes' point values
    pc = PolyCollection(polys, cmap=cmap, edgecolor=P["mesh"], lw=0.4, alpha=0.3)
    pc.set_array(np.asarray(field).ravel())
    pc.set_clim(lo, hi)
    ax.add_collection(pc)
    for name, d in data.items():
        if name == "points":
            ax.scatter(d["x"], d["y"], c=d["P"], cmap=cmap, vmin=lo, vmax=hi, s=70, edgecolor=P["fg"], lw=0.8,
                       zorder=3)
            continue
        xy = np.column_stack([d["x"], d["y"]])
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, linewidths=4.5, zorder=3)
        lc.set_array(0.5 * (d["P"][:-1] + d["P"][1:]))
        lc.set_clim(lo, hi)
        ax.add_collection(lc)
    labels = {"points": (0.2, 0.27, "points"), "line": (0.12, 0.56, "line"), "arc": (0.55, 0.9, "arc"),
              "curve": (0.75, 0.07, "curve x(s), y(s)"), "polyline": (1.2, 0.98, "polyline")}
    for x, y, text in labels.values():
        label(ax, x, y, text, P, size=12, box=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("pressure (point values)")
    cb.outline.set_visible(False)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.0)
    bare(ax)
    return fig


# ------------------------------------------------------ walls of an annulus

R_IN, R_OUT, MACH = 1.0, 1.4, 0.8


def annulus_msh(path, n_r=6, n_t=72):
    """Gmsh 2.2 O-grid of the annulus R_IN < r < R_OUT, zones "inner" and "outer"."""
    nodes, elements = [], []
    for j in range(n_r + 1):
        r = R_IN + (R_OUT - R_IN) * j / n_r
        for i in range(n_t):
            t = 2 * math.pi * i / n_t
            nodes.append((r * math.cos(t), r * math.sin(t)))
    nid = lambda i, j: j * n_t + (i % n_t) + 1  # noqa: E731
    for i in range(n_t):
        elements.append(f"1 2 1 1 {nid(i, 0)} {nid(i + 1, 0)}")
        elements.append(f"1 2 2 2 {nid(i, n_r)} {nid(i + 1, n_r)}")
    for j in range(n_r):
        for i in range(n_t):
            elements.append(f"3 2 0 1 {nid(i, j)} {nid(i + 1, j)} {nid(i + 1, j + 1)} {nid(i, j + 1)}")
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "2", '1 1 "inner"', '1 2 "outer"',
             "$EndPhysicalNames", "$Nodes", str(len(nodes))]
    lines += [f"{k + 1} {x:.17g} {y:.17g} 0" for k, (x, y) in enumerate(nodes)]
    lines += ["$EndNodes", "$Elements", str(len(elements))]
    lines += [f"{k + 1} {e}" for k, e in enumerate(elements)]
    lines += ["$EndElements"]
    Path(path).write_text("\n".join(lines) + "\n")


def vortex_p(r):
    return (1 + 0.2 * MACH**2 * (1 - 1 / r**2)) ** 3.5 / 1.4


@functools.cache
def wall_data(mallard):
    work = Path(tempfile.mkdtemp())
    annulus_msh(work / "annulus.msh")
    b = f"(1 + 0.2 * {MACH}^2 * (1 - 1 / (x^2 + y^2)))"
    toml = f"""
[run]
n_steps = 1
dt = 1.0e-9
[mesh]
type = "file"
filename = "annulus.msh"
[[mesh.curved]]
zone = "inner"
shape = "circle"
center = [0.0, 0.0]
radius = {R_IN}
[[mesh.curved]]
zone = "outer"
shape = "circle"
center = [0.0, 0.0]
radius = {R_OUT}
[initialize]
type = "analytical"
rho = "{b}^2.5"
u = ["-{MACH} * y / (x^2 + y^2)", "{MACH} * x / (x^2 + y^2)"]
p = "{b}^3.5 / 1.4"
[[boundaries]]
name = "inner"
type = "symmetry"
[[boundaries]]
name = "outer"
type = "wall_adiabatic"
[numerics.face_reconstruction]
type = "TENO"
order = 4
troubled_threshold = 1.0
[[write_data]]
prefix = "mesh"
format = "vtu"
interval = 1
variables = ["P"]
{PHYSICS}"""
    for zone in ("inner", "outer"):
        for sampling in ("cell", "point"):
            toml += (f'\n[[probes]]\nname = "{zone}_{sampling}"\nboundary = "{zone}"\nsampling = "{sampling}"\n'
                     f'variables = ["P"]\n')
    run(mallard, toml, work)
    polys, _ = polygons(work / "mesh_000000.vtu")
    data = {f"{z}_{s}": read_csv(work / f"probe_{z}_{s}.csv", step=0) for z in ("inner", "outer")
            for s in ("cell", "point")}
    return polys, data


def wall_polylines(P, args):
    polys, data = wall_data(args.mallard)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.25))
    ax.add_collection(PolyCollection(polys, facecolor="none", edgecolor=P["mesh"], lw=0.4))
    for zone, color in (("inner", P["teal"]), ("outer", P["flame"])):
        d = data[f"{zone}_point"]
        xy = np.column_stack([d["x"], d["y"]])
        ax.plot(*np.vstack([xy, xy[:1]]).T, color=color, lw=2.2)
        # Traversal direction: the fluid is on the left
        k = len(xy) // 8
        ax.add_patch(FancyArrowPatch(xy[k], xy[k + 3], arrowstyle="-|>", mutation_scale=16, color=color, lw=2))
        ax.plot(*xy[0], "o", color=color, ms=6)
        label(ax, *(1.18 * xy[k + 6] if zone == "outer" else 0.62 * xy[k + 6]), f"{zone}, s = 0 at the dot", P,
              color=color, size=11, box=True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    bare(ax)

    d_cell, d_point = data["inner_cell"], data["inner_point"]
    s = d_point["s"]
    ax2.plot(s, np.full_like(s, vortex_p(R_IN)), color=P["muted"], lw=1.2, ls="--", label="exact")
    ax2.plot(s, d_cell["P"], "o", color=P["plum"], ms=3.5, label="cell averages")
    ax2.plot(s, d_point["P"], "o", color=P["teal"], ms=3.5, label="point values")
    ax2.set_xlabel("arc length s along the inner wall")
    ax2.set_ylabel("wall pressure")
    ax2.legend(loc="upper right", ncols=3, fontsize=11, bbox_to_anchor=(1.0, 1.12))
    ax2.set_xlim(0, 2 * math.pi * R_IN)
    return fig


# ------------------------------------------------------- 3D wall heat flux

@functools.cache
def wall3d_data(mallard3d):
    work = Path(tempfile.mkdtemp())
    blob = "exp(-((x - 0.3)^2 + (z - 0.4)^2) / 0.015 - (y - 0.05)^2 / 0.01)"
    toml = f"""
[run]
n_steps = 120
cfl = 0.8
[mesh]
type = "cartesian"
Nx = 40
Ny = 10
Nz = 32
Lx = 1.0
Ly = 0.4
Lz = 0.8
[initialize]
type = "analytical"
rho = "1.0 / (1.0 + 0.6 * {blob})"
u = ["0.5 * y / 0.4", "0.0", "0.08 * sin(2 * pi * x)"]
p = "1.0"
[[boundaries]]
name = "bottom"
type = "wall_isothermal"
T = 1.0
[[boundaries]]
name = "top"
type = "wall_adiabatic"
u = [0.5, 0.0, 0.0]
""" + "".join(f'[[boundaries]]\nname = "{z}"\ntype = "symmetry"\n' for z in ("left", "right", "back", "front")) + f"""
[numerics.face_reconstruction]
type = "MUSCL"
[physics]
type = "navier_stokes"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0
mu = 0.01
Pr = 0.72
[[probes]]
name = "wall"
boundary = "bottom"
format = "vtp"
interval = 120
variables = ["Q_W", "CF"]
reference = {{rho = 1.0, u = [0.5, 0.0, 0.0], p = 1.0}}
"""
    run(mallard3d, toml, work)
    return read_vtp(work / "probe_wall_000120.vtp")


def wall_heat_flux(P, args):
    a = wall3d_data(args.mallard3d)
    pts, conn, offs, q = a["Points"], a["connectivity"][:, 0], a["offsets"][:, 0], a["Q_W"][:, 0]
    starts = np.concatenate([[0], offs[:-1]])
    # An oblique view of the wall (x right, z into the page)
    view = lambda p: np.column_stack([p[:, 0] + 0.45 * p[:, 2], 0.55 * p[:, 2] + p[:, 1]])  # noqa: E731
    faces = [view(pts[conn[s:e]]) for s, e in zip(starts, offs)]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    cmap = plt.get_cmap(P["cmap"])
    pc = PolyCollection(faces, cmap=cmap, edgecolor=P["bg"], lw=0.3)
    pc.set_array(-q)
    ax.add_collection(pc)
    # The rest of the box, as wire edges
    for a, b in [((0, 0, 0.8), (0, 0.4, 0.8)), ((1, 0, 0.8), (1, 0.4, 0.8)), ((0, 0.4, 0.8), (1, 0.4, 0.8)),
                 ((0, 0, 0), (0, 0.4, 0)), ((1, 0, 0), (1, 0.4, 0)), ((0, 0.4, 0), (1, 0.4, 0)),
                 ((0, 0.4, 0), (0, 0.4, 0.8)), ((1, 0.4, 0), (1, 0.4, 0.8))]:
        ax.plot(*view(np.array([a, b], dtype=float)).T, color=P["muted"], lw=0.9, ls=(0, (3, 2)))
    label(ax, 0.82, 0.97, "isothermal wall, moving lid above", P, size=11, color=P["muted"])
    ax.add_patch(FancyArrowPatch((0.05, -0.12), (0.45, -0.12), arrowstyle="-|>", mutation_scale=14, color=P["fg"]))
    label(ax, 0.25, -0.2, "flow", P, size=11)
    cb = fig.colorbar(pc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("heat flux into the wall, -Q_W")
    cb.outline.set_visible(False)
    ax.set_xlim(-0.05, 1.45)
    ax.set_ylim(-0.27, 1.02)
    bare(ax)
    return fig


class Args(tuple):
    def __new__(cls, mallard, mallard3d):
        return super().__new__(cls, (mallard, mallard3d))

    mallard = property(lambda self: self[0])
    mallard3d = property(lambda self: self[1])


FIGURES = {"probe_shapes": (probe_shapes, False), "probe_walls": (wall_polylines, False),
           "probe_wall_3d": (wall_heat_flux, True)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="2D Mallard executable")
    ap.add_argument("--mallard3d", help="3D Mallard executable, for the 3D figures")
    ap.add_argument("names", nargs="*")
    ns = ap.parse_args()
    args = Args(Path(ns.mallard).resolve(), Path(ns.mallard3d).resolve() if ns.mallard3d else None)
    for name, (draw, three_d) in FIGURES.items():
        if (ns.names and name not in ns.names) or (three_d and not args.mallard3d):
            continue
        figstyle.save(name, lambda P, d=draw: d(P, args))


if __name__ == "__main__":
    sys.exit(main())
