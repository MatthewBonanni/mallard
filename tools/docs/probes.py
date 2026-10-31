"""Figures of the probes and region snapshots sections of the user guide, from
short Mallard runs (seconds each on a CPU build): probe shapes over a 2D mesh,
boundary probes on curved walls (polylines with arc length), a 3D wall's heat
flux, an exact cut plane and a sphere, measurements around a cylinder at
Re = 40 (wall CP, CF and y+, fluxes through a closed circle and a cut,
reductions), and the snapshots of a disk's cells and of three slices.

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
        rows = [r for r in csv.DictReader(f) if not r["step"].startswith("#")]
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


def annulus_msh(path, n_r=6, n_t=72, r_in=R_IN, r_out=R_OUT, ratio=1.0):
    """Gmsh 2.2 O-grid of the annulus r_in < r < r_out, zones "inner" and "outer"; ratio
    is the growth of the radial spacing from cell to cell."""
    nodes, elements = [], []
    w = ratio ** np.arange(n_r)
    radii = r_in + (r_out - r_in) * np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    for j in range(n_r + 1):
        r = radii[j]
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


# ------------------------------------------- 3D cut plane and sphere surface

@functools.cache
def cut3d_data(mallard3d):
    work = Path(tempfile.mkdtemp())
    blob = "exp(-((x - 0.35)^2 + (y - 0.6)^2 + (z - 0.45)^2) / 0.03)"
    toml = f"""
[run]
n_steps = 1
dt = 1.0e-9
[mesh]
type = "cartesian_tet"
Nx = 10
Ny = 10
Nz = 10
[initialize]
type = "analytical"
rho = "1.0"
u = ["0.0", "0.0", "0.0"]
p = "1.0 + 0.5 * {blob} - 0.3 * exp(-((x - 0.7)^2 + (y - 0.3)^2 + (z - 0.6)^2) / 0.02)"
""" + "".join(f'[[boundaries]]\nname = "{z}"\ntype = "symmetry"\n'
              for z in ("left", "right", "bottom", "top", "back", "front")) + f"""
[numerics.face_reconstruction]
type = "TENO"
order = 3
troubled_threshold = 1.0
{PHYSICS}
[[probes]]
name = "cut"
shape = "cut_plane"
point = [0.5, 0.5, 0.5]
normal = [0.6, 0.35, 1.0]
sampling = "point"
format = "vtp"
variables = ["P"]

[[probes]]
name = "sphere"
shape = "sphere"
center = [0.5, 0.5, 0.5]
radius = 0.32
n_points = [24, 48]
sampling = "point"
format = "vtp"
variables = ["P"]
"""
    run(mallard3d, toml, work)
    return read_vtp(work / "probe_cut_000000.vtp"), read_vtp(work / "probe_sphere_000000.vtp")


def _view(p, azimuth=-0.6, elevation=0.45):
    """Orthographic view: screen (x, y) and depth of points p (n x 3)."""
    ca, sa, ce, se = math.cos(azimuth), math.sin(azimuth), math.cos(elevation), math.sin(elevation)
    x = ca * p[:, 0] - sa * p[:, 1]
    y0 = sa * p[:, 0] + ca * p[:, 1]
    return np.column_stack([x, ce * p[:, 2] - se * y0]), se * p[:, 2] + ce * y0


def _box(ax, P):
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
    for a in range(8):
        for b in range(a + 1, 8):
            if np.sum(np.abs(corners[a] - corners[b])) == 1:
                xy, _ = _view(corners[[a, b]])
                ax.plot(*xy.T, color=P["muted"], lw=0.8, ls=(0, (3, 2)))


def cut_and_sphere(P, args):
    cut, sphere = cut3d_data(args.mallard3d)
    lo, hi = 0.75, 1.45
    cmap = plt.get_cmap(P["cmap"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), gridspec_kw=dict(wspace=0.02))
    # Exact cut through tetrahedra: one polygon per cut cell, colored by its point value
    a = cut
    starts = np.concatenate([[0], a["offsets"][:-1, 0]])
    polys = [_view(a["Points"][a["connectivity"][s:e, 0]])[0] for s, e in zip(starts, a["offsets"][:, 0])]
    pc = PolyCollection(polys, cmap=cmap, edgecolor=P["mesh"], lw=0.3)
    pc.set_array(a["P"][:, 0])
    pc.set_clim(lo, hi)
    axes[0].add_collection(pc)
    _box(axes[0], P)
    axes[0].set_title("cut_plane through tetrahedra", fontsize=12)
    # Sphere: quadrilaterals of the grid facing the viewer, colored by the mean of their corners
    b = sphere
    pts, conn, offs, values = b["Points"], b["connectivity"][:, 0], b["offsets"][:, 0], b["P"][:, 0]
    starts = np.concatenate([[0], offs[:-1]])
    xy, depth = _view(pts)
    _, toward = _view(np.array([[0.5, 0.5, 0.5]]))
    quads, colors, order = [], [], []
    for s, e in zip(starts, offs):
        idx = conn[s:e]
        quads.append(xy[idx])
        colors.append(values[idx].mean())
        order.append(depth[idx].mean())
    keep = np.argsort(order)  # painter's algorithm: far first
    pc2 = PolyCollection([quads[k] for k in keep], cmap=cmap, edgecolor=P["bg"], lw=0.2)
    pc2.set_array(np.array(colors)[keep])
    pc2.set_clim(lo, hi)
    axes[1].add_collection(pc2)
    _box(axes[1], P)
    axes[1].set_title("sphere, resampled with point values", fontsize=12)
    corners, _ = _view(np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float))
    for ax in axes:
        ax.set_xlim(corners[:, 0].min() - 0.03, corners[:, 0].max() + 0.03)
        ax.set_ylim(corners[:, 1].min() - 0.03, corners[:, 1].max() + 0.03)
        bare(ax)
    cb = fig.colorbar(pc, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("pressure")
    cb.outline.set_visible(False)
    return fig


# ------------------------------------------ measurements around a cylinder

D, U_INF, RE, T_MEAS = 1.0, 0.3, 40.0, 20.0
CUT_X, CUT_H = 3.0, 4.0


@functools.cache
def measure_data(mallard):
    """Re = 40 flow past a cylinder: wall coefficients, fluxes through a closed circle and a cut, reductions."""
    work = Path(tempfile.mkdtemp())
    annulus_msh(work / "cylinder.msh", n_r=40, n_t=96, r_in=0.5 * D, r_out=15.0, ratio=1.1)
    p_inf, T_inf = 1 / 1.4, 1 / 1.4
    toml = f"""
[run]
t_stop = {T_MEAS}
cfl = 1.0
[mesh]
type = "file"
filename = "cylinder.msh"
[[mesh.curved]]
zone = "inner"
shape = "circle"
center = [0.0, 0.0]
radius = {0.5 * D}
[initialize]
type = "analytical"
rho = "1.0"
u = ["{U_INF} * (1 - exp(-(x^2 + y^2 - 0.25) / 0.5))", "0.0"]
p = "{p_inf}"
[[boundaries]]
name = "inner"
type = "wall_adiabatic"
[[boundaries]]
name = "outer"
type = "farfield"
u = [{U_INF}, 0.0]
p = {p_inf}
T = {T_inf}
[numerics.face_reconstruction]
type = "MUSCL"
[physics]
type = "navier_stokes"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0
mu = {U_INF * D / RE}
Pr = 0.72
[[write_data]]
prefix = "mesh"
format = "vtu"
time_interval = {T_MEAS}
variables = ["U_X"]
[[probes]]
name = "wall"
boundary = "inner"
variables = ["CP", "CF", "Y_PLUS", "TRACTION_PRESSURE", "TRACTION_VISCOUS"]
reference = {{rho = 1.0, u = [{U_INF}, 0.0], p = {p_inf}}}
reductions = ["integral", "mean", "max"]
time_interval = 0.5
[[probes]]
name = "circle"
shape = "circle"
center = [0.0, 0.0]
radius = 1.5
n_points = 120
variables = ["MASS_FLUX"]
reductions = ["integral"]
time_interval = 0.5
[[probes]]
name = "cut"
shape = "cut_plane"
point = [{CUT_X}, 0.0]
normal = [1.0, 0.0]
bounds = [[{CUT_X - 1}, {-CUT_H}], [{CUT_X + 1}, {CUT_H}]]
variables = ["MASS_FLUX"]
reductions = ["integral"]
time_interval = 0.5
"""
    run(mallard, toml, work)
    polys, _ = polygons(sorted(work.glob("mesh_*.vtu"))[-1])
    out = {name: read_csv(work / f"probe_{name}.csv") for name in ("wall", "circle", "cut")}
    red = {name: read_csv(work / f"probe_{name}_reductions.csv") for name in ("wall", "circle", "cut")}
    return polys, out, red


def last(d):
    """The rows of a probe CSV at its last output."""
    keep = d["step"] == d["step"].max()
    return {k: v[keep] for k, v in d.items()}


def measurements(P, args):
    polys, out, red = measure_data(args.mallard)
    wall, circle, cut = (last(out[k]) for k in ("wall", "circle", "cut"))
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), gridspec_kw=dict(hspace=0.42, wspace=0.32))
    (ax, ax_c), (ax_y, ax_t) = axes

    # The mesh, the closed circle and the cut, colored by their mass flux samples
    window = [q for q in polys if q[:, 0].min() < 5.2 and q[:, 0].max() > -2.2 and np.abs(q[:, 1]).min() < CUT_H + 0.9]
    ax.add_collection(PolyCollection(window, facecolor="none", edgecolor=P["mesh"], lw=0.3))
    ax.add_patch(Polygon(np.column_stack([0.5 * np.cos(np.linspace(0, 2 * np.pi, 97)),
                                          0.5 * np.sin(np.linspace(0, 2 * np.pi, 97))]),
                         facecolor=P["bg"], edgecolor=P["fg"], lw=1.2))
    for d in (circle, cut):
        order = np.argsort(d["y"]) if d is cut else np.arange(len(d["x"]))
        xy = np.column_stack([d["x"][order], d["y"][order]])
        if d is circle:
            xy = np.vstack([xy, xy[:1]])
        f = d["MASS_FLUX"][order] / U_INF
        f = np.concatenate([f, f[:1]]) if d is circle else f
        lc = LineCollection(np.stack([xy[:-1], xy[1:]], axis=1), cmap="coolwarm", linewidths=4.5, zorder=3)
        lc.set_array(0.5 * (f[:-1] + f[1:]))
        lc.set_clim(-1.2, 1.2)
        ax.add_collection(lc)
    label(ax, 0.0, 1.95, "closed circle", P, size=11, box=True)
    label(ax, CUT_X, CUT_H + 0.4, "cut_plane", P, size=11, box=True)
    ax.set_xlim(-2.2, 5.2)
    ax.set_ylim(-CUT_H - 0.4, CUT_H + 0.9)
    ax.set_title("mass flux through a circle and a cut", fontsize=12)
    bare(ax)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-1.2, 1.2))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
    cb.set_label(r"MASS_FLUX / $\rho_\infty U_\infty$")
    cb.outline.set_visible(False)

    # Wall coefficients against the angle from the front stagnation point (upper half)
    theta = np.degrees(np.arctan2(wall["y"], -wall["x"]))
    upper = np.argsort(theta)[theta[np.argsort(theta)] >= 0]
    th = theta[upper]
    ax_c.plot(th, wall["CP"][upper], color=P["teal"], label="CP")
    ax_c.set_ylabel("CP", color=P["teal"])
    ax_f = ax_c.twinx()
    ax_f.spines["right"].set_visible(True)
    ax_f.plot(th, wall["CF"][upper], color=P["flame"], label="CF")
    ax_f.axhline(0.0, color=P["muted"], lw=0.8, ls=":")
    ax_f.set_ylabel("CF", color=P["flame"])
    ax_c.set_xlabel("angle from the front stagnation point (deg)")
    ax_c.set_xlim(0, 180)
    ax_c.set_title("wall probe: CP and CF", fontsize=12)

    # y+ samples, with the probe's mean and max reductions
    r = red["wall"]
    k = np.argmax(r["step"])
    ax_y.plot(th, wall["Y_PLUS"][upper], "o", color=P["plum"], ms=3.5, label="samples")
    ax_y.axhline(r["max_Y_PLUS"][k], color=P["flame"], ls="--", lw=1.2, label="max")
    ax_y.axhline(r["mean_Y_PLUS"][k], color=P["teal"], ls="--", lw=1.2, label="mean")
    ax_y.set_xlabel("angle from the front stagnation point (deg)")
    ax_y.set_ylabel("first-cell y+")
    ax_y.set_xlim(0, 180)
    ax_y.set_ylim(0, 1.3 * r["max_Y_PLUS"][k])
    ax_y.legend(loc="upper right", ncols=3, fontsize=11)
    ax_y.set_title("Y_PLUS and its reductions", fontsize=12)

    # Integrals over time: drag from the wall tractions, net outflow through the closed circle
    q = 0.5 * U_INF**2 * D
    t = r["t"]
    ax_t.plot(t, r["integral_TRACTION_PRESSURE_X"] / q, color=P["teal"], label="wall: pressure drag, C$_D$")
    ax_t.plot(t, r["integral_TRACTION_VISCOUS_X"] / q, color=P["flame"], label="wall: viscous drag, C$_D$")
    rc = red["circle"]
    ax_t.plot(rc["t"], rc["integral_MASS_FLUX"] / (U_INF * D), color=P["plum"],
              label=r"circle: net outflow / $\rho_\infty U_\infty D$")
    ax_t.axhline(0.0, color=P["muted"], lw=0.8, ls=":")
    ax_t.set_xlabel("time")
    ax_t.set_ylabel("integral, normalized")
    ax_t.set_xlim(0, T_MEAS)
    ax_t.set_ylim(-0.6, 2.4)
    ax_t.legend(loc="upper right", fontsize=10.5)
    ax_t.set_title("integral reductions over time", fontsize=12)
    return fig


class Args(tuple):
    def __new__(cls, mallard, mallard3d):
        return super().__new__(cls, (mallard, mallard3d))

    mallard = property(lambda self: self[0])
    mallard3d = property(lambda self: self[1])


# ------------------------------------------------- snapshots of regions


@functools.cache
def snapshot_data(mallard):
    work = Path(tempfile.mkdtemp())
    disk = "{type = \"disk\", center = [0.5, 0.55], radius = 0.3}"
    cuts = ", ".join(f'{{type = "cut_plane", point = [{x}, 0.5], normal = [1.0, 0.25]}}' for x in (0.95, 1.15, 1.35))
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
[[write_data]]
prefix = "near"
format = "vtu"
interval = 1
variables = ["P"]
region = {disk}
[[write_data]]
prefix = "cut"
format = "vtu"
interval = 1
variables = ["P"]
regions = [{cuts}]
{PHYSICS}"""
    run(mallard, toml, work)
    mesh, _ = polygons(work / "mesh_000000.vtu")
    near, near_data = polygons(work / "near_000000.vtu")
    slices = []
    for k in range(3):
        a = read_vtp(work / f"cut_{k}_000000.vtp")
        xy, conn = a["Points"][:, :2], a["connectivity"].ravel()
        slices.append((np.stack([xy[conn[0::2]], xy[conn[1::2]]], axis=1), a["P"].ravel()))
    return mesh, near, near_data, slices


def region_snapshots(P, args):
    mesh, near, near_data, slices = snapshot_data(args.mallard)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    lo, hi = 0.8, 1.4
    cmap = plt.get_cmap(P["cmap"])
    for ax in axes:
        ax.add_collection(PolyCollection(mesh, facecolor="none", edgecolor=P["mesh"], lw=0.4, alpha=0.6))
        ax.add_patch(plt.Circle((0.5, 0.55), 0.3, fill=False, ls="--", lw=1.2, color=P["fg"], zorder=4))
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 1.0)
        bare(ax)
    frac = PolyCollection(near, cmap=plt.get_cmap(P["cmap"]), edgecolor=P["mesh"], lw=0.4)
    frac.set_array(near_data["REGION_FRACTION"].ravel())
    frac.set_clim(0.0, 1.0)
    axes[0].add_collection(frac)
    label(axes[0], 0.5, 0.94, "the disk's cells: REGION_FRACTION", P, size=12, box=True)
    cells = PolyCollection(near, cmap=cmap, edgecolor=P["mesh"], lw=0.4)
    cells.set_array(near_data["P"].ravel())
    cells.set_clim(lo, hi)
    axes[1].add_collection(cells)
    for segs, values in slices:
        lc = LineCollection(segs, cmap=cmap, linewidths=5.0, zorder=3)
        lc.set_array(values)
        lc.set_clim(lo, hi)
        axes[1].add_collection(lc)
    label(axes[1], 0.5, 0.94, "their pressure, and three slices", P, size=12, box=True)
    for ax, (m, lab) in zip(axes, ((plt.Normalize(0.0, 1.0), "cell fraction"), (plt.Normalize(lo, hi), "pressure"))):
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=m), ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(lab)
        cb.outline.set_visible(False)
    return fig


FIGURES = {"region_snapshots": (region_snapshots, False), "probe_shapes": (probe_shapes, False),
           "probe_walls": (wall_polylines, False), "probe_wall_3d": (wall_heat_flux, True),
           "probe_cut_3d": (cut_and_sphere, True), "probe_measurements": (measurements, False)}


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
