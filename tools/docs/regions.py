"""Figures of regions ([[sources]] region tables): the cell volume fractions
that Mallard computes on coarse unstructured meshes, read back from runs of
one step whose source is 1 per unit volume over the region (SOURCE_RHOE is
then the fraction), with the exact outline drawn over them; and the density
of an initial state built layer by layer ([[initialize.data]] and
[[initialize.regions]]) on the same mesh.

    python tools/docs/regions.py --mallard build/src/mallard [--mallard3d build-3d/src/mallard] [names...]

Without --mallard3d the 3D figure is skipped.
"""
import argparse
import functools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import figstyle
from figstyle import bare, label

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from mallard.vtu import probe_sampling, read_vtu_cells  # noqa: E402

PHYSICS = '[physics]\ntype = "euler"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n'


# ------------------------------------------------------------------ meshes

def triangle_mesh(path, n=12, jitter=0.3, seed=7):
    """Gmsh 2.2 file of the unit square: Delaunay triangles of jittered points, boundary zone "wall"."""
    rng = np.random.default_rng(seed)
    g = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inner = (pts[:, 0] > 0) & (pts[:, 0] < 1) & (pts[:, 1] > 0) & (pts[:, 1] < 1)
    pts[inner] += rng.uniform(-jitter, jitter, (inner.sum(), 2)) / n
    tri = Delaunay(pts).simplices
    edges = {}
    for t in tri:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (min(a, b), max(a, b))
            edges[key] = edges.get(key, 0) + 1
    boundary = [e for e, count in edges.items() if count == 1]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "1", '1 1 "wall"', "$EndPhysicalNames",
             "$Nodes", str(len(pts))]
    lines += [f"{i + 1} {x:.17g} {y:.17g} 0" for i, (x, y) in enumerate(pts)]
    lines += ["$EndNodes", "$Elements", str(len(boundary) + len(tri))]
    k = 1
    for a, b in boundary:
        lines.append(f"{k} 1 2 1 1 {a + 1} {b + 1}")
        k += 1
    for t in tri:
        # Counterclockwise
        p = pts[t]
        if (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0]) < 0:
            t = t[[0, 2, 1]]
        lines.append(f"{k} 2 2 0 2 {t[0] + 1} {t[1] + 1} {t[2] + 1}")
        k += 1
    lines.append("$EndElements")
    path.write_text("\n".join(lines) + "\n")


def tetrahedron_mesh(path, n=14, jitter=0.25, seed=11):
    """Gmsh 2.2 file of the unit cube: each jittered hexahedron split into 6 tetrahedra, boundary zone "wall"."""
    rng = np.random.default_rng(seed)
    g = np.linspace(0.0, 1.0, n + 1)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    inner = np.all((pts > 0) & (pts < 1), axis=1)
    pts[inner] += rng.uniform(-jitter, jitter, (inner.sum(), 3)) / n
    nid = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k  # noqa: E731
    tets, tris = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [nid(i + a, j + b, k + d) for a, b, d in
                     ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
                for t in ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)):
                    tets.append([c[v] for v in t])
    faces = {}
    for t in tets:
        for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            key = tuple(sorted(t[v] for v in f))
            faces[key] = faces.get(key, 0) + 1
    tris = [f for f, count in faces.items() if count == 1]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "1", '2 1 "wall"', "$EndPhysicalNames",
             "$Nodes", str(len(pts))]
    lines += [f"{i + 1} {x:.17g} {y:.17g} {z:.17g}" for i, (x, y, z) in enumerate(pts)]
    lines += ["$EndNodes", "$Elements", str(len(tris) + len(tets))]
    k = 1
    for f in tris:
        lines.append(f"{k} 2 2 1 1 {f[0] + 1} {f[1] + 1} {f[2] + 1}")
        k += 1
    for t in tets:
        p = pts[t]
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0]) < 0:
            t = [t[0], t[2], t[1], t[3]]
        lines.append(f"{k} 4 2 0 2 {' '.join(str(v + 1) for v in t)}")
        k += 1
    lines.append("$EndElements")
    path.write_text("\n".join(lines) + "\n")


def holed_mesh(path, n=14, jitter=0.25, seed=3, hole=(0.3, 0.5, 0.12)):
    """Gmsh 2.2 file of the unit square with a round hole: boundary zones "outer" and "hole", cell zones "left_half"
    and "right_half" (cell centroids left and right of x = 0.5)."""
    rng = np.random.default_rng(seed)
    hx, hy, hr = hole
    g = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inner = (pts[:, 0] > 0) & (pts[:, 0] < 1) & (pts[:, 1] > 0) & (pts[:, 1] < 1)
    pts[inner] += rng.uniform(-jitter, jitter, (inner.sum(), 2)) / n
    pts = pts[np.hypot(pts[:, 0] - hx, pts[:, 1] - hy) > hr + 0.4 / n]
    m = int(np.ceil(2 * np.pi * hr * n * 1.3))
    a = np.linspace(0.0, 2 * np.pi, m, endpoint=False)
    pts = np.vstack([pts, np.column_stack([hx + hr * np.cos(a), hy + hr * np.sin(a)])])
    tri = Delaunay(pts).simplices
    c = pts[tri].mean(axis=1)
    tri = tri[np.hypot(c[:, 0] - hx, c[:, 1] - hy) > hr]
    edges = {}
    for t in tri:
        for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (min(u, v), max(u, v))
            edges[key] = edges.get(key, 0) + 1
    boundary = [e for e, count in edges.items() if count == 1]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "4", '1 1 "outer"', '1 2 "hole"',
             '2 3 "left_half"', '2 4 "right_half"', "$EndPhysicalNames", "$Nodes", str(len(pts))]
    lines += [f"{i + 1} {x:.17g} {y:.17g} 0" for i, (x, y) in enumerate(pts)]
    lines += ["$EndNodes", "$Elements", str(len(boundary) + len(tri))]
    k = 1
    for u, v in boundary:
        mid = 0.5 * (pts[u] + pts[v])
        zone = 2 if np.hypot(mid[0] - hx, mid[1] - hy) < hr + 0.5 / n else 1
        lines.append(f"{k} 1 2 {zone} {zone} {u + 1} {v + 1}")
        k += 1
    for t in tri:
        q = pts[t]
        if (q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[1, 1] - q[0, 1]) * (q[2, 0] - q[0, 0]) < 0:
            t = t[[0, 2, 1]]
        zone = 3 if q[:, 0].mean() < 0.5 else 4
        lines.append(f"{k} 2 2 {zone} {zone} {t[0] + 1} {t[1] + 1} {t[2] + 1}")
        k += 1
    lines.append("$EndElements")
    path.write_text("\n".join(lines) + "\n")
    return pts, [pts[[u, v]] for u, v in boundary]


# ------------------------------------------------------------------- runs

def fractions(binary, mesh, region, work, name, dim, zones=("wall",)):
    """Cells of the mesh and the region's volume fraction in each, from a one-step run."""
    zero = "[0.0, 0.0]" if dim == 2 else "[0.0, 0.0, 0.0]"
    boundaries = "".join(f'[[boundaries]]\nname = "{z}"\ntype = "symmetry"\n' for z in zones)
    toml = f"""[run]
n_steps = 1
dt = 1e-9
[mesh]
type = "file"
filename = "{mesh}"
[initialize]
type = "constant"
u = {zero}
p = 1.0
T = 1.0
{boundaries}[[sources]]
rhoE = 1
region = {region}
[output]
check_interval = 1000000
[[write_data]]
prefix = "{name}"
format = "vtu"
interval = 1
fields = ["SOURCE_RHOE"]
{PHYSICS}"""
    (work / f"{name}.toml").write_text(toml)
    subprocess.run([str(binary), "-i", f"{name}.toml", "--kokkos-num-threads=2"], cwd=work, check=True,
                   stdout=subprocess.DEVNULL)
    pts, conn, offs, types, data = read_vtu_cells(str(work / f"{name}_000000.vtu"))
    starts = np.concatenate([[0], offs[:-1]])
    cells = [conn[a:b] for a, b in zip(starts, offs)]
    return pts, cells, data["SOURCE_RHOE"]


# 2D panels: (title, region table, inside(x, y) for the outline)
def _rot(x, y, deg, cx=0.5, cy=0.5):
    a = np.radians(deg)
    dx, dy = x - cx, y - cy
    return np.cos(a) * dx + np.sin(a) * dy, -np.sin(a) * dx + np.cos(a) * dy


PRIMITIVES = [
    ("disk", '{type = "disk", center = [0.5, 0.5], radius = 0.32}',
     lambda x, y: np.hypot(x - 0.5, y - 0.5) < 0.32),
    ("annulus", '{type = "annulus", center = [0.5, 0.5], radius = 0.38, inner_radius = 0.2}',
     lambda x, y: (np.hypot(x - 0.5, y - 0.5) < 0.38) & (np.hypot(x - 0.5, y - 0.5) > 0.2)),
    ("ellipsoid (slice), rotated", '{type = "ellipsoid", center = [0, 0], semi_axes = [0.4, 0.18], rotate = 30, '
     'translate = [0.5, 0.5]}',
     lambda x, y: (_rot(x, y, 30)[0] / 0.4) ** 2 + (_rot(x, y, 30)[1] / 0.18) ** 2 < 1),
    ("rectangle, rotated", '{type = "rectangle", center = [0.5, 0.5], size = [0.6, 0.3], rotate = -25, '
     'pivot = [0.5, 0.5]}',
     lambda x, y: (abs(_rot(x, y, -25)[0]) < 0.3) & (abs(_rot(x, y, -25)[1]) < 0.15)),
    ("polygon", '{type = "polygon", vertices = [[0.1, 0.15], [0.9, 0.3], [0.55, 0.5], [0.75, 0.9], [0.2, 0.7]]}',
     None),
    ("cone (slice)", '{type = "cone", base = [0.15, 0.5], axis = [1, 0], length = 0.7, base_radius = 0.35, '
     'top_radius = 0.08}',
     lambda x, y: (x > 0.15) & (x < 0.85) & (abs(y - 0.5) < 0.35 + (0.08 - 0.35) * (x - 0.15) / 0.7)),
    ("torus (slice)", '{type = "torus", center = [0.5, 0.5], axis = [0, 1, 0], major_radius = 0.25, '
     'minor_radius = 0.13}',
     lambda x, y: np.hypot(abs(x - 0.5) - 0.25, y - 0.5) < 0.13),
    ("implicit", '{type = "implicit", f = "hypot(x - 0.5, y - 0.5) - 0.28 - 0.08 * cos(5 * atan2(y - 0.5, x - 0.5))"}',
     lambda x, y: np.hypot(x - 0.5, y - 0.5) < 0.28 + 0.08 * np.cos(5 * np.arctan2(y - 0.5, x - 0.5))),
]

BOOLEANS = [
    ("difference", '{type = "difference", a = {type = "disk", center = [0.5, 0.5], radius = 0.38}, '
     'b = {type = "rectangle", center = [0.5, 0.5], size = [0.5, 0.16], rotate = 45, pivot = [0.5, 0.5]}}',
     lambda x, y: (np.hypot(x - 0.5, y - 0.5) < 0.38)
     & ~((abs(_rot(x, y, 45)[0]) < 0.25) & (abs(_rot(x, y, 45)[1]) < 0.08))),
    ("union", '{type = "union", regions = [{type = "disk", center = [0.32, 0.4], radius = 0.2}, '
     '{type = "disk", center = [0.66, 0.6], radius = 0.22}, {type = "rectangle", center = [0.5, 0.5], '
     'size = [0.7, 0.08]}]}',
     lambda x, y: (np.hypot(x - 0.32, y - 0.4) < 0.2) | (np.hypot(x - 0.66, y - 0.6) < 0.22)
     | ((abs(x - 0.5) < 0.35) & (abs(y - 0.5) < 0.04))),
    ("intersection", '{type = "intersection", regions = [{type = "annulus", center = [0.5, 0.5], radius = 0.4, '
     'inner_radius = 0.22}, {type = "half_space", point = [0.5, 0.5], normal = [1, -1]}]}',
     lambda x, y: (np.hypot(x - 0.5, y - 0.5) < 0.4) & (np.hypot(x - 0.5, y - 0.5) > 0.22) & (x - y < 0)),
    ("smooth edge", '{type = "disk", center = [0.5, 0.5], radius = 0.3, edge = 0.16}',
     lambda x, y: np.hypot(x - 0.5, y - 0.5) < 0.3),
]

POLYGON = np.array([[0.1, 0.15], [0.9, 0.3], [0.55, 0.5], [0.75, 0.9], [0.2, 0.7]])


@functools.cache
def data_2d(mallard):
    work = Path(tempfile.mkdtemp())
    mesh = work / "square.msh"
    triangle_mesh(mesh)
    out = {}
    for i, (title, region, _) in enumerate(PRIMITIVES + BOOLEANS):
        out[title] = fractions(mallard, mesh.name, region, work, f"r{i}", 2)
    return out


def panel(ax, P, pts, cells, f, inside, title, polygon=None):
    polys = [pts[c, :2] for c in cells]
    colors = plt.get_cmap(P["cmap"])(f)
    colors[f <= 0.0] = (0, 0, 0, 0)
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=P["mesh"], linewidths=0.5))
    if polygon is not None:
        closed = np.vstack([polygon, polygon[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=P["fg"], lw=1.4)
    elif inside is not None:
        g = np.linspace(0, 1, 300)
        X, Y = np.meshgrid(g, g)
        ax.contour(X, Y, inside(X, Y).astype(float), levels=[0.5], colors=[P["fg"]], linewidths=1.4)
    bare(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12)


def colorbar(fig, axes, P):
    sm = plt.cm.ScalarMappable(cmap=P["cmap"], norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("cell volume fraction")
    cb.outline.set_edgecolor(P["muted"])


def primitives_2d(P, args):
    data = data_2d(args.mallard)
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.6), gridspec_kw=dict(wspace=0.06, hspace=0.18))
    for (title, _, inside), ax in zip(PRIMITIVES, axes.flat):
        panel(ax, P, *data[title], inside, title, POLYGON if title == "polygon" else None)
    colorbar(fig, axes, P)
    return fig


def booleans_2d(P, args):
    data = data_2d(args.mallard)
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0), gridspec_kw=dict(wspace=0.06))
    for (title, _, inside), ax in zip(BOOLEANS, axes.flat):
        panel(ax, P, *data[title], inside, title)
    colorbar(fig, axes, P)
    return fig


def smooth_edge(P, args):
    """The smoothstep across the edge, and the cell fractions of a sharp and a smooth disk against radius."""
    data = data_2d(args.mallard)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    e = 0.16
    for title, color, name in (("disk", P["blue"], "sharp, R = 0.32"), ("smooth edge", P["flame"],
                                                                         f"edge = {e}, R = 0.3")):
        pts, cells, f = data[title]
        centroids = np.array([pts[c, :2].mean(axis=0) for c in cells])
        r = np.hypot(centroids[:, 0] - 0.5, centroids[:, 1] - 0.5)
        ax.scatter(r, f, s=10, color=color, alpha=0.7, label=f"cells, {name}", linewidths=0)
    r = np.linspace(0, 0.6, 400)
    s = np.clip(0.5 - (r - 0.3) / e, 0, 1)
    ax.plot(r, s * s * (3 - 2 * s), color=P["fg"], lw=1.6, label="smoothstep indicator")
    ax.axvspan(0.3 - e / 2, 0.3 + e / 2, color=P["faint"], zorder=0)
    label(ax, 0.3, 1.1, "edge", P, size=11, color=P["muted"])
    ax.set_xlabel("distance of the cell centroid from the center")
    ax.set_ylabel("volume fraction")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(-0.05, 1.18)
    ax.legend(loc="lower left", fontsize=10)
    return fig


# --------------------------------------------------------------------- 3D

REGIONS_3D = [
    ("annular cylinder, tilted",
     '{type = "cylinder", center = [0.5, 0.5, 0.5], axis = [1, 1, 0.6], radius = 0.33, inner_radius = 0.18, '
     'length = 0.5}'),
    ("box, rotated", '{type = "box", center = [0, 0, 0], size = [0.65, 0.35, 0.22], '
     'rotate = {axis = [1, 2, 3], angle = 35}, translate = [0.5, 0.5, 0.5]}'),
    ("sphere minus cylinder", '{type = "difference", a = {type = "sphere", center = [0.5, 0.5, 0.5], radius = 0.4}, '
     'b = {type = "cylinder", center = [0.5, 0.5, 0.5], axis = [0, 0, 1], radius = 0.17}}'),
]


@functools.cache
def data_3d(mallard3d):
    work = Path(tempfile.mkdtemp())
    mesh = work / "cube.msh"
    tetrahedron_mesh(mesh)
    return [fractions(mallard3d, mesh.name, region, work, f"r3d{i}", 3) for i, (_, region) in enumerate(REGIONS_3D)]


def regions_3d(P, args):
    """The cells cut by the plane y = 0.5, colored by their fraction, in front of the cells behind it that the
    region fills (faint): a cross-section through the solid."""
    data = data_3d(args.mallard3d)
    fig = plt.figure(figsize=(11.0, 4.0))
    cmap = plt.get_cmap(P["cmap"])
    tris = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    for k, ((title, _), (pts, cells, f)) in enumerate(zip(REGIONS_3D, data)):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d", computed_zorder=False)
        back, cut, colors = [], [], []
        for c, fc in zip(cells, f):
            p = pts[c]
            if p[:, 1].min() < 0.5 < p[:, 1].max():
                if fc > 0.0:
                    cut += [p[list(t)] for t in tris]
                    colors += [cmap(fc)] * 4
            elif p[:, 1].min() >= 0.5 and fc > 0.5:
                back += [p[list(t)] for t in tris]
        ax.add_collection3d(Poly3DCollection(back, facecolors=P["faint"], edgecolors=P["mesh"], linewidths=0.1,
                                             alpha=0.35, rasterized=True))
        ax.add_collection3d(Poly3DCollection(cut, facecolors=colors, edgecolors=P["mesh"], linewidths=0.2,
                                             rasterized=True))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_box_aspect((1, 1, 1), zoom=1.35)
        ax.view_init(elev=18, azim=-75)
        ax.set_axis_off()
        ax.set_title(title, fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=P["cmap"], norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.55, pad=0.02)
    cb.set_label("cell volume fraction")
    cb.outline.set_edgecolor(P["muted"])
    return fig


# ------------------------------------------------------- initial-state layers

LAYERS = [
    ("base: a tabulated profile", "", None),
    ("+ a rotated rectangle", """
[[initialize.regions]]
region = {type = "rectangle", center = [0.66, 0.38], size = [0.5, 0.16], pivot = [0.66, 0.38], rotate = 35}
rho = 0.25
""", lambda x, y: (abs(_rot(x, y, 35, 0.66, 0.38)[0]) < 0.25) & (abs(_rot(x, y, 35, 0.66, 0.38)[1]) < 0.08)),
    ("+ a disk on top, smooth edge", """
[[initialize.regions]]
region = {type = "disk", center = [0.7, 0.7], radius = 0.17, edge = 0.06}
priority = 1
rho = 1.6
""", lambda x, y: np.hypot(x - 0.7, y - 0.7) < 0.17),
]


@functools.cache
def layers_2d(mallard):
    """Density of each initial state of LAYERS (each adding its layer to the ones before) on the triangle mesh."""
    work = Path(tempfile.mkdtemp())
    mesh = work / "square.msh"
    triangle_mesh(mesh)
    r = np.linspace(0.0, 0.8, 9)
    (work / "profile.csv").write_text("r,rho\n" + "".join(f"{a:.3f},{1.0 - 0.6 * np.exp(-(a / 0.25) ** 2):.6f}\n"
                                                         for a in r))
    out, regions = [], ""
    for i, (_, layer, _) in enumerate(LAYERS):
        regions += layer
        toml = f"""[run]
n_steps = 1
dt = 1e-9
[mesh]
type = "file"
filename = "{mesh.name}"
[initialize]
type = "analytical"
rho = "profile_rho"
u = ["0", "0"]
p = "1"
[[initialize.data]]
name = "profile"
file = "profile.csv"
interpolation = "cubic"
at = ["hypot(x - 0.3, y - 0.35)"]
{regions}
[[boundaries]]
name = "wall"
type = "symmetry"
[output]
check_interval = 1000000
[[write_data]]
prefix = "layers{i}"
format = "vtu"
interval = 1
fields = ["RHO"]
{PHYSICS}"""
        (work / f"layers{i}.toml").write_text(toml)
        subprocess.run([str(mallard), "-i", f"layers{i}.toml", "--kokkos-num-threads=2"], cwd=work, check=True,
                       stdout=subprocess.DEVNULL)
        pts, conn, offs, _, data = read_vtu_cells(str(work / f"layers{i}_000000.vtu"))
        starts = np.concatenate([[0], offs[:-1]])
        out.append((pts, [conn[a:b] for a, b in zip(starts, offs)], data["RHO"]))
    return out


def initial_layers(P, args):
    data = layers_2d(args.mallard)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), gridspec_kw=dict(wspace=0.06))
    norm = plt.Normalize(0.0, 1.7)
    cmap = plt.get_cmap(P["cmap"])
    for k, (ax, (title, _, _), (pts, cells, rho)) in enumerate(zip(axes, LAYERS, data)):
        ax.add_collection(PolyCollection([pts[c, :2] for c in cells], facecolors=cmap(norm(rho)),
                                         edgecolors=P["mesh"], linewidths=0.4))
        g = np.linspace(0, 1, 300)
        X, Y = np.meshgrid(g, g)
        for _, _, inside in LAYERS[1:k + 1]:
            ax.contour(X, Y, inside(X, Y).astype(float), levels=[0.5], colors=[P["fg"]], linewidths=1.2)
        bare(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("cell-average density")
    cb.outline.set_edgecolor(P["muted"])
    return fig


HOLE = (0.3, 0.5, 0.12)
MESH_REGIONS = [
    ('"domain" minus a disk', '{type = "difference", regions = ["domain", {type = "disk", center = [0.62, 0.55], '
     'radius = 0.24}]}', lambda x, y: np.hypot(x - 0.62, y - 0.55) >= 0.24),
    ('"left_half" and inside a circle',
     '{type = "intersection", regions = ["left_half", {type = "interior", of = {type = "circle", '
     'center = [0.45, 0.5], radius = 0.33}}]}', lambda x, y: np.hypot(x - 0.45, y - 0.5) < 0.33),
    ('inside the zone "outer"', '{type = "interior", of = "outer"}', None),
]


@functools.cache
def data_mesh_regions(mallard):
    work = Path(tempfile.mkdtemp())
    mesh = work / "holed.msh"
    _, edges = holed_mesh(mesh, hole=HOLE)
    out = {title: fractions(mallard, mesh.name, region, work, f"m{i}", 2, zones=("outer", "hole"))
           for i, (title, region, _) in enumerate(MESH_REGIONS)}
    return out, edges


def mesh_regions(P, args):
    """Regions made with the mesh's own: the domain, a cell zone and the interior of a boundary zone."""
    data, edges = data_mesh_regions(args.mallard)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), gridspec_kw=dict(wspace=0.08))
    for (title, _, inside), ax in zip(MESH_REGIONS, axes.flat):
        panel(ax, P, *data[title], inside, title)
    hx, hy, hr = HOLE
    for e in edges:
        hole = np.hypot(e[:, 0].mean() - hx, e[:, 1].mean() - hy) < hr + 0.05
        axes[-1].plot(e[:, 0], e[:, 1], color=P["flame"] if not hole else P["blue"], lw=2.2, solid_capstyle="round")
    label(axes[-1], 0.5, -0.06, '"outer"', P, color=P["flame"], size=10, va="top")
    label(axes[-1], hx, hy, '"hole"', P, color=P["blue"], size=10)
    axes[1].axvline(0.5, color=P["muted"], lw=1.0, ls="--")
    colorbar(fig, axes, P)
    return fig


ORIENTED = [
    ("circle (exact)", '{type = "circle", center = [0.5, 0.5], radius = 0.3}'),
    ("rectangle minus a disk, boundary (extracted)",
     '{type = "boundary", of = {type = "difference", regions = [{type = "rectangle", center = [0.5, 0.5], '
     'size = [0.8, 0.6]}, {type = "disk", center = [0.75, 0.5], radius = 0.22}]}}'),
    ('"outer" in a box; a flipped ellipse',
     '{type = "intersection", regions = ["outer", {type = "rectangle", center = [0, 0.5], size = [0.6, 2]}]}'),
]


@functools.cache
def data_orientation(mallard):
    """The case view of a check whose probes are the oriented curves: per curve, segments and normals."""
    import h5py
    work = Path(tempfile.mkdtemp())
    mesh = work / "holed.msh"
    holed_mesh(mesh, hole=HOLE)
    regions = [r for _, r in ORIENTED] + ['{type = "ellipse", center = [0.62, 0.62], semi_axes = [0.2, 0.1], '
                                          'rotate = 30, pivot = [0.62, 0.62], flip = true}']
    probes = "".join(f'[[probes]]\nname = "c{i}"\nfields = ["P"]\nregion = {r}\n' for i, r in enumerate(regions))
    toml = f"""[run]
n_steps = 1
dt = 1e-9
[mesh]
type = "file"
filename = "{mesh.name}"
[initialize]
type = "constant"
u = [0.0, 0.0]
p = 1.0
T = 1.0
[[boundaries]]
name = "outer"
type = "symmetry"
[[boundaries]]
name = "hole"
type = "symmetry"
{probes}{PHYSICS}"""
    (work / "case.toml").write_text(toml)
    subprocess.run([str(mallard), "check", "case.toml", "--view", "view", "--kokkos-num-threads=2"], cwd=work,
                   check=True, stdout=subprocess.DEVNULL)
    out = {}
    with h5py.File(work / "view" / "case.vtkhdf") as f:
        for i in range(len(regions)):
            b = f["VTKHDF"][f'probes[name = "c{i}"].region']
            pts, conn, offs = b["Points"][:, :2], b["Connectivity"][:], b["Offsets"][:]
            segs = np.array([pts[conn[offs[k]:offs[k + 1]]] for k in range(len(offs) - 1)])
            out[i] = (segs, b["CellData"]["normal"][:, :2], b["CellData"]["length"][:])
    pts, cells, _ = fractions(mallard, mesh.name, '"domain"', work, "dom", 2, zones=("outer", "hole"))
    return out, pts, cells


def orientation(P, args):
    """Curves with their normals, as mallard check --view writes them."""
    data, pts, cells = data_orientation(args.mallard)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), gridspec_kw=dict(wspace=0.08))
    polys = [pts[c, :2] for c in cells]
    for k, ax in enumerate(axes):
        ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors=P["mesh"], linewidths=0.4))
        for i, color in ((k, P["blue"]), (3, P["flame"])) if k == 2 else ((k, P["blue"]),):
            segs, normals, lengths = data[i]
            for s in segs:
                ax.plot(s[:, 0], s[:, 1], color=color, lw=1.8, solid_capstyle="round")
            mid = segs.mean(axis=1)
            step = max(1, len(mid) // 28)
            ax.quiver(mid[::step, 0], mid[::step, 1], normals[::step, 0], normals[::step, 1], color=color,
                      angles="xy", scale_units="xy", scale=12.0, width=0.006)
        bare(ax)
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.12, 1.12)
        ax.set_title(ORIENTED[k][0], fontsize=11)
    return fig


# ------------------------------------------------- measures in the domain

R_IN, R_OUT = 1.0, 1.384
WALL_DISK = (1.384, 0.0, 0.3)


def annulus_mesh(path, n_r=3, n_t=32, seed=5):
    """Gmsh 2.2 file of the annulus R_IN <= r <= R_OUT: n_r rings of n_t quadrilaterals with jittered interior
    nodes, boundary zones "inner" and "outer" (straight lines, bent onto the circles by [[mesh.curved]])."""
    rng = np.random.default_rng(seed)
    nodes, idx = [], {}
    for j in range(n_r + 1):
        for i in range(n_t):
            jr, jt = rng.uniform(-0.15, 0.15, 2) if 0 < j < n_r else (0.0, 0.0)
            r = R_IN + (R_OUT - R_IN) * (j + jr) / n_r
            nodes.append((r * np.cos(2 * np.pi * (i + jt) / n_t), r * np.sin(2 * np.pi * (i + jt) / n_t)))
            idx[i, j] = len(nodes)
    c = lambda i, j: idx[i % n_t, j]  # noqa: E731
    elements = [f"1 2 {z} {z} {c(i, j)} {c(i + 1, j)}" for i in range(n_t) for z, j in ((1, 0), (2, n_r))]
    elements += [f"3 2 0 3 {c(i, j)} {c(i + 1, j)} {c(i + 1, j + 1)} {c(i, j + 1)}"
                 for j in range(n_r) for i in range(n_t)]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "2", '1 1 "inner"', '1 2 "outer"',
             "$EndPhysicalNames", "$Nodes", str(len(nodes))]
    lines += [f"{k + 1} {x:.17g} {y:.17g} 0" for k, (x, y) in enumerate(nodes)]
    lines += ["$EndNodes", "$Elements", str(len(elements))] + [f"{k + 1} {e}" for k, e in enumerate(elements)]
    path.write_text("\n".join(lines + ["$EndElements"]) + "\n")


def measured_run(binary, work, toml, view=False):
    """Run the case (and with view its check's case view); the cells, each one's SOURCE_RHOE (the fraction of the
    region of [[sources]]), the probe sampling block of disk.csv, the last row of surface.csv if any, and the
    warnings printed."""
    (work / "case.toml").write_text(toml)
    run = subprocess.run([str(binary), "-i", "case.toml", "--kokkos-num-threads=2"], cwd=work, check=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if view:
        subprocess.run([str(binary), "check", "case.toml", "--view", "view", "--kokkos-num-threads=2"], cwd=work,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pts, conn, offs, _, data = read_vtu_cells(str(work / "f_000000.vtu"))
    cells = [conn[a:b] for a, b in zip(np.concatenate([[0], offs[:-1]]), offs)]
    surface = None
    if (work / "surface.csv").exists():
        rows = [r for r in (work / "surface.csv").read_text().splitlines() if r and not r.startswith(("#", "step"))]
        header = (work / "surface.csv").read_text().splitlines()[0].split(",")
        surface = dict(zip(header, map(float, rows[-1].split(","))))
    warnings = [w for w in run.stderr.splitlines() if w.startswith("warning:")]
    return pts, cells, data["SOURCE_RHOE"], probe_sampling(str(work / "disk.csv")), surface, warnings


def lens(R, d, rho):
    """Area and perimeter of the part of a disk (radius rho, center at distance d) inside the circle of radius R."""
    alpha = np.arccos((d * d + R * R - rho * rho) / (2 * d * R))
    beta = np.arccos((d * d + rho * rho - R * R) / (2 * d * rho))
    area = R * R * alpha + rho * rho * beta - 0.5 * np.sqrt((-d + R + rho) * (d + R - rho) * (d - R + rho)
                                                            * (d + R + rho))
    return area, 2 * R * alpha + 2 * rho * beta


@functools.cache
def data_curved_wall(mallard):
    """A disk across the outer wall of a coarse curved annulus, in a uniform expansion u = 0.1 (x, y): its cell
    fractions, measures and the net mass flow out of its part in the domain, and the case view of that part's
    surface."""
    import h5py
    work = Path(tempfile.mkdtemp())
    annulus_mesh(work / "annulus.msh")
    cx, cy, rho = WALL_DISK
    curved = "".join(f'[[mesh.curved]]\nzone = "{z}"\ntype = "circle"\ncenter = [0.0, 0.0]\nradius = {r}\n'
                     for z, r in (("inner", R_IN), ("outer", R_OUT)))
    walls = "".join(f'[[boundaries]]\nname = "{z}"\ntype = "symmetry"\n' for z in ("inner", "outer"))
    toml = f"""[run]
n_steps = 1
dt = 1e-9
[mesh]
type = "file"
filename = "annulus.msh"
{curved}[initialize]
type = "analytical"
rho = "1.0"
u = ["0.1 * x", "0.1 * y"]
p = "1 / 1.4"
{walls}[numerics.face_reconstruction]
type = "TENO"
order = 4
troubled_threshold = 1.0
[regions]
disk = {{type = "disk", center = [{cx}, {cy}], radius = {rho}}}
[[sources]]
rhoE = 1
region = "disk"
[[probes]]
name = "disk"
region = "disk"
fields = ["RHO"]
reductions = ["integral"]
reductions_file = "disk.csv"
[[probes]]
name = "surface"
region = {{type = "boundary", of = {{type = "intersection", regions = ["domain", "disk"]}}}}
fields = ["MASS_FLUX"]
sampling = "point"
reductions = ["integral"]
reductions_file = "surface.csv"
[output]
check_interval = 1000000
[[write_data]]
prefix = "f"
format = "vtu"
interval = 1
fields = ["SOURCE_RHOE"]
{PHYSICS}"""
    pts, cells, f, sampling, surface, _ = measured_run(mallard, work, toml, view=True)
    with h5py.File(work / "view" / "case.vtkhdf") as h:
        b = h["VTKHDF"]['probes[name = "surface"].region']
        p, conn, offs = b["Points"][:, :2], b["Connectivity"][:], b["Offsets"][:]
        segs = np.array([p[conn[offs[k]:offs[k + 1]]] for k in range(len(offs) - 1)])
        normals = b["CellData"]["normal"][:, :2]
    return pts, cells, f, sampling, surface, segs, normals


def curved_polygons(pts, cells):
    """Each cell's outline with its faces on the walls bent onto them, as [[mesh.curved]] does."""
    polys = []
    for c in cells:
        q = pts[c, :2]
        out = []
        for a, b in zip(q, np.roll(q, -1, axis=0)):
            ra, rb = np.hypot(*a), np.hypot(*b)
            r = next((w for w in (R_IN, R_OUT) if abs(ra - w) < 1e-9 and abs(rb - w) < 1e-9), None)
            if r is None:
                out.append(a)
                continue
            ta, tb = np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0])
            tb = ta + (tb - ta + np.pi) % (2 * np.pi) - np.pi
            t = np.linspace(ta, tb, 12, endpoint=False)
            out.extend(np.column_stack([r * np.cos(t), r * np.sin(t)]))
        polys.append(np.array(out))
    return polys


def wall_view(ax, P, pts, cells, f=None):
    polys = curved_polygons(pts, cells)
    colors = "none"
    if f is not None:
        colors = plt.get_cmap(P["cmap"])(f)
        colors[f <= 0.0] = (0, 0, 0, 0)
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=P["mesh"], linewidths=0.6))
    t = np.linspace(-0.5, 0.5, 200)
    ax.plot(R_OUT * np.cos(t), R_OUT * np.sin(t), color=P["fg"], lw=2.0)
    bare(ax)
    ax.set_xlim(0.98, 1.78)
    ax.set_ylim(-0.42, 0.42)


def lens_outline(cx, cy, rho):
    """The part of the disk inside r = R_OUT, closed: its arc of the disk, then the wall's arc inside the disk."""
    t = np.linspace(0, 2 * np.pi, 721)
    x, y = cx + rho * np.cos(t), cy + rho * np.sin(t)
    inside = np.hypot(x, y) <= R_OUT
    start = np.argmin(inside) if inside[0] else 0
    x, y, inside = np.roll(x, -start), np.roll(y, -start), np.roll(inside, -start)
    x, y = x[inside], y[inside]
    a, b = np.arctan2(y[-1], x[-1]), np.arctan2(y[0], x[0])
    w = np.linspace(a, b, 100)
    return np.concatenate([x, R_OUT * np.cos(w)]), np.concatenate([y, R_OUT * np.sin(w)])


def region_curved_wall(P, args):
    """A disk across a curved wall: the fraction of each curved cell it covers, its part in the domain, and the
    measures the probe records against the closed form."""
    pts, cells, f, sampling, _, _, _ = data_curved_wall(args.mallard)
    cx, cy, rho = WALL_DISK
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    wall_view(ax, P, pts, cells, f)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(cx + rho * np.cos(t), cy + rho * np.sin(t), color=P["muted"], lw=1.2, ls=(0, (4, 3)))
    ax.plot(*lens_outline(cx, cy, rho), color=P["flame"], lw=2.0)
    label(ax, 1.42, 0.37, "outer wall (mesh.curved)", P, size=10, ha="left", box=True)
    label(ax, 1.63, -0.3, "the disk", P, size=10, color=P["muted"], box=True)
    label(ax, 1.2, -0.3, "its part in the domain", P, size=10, color=P["flame"], box=True)
    area, perimeter = lens(R_OUT, np.hypot(cx, cy), rho)
    m = sampling["measures"]["in_domain"]
    text = (f"in the domain       recorded      closed form\n"
            f"area                {m['area']:.7f}     {area:.7f}\n"
            f"perimeter           {m['perimeter']:.7f}     {perimeter:.7f}\n"
            f"outside: {100 * sampling['clipped']['area_fraction']:.3g}% of its area")
    ax.text(0.0, -0.02, text, fontsize=9, family="monospace", color=P["fg"], va="top", ha="left",
            transform=ax.transAxes)
    sm = plt.cm.ScalarMappable(cmap=P["cmap"], norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("cell volume fraction")
    cb.outline.set_edgecolor(P["muted"])
    return fig


def region_control_surface(P, args):
    """The surface of a disk's part in the domain, as the check's case view writes it (its faces on the wall bent
    onto it): the disk's arc inside the domain and the wall's inside the disk, with the outward normals, and the
    net mass flow out of it."""
    pts, cells, _, sampling, surface, segs, normals = data_curved_wall(args.mallard)
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    wall_view(ax, P, pts, cells)
    mid = segs.mean(axis=1)
    on_wall = np.all(np.abs(np.hypot(segs[:, :, 0], segs[:, :, 1]) - R_OUT) < 1e-9, axis=1)
    for part, color in ((~on_wall, P["blue"]), (on_wall, P["flame"])):
        for s in segs[part]:
            if part is on_wall:
                t = np.arctan2(s[:, 1], s[:, 0])
                t = np.linspace(t[0], t[1], 20)
                s = R_OUT * np.column_stack([np.cos(t), np.sin(t)])
                mid[np.argmin(np.hypot(*(mid - s[10]).T))] = s[10]
            ax.plot(s[:, 0], s[:, 1], color=color, lw=2.2, solid_capstyle="round")
        ax.quiver(mid[part, 0], mid[part, 1], normals[part, 0], normals[part, 1], color=color, angles="xy",
                  scale_units="xy", scale=14.0, width=0.005)
    label(ax, 1.55, 0.3, "the wall\ninside the disk", P, size=10, color=P["flame"], box=True)
    label(ax, 1.17, -0.33, "the disk's edge in the domain", P, size=10, color=P["blue"], box=True)
    area = sampling["measures"]["in_domain"]["area"]
    text = (f"u = 0.1 (x, y), div u = 0.2\n"
            f"net mass outflow    {surface['integral_MASS_FLUX']:.7f} kg/(m s)\n"
            f"0.2 x area inside   {0.2 * area:.7f}\n"
            f"perimeter           {surface['length']:.7f} m")
    ax.text(0.0, -0.02, text, fontsize=9, family="monospace", color=P["fg"], va="top", ha="left",
            transform=ax.transAxes)
    return fig


OUTSIDE_DISK = (0.72, 0.8, 0.32)


@functools.cache
def data_outside(mallard):
    work = Path(tempfile.mkdtemp())
    triangle_mesh(work / "square.msh")
    cx, cy, rho = OUTSIDE_DISK
    toml = f"""[run]
n_steps = 1
dt = 1e-9
[mesh]
type = "file"
filename = "square.msh"
[initialize]
type = "constant"
u = [0.0, 0.0]
p = 1.0
T = 1.0
[[boundaries]]
name = "wall"
type = "symmetry"
[regions]
disk = {{type = "disk", center = [{cx}, {cy}], radius = {rho}}}
[[sources]]
rhoE = 1
region = "disk"
[[probes]]
name = "disk"
region = "disk"
fields = ["RHO"]
reductions = ["mean"]
reductions_file = "disk.csv"
[output]
check_interval = 1000000
[[write_data]]
prefix = "f"
format = "vtu"
interval = 1
fields = ["SOURCE_RHOE"]
{PHYSICS}"""
    pts, cells, f, sampling, _, warnings = measured_run(mallard, work, toml)
    return pts, cells, f, sampling, warnings


def region_outside(P, args):
    """A disk partly outside a square domain: the cells' fractions of its part inside, its part outside shaded,
    and the warning the run prints."""
    pts, cells, f, sampling, warnings = data_outside(args.mallard)
    cx, cy, rho = OUTSIDE_DISK
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    g = np.linspace(-0.1, 1.15, 600)
    X, Y = np.meshgrid(g, g)
    out = (np.hypot(X - cx, Y - cy) < rho) & ((X > 1) | (Y > 1))
    ax.contourf(X, Y, out.astype(float), levels=[0.5, 1.5], colors=[P["faint"]], hatches=["///"])
    colors = plt.get_cmap(P["cmap"])(f)
    colors[f <= 0.0] = (0, 0, 0, 0)
    ax.add_collection(PolyCollection([pts[c, :2] for c in cells], facecolors=colors, edgecolors=P["mesh"],
                                     linewidths=0.5))
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=P["fg"], lw=1.6)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(cx + rho * np.cos(t), cy + rho * np.sin(t), color=P["flame"], lw=1.6)
    share = 100 * sampling["clipped"]["area_fraction"]
    label(ax, 0.98, 1.1, f"outside the domain:\n{share:.3g}% of the disk's area", P, size=10, box=True, ha="left")
    label(ax, 0.3, 0.3, "domain", P, size=11, color=P["muted"])
    bare(ax)
    ax.set_xlim(-0.02, 1.2)
    ax.set_ylim(-0.02, 1.2)
    warning = warnings[0].replace("warning: ", "")
    head, tail = warning.split(": ", 1)
    ax.set_title(f"warning: {head}:\n{tail.replace(', and', ',\nand')}", fontsize=8.5, family="monospace",
                 loc="left")
    return fig


# ------------------------------------------------ measures in the domain, 3D

BALL_R = 1.0
WALL_SPHERE = (1.0, 0.0, 0.0, 0.45)


def ball_mesh(path, shells=(1.0, 0.75, 0.5, 0.25), n_surface=260, seed=2):
    """Gmsh 2.2 file of the ball r <= BALL_R: Delaunay tetrahedra of points on concentric spheres (Fibonacci
    lattices, the inner ones rotated at random), boundary zone "wall" (flat triangles, bent onto the sphere by
    [[mesh.curved]])."""
    rng = np.random.default_rng(seed)
    pts = [np.zeros((1, 3))]
    for s in shells:
        n = max(8, int(n_surface * s * s))
        k = np.arange(n) + 0.5
        z = 1 - 2 * k / n
        t = np.pi * (1 + 5 ** 0.5) * k
        p = np.column_stack([np.sqrt(1 - z * z) * np.cos(t), np.sqrt(1 - z * z) * np.sin(t), z])
        if s < 1.0:
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            p = p @ q
        pts.append(BALL_R * s * p)
    pts = np.vstack(pts)
    tets = Delaunay(pts).simplices
    vol = np.einsum("ij,ij->i", np.cross(pts[tets[:, 1]] - pts[tets[:, 0]], pts[tets[:, 2]] - pts[tets[:, 0]]),
                    pts[tets[:, 3]] - pts[tets[:, 0]])
    tets = tets[np.abs(vol) > 1e-12]
    faces = {}
    for t in tets:
        for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            key = tuple(sorted(t[v] for v in f))
            faces[key] = faces.get(key, 0) + 1
    tris = [f for f, count in faces.items() if count == 1]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "1", '2 1 "wall"', "$EndPhysicalNames",
             "$Nodes", str(len(pts))]
    lines += [f"{i + 1} {x:.17g} {y:.17g} {z:.17g}" for i, (x, y, z) in enumerate(pts)]
    lines += ["$EndNodes", "$Elements", str(len(tris) + len(tets))]
    k = 1
    for f in tris:
        lines.append(f"{k} 2 2 1 1 {f[0] + 1} {f[1] + 1} {f[2] + 1}")
        k += 1
    for t in tets:
        p = pts[t]
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0]) < 0:
            t = [t[0], t[2], t[1], t[3]]
        lines.append(f"{k} 4 2 0 2 {' '.join(str(v + 1) for v in t)}")
        k += 1
    lines.append("$EndElements")
    path.write_text("\n".join(lines) + "\n")


def sphere_lens(R, d, r):
    """Volume and surface area of the part of a sphere (radius r, center at distance d) inside the sphere of
    radius R: two caps, the small sphere's and the wall's."""
    x = (d * d + R * R - r * r) / (2 * d)
    volume = np.pi * (R + r - d) ** 2 * (d * d + 2 * d * r - 3 * r * r + 2 * d * R + 6 * r * R - 3 * R * R) / (12 * d)
    return volume, 2 * np.pi * r * (r - d + x) + 2 * np.pi * R * (R - x)


@functools.cache
def data_curved_wall_3d(mallard3d):
    """A sphere across the curved wall of a coarse tetrahedral ball, in a uniform expansion u = 0.1 (x, y, z):
    its cell fractions, measures and the net mass flow out of its part in the domain, and the case view of that
    part's surface."""
    import h5py
    work = Path(tempfile.mkdtemp())
    ball_mesh(work / "ball.msh")
    cx, cy, cz, rho = WALL_SPHERE
    toml = f"""[run]
n_steps = 1
dt = 1e-9
[mesh]
type = "file"
filename = "ball.msh"
[[mesh.curved]]
zone = "wall"
type = "sphere"
center = [0.0, 0.0, 0.0]
radius = {BALL_R}
filled = false
[initialize]
type = "analytical"
rho = "1.0"
u = ["0.1 * x", "0.1 * y", "0.1 * z"]
p = "1 / 1.4"
[[boundaries]]
name = "wall"
type = "symmetry"
[numerics.face_reconstruction]
type = "TENO"
order = 4
troubled_threshold = 1.0
[regions]
ball = {{type = "sphere", center = [{cx}, {cy}, {cz}], radius = {rho}}}
[[sources]]
rhoE = 1
region = "ball"
[[probes]]
name = "disk"
region = "ball"
fields = ["RHO"]
reductions = ["integral"]
reductions_file = "disk.csv"
[[probes]]
name = "surface"
region = {{type = "boundary", of = {{type = "intersection", regions = ["domain", "ball"]}}}}
fields = ["MASS_FLUX"]
sampling = "point"
reductions = ["integral"]
reductions_file = "surface.csv"
[output]
check_interval = 1000000
[[write_data]]
prefix = "f"
format = "vtu"
interval = 1
fields = ["SOURCE_RHOE"]
{PHYSICS}"""
    pts, cells, f, sampling, surface, _ = measured_run(mallard3d, work, toml, view=True)
    with h5py.File(work / "view" / "case.vtkhdf") as h:
        b = h["VTKHDF"]['probes[name = "surface"].region']
        p, conn, offs = b["Points"][:], b["Connectivity"][:], b["Offsets"][:]
        polys = [p[conn[offs[k]:offs[k + 1]]] for k in range(len(offs) - 1)]
        normals = b["CellData"]["normal"][:]
    return pts, cells, f, sampling, surface, polys, normals


def plane_cut(p, z=0.0):
    """The polygon a tetrahedron's plane z = const cut makes, and for each of its edges whether it lies in a face
    whose vertices are all on the wall."""
    on = np.abs(np.linalg.norm(p, axis=1) - BALL_R) < 1e-9
    points, edges = [], []
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        if (p[i, 2] - z) * (p[j, 2] - z) < 0:
            s = (z - p[i, 2]) / (p[j, 2] - p[i, 2])
            points.append(p[i, :2] + s * (p[j, :2] - p[i, :2]))
            edges.append({i, j})
    middle = np.mean(points, axis=0)
    order = np.argsort([np.arctan2(*(q - middle)[::-1]) for q in points])
    points = [points[k] for k in order]
    edges = [edges[k] for k in order]
    out = []
    for k in range(len(points)):
        a, b = points[k], points[(k + 1) % len(points)]
        face = edges[k] | edges[(k + 1) % len(points)]
        out.append(a)
        if len(face) == 3 and all(on[v] for v in face):
            ta, tb = np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0])
            t = np.linspace(ta, ta + (tb - ta + np.pi) % (2 * np.pi) - np.pi, 10)[1:-1]
            out.extend(np.column_stack([BALL_R * np.cos(t), BALL_R * np.sin(t)]))
    return np.array(out)


def region_curved_wall_3d(P, args):
    """A sphere across the curved wall of a ball: the cells' volume fractions on the plane z = 0 through its
    center, the faces on the wall bent onto it; and the closed surface of the sphere's part in the domain as the
    check's case view writes it, with its outward normals. Below, the measures the probes record against the
    closed form and the net outflow of a uniform expansion."""
    pts, cells, f, sampling, surface, polys, normals = data_curved_wall_3d(args.mallard3d)
    cx, cy, cz, rho = WALL_SPHERE
    cmap = plt.get_cmap(P["cmap"])
    fig = plt.figure(figsize=(10.0, 5.4))

    ax = fig.add_axes((0.02, 0.3, 0.45, 0.62))
    cut, colors = [], []
    for c, fc in zip(cells, f):
        p = pts[c]
        if p[:, 2].min() < 0.0 < p[:, 2].max() and p[:, 0].max() > 0.3:
            cut.append(plane_cut(p))
            colors.append(cmap(fc) if fc > 0 else (0, 0, 0, 0))
    ax.add_collection(PolyCollection(cut, facecolors=colors, edgecolors=P["mesh"], linewidths=0.6))
    t = np.linspace(-0.9, 0.9, 200)
    ax.plot(BALL_R * np.cos(t), BALL_R * np.sin(t), color=P["fg"], lw=2.0)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(cx + rho * np.cos(t), cy + rho * np.sin(t), color=P["muted"], lw=1.2, ls=(0, (4, 3)))
    bare(ax)
    ax.set_xlim(0.4, 1.5)
    ax.set_ylim(-0.55, 0.55)
    ax.set_title("cell volume fractions on z = 0", fontsize=11)
    label(ax, 1.05, 0.52, "wall (mesh.curved sphere)", P, size=9, ha="left", box=True)
    label(ax, 1.3, -0.45, "the sphere", P, size=9, color=P["muted"], box=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("cell volume fraction")
    cb.outline.set_edgecolor(P["muted"])

    ax = fig.add_axes((0.5, 0.22, 0.5, 0.76), projection="3d", computed_zorder=False)
    mid = np.array([q.mean(axis=0) for q in polys])
    middle = np.array([cx, cy, cz])
    on_wall = np.abs(np.linalg.norm(mid, axis=1) - BALL_R) < np.abs(np.linalg.norm(mid - middle, axis=1) - rho)
    for part, color, alpha in ((~on_wall, P["blue"], 0.35), (on_wall, P["flame"], 0.7)):
        shown = [q for q, keep in zip(polys, part) if keep]
        ax.add_collection3d(Poly3DCollection(shown, facecolors=color, edgecolors=color, linewidths=0.1,
                                             alpha=alpha, rasterized=True))
        idx = np.flatnonzero(part)[::max(1, part.sum() // 24)]
        ax.quiver(mid[idx, 0], mid[idx, 1], mid[idx, 2], normals[idx, 0], normals[idx, 1], normals[idx, 2],
                  length=0.12, color=color, linewidths=0.9)
    ax.set_xlim(0.45, 1.25)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(-0.4, 0.4)
    ax.set_box_aspect((1, 1, 1), zoom=1.2)
    ax.view_init(elev=18, azim=-70)
    ax.set_axis_off()
    ax.set_title("its part in the domain: closed surface, normals out", fontsize=11)
    fig.text(0.56, 0.2, "the wall inside the sphere", color=P["flame"], fontsize=10)
    fig.text(0.56, 0.155, "the sphere inside the domain", color=P["blue"], fontsize=10)

    volume, area = sphere_lens(BALL_R, np.hypot(np.hypot(cx, cy), cz), rho)
    m = sampling["measures"]["in_domain"]
    text = (f"in the domain       recorded      closed form\n"
            f"volume              {m['volume']:.7f}     {volume:.7f}\n"
            f"surface area        {m['surface_area']:.7f}     {area:.7f}\n"
            f"outside: {100 * sampling['clipped']['volume_fraction']:.3g}% of its volume\n"
            f"u = 0.1 (x, y, z): net mass outflow {surface['integral_MASS_FLUX']:.7f} kg/s, "
            f"0.3 x volume inside {0.3 * m['volume']:.7f}")
    fig.text(0.03, 0.22, text, fontsize=9, family="monospace", color=P["fg"], va="top", ha="left")
    return fig


class Args(tuple):
    """Hashable options, so the runs are cached between the light and dark figures."""

    def __new__(cls, mallard, mallard3d):
        return super().__new__(cls, (mallard, mallard3d))

    mallard = property(lambda self: self[0])
    mallard3d = property(lambda self: self[1])


FIGURES = {"region_primitives": primitives_2d, "region_booleans": booleans_2d, "region_smooth_edge": smooth_edge,
           "region_mesh_regions": mesh_regions, "region_orientation": orientation, "regions_3d": regions_3d,
           "initial_layers": initial_layers, "region_curved_wall": region_curved_wall,
           "region_control_surface": region_control_surface, "region_outside": region_outside,
           "region_curved_wall_3d": region_curved_wall_3d}


def figures(mallard, mallard3d=None, names=()):
    args = Args(str(Path(mallard).resolve()), str(Path(mallard3d).resolve()) if mallard3d else None)
    for name, draw in FIGURES.items():
        if (names and name not in names) or (name.endswith("3d") and not args.mallard3d):
            continue
        figstyle.save(name.replace("_", "-"), lambda P, d=draw: d(P, args))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="2D Mallard executable")
    ap.add_argument("--mallard3d", help="3D Mallard executable, for the 3D figure")
    ap.add_argument("names", nargs="*")
    ns = ap.parse_args()
    figures(ns.mallard, ns.mallard3d, ns.names)


if __name__ == "__main__":
    main()
