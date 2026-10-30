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
from mallard.vtu import read_vtu_cells  # noqa: E402

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


# ------------------------------------------------------------------- runs

def fractions(binary, mesh, region, work, name, dim):
    """Cells of the mesh and the region's volume fraction in each, from a one-step run."""
    zero = "[0.0, 0.0]" if dim == 2 else "[0.0, 0.0, 0.0]"
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
[[boundaries]]
name = "wall"
type = "symmetry"
[[sources]]
rhoE = 1
region = {region}
[output]
check_interval = 1000000
[[write_data]]
prefix = "{name}"
format = "vtu"
interval = 1
variables = ["SOURCE_RHOE"]
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
    ("ellipse, rotated", '{type = "ellipse", center = [0, 0], semi_axes = [0.4, 0.18], rotate = 30, '
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
variables = ["RHO"]
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


class Args(tuple):
    """Hashable options, so the runs are cached between the light and dark figures."""

    def __new__(cls, mallard, mallard3d):
        return super().__new__(cls, (mallard, mallard3d))

    mallard = property(lambda self: self[0])
    mallard3d = property(lambda self: self[1])


FIGURES = {"region_primitives": primitives_2d, "region_booleans": booleans_2d, "region_smooth_edge": smooth_edge,
           "regions_3d": regions_3d, "initial_layers": initial_layers}


def figures(mallard, mallard3d=None, names=()):
    args = Args(str(Path(mallard).resolve()), str(Path(mallard3d).resolve()) if mallard3d else None)
    for name, draw in FIGURES.items():
        if (names and name not in names) or (name == "regions_3d" and not args.mallard3d):
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
