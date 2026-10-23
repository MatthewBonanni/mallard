"""Figures of the geometry guide (docs/guide/geometry.md).

    python tools/docs/geometry.py [--mallard build/src/mallard] [--mallard3d build3d/src/mallard] [names...]

The schematics (3D primitives, booleans, transforms, the signed distance)
evaluate a NumPy port of the signed distances and transforms of
src/common/region.h. The figures on meshes run Mallard and read back what it
computed: volume fractions as SOURCE_RHOE of a source of 1 per unit volume over
the region, probe samples from the probe CSV files. They need --mallard (2D)
and, for the 3D probes, --mallard3d.
"""
import argparse
import csv
import functools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyArrowPatch, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import figstyle
from figstyle import bare, label

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools")]
from mallard.vtu import read_vtu_cells  # noqa: E402

PHYSICS = '[physics]\ntype = "euler"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n'
MONO = dict(family="monospace")


# ------------------------------------------------------- signed distances (port of region.h, z = 0)

def to_local(x, y, center=(0.0, 0.0), rotate=0.0, pivot=(0.0, 0.0), scale=(1.0, 1.0), translate=(0.0, 0.0)):
    """Local coordinates of world points: world = pivot + R S (local - pivot) + translate."""
    a = np.radians(rotate)
    u, v = x - translate[0] - pivot[0], y - translate[1] - pivot[1]
    u, v = np.cos(a) * u + np.sin(a) * v, -np.sin(a) * u + np.cos(a) * v
    return u / scale[0] + pivot[0] - center[0], v / scale[1] + pivot[1] - center[1]


def disk(x, y, center, radius, **t):
    u, v = to_local(x, y, center, **t)
    return np.hypot(u, v) - radius


def rectangle(x, y, center, size, **t):
    u, v = to_local(x, y, center, **t)
    du, dv = np.abs(u) - size[0] / 2, np.abs(v) - size[1] / 2
    return np.minimum(np.maximum(du, dv), 0) + np.hypot(np.maximum(du, 0), np.maximum(dv, 0))


def transform_points(p, rotate=0.0, pivot=(0.0, 0.0), scale=(1.0, 1.0), translate=(0.0, 0.0)):
    a = np.radians(rotate)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.asarray(pivot) + ((np.asarray(p) - pivot) * scale) @ R.T + translate)


# ------------------------------------------------------------------ Mallard runs

def run(binary, work, name, toml):
    (work / f"{name}.toml").write_text(toml)
    subprocess.run([str(binary), "-i", f"{name}.toml", "--kokkos-num-threads=2"], cwd=work, check=True,
                   stdout=subprocess.DEVNULL)


def fractions(binary, work, name, mesh, boundaries, region, dim=2):
    """Cells and the region's volume fraction in each, from a one-step run."""
    zero = "[0.0, 0.0]" if dim == 2 else "[0.0, 0.0, 0.0]"
    bcs = "".join(f'[[boundaries]]\nname = "{b}"\ntype = "symmetry"\n' for b in boundaries)
    run(binary, work, name, f"""[run]
n_steps = 1
dt = 1e-9
[mesh]
{mesh}
[initialize]
type = "constant"
u = {zero}
p = 1.0
T = 1.0
{bcs}[[sources]]
rhoE = 1
region = {region}
[output]
check_interval = 1000000
[[write_data]]
prefix = "{name}"
format = "vtu"
interval = 1
variables = ["SOURCE_RHOE"]
{PHYSICS}""")
    pts, conn, offs, _, data = read_vtu_cells(str(work / f"{name}_000000.vtu"))
    starts = np.concatenate([[0], offs[:-1]])
    return pts, [conn[a:b] for a, b in zip(starts, offs)], data["SOURCE_RHOE"]


def probe_rows(path):
    with open(path) as f:
        lines = [line for line in f if not line.startswith("#")]
    return [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(lines)]


# ------------------------------------------------------------------ figure helpers

def fraction_cells(ax, P, pts, cells, f, edge=0.4):
    colors = plt.get_cmap(P["cmap"])(np.clip(f, 0, 1))
    colors[f <= 0.0] = (0, 0, 0, 0)
    ax.add_collection(PolyCollection([pts[c, :2] for c in cells], facecolors=colors, edgecolors=P["mesh"],
                                     linewidths=edge))


def fraction_colorbar(fig, axes, P, text="cell volume fraction", shrink=0.8):
    sm = plt.cm.ScalarMappable(cmap=P["cmap"], norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, shrink=shrink, pad=0.02)
    cb.set_label(text)
    cb.outline.set_edgecolor(P["muted"])
    return cb


def outline(ax, X, Y, phi, color, lw=1.5, ls="-"):
    ax.contour(X, Y, phi, levels=[0.0], colors=[color], linewidths=lw, linestyles=ls)


def fill(ax, X, Y, phi, color, alpha=0.55):
    ax.contourf(X, Y, phi, levels=[-1e9, 0.0], colors=[color], alpha=alpha)


def grid(lo, hi, n=300):
    return np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))


# ------------------------------------------------------------------ 3D primitives

def _shade(P, quads, base):
    """Face colors of quads (n, 4, 3) under a fixed light, lit on both sides."""
    n = np.cross(quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 1])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    light = np.array([-0.45, -0.6, 0.85])
    light /= np.linalg.norm(light)
    i = np.abs(n @ light)
    rgb = np.array(to_rgb(base))
    c = rgb[None] * (0.3 + 0.8 * i[:, None])
    c += (1 - c) * 0.4 * i[:, None] ** 16
    return np.clip(c, 0, 1)


def _quads(X, Y, Z):
    p = np.stack([X, Y, Z], axis=-1)
    return np.stack([p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]], axis=2).reshape(-1, 4, 3)


def _revolve(r, z, n=64):
    """Surface of revolution about z of the profile (r, z)."""
    t = np.linspace(0, 2 * np.pi, n + 1)
    R, T = np.meshgrid(r, t, indexing="ij")
    Zg, _ = np.meshgrid(z, t, indexing="ij")
    return _quads(R * np.cos(T), R * np.sin(T), Zg)


def _rotation(axis, deg):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _profile(*segments, n=16):
    r, z = [], []
    for (r0, z0), (r1, z1) in segments:
        s = np.linspace(0, 1, n)
        r += list(r0 + s * (r1 - r0))
        z += list(z0 + s * (z1 - z0))
    return np.array(r), np.array(z)


def primitive_quads():
    """(title, keys, quads) of each 3D primitive, about the origin and within [-1, 1]^3."""
    u = np.linspace(0, np.pi, 40)
    sphere = _revolve(0.9 * np.sin(u), 0.9 * np.cos(u))
    e = _revolve(np.sin(u), np.cos(u)) * [1.0, 0.55, 0.42]
    ellipsoid = e @ _rotation([0, 0, 1], 30).T
    cylinder = _revolve(*_profile(((0, 0.8), (0.55, 0.8)), ((0.55, 0.8), (0.55, -0.8)), ((0.55, -0.8), (0, -0.8))))
    annular = _revolve(*_profile(((0.42, 0.5), (0.8, 0.5)), ((0.8, 0.5), (0.8, -0.5)), ((0.8, -0.5), (0.42, -0.5)),
                                 ((0.42, -0.5), (0.42, 0.5))))
    annular = annular @ _rotation([1, 0, 0], 20).T
    cone = _revolve(*_profile(((0, -0.75), (0.8, -0.75)), ((0.8, -0.75), (0.28, 0.75)), ((0.28, 0.75), (0, 0.75))))
    v = np.linspace(0, 2 * np.pi, 33)
    torus = _revolve(0.62 + 0.27 * np.cos(v), 0.27 * np.sin(v), n=72)
    box = []
    h = np.array([0.8, 0.5, 0.35])
    s = np.linspace(-1, 1, 9)
    A, B = np.meshgrid(s, s, indexing="ij")
    for axis in range(3):
        for sign in (-1, 1):
            c = [None, None, None]
            o = [i for i in range(3) if i != axis]
            c[axis] = np.full_like(A, sign * h[axis])
            c[o[0]], c[o[1]] = A * h[o[0]], B * h[o[1]]
            box.append(_quads(*c))
    box = np.concatenate(box) @ (_rotation([0, 0, 1], 25) @ _rotation([1, 0, 0], 15)).T
    # Superellipsoid x^4 + y^4 + z^4 < 0.8^4
    th, ph = np.linspace(-np.pi / 2, np.pi / 2, 40), np.linspace(-np.pi, np.pi, 72)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    sp = lambda w: np.sign(w) * np.abs(w) ** 0.5  # noqa: E731
    implicit = _quads(0.8 * sp(np.cos(TH)) * sp(np.cos(PH)), 0.8 * sp(np.cos(TH)) * sp(np.sin(PH)), 0.8 * sp(np.sin(TH)))
    return [
        ("sphere", "center, radius", sphere),
        ("ellipsoid", "center, semi_axes", ellipsoid),
        ("cylinder", "axis, radius, length", cylinder),
        ("cylinder, annular", "+ inner_radius", annular),
        ("cone", "base, axis, length,\nbase_radius, top_radius", cone),
        ("box", "center, size\n(rotated)", box),
        ("torus", "center, axis,\nmajor_radius, minor_radius", torus),
        ("half_space", "point, normal", None),
        ("implicit", 'f = "x^4 + y^4 + z^4 - 0.41"', implicit),
    ]


def primitives_3d(P, args):
    shapes = primitive_quads()
    base = "#2f9c8f" if P["name"] == "light" else P["teal"]
    fig = plt.figure(figsize=(11.0, 9.6))
    for k, (title, keys, quads) in enumerate(shapes):
        ax = fig.add_subplot(3, 3, k + 1, projection="3d", computed_zorder=quads is not None)
        if quads is None:
            n = np.array([0.35, -0.3, 1.0])
            n /= np.linalg.norm(n)
            s = np.linspace(-0.95, 0.95, 12)
            A, B = np.meshgrid(s, s, indexing="ij")
            Z = -(n[0] * A + n[1] * B) / n[2] - 0.1
            q = _quads(A, B, Z)
            c = _shade(P, q, base)
            ax.add_collection3d(Poly3DCollection(q, facecolors=c, linewidths=0, alpha=0.75, rasterized=True))
            ax.quiver(0, 0, -0.1, *(0.8 * n), color=P["flame"], lw=2.6, arrow_length_ratio=0.25, zorder=10)
            ax.text(0.15, 0.1, -0.85, "inside: below", color=P["fg"], fontsize=11, ha="center")
            ax.text(*(0.95 * n + [0, 0, -0.1]), "normal", color=P["flame"], fontsize=11)
        else:
            c = _shade(P, quads, base)
            ax.add_collection3d(Poly3DCollection(quads, facecolors=c, edgecolors=c, linewidths=0.3, rasterized=True))
        for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            lim(-0.9, 0.9)
        ax.set_box_aspect((1, 1, 1), zoom=1.25)
        ax.view_init(elev=24, azim=-58)
        ax.set_axis_off()
        ax.set_title(title, fontsize=13, **MONO, pad=-2)
        ax.text2D(0.5, -0.02, keys, transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
                  color=P["muted"], **MONO)
    fig.subplots_adjust(wspace=0.0, hspace=0.32, left=0, right=1, top=0.96, bottom=0.05)
    return fig


# ------------------------------------------------------------------ booleans

A_DISK = dict(center=(0.42, 0.5), radius=0.27)
B_RECT = dict(center=(0.64, 0.5), size=(0.5, 0.26), rotate=25, pivot=(0.64, 0.5))


def csg(P, args):
    X, Y = grid((0.05, 0.12), (0.95, 0.88))
    a = disk(X, Y, **A_DISK)
    b = rectangle(X, Y, **B_RECT)
    panels = [("A", a, '"disk"'), ("B", b, '"rectangle"'), ("A ∪ B", np.minimum(a, b), '"union"'),
              ("A ∩ B", np.maximum(a, b), '"intersection"'), ("A − B", np.maximum(a, -b), '"difference"')]
    fig, axes = plt.subplots(1, 5, figsize=(11.0, 2.75), gridspec_kw=dict(wspace=0.04))
    for ax, (title, phi, kind) in zip(axes, panels):
        fill(ax, X, Y, phi, P["teal"], 0.6)
        outline(ax, X, Y, a, P["muted"], 1.0, "--")
        outline(ax, X, Y, b, P["muted"], 1.0, "--")
        outline(ax, X, Y, phi, P["fg"], 1.7)
        bare(ax)
        ax.set_title(title, fontsize=15)
        label(ax, 0.5, 0.05, kind, P, size=11, color=P["muted"], **MONO)
        if title in ("A", "B"):
            c = A_DISK["center"] if title == "A" else B_RECT["center"]
            label(ax, *c, title, P, size=13, color=P["bg"], weight="bold")
    return fig


# ------------------------------------------------------------------ transforms

FLAG = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.16], [0.16, 0.16], [0.16, 0.42], [0.0, 0.42]]) + [0.15, 0.1]


def transforms(P, args):
    panels = [
        ("translate", "translate = [0.55, 0.3]", dict(translate=(0.55, 0.3)), None),
        ("rotate", "rotate = 35\n(about the origin)", dict(rotate=35), (0.0, 0.0)),
        ("rotate about a pivot", "rotate = 35\npivot = [0.45, 0.3]", dict(rotate=35, pivot=(0.45, 0.3)), (0.45, 0.3)),
        ("scale", "scale = [1.8, 0.6]", dict(scale=(1.8, 0.6)), None),
        ("all, in order", "scale, rotate,\nthen translate",
         dict(scale=(1.4, 0.8), rotate=-30, pivot=(0.45, 0.3), translate=(0.25, 0.35)), (0.45, 0.3)),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(12.0, 3.4), gridspec_kw=dict(wspace=0.06))
    for ax, (title, keys, t, pivot) in zip(axes, panels):
        for v in np.arange(-0.2, 1.21, 0.2):
            ax.plot([v, v], [-0.08, 1.05], color=P["faint"], lw=0.8, zorder=0)
        for v in np.arange(0.0, 1.01, 0.2):
            ax.plot([-0.3, 1.25], [v, v], color=P["faint"], lw=0.8, zorder=0)
        ax.annotate("", (0.3, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color=P["muted"], lw=1.2))
        ax.annotate("", (0, 0.3), (0, 0), arrowprops=dict(arrowstyle="-|>", color=P["muted"], lw=1.2))
        ax.add_patch(Polygon(FLAG, closed=True, facecolor="none", edgecolor=P["muted"], lw=1.3, ls="--"))
        moved = transform_points(FLAG, **t)
        ax.add_patch(Polygon(moved, closed=True, facecolor=P["teal"], alpha=0.55, edgecolor="none"))
        ax.add_patch(Polygon(moved, closed=True, facecolor="none", edgecolor=P["fg"], lw=1.6))
        ax.plot(*FLAG[0], "o", color=P["muted"], ms=4)
        ax.plot(*moved[0], "o", color=P["fg"], ms=4)
        ax.add_patch(FancyArrowPatch(FLAG[0], moved[0], arrowstyle="-|>", mutation_scale=11, color=P["flame"],
                                     lw=1.3, connectionstyle="arc3,rad=0.25" if "rotate" in t else "arc3"))
        if pivot is not None:
            ax.plot(*pivot, marker="+", color=P["flame"], ms=13, mew=2.2)
        bare(ax)
        ax.set_xlim(-0.3, 1.25)
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(title, fontsize=13)
        label(ax, 0.47, -0.16, keys, P, size=10, color=P["muted"], va="top", linespacing=1.3, **MONO)
    return fig


# ------------------------------------------------------------------ signed distance

def signed_distance(P, args):
    X, Y = grid((0.0, 0.0), (1.0, 1.0), 400)
    a = disk(X, Y, center=(0.5, 0.5), radius=0.36)
    b = rectangle(X, Y, center=(0.5, 0.5), size=(0.52, 0.16), rotate=40, pivot=(0.5, 0.5))
    phi = np.maximum(a, -b)
    edge = 0.12
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw=dict(wspace=0.32))
    lim = 0.3
    cmap = LinearSegmentedColormap.from_list("sd", [P["teal"], P["bg"], P["flame"]])
    cs = axes[0].contourf(X, Y, np.clip(phi, -lim, lim), levels=np.linspace(-lim, lim, 25), cmap=cmap)
    cs.set_rasterized(True)
    axes[0].contour(X, Y, phi, levels=np.arange(-0.25, 0.31, 0.05), colors=[P["muted"]], linewidths=0.6)
    outline(axes[0], X, Y, phi, P["fg"], 2.0)
    axes[0].set_title("signed distance φ (difference)", fontsize=12)
    cb = fig.colorbar(cs, ax=axes[0], shrink=0.85, pad=0.03, ticks=[-0.3, -0.15, 0, 0.15, 0.3])
    cb.set_label("φ  (negative inside)")
    cb.outline.set_edgecolor(P["muted"])
    s = np.clip(0.5 - phi / edge, 0, 1)
    w = s * s * (3 - 2 * s)
    im = axes[1].imshow(w, extent=(0, 1, 0, 1), origin="lower", cmap=P["cmap"], vmin=0, vmax=1, rasterized=True)
    outline(axes[1], X, Y, phi + edge / 2, "#1f1a1d", 0.9, "--")
    outline(axes[1], X, Y, phi - edge / 2, "#ffffff", 0.9, "--")
    axes[1].set_title(f"indicator with edge = {edge}", fontsize=12)
    cb = fig.colorbar(im, ax=axes[1], shrink=0.85, pad=0.03)
    cb.set_label("indicator")
    cb.outline.set_edgecolor(P["muted"])
    for ax in axes:
        bare(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    return fig


# ------------------------------------------------------------------ adaptive subdivision

def subdivide(tri, phi, max_level, level=0, out=None):
    """Leaves of region.cpp's Sampler on a triangle (sharp, bounded region): (vertices, level, indicator)."""
    out = [] if out is None else out
    c = tri.mean(axis=0)
    radius = np.max(np.linalg.norm(tri - c, axis=1))
    d = phi(*c)
    if level >= max_level or abs(d) >= radius:
        out.append((tri, level, float(d < 0)))
        return out
    m01, m12, m02 = (tri[0] + tri[1]) / 2, (tri[1] + tri[2]) / 2, (tri[0] + tri[2]) / 2
    for child in ((tri[0], m01, m02), (m01, tri[1], m12), (m02, m12, tri[2]), (m01, m12, m02)):
        subdivide(np.array(child), phi, max_level, level + 1, out)
    return out


def subdivision(P, args):
    phi = lambda x, y: disk(x, y, (0.55, 0.62), 0.5)  # noqa: E731
    cells = [np.array(t) for t in (((0, 0), (1, 0), (0.45, 0.85)), ((1, 0), (1.25, 0.95), (0.45, 0.85)),
                                   ((0.45, 0.85), (1.25, 0.95), (0.6, 1.55)))]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), gridspec_kw=dict(wspace=0.05))
    X, Y = grid((-0.1, -0.25), (1.35, 1.65))
    for ax, depth in zip(axes, (0, 2, 6)):
        total = []
        for cell in cells:
            leaves = subdivide(cell, phi, depth)
            polys = [t for t, _, _ in leaves]
            inside = np.array([w for _, _, w in leaves])
            colors = [P["teal"] if w else P["panel"] for w in inside]
            ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=P["mesh"], linewidths=0.5,
                                             alpha=0.9))
            area = np.array([0.5 * abs((t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1]) - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0]))
                             for t in polys])
            total.append((area * inside).sum() / area.sum())
            ax.add_patch(Polygon(cell, closed=True, facecolor="none", edgecolor=P["fg"], lw=1.6))
        outline(ax, X, Y, phi(X, Y), P["flame"], 1.8)
        bare(ax)
        ax.set_xlim(-0.1, 1.35)
        ax.set_ylim(-0.25, 1.65)
        title = "no subdivision" if depth == 0 else f"{depth} levels"
        ax.set_title(title, fontsize=12)
        label(ax, 0.62, -0.1, "fractions " + ", ".join(f"{f:.2f}" for f in total), P, size=10.5,
              color=P["muted"], va="top")
    return fig


# ------------------------------------------------------------------ regions on meshes

ROT_BOX = '{type = "rectangle", center = [0.5, 0.5], size = [0.66, 0.28], rotate = 30, pivot = [0.5, 0.5]'
CART = 'type = "cartesian"\nNx = 18\nNy = 18\nLx = 1.0\nLy = 1.0'
CART_BCS = ("left", "right", "bottom", "top")


@functools.cache
def rotated_box_data(mallard):
    work = Path(tempfile.mkdtemp())
    return (fractions(mallard, work, "sharp", CART, CART_BCS, ROT_BOX + "}"),
            fractions(mallard, work, "smooth", CART, CART_BCS, ROT_BOX + ", edge = 0.1}"))


def rotated_box(P, args):
    sharp, smooth = rotated_box_data(args.mallard)
    X, Y = grid((0, 0), (1, 1))
    phi = rectangle(X, Y, center=(0.5, 0.5), size=(0.66, 0.28), rotate=30, pivot=(0.5, 0.5))
    pts, cells, f = sharp
    cx = np.array([pts[c, 0].mean() for c in cells])
    cy = np.array([pts[c, 1].mean() for c in cells])
    inside = (rectangle(cx, cy, center=(0.5, 0.5), size=(0.66, 0.28), rotate=30, pivot=(0.5, 0.5)) < 0).astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9), gridspec_kw=dict(wspace=0.05))
    titles = ("centroid inside (a staircase)", "volume fraction (Mallard)", "volume fraction, edge = 0.1")
    for ax, (p, c, v), title in zip(axes, ((pts, cells, inside), sharp, smooth), titles):
        fraction_cells(ax, P, p, c, v)
        outline(ax, X, Y, phi, P["fg"], 1.6)
        bare(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=12)
    fraction_colorbar(fig, axes, P, shrink=0.75)
    exact = 0.66 * 0.28
    area = 1.0 / 18 ** 2
    for ax, v in zip(axes[:2], (inside, f)):
        label(ax, 0.5, -0.07, f"area {v.sum() * area:.4f}  (exact {exact:.4f})", P, size=10.5, color=P["muted"])
    return fig


MESH_REGION = ('{type = "union", regions = ['
               '{type = "annulus", center = [0, 0], radius = 0.85, inner_radius = 0.5}, '
               '{type = "rectangle", center = [2.4, 0], size = [3.4, 0.9], rotate = 12, pivot = [0.7, 0]}]}')


@functools.cache
def mesh_region_data(mallard):
    work = Path(tempfile.mkdtemp())
    subprocess.run([sys.executable, str(ROOT / "tools" / "mesh" / "make_cylinder_mesh.py"), str(work / "cyl.msh"),
                    "--n-theta", "96", "--n-r", "40", "--r-far", "10", "--dr0", "0.015"], check=True,
                   stdout=subprocess.DEVNULL)
    return fractions(mallard, work, "cyl", 'type = "file"\nfilename = "cyl.msh"', ("cylinder", "farfield"),
                     MESH_REGION)


def mesh_region(P, args):
    pts, cells, f = mesh_region_data(args.mallard)
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    keep = [k for k, c in enumerate(cells) if pts[c, 0].max() > -1.4 and pts[c, 0].min() < 4.6
            and np.abs(pts[c, 1]).min() < 1.8]
    ax.add_collection(PolyCollection([pts[cells[k], :2] for k in keep], facecolors="none", edgecolors=P["mesh"],
                                     linewidths=0.4, rasterized=True))
    sel = [k for k in keep if f[k] > 0]
    fraction_cells(ax, P, pts, [cells[k] for k in sel], f[sel], edge=0.4)
    X, Y = grid((-1.4, -1.8), (4.6, 1.8), 600)
    phi = np.minimum(np.maximum(disk(X, Y, (0, 0), 0.85), -disk(X, Y, (0, 0), 0.5)),
                     rectangle(X, Y, center=(2.4, 0), size=(3.4, 0.9), rotate=12, pivot=(0.7, 0)))
    outline(ax, X, Y, phi, P["fg"], 1.4)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.fill(0.5 * np.cos(t), 0.5 * np.sin(t), color=P["panel"], ec=P["fg"], lw=1.6)
    bare(ax)
    ax.set_xlim(-1.4, 4.6)
    ax.set_ylim(-1.8, 1.8)
    fraction_colorbar(fig, ax, P, shrink=0.9)
    return fig


# ------------------------------------------------------------------ probes

PROBES_2D = [
    ("point", 'point = [0.12, 0.88]', (0.12, 0.96)),
    ("points", 'points = [[0.22, 0.6], [0.3, 0.72], [0.4, 0.58], [0.34, 0.5], [0.46, 0.7]]', (0.34, 0.8)),
    ("line", 'start = [0.08, 0.08]\nend = [0.7, 0.36]\nn_points = 12', (0.3, 0.33)),
    ("polyline", 'vertices = [[0.85, 0.08], [1.02, 0.44], [1.2, 0.12]]\nn_points = 6', (1.25, 0.42)),
    ("arc", 'center = [0.82, 0.62]\nradius = 0.24\nangles = [15, 165]\nn_points = 13', (0.82, 0.93)),
    ("circle", 'center = [1.62, 0.7]\nradius = 0.18\nn_points = 16', (1.62, 0.96)),
    ("curve", 'x = "1.38 + 0.6 * s"\ny = "0.22 + 0.12 * sin(4 * pi * s)"\nn_points = 25', (1.68, 0.44)),
    ("circle, transformed", 'center = [0, 0]\nradius = 0.16\nn_points = 16\nscale = [1.5, 0.6]\nrotate = 30\n'
     'translate = [1.18, 0.74]', (1.2, 0.52)),
]

PROBES_3D = [
    ("circle", 'center = [0.38, 0.4, 0.5]\nradius = 0.27\nnormal = [0.6, -1, 0.9]\nn_points = 24', (0.3, 0.3, 0.1)),
    ("curve", 'x = "0.75 + 0.15 * cos(4 * pi * s)"\ny = "0.75 + 0.15 * sin(4 * pi * s)"\nz = "0.08 + 0.84 * s"\n'
     'n_points = 49', (0.8, 0.8, 1.0)),
    ("line", 'start = [0.05, 0.1, 0.9]\nend = [0.95, 0.1, 0.9]\nn_points = 10', (0.1, 0.1, 0.98)),
]


def probe_inputs(probes, dim):
    out = ""
    for k, entry in enumerate(probes):
        name, keys = entry[0], entry[1]
        shape = name.split(",")[0]
        out += (f'[[probes]]\nname = "p{k}"\nvariables = ["RHO"]\nshape = "{shape}"\n{keys}\n'
                f'file = "p{k}.csv"\n')
    return out


@functools.cache
def probes_data(binary, dim):
    work = Path(tempfile.mkdtemp())
    probes = PROBES_2D if dim == 2 else PROBES_3D
    if dim == 2:
        mesh, u = 'type = "cartesian"\nNx = 4\nNy = 2\nLx = 2.0\nLy = 1.0', "[0.0, 0.0]"
        bcs = CART_BCS
    else:
        mesh, u = 'type = "cartesian"\nNx = 3\nNy = 3\nNz = 3\nLx = 1.0\nLy = 1.0\nLz = 1.0', "[0.0, 0.0, 0.0]"
        bcs = CART_BCS + ("front", "back")
    bc = "".join(f'[[boundaries]]\nname = "{b}"\ntype = "symmetry"\n' for b in bcs)
    run(binary, work, "probes", f"""[run]
n_steps = 1
dt = 1e-9
[mesh]
{mesh}
[initialize]
type = "constant"
u = {u}
p = 1.0
T = 1.0
{bc}[output]
check_interval = 1000000
{probe_inputs(probes, dim)}{PHYSICS}""")
    out = []
    for k in range(len(probes)):
        rows = [r for r in probe_rows(work / f"p{k}.csv") if r["step"] == 0]
        out.append(np.array([[r[c] for c in ("x", "y", "z")[:dim]] for r in rows]))
    return out


def probes_2d(P, args):
    data = probes_data(args.mallard, 2)
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.add_patch(Polygon([(0, 0), (2, 0), (2, 1), (0, 1)], closed=True, facecolor=P["panel"], edgecolor=P["muted"],
                         lw=1.0))
    colors = [P["flame"], P["plum"], P["blue"], P["green"], P["teal"], P["red"], P["gold"], P["line"]]
    for (name, _, at), x, color in zip(PROBES_2D, data, colors):
        if name not in ("point", "points"):
            closed = name.startswith("circle")
            path = np.vstack([x, x[:1]]) if closed else x
            ax.plot(path[:, 0], path[:, 1], color=color, lw=1.4, alpha=0.8)
        ax.plot(x[:, 0], x[:, 1], "o", color=color, ms=5.5, mec=P["bg"], mew=0.8)
        ax.plot(*x[0], "o", color=color, ms=9, mfc="none", mew=1.4)
        label(ax, *at, f'"{name.split(",")[0]}"' + (" + transforms" if "," in name else ""), P, size=11,
              color=color, **MONO)
    bare(ax)
    ax.set_xlim(-0.03, 2.03)
    ax.set_ylim(-0.03, 1.03)
    return fig


def probes_3d(P, args):
    data = probes_data(args.mallard3d, 3)
    fig = plt.figure(figsize=(6.0, 5.2))
    ax = fig.add_subplot(projection="3d")
    for a in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)):
        for d in range(3):
            if a[d] == 0:
                b = list(a)
                b[d] = 1
                ax.plot(*zip(a, b), color=P["muted"], lw=0.8)
    colors = [P["teal"], P["flame"], P["blue"]]
    for (name, _, at), x, color in zip(PROBES_3D, data, colors):
        path = np.vstack([x, x[:1]]) if name == "circle" else x
        ax.plot(*path.T, color=color, lw=1.3, alpha=0.8)
        ax.scatter(*x.T, color=color, s=16, depthshade=False)
        ax.text(*at, f'"{name}"', color=color, fontsize=11, ha="center", **MONO)
    n = np.array([0.6, -1, 0.9])
    n /= np.linalg.norm(n)
    c = np.array([0.38, 0.4, 0.5])
    ax.quiver(*c, *(0.3 * n), color=P["teal"], lw=1.6, arrow_length_ratio=0.25)
    ax.text(*(c + 0.36 * n), "normal", color=P["teal"], fontsize=10)
    for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        lim(0, 1)
    ax.set_box_aspect((1, 1, 1), zoom=1.1)
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()
    return fig


# ------------------------------------------------------------------ sampling convergence

SINE_RHO = "1.0 + 0.2 * sin(2 * pi * x) * sin(2 * pi * y)"
SCHEMES = [
    ("cell", 'type = "TENO"\norder = 5\ntroubled_threshold = 1.0', "cell", "cell average (sampling = \"cell\")"),
    ("muscl", 'type = "MUSCL"\nlimiter = "none"', "point", "point, MUSCL (linear least squares)"),
    ("teno3", 'type = "TENO"\norder = 3\ntroubled_threshold = 1.0', "point", "point, TENO order 3"),
    ("teno5", 'type = "TENO"\norder = 5\ntroubled_threshold = 1.0', "point", "point, TENO order 5"),
]
NS = (10, 20, 40, 80)


def sample_points(n):
    """Points at the same place within their cells on every mesh, off the centroids (as in probes_test.cpp)."""
    near = ((0.137, 0.462), (0.713, 0.291), (0.402, 0.858))
    within = (0.3, 0.7)
    return [[(np.floor(p[d] * n) + within[d]) / n for d in range(2)] for p in near]


@functools.cache
def convergence_data(mallard):
    work = Path(tempfile.mkdtemp())
    out = {}
    for key, recon, sampling, _ in SCHEMES:
        errors = []
        for n in NS:
            name = f"{key}_{n}"
            pts = ", ".join(f"[{float(x)!r}, {float(y)!r}]" for x, y in sample_points(n))
            run(mallard, work, name, f"""[run]
n_steps = 1
dt = 1e-6
[mesh]
type = "cartesian"
Nx = {n}
Ny = {n}
Lx = 1.0
Ly = 1.0
periodic = ["x", "y"]
[initialize]
type = "analytical"
rho = "{SINE_RHO}"
u = ["0.0", "0.0"]
p = "1.0"
[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"
[numerics.face_reconstruction]
{recon}
[output]
check_interval = 1000000
[[probes]]
name = "p"
file = "{name}.csv"
variables = ["RHO"]
sampling = "{sampling}"
shape = "points"
points = [{pts}]
{PHYSICS}""")
            rows = [r for r in probe_rows(work / f"{name}.csv") if r["step"] == 0]
            exact = [1 + 0.2 * np.sin(2 * np.pi * r["x"]) * np.sin(2 * np.pi * r["y"]) for r in rows]
            errors.append(max(abs(r["RHO"] - e) for r, e in zip(rows, exact)))
        out[key] = np.array(errors)
    return out


def sampling_convergence(P, args):
    data = convergence_data(args.mallard)
    h = 1.0 / np.array(NS)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = {"cell": P["red"], "muscl": P["gold"], "teno3": P["blue"], "teno5": P["teal"]}
    for key, _, _, text in SCHEMES:
        e = data[key]
        rate = np.log2(e[-2] / e[-1])
        ax.loglog(h, e, "o-", color=colors[key], label=f"{text}: order {rate:.1f}", ms=6)
    for order, key in ((1, "cell"), (2, "muscl"), (3, "teno3"), (5, "teno5")):
        e = data[key]
        ref = e[0] * 1.8 * (h / h[0]) ** order
        ax.loglog(h, ref, ls=":", color=P["muted"], lw=1.0)
        ax.text(h[-1] * 0.93, ref[-1], f"{order}", color=P["muted"], fontsize=10, ha="right", va="center")
    ax.set_xticks(h, [f"1/{n}" for n in NS])
    ax.minorticks_off()
    ax.set_xlabel("cell size h (sine density on a periodic Cartesian mesh)")
    ax.set_ylabel("max error of RHO at the samples")
    ax.invert_xaxis()
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, which="major", color=P["faint"], lw=0.7)
    return fig


# ------------------------------------------------------------------ main

class Args(tuple):
    """Hashable options, so the runs are cached between the light and dark figures."""

    def __new__(cls, mallard, mallard3d):
        return super().__new__(cls, (mallard, mallard3d))

    mallard = property(lambda self: self[0])
    mallard3d = property(lambda self: self[1])


# name -> (draw, needs): needs is None, "2d" or "3d"
FIGURES = {
    "geometry-primitives-3d": (primitives_3d, None),
    "geometry-csg": (csg, None),
    "geometry-transforms": (transforms, None),
    "geometry-signed-distance": (signed_distance, None),
    "geometry-subdivision": (subdivision, None),
    "geometry-rotated-box": (rotated_box, "2d"),
    "geometry-mesh-region": (mesh_region, "2d"),
    "geometry-probes": (probes_2d, "2d"),
    "geometry-probes-3d": (probes_3d, "3d"),
    "geometry-sampling": (sampling_convergence, "2d"),
}


def figures(mallard=None, mallard3d=None, names=()):
    args = Args(str(Path(mallard).resolve()) if mallard else None, str(Path(mallard3d).resolve()) if mallard3d else None)
    for name, (draw, needs) in FIGURES.items():
        if names and name not in names:
            continue
        if (needs == "2d" and not args.mallard) or (needs == "3d" and not args.mallard3d):
            continue
        figstyle.save(name, lambda P, d=draw: d(P, args))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", help="2D Mallard executable, for the figures on meshes")
    ap.add_argument("--mallard3d", help="3D Mallard executable, for the 3D probes")
    ap.add_argument("names", nargs="*")
    ns = ap.parse_args()
    figures(ns.mallard, ns.mallard3d, ns.names)


if __name__ == "__main__":
    main()
