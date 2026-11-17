"""Figures of the local coordinates of boundary faces (docs/guide/boundary_conditions.md):
one face of a pipe's inlet in its 3D mesh, the coordinates over three inlets
(rectangular, circular, annular), and the arc length s along a curved 2D inlet.

    python tools/docs/boundary_coordinates.py [--mallard build/src/mallard] [--mallard3d build-3d/src/mallard]
                                              [names...]

The coordinates come from a NumPy port of src/boundary/boundary_coordinates.cpp
(face centers and area vectors as src/mesh/mesh_3d.cpp computes them). With
--mallard and --mallard3d, Mallard reads the same meshes and the frame and the
range of every coordinate in its log must match the port's.
"""
import argparse
import functools
import itertools
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import figstyle
from figstyle import bare, label


# Meshes: nodes (N, dim), cells (node lists), zones {name: [face node lists]}

def _extrude(base, quads, length, layers):
    """A 2D mesh of quadrilaterals in the (y, z) plane extruded along +x into hexahedra; zones by position."""
    m = len(base)
    xs = np.linspace(0.0, length, layers + 1)
    nodes = np.vstack([np.column_stack([np.full(m, x), base]) for x in xs])
    cells = [[v + k * m for v in q] + [v + (k + 1) * m for v in q] for k in range(layers) for q in quads]
    sides = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    count = {}
    for c in cells:
        for s in sides:
            face = tuple(c[i] for i in s)
            key = tuple(sorted(face))
            count[key] = (count[key][0] + 1, face) if key in count else (1, face)
    zones = {"inlet": [], "outlet": [], "wall": []}
    for n, face in count.values():
        if n == 1:
            x = nodes[list(face), 0]
            zones["inlet" if np.all(x == 0.0) else "outlet" if np.all(x == length) else "wall"].append(list(face))
    return nodes, cells, zones


def _structured(Y, Z):
    """Nodes and quadrilaterals of a structured (i, j) grid of the (y, z) plane."""
    ni, nj = Y.shape
    quads = [[i * nj + j, (i + 1) * nj + j, (i + 1) * nj + j + 1, i * nj + j + 1]
             for i in range(ni - 1) for j in range(nj - 1)]
    return np.column_stack([Y.ravel(), Z.ravel()]), quads


def _stretched(n, beta=1.6):
    """n + 1 points on [0, 1], finer at both ends."""
    return 0.5 * (1.0 + np.tanh(beta * np.linspace(-1.0, 1.0, n + 1)) / np.tanh(beta))


def _merge(parts):
    """Join (nodes, quads) blocks, merging coincident nodes."""
    nodes, quads, index = [], [], {}
    for pts, qs in parts:
        ids = []
        for p in pts:
            key = tuple(np.round(p, 12))
            if key not in index:
                index[key] = len(nodes)
                nodes.append(p)
            ids.append(index[key])
        quads += [[ids[v] for v in q] for q in qs]
    return np.array(nodes), quads


def duct_mesh(width=2.0, height=1.0, ny=16, nz=8, length=0.5, layers=2):
    Y, Z = np.meshgrid(width * _stretched(ny), height * _stretched(nz), indexing="ij")
    return _extrude(*_structured(Y, Z), length, layers)


def pipe_mesh(radius=1.0, m=8, k=6, length=0.5, layers=2):
    """An O-grid of the disk: a square core and four blocks out to the circle."""
    a = 0.42 * radius
    s = np.linspace(-1.0, 1.0, m + 1)
    parts = [_structured(*np.meshgrid(a * s, a * s, indexing="ij"))]
    t = np.linspace(0.0, 1.0, k + 1) ** 0.8
    for q in range(4):
        c, sn = np.cos(q * np.pi / 2), np.sin(q * np.pi / 2)
        rot = np.array([[c, -sn], [sn, c]])
        inner = np.column_stack([np.full(m + 1, a), a * s])
        phi = np.pi / 4 * s
        outer = radius * np.column_stack([np.cos(phi), np.sin(phi)])
        P = inner[:, None, :] * (1 - t[None, :, None]) + outer[:, None, :] * t[None, :, None]
        P = P @ rot.T
        parts.append(_structured(P[..., 0].T, P[..., 1].T))
    return _extrude(*_merge(parts), length, layers)


def annulus_mesh(inner=0.45, outer=1.0, nr=8, nt=48, length=0.5, layers=2):
    r = inner + (outer - inner) * _stretched(nr, 1.2)
    t = np.linspace(0.0, 2 * np.pi, nt + 1)
    R, T = np.meshgrid(r, t, indexing="ij")
    pts, quads = _structured(R * np.cos(T), R * np.sin(T))
    return _extrude(*_merge([(pts, quads)]), length, layers)


def half_ring_mesh(inner=1.0, outer=2.0, angle=180.0, nr=6, nt=36):
    """2D: half a ring; inlet on the outer arc, outlet on the inner one, walls on the straight sides."""
    r = inner + (outer - inner) * np.linspace(0, 1, nr + 1)
    t = np.deg2rad(angle) * np.linspace(0, 1, nt + 1)
    R, T = np.meshgrid(r, t, indexing="ij")
    nodes, quads = _structured(R * np.cos(T), R * np.sin(T))
    nj = nt + 1
    zones = {"inlet": [[nr * nj + j + 1, nr * nj + j] for j in range(nt)],
             "outlet": [[j, j + 1] for j in range(nt)],
             "wall": [[(i + 1) * nj, i * nj] for i in range(nr)]
             + [[i * nj + nt, (i + 1) * nj + nt] for i in range(nr)]}
    return nodes, quads, zones


def write_gmsh(path, nodes, cells, zones):
    dim = nodes.shape[1]
    types = {2: 1, 3: 2, 4: 3, 8: 5}
    names = list(zones)
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names) + 1)]
    lines += [f'{dim - 1} {i + 1} "{n}"' for i, n in enumerate(names)] + [f'{dim} {len(names) + 1} "fluid"']
    lines += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    lines += [f"{i + 1} " + " ".join(f"{x:.17g}" for x in (*p, 0.0)[:3]) for i, p in enumerate(nodes)]
    elements = [(i + 1, f) for i, n in enumerate(names) for f in zones[n]] + [(len(names) + 1, c) for c in cells]
    lines += ["$EndNodes", "$Elements", str(len(elements))]
    lines += [f"{k + 1} {types[len(e)]} 2 {tag} {tag} " + " ".join(str(v + 1) for v in e)
              for k, (tag, e) in enumerate(elements)]
    lines.append("$EndElements")
    path.write_text("\n".join(lines) + "\n")


# The port of src/boundary/boundary_coordinates.cpp

def face_geometry(nodes, face):
    """Center and outward area vector of a face, as Mallard computes them (a triangle fan in 3D)."""
    p = nodes[face]
    if p.shape[1] == 2:
        d = p[1] - p[0]
        return 0.5 * (p[0] + p[1]), np.array([d[1], -d[0]])
    c = p.mean(axis=0)
    a = np.cross(p - c, np.roll(p, -1, axis=0) - c)
    A = 0.5 * a.sum(axis=0)
    w = 0.5 * a @ A / np.linalg.norm(A)
    x = (w[:, None] * (c + p + np.roll(p, -1, axis=0)) / 3.0).sum(axis=0) / w.sum()
    return x, A


def _inward(nodes, cells, faces):
    """Each face's center and area vector, turned into the domain (the side of its cell)."""
    owner = {}
    for c in cells:
        cs = set(c)
        for i, f in enumerate(faces):
            if set(f) <= cs:
                owner[i] = nodes[c].mean(axis=0)
    centers, areas = [], []
    for i, f in enumerate(faces):
        x, A = face_geometry(nodes, f)
        if A @ (owner[i] - x) < 0:
            A = -A
        centers.append(x)
        areas.append(A)
    return np.array(centers), np.array(areas)


def _closest_on_segment(x, a, b):
    ab = b - a
    t = np.clip(((x - a) * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
    return a + t[..., None] * ab


def _closest_on_triangles(x, a, b, c):
    """Closest points to the points x (P, 1, 3) on the triangles (a, b, c) (T, 3): (P, T, 3)."""
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    q = x - ((x - a) * n).sum(-1, keepdims=True) * n
    inside = np.ones(q.shape[:-1], bool)
    for u, v in ((a, b), (b, c), (c, a)):
        inside &= (np.cross(v - u, q - u) * n).sum(-1) >= 0
    d = np.where(inside, np.linalg.norm(x - q, axis=-1), np.inf)
    for u, v in ((a, b), (b, c), (c, a)):
        e = _closest_on_segment(x, u, v)
        de = np.linalg.norm(x - e, axis=-1)
        better = de < d
        q[better], d[better] = e[better], de[better]
    return q


def wall_closest(nodes, walls, x):
    """The points of the wall faces nearest to the points x (P, dim): segments in 2D, triangles fanned from each
    face's lowest node in 3D."""
    x = np.atleast_2d(x)
    if nodes.shape[1] == 2:
        pieces = (nodes[[f[0] for f in walls]], nodes[[f[1] for f in walls]])
        closest = _closest_on_segment
    else:
        tris = []
        for f in walls:
            v = nodes[f]
            k = min(range(len(f)), key=lambda i: tuple(v[i]))
            v = np.roll(v, -k, axis=0)
            tris += [(v[0], v[j], v[j + 1]) for j in range(1, len(f) - 1)]
        pieces = tuple(np.array(t) for t in zip(*tris))
        closest = _closest_on_triangles
    out = np.empty_like(x)
    for i in range(0, len(x), 500):
        chunk = x[i:i + 500, None, :]
        pts = closest(chunk, *pieces)
        k = np.argmin(np.linalg.norm(pts - chunk, axis=-1), axis=1)
        out[i:i + 500] = pts[np.arange(len(k)), k]
    return out


def wall_distance(nodes, walls, x):
    return np.linalg.norm(wall_closest(nodes, walls, x) - np.atleast_2d(x), axis=-1)


def frame(centers, areas):
    """origin, normal, tangent (and span in 3D) of the faces with the default frame."""
    dim = centers.shape[1]
    area = np.linalg.norm(areas, axis=1)
    origin = (area[:, None] * centers).sum(0) / area.sum()
    normal = areas.sum(0)
    normal /= np.linalg.norm(normal)
    axis = np.zeros(dim)
    axis[(int(np.argmax(np.abs(normal))) + 1) % dim] = 1.0
    tangent = axis - (axis @ normal) * normal
    tangent /= np.linalg.norm(tangent)
    out = dict(origin=origin, normal=normal, tangent=tangent)
    if dim == 3:
        span = np.cross(normal, tangent)
        out["span"] = span if span[np.argmax(np.abs(span))] >= 0 else -span
    return out


def arc_length(nodes, faces, tangent):
    """s at each face's center, from the end of the curve with the lowest coordinate along tangent."""
    at = {}
    for k, f in enumerate(faces):
        for v in f:
            at.setdefault(v, []).append(k)
    ends = [v for v, ks in at.items() if len(ks) == 1]
    v = min(ends, key=lambda v: (nodes[v] @ tangent, *nodes[v]))
    s, length, seen = np.zeros(len(faces)), 0.0, set()
    for _ in faces:
        k = next(k for k in at[v] if k not in seen)
        seen.add(k)
        w = faces[k][0] if faces[k][1] == v else faces[k][1]
        h = np.linalg.norm(nodes[w] - nodes[v])
        s[k] = length + 0.5 * h
        length += h
        v = w
    return s


@functools.cache
def case(name):
    """The mesh of a case and the local coordinates of its inlet faces."""
    nodes, cells, zones = MESHES[name]()
    faces = zones["inlet"]
    centers, areas = _inward(nodes, cells, faces)
    F = frame(centers, areas)
    off = centers - F["origin"]
    c = dict(n=wall_distance(nodes, zones["wall"], centers),
             xi=off @ F["tangent"])
    if nodes.shape[1] == 3:
        c["eta"] = off @ F["span"]
        c["r"] = np.hypot(c["xi"], c["eta"])
        c["theta"] = np.arctan2(c["eta"], c["xi"])
    else:
        c["s"] = arc_length(nodes, faces, F["tangent"])
    return dict(nodes=nodes, cells=cells, zones=zones, centers=centers, frame=F, coords=c)


def turned(mesh, angles=(0.5, 0.3, 0.2), shift=(0.3, -0.2, 0.7)):
    """The mesh rotated about z, y and x by angles (rad) and shifted: a frame on no axis."""
    nodes, cells, zones = mesh
    c, s = np.cos(angles), np.sin(angles)
    rz = np.array([[c[0], -s[0], 0], [s[0], c[0], 0], [0, 0, 1]])
    ry = np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
    rx = np.array([[1, 0, 0], [0, c[2], -s[2]], [0, s[2], c[2]]])
    return nodes @ (rx @ ry @ rz).T + shift, cells, zones


MESHES = {"duct": duct_mesh, "pipe": pipe_mesh, "pipe_turned": lambda: turned(pipe_mesh()), "annulus": annulus_mesh,
          "half_ring": half_ring_mesh, "pipe_long": lambda: pipe_mesh(m=6, k=5, length=2.4, layers=10)}


# The check against Mallard

def _vector(text):
    return np.array([float(v) for v in text.split(",")])


def check(binary, name):
    """Mallard's frame and coordinate ranges of the case's inlet (from its log) must match the port's."""
    data = case(name)
    dim = data["nodes"].shape[1]
    used = ["n", "xi", "eta", "r", "theta"] if dim == 3 else ["n", "xi", "s"]
    zero = ", ".join(["0.0"] * dim)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_gmsh(work / "mesh.msh", data["nodes"], data["cells"], data["zones"])
        (work / "input.toml").write_text(f"""[run]
n_steps = 1
cfl = 0.5
[mesh]
type = "file"
filename = "mesh.msh"
[initialize]
type = "constant"
u = [{zero}]
p = 1.0
T = 1.0
[[boundaries]]
name = "inlet"
type = "inlet_state"
u = "0.1 + 0 * ({' + '.join(used)})"
p = 1.0
T = 1.0
walls = ["wall"]
[[boundaries]]
name = "wall"
type = "symmetry"
[[boundaries]]
name = "outlet"
type = "outlet_pressure"
p = 1.0
[numerics.face_reconstruction]
type = "MUSCL"
[physics]
type = "euler"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0
""")
        log = subprocess.run([str(binary), "check", "input.toml", "--kokkos-num-threads=2"], cwd=work,
                             capture_output=True, text=True, check=True).stdout
    lines = log.splitlines()
    start = next((i for i, line in enumerate(lines) if "local coordinates:" in line), None)
    if start is None:
        raise RuntimeError(f"{name}: no local coordinates in Mallard's log:\n{log}")
    indent = len(lines[start + 1]) - len(lines[start + 1].lstrip())
    wrapped = [lines[start]] + list(itertools.takewhile(lambda line: line[:indent].isspace(), lines[start + 1:]))
    text = " ".join(" ".join(wrapped).split()).split("local coordinates: ")[1]
    for key, value in data["frame"].items():
        got = _vector(re.search(key + r" \(([^)]*)\)", text).group(1))
        if not np.allclose(got, value, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"{name}: Mallard's {key} {got}, the port's {value}")
    for key in used:
        lo, hi = (float(v) for v in re.search(rf"\b{key} (\S+) to (\S+?)(?:;|,|$)", text).groups())
        mine = data["coords"][key]
        if not np.allclose([lo, hi], [mine.min(), mine.max()], rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"{name}: Mallard's {key} {lo} to {hi}, the port's {mine.min()} to {mine.max()}")
    print(f"{name}: the port matches Mallard ({text.split(';')[0]})")


# Figures

SYMBOLS = dict(n=r"$n$", xi=r"$\xi$", eta=r"$\eta$", r=r"$r$", theta=r"$\theta$", s=r"$s$")


def _arrow3(ax, a, b, color, lw=2.2):
    ax.quiver(*a, *(np.asarray(b) - a), color=color, lw=lw, arrow_length_ratio=0.14)


def boundary_frame(P):
    """A pipe's inlet in its mesh, the frame, and the coordinates of one face."""
    data = case("pipe_long")
    nodes, zones, F, c = data["nodes"], data["zones"], data["frame"], data["coords"]
    L = nodes[:, 0].max()
    fig = plt.figure(figsize=(10.4, 5.4))
    ax = fig.add_axes([0.0, 0.0, 0.66, 1.0], projection="3d", computed_zorder=False)
    view = np.array([-1.0, -0.55, 0.45])
    back = [nodes[f] for f in zones["wall"] if np.array([0.0, *nodes[f].mean(axis=0)[1:]]) @ view < 0]
    ax.add_collection3d(Poly3DCollection(back, facecolors=P["faint"], edgecolors=P["mesh"], linewidths=0.75))
    ax.add_collection3d(Poly3DCollection([nodes[f] for f in zones["outlet"]], facecolors="none",
                                         edgecolors=P["mesh"], linewidths=0.4))
    t = np.linspace(0, 2 * np.pi, 200)
    for x in (0.0, L):
        ax.plot(np.full_like(t, x), np.cos(t), np.sin(t), color=P["muted"], lw=1.0)
    for phi in (np.pi * 0.5 + 0.5, -np.pi * 0.5 + 0.5):
        ax.plot([0, L], [np.cos(phi)] * 2, [np.sin(phi)] * 2, color=P["muted"], lw=0.8)
    ax.add_collection3d(Poly3DCollection([nodes[f] for f in zones["inlet"]], facecolors=P["flame"], alpha=0.16,
                                         edgecolors=P["flame"], linewidths=0.7))
    target = int(np.argmin(np.linalg.norm(data["centers"] - [0.0, 0.5, 0.42], axis=1)))
    ax.add_collection3d(Poly3DCollection([nodes[zones["inlet"][target]]], facecolors=P["flame"], edgecolors=P["fg"],
                                         linewidths=1.0, alpha=0.9))
    o, x = F["origin"], data["centers"][target]
    xi, eta, r, theta, n = (c[k][target] for k in ("xi", "eta", "r", "theta", "n"))
    T, S, N = F["tangent"], F["span"], F["normal"]
    _arrow3(ax, o, o + 1.5 * N, P["plum"])
    _arrow3(ax, o, o + 1.4 * T, P["blue"])
    _arrow3(ax, o, o + 1.4 * S, P["blue"])
    ax.text(*(o + 1.6 * N + 0.12 * S), "normal", color=P["plum"], fontsize=12)
    ax.text(*(o + 1.4 * T - 0.18 * S), r"tangent ($\xi$)", color=P["blue"], fontsize=12, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=P["bg"], edgecolor="none", alpha=0.85))
    ax.text(*(o + 1.5 * S), r"span ($\eta$)", color=P["blue"], fontsize=12, ha="center")
    ax.plot(*np.array([o, o + xi * T, x]).T, color=P["blue"], lw=1.2, ls="--")
    ax.text(*(o + 0.5 * xi * T - 0.13 * S), r"$\xi$", color=P["blue"], fontsize=13, ha="center", va="center")
    ax.text(*(o + xi * T + 0.5 * eta * S + 0.1 * T), r"$\eta$", color=P["blue"], fontsize=13, va="center")
    ax.plot(*np.array([o, x]).T, color=P["fg"], lw=1.4)
    ax.text(*(o + 0.55 * (x - o) - 0.1 * T + 0.08 * S), r"$r$", color=P["fg"], fontsize=13, ha="center")
    a = np.linspace(0, theta, 30)
    ax.plot(*(o + 0.3 * (np.cos(a)[:, None] * T + np.sin(a)[:, None] * S)).T, color=P["fg"], lw=1.2)
    ax.text(*(o + 0.42 * (np.cos(theta / 2) * T + np.sin(theta / 2) * S)), r"$\theta$", color=P["fg"], fontsize=13,
            ha="center", va="center")
    wall = wall_closest(nodes, zones["wall"], x)[0]
    ax.plot(*np.array([x, wall]).T, color=P["teal"], lw=2.6)
    ax.text(*(0.5 * (x + wall) + 0.1 * S - 0.06 * T), r"$n$", color=P["teal"], fontsize=14, ha="right")
    ax.scatter(*o, color=P["fg"], s=22, depthshade=False)
    ax.text(*(o - 0.12 * S - 0.05 * T), "origin", color=P["fg"], fontsize=12, ha="left", va="top")
    ax.scatter(*x, color=P["fg"], s=14, depthshade=False)
    ax.text(0.5 * L, -0.3, -0.95, "wall", color=P["muted"], fontsize=12, ha="center")
    ax.text(0.0, 0.25, -1.2, "inlet", color=P["flame"], fontsize=12, ha="center", va="top")
    ax.set_xlim(-0.2, L)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_box_aspect((L + 0.2, 2.6, 2.6), zoom=1.15)
    ax.view_init(elev=22, azim=-152)
    ax.set_axis_off()

    tx = fig.add_axes([0.67, 0.06, 0.33, 0.88])
    tx.axis("off")
    rows = [("frame of the inlet", None, None),
            ("origin", "area-weighted center of the faces", P["fg"]),
            ("normal", "their mean normal, into the domain", P["plum"]),
            ("tangent", "the axis after the normal's\nlargest component (here y)", P["blue"]),
            ("span", "normal × tangent", P["blue"]),
            ("the highlighted face", None, None),
            (r"$n$ = " + f"{n:.2f}", "to the nearest wall face", P["teal"]),
            (r"$\xi$ = " + f"{xi:.2f}", "along the tangent", P["blue"]),
            (r"$\eta$ = " + f"{eta:.2f}", "along the span", P["blue"]),
            (r"$r$ = " + f"{r:.2f}", "hypot($\\xi$, $\\eta$)", P["fg"]),
            (r"$\theta$ = " + f"{theta:.2f}", "atan2($\\eta$, $\\xi$)", P["fg"])]
    y = 1.0
    for head, text, color in rows:
        if text is None:
            y -= 0.04 if y < 1 else 0
            tx.text(0.0, y, head, fontsize=12.5, fontweight="bold", va="top", color=P["fg"])
            y -= 0.08
            continue
        tx.text(0.03, y, head, fontsize=12, va="top", color=color)
        tx.text(0.3, y, text, fontsize=11, va="top", color=P["muted"])
        y -= 0.075 + 0.05 * text.count("\n")
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)
    return fig


def _plane(data, spacing=0.012):
    """The inlet's faces as polygons in (xi, eta), and the coordinates on a grid of points of the inlet."""
    F, nodes = data["frame"], data["nodes"]
    to2 = lambda p: np.column_stack([(p - F["origin"]) @ F["tangent"], (p - F["origin"]) @ F["span"]])  # noqa: E731
    polys = [to2(nodes[f]) for f in data["zones"]["inlet"]]
    lo, hi = np.min([p.min(0) for p in polys], axis=0), np.max([p.max(0) for p in polys], axis=0)
    g = [np.arange(lo[d], hi[d] + spacing, spacing) for d in range(2)]
    XI, ETA = np.meshgrid(*g, indexing="ij")
    pts = np.column_stack([XI.ravel(), ETA.ravel()])
    inside = np.zeros(len(pts), bool)
    for p in polys:
        inside |= MplPath(p).contains_points(pts, radius=1e-9)
    x3 = F["origin"] + pts[inside, :1] * F["tangent"] + pts[inside, 1:] * F["span"]
    field = dict(xi=XI, eta=ETA, r=np.hypot(XI, ETA), theta=np.arctan2(ETA, XI), n=np.full(XI.size, np.nan))
    field["n"][inside] = wall_distance(nodes, data["zones"]["wall"], x3)
    field["n"] = field["n"].reshape(XI.shape)
    out = ~inside.reshape(XI.shape)
    cut = (XI < 0) & (np.abs(ETA) < 1.5 * spacing)
    for key in field:
        field[key] = np.ma.masked_where(out | (cut if key == "theta" else False), field[key])
    return polys, XI, ETA, field


def _step(span):
    return next(s for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0) if span / s <= 4)


SURFACES = (("duct", "rectangle"), ("pipe", "circle"), ("annulus", "annulus"))
COLUMNS = (("n", "distance to the wall"), ("xi", "along the tangent"), ("eta", "along the span"),
           ("r", "from the origin"), ("theta", "from the tangent"))


def boundary_coordinates(P):
    """Each coordinate (columns) over the inlet faces of three ducts (rows): the face values and contour lines."""
    fig, axes = plt.subplots(len(SURFACES), len(COLUMNS), figsize=(13.0, 8.6),
                             gridspec_kw=dict(height_ratios=[1.0, 2.0, 2.0], hspace=0.22, wspace=0.08))
    cmap = plt.get_cmap(P["cmap"])
    for row, (name, title) in enumerate(SURFACES):
        data = case(name)
        polys, XI, ETA, field = _plane(data)
        for col, (key, meaning) in enumerate(COLUMNS):
            ax = axes[row, col]
            v = data["coords"][key]
            lo, hi = v.min(), v.max()
            norm = plt.Normalize(lo, hi)
            ax.add_collection(PolyCollection(polys, facecolors=cmap(norm(v)), edgecolors=P["bg"], linewidths=0.35))
            if key == "theta":
                step, levels = np.pi / 4, np.pi / 4 * np.arange(-3, 4)
                note = r"$-\pi$ to $\pi$, lines every $\pi/4$"
            else:
                step = _step(hi - lo)
                levels = step * np.arange(np.floor(field[key].min() / step), np.ceil(field[key].max() / step) + 1)
                levels = levels[(levels > field[key].min()) & (levels < field[key].max())]
                note = f"{lo:.2f} to {hi:.2f}, lines every {step:g}"
            ax.contour(XI, ETA, field[key], levels=levels, colors=P["fg"], linewidths=0.9, linestyles="solid")
            ax.plot(0, 0, "+", color=P["red"], ms=10, mew=2)
            bare(ax)
            ax.autoscale_view()
            ax.set_title(note, fontsize=10.5, color=P["muted"], pad=4)
            if row == 0:
                ax.text(0.5, 1.42, f"{SYMBOLS[key]}  {meaning}", transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=13)
            if col == 0:
                ax.text(-0.06, 0.5, title, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=13)
    return fig


def boundary_arc_length(P):
    """A curved 2D inlet: its frame, a face's n, xi and s, and the three along the inlet."""
    data = case("half_ring")
    nodes, zones, F, c = data["nodes"], data["zones"], data["frame"], data["coords"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.22))
    ax.add_collection(PolyCollection([nodes[q] for q in data["cells"]], facecolors=P["faint"], edgecolors=P["mesh"],
                                     linewidths=0.5))
    for f in zones["wall"]:
        ax.plot(*nodes[f].T, color=P["fg"], lw=2.6, solid_capstyle="butt")
    cmap = plt.get_cmap(P["cmap"])
    norm = plt.Normalize(0, c["s"].max())
    for k, f in enumerate(zones["inlet"]):
        ax.plot(*nodes[f].T, color=cmap(norm(c["s"][k])), lw=7, solid_capstyle="butt")
    o, T, N = F["origin"], F["tangent"], F["normal"]
    ax.add_patch(FancyArrowPatch(o, o + 0.6 * N, arrowstyle="-|>", mutation_scale=12, color=P["plum"], lw=2))
    label(ax, *(o + 0.6 * N + [0.05, -0.1]), "normal", P, size=11, color=P["plum"], ha="left")
    ax.add_patch(FancyArrowPatch(o, o - 0.75 * T, arrowstyle="-", color=P["blue"], lw=1.0, ls=":"))
    ax.add_patch(FancyArrowPatch(o, o + 0.75 * T, arrowstyle="-|>", mutation_scale=12, color=P["blue"], lw=2))
    label(ax, *(o + 0.4 * T - 0.17 * N), "tangent", P, size=11, color=P["blue"])
    k = int(np.argmin(np.abs(np.arctan2(data["centers"][:, 1], data["centers"][:, 0]) - np.deg2rad(55))))
    x = data["centers"][k]
    wall = wall_closest(nodes, zones["wall"], x)[0]
    ax.plot(*np.array([x, wall]).T, color=P["teal"], lw=2)
    label(ax, *(0.5 * (x + wall) + [0.12, 0.0]), r"$n$", P, size=13, color=P["teal"])
    foot = o + c["xi"][k] * T + 0.3 * N
    ax.annotate("", xy=foot, xytext=o + 0.3 * N, arrowprops=dict(arrowstyle="<->", color=P["blue"], lw=1.4,
                                                                    shrinkA=0, shrinkB=0))
    ax.plot(*np.array([x, foot]).T, ls=":", color=P["muted"], lw=1.0)
    label(ax, *(0.5 * (o + foot) + 0.15 * N + 0.12 * T), r"$\xi$", P, size=13, color=P["blue"])
    start = nodes[[v for f in zones["inlet"] for v in f]]
    start = start[np.argmin(start @ T)]
    a = np.linspace(np.arctan2(start[1], start[0]), np.arctan2(x[1], x[0]), 80)
    ax.plot(2.22 * np.cos(a), 2.22 * np.sin(a), color=P["gold"], lw=1.8)
    ax.plot(2.22 * np.cos(a[-1]), 2.22 * np.sin(a[-1]), "o", color=P["gold"], ms=4)
    label(ax, 2.4 * np.cos(a.mean()), 2.4 * np.sin(a.mean()), r"$s$", P, size=13, color=P["gold"])
    ax.plot(*o, "o", color=P["fg"], ms=5, zorder=5)
    label(ax, o[0] - 0.08, o[1] + 0.1, "origin", P, size=11, ha="right")
    ax.plot(*x, "o", color=P["fg"], ms=5, zorder=5)
    label(ax, -1.5, -0.15, "wall", P, size=11, color=P["muted"], va="top")
    label(ax, 1.5, -0.15, "wall", P, size=11, color=P["muted"], va="top")
    label(ax, 2.2, 1.25, "inlet,\ncolored\nby $s$", P, size=11, color=P["fg"], ha="left")
    bare(ax)
    ax.set_xlim(-2.5, 2.95)
    ax.set_ylim(-0.35, 2.55)

    order = np.argsort(c["s"])
    for key, color in (("s", P["gold"]), ("xi", P["blue"]), ("n", P["teal"])):
        bx.plot(c["s"][order], c[key][order], "o-", ms=3.5, lw=1.6, color=color, label=SYMBOLS[key])
    bx.axhline(0, color=P["muted"], lw=0.8)
    bx.set_xlabel(r"$s$ of the inlet face [m]")
    bx.set_ylabel("coordinate [m]")
    bx.legend(loc="upper left")
    return fig


def figures(mallard=None, mallard3d=None, names=None):
    if mallard3d:
        for name in ("duct", "pipe", "pipe_turned", "annulus", "pipe_long"):
            check(mallard3d, name)
    if mallard:
        check(mallard, "half_ring")
    for name, draw in (("boundary_arc_length", boundary_arc_length), ("boundary_coordinates", boundary_coordinates),
                       ("boundary_frame", boundary_frame)):
        if not names or name in names:
            figstyle.save(name, draw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", help="2D Mallard executable, to check the 2D coordinates")
    ap.add_argument("--mallard3d", help="3D Mallard executable, to check the 3D coordinates")
    ap.add_argument("names", nargs="*")
    ns = ap.parse_args()
    figures(ns.mallard, ns.mallard3d, ns.names)


if __name__ == "__main__":
    sys.exit(main())
