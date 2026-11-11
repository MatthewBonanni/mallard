"""Figures of the mesh and stencil fields of mallard check --mesh-fields: quality and
stencil fields and one cell's stencils on a mesh of quadrilaterals and triangles, and
the same on a mesh of hexahedra and prisms in 3D.

    python tools/docs/mesh_fields.py --mallard build/src/mallard [--mallard3d build-3d/src/mallard] [names...]

Without --mallard3d the 3D figure is skipped. Needs h5py.
"""
import argparse
import functools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import figstyle
from figstyle import bare

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from mallard.stencils import MIRROR, read_stencils  # noqa: E402
from mallard.vtu import read_vtu_cells, snapshot  # noqa: E402

PHYSICS = '[physics]\ntype = "euler"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n'
N = 12


def _grid(n, jitter, seed):
    rng = np.random.default_rng(seed)
    g = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inner = (pts[:, 0] > 0) & (pts[:, 0] < 1) & (pts[:, 1] > 0) & (pts[:, 1] < 1)
    pts[inner] += rng.uniform(-jitter, jitter, (inner.sum(), 2)) / n
    return pts


def _faces2d(n):
    """Quadrilaterals left of x = 0.5, each quadrilateral split into two triangles right of it, counterclockwise."""
    nid = lambda i, j: i * (n + 1) + j  # noqa: E731
    cells = []
    for i in range(n):
        for j in range(n):
            q = [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
            if i < n // 2:
                cells.append(q)
            elif (i + j) % 2:
                cells += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
            else:
                cells += [[q[0], q[1], q[3]], [q[1], q[2], q[3]]]
    return cells


def _write_gmsh(path, pts, boundary, cells, dim):
    types2 = {2: 1, 3: 2, 4: 3}
    types3 = {3: 2, 4: 3, 6: 6, 8: 5}
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "1", f'{dim - 1} 1 "wall"',
             "$EndPhysicalNames", "$Nodes", str(len(pts))]
    lines += [f"{i + 1} " + " ".join(f"{x:.17g}" for x in (*p, 0.0)[:3]) for i, p in enumerate(pts)]
    lines += ["$EndNodes", "$Elements", str(len(boundary) + len(cells))]
    k = 1
    for f in boundary:
        t = types2[len(f)] if dim == 2 else types3[len(f)]
        lines.append(f"{k} {t} 2 1 1 " + " ".join(str(v + 1) for v in f))
        k += 1
    for c in cells:
        t = types2[len(c)] if dim == 2 else types3[len(c)]
        lines.append(f"{k} {t} 2 0 2 " + " ".join(str(v + 1) for v in c))
        k += 1
    lines.append("$EndElements")
    path.write_text("\n".join(lines) + "\n")


def mixed_mesh(path, n=N, jitter=0.25, seed=5):
    """Gmsh 2.2 file of the unit square: jittered quadrilaterals and triangles, boundary zone "wall"."""
    pts = _grid(n, jitter, seed)
    cells = _faces2d(n)
    edges = {}
    for c in cells:
        for a, b in zip(c, c[1:] + c[:1]):
            edges[(min(a, b), max(a, b))] = edges.get((min(a, b), max(a, b)), 0) + 1
    _write_gmsh(path, pts, [e for e, count in edges.items() if count == 1], cells, 2)


def extruded_mesh(path, n=8, layers=4, depth=0.5, jitter=0.25, seed=5):
    """Gmsh 2.2 file of the box [0, 1]^2 x [0, depth]: the 2D mesh extruded into hexahedra and prisms."""
    base = _grid(n, jitter, seed)
    m = len(base)
    pts = np.vstack([np.column_stack([base, np.full(m, depth * k / layers)]) for k in range(layers + 1)])
    cells = []
    for k in range(layers):
        for f in _faces2d(n):
            cells.append([v + k * m for v in f] + [v + (k + 1) * m for v in f])
    sides = {4: ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
             3: ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5))}
    faces = {}
    for c in cells:
        for f in sides[len(c) // 2]:
            nodes = tuple(c[v] for v in f)
            key = tuple(sorted(nodes))
            faces[key] = (faces[key][0] + 1, nodes) if key in faces else (1, nodes)
    _write_gmsh(path, pts, [nodes for count, nodes in faces.values() if count == 1], cells, 3)


def check(binary, mesh, work, dim):
    """mallard check --mesh-fields of a fifth-order TENO case on the mesh, every cell's stencils."""
    zero = "[0.0, 0.0]" if dim == 2 else "[0.0, 0.0, 0.0]"
    (work / "input.toml").write_text(f"""[run]
n_steps = 1
cfl = 0.5
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
[numerics.face_reconstruction]
type = "TENO"
order = 5
{PHYSICS}""")
    subprocess.run([str(binary), "check", "input.toml", "--mesh-fields", "mesh", "--stencils", "all",
                    "--kokkos-num-threads=2"], cwd=work, check=True, stdout=subprocess.DEVNULL)
    pts, conn, offs, types, arrays = read_vtu_cells(snapshot(str(work / "mesh_000000")))
    starts = np.concatenate([[0], offs[:-1]])
    return pts, [conn[a:b] for a, b in zip(starts, offs)], arrays, read_stencils(work / "mesh_stencils.h5")


@functools.cache
def data_2d(mallard):
    work = Path(tempfile.mkdtemp())
    mixed_mesh(work / "mixed.msh")
    return check(mallard, "mixed.msh", work, 2)


@functools.cache
def data_3d(mallard3d):
    work = Path(tempfile.mkdtemp())
    extruded_mesh(work / "mixed.msh")
    return check(mallard3d, "mixed.msh", work, 3)


def _reflect(poly, a, b):
    d = (b - a) / np.linalg.norm(b - a)
    r = poly - a
    return a + 2 * np.outer(r @ d, d) - r


def _boundary_edge(poly):
    """The edge of a boundary cell's polygon on the boundary of the unit square."""
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        for x in range(2):
            if (abs(a[x]) < 1e-12 and abs(b[x]) < 1e-12) or (abs(a[x] - 1) < 1e-12 and abs(b[x] - 1) < 1e-12):
                return a, b
    raise ValueError("not a boundary cell")


def _field(fig, ax, P, polys, values, label):
    cells = PolyCollection(polys, array=values, cmap=P["cmap"], edgecolors=P["mesh"], linewidths=0.3, rasterized=True)
    ax.add_collection(cells)
    bar = fig.colorbar(cells, ax=ax, fraction=0.046, pad=0.02, location="bottom")
    bar.set_label(label)
    bar.outline.set_edgecolor(P["muted"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    bare(ax)


def mesh_fields(P, args):
    pts, cells, arrays, stencils = data_2d(args.mallard)
    polys = [pts[c, :2] for c in cells]
    centers = np.array([p.mean(axis=0) for p in polys])
    target = int(np.argmin(np.hypot(centers[:, 0] - 0.52, centers[:, 1] - 0.13)))
    s = stencils.stencil(target)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.9), layout="constrained")
    _field(fig, axes[0], P, polys, arrays["NON_ORTHOGONALITY"], "NON_ORTHOGONALITY [deg]")
    axes[0].set_title("non-orthogonality", fontsize=12)
    _field(fig, axes[1], P, polys, arrays["STENCIL_CONDITION"], "STENCIL_CONDITION")
    axes[1].set_title("condition of the central stencil's fit", fontsize=12)
    ax = axes[2]
    ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors=P["mesh"], linewidths=0.6))
    central = s["stencil"] == 0
    n_mirror = 0
    for c, role, face_cell in zip(s["cell"][central], s["role"][central], s["face_cell"][central]):
        poly = polys[c]
        if role & MIRROR:
            poly = _reflect(poly, *_boundary_edge(polys[face_cell]))
            n_mirror += 1
            ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor=P["flame"], hatch="////", lw=0.8))
        else:
            ax.add_patch(Polygon(poly, closed=True, facecolor=P["teal"], alpha=0.38, edgecolor=P["teal"], lw=0.8))
    ax.add_patch(Polygon(polys[target], closed=True, facecolor=P["fg"], edgecolor=P["fg"], lw=1))
    ax.axhline(0.0, color=P["line"], lw=1.4)
    c0 = centers[target]
    ax.set_xlim(c0[0] - 0.33, c0[0] + 0.33)
    ax.set_ylim(-0.2, 0.46)
    bare(ax)
    ax.set_title(f"cell {target}: {int(central.sum())} cells, {n_mirror} mirrored across the boundary", fontsize=12)
    return fig


def _cell_faces(p):
    """The faces of a hexahedron or a wedge (VTK node order), as polygons."""
    if len(p) == 8:
        sides = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    else:
        sides = ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5))
    return [p[list(f)] for f in sides]


def mesh_fields_3d(P, args):
    """The surface cells colored by non-orthogonality, and one cell's central stencil (mirror images left out)."""
    pts, cells, arrays, stencils = data_3d(args.mallard3d)
    centers = np.array([pts[c].mean(axis=0) for c in cells])
    target = int(np.argmin(np.linalg.norm(centers - [0.5, 0.45, 0.25], axis=1)))
    s = stencils.stencil(target)
    fig = plt.figure(figsize=(11.0, 4.6))
    values = arrays["NON_ORTHOGONALITY"]
    cmap, norm = plt.get_cmap(P["cmap"]), plt.Normalize(values.min(), values.max())
    ax = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    polys, colors = [], []
    for c, nodes in enumerate(cells):
        for f in _cell_faces(pts[nodes]):
            if np.all(np.isclose(f[:, 1], 0.0)) or np.all(np.isclose(f[:, 0], 1.0)) or np.all(np.isclose(f[:, 2], 0.5)):
                polys.append(f)
                colors.append(cmap(norm(values[c])))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=colors, edgecolors=P["mesh"], linewidths=0.2,
                                         rasterized=True))
    ax.set_title("non-orthogonality on the surface cells", fontsize=12)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.08)
    bar.set_label("NON_ORTHOGONALITY [deg]")
    bar.outline.set_edgecolor(P["muted"])
    ax2 = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
    members = sorted({int(c) for c, st, role in zip(s["cell"], s["stencil"], s["role"])
                      if st == 0 and not role & MIRROR and c != target})
    ax2.add_collection3d(Poly3DCollection([f for c in members for f in _cell_faces(pts[cells[c]])],
                                          facecolors=P["teal"], edgecolors=P["bg"], linewidths=0.3, alpha=0.3,
                                          rasterized=True))
    ax2.add_collection3d(Poly3DCollection(_cell_faces(pts[cells[target]]), facecolors=P["fg"], edgecolors=P["fg"],
                                          linewidths=0.4))
    n_mirror = int(np.sum((s["stencil"] == 0) & ((s["role"] & MIRROR) != 0)))
    ax2.set_title(f"central stencil of cell {target}: {int(s['sizes'][0])} cells"
                  + (f", {n_mirror} mirrored (not drawn)" if n_mirror else ""), fontsize=12)
    box = np.vstack([pts[cells[c]] for c in members])
    lo, hi = box.min(axis=0), box.max(axis=0)
    half = 0.5 * (hi - lo).max()
    mid = 0.5 * (lo + hi)
    for a, limits, aspect in ((ax, [(0, 1), (0, 1), (0, 0.5)], (1, 1, 0.5)),
                              (ax2, [(m - half, m + half) for m in mid], (1, 1, 1))):
        a.set_xlim(*limits[0])
        a.set_ylim(*limits[1])
        a.set_zlim(*limits[2])
        a.set_box_aspect(aspect, zoom=1.0 if a is ax else 1.25)
        a.view_init(elev=24, azim=-60)
        a.set_axis_off()
    return fig


def figures(mallard, mallard3d=None):
    args = argparse.Namespace(mallard=Path(mallard).resolve(),
                              mallard3d=Path(mallard3d).resolve() if mallard3d else None)
    figstyle.save("mesh_fields", lambda P: mesh_fields(P, args))
    if mallard3d:
        figstyle.save("mesh_fields_3d", lambda P: mesh_fields_3d(P, args))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="2D Mallard executable (an HDF5 build)")
    ap.add_argument("--mallard3d", help="3D Mallard executable (an HDF5 build), for the 3D figure")
    ns = ap.parse_args()
    figures(ns.mallard, ns.mallard3d)


if __name__ == "__main__":
    main()
