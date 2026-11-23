"""Figure of the high-order view ([[write_data]] high_order): the same TENO
snapshot drawn from its cell averages and from its Lagrange cells, read back
from a run of one step on a coarse mesh of quadrilaterals and triangles.

    python tools/docs/high_order_view.py --mallard build/src/mallard
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt

import figstyle
from figstyle import bare

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from mallard.vtu import read_vtu_cells  # noqa: E402

N = 7
ORDER = 5
RHO = "1.0 + 0.05 * exp(-((x - 0.45)^2 + (y - 0.55)^2) / 0.05)"


def mixed_mesh(path, n=N, jitter=0.18, seed=5):
    """Gmsh 2.2 file of the unit square: jittered quadrilaterals, every other column split in triangles."""
    rng = np.random.default_rng(seed)
    nid = lambda i, j: j * (n + 1) + i + 1  # noqa: E731
    pts = []
    for j in range(n + 1):
        for i in range(n + 1):
            inner = 0 < i < n and 0 < j < n
            d = rng.uniform(-jitter, jitter, 2) if inner else np.zeros(2)
            pts.append(((i + d[0]) / n, (j + d[1]) / n))
    elements = []
    for i in range(n):
        elements.append(f"1 2 1 1 {nid(i, 0)} {nid(i + 1, 0)}")
        elements.append(f"1 2 1 1 {nid(i + 1, n)} {nid(i, n)}")
        elements.append(f"1 2 1 1 {nid(0, i + 1)} {nid(0, i)}")
        elements.append(f"1 2 1 1 {nid(n, i)} {nid(n, i + 1)}")
    for j in range(n):
        for i in range(n):
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            if i % 2 == 0:
                elements.append(f"3 2 0 2 {a} {b} {c} {d}")
            else:
                elements += [f"2 2 0 2 {a} {b} {c}", f"2 2 0 2 {a} {c} {d}"]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "1", '1 1 "sides"', "$EndPhysicalNames",
             "$Nodes", str(len(pts))]
    lines += [f"{k + 1} {x:.17g} {y:.17g} 0" for k, (x, y) in enumerate(pts)]
    lines += ["$EndNodes", "$Elements", str(len(elements))]
    lines += [f"{k + 1} {e}" for k, e in enumerate(elements)]
    lines.append("$EndElements")
    path.write_text("\n".join(lines) + "\n")


def run(mallard, work):
    mixed_mesh(work / "mesh.msh")
    (work / "input.toml").write_text(
        '[run]\nn_steps = 1\ncfl = 0.2\n[mesh]\ntype = "file"\nfilename = "mesh.msh"\n'
        f'[initialize]\ntype = "analytical"\nrho = "{RHO}"\nu = ["0.0", "0.0"]\np = "1.0"\n'
        '[[boundaries]]\nname = "sides"\ntype = "outlet_extrapolation"\n'
        '[physics]\ntype = "euler"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n'
        f'[numerics.face_reconstruction]\ntype = "TENO"\norder = {ORDER}\n'
        '[[write_data]]\nprefix = "out/flow"\nformat = "vtu"\ninterval = 1\nfields = ["RHO"]\n'
        '[output]\nrun_log = ""\n')
    subprocess.run([mallard, "-i", "input.toml", "--kokkos-num-threads=2"], cwd=work, check=True,
                   stdout=subprocess.DEVNULL)
    return read_vtu_cells(str(work / "out/flow_000000.vtu")), read_view(work / "out/flow_high_order_000000.vtu")


def read_view(path):
    """Points, offsets, types and the point array RHO of a view's VTU (raw appended, little endian)."""
    raw = path.read_bytes()
    head = raw[:raw.index(b"<AppendedData")].decode()
    base = raw.index(b"_", raw.index(b"<AppendedData")) + 1
    arrays = {}
    for tag in head.split("<DataArray")[1:]:
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag.split(">")[0]))
        if "offset" not in attrs:
            continue
        at = base + int(attrs["offset"])
        size = int(np.frombuffer(raw[at:at + 8], "<u8")[0])
        dtype = {"Float64": "<f8", "Float32": "<f4", "Int64": "<i8", "UInt8": "u1", "Int32": "<i4"}[attrs["type"]]
        arrays[attrs.get("Name", "Points")] = np.frombuffer(raw[at + 8:at + 8 + size], dtype)
    return arrays["Points"].reshape(-1, 3), arrays["offsets"], arrays["types"], arrays["RHO"]


def lagrange_cells(points, offsets, types, values, p, n_sub=12):
    """Each Lagrange cell drawn as small triangles colored by its polynomial: the nodes' reference coordinates
    are those of the cell's own map from its vertices (affine triangles, bilinear quadrilaterals), and the
    polynomial through the node values (P_p, Q_p) is evaluated on a finer lattice."""
    polys, colors = [], []
    start = 0
    for end, t in zip(offsets, types):
        x, f = points[start:end, :2], values[start:end]
        if t == 69:
            v = x[:3]
            ref = np.linalg.solve(np.column_stack([v[1] - v[0], v[2] - v[0]]), (x - v[0]).T).T
            mono = [(a, b) for a in range(p + 1) for b in range(p + 1 - a)]
            g = [(i / n_sub, j / n_sub) for j in range(n_sub + 1) for i in range(n_sub + 1 - j)]
            index = {(round(a * n_sub), round(b * n_sub)): k for k, (a, b) in enumerate(g)}
            tri = []
            for j in range(n_sub):
                for i in range(n_sub - j):
                    tri.append((index[i, j], index[i + 1, j], index[i, j + 1]))
                    if i + j < n_sub - 1:
                        tri.append((index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]))
            to_x = lambda r, s: v[0] + r * (v[1] - v[0]) + s * (v[2] - v[0])  # noqa: E731
        else:
            v = x[:4]
            ref = np.array([bilinear_inverse(v, q) for q in x])
            mono = [(a, b) for a in range(p + 1) for b in range(p + 1)]
            g = [(i / n_sub, j / n_sub) for j in range(n_sub + 1) for i in range(n_sub + 1)]
            tri = []
            for j in range(n_sub):
                for i in range(n_sub):
                    k = j * (n_sub + 1) + i
                    tri += [(k, k + 1, k + n_sub + 2), (k, k + n_sub + 2, k + n_sub + 1)]
            to_x = lambda r, s: bilinear(v, r, s)  # noqa: E731
        A = np.array([[r ** a * s ** b for a, b in mono] for r, s in ref])
        coef = np.linalg.solve(A, f)
        gv = np.array([[r ** a * s ** b for a, b in mono] for r, s in g]) @ coef
        gx = np.array([to_x(r, s) for r, s in g])
        for a, b, c in tri:
            polys.append(gx[[a, b, c]])
            colors.append((gv[a] + gv[b] + gv[c]) / 3.0)
        start = end
    return polys, np.array(colors)


def bilinear(v, r, s):
    return (1 - r) * (1 - s) * v[0] + r * (1 - s) * v[1] + r * s * v[2] + (1 - r) * s * v[3]


def bilinear_inverse(v, q):
    rs = np.array([0.5, 0.5])
    for _ in range(30):
        r, s = rs
        J = np.column_stack([(1 - s) * (v[1] - v[0]) + s * (v[2] - v[3]), (1 - r) * (v[3] - v[0]) + r * (v[2] - v[1])])
        step = np.linalg.solve(J, bilinear(v, r, s) - q)
        rs = rs - step
        if np.abs(step).max() < 1e-14:
            break
    return rs


def draw(P, data):
    (pts, conn, offs, types, arrays), view = data
    starts = np.concatenate([[0], offs[:-1]])
    cells = [pts[conn[a:b], :2] for a, b in zip(starts, offs)]
    rho = np.asarray(arrays["RHO"])
    polys, colors = lagrange_cells(*view, ORDER - 1)
    lo, hi = min(rho.min(), colors.min()), max(rho.max(), colors.max())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    for ax, title in zip(axes, ["Cell averages (flow_000000)", f"High-order view, degree {ORDER - 1} "
                                                              "(flow_high_order_000000)"]):
        bare(ax)
        ax.set_title(title, color=P["fg"], fontsize=12)
    axes[0].add_collection(PolyCollection(cells, array=rho, cmap=P["cmap"], clim=(lo, hi), edgecolors="none"))
    axes[1].add_collection(PolyCollection(polys, array=colors, cmap=P["cmap"], clim=(lo, hi), edgecolors="face",
                                          linewidths=0.3, antialiased=False, rasterized=True))
    for ax in axes:
        ax.add_collection(PolyCollection(cells, facecolors="none", edgecolors=P["bg"], linewidths=0.6))
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
    fig.subplots_adjust(wspace=0.06)
    return fig


def figures(mallard):
    with tempfile.TemporaryDirectory() as tmp:
        data = run(str(Path(mallard).resolve()), Path(tmp))
    figstyle.save("high-order-view", lambda P: draw(P, data))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="2D Mallard executable")
    figures(ap.parse_args().mallard)


if __name__ == "__main__":
    main()
