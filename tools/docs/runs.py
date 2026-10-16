"""Figures of the user guide drawn from short Mallard runs (seconds each, on a
CPU build): TENO-E stencils read back from a stencil cache, the Sod shock
tube with each reconstruction (through a line probe), MPI partitions, a
MallardReactor ignition, and a remap from quadrilaterals to triangles.

    python tools/docs/runs.py --mallard build/src/Mallard [--mpirun mpirun] [names...]

MallardReactor is taken next to Mallard. Partitions need an MPI build.
"""
import argparse
import csv
import functools
import struct
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
import matplotlib.path
import matplotlib.pyplot as plt

import figstyle
from figstyle import bare, label

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from mallard_vtu import read_vtu_cells  # noqa: E402
from plot_sod import exact_sod  # noqa: E402

PHYSICS = '\n[physics]\ntype = "euler"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n'


def run(binary, toml, workdir, ranks=1, mpirun="mpirun", name="input.toml"):
    path = Path(workdir) / name
    path.write_text(toml)
    cmd = [str(binary), "-i", name, "--kokkos-num-threads=2"]
    if ranks > 1:
        cmd = [mpirun, "-n", str(ranks), "--oversubscribe", *cmd]
    subprocess.run(cmd, cwd=workdir, check=True, stdout=subprocess.DEVNULL)


def polygons(path):
    pts, conn, offs, types, data = read_vtu_cells(str(path))
    starts = np.concatenate([[0], offs[:-1]])
    return [pts[conn[a:b], :2] for a, b in zip(starts, offs)], data


# ------------------------------------------------------------- TENO stencils

def read_teno_cache(path, nk, nk_small=5, n_faces=4):
    """Stencils of a 2D, double-precision TENO cache (src/numerics/teno.cpp, save_cache)."""
    raw = open(path, "rb").read()
    off = 16 + 16 + 8 + 8 + 1
    (n_cells,) = struct.unpack_from("<I", raw, off)
    off += 4
    cells = []
    for _ in range(n_cells):
        off += 1 + 8
        (n_large,) = struct.unpack_from("<H", raw, off)
        off += 2
        small = struct.unpack_from(f"<{n_faces}H", raw, off)
        off += 2 * n_faces
        off += 8 * nk + 8 * (nk * (nk + 1) // 2)
        large = struct.unpack_from(f"<{n_large}i", raw, off)
        off += 4 * n_large
        large_faces = struct.unpack_from(f"<{n_large}i", raw, off)
        off += 4 * n_large + 8 * n_large * nk
        n_small = sum(small)
        small_cells = struct.unpack_from(f"<{n_small}i", raw, off)
        off += 4 * n_small
        small_faces = struct.unpack_from(f"<{n_small}i", raw, off)
        off += 4 * n_small + 8 * n_small * nk_small
        cells.append(dict(large=large, large_faces=large_faces, small=small, small_cells=small_cells,
                          small_faces=small_faces))
    assert off == len(raw), "unexpected TENO cache layout"
    return cells


@functools.cache
def teno_data(args):
    work = Path(tempfile.mkdtemp())
    toml = f"""
[run]
n_steps = 1
cfl = 0.5
[mesh]
type = "cartesian_tri"
Nx = 9
Ny = 9
[initialize]
type = "constant"
u = [0.0, 0.0]
p = 1.0
T = 1.0
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
order = 5
cache_file = "teno.cache"
[[write_data]]
prefix = "mesh"
format = "vtu"
interval = 1
variables = ["RHO"]
{PHYSICS}"""
    run(args.mallard, toml, work)
    polys, _ = polygons(work / "mesh_000000.vtu")
    return polys, read_teno_cache(work / "teno.cache", nk=14)


def teno_stencils(P, args):
    polys, stencils = teno_data(args)
    target = min(range(len(polys)), key=lambda c: np.linalg.norm(polys[c].mean(axis=0) - [0.5, 0.5]))
    s = stencils[target]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw=dict(wspace=0.06))
    colors = [P["teal"], P["flame"], P["blue"], P["plum"]]
    for k, ax in enumerate(axes):
        for poly in polys:
            ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor=P["mesh"], lw=0.8))
        if k == 0:
            for c in s["large"]:
                ax.add_patch(Polygon(polys[c], closed=True, facecolor=P["teal"], alpha=0.35, edgecolor=P["teal"],
                                     lw=0.8))
            ax.set_title(f"central stencil: {len(s['large'])} cells", fontsize=12)
        else:
            start = 0
            for f, n in enumerate(s["small"]):
                for c in s["small_cells"][start:start + n]:
                    if c == target:
                        continue
                    ax.add_patch(Polygon(polys[c], closed=True, facecolor=colors[f + 1], alpha=0.38,
                                         edgecolor=colors[f + 1], lw=0.8))
                start += n
            n_sectors = sum(1 for n in s["small"] if n)
            ax.set_title(f"{n_sectors} sector stencils, one per face", fontsize=12)
        ax.add_patch(Polygon(polys[target], closed=True, facecolor=P["fg"], edgecolor=P["fg"], lw=1))
        bare(ax)
        c0 = polys[target].mean(axis=0)
        ax.set_xlim(c0[0] - 0.36, c0[0] + 0.36)
        ax.set_ylim(c0[1] - 0.36, c0[1] + 0.36)
    return fig


# ---------------------------------------------------------------- Sod schemes

@functools.cache
def sod_data(args):
    schemes = [
        ("first order", 'type = "FO"', "muted"),
        ("MUSCL", 'type = "MUSCL"', "blue"),
        ("TENO-E 3", 'type = "TENO"\norder = 3', "flame"),
        ("TENO-E 5", 'type = "TENO"\norder = 5', "teal"),
    ]
    base = tomllib.load(open(ROOT / "examples" / "sod" / "input.toml", "rb"))
    nx = base["mesh"]["Nx"]
    results = []
    for name, recon, color in schemes:
        work = Path(tempfile.mkdtemp())
        text = (ROOT / "examples" / "sod" / "input.toml").read_text()
        text = text.split("[numerics.face_reconstruction]")[0] + "[numerics.face_reconstruction]\n" + recon + "\n"
        text += PHYSICS
        text += f"""
[[probes]]
name = "line"
variables = ["RHO"]
start = [{0.5 / nx}, 0.01]
end = [{1 - 0.5 / nx}, 0.01]
n_points = {nx}
interval = 1
"""
        run(args.mallard, text, work)
        rows = list(csv.DictReader(open(work / "probe_line.csv")))
        last = max(float(r["t"]) for r in rows)
        x = np.array([float(r["x"]) for r in rows if float(r["t"]) == last])
        rho = np.array([float(r["RHO"]) for r in rows if float(r["t"]) == last])
        results.append((name, color, x, rho, last))
    return results


def sod_schemes(P, args):
    results = sod_data(args)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.28))
    xe = np.linspace(0, 1, 2000)
    t = results[0][4]
    re = exact_sod(xe, t)
    for ax, (x0, x1) in zip(axes, ((0, 1), (0.62, 0.92))):
        ax.plot(xe, re, color=P["fg"], lw=1.2, label="exact")
        for name, color, x, rho, _ in results:
            ax.plot(x, rho, "-", color=P[color], lw=1.6, label=name, marker="o" if ax is axes[1] else None, ms=2.5)
        ax.set_xlim(x0, x1)
        ax.set_xlabel("x")
    axes[1].set_ylim(0.1, 0.45)
    axes[0].set_ylabel(r"density, $t = %.1f$" % t)
    axes[1].set_title("contact and shock", fontsize=12, color=P["muted"])
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.18), ncol=5, columnspacing=1.0, handlelength=1.5)
    return fig


# ----------------------------------------------------------------- partitions

@functools.cache
def partition_data(args):
    work = Path(tempfile.mkdtemp())
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_cylinder_mesh.py"), str(work / "cyl.msh"),
                    "--n-theta", "96", "--n-r", "40", "--r-far", "6", "--dr0", "0.03"], check=True,
                   stdout=subprocess.DEVNULL)
    cases = []
    for partitioner in ("hilbert", "multijagged"):
        for mesh in ('type = "cartesian_tri"\nNx = 40\nNy = 20\nLx = 2.0\nLy = 1.0', 'type = "file"\nfilename = "cyl.msh"'):
            box = "cartesian" in mesh
            zones = ("left", "right", "top", "bottom") if box else ("cylinder", "farfield")
            bcs = "".join(f'[[boundaries]]\nname = "{z}"\ntype = "symmetry"\n' for z in zones)
            prefix = f"{partitioner}_{'box' if box else 'cyl'}"
            toml = f"""
[run]
n_steps = 1
cfl = 0.5
[mesh]
{mesh}
[initialize]
type = "constant"
u = [0.0, 0.0]
p = 1.0
T = 1.0
{bcs}
[parallel]
partitioner = "{partitioner}"
[[write_data]]
prefix = "{prefix}"
format = "vtu"
interval = 1
variables = ["RHO"]
{PHYSICS}"""
            run(args.mallard, toml, work, ranks=args.ranks, mpirun=args.mpirun, name=prefix + ".toml")
            cases.append((partitioner, box, [polygons(work / f"{prefix}_000000_p{r:04d}.vtu")[0]
                                              for r in range(args.ranks)]))
    return cases


def partitions(P, args):
    cases = partition_data(args)
    palette = [P["teal"], P["flame"], P["blue"], P["plum"], P["green"], P["gold"], P["red"], P["muted"]]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0), gridspec_kw=dict(wspace=0.08, hspace=0.18))
    for (partitioner, box, pieces), ax in zip(cases, axes.T.flat):
        for r, polys in enumerate(pieces):
            ax.add_collection(PolyCollection(polys, facecolors=palette[r % len(palette)], alpha=0.6,
                                             edgecolors=P["bg"], linewidths=0.25, rasterized=True))
        bare(ax)
        if box:
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 1)
            ax.set_title(f'partitioner = "{partitioner}"', fontsize=12, family="monospace")
        else:
            ax.set_xlim(-6.2, 6.2)
            ax.set_ylim(-6.2, 6.2)
    return fig


# --------------------------------------------------------------- ignition

@functools.cache
def ignition_data(args):
    work = Path(tempfile.mkdtemp())
    text = (ROOT / "examples" / "h2_ignition" / "input.toml").read_text()
    text = text.replace("../../mechanisms/", str(ROOT / "mechanisms") + "/")
    reactor = Path(args.mallard).parent / "MallardReactor"
    run(reactor, text, work)
    return list(csv.DictReader(open(work / "reactor.csv")))


def ignition(P, args):
    rows = ignition_data(args)
    t = np.array([float(r["t"]) for r in rows]) * 1e6
    T = np.array([float(r["T"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(t, T, color=P["fg"], lw=2.6, label="T")
    ax.set_xlabel(r"t [$\mu$s]")
    ax.set_ylabel("T [K] (thick line)")
    ax2 = ax.twinx()
    for sp, col in (("H2", P["blue"]), ("O2", P["teal"]), ("H2O", P["plum"]), ("OH", P["gold"])):
        y = np.array([float(r["Y_" + sp]) for r in rows])
        ax2.plot(t, y * (10 if sp == "OH" else 1), color=col, lw=1.6, label=sp + (" x 10" if sp == "OH" else ""))
    ax2.set_ylabel("mass fraction")
    ax2.spines["right"].set_visible(True)
    ok = np.diff(t) > 0
    rate = np.diff(T)[ok] / np.diff(t)[ok]
    t_ig = 0.5 * (t[:-1] + t[1:])[ok][np.argmax(rate)]
    ax.axvline(t_ig, color=P["muted"], lw=1, ls=":")
    label(ax, t_ig + 4, 2050, f"ignition delay\n{t_ig:.0f} $\\mu$s", P, ha="left", size=11, color=P["muted"])
    ax2.legend(loc="center right", fontsize=11)
    ax.set_xlim(0, 100)
    return fig


# ------------------------------------------------------------------ remap

REMAP_FIELD = """
[initialize]
type = "analytical"
rho = "1.0 + 0.6 * exp(-30 * ((x - 0.42)^2 + (y - 0.58)^2)) + (y > 0.22 + 0.35 * x ? 0.5 : 0.0)"
u = ["0.0", "0.0"]
p = "1.0"
"""
REMAP_BCS = "".join(f'[[boundaries]]\nname = "{z}"\ntype = "symmetry"\n' for z in ("left", "right", "top", "bottom"))


@functools.cache
def remap_data(args):
    """A field on 16 x 16 quads (source), remapped to 2 x 37 x 37 triangles (target)."""
    work = Path(tempfile.mkdtemp())
    source_mesh = 'type = "cartesian"\nNx = 16\nNy = 16\n'
    run(args.mallard, f"""
[run]
n_steps = 1
cfl = 0.5
[mesh]
{source_mesh}{REMAP_FIELD}{REMAP_BCS}
[[write_data]]
prefix = "source"
format = "vtu"
interval = 1
variables = ["RHO"]
[[write_data]]
prefix = "source"
format = "restart"
interval = 1
{PHYSICS}""", work, name="source.toml")
    run(args.mallard, f"""
[run]
n_steps = 1
cfl = 0.5
[mesh]
type = "cartesian_tri"
Nx = 37
Ny = 37
[initialize]
type = "remap"
file = "source_000000.restart"
degree = 3
[initialize.mesh]
{source_mesh}{REMAP_BCS}
[[write_data]]
prefix = "target"
format = "vtu"
interval = 1
variables = ["RHO"]
{PHYSICS}""", work, name="target.toml")
    source, sdata = polygons(work / "source_000000.vtu")
    target, tdata = polygons(work / "target_000000.vtu")
    return source, np.asarray(sdata["RHO"]), target, np.asarray(tdata["RHO"])


def _clip(subject, a, b):
    """Part of convex polygon subject left of the directed line a -> b."""
    out = []
    for i, p in enumerate(subject):
        q = subject[(i + 1) % len(subject)]
        sp = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        sq = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
        if sp >= 0:
            out.append(p)
        if (sp >= 0) != (sq >= 0):
            out.append(p + sp / (sp - sq) * (q - p))
    return out


def _area(p):
    return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])


def _intersect(target, source):
    s = source if _area(source) > 0 else source[::-1]
    piece = list(target)
    for i in range(len(s)):
        if not piece:
            break
        piece = _clip(piece, s[i], s[(i + 1) % len(s)])
    return np.array(piece)


def remap_supermesh(P, args):
    source, _, target, _ = remap_data(args)
    window = (0.30, 0.58, 0.30, 0.58)

    def inside(c):
        return (c[:, 0].max() > window[0] and c[:, 0].min() < window[1] and c[:, 1].max() > window[2]
                and c[:, 1].min() < window[3])

    def pieces(t):
        return [q for q in (_intersect(t, s) for s in source) if len(q) >= 3 and abs(_area(q)) > 0.02 * abs(_area(t))]

    near = [t for t in target if inside(t)]
    center = np.array([0.5 * (window[0] + window[1]), 0.5 * (window[2] + window[3])])
    chosen = max(near, key=lambda t: (len(pieces(t)), -np.linalg.norm(t.mean(axis=0) - center)))
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    colors = [P["teal"], P["flame"], P["plum"], P["gold"]]
    for k, q in enumerate(pieces(chosen)):
        ax.add_patch(Polygon(q, closed=True, facecolor=colors[k % 4], alpha=0.75, edgecolor="none"))
    ax.add_collection(PolyCollection(near, facecolors="none", edgecolors=P["mesh"], linewidths=0.8))
    ax.add_collection(PolyCollection([s for s in source if inside(s)], facecolors="none", edgecolors=P["blue"],
                                     linewidths=2.0))
    ax.add_patch(Polygon(chosen, closed=True, facecolor="none", edgecolor=P["fg"], lw=1.8))
    bare(ax)
    c0 = chosen.mean(axis=0)
    ax.set_xlim(c0[0] - 0.07, c0[0] + 0.07)
    ax.set_ylim(c0[1] - 0.07, c0[1] + 0.07)
    label(ax, 0.5, -0.04, "source quadrilaterals (blue), target triangles (gray):\n"
          "one target cell in its pieces, one per source cell it overlaps", P, size=11, va="top",
          transform=ax.transAxes)
    return fig


def remap_fields(P, args):
    source, rho_s, target, rho_t = remap_data(args)
    lo, hi = min(rho_s.min(), rho_t.min()), max(rho_s.max(), rho_t.max())
    fig = plt.figure(figsize=(10.4, 3.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.12)
    y_cut = 0.58
    for k, (cells, rho, title) in enumerate([(source, rho_s, "before: 256 quadrilaterals"),
                                             (target, rho_t, "after: 2,738 triangles")]):
        ax = fig.add_subplot(grid[k])
        ax.add_collection(PolyCollection(cells, array=rho, cmap=P["cmap"], clim=(lo, hi), edgecolors="none",
                                         antialiaseds=False))
        ax.plot([0, 1], [y_cut, y_cut], color=P["bg"], lw=1.4)
        bare(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=12)
    ax = fig.add_subplot(grid[2])
    x = np.linspace(0.0005, 0.9995, 2000)
    for cells, rho, name, color, lw in [(source, rho_s, "before", P["blue"], 1.4), (target, rho_t, "after", P["flame"], 2.0)]:
        values = np.full(x.size, np.nan)
        for c, v in zip(cells, rho):
            if c[:, 1].min() <= y_cut <= c[:, 1].max():
                path = matplotlib.path.Path(c)
                hit = path.contains_points(np.column_stack([x, np.full(x.size, y_cut)]))
                values[hit] = v
        ax.plot(x, values, color=color, lw=lw, label=name)
    ax.set_xlabel(f"x at y = {y_cut}")
    ax.set_ylabel("density")
    ax.legend(loc="upper right")
    return fig


class Args(tuple):
    """Hashable options, so the runs are cached between the light and dark figures."""

    def __new__(cls, mallard, mpirun, ranks):
        return super().__new__(cls, (mallard, mpirun, ranks))

    mallard = property(lambda self: self[0])
    mpirun = property(lambda self: self[1])
    ranks = property(lambda self: self[2])


FIGURES = {"teno_stencils": teno_stencils, "sod_schemes": sod_schemes, "partitions": partitions,
           "ignition": ignition, "remap-supermesh": remap_supermesh, "remap-fields": remap_fields}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="2D Mallard executable (MPI build for partitions)")
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--ranks", type=int, default=6)
    ap.add_argument("names", nargs="*")
    ns = ap.parse_args()
    args = Args(Path(ns.mallard).resolve(), ns.mpirun, ns.ranks)
    for name, draw in FIGURES.items():
        if not ns.names or name in ns.names:
            figstyle.save(name, lambda P, d=draw: d(P, args))


if __name__ == "__main__":
    main()
