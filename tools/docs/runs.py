"""Figures of the user guide drawn from short Mallard runs (seconds each, on a
CPU build): TENO-E stencils read back from mallard check's stencil file, the Sod shock
tube with each reconstruction (through a line probe), MPI partitions, a
mallard reactor ignition, a sector against the full annulus (rotational
periodicity), the mesh-import fixtures as Mallard reads them
(through mallard mesh convert, which needs h5py and an HDF5 and CGNS build), a
remap from quadrilaterals to triangles, the mesh quality from mallard check,
the halo layers of one rank, and a grid-convergence study through mallard post
convergence.

    python tools/docs/runs.py --mallard build/src/mallard [--mpirun mpirun] [names...]

Partitions and halos need an MPI build.
"""
import argparse
import csv
import functools
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Polygon
import matplotlib.path
import matplotlib.pyplot as plt

import figstyle
from figstyle import bare, label

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools" / "mesh"), str(ROOT / "examples" / "sod"),
                str(ROOT / "examples" / "rotational_sector")]
from mallard.stencils import read_stencils  # noqa: E402
from mallard.vtu import read_vtu_cells, snapshot  # noqa: E402
from plot_mesh_quality import LABELS, draw_quality, read_quality  # noqa: E402
from plot_sod import exact_sod  # noqa: E402

PHYSICS = '\n[physics]\ntype = "euler"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n'


def run(binary, toml, workdir, ranks=1, mpirun="mpirun", name="input.toml", command=()):
    path = Path(workdir) / name
    path.write_text(toml)
    cmd = [str(binary), *command, "-i", name, "--kokkos-num-threads=2"]
    if ranks > 1:
        cmd = [mpirun, "-n", str(ranks), "--oversubscribe", *cmd]
    subprocess.run(cmd, cwd=workdir, check=True, stdout=subprocess.DEVNULL)


def polygons(path):
    pts, conn, offs, types, data = read_vtu_cells(str(path))
    starts = np.concatenate([[0], offs[:-1]])
    return [pts[conn[a:b], :2] for a, b in zip(starts, offs)], data


# ------------------------------------------------------------- TENO stencils

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
{PHYSICS}"""
    (work / "input.toml").write_text(toml)
    subprocess.run([str(args.mallard), "check", "input.toml", "--mesh-fields", "mesh", "--stencils", "all",
                    "--kokkos-num-threads=2"], cwd=work, check=True, stdout=subprocess.DEVNULL)
    polys, _ = read_quality(snapshot(str(work / "mesh_000000")))
    return polys, read_stencils(work / "mesh_stencils.h5")


def teno_stencils(P, args):
    polys, stencils = teno_data(args)
    target = min(range(len(polys)), key=lambda c: np.linalg.norm(polys[c].mean(axis=0) - [0.5, 0.5]))
    s = stencils.stencil(target)
    central = s["cell"][s["stencil"] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw=dict(wspace=0.06))
    colors = [P["teal"], P["flame"], P["blue"], P["plum"]]
    for k, ax in enumerate(axes):
        for poly in polys:
            ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor=P["mesh"], lw=0.8))
        if k == 0:
            for c in central:
                ax.add_patch(Polygon(polys[c], closed=True, facecolor=P["teal"], alpha=0.35, edgecolor=P["teal"],
                                     lw=0.8))
            ax.set_title(f"central stencil: {len(central)} cells", fontsize=12)
        else:
            sectors = stencils.sectors(target)
            for f, cells in sectors.items():
                for c in cells:
                    if c == target:
                        continue
                    ax.add_patch(Polygon(polys[c], closed=True, facecolor=colors[f + 1], alpha=0.38,
                                         edgecolor=colors[f + 1], lw=0.8))
            n_sectors = len(sectors)
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
fields = ["RHO"]
start = [{0.5 / nx}, 0.01]
end = [{1 - 0.5 / nx}, 0.01]
n_points = {nx}
interval = 1
"""
        run(args.mallard, text, work)
        rows = list(csv.DictReader(line for line in open(work / "probe_line.csv") if not line.startswith("#")))
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
    subprocess.run([sys.executable, str(ROOT / "tools" / "mesh" / "make_cylinder_mesh.py"), str(work / "cyl.msh"),
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
fields = ["RHO"]
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
    run(args.mallard, text, work, command=["reactor"])
    return list(csv.DictReader(line for line in open(work / "reactor.csv") if not line.startswith("#")))


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
    label(ax, t_ig - 3, 1900, f"ignition delay\n{t_ig:.0f} $\\mu$s", P, ha="right", size=11, color=P["muted"])
    ax2.legend(loc="center right", fontsize=11)
    ax.set_xlim(0, 100)
    return fig


# ------------------------------------------------------- rotational periodicity

@functools.cache
def rotational_data(args):
    """The 2D example's sector and full annulus, run to t = 0.25, compared copy by copy."""
    from rotational_sector import compare
    work = Path(tempfile.mkdtemp())
    for name in ("sector", "annulus"):
        text = (ROOT / "examples" / "rotational_sector" / f"{name}.toml").read_text()
        text = text.replace("t_stop = 0.5", "t_stop = 0.25").replace("time_interval = 0.5", "time_interval = 0.25")
        run(args.mallard, text, work, name=f"{name}.toml")
    return compare(snapshot(str(work / "solut" / "sector_000001")), snapshot(str(work / "solut" / "annulus_000001")), 4)


def rotational_sector(P, args):
    polys, fields, copies, d, own = rotational_data(args)
    polys = [p[:, :2] for p in polys]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw=dict(wspace=0.32))
    rho = PolyCollection(polys, array=fields["RHO"], cmap=P["cmap"], edgecolors="face", linewidths=0.2,
                         rasterized=True)
    axes[0].add_collection(rho)
    err = PolyCollection(polys, array=np.log10(np.maximum(d, 1e-17)), cmap="magma", edgecolors="face", linewidths=0.2,
                         rasterized=True)
    axes[1].add_collection(err)
    for ax, coll, title in ((axes[0], rho, "full annulus: density"),
                            (axes[1], err, "log$_{10}$ |sector $-$ annulus|")):
        bare(ax)
        ax.autoscale_view()
        ax.set_title(title, fontsize=12, color=P["muted"])
        cb = fig.colorbar(coll, ax=ax, shrink=0.75)
        cb.outline.set_visible(False)
    for t in (0.0, np.pi / 2):
        axes[0].plot([0.95 * np.cos(t), 2.05 * np.cos(t)], [0.95 * np.sin(t), 2.05 * np.sin(t)], color=P["flame"],
                     lw=2)
    label(axes[1], 0.5, -0.04, f"largest {d.max():.0e}; annulus against itself {own.max():.0e}", P, size=11,
          color=P["muted"], va="top", transform=axes[1].transAxes)
    return fig


# ------------------------------------------------------------- mesh quality

@functools.cache
def mesh_quality_data(args):
    """mallard check of examples/mallard on a coarse mesh (make_mallard_mesh.py needs the gmsh module)."""
    work = Path(tempfile.mkdtemp())
    subprocess.run([sys.executable, str(ROOT / "examples" / "mallard" / "make_mallard_mesh.py"), str(work / "mallard.msh"),
                    "--h-wall", "0.02", "--h-far", "0.1"], check=True, stdout=subprocess.DEVNULL)
    (work / "input.toml").write_text((ROOT / "examples" / "mallard" / "input.toml").read_text())
    subprocess.run([str(args.mallard), "check", "input.toml", "--mesh-fields", "quality",
                    "--kokkos-num-threads=2"], cwd=work, check=True, stdout=subprocess.DEVNULL)
    return read_quality(snapshot(str(work / "quality_000000")))


def mesh_quality(P, args):
    polys, arrays = mesh_quality_data(args)
    xlim, ylim = (-0.3, 1.6), (-0.75, 0.75)
    centers = np.array([p.mean(axis=0) for p in polys])
    shown = ((centers[:, 0] > xlim[0] + 0.03) & (centers[:, 0] < xlim[1] - 0.03)
             & (centers[:, 1] > ylim[0] + 0.03) & (centers[:, 1] < ylim[1] - 0.03))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), layout="constrained")
    for ax, field in zip(axes, ("NON_ORTHOGONALITY", "SKEWNESS")):
        values = arrays[field]
        cells = draw_quality(ax, polys, values, 6, P["red"], P["mesh"], cmap=P["cmap"], fontsize=11, among=shown)
        bar = fig.colorbar(cells, ax=ax, fraction=0.04, pad=0.02)
        bar.set_label(LABELS[field])
        bar.outline.set_edgecolor(P["muted"])
        ax.set_title(f"max {values.max():.3g}, mean {values.mean():.3g}; worst 6 in view circled", fontsize=12)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axis("off")
    return fig


# ------------------------------------------------------------- mesh import

IMPORTS = (("Gmsh", "cyl2d.msh"), ("CGNS", "cyl2d.cgns"), ("Fluent", "cyl2d_fluent.msh"),
           ("OpenFOAM", "cyl2d_foam"))
TRANSFORM = "[mesh.transform]\nscale = [1.5, 1]\nrotate = 30\ntranslate = [0.5, 0.25]\n"


@functools.cache
def mesh_read(args, name, transform=""):
    """Mallard's read of a test/data/meshes fixture (through mallard mesh convert, which needs an HDF5 and,
    for CGNS, a CGNS build): nodes, cells and boundary faces by zone."""
    import h5py
    work = Path(tempfile.mkdtemp())
    (work / "in.toml").write_text('[mesh]\ntype = "file"\nfilename = "%s"\n%s'
                                  % (ROOT / "test" / "data" / "meshes" / name, transform))
    subprocess.run([str(args.mallard), "mesh", "convert", "in.toml", "out.h5"], cwd=work,
                   check=True, stdout=subprocess.DEVNULL)
    with h5py.File(work / "out.h5", "r") as h:
        x = h["nodes/coordinates"][:]
        co, cn = h["cells/offsets"][:], h["cells/nodes"][:]
        fo, fn, fz = h["boundary/offsets"][:], h["boundary/nodes"][:], h["boundary/zone"][:]
        names = [z.decode() if isinstance(z, bytes) else str(z) for z in h["boundary"].attrs["zone_names"]]
    cells = [x[cn[co[c]:co[c + 1]]] for c in range(len(co) - 1)]
    zones = {n: [x[fn[fo[f]:fo[f + 1]]] for f in range(len(fo) - 1) if fz[f] == z] for z, n in enumerate(names)}
    return x, cells, zones


def draw_import(ax, P, mesh, labels=True):
    x, cells, zones = mesh
    colors = {"inlet": P["blue"], "outlet": P["red"], "walls": P["green"], "cylinder": P["flame"]}
    ax.add_collection(PolyCollection(cells, facecolors=P["panel"], edgecolors=P["mesh"], linewidths=0.6))
    for name, segs in zones.items():
        ax.add_collection(LineCollection(segs, colors=colors.get(name, P["plum"]), linewidths=3.0,
                                         capstyle="round"))
        if not labels or not segs:
            continue
        if name == "walls":
            segs = [s for s in segs if s[:, 1].mean() > np.median(x[:, 1])] or segs
        mid = np.concatenate(segs).mean(axis=0)
        label(ax, *mid, name, P, color=colors.get(name, P["plum"]), size=11, box=True, fontweight="bold")
    ax.autoscale_view()


def mesh_import(P, args):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.4), gridspec_kw=dict(wspace=0.06, hspace=0.22))
    for (fmt, name), ax in zip(IMPORTS, axes.flat):
        mesh = mesh_read(args, name)
        draw_import(ax, P, mesh)
        bare(ax)
        ax.set_title(f"{fmt}: {name}", fontsize=12)
    return fig


def mesh_transform(P, args):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.08))
    meshes = (mesh_read(args, "cyl2d.msh"), mesh_read(args, "cyl2d.msh", TRANSFORM))
    for ax, mesh, title in zip(axes, meshes, ("as read", "scale = [1.5, 1], rotate = 30, translate = [0.5, 0.25]")):
        draw_import(ax, P, mesh)
        ax.plot(0, 0, "+", color=P["fg"], ms=12, mew=1.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.grid(True, color=P["faint"], lw=0.8)
    xy = np.concatenate([m[0] for m in meshes])
    axes[0].set_xlim(xy[:, 0].min() - 0.15, xy[:, 0].max() + 0.15)
    axes[0].set_ylim(xy[:, 1].min() - 0.15, xy[:, 1].max() + 0.15)
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
fields = ["RHO"]
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
fields = ["RHO"]
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


# ------------------------------------------------------------------- halo

HALO_RANKS = 3


@functools.cache
def halo_data(args):
    """Owned cells of every rank and Mallard's halo depth, for MUSCL and TENO order 5."""
    work = Path(tempfile.mkdtemp())
    cases = []
    for name, numerics in (("MUSCL", 'type = "MUSCL"'), ("TENO", 'type = "TENO"\norder = 5')):
        bcs = "".join(f'[[boundaries]]\nname = "{z}"\ntype = "symmetry"\n' for z in ("left", "right", "top", "bottom"))
        toml = f"""
[run]
n_steps = 1
cfl = 0.5
[mesh]
type = "cartesian_tri"
Nx = 36
Ny = 18
Lx = 2.0
Ly = 1.0
[initialize]
type = "constant"
u = [0.0, 0.0]
p = 1.0
T = 1.0
{bcs}
[numerics.face_reconstruction]
{numerics}
[[write_data]]
prefix = "{name}"
format = "vtu"
interval = 1
fields = ["RHO"]
{PHYSICS}"""
        (work / "in.toml").write_text(toml)
        out = subprocess.run([args.mpirun, "-n", str(HALO_RANKS), "--oversubscribe", str(args.mallard), "-i", "in.toml",
                              "--kokkos-num-threads=1"], cwd=work, check=True, capture_output=True, text=True).stdout
        quantity = {}
        for line in out.splitlines():
            for key in ("halo layers", "halo cells", "halo layer-1 cells"):
                if line.strip().startswith(key + " "):
                    quantity[key] = int(line.split()[len(key.split())].replace(",", ""))
        ranks = [polygons(work / f"{name}_000000_p{r:04d}.vtu")[0] for r in range(HALO_RANKS)]
        cases.append((name, quantity, ranks))
    return cases


def halo_layers(ranks, rank, depth):
    """Vertex-neighbor layers around a rank's owned cells, as design/mpi.md builds the halo."""
    cells = [c for polys in ranks for c in polys]
    owner = np.concatenate([np.full(len(polys), r) for r, polys in enumerate(ranks)])
    by_node = {}
    for c, poly in enumerate(cells):
        for p in np.round(poly, 9):
            by_node.setdefault(tuple(p), []).append(c)
    neighbors = [set() for _ in cells]
    for around in by_node.values():
        for c in around:
            neighbors[c].update(around)
    layer = {c: 0 for c in np.flatnonzero(owner == rank)}
    front = list(layer)
    for k in range(1, depth + 1):
        reached = []
        for c in front:
            for n in neighbors[c]:
                if n not in layer:
                    layer[n] = k
                    reached.append(n)
        front = reached
    return cells, owner, layer


def halo(P, args):
    cases = halo_data(args)
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 8.0), gridspec_kw=dict(hspace=0.22))
    for (name, quantity, ranks), ax in zip(cases, axes):
        depth = quantity["halo layers"]
        counts = [halo_layers(ranks, r, depth)[2] for r in range(HALO_RANKS)]
        halo_cells = [sum(1 for v in c.values() if v > 0) for c in counts]
        layer1 = [sum(1 for v in c.values() if v == 1) for c in counts]
        assert max(halo_cells) == quantity["halo cells"] and max(layer1) == quantity["halo layer-1 cells"], \
            (name, halo_cells, layer1, quantity)
        cells, owner, layer = halo_layers(ranks, 1, depth)
        ax.add_collection(PolyCollection(cells, facecolors="none", edgecolors=P["mesh"], linewidths=0.4,
                                         rasterized=True))
        ax.add_collection(PolyCollection([cells[c] for c, k in layer.items() if k == 0], facecolors=P["teal"],
                                         alpha=0.55, edgecolors=P["bg"], linewidths=0.4, rasterized=True))
        for k in range(1, depth + 1):
            ax.add_collection(PolyCollection([cells[c] for c, v in layer.items() if v == k], facecolors=P["flame"],
                                             alpha=0.75 * (1 - 0.8 * (k - 1) / max(depth - 1, 1)) + 0.1,
                                             edgecolors=P["bg"], linewidths=0.4, rasterized=True))
        bare(ax)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        what = "1 layer" if depth == 1 else f"{depth} layers"
        recon = 'type = "MUSCL"' if name == "MUSCL" else 'type = "TENO", order = 5'
        ax.set_title(f"{recon}: halo of {what}, {quantity['halo cells']} cells", fontsize=12)
    handles = [Polygon([[0, 0]], facecolor=P["teal"], alpha=0.55, label="owned by rank 1"),
               Polygon([[0, 0]], facecolor=P["flame"], alpha=0.85, label="halo layer 1"),
               Polygon([[0, 0]], facecolor=P["flame"], alpha=0.25, label="deeper halo layers")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=3, fontsize=11.5)
    return fig


# --------------------------------------------------------- grid convergence

CONVERGENCE_EXACT = "1 + 0.2 * sin(2 * pi * (x - t)) * sin(2 * pi * (y - 0.5 * t))"


@functools.cache
def convergence_data(args):
    """The density wave of docs/guide/convergence.md on 16^2, 32^2 and 64^2 cells, through mallard post convergence."""
    import json
    work = Path(tempfile.mkdtemp())
    for n in (16, 32, 64):
        (work / f"n{n}").mkdir()
        run(args.mallard, f"""
[run]
t_stop = 0.05
cfl = 0.5
[mesh]
type = "cartesian"
Nx = {n}
Ny = {n}
[[periodic]]
zones = ["left", "right"]
[[periodic]]
zones = ["bottom", "top"]
[initialize]
type = "analytical"
rho = "1.0 + 0.2 * sin(2 * pi * x) * sin(2 * pi * y)"
u = ["1.0", "0.5"]
p = "1.0"
[numerics.face_reconstruction]
type = "MUSCL"
limiter = "none"
[integrals]
interval = 1
[[write_data]]
prefix = "restart"
format = "restart"
time_interval = 0.05
{PHYSICS}""", work / f"n{n}")
    report = {}
    for key, extra in (("quantity", ["--quantity", "density_squared"]), ("field", ["--field", "RHO", "--exact", CONVERGENCE_EXACT])):
        out = subprocess.run([str(args.mallard), "post", "convergence", "n16", "n32", "n64", *extra, "--format", "json"],
                             cwd=work, capture_output=True, text=True, check=True).stdout
        report[key] = json.loads(out)
    return report


def grid_convergence(P, args):
    report = convergence_data(args)
    q = report["quantity"]["quantities"][0]
    e = q["estimate"]
    h = np.array([r["h"] for r in report["quantity"]["runs"]])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.3), gridspec_kw=dict(wspace=0.38))

    p = e["order"]
    x = (h / h.min()) ** p
    phi, ext = np.array(q["values"]), e["extrapolated"]
    band = e["gci"] * phi[-1]
    inset = ax.inset_axes([0.4, 0.55, 0.58, 0.42])
    for a, big in ((ax, True), (inset, False)):
        a.axhline(1.01, color=P["muted"], lw=1.0, ls=(0, (4, 2)), zorder=0)
        a.plot([0, x[0]], [ext, phi[0]], color=P["teal"], lw=1.0, ls=":")
        a.errorbar([1.0], [phi[-1]], yerr=[[band], [band]], color=P["teal"], capsize=4, lw=1.2, zorder=2)
        a.plot(x, phi, "o", color=P["teal"], ms=6, zorder=3)
        a.plot([0], [ext], "o", mfc=P["bg"], mec=P["teal"], mew=1.5, ms=7, zorder=3, clip_on=False)
        a.ticklabel_format(axis="y", useOffset=False)
    for xi, yi, run_name in zip(x[:2], phi[:2], ("16²", "32²")):
        figstyle.label(ax, xi + 0.6, yi, run_name, P, size=11, ha="left")
    figstyle.label(ax, 2.2, 1.01, "exact, 1.01", P, color=P["muted"], size=11, ha="left", va="bottom")
    ax.indicate_inset_zoom(inset, edgecolor=P["muted"])
    ax.set_xlim(0, x.max() * 1.15)
    ax.set_ylim(phi[0] - 3e-5, 1.01 + 2.6e-4)
    ax.set_xlabel(rf"$(h/h_1)^p$, observed $p$ = {p:.2f}")
    ax.set_ylabel(r"$\int\rho^2\,\mathrm{d}V$")
    ax.set_title("density_squared: extrapolation and GCI", fontsize=13)
    inset.set_xlim(-0.15, 1.9)
    inset.set_ylim(phi[-1] - 1.5 * band, 1.01 + 1.2 * band)
    inset.set_facecolor(P["bg"])
    inset.tick_params(labelsize=9.5)
    inset.set_xticks([0, 1])
    inset.set_yticks([])
    figstyle.label(inset, 1.15, phi[-1], f"64², GCI {100 * e['gci']:.5f}%", P, color=P["teal"], size=10.5, ha="left")
    figstyle.label(inset, 0.12, ext - 0.25 * band, rf"$\phi_\mathrm{{ext}}$", P, color=P["teal"], size=11, ha="left",
                   va="top")
    for side in ("top", "right"):
        inset.spines[side].set_visible(True)

    norms = report["field"]["field"]["norms"]
    cells = np.array([r["cells"] for r in report["field"]["runs"]])
    colors = {"L1": P["teal"], "L2": P["blue"], "Linf": P["plum"]}
    for n in norms:
        bx.loglog(cells, n["errors"], "o-", color=colors[n["norm"]], ms=6)
        shift = {"L1": 0.85, "L2": 1.18, "Linf": 1.0}[n["norm"]]
        figstyle.label(bx, cells[-1] * 1.15, n["errors"][-1] * shift, n["norm"].replace("inf", r"$_\infty$").replace("L1", "L$_1$")
                       .replace("L2", "L$_2$"), P, color=colors[n["norm"]], size=11.5, ha="left")
    ref = cells.astype(float)
    bx.loglog(ref, 0.6 * norms[0]["errors"][0] * (ref / ref[0]) ** -1.0, color=P["muted"], lw=1.0, ls=":")
    figstyle.label(bx, ref[1], 0.6 * norms[0]["errors"][0] * (ref[1] / ref[0]) ** -1.0 * 0.55, "order 2", P,
                   color=P["muted"], size=10.5, rotation=-28)
    bx.set_xticks(cells, [f"{c:,}" for c in cells])
    bx.minorticks_off()
    bx.set_xlim(cells[0] / 1.4, cells[-1] * 2.0)
    bx.set_xlabel("cells")
    bx.set_ylabel("error of RHO")
    bx.set_title("Error against the exact solution", fontsize=13)
    return fig


class Args(tuple):
    """Hashable options, so the runs are cached between the light and dark figures."""

    def __new__(cls, mallard, mpirun, ranks):
        return super().__new__(cls, (mallard, mpirun, ranks))

    mallard = property(lambda self: self[0])
    mpirun = property(lambda self: self[1])
    ranks = property(lambda self: self[2])


FIGURES = {"teno_stencils": teno_stencils, "sod_schemes": sod_schemes, "partitions": partitions,
           "ignition": ignition, "mesh_import": mesh_import, "mesh_transform": mesh_transform,
           "remap-supermesh": remap_supermesh, "remap-fields": remap_fields, "mesh_quality": mesh_quality,
           "rotational_sector": rotational_sector, "halo": halo, "grid_convergence": grid_convergence}


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
