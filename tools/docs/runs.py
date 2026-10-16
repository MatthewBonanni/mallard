"""Figures of the user guide drawn from short Mallard runs (seconds each, on a
CPU build): TENO-E stencils read back from a stencil cache, the Sod shock
tube with each reconstruction (through a line probe), MPI partitions, a
MallardReactor ignition, a sector against the full annulus (rotational
periodicity), and the mesh-import fixtures as Mallard reads them
(through mallard-mesh-convert, which needs h5py and an HDF5 and CGNS build).

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
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Polygon
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
    return compare(str(work / "solut" / "sector_000001.vtu"), str(work / "solut" / "annulus_000001.vtu"), 4)


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


# ------------------------------------------------------------- mesh import

IMPORTS = (("Gmsh", "cyl2d.msh"), ("CGNS", "cyl2d.cgns"), ("Fluent", "cyl2d_fluent.msh"),
           ("OpenFOAM", "cyl2d_foam"))
TRANSFORM = "[mesh.transform]\nscale = [1.5, 1]\nrotate = 30\ntranslate = [0.5, 0.25]\n"


@functools.cache
def mesh_read(args, name, transform=""):
    """Mallard's read of a test/data/meshes fixture (through mallard-mesh-convert, which needs an HDF5 and,
    for CGNS, a CGNS build): nodes, cells and boundary faces by zone."""
    import h5py
    work = Path(tempfile.mkdtemp())
    (work / "in.toml").write_text('[mesh]\ntype = "file"\nfilename = "%s"\n%s'
                                  % (ROOT / "test" / "data" / "meshes" / name, transform))
    subprocess.run([str(Path(args.mallard).parent / "mallard-mesh-convert"), "in.toml", "out.h5"], cwd=work,
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
    colours = {"inlet": P["blue"], "outlet": P["red"], "walls": P["green"], "cylinder": P["flame"]}
    ax.add_collection(PolyCollection(cells, facecolors=P["panel"], edgecolors=P["mesh"], linewidths=0.6))
    for name, segs in zones.items():
        ax.add_collection(LineCollection(segs, colors=colours.get(name, P["plum"]), linewidths=3.0,
                                         capstyle="round"))
        if not labels or not segs:
            continue
        if name == "walls":
            segs = [s for s in segs if s[:, 1].mean() > np.median(x[:, 1])] or segs
        mid = np.concatenate(segs).mean(axis=0)
        label(ax, *mid, name, P, color=colours.get(name, P["plum"]), size=11, box=True, fontweight="bold")
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


class Args(tuple):
    """Hashable options, so the runs are cached between the light and dark figures."""

    def __new__(cls, mallard, mpirun, ranks):
        return super().__new__(cls, (mallard, mpirun, ranks))

    mallard = property(lambda self: self[0])
    mpirun = property(lambda self: self[1])
    ranks = property(lambda self: self[2])


FIGURES = {"teno_stencils": teno_stencils, "sod_schemes": sod_schemes, "partitions": partitions,
           "ignition": ignition, "mesh_import": mesh_import, "mesh_transform": mesh_transform,
           "rotational_sector": rotational_sector}


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
