"""Figures of the tools page (tools/README.md): what each tool produces, drawn
from its own output, light and dark, into docs/images/tools/.

    python tools/docs/tools.py [--mallard build/src/mallard] [--work DIR] [names...]

Every figure runs the tool it shows (as a command, like a user would) and
draws what it wrote. Needs numpy, scipy, matplotlib, gmsh and cantera
(pip install gmsh cantera); the figures from Mallard runs (plot_vtu.py,
compare_png.py, animate.py, animate_cartesian.py) need --mallard, a 2D
build, and take about a minute on a few CPU threads. --work keeps the
tools' output between invocations (default: a temporary directory).
"""
import argparse
import csv
import re
import struct
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

import figstyle
import meshes
from figstyle import bare, label

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
OUT = ROOT / "docs" / "images" / "tools"
sys.path.insert(0, str(TOOLS))
from mallard_vtu import read_vtu  # noqa: E402

WORK = None
MALLARD = None


def tool(name, *args, cwd=None):
    """Run tools/NAME with ARGS in the work directory; returns its stdout."""
    cmd = [sys.executable, str(TOOLS / name), *map(str, args)]
    return subprocess.run(cmd, cwd=cwd or WORK, check=True, capture_output=True, text=True).stdout


def once(path, make):
    """WORK/path, made by make(path) unless it is there already."""
    p = WORK / path
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        make(p)
    return p


def read_restart(path):
    """Fields by name of a restart file of version 2 or 3 (double precision)."""
    with open(path, "rb") as f:
        f.read(16)
        version, _ = struct.unpack("<II", f.read(8))
        n_cells, n_vars, _ = struct.unpack("<QQQ", f.read(24))
        f.read(8)

        def name():
            (n,) = struct.unpack("<I", f.read(4))
            return f.read(n).decode()

        names = [name() for _ in range(n_vars)]
        if version >= 3:
            (na,) = struct.unpack("<Q", f.read(8))
            for _ in range(na):
                name()
                f.read(8)
        if version >= 4:
            f.read(12)
        data = np.fromfile(f, dtype="<f8", count=n_vars * n_cells).reshape(n_vars, n_cells)
    return dict(zip(names, data))


def edges(ax, mesh, P, lw=0.3, color=None, rasterized=True):
    ax.add_collection(LineCollection(mesh.edges(), colors=color or P["mesh"], linewidths=lw, rasterized=rasterized))


def zones(ax, mesh, P, colors, lw=1.6):
    for z, c in colors.items():
        seg = mesh.zone_segments(z)
        if len(seg):
            ax.add_collection(LineCollection(seg, colors=c, linewidths=lw))


# ------------------------------------------------------------------ workflow

def workflow(P):
    stages = [
        ("Mesh", P["teal"], ["make_*_mesh.py", "mesh convert"]),
        ("Set up", P["blue"], ["*_restart.py", "*_reference.py"]),
        ("Run", P["flame"], ["mallard -i", "mallard reactor"]),
        ("Post-process", P["plum"], ["plot_vtu.py", "animate*.py"]),
        ("Compare", P["green"], ["plot_<case>.py", "perf_suite.py"]),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.3))
    w, gap = 2.1, 0.45
    for k, (title, color, lines) in enumerate(stages):
        x = k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, 1.5, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=P["panel"], edgecolor=color, linewidth=2))
        label(ax, x + w / 2, 1.17, title, P, color=color, size=14, weight="bold")
        for j, line in enumerate(lines):
            label(ax, x + w / 2, 0.72 - 0.4 * j, line, P, size=10.5, family="monospace")
        if k:
            ax.annotate("", (x - 0.03, 0.75), (x - gap + 0.03, 0.75),
                        arrowprops=dict(arrowstyle="-|>", color=P["muted"], lw=1.6))
    ax.set_xlim(-0.05, len(stages) * (w + gap) - gap + 0.05)
    ax.set_ylim(-0.1, 1.6)
    bare(ax)
    return fig


# --------------------------------------------------------------------- meshes

def annulus(P):
    variants = [("quadrilaterals", []), ("--triangles", ["--triangles"]), ("--jitter 0.3", ["--jitter", "0.3"])]
    ms = [meshes.read_gmsh22(once(f"annulus{k}.msh", lambda p, a=a: tool(
        "make_annulus_mesh.py", p, "--n-r", 6, "--n-theta", 48, "--r-out", 2, *a))) for k, (_, a) in enumerate(variants)]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (title, _), m in zip(axs, variants, ms):
        edges(ax, m, P, lw=0.6, color=P["muted"])
        zones(ax, m, P, {"inner": P["teal"], "outer": P["flame"]}, lw=2)
        ax.set_title(title, fontsize=12, family="monospace")
        ax.set_xlim(-2.05, 2.05)
        ax.set_ylim(-2.05, 2.05)
        bare(ax)
    axs[0].plot([], [], color=P["teal"], lw=2, label="inner")
    axs[0].plot([], [], color=P["flame"], lw=2, label="outer")
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    return fig


def gmsh_triangles(path, surface=None):
    """Nodes (n, 3) and triangles (m, 3) of a Gmsh file: the 2D cells, or the faces of a physical surface."""
    import gmsh
    gmsh.initialize(["", "-v", "0"])
    try:
        gmsh.open(str(path))
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        index = np.zeros(int(tags.max()) + 1, int)
        index[tags.astype(int)] = np.arange(len(tags))
        xyz = xyz.reshape(-1, 3)
        if surface is None:
            types, _, nodes = gmsh.model.mesh.getElements(2)
            tri = nodes[list(types).index(2)]
        else:
            group = next(t for d, t in gmsh.model.getPhysicalGroups(2)
                         if gmsh.model.getPhysicalName(2, t) == surface)
            tri = np.concatenate([gmsh.model.mesh.getElements(2, e)[2][0]
                                  for e in gmsh.model.getEntitiesForPhysicalGroup(2, group)])
        return xyz, index[tri.astype(int)].reshape(-1, 3)
    finally:
        gmsh.finalize()


def tri_edges(xyz, tri, a=0, b=1):
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    return xyz[e][:, :, [a, b]]


def mallard_mesh(P):
    path = once("duck.msh", lambda p: tool("make_mallard_mesh.py", p, "--h-wall", 0.01, "--h-far", 0.06))
    xyz, tri = gmsh_triangles(path)
    fig, ax = plt.subplots(figsize=(8, 3.9))
    ax.add_collection(LineCollection(tri_edges(xyz, tri), colors=P["muted"], linewidths=0.25, rasterized=True))
    ax.set_xlim(-0.35, 1.45)
    ax.set_ylim(-0.42, 0.5)
    bare(ax)
    ax.set_title("make_mallard_mesh.py duck.msh --h-wall 0.01 --h-far 0.06 (near field)", fontsize=11,
                 family="monospace")
    return fig


def case_meshes(P):
    volvo = once("volvo.msh", lambda p: tool("make_volvo_mesh.py", p, "--ds", 2.5e-3, "--nz", 2))
    sphere = once("sphere.msh", lambda p: tool("make_sphere_mesh.py", p, "--h-wall", 0.04, "--h-shock", 0.045,
                                               "--h-far", 0.25))
    v = meshes.read_gmsh22(volvo, plane_zone="back")
    xyz, tri = gmsh_triangles(sphere, "symmetry_z")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axs[0]
    edges(ax, v, P, lw=0.3, color=P["muted"])
    zones(ax, v, P, {"bluffbody": P["flame"], "inlet": P["teal"], "outlet": P["plum"],
                     "top": P["fg"], "bottom": P["fg"]}, lw=1.8)
    ax.set_xlim(-0.08, 0.2)
    ax.set_ylim(-0.065, 0.065)
    bare(ax)
    ax.set_title("make_volvo_mesh.py: hexahedra around the flameholder", fontsize=11)
    ax = axs[1]
    ax.add_collection(LineCollection(tri_edges(xyz, tri), colors=P["muted"], linewidths=0.25, rasterized=True))
    t = np.linspace(0, np.pi / 2, 100)
    ax.plot(0.5 * np.cos(t + np.pi / 2), 0.5 * np.sin(t + np.pi / 2), color=P["flame"], lw=1.8)
    ax.set_xlim(-1.5, 1.2)
    ax.set_ylim(0, 2.2)
    bare(ax)
    ax.set_title("make_sphere_mesh.py: symmetry plane", fontsize=11)
    return fig


# -------------------------------------------------------------- initial states

def isotropic(P):
    path = once("hit.restart", lambda p: tool("isotropic_turbulence_restart.py", 64, p))
    f = read_restart(path)
    n = 64
    u = np.stack([f[c] / f["RHO"] for c in ("RHOU_X", "RHOU_Y", "RHOU_Z")]).reshape(3, n, n, n)
    k = np.fft.fftfreq(n, 1.0 / n)
    K = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    e = 0.5 * sum(np.abs(np.fft.fftn(c)) ** 2 for c in u) / n ** 6
    shells = np.arange(1, n // 2)
    E = np.array([e[(K >= s - 0.5) & (K < s + 0.5)].sum() for s in shells])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    im = ax.imshow(u[0, :, :, 0].T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi], cmap="RdBu_r",
                   rasterized=True)
    fig.colorbar(im, ax=ax, shrink=0.85, label="u")
    ax.set_title("u on the plane z = 0 (64³ cells)", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax = axs[1]
    ax.loglog(shells, E, "o", color=P["teal"], ms=4, label="restart file")
    ks = np.linspace(1, n // 2, 200)
    model = ks ** 4 * np.exp(-2 * (ks / 4) ** 2)
    ax.loglog(ks, model * E[3] / model[np.searchsorted(ks, 4)], color=P["flame"], label=r"$k^4\,e^{-2(k/k_0)^2}$")
    ax.set_ylim(E.max() * 1e-8, E.max() * 3)
    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")
    ax.legend()
    ax.set_title("shell spectrum", fontsize=12)
    return fig


def channel(P):
    inp = tomllib.loads((ROOT / "examples" / "channel_retau180" / "input.toml").read_text())
    m = inp["mesh"]
    small = dict(m, Nx=m["Nx"] // 2, Ny=m["Ny"] // 2, Nz=m["Nz"] // 2)
    toml = "[mesh]\n" + "\n".join(f"{k} = {v!r}".replace("'", '"').replace("True", "true").replace("False", "false")
                                   for k, v in small.items() if not isinstance(v, dict)) + "\n"
    for table in ("physics", "source"):
        if table in inp:
            toml += f"[{table}]\n" + "\n".join(
                f"{k} = {v!r}".replace("'", '"').replace("True", "true").replace("False", "false")
                for k, v in inp[table].items() if not isinstance(v, dict)) + "\n"
    (WORK / "channel.toml").write_text(toml)
    path = once("channel.restart", lambda p: tool("channel_init.py", WORK / "channel.toml", p))
    f = read_restart(path)
    nx, ny, nz = small["Nx"], small["Ny"], small["Nz"]
    u = (f["RHOU_X"] / f["RHO"]).reshape(nx, ny, nz)
    from channel_init import nodes
    yc = nodes(ny, small["Ly"], small.get("stretching", [0, 0, 0])[1])
    yc = 0.5 * (yc[1:] + yc[:-1])
    zc = (np.arange(nz) + 0.5) * small["Lz"] / nz
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1, 2.2]))
    ax = axs[0]
    ax.plot(u.mean(axis=(0, 2)), yc, color=P["teal"], lw=2.2, label="mean")
    ax.fill_betweenx(yc, u.mean(axis=(0, 2)) - u.std(axis=(0, 2)), u.mean(axis=(0, 2)) + u.std(axis=(0, 2)),
                     color=P["teal"], alpha=0.25, lw=0, label="± rms")
    ax.set_xlabel("u / U_b")
    ax.set_ylabel("y / h")
    ax.legend(loc="center left")
    ax.set_title("streamwise velocity", fontsize=12)
    ax = axs[1]
    Z, Y = np.meshgrid(zc, yc)
    pc = ax.pcolormesh(Z, Y, (u[nx // 2] - u.mean(axis=(0, 2))[:, None]), cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                       rasterized=True, shading="auto")
    fig.colorbar(pc, ax=ax, label="u' / U_b")
    ax.set_aspect("equal")
    ax.set_xlabel("z / h")
    ax.set_title("divergence-free perturbation, cross-section x = Lx / 2", fontsize=12)
    return fig


def tile(P):
    src = once("hit32.restart", lambda p: tool("isotropic_turbulence_restart.py", 32, p))
    out = once("tiled.restart", lambda p: tool("tile_restart.py", src, 32, 3, p))
    a, b = read_restart(src), read_restart(out)
    ua = (a["RHOU_X"] / a["RHO"]).reshape(32, 32, 32)[:, :, 0]
    ub = (b["RHOU_X"] / b["RHO"]).reshape(96, 32, 32)[:, :, 0]
    fig, axs = plt.subplots(1, 2, figsize=(11, 2.9), gridspec_kw=dict(width_ratios=[1, 3]))
    lim = np.abs(ub).max()
    for ax, u, title in ((axs[0], ua, "IN.restart: 32 cells along x"),
                         (axs[1], ub, "tile_restart.py IN.restart 32 3 OUT.restart: 96 cells")):
        ax.imshow(u.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, rasterized=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("i (x)")
        ax.set_yticks([])
    for x in (31.5, 63.5):
        axs[1].axvline(x, color=P["fg"], lw=1, ls="--")
    return fig


def hex_layer(path):
    """Bottom faces (z = 0 layer) of the hexahedra of an ASCII Gmsh 2.2 file, as (cell index, 4 x 2 corners)."""
    text = open(path).read()
    nodes = np.array([[float(v) for v in l.split()] for l in
                      re.search(r"\$Nodes\n\d+\n(.*?)\$EndNodes", text, re.S).group(1).strip().splitlines()])
    index = {int(n): k for k, n in enumerate(nodes[:, 0])}
    xyz = nodes[:, 1:4]
    cells, polys = [], []
    hexes = [l.split() for l in re.search(r"\$Elements\n\d+\n(.*?)\$EndElements", text, re.S).group(1)
             .strip().splitlines() if l.split()[1] == "5"]
    for c, e in enumerate(hexes):
        v = xyz[[index[int(g)] for g in e[3 + int(e[2]):]]]
        if v[:, 2].mean() < 0.5 * np.ptp(v[:, 2]) + v[:, 2].min() + 1e-12 and v[:, 2].min() < 1e-12:
            low = v[v[:, 2] < v[:, 2].min() + 1e-9][:, :2]
            ang = np.arctan2(*(low - low.mean(0)).T[::-1])
            cells.append(c)
            polys.append(low[np.argsort(ang)])
    return np.array(cells), polys


def interpolate(P):
    from matplotlib.collections import PolyCollection
    from interpolate_restart import cell_centroids, write_restart
    coarse = once("volvo_coarse.msh", lambda p: tool("make_volvo_mesh.py", p, "--ds", 6e-3, "--nz", 2))
    fine = once("volvo_fine.msh", lambda p: tool("make_volvo_mesh.py", p, "--ds", 2e-3, "--nz", 2))
    cc = cell_centroids(coarse)

    def make_old(p):
        x, y = cc[:, 0], cc[:, 1]
        u = 1.0 - 1.3 * np.exp(-(y / 0.02) ** 2) * np.exp(-((x - 0.06) / 0.08) ** 2) * (x > 0) \
            + 0.25 * np.sin(60 * x) * np.exp(-(y / 0.03) ** 2) * (x > 0.1)
        rho = np.ones(len(cc))
        write_restart(p, {"version": 2, "step": 0, "t": 0.0, "names": ["RHO", "RHOU_X", "RHOU_Y", "RHOU_Z", "RHOE"],
                          "attributes": []}, np.stack([rho, u, 0 * u, 0 * u, 2.5 + 0.5 * u ** 2]))
    old = once("volvo_coarse.restart", make_old)
    new = once("volvo_fine.restart", lambda p: tool("interpolate_restart.py", coarse, old, fine, p))
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.6))
    for ax, msh, rs, title in ((axs[0], coarse, old, "OLD.msh and OLD.restart"),
                               (axs[1], fine, new, "interpolate_restart.py OLD.msh OLD.restart NEW.msh NEW.restart")):
        cells, polys = hex_layer(msh)
        u = read_restart(rs)["RHOU_X"][cells]
        pc = PolyCollection(polys, array=u, cmap=P["cmap"], edgecolors=P["bg"], linewidths=0.2, rasterized=True)
        pc.set_clim(-0.3, 1.25)
        ax.add_collection(pc)
        ax.set_xlim(-0.1, 0.45)
        ax.set_ylim(-0.06, 0.06)
        bare(ax)
        ax.set_title(f"{title} ({len(read_restart(rs)['RHO']):,} cells)", fontsize=11)
    fig.colorbar(pc, ax=axs, shrink=0.8, label="ρu")
    return fig


def flame(P):
    full = WORK / "flame_full"
    once("flame_full/done", lambda p: (tool("flame_reference.py", "--single", "h2", 1.0, 300, full), p.write_text("")))
    ref_csv = full / "h2_phi1_T300_mix_full.csv"
    mech = ROOT / "mechanisms" / "h2o2.yaml"
    run, run2 = WORK / "flame_run", WORK / "flame_run2d"
    once("flame_run/flame.restart", lambda p: tool("flame_restart.py", ref_csv, mech, "ohmech", 8, run))
    once("flame_run2d/flame.restart", lambda p: tool("flame_restart.py", ref_csv, mech, "ohmech", 8, run2,
                                                     "--ny", 48, "--perturb", 1.5))
    ref = np.genfromtxt(ref_csv, delimiter=",", names=True)
    m = tomllib.loads((run / "input.toml").read_text())["mesh"]
    f1 = read_restart(run / "flame.restart")
    xc = (np.arange(m["Nx"]) + 0.5) * m["Lx"] / m["Nx"]

    def mid(x, y):
        return np.interp(0.5 * (y.min() + y.max()), y, x)
    y_run = f1["RHOY_H2O"] / f1["RHO"]
    shift = mid(xc, y_run) - mid(ref["x"], ref["Y_H2O"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    for sp, col in (("H2", P["blue"]), ("O2", P["teal"]), ("H2O", P["plum"])):
        ax.plot(xc * 1e3, f1["RHOY_" + sp] / f1["RHO"], "o", ms=3.2, color=col, label=f"Y {sp} (restart)")
        ax.plot((ref["x"] + shift) * 1e3, ref["Y_" + sp], color=col, lw=1.2)
    ax.plot([], [], color=P["muted"], lw=1.2, label="Cantera")
    ax.set_xlim(0, m["Lx"] * 1e3)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("mass fraction")
    ax.legend(fontsize=9.5, loc="center right")
    ax.set_title("1D strip, 8 cells per thermal thickness", fontsize=12)
    ax = axs[1]
    m2 = tomllib.loads((run2 / "input.toml").read_text())["mesh"]
    f2 = read_restart(run2 / "flame.restart")
    y2 = (f2["RHOY_H2O"] / f2["RHO"]).reshape(m2["Nx"], m2["Ny"])
    im = ax.imshow(y2.T, origin="lower", extent=[0, m2["Lx"] * 1e3, 0, m2["Ly"] * 1e3], cmap=P["cmap"],
                   rasterized=True)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Y H2O")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_xlim(1.0, 3.6)
    ax.set_title("--ny 48 --perturb 1.5: wrinkled front", fontsize=12)
    return fig


def znd(P):
    mech = ROOT / "mechanisms" / "h2o2.yaml"
    zcsv = once("znd.csv", lambda p: tool("detonation_reference.py", mech, "ohmech", "H2:2, O2:1, AR:7", 298, 6670, p))
    rows = [r for r in csv.reader(open(zcsv)) if r and not r[0].startswith("#")]
    head, data = rows[0], np.array(rows[1:], float)
    col = {h: k for k, h in enumerate(head)}
    nx, ny, lx, ly = 400, 100, 0.08, 0.02
    out = once("znd2d.restart", lambda p: tool("znd_restart.py", zcsv, mech, "ohmech", nx, lx, 0.05, p, 2,
                                               "--ny", ny, "--ly", ly, "--pocket", 0.045, 0.006, 0.0015,
                                               "--pocket", 0.044, 0.014, 0.0012))
    f = read_restart(out)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw=dict(width_ratios=[1, 1.5], wspace=0.5))
    ax = axs[0]
    x = data[:, col["x"]] * 1e3
    ax.plot(x, data[:, col["T"]], color=P["flame"], lw=2, label="T")
    ax.set_xlabel("distance behind the shock [mm]")
    ax.set_ylabel("T [K]")
    T = data[:, col["T"]]
    ax.set_xlim(-0.02, 4 * x[np.argmax(np.gradient(T, x))])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, data[:, col["p"]] / 6670, color=P["teal"], lw=2, label="p / p0")
    ax2.set_ylabel("p / p0")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    ax.set_title("detonation_reference.py: ZND profile", fontsize=12)
    ax = axs[1]
    y = (f["RHOY_H2O"] / f["RHO"]).reshape(nx, ny)
    im = ax.imshow(y.T, origin="lower", extent=[0, lx * 1e3, 0, ly * 1e3], cmap=P["cmap"], rasterized=True)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Y H2O")
    ax.set_xlim(30, 60)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("znd_restart.py --ny 100 --pocket ... (2D start)", fontsize=12)
    return fig


def sedov(P):
    import sedov as sd
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    eta, f, g, h = sd.similarity_profiles(1.4)[:4]
    ax.plot(eta, g / g.max(), color=P["teal"], lw=2, label="ρ / ρ_shock")
    ax.plot(eta, f / f.max(), color=P["flame"], lw=2, label="u / u_shock")
    ax.plot(eta, h / h.max(), color=P["plum"], lw=2, label="p / p_shock")
    ax.set_xlabel("r / R(t)")
    ax.set_xlim(0, 1.02)
    ax.legend(loc="upper left")
    ax.set_title(f"sedov.py: similarity solution, ξ0 = {sd.xi0(1.4):.4f} (γ = 1.4)", fontsize=12)
    return fig


# ------------------------------------------------------------------- Mallard runs

RIEMANN = (ROOT / "examples" / "riemann_2d_quads" / "input.toml").read_text()


def riemann(order):
    """A 200 x 200 run of the 2D Riemann problem, first order or TENO5; returns its last VTU."""
    def make(p):
        text = RIEMANN.replace("Nx = 400", "Nx = 200").replace("Ny = 400", "Ny = 200")
        if order == "FO":
            text = text.replace('type = "TENO"\norder = 5', 'type = "FO"')
        if "time_interval" not in text:
            text = text.replace('prefix = "./solut/riemann"', 'prefix = "./solut/riemann"\ntime_interval = 0.1')
        p.mkdir(parents=True, exist_ok=True)
        (p / "input.toml").write_text(text)
        subprocess.run([str(MALLARD), "-i", "input.toml", "--kokkos-num-threads=4"], cwd=p, check=True,
                       stdout=subprocess.DEVNULL)
    d = once(f"riemann_{order}", make)
    return d, sorted((d / "solut").glob("riemann_*.vtu"))[-1]


def tripanel(ax, vtu, P, contours=0):
    pts, tris, tc, arrays = read_vtu(str(vtu))
    v = arrays["RHO"][tc]
    tpc = ax.tripcolor(pts[:, 0], pts[:, 1], tris, facecolors=v, cmap=P["cmap"], rasterized=True)
    if contours:
        import matplotlib.tri as mtri
        node = np.zeros(len(pts))
        cnt = np.zeros(len(pts))
        np.add.at(node, tris.ravel(), np.repeat(v, 3))
        np.add.at(cnt, tris.ravel(), 1)
        ax.tricontour(mtri.Triangulation(pts[:, 0], pts[:, 1], tris), node / np.maximum(cnt, 1), contours,
                      colors=P["bg"], linewidths=0.4).set_rasterized(True)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    return tpc, v


def plot_vtu(P):
    _, vtu = riemann("TENO")
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    tpc, v = tripanel(ax, vtu, P, contours=24)
    fig.colorbar(tpc, ax=ax, label="RHO")
    ax.set_title(f"RHO  min={v.min():.4g} max={v.max():.4g}", fontsize=12)
    return fig


def compare(P):
    runs = [("FO", "first order"), ("TENO", "TENO5")]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (order, title) in zip(axs, runs):
        _, vtu = riemann(order)
        tpc, v = tripanel(ax, vtu, P)
        tpc.set_clim(0.1, 1.8)
        ax.set_title(f"{title}  RHO [{v.min():.3f}, {v.max():.3f}]", fontsize=12)
    fig.colorbar(tpc, ax=axs, shrink=0.85, label="RHO")
    return fig


def animate_frame(P):
    d, _ = riemann("TENO")
    frames = d / "frames"
    if not frames.exists():
        tool("animate.py", d / "solut", d / "riemann", "--res", 500, "--frames-dir", frames, cwd=d)
    pngs = sorted(frames.glob("*.png"))
    img = plt.imread(pngs[-1])
    img = img[::max(1, img.shape[1] // 1100), ::max(1, img.shape[1] // 1100)]
    fig, ax = plt.subplots(figsize=(10, 10 * img.shape[0] / img.shape[1]))
    ax.imshow(img, rasterized=True)
    ax.axis("off")
    return fig


def cartesian_still(P):
    d, _ = riemann("TENO")
    grid = d / "grid"
    if not grid.exists():
        tool("animate_cartesian.py", "extract", d / "solut", grid, cwd=d)
    last = sorted(p for p in grid.glob("*.npy"))[-1]
    tool("animate_cartesian.py", "still", last, d / "still", cwd=d)
    a, b = plt.imread(d / "still_density.png"), plt.imread(d / "still_schlieren.png")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, title in ((axs[0], a, "still_density.png"), (axs[1], b, "still_schlieren.png")):
        ax.imshow(img, rasterized=True, interpolation="nearest")
        ax.set_title(title, fontsize=11, family="monospace")
        ax.axis("off")
    return fig


# ---------------------------------------------------------------- performance

def perf(P):
    rows = list(csv.DictReader(open(ROOT / "benchmarks" / "perf" / "baselines.csv")))
    hardware = list(dict.fromkeys(r["hardware"] for r in rows))
    benches = list(dict.fromkeys(r["benchmark"] for r in rows))
    val = {(r["hardware"], r["benchmark"]): float(r["cells_per_s"]) for r in rows}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.8 / len(hardware)
    colors = [P["teal"], P["flame"], P["plum"], P["blue"]]
    for k, hw in enumerate(hardware):
        xs = [i + (k - (len(hardware) - 1) / 2) * w for i, b in enumerate(benches) if (hw, b) in val]
        ys = [val[(hw, b)] for b in benches if (hw, b) in val]
        ax.bar(xs, ys, w * 0.92, color=colors[k % 4], label=hw)
    ax.set_yscale("log")
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels(benches, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("cells / s")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("benchmarks/perf/baselines.csv: what perf_suite.py compare checks against", fontsize=12)
    return fig


FIGURES = {
    "workflow": workflow,
    "annulus_mesh": annulus,
    "mallard_mesh": mallard_mesh,
    "case_meshes": case_meshes,
    "isotropic_turbulence_restart": isotropic,
    "channel_init": channel,
    "tile_restart": tile,
    "interpolate_restart": interpolate,
    "flame_restart": flame,
    "znd_restart": znd,
    "sedov": sedov,
    "perf_baselines": perf,
}
RUN_FIGURES = {
    "plot_vtu": plot_vtu,
    "compare_png": compare,
    "animate": animate_frame,
    "animate_cartesian": cartesian_still,
}


def main():
    global WORK, MALLARD
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", help="2D Mallard executable, for the figures from runs")
    ap.add_argument("--work", help="directory for the tools' output (kept between invocations)")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    WORK = Path(args.work or tempfile.mkdtemp()).resolve()
    WORK.mkdir(parents=True, exist_ok=True)
    figures = dict(FIGURES)
    if args.mallard:
        MALLARD = Path(args.mallard).resolve()
        figures |= RUN_FIGURES
    for name, draw in figures.items():
        if not args.names or name in args.names:
            figstyle.save(name, draw, out=OUT)


if __name__ == "__main__":
    main()
