"""Effective resolution of Mallard's face reconstructions (#660): the modified
wavenumber of each scheme, measured by driving the real code, and the ratio
r = Delta_eff / Delta of its effective filter width to the cell size.

    python tools/docs/effective_resolution.py --mallard build/src/mallard [--mallard3d build-3d/src/mallard]
        [--table] [--no-figure]

Each measurement is one forward-Euler step of the Euler equations with
uniform velocity a and pressure and a density wave rho = 1 + eps cos(k . x):
with HLLC the mass flux is then the exact upwind flux of the reconstructed
density, so the step gives the semi-discrete operator of linear advection,
d rho / dt = -i a k~ rho, at every wavenumber, with the reconstruction, face
quadrature and flux of a real run. The modified wavenumber is the projection
k~ = i (sum_c V_c R_c e^{-ik.x_c}) / (a sum_c V_c rho'_c e^{-ik.x_c}) of the
right-hand side R onto the wave. Linear schemes on uniform meshes take one run
with every wavenumber superposed (the modes do not mix); nonlinear ones (TENO
with every cell troubled, MUSCL with its limiter) and irregular meshes take one
run per wavenumber (Pirozzoli's approximate dispersion relation).

The effective wavenumber k_eff is where the numerical dissipation reaches 1%
per cell, -Im(k~) Delta = 0.01 (after Moura, Sherwin & Peiro's "1% rule"), and
r = pi / (k_eff Delta) with Delta = V^(1/d) the mean cell size. --table prints
the ratios to MUSCL that src/physics/les.cpp holds (EFFECTIVE_WIDTH_RATIO).
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import figstyle

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools")]
from mallard.vtu import read_vtu_cells  # noqa: E402

OUT = figstyle.OUT.parent / "theory"
DATA = Path(__file__).resolve().parent / "effective_resolution.json"
EPS = 1e-2       # wave amplitude (relative to rho = 1)
SPEED = 0.5      # advection speed a; p = 1, rho = 1: Mach 0.42
TOLERANCE = 0.01  # -Im(k~) Delta at k_eff

SCHEMES = {
    "FO": dict(recon='type = "FO"', linear=True),
    "MUSCL": dict(recon='type = "MUSCL"\nlimiter = "none"', linear=True),
    "MUSCL, Venkatakrishnan": dict(recon='type = "MUSCL"\nlimiter = "venkatakrishnan"', linear=False),
    **{f"TENO{k} linear": dict(recon=f'type = "TENO"\norder = {k}\ntroubled_threshold = 1e30', linear=True)
       for k in (3, 4, 5, 6)},
    **{f"TENO{k}, all troubled": dict(recon=f'type = "TENO"\norder = {k}\ntroubled_threshold = 0.0', linear=False)
       for k in (3, 5)},
    # The troubled-cell fallback: a cutoff that rejects the central polynomial and keeps every sector one
    "TENO5, sector stencils": dict(recon='type = "TENO"\norder = 5\ntroubled_threshold = 0.0\nC_T = 0.22',
                                   linear=False),
}


def run(binary, toml, workdir):
    (Path(workdir) / "input.toml").write_text(toml)
    subprocess.run([str(Path(binary).resolve()), "-i", "input.toml", "--kokkos-num-threads=2"], cwd=workdir, check=True,
                   stdout=subprocess.DEVNULL)


def cell_geometry(path, dim):
    """Centroids and volumes of the cells of a VTU (2D polygons; 3D boxes)."""
    pts, conn, offs, _, data = read_vtu_cells(str(path))
    starts = np.concatenate([[0], offs[:-1]])
    x, v = [], []
    for a, b in zip(starts, offs):
        p = pts[conn[a:b], :dim]
        if dim == 2:
            q = np.roll(p, -1, axis=0)
            cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
            area = 0.5 * cross.sum()
            x.append(((p + q) * cross[:, None]).sum(axis=0) / (6.0 * area))
            v.append(abs(area))
        else:
            x.append(p.mean(axis=0))
            v.append(np.prod(p.max(axis=0) - p.min(axis=0)))
    return np.array(x), np.array(v), data


def jittered_triangles(path, n, length=1.0, jitter=0.3, seed=7):
    """Periodic Gmsh mesh of n x n squares, interior nodes moved by up to jitter h, each square split along a
    random diagonal: an irregular triangulation whose sides still match for [[periodic]]."""
    rng = np.random.default_rng(seed)
    h = length / n
    xy = np.stack(np.meshgrid(np.arange(n + 1) * h, np.arange(n + 1) * h, indexing="ij"), axis=-1)
    xy[1:n, 1:n] += rng.uniform(-jitter, jitter, (n - 1, n - 1, 2)) * h

    def node(i, j):
        return i * (n + 1) + j + 1

    names = ['1 1 "left"', '1 2 "right"', '1 3 "bottom"', '1 4 "top"']
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names)), *names,
             "$EndPhysicalNames", "$Nodes", str((n + 1) ** 2)]
    for i in range(n + 1):
        for j in range(n + 1):
            lines.append(f"{node(i, j)} {xy[i, j, 0]:.17g} {xy[i, j, 1]:.17g} 0")
    lines.append("$EndNodes")
    el = []
    for j in range(n):
        el.append(f"1 2 1 1 {node(0, j)} {node(0, j + 1)}")
        el.append(f"1 2 2 2 {node(n, j)} {node(n, j + 1)}")
    for i in range(n):
        el.append(f"1 2 3 3 {node(i, 0)} {node(i + 1, 0)}")
        el.append(f"1 2 4 4 {node(i, n)} {node(i + 1, n)}")
    for i in range(n):
        for j in range(n):
            a, b, c, d = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
            if rng.random() < 0.5:
                el += [f"2 2 0 1 {a} {b} {c}", f"2 2 0 1 {a} {c} {d}"]
            else:
                el += [f"2 2 0 1 {a} {b} {d}", f"2 2 0 1 {b} {c} {d}"]
    lines += ["$Elements", str(len(el)), *[f"{k + 1} {e}" for k, e in enumerate(el)], "$EndElements"]
    Path(path).write_text("\n".join(lines) + "\n")


def case_toml(mesh, dim, recon, rho, dt):
    # A file mesh's seams need their translations; the 2D file meshes here are unit squares
    shift = ("translation = [1.0, 0.0]\n", "translation = [0.0, 1.0]\n") if "file" in mesh else ("", "")
    periodic = f'[[periodic]]\nzones = ["left", "right"]\n{shift[0]}[[periodic]]\nzones = ["bottom", "top"]\n{shift[1]}'
    if dim == 3:
        periodic += '[[periodic]]\nzones = ["back", "front"]\n'
    u = f"[{SPEED}, 0.0]" if dim == 2 else f"[{SPEED}, 0.0, 0.0]"
    return f"""[run]
n_steps = 1
dt = {dt}
[mesh]
{mesh}
{periodic}[initialize]
type = "analytical"
rho = "{rho}"
u = {u}
p = "1.0"
[numerics]
time_integrator = "FE"
riemann_solver = "HLLC"
[numerics.face_reconstruction]
{recon}
[physics]
type = "euler"
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0
[output]
check_interval = 1000000
[[write_data]]
prefix = "w"
format = "vtu"
interval = 1
variables = ["RHO"]
"""


def wave(k, phase=0.0):
    terms = [f"{c:.17g} * {v}" for c, v in zip(k, "xyz") if c != 0.0]
    return f"cos({' + '.join(terms)} + {phase:.17g})"


def measure(binary, mesh, dim, recon, waves, multi, dt):
    """k~ / |k| for each wave vector of waves: all in one run (multi) or one run each."""
    out = []
    groups = [waves] if multi else [[k] for k in waves]
    rng = np.random.default_rng(3)
    for group in groups:
        phases = rng.uniform(0, 2 * np.pi, len(group)) if multi else [0.0]
        rho = "1.0 + " + " + ".join(f"{EPS} * {wave(k, p)}" for k, p in zip(group, phases))
        with tempfile.TemporaryDirectory() as work:
            run(binary, case_toml(mesh, dim, recon, rho, dt), work)
            x, v, d0 = cell_geometry(Path(work) / "w_000000.vtu", dim)
            _, _, d1 = cell_geometry(Path(work) / "w_000001.vtu", dim)
        r0 = d0["RHO"] - 1.0
        R = (d1["RHO"] - d0["RHO"]) / dt
        for k in group:
            e = np.exp(-1j * (x @ np.asarray(k[:dim])))
            lam = np.sum(v * R * e) / np.sum(v * r0 * e)
            out.append(1j * lam / SPEED / np.linalg.norm(k))
    return out


def k_eff(kd, kmod):
    """First k Delta where -Im(k~) Delta reaches TOLERANCE, interpolated linearly; nan if never."""
    damp = -np.imag(kmod) * kd
    above = np.nonzero(damp >= TOLERANCE)[0]
    if len(above) == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return kd[0] * TOLERANCE / damp[0]
    return kd[i - 1] + (TOLERANCE - damp[i - 1]) * (kd[i] - kd[i - 1]) / (damp[i] - damp[i - 1])


def measure_all(mallard, mallard3d):
    results = {}
    with tempfile.TemporaryDirectory() as meshdir:
        # Uniform quadrilaterals: 128 x 8, waves along x at every resolved wavenumber
        nx, ny = 128, 8
        quads = f'type = "cartesian"\nNx = {nx}\nNy = {ny}\nLx = 1.0\nLy = {ny / nx}'
        modes = np.arange(1, nx // 2)
        for name, s in SCHEMES.items():
            waves = [(2 * np.pi * m, 0.0) for m in (modes if s["linear"] else modes[1::2])]
            kmod = measure(mallard, quads, 2, s["recon"], waves, s["linear"], 1e-5)
            results[f"quadrilaterals/{name}"] = dict(kd=[w[0] / nx for w in waves], re=np.real(kmod).tolist(),
                                                     im=np.imag(kmod).tolist())
            print(f"quadrilaterals  {name:24s} done", flush=True)
        # Irregular triangles: 40 x 40 jittered squares, waves along x and along the diagonal
        n = 40
        msh = Path(meshdir) / "triangles.msh"
        jittered_triangles(msh, n)
        tris = f'type = "file"\nfilename = "{msh}"'
        delta = np.sqrt(0.5) / n
        for name in ("MUSCL", "TENO3 linear", "TENO5 linear", "TENO5, all troubled"):
            s = SCHEMES[name]
            curves = []
            for direction in ((1, 0), (1, 1)):
                waves = [(2 * np.pi * m * direction[0], 2 * np.pi * m * direction[1]) for m in range(1, n // 2)]
                kmod = measure(mallard, tris, 2, s["recon"], waves, False, 1e-5)
                curves.append(dict(kd=[float(np.linalg.norm(w)) * delta for w in waves],
                                   re=np.real(kmod).tolist(), im=np.imag(kmod).tolist()))
            results[f"triangles/{name}"] = dict(directions=curves)
            print(f"triangles       {name:24s} done", flush=True)
    if mallard3d:
        nx, ny = 64, 6
        hexes = f'type = "cartesian"\nNx = {nx}\nNy = {ny}\nNz = {ny}\nLx = 1.0\nLy = {ny / nx}\nLz = {ny / nx}'
        modes = np.arange(1, nx // 2)
        for name in ("MUSCL", "TENO3 linear", "TENO4 linear", "TENO5 linear", "TENO6 linear",
                     "TENO5, all troubled", "TENO5, sector stencils"):
            s = dict(SCHEMES[name])
            # Six sectors in 3D: equal weights are 1/7 (central) and 1/6 (each sector)
            s["recon"] = s["recon"].replace("C_T = 0.22", "C_T = 0.155")
            waves = [(2 * np.pi * m, 0.0, 0.0) for m in (modes if s["linear"] else modes[1::2])]
            kmod = measure(mallard3d, hexes, 3, s["recon"], waves, s["linear"], 1e-5)
            results[f"hexahedra/{name}"] = dict(kd=[w[0] / nx for w in waves], re=np.real(kmod).tolist(),
                                                im=np.imag(kmod).tolist())
            print(f"hexahedra       {name:24s} done", flush=True)
    return results


def ratios(results):
    """r = pi / (k_eff Delta) per mesh and scheme (triangles: the mean over the directions)."""
    r = {}
    for key, c in results.items():
        curves = c.get("directions", [c])
        ks = [k_eff(np.array(d["kd"]), np.array(d["re"]) + 1j * np.array(d["im"])) for d in curves]
        r[key] = dict(k_eff=float(np.mean(ks)), r=float(np.pi / np.mean(ks)),
                      per_direction=[float(np.pi / k) for k in ks])
    return r


COLORS = {"FO": "muted", "MUSCL": "plum", "MUSCL, Venkatakrishnan": "red", "TENO3 linear": "blue",
          "TENO4 linear": "green", "TENO5 linear": "teal", "TENO6 linear": "gold",
          "TENO3, all troubled": "blue", "TENO5, all troubled": "flame", "TENO5, sector stencils": "red"}


def figure(results):
    import matplotlib.pyplot as plt

    shown = ["FO", "MUSCL", "TENO3 linear", "TENO5 linear", "TENO5, all troubled", "TENO5, sector stencils"]

    def draw(P):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for name in shown:
            c = results[f"quadrilaterals/{name}"]
            kd, re, im = np.array(c["kd"]), np.array(c["re"]), np.array(c["im"])
            style = "-" if "linear" in name or "TENO" not in name else "--"
            axes[0].plot(kd, re * kd, style, color=P[COLORS[name]], label=name)
            axes[1].semilogy(kd, np.maximum(-im * kd, 1e-7), style, color=P[COLORS[name]], label=name)
            tri = results.get(f"triangles/{name}")
            if tri:
                d = tri["directions"][0]
                kt, it = np.array(d["kd"]), np.array(d["im"])
                axes[1].semilogy(kt, np.maximum(-it * kt, 1e-7), ":", color=P[COLORS[name]], lw=1.4)
        axes[0].plot([0, np.pi], [0, np.pi], color=P["line"], lw=1.0, label="exact")
        axes[0].set(xlabel=r"$k\Delta$", ylabel=r"$\mathrm{Re}(\tilde k)\Delta$", xlim=(0, np.pi), ylim=(0, np.pi),
                    title="Dispersion")
        axes[1].axhline(TOLERANCE, color=P["muted"], lw=1.0)
        figstyle.label(axes[1], 2.3, TOLERANCE * 0.55, "1% rule", P, size=11, ha="left")
        axes[1].set(xlabel=r"$k\Delta$", ylabel=r"$-\mathrm{Im}(\tilde k)\Delta$", xlim=(0, np.pi), ylim=(1e-6, 5),
                    title="Dissipation (dotted: irregular triangles)")
        axes[0].legend(loc="upper left", fontsize=10.5)
        fig.tight_layout()
        return fig

    figstyle.save("effective_resolution", draw, OUT)


def figures():
    """The figure from the saved measurements (make_figures.py)."""
    figure(json.loads(DATA.read_text()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", help="2D Mallard executable (without it, the saved measurements are redrawn)")
    ap.add_argument("--mallard3d", help="3D Mallard executable, for hexahedra")
    ap.add_argument("--table", action="store_true", help="print the ratios to MUSCL of the linear TENO stencils")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    if args.mallard:
        results = measure_all(args.mallard, args.mallard3d)
        DATA.write_text(json.dumps(results) + "\n")
    else:
        results = json.loads(DATA.read_text())
    r = ratios(results)
    print(f"{'mesh / scheme':44s} {'k_eff Delta':>11s} {'r':>6s}")
    for key, v in r.items():
        extra = "  (" + ", ".join(f"{x:.2f}" for x in v["per_direction"]) + ")" if len(v["per_direction"]) > 1 else ""
        print(f"{key:44s} {v['k_eff']:11.3f} {v['r']:6.2f}{extra}")
    if args.table:
        for mesh in ("quadrilaterals", "hexahedra"):
            if f"{mesh}/MUSCL" not in r:
                continue
            ref = r[f"{mesh}/MUSCL"]["r"]
            print(mesh, {k: round(r[f"{mesh}/TENO{k} linear"]["r"] / ref, 3) for k in (3, 4, 5, 6)
                         if f"{mesh}/TENO{k} linear" in r})
    if not args.no_figure:
        figure(results)


if __name__ == "__main__":
    main()
