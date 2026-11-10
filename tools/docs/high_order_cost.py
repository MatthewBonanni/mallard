"""Error against cost of each order of reconstruction: the isentropic vortex
(Shu 1998) convected across a periodic box of irregular triangles, with FO,
MUSCL and TENO-E orders 3 and 5, each on several meshes.

    python tools/docs/high_order_cost.py --mallard build/src/mallard [--threads 2]

The box is 10 x 10 and the vortex (strength 5, free stream rho = p = 1,
u = 1) is convected to t = 2, where the exact solution is the initial one
shifted by 2. The meshes (triangles()) are n x n squares, interior nodes
moved by up to 0.25 h, each square split along its shorter diagonal. Every
scheme runs with HLLC, SSPRK3 and cfl = 1.0 (MUSCL unlimited: the flow is
smooth). The error is the L2 norm of the density error over the box, against
exact cell averages (a degree-5 rule on 16 subtriangles per cell). The cost
of a run is its CPU time (user + system, setup, TENO-E stencils and output
included) on --threads threads, which a busy machine disturbs less than its
wall time (also recorded); runs under 30 s are the best of three. The cell
count is the cost that does not depend on the machine. The measurements go
to high_order_cost.json, which make_figures.py draws without running anything.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import figstyle

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools")]
from mallard.vtu import read_vtu_cells  # noqa: E402

DATA = Path(__file__).resolve().parent / "high_order_cost.json"
LENGTH, T_STOP, BETA, GAMMA, CFL = 10.0, 2.0, 5.0, 1.4, 1.0
SCHEMES = {
    "FO": dict(recon='type = "FO"', meshes=(16, 32, 64, 128, 256, 512)),
    "MUSCL": dict(recon='type = "MUSCL"\nlimiter = "none"', meshes=(16, 32, 64, 128, 256, 512)),
    "TENO3": dict(recon='type = "TENO"\norder = 3', meshes=(16, 32, 64, 128, 256)),
    "TENO5": dict(recon='type = "TENO"\norder = 5', meshes=(16, 32, 64, 128)),
}
TARGETS = (1e-3, 1e-4)
COLORS = {"FO": "red", "MUSCL": "gold", "TENO3": "blue", "TENO5": "teal"}


def triangles(path, n, jitter=0.25, seed=7):
    """Periodic Gmsh mesh of n x n squares of the box, interior nodes moved by up to jitter h, each square split
    along its shorter diagonal: irregular triangles without slivers, whose sides match for [[periodic]]."""
    rng = np.random.default_rng(seed)
    h = LENGTH / n
    xy = np.stack(np.meshgrid(np.arange(n + 1) * h, np.arange(n + 1) * h, indexing="ij"), axis=-1)
    xy[1:n, 1:n] += rng.uniform(-jitter, jitter, (n - 1, n - 1, 2)) * h

    def node(i, j):
        return i * (n + 1) + j + 1

    names = ['1 1 "left"', '1 2 "right"', '1 3 "bottom"', '1 4 "top"']
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names)), *names,
             "$EndPhysicalNames", "$Nodes", str((n + 1) ** 2)]
    lines += [f"{node(i, j)} {xy[i, j, 0]:.17g} {xy[i, j, 1]:.17g} 0" for i in range(n + 1) for j in range(n + 1)]
    lines.append("$EndNodes")
    el = []
    for j in range(n):
        el += [f"1 2 1 1 {node(0, j)} {node(0, j + 1)}", f"1 2 2 2 {node(n, j)} {node(n, j + 1)}"]
    for i in range(n):
        el += [f"1 2 3 3 {node(i, 0)} {node(i + 1, 0)}", f"1 2 4 4 {node(i, n)} {node(i + 1, n)}"]
    for i in range(n):
        for j in range(n):
            a, b, c, d = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
            if np.linalg.norm(xy[i + 1, j + 1] - xy[i, j]) < np.linalg.norm(xy[i, j + 1] - xy[i + 1, j]):
                el += [f"2 2 0 1 {a} {b} {c}", f"2 2 0 1 {a} {c} {d}"]
            else:
                el += [f"2 2 0 1 {a} {b} {d}", f"2 2 0 1 {b} {c} {d}"]
    lines += ["$Elements", str(len(el)), *[f"{k + 1} {e}" for k, e in enumerate(el)], "$EndElements"]
    Path(path).write_text("\n".join(lines) + "\n")


def exact(x, y, t=0.0):
    """Density, velocity and pressure of the vortex, centered in the box at t = 0, convected at u = 1."""
    dx = (x - LENGTH / 2 - t + LENGTH / 2) % LENGTH - LENGTH / 2
    dy = y - LENGTH / 2
    f = np.exp(0.5 * (1.0 - dx * dx - dy * dy))
    rho = (1.0 - (GAMMA - 1.0) * BETA**2 / (8.0 * GAMMA * np.pi**2) * f * f) ** (1.0 / (GAMMA - 1.0))
    return rho, 1.0 - BETA / (2 * np.pi) * f * dy, BETA / (2 * np.pi) * f * dx, rho**GAMMA


def kexpr():
    f = f"exp(0.5 * (1.0 - (x - {LENGTH / 2}) * (x - {LENGTH / 2}) - (y - {LENGTH / 2}) * (y - {LENGTH / 2})))"
    rho = f"pow(1.0 - {(GAMMA - 1.0) * BETA**2 / (8.0 * GAMMA * np.pi**2):.17g} * {f} * {f}, {1.0 / (GAMMA - 1.0):.17g})"
    a = BETA / (2 * np.pi)
    return dict(rho=rho, u=f"1.0 - {a:.17g} * {f} * (y - {LENGTH / 2})", v=f"{a:.17g} * {f} * (x - {LENGTH / 2})",
                p=f"pow({rho}, {GAMMA})")


def case_toml(mesh, recon):
    e = kexpr()
    return f"""[run]
t_stop = {T_STOP}
cfl = {CFL}
[mesh]
type = "file"
filename = "{mesh}"
[[periodic]]
zones = ["left", "right"]
translation = [{LENGTH}, 0.0]
[[periodic]]
zones = ["bottom", "top"]
translation = [0.0, {LENGTH}]
[initialize]
type = "analytical"
rho = "{e['rho']}"
u = ["{e['u']}", "{e['v']}"]
p = "{e['p']}"
[numerics]
riemann_solver = "HLLC"
time_integrator = "SSPRK3"
[numerics.face_reconstruction]
{recon}
[physics]
type = "euler"
gamma = {GAMMA}
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0
[output]
check_interval = 1000000
[[write_data]]
prefix = "w"
format = "vtu"
interval = 1000000
variables = ["RHO"]
"""


# Degree-5 rule on the reference triangle (Dunavant, 7 points): barycentric coordinates and weights
_A, _B = (6 - np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21
_WA, _WB = (155 - np.sqrt(15)) / 1200, (155 + np.sqrt(15)) / 1200
BARY = np.array([[1 / 3, 1 / 3, 1 / 3], [_A, _A, 1 - 2 * _A], [_A, 1 - 2 * _A, _A], [1 - 2 * _A, _A, _A],
                 [_B, _B, 1 - 2 * _B], [_B, 1 - 2 * _B, _B], [1 - 2 * _B, _B, _B]])
WEIGHTS = np.array([9 / 40, _WA, _WA, _WA, _WB, _WB, _WB])


def subdivide(tri, levels=2):
    """Each triangle (N, 3, 2) split into 4**levels congruent ones."""
    for _ in range(levels):
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tri = np.concatenate([np.stack(t, axis=1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return tri


def density_error(vtu, t):
    pts, conn, offs, _, data = read_vtu_cells(str(vtu))
    if np.any(np.diff(np.concatenate([[0], offs])) != 3):
        raise RuntimeError("expected triangles only")
    tri = pts[conn.reshape(-1, 3), :2]
    area = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) -
                        (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    sub = subdivide(tri).reshape(16, len(tri), 3, 2)
    q = np.einsum("qk,sckd->scqd", BARY, sub)
    average = (exact(q[..., 0], q[..., 1], t)[0] * WEIGHTS).sum(axis=2).mean(axis=0)
    return float(np.sqrt(np.sum(area * (data["RHO"] - average) ** 2) / np.sum(area)))


def run_case(binary, mesh, recon, threads, work):
    (Path(work) / "input.toml").write_text(case_toml(mesh, recon))
    start, cpu_start = time.perf_counter(), os.times()
    out = subprocess.run([str(Path(binary).resolve()), "-i", "input.toml", f"--kokkos-num-threads={threads}"],
                         cwd=work, check=True, capture_output=True, text=True).stdout
    wall, cpu_end = time.perf_counter() - start, os.times()
    cpu = (cpu_end.children_user - cpu_start.children_user) + (cpu_end.children_system - cpu_start.children_system)
    final = sorted(Path(work).glob("w_*.vtu"))[-1]
    steps = re.findall(r"^\s+([\d,]+)\s+\d\.\d+e[+-]\d+", out, re.M)
    return dict(error=density_error(final, T_STOP), wall=wall, cpu=cpu, steps=int(steps[-1].replace(',', '')) if steps else None)


def measure(binary, threads):
    results = dict(machine=f"{platform.machine()} {platform.system()}, {threads} threads", runs={})
    with tempfile.TemporaryDirectory() as work:
        for n in sorted({n for s in SCHEMES.values() for n in s["meshes"]}):
            mesh = Path(work) / f"tri{n}.msh"
            triangles(mesh, n)
            for name, s in SCHEMES.items():
                if n not in s["meshes"]:
                    continue
                tries = []
                while not tries or (sum(t["wall"] for t in tries) < 30.0 and len(tries) < 3):  # short: best of 3
                    with tempfile.TemporaryDirectory() as case:
                        tries.append(run_case(binary, mesh, s["recon"], threads, case))
                r = min(tries, key=lambda t: t["cpu"])
                r["cells"] = 2 * n * n
                results["runs"].setdefault(name, []).append(r)
                print(f"{name:6s} {2 * n * n:7d} cells  error {r['error']:.3e}  cpu {r['cpu']:8.2f} s  "
                      f"wall {r['wall']:8.2f} s  steps {r['steps']}", flush=True)
    return results


def orders(runs):
    """Measured order between successive meshes, from the error against h = cells^(-1/2)."""
    c, e = np.array([r["cells"] for r in runs], float), np.array([r["error"] for r in runs])
    return list(-2 * np.diff(np.log(e)) / np.diff(np.log(c)))


def at_error(runs, key, target):
    """The cells or wall time that reach the target error, interpolated in log-log between the two runs around
    it; None if the runs do not reach it."""
    e = np.array([r["error"] for r in runs])
    x = np.array([r[key] for r in runs], float)
    for i in range(len(e) - 1):
        if e[i] >= target >= e[i + 1]:
            s = (np.log(target) - np.log(e[i])) / (np.log(e[i + 1]) - np.log(e[i]))
            return float(np.exp(np.log(x[i]) + s * (np.log(x[i + 1]) - np.log(x[i]))))
    return None


def table(results):
    runs = results["runs"]
    for name, r in runs.items():
        print(f"{name}: orders " + ", ".join(f"{p:.2f}" for p in orders(r)))
    for target in TARGETS:
        print(f"error {target:g}:")
        for name, r in runs.items():
            cells, cpu = at_error(r, "cells", target), at_error(r, "cpu", target)
            print(f"  {name:6s} " + ("not reached" if cells is None else f"{cells:9.0f} cells {cpu:8.2f} s CPU"))
    for name in ("FO", "MUSCL", "TENO3"):
        print(f"{name} / TENO5 CPU time at equal error, and where it crosses 1:")
        crossing = None
        for target in np.geomspace(1e-2, 1e-6, 81):
            a, b = at_error(runs[name], "cpu", target), at_error(runs["TENO5"], "cpu", target)
            if a is None or b is None:
                continue
            if crossing is None and a > b:
                crossing = target
            if any(np.isclose(target, t) for t in TARGETS + (1e-5,)):
                print(f"  error {target:g}: {a / b:.2f}")
        print(f"  TENO5 cheaper below error {crossing:.2g}" if crossing else "  TENO5 never cheaper in range")


def figure(results):
    import matplotlib.pyplot as plt

    def draw(P):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
        for name, runs in results["runs"].items():
            e = [r["error"] for r in runs]
            for ax, key in zip(axes, ("cells", "cpu")):
                ax.loglog([r[key] for r in runs], e, "o-", color=P[COLORS[name]], label=name)
        axes[0].set(xlabel="cells", ylabel=r"$L_2$ error of $\rho$", title="Error against mesh size")
        axes[1].set(xlabel=f"CPU time, s ({results['machine']})", title="Error against cost")
        axes[0].legend(loc="lower left", fontsize=10.5)
        fig.tight_layout()
        return fig

    figstyle.save("high_order_cost", draw)


def figures():
    """The figure from the saved measurements (make_figures.py)."""
    figure(json.loads(DATA.read_text()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", help="2D Mallard executable (without it, the saved measurements are redrawn)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    if args.mallard:
        results = measure(args.mallard, args.threads)
        DATA.write_text(json.dumps(results, indent=1) + "\n")
    else:
        results = json.loads(DATA.read_text())
    table(results)
    if not args.no_figure:
        figure(results)


if __name__ == "__main__":
    main()
