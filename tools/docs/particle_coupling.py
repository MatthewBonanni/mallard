"""Figures of two-way coupled particles, in the particles pages of the user
guide (docs/images/guide/) and the theory manual (docs/images/theory/).

    python tools/docs/particle_coupling.py --mallard build/src/mallard [names...]

From short Mallard runs (a 2D build, seconds each on 2 CPU threads): the
cloud of examples/particle_cloud falling through gas at rest, coupled one way and
two ways (particle_cloud_coupling), and the deposit of one particle's volume
fraction over a Gaussian of deposition_width on meshes of 16 to 128 cells per
side against the Gaussian itself (particle_deposition). Particle positions
are read from restart files (tools/mallard/restart.py), fields from VTU
snapshots.
"""
import argparse
import csv
import functools
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import figstyle
from figstyle import bare

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from mallard import restart  # noqa: E402
from mallard.vtu import outlines, read_vtu_cells, snapshot  # noqa: E402

THEORY = figstyle.OUT.parent / "theory"


def run(binary, toml, workdir):
    (Path(workdir) / "input.toml").write_text(toml)
    subprocess.run([str(binary), "-i", "input.toml", "--kokkos-num-threads=2"], cwd=workdir, check=True,
                   stdout=subprocess.DEVNULL)


def cell_polygons(pts, conn, offs, types):
    polys = [None] * len(offs)
    for cells, ids in outlines(conn, offs, types):
        for c, row in zip(cells, ids):
            polys[c] = pts[row][:, :2]
    return polys


def last_snapshot(work, prefix):
    stems = sorted({p.with_suffix("") for p in Path(work).glob(f"{prefix}_*") if p.suffix in (".vtu", ".pvtu")})
    return read_vtu_cells(snapshot(str(stems[-1])))


def probe(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    return {name: np.array([float(r[k]) for r in rows[1:]]) for k, name in enumerate(rows[0])}


# ------------------------------------------------------------ the falling cloud

@functools.cache
def cloud_run(binary, coupling):
    """examples/particle_cloud with the given coupling: its probe, the last snapshot and the particles' positions."""
    work = Path(tempfile.mkdtemp())
    cfg = (ROOT / "examples" / "particle_cloud" / "input.toml").read_text()
    cfg = cfg.replace('coupling = "two-way"', f'coupling = "{coupling}"')
    cfg = cfg.replace('prefix = "cloud"', 'prefix = "cloud"\nformat = "vtu"').replace(', "PARTICLE_SOURCE_RHOU_Y"', "")
    cfg = re.sub(r"particle_fields = .*\n", "", cfg)
    cfg += '\n[[write_data]]\nprefix = "end"\nformat = "restart"\ntime_interval = 2.0\n'
    run(binary, cfg, work)
    r = restart.read(str(sorted(work.glob("end_*.restart"))[-1]))
    x = np.asarray(r.arrays["PARTICLE_X"], dtype=float).reshape(-1, 2)
    return probe(work / "particle_probe_cloud.csv"), last_snapshot(work, "cloud"), x


def particle_cloud_coupling(P, args):
    fig = plt.figure(figsize=(10.4, 4.3))
    a = fig.add_axes([0.07, 0.14, 0.4, 0.76])
    b = fig.add_axes([0.55, 0.06, 0.4, 0.86])
    for coupling, color, name in [("one-way", P["muted"], "one-way"), ("two-way", P["flame"], "two-way")]:
        p, _, _ = cloud_run(args.mallard, coupling)
        a.plot(np.concatenate([[0], p["t"]]), np.concatenate([[0], -p["V_Y mean"]]), color=color, label=name)
    tau, g = 0.1, 1.0
    t = np.linspace(0, 2, 200)
    a.plot(t, tau * g * (1 - np.exp(-t / tau)), color=P["blue"], ls="--", lw=1.2,
           label=r"one particle, $\tau g (1 - e^{-t/\tau})$")
    a.set_xlabel("t")
    a.set_ylabel("mean fall velocity of the cloud")
    a.set_xlim(0, 2)
    a.set_ylim(0, None)
    a.legend(fontsize=10, loc="upper left")
    a.set_title("81 parcels, mass loading 1", fontsize=12)
    _, (pts, conn, offs, types, arrays), x = cloud_run(args.mallard, "two-way")
    u = np.asarray(arrays["U"], dtype=float).reshape(len(offs), -1)[:, 1]
    lim = np.abs(u).max()
    pc = PolyCollection(cell_polygons(pts, conn, offs, types), array=u, cmap="RdBu_r", edgecolors="face",
                        linewidths=0.2, clim=(-lim, lim), rasterized=True)
    b.add_collection(pc)
    b.plot(x[:, 0], x[:, 1], "o", color=P["fg"], ms=2.2)
    bare(b)
    b.set_xlim(0, 1)
    b.set_ylim(0, 1)
    cb = fig.colorbar(pc, ax=b, fraction=0.04, pad=0.02)
    cb.set_label("gas velocity $u_y$", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_visible(False)
    b.set_title("two-way, t = 2: the gas dragged down", fontsize=12)
    return fig


# ------------------------------------------------------------ deposition

# The error's constant depends on where the particle sits in its cell (the deposit's covariance is short by
# d d^T, d its offset from the cell center), so the particle sits a third of a cell from a corner, the same
# |d| = h / 6 on every mesh of 16 * 2^k cells per side: then the study measures the order, not the offset
SIGMA, POINT, DIAMETER, DENSITY = 0.08, ((6 + 1 / 3) / 16, (9 + 2 / 3) / 16), 1e-3, 1000.0


def single_particle(n, sigma):
    """One coupled particle at rest in a periodic unit box of gas at rest, its fields written after one tiny step."""
    return f"""[run]
n_steps = 1
dt = 1e-9
[mesh]
type = "cartesian"
Nx = {n}
Ny = {n}
Lx = 1.0
Ly = 1.0
[[periodic]]
zones = ["left", "right"]
[[periodic]]
zones = ["bottom", "top"]
[initialize]
type = "constant"
T = 0.7142857142857143
u = [0.0, 0.0]
p = 0.7142857142857143
[numerics]
preset = "first_order"
[physics]
type = "navier_stokes"
units = "nondimensional"
mu = 1e-3
gamma = 1.4
p_ref = 1.0
T_ref = 1.0
rho_ref = 1.0
[particles]
type = "solid"
coupling = "two-way"
diameter = {DIAMETER}
density = {DENSITY}
deposition_width = {sigma}
velocity = [0.0, 0.0]
points = [[{POINT[0]}, {POINT[1]}]]
[[write_data]]
prefix = "deposit"
format = "vtu"
fields = ["PARTICLE_VOLUME_FRACTION"]
interval = 1
"""


def gaussian(x, y, sigma):
    """The volume fraction of the particle spread over a periodic Gaussian about it."""
    volume = math.pi / 6.0 * DIAMETER ** 3
    dx = x - POINT[0] - np.round(x - POINT[0])
    dy = y - POINT[1] - np.round(y - POINT[1])
    return volume * np.exp(-0.5 * (dx * dx + dy * dy) / sigma ** 2) / (2.0 * math.pi * sigma ** 2)


@functools.cache
def deposit(binary, n, sigma):
    """Cell centers, cell areas, polygons and volume fraction of the deposit on n x n cells."""
    work = Path(tempfile.mkdtemp())
    run(binary, single_particle(n, sigma), work)
    pts, conn, offs, types, arrays = last_snapshot(work, "deposit")
    polys = cell_polygons(pts, conn, offs, types)
    centers = np.array([p.mean(axis=0) for p in polys])
    return centers, np.full(len(polys), 1.0 / n ** 2), polys, np.asarray(arrays["PARTICLE_VOLUME_FRACTION"], float)


def particle_deposition(P, args):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw=dict(wspace=0.3, width_ratios=[1.35, 1]))
    peak = gaussian(np.array(POINT[0]), np.array(POINT[1]), SIGMA)
    xs = np.linspace(0, 1, 400)
    a.plot(xs, gaussian(xs, POINT[1], SIGMA) / peak, color=P["fg"], lw=1.4, label=rf"Gaussian, $\sigma$ = {SIGMA:g}")
    ns = [16, 32, 64, 128]
    colors = [P["flame"], P["teal"], P["blue"], P["plum"]]
    errors = []
    for n, color in zip(ns, colors):
        c, area, _, alpha = deposit(args.mallard, n, SIGMA)
        exact = gaussian(c[:, 0], c[:, 1], SIGMA)
        volume = math.pi / 6.0 * DIAMETER ** 3
        errors.append(np.sum(np.abs(alpha - exact) * area) / volume)
        row = np.abs(c[:, 1] - POINT[1]) <= 0.5 / n + 1e-12
        order = np.argsort(c[row, 0])
        edges = np.concatenate([c[row, 0][order] - 0.5 / n, [c[row, 0][order][-1] + 0.5 / n]])
        a.stairs(alpha[row][order] / peak, edges, color=color, lw=1.3,
                 label=rf"{n} cells per side, $\sigma / h$ = {SIGMA * n:.1f}")
    a.axvline(POINT[0], color=P["muted"], lw=0.8, ls=":")
    a.set_xlim(0.05, 1.0)
    a.set_xlabel("x, through the particle's row of cells")
    a.set_ylabel("volume fraction / Gaussian's peak")
    a.legend(fontsize=9.5, loc="upper right")
    a.set_title("deposit of one particle", fontsize=12)
    h = 1.0 / np.array(ns)
    b.loglog(h, errors, "o-", color=P["blue"], label="Mallard")
    b.loglog(h, errors[-1] * (h / h[-1]) ** 2, color=P["muted"], ls="--", lw=1.0, label="second order")
    b.set_xticks(h, [f"1/{n}" for n in ns])
    b.minorticks_off()
    b.set_xlabel("cell size h")
    b.set_ylabel(r"$L_1$ error / particle volume")
    b.legend(fontsize=10, loc="lower right")
    b.set_title("convergence at fixed $\\sigma$", fontsize=12)
    return fig


GUIDE_RUNS = {"particle_cloud_coupling": particle_cloud_coupling}
THEORY_RUNS = {"particle_deposition": particle_deposition}


def figures(mallard=None, names=None):
    if not mallard:
        return
    args = argparse.Namespace(mallard=str(Path(mallard).resolve()))
    want = lambda name: not names or name in names  # noqa: E731
    for name, draw in GUIDE_RUNS.items():
        if want(name):
            figstyle.save(name, lambda P, draw=draw: draw(P, args))
    for name, draw in THEORY_RUNS.items():
        if want(name):
            figstyle.save(name, lambda P, draw=draw: draw(P, args), THEORY)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="2D Mallard executable")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    figures(args.mallard, args.names)


if __name__ == "__main__":
    main()
