"""Figures of particle output, in the particles page of the user guide
(docs/images/guide/).

    python tools/docs/particle_output.py --mallard build/src/mallard [names...]

From short runs of examples/particle_vortex (a 2D build with HDF5, about a
minute on 2 CPU threads): particle snapshots of [[write_data]] over the gas
speed of a VTU snapshot at the same times (particle_snapshots), and the
crossings of the boundary of the core probe's disk against the probe's count
(particle_crossings), and the trajectories of a fraction of the particles
colored by their temperature (particle_trajectories). The particle files are read with h5py.
"""
import argparse
import csv
import functools
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

import figstyle
from figstyle import bare

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from mallard.vtu import outlines, read_vtu_cells, snapshot  # noqa: E402

SNAPSHOT_TIMES = (0, 3, 6)


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


def vortex_input(extra=""):
    """examples/particle_vortex without its own outputs, with the given ones."""
    cfg = (ROOT / "examples" / "particle_vortex" / "input.toml").read_text()
    cfg = re.sub(r"\[\[write_data\]\]\n(?:[^\[\n][^\n]*\n)*", "", cfg)
    return cfg + extra


def read_particles(path):
    """{name: array} of a particle snapshot file: positions x and every point array."""
    with h5py.File(path, "r") as f:
        out = {name: np.asarray(f["VTKHDF/PointData"][name]) for name in f["VTKHDF/PointData"]}
        out["x"] = np.asarray(f["VTKHDF/Points"])[:, :2]
    return out


@functools.cache
def snapshot_run(binary):
    """The gas speed and the particles at each snapshot time of examples/particle_vortex."""
    work = Path(tempfile.mkdtemp())
    run(binary, vortex_input('[[write_data]]\nprefix = "gas"\nformat = "vtu"\nfields = ["U"]\ntime_interval = 1.0\n'
                             '[[write_data]]\nprefix = "drops"\nparticle_fields = ["T_P", "SLIP_MAX"]\n'
                             'time_interval = 1.0\n'), work)
    out = []
    for k in SNAPSHOT_TIMES:
        pts, conn, offs, types, arrays = read_vtu_cells(snapshot(str(work / f"gas_{k:06d}")))
        speed = np.linalg.norm(np.asarray(arrays["U"], dtype=float).reshape(len(offs), -1)[:, :2], axis=1)
        out.append((k, cell_polygons(pts, conn, offs, types), speed,
                    read_particles(work / f"drops_particles_{k:06d}.vtkhdf")))
    return out


def particle_snapshots(P, args):
    snaps = snapshot_run(args.mallard)
    fig, axes = plt.subplots(1, len(snaps), figsize=(11.0, 4.3), gridspec_kw=dict(wspace=0.06))
    vmax = max(s.max() for _, _, s, _ in snaps)
    for ax, (t, polys, speed, parts) in zip(axes, snaps):
        ax.add_collection(PolyCollection(polys, array=speed, cmap="gray_r" if P["name"] == "light" else "gray",
                                         clim=(0, 1.6 * vmax), edgecolors="face", linewidths=0.2, rasterized=True))
        sc = ax.scatter(parts["x"][:, 0], parts["x"][:, 1], c=parts["T_P"], s=9, cmap="plasma", vmin=0.4,
                        vmax=1 / 1.4, linewidths=0)
        bare(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"t = {t:g}", fontsize=12)
    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=50)
    cb.set_label("particle temperature T_P (snapshot field); gray: gas speed, zero at the vortex centers", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_visible(False)
    return fig


CORE = (0.25, 0.25, 0.1)


def read_csv(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    return {name: np.array([float(r[k]) for r in rows[1:]]) for k, name in enumerate(rows[0])}


@functools.cache
def crossing_run(binary):
    """The crossings of the boundary of the probe's disk, and the probe, of examples/particle_vortex."""
    work = Path(tempfile.mkdtemp())
    cx, cy, r = CORE
    run(binary, vortex_input('[[particles.crossings]]\nname = "core"\n'
                             f'region = {{type = "boundary", of = {{type = "disk", center = [{cx}, {cy}], radius = {r}}}}}\n'),
        work)
    return read_csv(work / "particle_crossings_core.csv"), read_csv(work / "particle_probe_core.csv")


def particle_crossings(P, args):
    crossings, probe = crossing_run(args.mallard)
    cx, cy, r = CORE
    lattice = np.linspace(0.1, 0.4, 9)
    X, Y = np.meshgrid(lattice, lattice)
    inside0 = int(np.sum(np.hypot(X - cx, Y - cy) < r))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.3), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.35]))
    theta = np.linspace(0, 2 * np.pi, 300)
    a.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), color=P["line"], lw=1.2)
    a.plot(X.ravel(), Y.ravel(), "o", color=P["muted"], ms=2.5, label="seeds")
    for direction, color, name in [(1, P["flame"], "out"), (-1, P["blue"], "in")]:
        sel = crossings["direction"] == direction
        a.plot(crossings["x"][sel], crossings["y"][sel], "o", color=color, ms=5, mfc="none",
               label=f"{name} (direction {direction}): {sel.sum()}")
    bare(a)
    a.set_xlim(0.05, 0.45)
    a.set_ylim(0.05, 0.45)
    a.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, handletextpad=0.2, columnspacing=0.8)
    a.set_title("crossings of the core's circle", fontsize=12)
    order = np.argsort(crossings["t"], kind="stable")
    t = np.concatenate([[0], crossings["t"][order], [probe["t"][-1]]])
    net = np.cumsum(crossings["direction"][order])
    count = inside0 - np.concatenate([[0], net, net[-1:]])
    b.step(t, count, where="post", color=P["flame"], label="seeds inside less net crossings out")
    sampled = probe["COUNT"] if "COUNT" in probe else probe["count"]  # probe columns are uppercase since #1466
    b.plot(probe["t"], sampled, "o", color=P["blue"], ms=4, mfc="none", label="probe count")
    b.set_xlabel("t")
    b.set_ylabel("particles in the core")
    b.set_xlim(0, None)
    b.set_ylim(0, None)
    b.yaxis.get_major_locator().set_params(integer=True)
    b.legend(fontsize=10, loc="upper right")
    b.set_title("the crossings account for the probe's count", fontsize=12)
    return fig


@functools.cache
def trajectory_run(binary):
    """A quarter of the particles of examples/particle_vortex every 10 steps, with their temperature."""
    work = Path(tempfile.mkdtemp())
    run(binary, vortex_input('[[particles.trajectories]]\nname = "sample"\ninterval = 10\nfraction = 0.25\n'
                             'fields = ["T_P"]\n'), work)
    return read_csv(work / "particle_trajectories_sample.csv")


def particle_trajectories(P, args):
    rows = trajectory_run(args.mallard)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    segments, values = [], []
    for i in np.unique(rows["id"]):
        sel = rows["id"] == i
        x, y, T = rows["x"][sel], rows["y"][sel], rows["T_P"][sel]
        for k in range(len(x) - 1):
            if abs(x[k + 1] - x[k]) < 0.5 and abs(y[k + 1] - y[k]) < 0.5:  # not across a periodic side
                segments.append([(x[k], y[k]), (x[k + 1], y[k + 1])])
                values.append(0.5 * (T[k] + T[k + 1]))
        ax.plot(x[0], y[0], "o", color=P["fg"], ms=3)
    lines = LineCollection(segments, array=np.array(values), cmap="plasma", linewidths=1.4)
    lines.set_clim(0.4, 1 / 1.4)
    ax.add_collection(lines)
    ax.plot([0, 0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5, 0], color=P["mesh"], lw=0.8)
    bare(ax)
    ax.set_xlim(-0.01, 0.51)
    ax.set_ylim(-0.01, 0.51)
    cb = fig.colorbar(lines, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("particle temperature T_P", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_visible(False)
    ax.set_title(f"{len(np.unique(rows['id'])):g} of 81 particles, t = 0 to 10, in their vortex", fontsize=12)
    return fig


GUIDE_RUNS = {"particle_crossings": particle_crossings, "particle_snapshots": particle_snapshots,
              "particle_trajectories": particle_trajectories}


def figures(mallard=None, names=None):
    if not mallard:
        return
    args = argparse.Namespace(mallard=str(Path(mallard).resolve()))
    for name, draw in GUIDE_RUNS.items():
        if not names or name in names:
            figstyle.save(name, lambda P, draw=draw: draw(P, args))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", required=True, help="2D Mallard executable, built with HDF5")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    figures(args.mallard, args.names)


if __name__ == "__main__":
    main()
