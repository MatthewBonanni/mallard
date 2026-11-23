"""Figures of particle output, in the particles page of the user guide
(docs/images/guide/).

    python tools/docs/particle_output.py --mallard build/src/mallard [names...]

From short runs of examples/particle_vortex (a 2D build with HDF5, about a
minute on 2 CPU threads): particle snapshots of [[write_data]] over the gas
speed of a VTU snapshot at the same times (particle_snapshots). The particle
files are read with h5py.
"""
import argparse
import functools
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

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
                             '[[write_data]]\nprefix = "drops"\nparticle_fields = ["T_P", "slip_max"]\n'
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


GUIDE_RUNS = {"particle_snapshots": particle_snapshots}


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
